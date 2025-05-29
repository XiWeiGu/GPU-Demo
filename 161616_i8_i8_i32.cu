#include <mma.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <chrono>

#ifdef CUBLAS_TEST
#include <cublas_v2.h>
#include <cublasLt.h>

#endif

using namespace nvcuda;

const int FACTOR = 512;
// 定义矩阵大小和分块尺寸
constexpr int M = 16 * FACTOR;
constexpr int N = 16 * FACTOR;
constexpr int K = 16 * FACTOR;

// 定义MMA分块尺寸
constexpr int BM = 16;
constexpr int BN = 16;
constexpr int BK = 16;

constexpr int WARP_SIZE = 32;
constexpr int WARPS_PER_BLOCK = 1;

// 使用int8_t数据类型的MMA核函数
__global__ void mma_int8_kernel(
    const int8_t* __restrict__ A,
    const int8_t* __restrict__ B,
    int32_t* __restrict__ C) {

    // Warp和线程标识
    const int lane_id = threadIdx.x % 32;
    const int warp_id = threadIdx.x / 32;

    // Block负责的全局矩阵起始位置
    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;

    //const int SUB_BLOCKS_PER_WRAP = (BM / BM) * (BN / BN) / WARPS_PER_BLOCK;

    const int sub_block_row = warp_id / (BM / BM) * BM;
    const int sub_block_col = warp_id % (BN / BN) * BN;

    // 声明WMMA片段
    wmma::fragment<wmma::matrix_a, BM, BN, BK, int8_t, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, BM, BN, BK, int8_t, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, BM, BN, BK, int32_t> acc_frag;

    // 初始化累加器
    wmma::fill_fragment(acc_frag, 0);

    // 遍历K维度的分块
    for (int k = 0; k < K; k += BK) {
        // 加载矩阵A和B的分块
        const int8_t* a_ptr = A + (block_row + sub_block_row) * K + k;
        const int8_t* b_ptr = B + (block_col + sub_block_col) * K + k;

        // 确保内存对齐（重要！）
        wmma::load_matrix_sync(a_frag, a_ptr, K);
        wmma::load_matrix_sync(b_frag, b_ptr, K);

        // 执行MMA操作：C = A * B + C
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }

    // 存储结果到全局内存
    int32_t* c_ptr = C + (block_row + sub_block_row) * N + (block_col + sub_block_col);
    wmma::store_matrix_sync(c_ptr, acc_frag, N, wmma::mem_row_major);
}

// CPU参考实现（支持列主序B矩阵）
void cpu_matmul_colmajor_b(const int8_t* A, const int8_t* B, int32_t* C) {
    // A: MxK (行主序), B: KxN (列主序)
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            int32_t sum = 0;
            // 访问B的第j列数据（列主序存储）
            const int8_t* B_col_j = B + j * K;  // 第j列起始位置
            for (int k = 0; k < K; ++k) {
                sum += static_cast<int32_t>(A[i*K + k]) *
                       static_cast<int32_t>(B_col_j[k]);
            }
            C[i*N + j] = sum;
        }
    }
}

// 错误检查宏
#define CHECK_CUDA_ERROR(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__ \
                      << ": " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// CPU矩阵转置函数
void cpu_transpose_matrix(const int32_t* src, int32_t* dst, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            dst[j * rows + i] = src[i * cols + j];
        }
    }
}

int main() {
    // 检查CUDA设备
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        printf("No CUDA devices found\n");
        return 1;
    }

    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, 0);
    printf("Using device: %s, SM %d.%d\n", props.name, props.major, props.minor);

    if (props.major < 8) {
        printf("Warning: WMMA requires SM 8.0 or higher\n");
    }

    // 分配主机和设备内存
    int8_t *h_A = new int8_t[M * K];
    int8_t *h_B = new int8_t[K * N];
    int32_t *h_C = new int32_t[M * N];
    int32_t *h_C_ref  = new int32_t[M * N];
    int32_t *h_C_ref2 = new int32_t[M * N];

    int8_t *d_A, *d_B;
    int32_t *d_C;

    cudaMalloc(&d_A, M * K * sizeof(int8_t));
    cudaMalloc(&d_B, K * N * sizeof(int8_t));
    cudaMalloc(&d_C, M * N * sizeof(int32_t));

    // 初始化输入数据
    for (int i = 0; i < M * K; i++) {
        h_A[i] = static_cast<int8_t>(rand() % 10);
    }

    for (int i = 0; i < K * N; i++) {
        h_B[i] = static_cast<int8_t>(rand() % 10);
    }

    // 将数据复制到设备
    cudaMemcpy(d_A, h_A, M * K * sizeof(int8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(int8_t), cudaMemcpyHostToDevice);

    // 初始化C为0
    memset(h_C, 0, M * N * sizeof(int32_t));
    memset(h_C_ref, 0, M * N * sizeof(int32_t));
    cudaMemcpy(d_C, h_C, M * N * sizeof(int32_t), cudaMemcpyHostToDevice);

    // 配置网格和线程块
    dim3 gridDim(M / BM, N / BN);
    dim3 blockDim(WARP_SIZE * WARPS_PER_BLOCK); // 4个线程束

    // 创建CUDA事件计时
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    // 执行GPU核函数
    CHECK_CUDA_ERROR(cudaEventRecord(start));

    // 启动核自定义函数
    mma_int8_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C);

    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // 计算耗时
    float milliseconds = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));
    std::cout << "MMA GPU Time: " << milliseconds << " ms" << std::endl;

    // 检查错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // 同步设备
    cudaDeviceSynchronize();

#ifdef DISABLE_CPU
    // 将结果复制回主机
    cudaMemcpy(h_C, d_C, M * N * sizeof(int32_t), cudaMemcpyDeviceToHost);

    // 执行CPU参考计算
    auto cpu_start = std::chrono::high_resolution_clock::now();
    cpu_matmul_colmajor_b(h_A, h_B, h_C_ref);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    auto cpu_duration = std::chrono::duration_cast<std::chrono::milliseconds>(cpu_end - cpu_start);
    std::cout << "CPU Time: " << cpu_duration.count() << " ms" << std::endl;

    // 转置CPU计算的结果
    cpu_transpose_matrix(h_C_ref, h_C_ref2, M, N);

    // 验证结果
    int errors = 0;
    const double epsilon = 1e-2;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            int idx = i * N + j;
            if (abs(h_C[idx] - h_C_ref[idx]) > 1) { // 允许1的误差
                if (++errors <= 16) {
                    std::cerr << "Error at (" << i << ", " << j
                              << "): GPU=" << h_C[idx]
                              << " CPU=" << h_C_ref[idx] << std::endl;
                }
            }
        }
    }
#endif

    // 调用cuBLAS
#ifdef CUBLAS_TEST
    // 初始化cuBLAS库
    cublasHandle_t handle;
    cublasStatus_t cublasStatus = cublasCreate(&handle);
    if (cublasStatus != CUBLAS_STATUS_SUCCESS) {
        printf("Failed to initialize cuBLAS: %d\n", cublasStatus);
        return 1;
    }
    // 定义cuBLAS参数
    const int lda = K;  // A的leading dimension
    const int ldb = K;  // B的leading dimension（转置后）
    const int ldc = N;  // C的leading dimension

    // 标量参数 (alpha和beta)
    const int32_t alpha = 1;
    const int32_t  beta = 0;
    memset(h_C, 0, M * N * sizeof(int32_t));
    cudaMemcpy(d_C, h_C, M * N * sizeof(int32_t), cudaMemcpyHostToDevice);

    CHECK_CUDA_ERROR(cudaEventRecord(start));
    // 执行int8矩阵乘法，B矩阵转置
    cublasStatus = cublasGemmEx(
        handle,               // cuBLAS句柄
        CUBLAS_OP_T,          // A的转置操作 (转置)
        CUBLAS_OP_N,          // B的转置操作 (不转置转置)
        N,                    // A和C的行数
        M,                    // B和C的列数
        K,                    // A的列数和B的行数
        &alpha,               // 标量alpha
        d_A,                  // 矩阵A的设备指针
        CUDA_R_8I,            // A的数据类型 (int8)
        lda,                  // A的leading dimension
        d_B,                  // 矩阵B的设备指针
        CUDA_R_8I,            // B的数据类型 (int8)
        ldb,                  // B的leading dimension（转置后）
        &beta,                // 标量beta
        d_C,                  // 矩阵C的设备指针
        CUDA_R_32I,           // C的数据类型 (int32)
        ldc,                  // C的leading dimension
        CUBLAS_COMPUTE_32I,   // 计算精度 (int32)
        //CUBLAS_GEMM_DEFAULT   // 算法选择
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // 计算耗时
    milliseconds = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));
    std::cout << "cublas GPU Time: " << milliseconds << " ms" << std::endl;

    if (cublasStatus != CUBLAS_STATUS_SUCCESS) {
        printf("cuBLAS GEMM failed: %d\n", cublasStatus);
        cublasDestroy(handle);
        return 1;
    }

    // 同步设备
    cudaDeviceSynchronize();

#ifdef DISABLE_CPU
    // 将结果复制回主机
    cudaMemcpy(h_C, d_C, M * N * sizeof(int32_t), cudaMemcpyDeviceToHost);
    // 验证结果
    errors = 0;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            int idx = i * N + j;
            if (abs(h_C[idx] - h_C_ref2[idx]) > 1) { // 允许1的误差
                if (++errors <= 16) {
                    std::cerr << "Error at (" << i << ", " << j
                              << "): GPU=" << h_C[idx]
                              << " CPU=" << h_C_ref[idx] << std::endl;
                }
            }
        }
    }
#endif


#endif

    // 释放内存
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_C_ref;
    delete[] h_C_ref2;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
