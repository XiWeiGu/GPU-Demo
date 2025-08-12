#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <hip/hip_runtime.h>
#include "hipblaslt/hipblaslt.h"
#include <omp.h> // OpenMP 头文件
		 //
/*******************************
 * hipcc -std=c++11 -O3 -fopenmp -I./build/debug/include -L./build/debug/library -lhipblaslt-d demo.cpp
 * export LD_LIBRARY_PATH=/home/gxw/hipBLASLt-0.12.1-unofficial/build/debug/library:$LD_LIBRARY_PATH
********************************/

#define CHECK_HIP_ERROR(status) \
    if (status != hipSuccess) { \
        std::cerr << "HIP error (" << __FILE__ << ":" << __LINE__ << "): " << hipGetErrorString(status) << std::endl; \
        exit(EXIT_FAILURE); \
    }

#define CHECK_HIPBLASLT_ERROR(status) \
    if (status != HIPBLAS_STATUS_SUCCESS) { \
        std::cerr << "hipBLASLt error (" << __FILE__ << ":" << __LINE__ << "): " << status << std::endl; \
        exit(EXIT_FAILURE); \
    }

// CPU矩阵乘法 (列主序)
void cpu_gemm_colmajor(int m, int n, int k, const float* A, const float* B, float* C) {
    for (int col = 0; col < n; ++col) {      // 遍历C的列
        for (int row = 0; row < m; ++row) {  // 遍历C的行
            float sum = 0.0f;
            for (int i = 0; i < k; ++i) {
                // A[row + i * m] : A(row, i)  (列主序)
                // B[col + i * n] : B(col, i) = B^T(i, col) (列主序)
                sum += A[row + i * m] * B[col + i * n];
            }
            C[row + col * m] = sum;  // 列主序存储结果
        }
    }
}

// 验证两个矩阵是否相似
bool validateResults(const float* cpuResult, const float* gpuResult, int m, int n, float epsilon = 1e-3) {
    double totalError = 0.0;
    double maxError = 0.0;
    int errorCount = 0;
    const int totalElements = m * n;
    
    for (int i = 0; i < totalElements; i++) {
        float diff = std::abs(cpuResult[i] - gpuResult[i]);
        float ref = std::max(std::abs(cpuResult[i]), std::abs(gpuResult[i]));
        
        // 相对误差
        float relError = (ref > epsilon) ? diff / ref : diff;
        
        if (relError > epsilon) {
            if (errorCount < 10) {
                std::cerr << "Mismatch at element " << i << ": CPU=" << cpuResult[i] 
                          << ", GPU=" << gpuResult[i] << ", diff=" << diff 
                          << ", relError=" << relError << std::endl;
            }
            errorCount++;
        }
        
        totalError += diff;
        if (diff > maxError) maxError = diff;
    }
    
    double avgError = totalError / totalElements;
    double errorRate = static_cast<double>(errorCount) / totalElements * 100.0;
    
    std::cout << "Validation results:" << std::endl;
    std::cout << "  Total elements: " << totalElements << std::endl;
    std::cout << "  Mismatched elements: " << errorCount << " (" << errorRate << "%)" << std::endl;
    std::cout << "  Average absolute error: " << avgError << std::endl;
    std::cout << "  Maximum absolute error: " << maxError << std::endl;
    
    if (errorRate > 0.1) { // 超过0.1%的错误率认为失败
        std::cerr << "Validation FAILED: Too many mismatches!" << std::endl;
        return false;
    }
    
    if (maxError > 10 * epsilon) { // 最大误差过大
        std::cerr << "Validation FAILED: Maximum error too large!" << std::endl;
        return false;
    }
    
    std::cout << "Validation PASSED!" << std::endl;
    return true;
}

int main() {
    // 矩阵维度设置
    const int m = 512;  // 行数 (减小规模以便更快验证)
    const int n = 256;  // 列数
    const int k = 1024; // 内部维度
    
    // 步长设置
    const int lda = m; // A 矩阵的 leading dimension
    const int ldb = n; // B 矩阵的 leading dimension (转置前)
    const int ldc = m; // C 矩阵的 leading dimension
    
    // 矩阵大小计算
    const size_t sizeA = m * k * sizeof(float);
    const size_t sizeB = n * k * sizeof(float);
    const size_t sizeC = m * n * sizeof(float);
    
    std::cout << "Matrix dimensions: A(" << m << "x" << k << "), "
              << "B(" << n << "x" << k << " transposed), "
              << "C(" << m << "x" << n << ")" << std::endl;
    
    // 初始化矩阵数据
    std::vector<float> h_A(m * k);
    std::vector<float> h_B(n * k);
    std::vector<float> h_C_gpu(m * n, 0.0f);
    std::vector<float> h_C_cpu(m * n, 0.0f);
    
    // 填充随机数据 (范围 [0, 1])
    std::generate(h_A.begin(), h_A.end(), []{ return static_cast<float>(rand()) / RAND_MAX; });
    std::generate(h_B.begin(), h_B.end(), []{ return static_cast<float>(rand()) / RAND_MAX; });
    
    // 分配设备内存
    float *d_A, *d_B, *d_C;
    CHECK_HIP_ERROR(hipMalloc(&d_A, sizeA));
    CHECK_HIP_ERROR(hipMalloc(&d_B, sizeB));
    CHECK_HIP_ERROR(hipMalloc(&d_C, sizeC));
    
    // 拷贝数据到设备
    CHECK_HIP_ERROR(hipMemcpy(d_A, h_A.data(), sizeA, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_B, h_B.data(), sizeB, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemset(d_C, 0, sizeC)); // 初始化C为0
    
    // 创建hipBLASLt句柄
    hipblasLtHandle_t handle;
    CHECK_HIPBLASLT_ERROR(hipblasLtCreate(&handle));
    
    // 创建矩阵描述符
    hipblasLtMatrixLayout_t matA, matB, matC;
    hipblasLtMatrixLayoutCreate(&matA, HIP_R_32F, m, k, lda);
    hipblasLtMatrixLayoutCreate(&matB, HIP_R_32F, k, n, ldb); // 注意维度
    hipblasLtMatrixLayoutCreate(&matC, HIP_R_32F, m, n, ldc);
    
    // 设置操作描述符
    hipblasLtMatmulDesc_t operation;
    hipblasLtMatmulDescCreate(&operation, HIPBLAS_COMPUTE_32F, HIP_R_32F);
    
    // 设置转置选项
    hipblasOperation_t transA = HIPBLAS_OP_N; // A不转置
    hipblasOperation_t transB = HIPBLAS_OP_T; // B转置
    
    hipblasLtMatmulDescSetAttribute(operation, HIPBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(transA));
    hipblasLtMatmulDescSetAttribute(operation, HIPBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(transB));
    
    // 设置算法偏好
    hipblasLtMatmulPreference_t preference;
    hipblasLtMatmulPreferenceCreate(&preference);
    const int workspaceSize = 32 * 1024 * 1024; // 32 MB
    hipblasLtMatmulPreferenceSetAttribute(preference, 
                                         HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, 
                                         &workspaceSize, 
                                         sizeof(workspaceSize));
    
    // 获取算法
    const int requestAlgoCount = 10;
    int returnedAlgoCount = 0;
    hipblasLtMatmulHeuristicResult_t heuristicResults[requestAlgoCount];
    
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulAlgoGetHeuristic(
        handle, operation, matA, matB, matC, matC, preference,
        requestAlgoCount, heuristicResults, &returnedAlgoCount));
    
    if (returnedAlgoCount == 0) {
        std::cerr << "No valid algorithms found!" << std::endl;
        exit(EXIT_FAILURE);
    }
    
    std::cout << "Found " << returnedAlgoCount << " valid algorithms. Using the first one." << std::endl;
    
    // 分配工作空间
    void *d_workspace = nullptr;
    CHECK_HIP_ERROR(hipMalloc(&d_workspace, workspaceSize));
    
    // 设置标量参数
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // 创建流
    hipStream_t stream;
    CHECK_HIP_ERROR(hipStreamCreate(&stream));
    
    // 预热运行
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmul(
        handle, operation,
        &alpha,
        d_A, matA,
        d_B, matB,
        &beta,
        d_C, matC,
        d_C, matC,
        &heuristicResults[0].algo,
        d_workspace, workspaceSize,
        stream));
    CHECK_HIP_ERROR(hipStreamSynchronize(stream));
    
    // 计时运行
    const int numRuns = 10;
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < numRuns; i++) {
        CHECK_HIPBLASLT_ERROR(hipblasLtMatmul(
            handle, operation,
            &alpha,
            d_A, matA,
            d_B, matB,
            &beta,
            d_C, matC,
            d_C, matC,
            &heuristicResults[0].algo,
            d_workspace, workspaceSize,
            stream));
    }
    
    CHECK_HIP_ERROR(hipStreamSynchronize(stream));
    auto end = std::chrono::high_resolution_clock::now();
    
    // 计算性能
    double totalTime = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000000.0;
    double avgTime = totalTime / numRuns;
    double flops = 2.0 * m * n * k;
    double gflops = (flops / avgTime) / 1e9;
    
    std::cout << "hipBLASLt performance:" << std::endl;
    std::cout << "  Average time: " << avgTime * 1000 << " ms" << std::endl;
    std::cout << "  GFLOPS: " << gflops << std::endl;
    
    // 拷贝结果回主机
    CHECK_HIP_ERROR(hipMemcpy(h_C_gpu.data(), d_C, sizeC, hipMemcpyDeviceToHost));
    
    // 在 CPU 上计算参考结果
    std::cout << "\nStarting CPU computation..." << std::endl;
    auto cpuStart = std::chrono::high_resolution_clock::now();
    
    // 使用 OpenMP 加速
    omp_set_num_threads(omp_get_max_threads());
    cpu_gemm_colmajor(m, n, k, h_A.data(), h_B.data(), h_C_cpu.data());
    
    auto cpuEnd = std::chrono::high_resolution_clock::now();
    double cpuTime = std::chrono::duration_cast<std::chrono::microseconds>(cpuEnd - cpuStart).count() / 1000000.0;
    std::cout << "CPU computation time: " << cpuTime * 1000 << " ms" << std::endl;
    
    // 验证结果
    std::cout << "\nValidating results..." << std::endl;
    bool isValid = validateResults(h_C_cpu.data(), h_C_gpu.data(), m, n);
    
    // 清理资源
    hipblasLtMatmulPreferenceDestroy(preference);
    hipblasLtMatrixLayoutDestroy(matA);
    hipblasLtMatrixLayoutDestroy(matB);
    hipblasLtMatrixLayoutDestroy(matC);
    hipblasLtMatmulDescDestroy(operation);
    hipblasLtDestroy(handle);
    
    hipFree(d_workspace);
    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);
    hipStreamDestroy(stream);
    
    return isValid ? EXIT_SUCCESS : EXIT_FAILURE;
}
