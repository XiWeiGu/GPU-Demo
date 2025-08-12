export LD_LIBRARY_PATH=//media/gxw/f0624adb-6f05-4272-97fe-c9f66e0776ca/gxw-github/hipBLASLt-0.12.1-unofficial/build/debug/library:$LD_LIBRARY_PATH
hipcc -std=c++11 -O3 -fopenmp -I./build/debug/include -L./build/debug/library -lhipblaslt-d *
