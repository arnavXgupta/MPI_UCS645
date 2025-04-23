#include <iostream>
#include <cuda_runtime.h>
using namespace std;

#define N 1024

__device__ __managed__ float A[N];
__device__ __managed__ float B[N];
__device__ __managed__ float C[N];

__global__ void vectorAddStatic() {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    for (int i = 0; i < N; ++i) {
        A[i] = static_cast<float>(i);
        B[i] = static_cast<float>(i * 2);
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    cudaEventRecord(start);
    vectorAddStatic<<<blocksPerGrid, threadsPerBlock>>>();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Kernel execution time: " << milliseconds << " ms\n";

    cout << "Result (first 5 values):\n";
    for (int i = 0; i < 5; ++i)
        cout << A[i] << " + " << B[i] << " = " << C[i] << "\n";

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    int memClock = prop.memoryClockRate; 
    int memBus = prop.memoryBusWidth;    
    float theoBW = 2.0f * memClock * memBus / 8 / 1e6;

    cout << "Device: " << prop.name << "\n";
    cout << "Theoretical Bandwidth: " << theoBW << " GB/s\n";

    return 0;
}
