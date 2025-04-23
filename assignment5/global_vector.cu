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

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAddStatic<<<blocksPerGrid, threadsPerBlock>>>();

    cudaDeviceSynchronize();

    cout << "Vector addition result (first 10 values):\n";
    for (int i = 0; i < 10; ++i) {
        cout << A[i] << " + " << B[i] << " = " << C[i] << "\n";
    }

    return 0;
}
