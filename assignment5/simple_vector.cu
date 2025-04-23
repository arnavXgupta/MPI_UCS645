%%writefile simple_vector.cu
#include <iostream>
#include <cuda_runtime.h>\
using namespace std;

#define N 1024  

__global__ void vectorAdd(float* A, float* B, float* C, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;

    size_t size = N * sizeof(float);

    h_A = new float[N];
    h_B = new float[N];
    h_C = new float[N];

    for (int i = 0; i < N; ++i) {
        h_A[i] = i * 1.0f;
        h_B[i] = i * 2.0f;
    }

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    cout << "Result (first 10 values):\n";
    for (int i = 0; i < 10; ++i) {
        cout << h_A[i] << " + " << h_B[i] << " = " << h_C[i] <<endl;
    }

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}