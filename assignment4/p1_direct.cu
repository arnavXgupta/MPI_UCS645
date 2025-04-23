#include <iostream>
#include <cuda_runtime.h>
using namespace std;

#define N 1024  // Step 1

// Step 7
__global__ void computeSumUsingFormula(int *output) {
    int tid = threadIdx.x;

    // Only one thread performs the task using formula
    if (tid == 0) {
        int sum = (N * (N + 1)) / 2;
        *output = sum;
    }
}

int main() {
    // Step 2
    int *h_output;
    int *d_output;

    // Step 3
    h_output = new int;
    cudaMalloc((void**)&d_output, sizeof(int));

    // Step 6 & 7
    computeSumUsingFormula<<<32, 256>>>(d_output);

    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, sizeof(int), cudaMemcpyDeviceToHost);

    cout << "Sum of first " << N << " integers using formula is: " << *h_output << endl;

    delete h_output;
    cudaFree(d_output);

    return 0;
}