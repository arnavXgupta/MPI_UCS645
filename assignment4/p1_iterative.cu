#include <iostream>
#include <cuda_runtime.h>
using namespace std;

#define N 1024  // Number of integers (step 1)

//(step 7)
__global__ void computeTasks(int *input, int *output) {
    int tid = threadIdx.x;

    
    if (tid == 0) {
        int sum = 0;
        for (int i = 0; i < N; ++i) {
            sum += input[i];
        }
        *output = sum;
    }
}

int main() {
    //(step 2)
    int *h_input, *h_output;
    int *d_input, *d_output;

    h_input = new int[N];
    h_output = new int;

    // (step 3)
    cudaMalloc((void**)&d_input, N * sizeof(int));
    cudaMalloc((void**)&d_output, sizeof(int));

    // (step 4)
    for (int i = 0; i < N; ++i) {
        h_input[i] = i + 1;
    }

    // (step 5)
    cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice);

    // (step 6)
    computeTasks<<<32, 256>>>(d_input, d_output);  

    cudaMemcpy(h_output, d_output, sizeof(int), cudaMemcpyDeviceToHost);

    cout << "Sum of first " << N << " integers is: " << *h_output << endl;

    delete[] h_input;
    delete h_output;
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}