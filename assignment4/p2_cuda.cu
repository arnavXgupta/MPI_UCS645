#include <iostream>
#include <cuda.h>
#include <algorithm>
#include <cstdlib>
using namespace std;

#define N 1000

__global__ void mergePass(int* input, int* output, int width, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int start = idx * (2 * width);

    if (start >= n) return;

    int mid = min(start + width, n);
    int end = min(start + 2 * width, n);

    int i = start, j = mid, k = start;

    while (i < mid && j < end) {
        if (input[i] <= input[j]){
            output[k++] = input[i++];
        }else{
            output[k++] = input[j++];
        }
    }

    while (i < mid) output[k++] = input[i++];
    while (j < end) output[k++] = input[j++];
}

int main() {
    int* h_array = new int[N];
    int* d_array1, * d_array2;

    for (int i = 0; i < N; ++i){
        h_array[i] = rand() % 1000;
    }

    cout << "Unsorted Array : ";
    for (int i = 0; i < 1000; ++i){
        cout << h_array[i] << " ";
    }
    cout << "\n";

    cudaMalloc(&d_array1, N * sizeof(int));
    cudaMalloc(&d_array2, N * sizeof(int));

    cudaMemcpy(d_array1, h_array, N * sizeof(int), cudaMemcpyHostToDevice);

    int* in = d_array1;
    int* out = d_array2;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    for (int width = 1; width < N; width *= 2) {
        int numThreads = (N + (2 * width) - 1) / (2 * width);
        int blockSize = 256;
        int gridSize = (numThreads + blockSize - 1) / blockSize;

        mergePass<<<gridSize, blockSize>>>(in, out, width, N);

        swap(in, out);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "CUDA Merge Sort Time: " << milliseconds << " ms\n";
    
    cudaMemcpy(h_array, in, N * sizeof(int), cudaMemcpyDeviceToHost);

    cout << "Sorted Array : ";
    for (int i = 0; i < 1000; ++i){
        cout << h_array[i] << " ";
    }
    cout << "\n";

    cudaFree(d_array1);
    cudaFree(d_array2);
    delete[] h_array;

    return 0;
}