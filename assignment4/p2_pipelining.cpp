#include <mpi.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
using namespace std::chrono;
using namespace std;

#define N 1000

vector<int> merge(vector<int>& left, vector<int>& right) {
    auto start = high_resolution_clock::now();
    vector<int> result;
    auto it1 = left.begin();
    auto it2 = right.begin();

    while (it1 != left.end() && it2 != right.end()) {
        if (*it1 <= *it2) result.push_back(*it1++);
        else result.push_back(*it2++);
    }
    result.insert(result.end(), it1, left.end());
    result.insert(result.end(), it2, right.end());

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
    cout << "MPI Merge Sort Time: " << duration.count() << " ms\n";

    return result;
}

int main(int argc, char** argv) {
    int rank, size;

    MPI_Init(&argc, &argv);               
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
    MPI_Comm_size(MPI_COMM_WORLD, &size); 

    int chunkSize = N / size;
    vector<int> fullArray;

    if (rank == 0) {
        fullArray.resize(N);
        for (int i = 0; i < N; ++i) fullArray[i] = rand() % 1000;

        /* cout << "Unsorted elements: ";
        for (int i = 0; i < 1000; ++i){
            cout << fullArray[i] << " ";
        }
        cout << "\n"; */
    }

    vector<int> subArray(chunkSize);
    MPI_Scatter(fullArray.data(), 
                chunkSize, 
                MPI_INT,
                subArray.data(), 
                chunkSize, 
                MPI_INT,
                0, 
                MPI_COMM_WORLD);

    sort(subArray.begin(), subArray.end());

    vector<int> gatheredArray;
    if (rank == 0) gatheredArray.resize(N);

    MPI_Gather(subArray.data(), 
               chunkSize, 
               MPI_INT,
               gatheredArray.data(), 
               chunkSize, 
               MPI_INT,
               0, 
               MPI_COMM_WORLD);

    if (rank == 0) {
        vector<int> result = { gatheredArray.begin(), gatheredArray.begin() + chunkSize };
        for (int i = 1; i < size; ++i) {
            vector<int> nextChunk(gatheredArray.begin() + i * chunkSize, gatheredArray.begin() + (i + 1) * chunkSize);
            result = merge(result, nextChunk);  
        }

        cout << "Sorted element): ";
        for (int i = 0; i < 1000; ++i){
            cout << result[i] << " ";
        }
        cout << "\n"; 
    }

    MPI_Finalize();
    return 0;
}