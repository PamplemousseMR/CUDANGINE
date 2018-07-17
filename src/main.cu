#include <stdio.h>
#include <iostream>

#include "buffer.hpp"
#include "buffer.hxx"

/*#define N 10

__constant__ __device__ int fact = 2;

double X[N] = {1,-2,3,-4,5,-6,7,-8,9,-10};
double Y[N] = {0};
__device__ double DEV_X[N];

__global__ void Kernel_double(int niters, int* d_inputs)
{
    for(int i=0 ; i<niters ; ++i) {
        printf("%d ", d_inputs[i] * fact);
    }
    printf("\n");
    for(int i=0 ; i<niters ; ++i) {
        printf("%f ", DEV_X[i]);
    }
    for(int i=0 ; i<niters ; ++i) {
        DEV_X[i] /= 2.0;
    }
    printf("\n");
}

int main(int argc, char **argv)
{    
    cudaMemcpyToSymbol(DEV_X, X,10*sizeof(double));
    {
        int inputs[N];
        for (int i = 0; i<N; i++){
            inputs[i] = i*2;
        }

        cudangine::Buffer<int> buf(N, inputs);
        Kernel_double<<<1,1>>>(N, buf);
        cudaDeviceSynchronize();
    }
    cudaMemcpyFromSymbol(Y, DEV_X, 10*sizeof(double));
    for(int i=0 ; i<N ; ++i) {
        printf("%f ", Y[i]);
    }

    return EXIT_SUCCESS;
}*/

template< typename T >
__host__ __device__ void affect(T* _arr, size_t _index, T _value)
{
    _arr[_index] = _value;
}

__global__ void kernelPlusEquals(int* const _vecA, const int* const _vecB, unsigned _size)
{
    printf("gridSize: %d, blockId: %d, threadByBlock: %d, threadIdInBlock: %d, warpSize: %d\n", gridDim.x, blockIdx.x, blockDim.x, threadIdx.x, warpSize);
    const unsigned globalSize = gridDim.x * blockDim.x;
    const double localSize = _size/__uint2double_rn(globalSize);
    const unsigned index = blockIdx.x*blockDim.x + threadIdx.x;
    const unsigned begin = index * localSize;
    const unsigned end = index * localSize + localSize;
    printf("global: %d, local: %f, begin: %d, end: %d\n", globalSize, localSize, begin, end);
    for(int i = begin ; i < end ; ++i)
    {
        affect(_vecA, i, _vecA[i] + _vecB[i]);
    }
}

int main(int argc, char **argv)
{
    {
        const unsigned size = 35;
        int vecA[size];
        int vecB[size];

        cudangine::Buffer<int> bufVecA(size, vecA);

        for (int i = 0; i < size ; ++i)
        {
            affect(vecA,i,i);
            affect(vecB,i,i);
        }

        bufVecA.synchronizeDevice();
        cudangine::Buffer<int> bufVecB(size, vecB);

        kernelPlusEquals<<<2,4>>>(bufVecA, bufVecB, size);
        cudaDeviceSynchronize();

        for (int i = 0; i < size ; ++i)
        {
            std::cout << vecA[i] << " " ;
        }
        std::cout << std::endl;

        bufVecA.synchronizeHost();

        for (int i = 0; i < size ; ++i)
        {
            std::cout << vecA[i] << " " ;
        }
        std::cout << std::endl;
    }

    return EXIT_SUCCESS;
}
