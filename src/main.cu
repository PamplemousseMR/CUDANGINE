#include <stdio.h>
#include <iostream>

#include "buffer.hpp"
#include "buffer.hxx"

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

//----------------------------------------------------------------------------------------------------

__device__ int deviceBufA[10];
__device__ int deviceBufB[10];

__global__ void kernelMultEquals()
{
    printf("gridSize: %d, blockId: %d, threadByBlock: %d, threadIdInBlock: %d, warpSize: %d\n", gridDim.x, blockIdx.x, blockDim.x, threadIdx.x, warpSize);
    const unsigned globalSize = gridDim.x * blockDim.x;
    const double localSize = 10/__uint2double_rn(globalSize);
    const unsigned index = blockIdx.x*blockDim.x + threadIdx.x;
    const unsigned begin = index * localSize;
    const unsigned end = index * localSize + localSize;
    printf("global: %d, local: %f, begin: %d, end: %d\n", globalSize, localSize, begin, end);
    for(int i = begin ; i < end ; ++i)
    {
        deviceBufA[i] *= deviceBufB[i];
    }
}

//----------------------------------------------------------------------------------------------------

int main(int argc, char **argv)
{
    // Basic exemple
    /*{
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
    }*/

    // Memory copy exemple
    {
        const unsigned size = 10;
        int vecA[size];
        int vecB[size];

        for (int i = 0; i < size ; ++i)
        {
            vecA[i] = i;
            vecB[i] = i;
        }

        cudaMemcpyToSymbol(deviceBufA, vecA,size*sizeof(int));
        cudaMemcpyToSymbol(deviceBufB, vecB,size*sizeof(int));

        kernelMultEquals<<<2,4>>>();
        cudaDeviceSynchronize();

        for (int i = 0; i < size ; ++i)
        {
            std::cout << vecA[i] << " " ;
        }
        std::cout << std::endl;

        cudaMemcpyFromSymbol(vecA, deviceBufA,size*sizeof(int));
        cudaMemcpyFromSymbol(vecB, deviceBufB,size*sizeof(int));

        for (int i = 0; i < size ; ++i)
        {
            std::cout << vecA[i] << " " ;
        }
        std::cout << std::endl;
    }

    return EXIT_SUCCESS;
}
