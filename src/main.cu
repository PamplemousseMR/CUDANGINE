#include <stdio.h>
#include <iostream>

#include "buffer.hpp"
#include "buffer.hxx"

#include "exception.hpp"

#include "point.hpp"

//http://www.icl.utk.edu/~mgates3/docs/cuda.html

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

__device__ __constant__ short divider = 2;

__global__ void kernelDivEquals(double* const _vecA, unsigned _size)
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
        _vecA[i] /= divider;
    }
}

//----------------------------------------------------------------------------------------------------

__global__ void kernelPointPlusEquals(cudangine::Point* const _vecA, const cudangine::Point* const _vecB, unsigned _size)
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
        _vecA[i].m_x += _vecB[i].m_x;
        _vecA[i].m_y += _vecB[i].m_y;
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
    /*{
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
    }*/

    // Stream exemple
    /*{
        const unsigned size = 35;
        double vecA[size];
        double vecB[size];

        for (int i = 0; i < size ; ++i)
        {
            vecA[i] = i+1;
            vecB[i] = i*4;
        }

        cudangine::Buffer<double> bufVecA(size, vecA);
        cudangine::Buffer<double> bufVecB(size, vecB);

        cudaStream_t stream1, stream2;
        cudaError_t err = cudaStreamCreate(&stream1);
        if(err != cudaSuccess)
        {
            throw cudangine::Exception(err);
        }
        err = cudaStreamCreate(&stream2);
        if(err != cudaSuccess)
        {
            throw cudangine::Exception(err);
        }

        kernelDivEquals<<<2,4, 0, stream1>>>(bufVecA, size);
        kernelDivEquals<<<2,4, 0, stream1>>>(bufVecA, size);
        kernelDivEquals<<<2,4, 0, stream2>>>(bufVecB, size);
        kernelDivEquals<<<2,4, 0, stream2>>>(bufVecB, size);
        cudaStreamSynchronize(stream1);
        cudaStreamSynchronize(stream2);

        bufVecA.synchronizeHost();
        bufVecB.synchronizeHost();

        for (int i = 0; i < size ; ++i)
        {
            std::cout << vecA[i] << " " ;
        }
        std::cout << std::endl;

        for (int i = 0; i < size ; ++i)
        {
            std::cout << vecB[i] << " " ;
        }
        std::cout << std::endl;
    }*/

    // Class exemple
    {
        const unsigned size = 10;
        cudangine::Point vecA[size];
        cudangine::Point vecB[size];

        for (int i = 0; i < size ; ++i)
        {
            vecA[i] = cudangine::Point(i,i);
            vecB[i] = cudangine::Point(i,i);
        }

        cudangine::Buffer<cudangine::Point> bufVecA(size, vecA);
        cudangine::Buffer<cudangine::Point> bufVecB(size, vecB);

        kernelPointPlusEquals<<<2,4>>>(bufVecA, bufVecB, size);
        cudaDeviceSynchronize();

        for (int i = 0; i < size ; ++i)
        {
            std::cout << "{" << vecA[i].m_x << ", " << vecA[i].m_y << "} ";
        }
        std::cout << std::endl;

        bufVecA.synchronizeHost();

        for (int i = 0; i < size ; ++i)
        {
            std::cout << "{" << vecA[i].m_x << ", " << vecA[i].m_y << "} ";
        }
        std::cout << std::endl;
    }

    return EXIT_SUCCESS;
}
