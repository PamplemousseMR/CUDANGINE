#include <stdio.h>

#include <buffer.hpp>
#include <buffer.hxx>

template< typename T >
__host__ __device__ void affect(T* _arr, size_t _index, T _value)
{
    _arr[_index] = _value;
}

__global__ void kernelPlusEquals(int* const _vecA, const int* const _vecB, unsigned _size)
{
    const unsigned globalSize = gridDim.x * blockDim.x;
    const double localSize = _size/__uint2double_rn(globalSize);
    const unsigned index = blockIdx.x*blockDim.x + threadIdx.x;
    const unsigned begin = index * localSize;
    const unsigned end = index * localSize + localSize;
    for(int i = begin ; i < end ; ++i)
    {
        affect(_vecA, i, _vecA[i] + _vecB[i]);
    }
}

int main(int argc, char **argv)
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

    bufVecA.synchronizeHost();

    return EXIT_SUCCESS;
}
