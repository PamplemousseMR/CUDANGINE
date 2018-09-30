#include <stdio.h>

#include <buffer.hpp>
#include <buffer.hxx>
#include <exception.hpp>

__device__ __constant__ short divider = 2;

template< typename T >
__host__ __device__ void affect(T* _arr, size_t _index, T _value)
{
    _arr[_index] = _value;
}

__global__ void kernelDivEquals(double* const _vecA, unsigned _size)
{
    const unsigned globalSize = gridDim.x * blockDim.x;
    const double localSize = _size/__uint2double_rn(globalSize);
    const unsigned index = blockIdx.x*blockDim.x + threadIdx.x;
    const unsigned begin = index * localSize;
    const unsigned end = index * localSize + localSize;
    for(int i = begin ; i < end ; ++i)
    {
        _vecA[i] /= divider;
    }
}

int main(int argc, char **argv)
{
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

    return EXIT_SUCCESS;
}
