#include <stdio.h>

#include <buffer.hpp>
#include <buffer.hxx>

#include "point.hpp"

__global__ void kernelPointPlusEquals(cudangine::Point* const _vecA, const cudangine::Point* const _vecB, unsigned _size)
{
    const unsigned globalSize = gridDim.x * blockDim.x;
    const double localSize = _size/__uint2double_rn(globalSize);
    const unsigned index = blockIdx.x*blockDim.x + threadIdx.x;
    const unsigned begin = index * localSize;
    const unsigned end = index * localSize + localSize;
    for(int i = begin ; i < end ; ++i)
    {
        _vecA[i].m_x += _vecB[i].m_x;
        _vecA[i].m_y += _vecB[i].m_y;
    }
}

int main(int argc, char **argv)
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


    bufVecA.synchronizeHost();

    return EXIT_SUCCESS;
}
