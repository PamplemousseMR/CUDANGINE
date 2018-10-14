#include <stdio.h>

#include <buffer.hpp>
#include <buffer.hxx>

__device__ int deviceBufA[10];
__device__ int deviceBufB[10];

__global__ void kernelMultEquals()
{
    const unsigned globalSize = gridDim.x * blockDim.x;
    const double localSize = 10/__uint2double_rn(globalSize);
    const unsigned index = blockIdx.x*blockDim.x + threadIdx.x;
    const unsigned begin = index * localSize;
    const unsigned end = index * localSize + localSize;
    for(int i = begin ; i < end ; ++i)
    {
        deviceBufA[i] *= deviceBufB[i];
    }
}
int main(int argc, char **argv)
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

    cudaMemcpyFromSymbol(vecA, deviceBufA,size*sizeof(int));
    cudaMemcpyFromSymbol(vecB, deviceBufB,size*sizeof(int));

    return EXIT_SUCCESS;
}
