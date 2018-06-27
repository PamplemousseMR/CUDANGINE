#include <stdio.h>
#include <iostream>

#include "buffer.hpp"
#include "buffer.hxx"

#define N 32

__global__ void Kernel_double(int niters, int* d_inputs)
{
    for(int i=0 ; i<niters ; ++i) {
        printf("%d ", d_inputs[i]);
    }
    printf("\n");
}

int main(int argc, char **argv)
{    
    {
        int inputs[N];
        for (int i = 0; i<N; i++){
            inputs[i] = i*2;
        }

        cudangine::Buffer<int> buf(N, inputs);
        Kernel_double<<<1,1>>>(N, buf);
        cudaDeviceSynchronize();
    }

    return EXIT_SUCCESS;
}
