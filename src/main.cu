#include <stdio.h>

#include "exemples.hpp"

//http://www.icl.utk.edu/~mgates3/docs/cuda.html

int main(int argc, char **argv)
{
    basicTest();
    memoryTest();
    streamTest();
    classTest();

    return EXIT_SUCCESS;
}
