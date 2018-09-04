#pragma once

template< typename T >
__host__ __device__ void affect(T* _arr, size_t _index, T _value)
{
    _arr[_index] = _value;
}

void basicTest();
void memoryTest();
void streamTest();
void classTest();
