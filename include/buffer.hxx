#include "buffer.hpp"

namespace cudangine
{

template<class T>
Buffer<T>::Buffer(int _elem, T* _data) throw(...)
{
    const size_t size = _elem * sizeof(T);
    cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&m_data), size);
    if(err != cudaSuccess)
    {
        throw Exception(err);
    }
    err = cudaMemcpy(m_data, _data, size, cudaMemcpyHostToDevice);
    if(err != cudaSuccess)
    {
        throw Exception(err);
    }
}

template<class T>
Buffer<T>::~Buffer()
{
    cudaFree(m_data);
}

template<class T>
Buffer<T>::operator T*()
{
    return m_data;
}

}
