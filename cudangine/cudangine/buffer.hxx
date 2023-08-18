#include "buffer.hpp"

#include <assert.h>

namespace cudangine
{

template<class T>
Buffer<T>::Buffer(size_t _elem, T* _data) :
    m_dataH(_data, _data + _elem),
    m_elem(_elem)
{
    assert(_data != nullptr);
    const size_t size = m_elem * sizeof(T);
    cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&m_dataD), size);
    if(err != cudaSuccess)
    {
        throw Exception(err);
    }
    err = cudaMemcpy(m_dataD, m_dataH.data(), size, cudaMemcpyHostToDevice);
    if(err != cudaSuccess)
    {
        throw Exception(err);
    }
}

template<class T>
Buffer<T>::~Buffer()
{
    const cudaError_t err = cudaFree(m_dataD);
    assert(err == cudaSuccess);
}

template<class T>
void Buffer<T>::synchronizeHost()
{
    const size_t size = m_elem * sizeof(T);
    const cudaError_t err = cudaMemcpy(m_dataH.data(), m_dataD, size, cudaMemcpyDeviceToHost);
    if(err != cudaSuccess)
    {
        throw Exception(err);
    }
}

template<class T>
void Buffer<T>::synchronizeDevice()
{
    const size_t size = m_elem * sizeof(T);
    const cudaError_t err = cudaMemcpy(m_dataD, m_dataH.data(), size, cudaMemcpyHostToDevice);
    if(err != cudaSuccess)
    {
        throw Exception(err);
    }
}

template<class T>
std::vector<T>& Buffer<T>::data()
{
    return m_dataH;
}

template<class T>
Buffer<T>::operator T*() const noexcept
{
    return m_dataD;
}

}
