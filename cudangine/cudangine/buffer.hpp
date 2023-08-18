#pragma once

#include "exception.hpp"

#include <vector>

namespace cudangine
{

template<class T>
class Buffer
{

public:

    Buffer(size_t, T*);

    ~Buffer();

    void synchronizeHost();

    void synchronizeDevice();

    std::vector<T>& data();

    operator T*() const noexcept;

private:

    std::vector<T> m_dataH;

    T* m_dataD;

    const size_t m_elem;

};

}
