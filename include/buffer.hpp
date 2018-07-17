#pragma once

#include "exception.hpp"

namespace cudangine
{

template<class T>
class Buffer
{

public:

    Buffer(size_t, T*) throw(Exception);

    ~Buffer();

    void synchronizeHost() throw(Exception);

    void synchronizeDevice() throw(Exception);

    operator T*() const noexcept;

private:

    T* m_dataH;

    T* m_dataD;

    const size_t m_elem;

};

}
