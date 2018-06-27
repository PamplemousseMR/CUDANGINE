#pragma once

#include "exception.hpp"

namespace cudangine
{

template<class T>
class Buffer
{

public:

    Buffer(int, T*) throw(Exception);

    ~Buffer();

    operator T*();

private:

    T* m_data;

};

}
