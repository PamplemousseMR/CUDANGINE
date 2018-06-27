#include "exception.hpp"

namespace cudangine
{

    Exception::Exception(const cudaError_t& _error) noexcept
        :   std::exception()
    {
        m_message = cudaGetErrorString(_error);
    }

    const char* Exception::what() const noexcept
    {
        return m_message.c_str();
    }

}
