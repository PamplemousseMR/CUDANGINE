#pragma once

#include <string>

namespace cudangine
{

class Exception : public std::exception
{

public:

    Exception(const cudaError_t&) noexcept;

    virtual const char* what() const noexcept override;

private:

    std::string m_message;

};

}
