#pragma once

#include <string>

namespace cudangine
{

class Exception : public std::exception
{

public:

    Exception(const cudaError_t&) noexcept;

    virtual ~Exception() override;

    const char* what() const noexcept override;

private:

    std::string m_message;

};

}
