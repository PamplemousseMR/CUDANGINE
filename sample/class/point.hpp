#pragma once

namespace cudangine
{

class Point
{

public:

    __host__ __device__ Point() noexcept
    {
    }

    __host__ __device__ Point(int _x, int _y) noexcept :
        m_x(_x),
        m_y(_y)
    {
    }

    int m_x {0};

    int m_y {0};

};

}
