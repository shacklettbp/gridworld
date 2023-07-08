#pragma once

namespace madgrid {

enum class CellFlag : uint32_t {
    Nothing = 0,
    Wall    = 1 << 0,
    End     = 1 << 1,
};

struct Cell {
    float reward;
    CellFlag flags;
};

struct GridState {
    const Cell *cells;

    uint32_t startX;
    uint32_t startY;
    uint32_t width;
    uint32_t height;
};

inline CellFlag & operator|=(CellFlag &a, CellFlag b)
{
    a = CellFlag(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
    return a;
}

inline bool operator&(CellFlag a, CellFlag b)
{
    return (static_cast<uint32_t>(a) & static_cast<uint32_t>(b)) > 0;
}

inline CellFlag operator|(CellFlag a, CellFlag b)
{
    a |= b;

    return a;
}

}
