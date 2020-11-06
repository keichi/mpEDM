#ifndef __NEAREST_NEIGHBORS_CPU_H__
#define __NEAREST_NEIGHBORS_CPU_H__

#include "lut.h"
#include "nearest_neighbors.h"

class NearestNeighborsCPU : public NearestNeighbors
{
public:
    NearestNeighborsCPU(uint32_t tau, uint32_t Tp, bool verbose);

    void compute_lut(LUT &out, const Series &library, const Series &target,
                     uint32_t E, uint32_t top_k) override;

protected:
    std::vector<float> distances;
};

template <class T> class Counter
{
private:
    T i_;

public:
    using difference_type = T;
    using value_type = T;
    using pointer = T;
    using reference = T &;
    using iterator_category = std::input_iterator_tag;

    Counter(T i) : i_(i) {}
    T operator*() const noexcept { return i_; }
    Counter &operator++() noexcept
    {
        i_++;
        return *this;
    }
    bool operator==(const Counter &rhs) const { return i_ == rhs.i_; }
    bool operator!=(const Counter &rhs) const { return i_ != rhs.i_; }
};

#endif
