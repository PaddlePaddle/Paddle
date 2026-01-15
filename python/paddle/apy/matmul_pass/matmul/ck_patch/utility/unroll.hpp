#pragma once

#include "ck/utility/functional2.hpp"

namespace ck {

template <int NUnroll>
struct unroll {

    template <class F>
    __host__ __device__ constexpr void operator()(F f) const {
        ck::static_for<0, NUnroll, 1>{}(f);
    }

}; 

} // namespace ck