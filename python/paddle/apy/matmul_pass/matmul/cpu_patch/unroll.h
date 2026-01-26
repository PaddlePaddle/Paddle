#pragma once

#include <type_traits>
#include "cpu_patch/data_type.h"
#include <utility>

namespace gops {

template <int NUnroll>
struct unroll {

    template <class F>
    constexpr void operator()(F f) const {
        unroll_impl(f, std::make_index_sequence<NUnroll>{});
    }

private:
    template <class F, std::size_t... Is>
    constexpr void unroll_impl(F f, std::index_sequence<Is...>) const {
        (f(Number<Is>{}), ...);
    }

}; 

} // namespace ck