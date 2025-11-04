#pragma once

namespace cutlass {

template <int NUnroll>
struct Unroll {

  template <class F>
  CUTLASS_HOST_DEVICE void operator()(F&& f) const {
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < NUnroll; ++i) {
      f(i);
    }
  }
  
};

} // namespace cutlass