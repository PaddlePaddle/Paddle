/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <complex>
#include <type_traits>
#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/kernels/funcs/eigen/extensions.h"

// #include "paddle/phi/kernels/primitive/compute_primitives.h"

namespace phi {
namespace kps {

template <typename InT, typename MPType = InT, typename OutT = MPType>
struct SumOps {
  inline DEVICE MPType compute(MPType a, InT b) const {
    return reduce(a, static_cast<MPType>(b));
  }

  inline DEVICE MPType reduce(MPType a, MPType b) const { return a + b; }

  inline DEVICE OutT post_process(MPType a) const {
    return static_cast<OutT>(a);
  }

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  inline DEVICE MPType shfl_sync(unsigned mask, MPType data, int offset) const {
    return phi::backends::gpu::CudaShuffleDownSync(mask, data, offset);
  }
#endif

  SumOps() {}
};

template <typename InT, typename MPType = InT, typename OutT = MPType>
struct ProdOps {
  inline DEVICE MPType compute(MPType a, InT b) const {
    return reduce(a, static_cast<MPType>(b));
  }

  inline DEVICE MPType reduce(MPType a, MPType b) const { return a * b; }

  inline DEVICE OutT post_process(MPType a) const {
    return static_cast<OutT>(a);
  }

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  inline DEVICE MPType shfl_sync(unsigned mask, MPType data, int offset) const {
    return phi::backends::gpu::CudaShuffleDownSync(mask, data, offset);
  }
#endif

  ProdOps() {}
};

template <typename InT, typename MPType = InT, typename OutT = MPType>
struct MeanOps {
  MPType factor;

  inline DEVICE MPType compute(MPType a, InT b) const {
    return reduce(a, static_cast<MPType>(b));
  }

  inline DEVICE MPType reduce(MPType a, MPType b) const { return a + b; }

  inline DEVICE OutT post_process(MPType a) const {
    return static_cast<OutT>(a * factor);
  }

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  inline DEVICE MPType shfl_sync(unsigned mask, MPType data, int offset) const {
    return phi::backends::gpu::CudaShuffleDownSync(mask, data, offset);
  }
#endif

  explicit MeanOps(MPType factor) : factor(factor) {}
};

template <typename InT, typename MPType = InT, typename OutT = MPType>
struct MinOps {
  inline DEVICE MPType compute(MPType a, InT b) const {
    return reduce(a, static_cast<MPType>(b));
  }

  inline DEVICE MPType reduce(MPType a, MPType b) const {
    if constexpr ((std::is_floating_point<InT>::value) &&
                  (!(std::is_same<InT, int32_t>::value ||
                     (std::is_same<InT, int64_t>::value)))) {
      if (isnan(a)) {
        return a;
      }
      if (isnan(b)) {
        return b;
      }
    }
    return (a < b ? a : b);
  }

  inline DEVICE OutT post_process(MPType a) const {
    return static_cast<OutT>(a);
  }

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  inline DEVICE MPType shfl_sync(unsigned mask, MPType data, int offset) const {
    return phi::backends::gpu::CudaShuffleDownSync(mask, data, offset);
  }
#endif

  MinOps() {}
};

template <>
struct MinOps<bool, bool, bool> {
  inline DEVICE bool compute(bool a, bool b) const { return reduce(a, b); }

  inline DEVICE bool reduce(bool a, bool b) const { return a & b; }

  inline DEVICE bool post_process(bool a) const { return a; }

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  inline DEVICE bool shfl_sync(unsigned mask, bool data, int offset) const {
    return phi::backends::gpu::CudaShuffleDownSync(mask, data, offset);
  }
#endif

  MinOps() {}
};

template <typename InT, typename MPType = InT, typename OutT = MPType>
struct MaxOps {
  MPType factor;

  inline DEVICE MPType compute(MPType a, InT b) const {
    return reduce(a, static_cast<MPType>(b));
  }

  inline DEVICE MPType reduce(MPType a, MPType b) const {
    if constexpr ((std::is_floating_point<InT>::value) &&
                  (!(std::is_same<InT, int32_t>::value ||
                     (std::is_same<InT, int64_t>::value)))) {
      if (isnan(a)) {
        return a;
      }
      if (isnan(b)) {
        return b;
      }
    }
    return (a > b ? a : b);
  }

  inline DEVICE OutT post_process(MPType a) const {
    return static_cast<OutT>(a);
  }

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  inline DEVICE MPType shfl_sync(unsigned mask, MPType data, int offset) const {
    return phi::backends::gpu::CudaShuffleDownSync(mask, data, offset);
  }
#endif

  MaxOps() {}
};

template <>
struct MaxOps<bool, bool, bool> {
  inline DEVICE bool compute(bool a, bool b) const { return reduce(a, b); }

  inline DEVICE bool reduce(bool a, bool b) const { return a | b; }

  inline DEVICE bool post_process(bool a) const { return a; }

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  inline DEVICE bool shfl_sync(unsigned mask, bool data, int offset) const {
    return phi::backends::gpu::CudaShuffleDownSync(mask, data, offset);
  }
#endif

  MaxOps() {}
};

template <typename InT, typename MPType = InT, typename OutT = MPType>
struct LogicalAndOps {
  inline DEVICE MPType compute(MPType a, InT b) const {
    return reduce(a, static_cast<MPType>(b));
  }

  inline DEVICE MPType reduce(MPType a, MPType b) const { return (b && a); }

  inline DEVICE OutT post_process(MPType a) const {
    return static_cast<OutT>(a);
  }

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  inline DEVICE MPType shfl_sync(unsigned mask, MPType data, int offset) const {
    return phi::backends::gpu::CudaShuffleDownSync(mask, data, offset);
  }
#endif

  LogicalAndOps() {}
};

template <typename InT, typename MPType = InT, typename OutT = MPType>
struct LogicalOrOps {
  inline DEVICE MPType compute(MPType a, InT b) const {
    return reduce(a, static_cast<MPType>(b));
  }

  inline DEVICE MPType reduce(MPType a, MPType b) const { return (b || a); }

  inline DEVICE OutT post_process(MPType a) const {
    return static_cast<OutT>(a);
  }

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  inline DEVICE MPType shfl_sync(unsigned mask, MPType data, int offset) const {
    return phi::backends::gpu::CudaShuffleDownSync(mask, data, offset);
  }
#endif

  LogicalOrOps() {}
};
}  // namespace kps
}  // namespace phi
