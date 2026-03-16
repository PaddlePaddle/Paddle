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

// CUDA and HIP use ReduceGpuKernel API
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)

#include <complex>
#include <cstring>
#include <type_traits>
#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/kernels/funcs/eigen/extensions.h"

#include "paddle/phi/kernels/funcs/p_norm_utils.h"

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

  inline DEVICE MPType shfl_sync(unsigned mask, MPType data, int offset) const {
    return phi::backends::gpu::CudaShuffleDownSync(mask, data, offset);
  }

  SumOps() {}
};

namespace detail {

template <typename T>
HOSTDEVICE inline bool IsNan(T val) {
  if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
    return isnan(val);
  }
  if constexpr (std::is_same_v<T, phi::dtype::float16> ||
                std::is_same_v<T, phi::dtype::bfloat16> ||
                std::is_same_v<T, phi::dtype::complex<float>> ||
                std::is_same_v<T, phi::dtype::complex<double>>) {
    return phi::dtype::isnan(val);
  }
  return false;  // int or bool
}

}  // namespace detail

template <typename InT, typename MPType = InT, typename OutT = MPType>
struct NansumOps {
  inline DEVICE MPType compute(MPType a, InT b) const {
    return reduce(a, static_cast<MPType>(b));
  }

  inline DEVICE MPType reduce(MPType a, MPType b) const {
    if (detail::IsNan(b)) return a;
    return a + b;
  }

  inline DEVICE OutT post_process(MPType a) const {
    return static_cast<OutT>(a);
  }

  inline DEVICE MPType shfl_sync(unsigned mask, MPType data, int offset) const {
    return phi::backends::gpu::CudaShuffleDownSync(mask, data, offset);
  }

  NansumOps() {}
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

  inline DEVICE MPType shfl_sync(unsigned mask, MPType data, int offset) const {
    return phi::backends::gpu::CudaShuffleDownSync(mask, data, offset);
  }

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

  inline DEVICE MPType shfl_sync(unsigned mask, MPType data, int offset) const {
    return phi::backends::gpu::CudaShuffleDownSync(mask, data, offset);
  }

  explicit MeanOps(MPType factor) : factor(factor) {}
};

template <typename InT, typename MPType = InT, typename OutT = MPType>
struct MinOps {
  inline DEVICE MPType compute(MPType a, InT b) const {
    return reduce(a, static_cast<MPType>(b));
  }

  inline DEVICE MPType reduce(MPType a, MPType b) const {
    if constexpr (std::is_floating_point<InT>::value) {
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

  inline DEVICE MPType shfl_sync(unsigned mask, MPType data, int offset) const {
    return phi::backends::gpu::CudaShuffleDownSync(mask, data, offset);
  }

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
    if constexpr (std::is_floating_point<InT>::value) {
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

  inline DEVICE MPType shfl_sync(unsigned mask, MPType data, int offset) const {
    return phi::backends::gpu::CudaShuffleDownSync(mask, data, offset);
  }

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
struct AbsMaxOps {
  inline DEVICE MPType compute(MPType a, InT b) const {
    MPType b_ = static_cast<MPType>(inline_abs(b));
    return reduce(a, b_);
  }

  inline DEVICE MPType reduce(MPType a, MPType b) const {
    if constexpr (std::is_floating_point<InT>::value) {
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

  inline DEVICE MPType shfl_sync(unsigned mask, MPType data, int offset) const {
    return phi::backends::gpu::CudaShuffleDownSync(mask, data, offset);
  }

  AbsMaxOps() {}
};

template <typename InT, typename MPType = InT, typename OutT = MPType>
struct AbsMinOps {
  inline DEVICE MPType compute(MPType a, InT b) const {
    MPType b_ = static_cast<MPType>(inline_abs(b));
    return reduce(a, b_);
  }

  inline DEVICE MPType reduce(MPType a, MPType b) const {
    if constexpr (std::is_floating_point<InT>::value) {
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

  inline DEVICE MPType shfl_sync(unsigned mask, MPType data, int offset) const {
    return phi::backends::gpu::CudaShuffleDownSync(mask, data, offset);
  }

  AbsMinOps() {}
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

  inline DEVICE MPType shfl_sync(unsigned mask, MPType data, int offset) const {
    return phi::backends::gpu::CudaShuffleDownSync(mask, data, offset);
  }

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

  inline DEVICE MPType shfl_sync(unsigned mask, MPType data, int offset) const {
    return phi::backends::gpu::CudaShuffleDownSync(mask, data, offset);
  }

  LogicalOrOps() {}
};

template <typename InT, typename MPType = InT, typename OutT = MPType>
struct L0NormOps {
  inline DEVICE MPType compute(MPType a, InT b) const {
    MPType b_ = (b == static_cast<InT>(0)) ? static_cast<MPType>(0)
                                           : static_cast<MPType>(1);
    return reduce(a, b_);
  }

  inline DEVICE MPType reduce(MPType a, MPType b) const { return (a + b); }

  inline DEVICE OutT post_process(MPType a) const {
    return static_cast<OutT>(a);
  }

  inline DEVICE MPType shfl_sync(unsigned mask, MPType data, int offset) const {
    return phi::backends::gpu::CudaShuffleDownSync(mask, data, offset);
  }

  L0NormOps() {}
};

template <typename InT, typename MPType = InT, typename OutT = MPType>
struct L1NormOps {
  inline DEVICE MPType compute(MPType a, InT b) const {
    MPType b_ = static_cast<MPType>(inline_abs(b));
    return reduce(a, b_);
  }

  inline DEVICE MPType reduce(MPType a, MPType b) const { return (a + b); }

  inline DEVICE OutT post_process(MPType a) const {
    return static_cast<OutT>(a);
  }

  inline DEVICE MPType shfl_sync(unsigned mask, MPType data, int offset) const {
    return phi::backends::gpu::CudaShuffleDownSync(mask, data, offset);
  }

  L1NormOps() {}
};

template <typename InT, typename MPType = InT, typename OutT = MPType>
struct L2NormOps {
  inline DEVICE MPType compute(MPType a, InT b) const {
    MPType b_ = static_cast<MPType>(b) * static_cast<MPType>(b);
    return reduce(a, b_);
  }

  inline DEVICE MPType reduce(MPType a, MPType b) const { return (a + b); }

  inline DEVICE OutT post_process(MPType a) const {
    return static_cast<OutT>(std::sqrt(a));
  }

  inline DEVICE MPType shfl_sync(unsigned mask, MPType data, int offset) const {
    return phi::backends::gpu::CudaShuffleDownSync(mask, data, offset);
  }

  L2NormOps() {}
};

template <typename InT, typename MPType = InT, typename OutT = MPType>
struct GenericPNormOps {
  MPType norm_;

  inline DEVICE MPType compute(MPType a, InT b) const {
    MPType b_ = std::pow(static_cast<MPType>(inline_abs(b)), norm_);
    return reduce(a, b_);
  }

  inline DEVICE MPType reduce(MPType a, MPType b) const { return (a + b); }

  inline DEVICE OutT post_process(MPType a) const {
    return static_cast<OutT>(std::pow(a, static_cast<MPType>(1.0) / norm_));
  }

  inline DEVICE MPType shfl_sync(unsigned mask, MPType data, int offset) const {
    return phi::backends::gpu::CudaShuffleDownSync(mask, data, offset);
  }

  explicit GenericPNormOps(MPType p_norm) : norm_(p_norm) {}
};

}  // namespace kps
}  // namespace phi

#endif
