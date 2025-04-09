// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/kernels/funcs/aligned_vector.h"

namespace phi {
namespace funcs {

template <typename T>
__device__ __inline__ T ClipFunc(const T v, const T min, const T max) {
  if (v > max) return max;
  if (v < min) return min;
  return v;
}

template <typename InType, typename OutType>
__forceinline__ __device__ OutType QuantHelperFunc(const InType input,
                                                   const float scale,
                                                   const int round_type,
                                                   const float max_bound,
                                                   const float min_bound) {
  float quant_value = max_bound * scale * input;

  if (round_type == 0) {
    quant_value = static_cast<float>(rint(quant_value));
  } else {
    quant_value = static_cast<float>(round(quant_value));
  }
  return static_cast<OutType>(
      ClipFunc<float>(quant_value, min_bound, max_bound));
}

template <typename InType, typename OutType>
__forceinline__ __device__ OutType FP8QuantHelperFunc(const InType input,
                                                      const float scale,
                                                      const int round_type,
                                                      const float max_bound,
                                                      const float min_bound) {
  float quant_value = max_bound * scale * input;
  return static_cast<OutType>(
      ClipFunc<float>(quant_value, min_bound, max_bound));
}

template <typename T>
struct Load {
  explicit Load(const T *src) : src_(src) {}

  template <int VecSize>
  __device__ void load(phi::AlignedVector<T, VecSize> *dst, int64_t idx) {
    phi::Load<T, VecSize>(src_ + idx, dst);
  }

  const T *src_;
};

template <typename T, bool Smooth = false>
struct Store {
  explicit Store(T *dst) : dst_(dst) {}

  template <int VecSize>
  __device__ void store(phi::AlignedVector<T, VecSize> &src, int64_t idx) {
    phi::Store<T, VecSize>(src, dst_ + idx);
  }

  T *dst_;
};

template <typename T>
struct Store<T, true> {
  Store(T *dst, const T *shift, const T *smooth, const int cols)
      : dst_(dst), shift_(shift), smooth_(smooth), cols_(cols) {}

  template <int VecSize>
  __device__ void store(phi::AlignedVector<T, VecSize> &src, int64_t idx) {
    using Vec = phi::AlignedVector<T, VecSize>;
    Vec shift_vec;
    Vec smooth_vec;

    phi::Load<T, VecSize>(shift_ + idx % cols_, &shift_vec);
    phi::Load<T, VecSize>(smooth_ + idx % cols_, &smooth_vec);
#pragma unroll
    for (int i = 0; i < VecSize; i++) {
      src[i] = (src[i] + shift_vec[i]) * smooth_vec[i];
    }
    phi::Store<T, VecSize>(src, dst_ + idx);
  }

  T *dst_;
  const T *shift_;
  const T *smooth_;
  const int cols_;
};

template <typename T>
struct DequantLoad {
  DequantLoad(const int32_t *src, const float *dequant_scales, const int cols)
      : src_(src), dequant_scales_(dequant_scales), cols_(cols) {}

  template <int VecSize>
  __device__ void load(phi::AlignedVector<T, VecSize> *dst, int64_t idx) {
    using SrcVec = phi::AlignedVector<int32_t, VecSize>;
    using DstVec = phi::AlignedVector<T, VecSize>;
    using ScaleVec = phi::AlignedVector<float, VecSize>;

    SrcVec src_vec;
    DstVec dst_vec;
    ScaleVec scale_vec;

    phi::Load<int32_t, VecSize>(src_ + idx, &src_vec);
    phi::Load<float, VecSize>(dequant_scales_ + idx % cols_, &scale_vec);
#pragma unroll
    for (int i = 0; i < VecSize; i++) {
      dst_vec[i] =
          static_cast<T>(static_cast<float>(src_vec[i]) * scale_vec[i]);
    }
    *dst = dst_vec;
  }

  const int32_t *src_;
  const float *dequant_scales_;
  const int cols_;
};

template <typename T, typename OutT, bool Smooth = false>
struct QuantStore {
  QuantStore(OutT *dst,
             const int quant_round_type,
             const float quant_scale,
             const float quant_max_bound,
             const float quant_min_bound)
      : dst_(dst),
        quant_round_type_(quant_round_type),
        quant_scale_(quant_scale),
        quant_max_bound_(quant_max_bound),
        quant_min_bound_(quant_min_bound) {}

  template <int VecSize>
  __device__ void store(phi::AlignedVector<T, VecSize> &src,  // NOLINT
                        int64_t idx) {                        // NOLINT
    using DstVec = phi::AlignedVector<OutT, VecSize>;

    DstVec dst_vec;
#pragma unroll
    for (int i = 0; i < VecSize; i++) {
      if constexpr (std::is_same_v<OutT, phi::dtype::float8_e4m3fn>) {
        dst_vec[i] = FP8QuantHelperFunc<float, OutT>(static_cast<float>(src[i]),
                                                     quant_scale_,
                                                     quant_round_type_,
                                                     quant_max_bound_,
                                                     quant_min_bound_);
      } else {
        dst_vec[i] = QuantHelperFunc<float, OutT>(static_cast<float>(src[i]),
                                                  quant_scale_,
                                                  quant_round_type_,
                                                  quant_max_bound_,
                                                  quant_min_bound_);
      }
    }

    phi::Store<OutT, VecSize>(dst_vec, dst_ + idx);
  }

  OutT *dst_;
  const int quant_round_type_;
  const float quant_scale_;
  const float quant_max_bound_;
  const float quant_min_bound_;
};

template <typename T>
struct QuantStore<T, int8_t, true> {
  QuantStore(int8_t *dst,
             const T *shift,
             const T *smooth,
             const int cols,
             const int quant_round_type,
             const float quant_scale,
             const float quant_max_bound,
             const float quant_min_bound)
      : dst_(dst),
        shift_(shift),
        smooth_(smooth),
        cols_(cols),
        quant_round_type_(quant_round_type),
        quant_scale_(quant_scale),
        quant_max_bound_(quant_max_bound),
        quant_min_bound_(quant_min_bound) {}

  template <int VecSize>
  __device__ void store(phi::AlignedVector<T, VecSize> &src,  // NOLINT
                        int64_t idx) {                        // NOLINT
    using DstVec = phi::AlignedVector<int8_t, VecSize>;
    using Vec = phi::AlignedVector<T, VecSize>;

    DstVec dst_vec;
    Vec shift_vec;
    Vec smooth_vec;

    phi::Load<T, VecSize>(shift_ + idx % cols_, &shift_vec);
    phi::Load<T, VecSize>(smooth_ + idx % cols_, &smooth_vec);
#pragma unroll
    for (int i = 0; i < VecSize; i++) {
      src[i] = (src[i] + shift_vec[i]) * smooth_vec[i];
      dst_vec[i] = QuantHelperFunc<float, int8_t>(static_cast<float>(src[i]),
                                                  quant_scale_,
                                                  quant_round_type_,
                                                  quant_max_bound_,
                                                  quant_min_bound_);
    }

    phi::Store<int8_t, VecSize>(dst_vec, dst_ + idx);
  }

  int8_t *dst_;
  const int quant_round_type_;
  const float quant_scale_;
  const float quant_max_bound_;
  const float quant_min_bound_;
  const T *shift_;
  const T *smooth_;
  const int cols_;
};

}  // namespace funcs
}  // namespace phi
