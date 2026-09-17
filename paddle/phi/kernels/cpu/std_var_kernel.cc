// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/std_var_kernel.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>
#include <vector>

#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/funcs/cascade_sum.h"
#include "paddle/phi/kernels/reduce_mean_kernel.h"

#if defined(PADDLE_WITH_OPENMP)
#include <omp.h>
#endif

namespace phi {
namespace {

using funcs::cascade_sum::DimInfo;
constexpr int64_t kGrainSize = 32768;

// Keep IEEE classifications when narrowing the double accumulator. In
// particular, NaN must not be converted to +Inf, and -Inf must keep its sign.
template <typename T>
inline T StoreResult(double value) {
  if (std::isnan(value)) {
    return static_cast<T>(std::numeric_limits<double>::quiet_NaN());
  }
  if (std::isinf(value)) {
    return static_cast<T>(value);
  }
  if constexpr (std::is_same_v<T, float>) {
    if (value > static_cast<double>(std::numeric_limits<float>::max())) {
      return std::numeric_limits<float>::infinity();
    }
    if (value < static_cast<double>(std::numeric_limits<float>::lowest())) {
      return -std::numeric_limits<float>::infinity();
    }
  }
  return static_cast<T>(value);
}

template <typename Fn>
void ParallelFor(int64_t count, Fn&& fn) {
#if defined(PADDLE_WITH_OPENMP)
#pragma omp parallel for schedule(static) if (count >= kGrainSize)
#endif
  for (int64_t i = 0; i < count; ++i) {
    fn(i);
  }
}

inline int64_t Numel(const std::vector<DimInfo>& dims) {
  int64_t result = 1;
  for (const auto& dim : dims) {
    result *= dim.size;
  }
  return result;
}

inline int64_t OffsetForLinear(const std::vector<DimInfo>& dims,
                               int64_t linear) {
  int64_t offset = 0;
  for (const auto& dim : dims) {
    const int64_t index = linear % dim.size;
    linear /= dim.size;
    offset += index * dim.in_stride;
  }
  return offset;
}

inline std::vector<DimInfo> MakeElementwiseDims(
    const std::vector<int64_t>& shape,
    const std::vector<int64_t>& strides,
    int64_t elem_size) {
  std::vector<int64_t> byte_strides(strides.size());
  for (size_t i = 0; i < strides.size(); ++i) {
    byte_strides[i] = strides[i] * elem_size;
  }
  return funcs::cascade_sum::BuildDims(
      shape, byte_strides, byte_strides, /*is_reduction=*/false);
}

// PyTorch's std_var_all_cpu uses a mean followed by a squared-difference
// reduction for a single output. Each parallel chunk computes a local sum and
// partials are folded in chunk order, keeping the serial result deterministic.
template <typename T>
double SumSquaredDifferences(const T* x_data,
                             double mean,
                             const std::vector<DimInfo>& dims) {
  const int64_t total = Numel(dims);
  const int64_t chunks = (total + kGrainSize - 1) / kGrainSize;
  std::vector<double> partials(static_cast<size_t>(chunks), 0.0);
  const char* base = reinterpret_cast<const char*>(x_data);

  ParallelFor(chunks, [&](int64_t chunk) {
    const int64_t begin = chunk * kGrainSize;
    const int64_t end = std::min(total, begin + kGrainSize);
    double local = 0.0;
    for (int64_t linear = begin; linear < end; ++linear) {
      const auto offset = OffsetForLinear(dims, linear);
      const double value =
          static_cast<double>(*reinterpret_cast<const T*>(base + offset));
      const double delta = value - mean;
      local += delta * delta;
    }
    partials[static_cast<size_t>(chunk)] = local;
  });

  double result = 0.0;
  for (double partial : partials) {
    result += partial;
  }
  return result;
}

struct WelfordAccumulator {
  double mean{0.0};
  double m2{0.0};
  int64_t count{0};
};

inline void WelfordUpdate(WelfordAccumulator* acc, double value) {
  ++acc->count;
  const double count = static_cast<double>(acc->count);
  const double delta = value - acc->mean;
  const double new_mean = acc->mean + delta / count;
  acc->m2 += delta * (value - new_mean);
  acc->mean = new_mean;
}

template <typename T>
void WelfordReduce(const T* x_data,
                   const std::vector<DimInfo>& dims,
                   T* out_data,
                   double correction,
                   bool take_sqrt) {
  int reduced_dims = 0;
  while (reduced_dims < static_cast<int>(dims.size()) &&
         dims[static_cast<size_t>(reduced_dims)].out_stride == 0) {
    ++reduced_dims;
  }

  // WelfordReduce is only reached for multi-output reductions (a single
  // output goes through the two-pass path above), which implies a non-empty
  // partial axis list, so dims always leads with at least one reduced
  // dimension after reorder_dimensions().
  std::vector<DimInfo> reduce_dims(dims.begin(), dims.begin() + reduced_dims);

  const int64_t reduce_numel = Numel(reduce_dims);
  int64_t output_numel = 1;
  for (size_t i = static_cast<size_t>(reduced_dims); i < dims.size(); ++i) {
    output_numel *= dims[i].size;
  }

  const char* input_base = reinterpret_cast<const char*>(x_data);
  ParallelFor(output_numel, [&](int64_t linear) {
    int64_t input_offset = 0;
    int64_t output_offset = 0;
    int64_t rest = linear;
    for (size_t i = static_cast<size_t>(reduced_dims); i < dims.size(); ++i) {
      const int64_t index = rest % dims[i].size;
      rest /= dims[i].size;
      input_offset += index * dims[i].in_stride;
      output_offset += index * dims[i].out_stride;
    }

    WelfordAccumulator acc;
    for (int64_t reduce_linear = 0; reduce_linear < reduce_numel;
         ++reduce_linear) {
      const int64_t offset =
          input_offset + OffsetForLinear(reduce_dims, reduce_linear);
      const double value =
          static_cast<double>(*reinterpret_cast<const T*>(input_base + offset));
      WelfordUpdate(&acc, value);
    }

    const double divisor =
        std::max(0.0, static_cast<double>(acc.count) - correction);
    double result = acc.m2 / divisor;
    if (take_sqrt) {
      result = std::sqrt(result);
    }
    auto* output =
        reinterpret_cast<T*>(reinterpret_cast<char*>(out_data) + output_offset);
    *output = StoreResult<T>(result);
  });
}

template <typename T, typename Context>
void StdVarImpl(const Context& dev_ctx,
                const DenseTensor& x,
                const std::vector<int64_t>& axis,
                double correction,
                bool take_sqrt,
                DenseTensor* out) {
  if (x.numel() == 0) {
    Full<T, Context>(dev_ctx,
                     out->dims(),
                     static_cast<T>(std::numeric_limits<double>::quiet_NaN()),
                     out);
    return;
  }

  const auto axes = funcs::NormalizeReduceAxes(x.dims(), axis, false);
  T* out_data = dev_ctx.template Alloc<T>(out);
  const auto shape = common::vectorize(x.dims());
  const auto strides = common::vectorize(x.strides());

  // CPU float/double reductions with one output use the two-pass implementation
  // in ATen.
  if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
    if (out->numel() == 1) {
      DenseTensor mean = Mean<T, Context>(dev_ctx, x, axes, true);
      const double mean_value = static_cast<double>(mean.data<T>()[0]);
      const auto dims = MakeElementwiseDims(shape, strides, sizeof(T));
      const double sum =
          SumSquaredDifferences<T>(x.data<T>(), mean_value, dims);
      const double divisor =
          std::max(0.0, static_cast<double>(x.numel()) - correction);
      double result = sum / divisor;
      if (take_sqrt) {
        result = std::sqrt(result);
      }
      out_data[0] = StoreResult<T>(result);
      return;
    }
  }

  const auto dims = funcs::cascade_sum::MakeReduceDims(
      shape, strides, axes, static_cast<int64_t>(sizeof(T)));
  WelfordReduce<T>(x.data<T>(), dims, out_data, correction, take_sqrt);
}

}  // namespace

template <typename T, typename Context>
void VarKernel(const Context& dev_ctx,
               const DenseTensor& x,
               const std::vector<int64_t>& axis,
               bool keepdim,
               bool unbiased,
               double correction,
               DenseTensor* out) {
  (void)keepdim;
  (void)unbiased;
  StdVarImpl<T, Context>(
      dev_ctx, x, axis, correction, /*take_sqrt=*/false, out);
}

template <typename T, typename Context>
void StdKernel(const Context& dev_ctx,
               const DenseTensor& x,
               const std::vector<int64_t>& axis,
               bool keepdim,
               bool unbiased,
               double correction,
               DenseTensor* out) {
  (void)keepdim;
  (void)unbiased;
  StdVarImpl<T, Context>(dev_ctx, x, axis, correction, /*take_sqrt=*/true, out);
}

}  // namespace phi

PD_REGISTER_KERNEL(var, CPU, ALL_LAYOUT, phi::VarKernel, float, double) {}
PD_REGISTER_KERNEL(std, CPU, ALL_LAYOUT, phi::StdKernel, float, double) {}
