// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/truncated_gaussian_random_kernel.h"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/random.h>
#include <thrust/transform.h>

#include <algorithm>
#include <cmath>
#include <limits>

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_info.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/distribution_helper.h"

namespace phi {

// Legacy generator, only used when the op carries a non-zero `seed` attribute.
// It keeps the historical inverse-CDF behavior so that graphs pinning an op
// seed keep reproducing the same values.
template <typename T, typename MT>
struct GPUTruncatedNormal {
  MT mean, std, a, b;
  MT a_normal_cdf;
  MT b_normal_cdf;
  unsigned int seed;
  MT numeric_min;

  __host__ __device__
  GPUTruncatedNormal(MT mean, MT std, MT numeric_min, int seed, MT a, MT b)
      : mean(mean), std(std), seed(seed), numeric_min(numeric_min), a(a), b(b) {
    a_normal_cdf = (1.0 + erff((a - mean) / std / sqrtf(2.0))) / 2.0;
    b_normal_cdf = (1.0 + erff((b - mean) / std / sqrtf(2.0))) / 2.0;
  }

  __host__ __device__ T operator()(const unsigned int n) const {
    thrust::minstd_rand rng;
    rng.seed(seed);
    thrust::uniform_real_distribution<MT> dist(numeric_min, 1);
    rng.discard(n);
    MT value = dist(rng);
    auto p = a_normal_cdf + (b_normal_cdf - a_normal_cdf) * value;
    MT ret = std::sqrt(2.0) * erfinvf(2 * p - 1) * std + mean;
    return static_cast<T>(std::clamp(ret, a, b));
  }
};

// Marks `flag` when any element of `data` falls outside [lo, hi]. Concurrent
// threads only ever store the same value, so the race on `flag` is benign.
template <typename T, typename MT>
__global__ void MarkOutOfRangeKernel(
    const T* data, int64_t numel, MT lo, MT hi, int* flag) {
  int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < numel;
       i += stride) {
    MT value = static_cast<MT>(data[i]);
    if (value < lo || value > hi) {
      *flag = 1;
    }
  }
}

// Fuses torch's `result = where(mask, fresh, result)` with the mask
// recomputation of the next loop iteration: elements left untouched were
// already inside [lo, hi], so `flag` ends up as `mask.any()` of the update.
template <typename T, typename MT>
__global__ void ReplaceOutOfRangeKernel(
    T* data, const T* fresh, int64_t numel, MT lo, MT hi, int* flag) {
  int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < numel;
       i += stride) {
    MT value = static_cast<MT>(data[i]);
    if (value < lo || value > hi) {
      T replacement = fresh[i];
      data[i] = replacement;
      MT new_value = static_cast<MT>(replacement);
      if (new_value < lo || new_value > hi) {
        *flag = 1;
      }
    }
  }
}

// Port of the `p <= 0.3` branch of torch's `_no_grad_trunc_normal_`: uniform
// proposals on [a, b] accepted with probability pdf(x)/pdf(mode).
//
// The log-pdf is evaluated with the same per-op rounding as the chain
// `candidates.sub_(mean).div_(std).pow_(2).mul_(-0.5).sub_(log_peak)`, which
// stores an intermediate of dtype T after every step. That rounding is
// observable for float16/bfloat16, hence the casts.
template <typename T, typename MT>
__global__ void AcceptRejectKernel(T* result,
                                   const T* proposal,
                                   const T* accept_rand,
                                   bool* pending,
                                   int64_t numel,
                                   MT mean,
                                   MT std,
                                   MT log_peak,
                                   bool is_first_round,
                                   int* flag) {
  int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < numel;
       i += stride) {
    if (!is_first_round && !pending[i]) {
      continue;
    }
    T candidate = is_first_round ? result[i] : proposal[i];
    if (!is_first_round) {
      result[i] = candidate;
    }

    MT value = static_cast<MT>(candidate);
    value = static_cast<MT>(static_cast<T>(value - mean));
    value = static_cast<MT>(static_cast<T>(value / std));
    value = static_cast<MT>(static_cast<T>(value * value));
    value = static_cast<MT>(static_cast<T>(value * static_cast<MT>(-0.5)));
    MT log_pdf = static_cast<MT>(static_cast<T>(value - log_peak));

    MT log_u =
        static_cast<MT>(static_cast<T>(log(static_cast<MT>(accept_rand[i]))));

    bool rejected = log_u > log_pdf;
    pending[i] = rejected;
    if (rejected) {
      *flag = 1;
    }
  }
}

template <typename Context>
class OutOfRangeFlag {
 public:
  explicit OutOfRangeFlag(const Context& dev_ctx) : dev_ctx_(dev_ctx) {
    flag_.Resize({1});
    data_ = dev_ctx_.template Alloc<int>(&flag_);
  }

  int* data() const { return data_; }

  void Reset() {
    phi::backends::gpu::GpuMemsetAsync(
        data_, 0, sizeof(int), dev_ctx_.stream());
  }

  // Mirrors the `mask.any()` device-to-host synchronization that torch's
  // python-level loop performs once per iteration.
  bool Read() {
    int host_flag = 0;
    memory_utils::Copy(phi::CPUPlace(),
                       &host_flag,
                       dev_ctx_.GetPlace(),
                       data_,
                       sizeof(int),
                       dev_ctx_.stream());
    dev_ctx_.Wait();
    return host_flag != 0;
  }

 private:
  const Context& dev_ctx_;
  DenseTensor flag_;
  int* data_ = nullptr;
};

// Rejection sampling on plain normal draws, matching the `p > 0.3` branch of
// torch's `_no_grad_trunc_normal_`. Every round redraws a full-size tensor, so
// the philox offsets consumed here line up with torch's sequence of
// `normal_()` launches.
template <typename T, typename Context>
void RejectionSampleFromNormal(const Context& dev_ctx,
                               double mean,
                               double std,
                               double a,
                               double b,
                               DenseTensor* out) {
  using MT = typename MPTypeTrait<T>::Type;
  int64_t numel = out->numel();

  funcs::normal_distribution<MT> dist;
  funcs::normal_transform<MT> trans(static_cast<MT>(mean),
                                    static_cast<MT>(std));
  funcs::distribution_and_transform<T>(dev_ctx, out, dist, trans);

  // torch compares against bounds that were first rounded to the output dtype.
  MT lo = static_cast<MT>(static_cast<T>(a));
  MT hi = static_cast<MT>(static_cast<T>(b));

  auto config = phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, numel);
  OutOfRangeFlag<Context> flag(dev_ctx);

  flag.Reset();
  MarkOutOfRangeKernel<T, MT>
      <<<config.block_per_grid, config.thread_per_block, 0, dev_ctx.stream()>>>(
          out->data<T>(), numel, lo, hi, flag.data());
  if (!flag.Read()) {
    return;
  }

  DenseTensor fresh;
  fresh.Resize(out->dims());
  dev_ctx.template Alloc<T>(&fresh);
  while (true) {
    funcs::distribution_and_transform<T>(dev_ctx, &fresh, dist, trans);
    flag.Reset();
    ReplaceOutOfRangeKernel<T, MT><<<config.block_per_grid,
                                     config.thread_per_block,
                                     0,
                                     dev_ctx.stream()>>>(
        out->data<T>(), fresh.data<T>(), numel, lo, hi, flag.data());
    if (!flag.Read()) {
      return;
    }
  }
}

// Rejection sampling on uniform proposals, matching the `p <= 0.3` branch of
// torch's `_no_grad_trunc_normal_`.
template <typename T, typename Context>
void RejectionSampleFromUniform(const Context& dev_ctx,
                                double mean,
                                double std,
                                double a,
                                double b,
                                DenseTensor* out) {
  using MT = typename MPTypeTrait<T>::Type;
  int64_t numel = out->numel();

  double mode = std::max(a, std::min(mean, b));
  double log_peak = -0.5 * std::pow((mode - mean) / std, 2);

  funcs::uniform_distribution<MT> dist;
  // Matches torch's `uniform_kernel`, which folds the bounds to the output
  // dtype before deriving the range: `range = MT(T(to)) - MT(T(from))`.
  funcs::uniform_real_transform<MT, T> proposal_trans(
      static_cast<MT>(static_cast<T>(a)), static_cast<MT>(static_cast<T>(b)));
  funcs::uniform_real_transform<MT, T> accept_trans(
      static_cast<MT>(static_cast<T>(0.0)),
      static_cast<MT>(static_cast<T>(1.0)));

  DenseTensor accept_rand;
  accept_rand.Resize(out->dims());
  dev_ctx.template Alloc<T>(&accept_rand);

  DenseTensor pending;
  pending.Resize(out->dims());
  bool* pending_data = dev_ctx.template Alloc<bool>(&pending);

  auto config = phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, numel);
  OutOfRangeFlag<Context> flag(dev_ctx);

  // First round proposes directly into `out`, as torch does.
  funcs::distribution_and_transform<T>(dev_ctx, out, dist, proposal_trans);
  funcs::distribution_and_transform<T>(
      dev_ctx, &accept_rand, dist, accept_trans);
  flag.Reset();
  AcceptRejectKernel<T, MT>
      <<<config.block_per_grid, config.thread_per_block, 0, dev_ctx.stream()>>>(
          out->data<T>(),
          nullptr,
          accept_rand.data<T>(),
          pending_data,
          numel,
          static_cast<MT>(mean),
          static_cast<MT>(std),
          static_cast<MT>(log_peak),
          true,
          flag.data());
  if (!flag.Read()) {
    return;
  }

  DenseTensor proposal;
  proposal.Resize(out->dims());
  dev_ctx.template Alloc<T>(&proposal);
  while (true) {
    funcs::distribution_and_transform<T>(
        dev_ctx, &proposal, dist, proposal_trans);
    funcs::distribution_and_transform<T>(
        dev_ctx, &accept_rand, dist, accept_trans);
    flag.Reset();
    AcceptRejectKernel<T, MT><<<config.block_per_grid,
                                config.thread_per_block,
                                0,
                                dev_ctx.stream()>>>(out->data<T>(),
                                                    proposal.data<T>(),
                                                    accept_rand.data<T>(),
                                                    pending_data,
                                                    numel,
                                                    static_cast<MT>(mean),
                                                    static_cast<MT>(std),
                                                    static_cast<MT>(log_peak),
                                                    false,
                                                    flag.data());
    if (!flag.Read()) {
      return;
    }
  }
}

template <typename T, typename Context>
void TruncatedGaussianRandomKernel(const Context& dev_ctx,
                                   const std::vector<int>& shape,
                                   double mean,
                                   double std,
                                   int seed,
                                   double a,
                                   double b,
                                   DataType dtype,
                                   DenseTensor* out) {
  T* data = dev_ctx.template Alloc<T>(out);

  using MT = typename MPTypeTrait<T>::Type;

  int64_t size = out->numel();
  if (size == 0) {
    return;
  }

  if (seed != 0) {
    // use OP seed, keeping the legacy inverse-CDF sampling
    thrust::counting_iterator<int64_t> index_sequence_begin(0);
    thrust::transform(index_sequence_begin,
                      index_sequence_begin + size,
                      thrust::device_ptr<T>(data),
                      GPUTruncatedNormal<T, MT>(static_cast<MT>(mean),
                                                static_cast<MT>(std),
                                                std::numeric_limits<MT>::min(),
                                                seed,
                                                static_cast<MT>(a),
                                                static_cast<MT>(b)));
    return;
  }

  // use global Generator seed. Both branches below reproduce torch's
  // `torch.nn.init.trunc_normal_` bit-for-bit, including which of the two
  // rejection schemes is picked.
  auto normal_cdf = [](double x) {
    return (1.0 + std::erf(x / std::sqrt(2.0))) / 2.0;
  };
  double p = normal_cdf((b - mean) / std) - normal_cdf((a - mean) / std);

  if (p > 0.3) {
    RejectionSampleFromNormal<T, Context>(dev_ctx, mean, std, a, b, out);
  } else {
    RejectionSampleFromUniform<T, Context>(dev_ctx, mean, std, a, b, out);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(truncated_gaussian_random,
                   GPU,
                   ALL_LAYOUT,
                   phi::TruncatedGaussianRandomKernel,
                   float,
                   double,
                   phi::float16,
                   phi::bfloat16) {}
