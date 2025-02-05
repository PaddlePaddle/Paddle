/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/kernels/multinomial_kernel.h"
#include "paddle/phi/kernels/funcs/multinomial_kernel_helper.h"

#ifdef __NVCC__
#include "cub/cub.cuh"
#endif
#ifdef __HIPCC__
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#endif

#include "paddle/common/ddim.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/arg_min_max_kernel.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/funcs/distribution_helper.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/for_range.h"
#include "paddle/phi/kernels/funcs/inclusive_scan.h"
#include "paddle/phi/kernels/funcs/multinomial_functor.h"
#include "paddle/phi/kernels/top_k_kernel.h"

namespace phi {

template <typename T, typename MT>
__global__ void NormalizeProbability(MT* norm_probs,
                                     const T* in_data,
                                     MT* sum_rows,
                                     int64_t num_distributions,
                                     int64_t num_categories) {
  int id = threadIdx.x + blockIdx.x * blockDim.x +
           blockIdx.y * gridDim.x * blockDim.x;
  if (id < num_distributions * num_categories) {
    PADDLE_ENFORCE(
        static_cast<MT>(in_data[id]) >= 0.0,
        "The input of multinomial distribution should be >= 0, but got %f.",
        static_cast<MT>(in_data[id]));
    int64_t row_id = id / num_categories;
    PADDLE_ENFORCE(sum_rows[row_id] > 0.0,
                   "The sum of one multinomial distribution probability should "
                   "be > 0, but got %f.",
                   sum_rows[row_id]);
    norm_probs[id] = static_cast<MT>(in_data[id]) / sum_rows[row_id];
  }
}

template <typename T>
__device__ int binarySearchFunctor(T* cumulative_probs_data,
                                   T* norm_probs_data,
                                   int num_categories,
                                   T rng_number) {
  int left = 0;
  int right = num_categories;

  while (right - left > 0) {
    int mid = left + (right - left) / 2;

    T temp_prob = cumulative_probs_data[mid];
    if (temp_prob < rng_number) {
      left = mid + 1;
    } else {
      right = mid;
    }
  }

  if (left == num_categories) {
    left = num_categories - 1;
  }

  while (left >= 1 && norm_probs_data[left] == 0) left--;

  return left;
}

template <typename T>
__global__ void sampleMultinomialWithReplacement(
    const int64_t num_samples,
    int64_t* out_data,
    const int64_t num_distributions,
    const int64_t num_categories,
    T* cumulative_probs_data,
    T* norm_probs_data,
    uint64_t seed,
    uint64_t offset) {
  // use binary search to get the selected category sample id.
  // let cumulative_probs_data[id-1] < rng_number < cumulative_probs_data[id].
  size_t idx = gridDim.x * blockDim.x * blockIdx.y + blockDim.x * blockIdx.x +
               threadIdx.x;

#if defined(__NVCC__)
  curandStatePhilox4_32_10_t state;
  curand_init(seed, idx, offset, &state);
#else
  hiprandStatePhilox4_32_10_t state;
  hiprand_init(seed, idx, offset, &state);
#endif

  int sample = blockIdx.x * blockDim.x + threadIdx.x;
  for (int dist = blockIdx.y; dist < num_distributions; dist += gridDim.y) {
    if (sample < num_samples) {
#if defined(__NVCC__)
      T rng_number = static_cast<T>(curand_uniform4(&state).x);
#else
      T rng_number = static_cast<T>(hiprand_uniform4(&state).x);
#endif
      int selected_category =
          binarySearchFunctor<T>(cumulative_probs_data + dist * num_categories,
                                 norm_probs_data + dist * num_categories,
                                 num_categories,
                                 rng_number);

      out_data[sample + dist * num_samples] = selected_category;
    }
  }
}

template <typename T, typename Context>
void MultinomialKernel(const Context& dev_ctx,
                       const DenseTensor& x,
                       const Scalar& num_samples,
                       bool replacement,
                       DenseTensor* out) {
  using MT = typename phi::dtype::MPTypeTrait<T>::Type;

  auto int_num_samples = num_samples.to<int>();
  auto* in_data = x.data<T>();
  int64_t* out_data = dev_ctx.template Alloc<int64_t>(out);
  auto in_dims = x.dims();
  int64_t dim_size = in_dims.size();
  const int64_t num_categories = in_dims[dim_size - 1];
  const int64_t num_distributions = dim_size > 1 ? in_dims[dim_size - 2] : 1;
  // If replacement is False, it's not a replaceable sample. Every category
  // can be used only once.
  if (!replacement) {
    MultinomialInputChecker<T, Context>(dev_ctx, x, num_samples);
    // Refer to [gumbel softmax algorithm]
    DenseTensor rand = EmptyLike<T, Context>(dev_ctx, x);
    T* rand_data = rand.data<T>();
    funcs::uniform_distribution<MT> dist;
    funcs::exponential_transform<MT> trans(1.0);
    funcs::distribution_and_transform<T>(dev_ctx, &rand, dist, trans);

    funcs::ForRange<Context> for_range(dev_ctx, x.numel());
    for_range([rand_data, in_data] __device__(size_t idx) {
      rand_data[idx] = in_data[idx] / rand_data[idx];
    });

    if (int_num_samples == 1) {
      ArgMaxKernel<T, Context>(
          dev_ctx, rand, -1, true, false, DataType::INT64, out);
    } else {
      std::vector<int64_t> out_dim_vec =
          common::vectorize<int64_t>(out->dims());
      DenseTensor value = Empty<T, Context>(dev_ctx, IntArray(out_dim_vec));
      TopkKernel<T, Context>(
          dev_ctx, rand, num_samples, -1, true, true, &value, out);
    }
    return;
  }

  // Sum of input may not be 1. To get probability in range [0, 1], calculate
  // sum of each row of input, and then use the sum to normalize the input.
  // sum_row_data: sum of each row
  DenseTensor sum_rows_tensor;
  sum_rows_tensor.Resize({num_distributions});
  auto* sum_rows_data = dev_ctx.template Alloc<MT>(&sum_rows_tensor);
  auto& place = *dev_ctx.eigen_device();

  if (num_distributions == 1) {
    auto eigen_input = EigenVector<T>::Flatten(x);
    auto eigen_sum_rows = EigenVector<MT>::Flatten(sum_rows_tensor);
    eigen_sum_rows.device(place) =
        eigen_input.sum(Eigen::DSizes<int, 1>(1))
            .template cast<MT>()
            .eval()
            .template cast<MT>()
            .reshape(Eigen::DSizes<int, 1>(sum_rows_tensor.dims()[0]))
            .template cast<MT>();
  } else {
    auto eigen_input = EigenMatrix<T>::From(x);
    auto eigen_sum_rows = EigenVector<MT>::Flatten(sum_rows_tensor);
    eigen_sum_rows.device(place) =
        eigen_input.sum(Eigen::DSizes<int, 1>(1)).template cast<MT>();
  }
  // Normalize row of each distribution to get the probability in range [0,
  // 1].
  // norm_probs_data: probability of the distribution
  DenseTensor norm_probs_tensor;
  norm_probs_tensor.Resize({num_distributions, num_categories});
  auto* norm_probs_data = dev_ctx.template Alloc<MT>(&norm_probs_tensor);
  // number of threads in a block is min(num_categories, 512)
  int block_size = num_categories < 512 ? num_categories : 512;
  dim3 block_norm(block_size);
  dim3 grid_norm((num_distributions * num_categories - 1) / block_norm.x + 1);

  NormalizeProbability<T, MT>
      <<<grid_norm, block_norm, 0, dev_ctx.stream()>>>(norm_probs_data,
                                                       in_data,
                                                       sum_rows_data,
                                                       num_distributions,
                                                       num_categories);
  // Get cumulative probability of each distribution. It's the same function
  // of ``cumsum`` op.
  DenseTensor cumulative_probs_tensor;
  cumulative_probs_tensor.Resize({num_distributions, num_categories});
  auto* cumulative_probs_data =
      dev_ctx.template Alloc<MT>(&cumulative_probs_tensor);
  // 'phi::funcs::InclusiveScan' has higher accuracy than
  // 'thrust::inclusive_scan'
  funcs::InclusiveScan<MT, std::plus<MT>>(
      /*in*/ norm_probs_data,
      /*out*/ cumulative_probs_data,
      /*outer_dim*/ static_cast<size_t>(num_distributions),
      /*mid_dim*/ static_cast<size_t>(num_categories),
      /*inner_dim*/ static_cast<size_t>(1),
      /*init*/ static_cast<T>(0),
      std::plus<MT>(),
      /*reverse=*/false,
      dev_ctx);
  // Sample the multinomial distributions.
  dim3 block(128);
  int64_t device_id = dev_ctx.GetPlace().GetDeviceId();
  const auto& prop = phi::backends::gpu::GetDeviceProperties(device_id);
  int grid_y = std::min<int64_t>(num_distributions, prop.maxGridSize[1]);
  dim3 grid((int_num_samples - 1) / block.x + 1, grid_y);

  auto gen_cuda = dev_ctx.GetGenerator();
  size_t curand4_loop_times =
      (num_distributions + 4 * grid_y - 1) / (4 * grid_y);
  // 'increment' shoulde be multiple of 4
  uint64_t increment = curand4_loop_times * 4;
  auto seed_offset = gen_cuda->IncrementOffset(increment);

  sampleMultinomialWithReplacement<MT>
      <<<grid, block, 0, dev_ctx.stream()>>>(int_num_samples,
                                             out_data,
                                             num_distributions,
                                             num_categories,
                                             cumulative_probs_data,
                                             norm_probs_data,
                                             seed_offset.first,
                                             seed_offset.second);
}

}  // namespace phi

PD_REGISTER_KERNEL(multinomial,
                   GPU,
                   ALL_LAYOUT,
                   phi::MultinomialKernel,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   float,
                   double) {
  kernel->OutputAt(0).SetDataType(phi::DataType::INT64);
}
