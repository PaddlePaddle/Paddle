/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/scan.h>
#include <thrust/transform.h>

#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/operators/multinomial_op.h"
#include "paddle/fluid/platform/transform.h"

namespace paddle {
namespace operators {

template <typename T>
__global__ void NormalizeProbability(T* norm_probs, const T* in_data,
                                     T* sum_rows) {
  int id = threadIdx.x + blockIdx.x * blockDim.x +
           blockIdx.y * gridDim.x * blockDim.x;
  norm_probs[id] = in_data[id] / sum_rows[blockIdx.y];
}

template <typename T>
__global__ void Cumsum(T* norm_probs_data, int64_t num_distributions,
                       int64_t num_categories, T* cumulative_probs) {
  for (int id = blockIdx.x; id < num_distributions; id += gridDim.x) {
    thrust::inclusive_scan(thrust::device,
                           norm_probs_data + id * num_categories,
                           norm_probs_data + (id + 1) * num_categories,
                           cumulative_probs + id * num_categories);
  }
}

template <typename T>
struct RandomGeneratorCudaFunctor {
  unsigned int seed_;
  __host__ __device__ RandomGeneratorCudaFunctor(int seed) : seed_(seed) {}

  __host__ __device__ T operator()(const unsigned int n) const {
    thrust::minstd_rand rng;
    rng.seed(seed_);
    thrust::uniform_real_distribution<T> dist(0.0, 1.0);
    rng.discard(n);
    return dist(rng);
  }
};

template <typename T>
__device__ int binarySearchFunctor(T* cumdist, T* dist, int size, T val) {
  int left = 0;
  int right = size;
  // cumdist[size - 1] = 0 => all zero prob dist
  // CUDA_KERNEL_ASSERT(cumdist[size - 1] > static_cast<T>(0));

  while (right - left > 0) {
    int mid = left + (right - left) / 2;

    T midVal = cumdist[mid];
    if (midVal < val) {
      left = mid + 1;
    } else {
      right = mid;
    }
  }

  if (left == size) {
    // No probability mass or precision problems; just return the
    // first non-zero element by setting left to size-1 here,
    // the code below will move it to the last non-zero probability
    // this actually can happen when the random number is 1
    // (github pytorch issue #4858).
    left = size - 1;
  }

  while (left >= 1 && dist[left] == 0) left--;

  return left;
}

template <typename T>
__global__ void sampleMultinomialWithReplacement(
    T* rng_data, const int64_t num_samples, T* out_data,
    const int64_t num_distributions, const int64_t num_categories,
    T* cumulative_probs, T* norm_probs_data) {
  // At the moment, each warp computes one sample value in the binary
  // search due to divergence. It seems possible to compute multiple
  // values and limit divergence though later on.

  // global index formula for 2D grid of 1D blocks
  // int idx = blockIdx.y * gridDim.x * blockDim.x + blockIdx.x * blockDim.x +
  // threadIdx.x;

  // int idx = blockIdx.x * blockDim.x + threadIdx.x;

  int idx = threadIdx.x + blockIdx.x * blockDim.x +
            blockIdx.y * gridDim.x * blockDim.x;

  for (int curDist = blockIdx.y; curDist < num_distributions;
       curDist += gridDim.y) {
    for (int sample = blockIdx.x * blockDim.x + threadIdx.x;
         sample < num_samples; sample += blockDim.x * gridDim.x) {
      // we are losing 3 out of 4 generated numbers but it's ok
      // this kernel is not very efficient anyway

      // T uniform_random = dist(rng);
      T uniform_random = rng_data[sample + curDist * num_samples];

      // Find the bucket that a uniform sample lies in
      int choice =
          binarySearchFunctor<T>(cumulative_probs + curDist * num_categories,
                                 norm_probs_data + curDist * num_categories,
                                 num_categories, uniform_random);

      out_data[sample + curDist * num_samples] = choice;
    }
  }
}

template <typename T>
class MultinomialOpKernel<platform::CUDADeviceContext, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const auto x = ctx.Input<framework::Tensor>("X");
    auto out = ctx.Output<framework::Tensor>("Out");

    const int64_t num_samples = ctx.Attr<int>("num_samples");
    const bool replacement = ctx.Attr<bool>("replacement");

    auto* in_data = x->data<T>();
    auto* out_data = out->mutable_data<T>(ctx.GetPlace());

    auto in_dims = x->dims();
    int64_t in_rank = in_dims.size();
    const int64_t num_categories = in_dims[in_rank - 1];
    const int64_t num_distributions = in_rank > 1 ? in_dims[in_rank - 2] : 1;

    if (!replacement) {
      int in_data_numel = x->numel();
      int out_data_numel = out->numel();

      T* cpu_in_data = new T[in_data_numel];
      T* cpu_out_data = new T[out_data_numel];

      cudaMemcpy(cpu_in_data, in_data, in_data_numel * sizeof(T),
                 cudaMemcpyDeviceToHost);

      MultinomialFunctor<T>(cpu_out_data, cpu_in_data, num_samples, replacement,
                            num_categories, num_distributions);
      cudaMemcpy(out_data, cpu_out_data, out_data_numel * sizeof(T),
                 cudaMemcpyHostToDevice);

      delete[] cpu_in_data;
      delete[] cpu_out_data;
      return;
    }

    framework::Tensor sum_rows_t;
    auto* sum_rows_data =
        sum_rows_t.mutable_data<T>({num_distributions}, ctx.GetPlace());

    auto& place = *ctx.template device_context<platform::CUDADeviceContext>()
                       .eigen_device();

    if (num_distributions == 1) {
      auto eigen_input = framework::EigenVector<T>::Flatten(*x);
      auto eigen_sum_rows = framework::EigenVector<T>::From(sum_rows_t);
      eigen_sum_rows.device(place) =
          eigen_input.sum(Eigen::DSizes<int, 1>(1))
              .eval()
              .reshape(Eigen::DSizes<int, 1>(sum_rows_t.dims()[0]));
    } else {
      auto eigen_input = framework::EigenMatrix<T>::From(*x);
      auto eigen_sum_rows = framework::EigenVector<T>::From(sum_rows_t);
      eigen_sum_rows.device(place) = eigen_input.sum(Eigen::DSizes<int, 1>(1));
    }

    framework::Tensor norm_probs_t;
    auto* norm_probs_data = norm_probs_t.mutable_data<T>(
        {num_distributions, num_categories}, ctx.GetPlace());

    dim3 block(num_categories < 512 ? num_categories : 512);
    dim3 grid((num_categories - 1) / block.x + 1, num_distributions);
    NormalizeProbability<
        T><<<grid, block, 0, ctx.cuda_device_context().stream()>>>(
        norm_probs_data, in_data, sum_rows_data);

    framework::Tensor cumulative_probs_t;
    auto* cumulative_probs = cumulative_probs_t.mutable_data<T>(
        {num_distributions, num_categories}, ctx.GetPlace());
    dim3 block1(1);
    dim3 grid1(num_distributions);
    Cumsum<T><<<grid1, block1, 0, ctx.cuda_device_context().stream()>>>(
        norm_probs_data, num_distributions, num_categories, cumulative_probs);

    VLOG(3) << "Print cumsum " << cumulative_probs << "\n";

    if (replacement) {
      dim3 block(128);
      // int grid_y = 1;
      dim3 grid((num_samples - 1) / block.x + 1, num_distributions);

      std::random_device rd;
      auto seed = rd();

      framework::Tensor rng_data_t;
      auto* rng_data = rng_data_t.mutable_data<T>(
          {num_distributions, num_samples}, ctx.GetPlace());

      thrust::counting_iterator<unsigned int> index_sequence_begin(0);
      platform::Transform<platform::CUDADeviceContext> trans;
      auto* context = static_cast<const platform::CUDADeviceContext*>(
          &ctx.device_context());
      trans(*context, index_sequence_begin,
            index_sequence_begin + num_distributions * num_samples, rng_data,
            RandomGeneratorCudaFunctor<T>(seed));

      sampleMultinomialWithReplacement<
          T><<<grid, block, 0, ctx.cuda_device_context().stream()>>>(
          rng_data, num_samples, out_data, num_distributions, num_categories,
          cumulative_probs, norm_probs_data);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_CUDA_KERNEL(
    multinomial, ops::MultinomialOpKernel<plat::CUDADeviceContext, float>,
    ops::MultinomialOpKernel<plat::CUDADeviceContext, double>);
