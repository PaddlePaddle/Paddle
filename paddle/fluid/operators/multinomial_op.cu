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

/*
template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenVector = framework::EigenVector<T, MajorType, IndexType>;
template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenMatrix = framework::EigenMatrix<T, MajorType, IndexType>;
*/

/*
template <class T>
__global__ void SumArrayCUDAKernel(T **in, T *out, size_t in_size) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  // T total(read_dst ? out[id] : static_cast<T>(0));
  T total(static_cast<T>(0))
  for (int i = 0; i < in_size; ++i) {
    const T *tmp = in[i];
    if (tmp) {
      total += tmp[id];
    }
  }
  out[id] = total;
  id += blockDim.x * gridDim.x;
}*/

/*
template <typename T>
__global__ void NormalizeProbability(T* probs, int64_t rows, int64_t cols) {
  extern __shared__ std::vector<T> sum_rows(rows);
  T val;
  for (int64_t i = blockId.x; i < rows; i += gridDim.x) {
    T sum = static_cast<T>(0);
    for (int64_t j = threadIdx.x; j < cols; j += blockDim.x) {
      val = probs[i * cols + j];
      sum += val;
    }

  }
}*/

template <typename T>
__global__ void NormalizeProbability(T* norm_probs, const T* in_data,
                                     T* sum_rows) {
  // int id = blockIdx.x * blockDim.x + threadIdx.x;
  int id = threadIdx.x;
  norm_probs[id] = in_data[id] / sum_rows[0];
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

/*
template <typename T>
class MultinomialCudaFunctor(T* out_data, const T* in_data,
                        const int64_t num_samples, const bool replacement,
                        const int64_t num_categories,
                        const int64_t num_distributions) {

}*/

template <typename T>
__device__ int binarySearchForMultinomial(T* cumdist, T* dist, int size,
                                          T val) {
  int start = 0;
  int end = size;
  // cumdist[size - 1] = 0 => all zero prob dist
  // CUDA_KERNEL_ASSERT(cumdist[size - 1] > static_cast<T>(0));

  while (end - start > 0) {
    int mid = start + (end - start) / 2;

    T midVal = cumdist[mid];
    if (midVal < val) {
      start = mid + 1;
    } else {
      end = mid;
    }
  }

  if (start == size) {
    // No probability mass or precision problems; just return the
    // first non-zero element by setting start to size-1 here,
    // the code below will move it to the last non-zero probability
    // this actually can happen when the random number is 1
    // (github pytorch issue #4858).
    start = size - 1;
  }

  while (start >= 1 && dist[start] == 0) start--;

  return start;
}

template <typename T>
__global__ void sampleMultinomialWithReplacement(
    T* rng, const int64_t totalSamples, T* dest, const int64_t distributions,
    const int64_t categories, T* normDistPrefixSum, T* normDist) {
  // At the moment, each warp computes one sample value in the binary
  // search due to divergence. It seems possible to compute multiple
  // values and limit divergence though later on.

  // global index formula for 2D grid of 1D blocks
  // int idx = blockIdx.y * gridDim.x * blockDim.x + blockIdx.x * blockDim.x +
  // threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  for (int sample = blockIdx.x * blockDim.x + threadIdx.x;
       sample < totalSamples; sample += blockDim.x * gridDim.x) {
    // we are losing 3 out of 4 generated numbers but it's ok
    // this kernel is not very efficient anyway

    // T uniform_random = dist(rng);
    T uniform_random = rng[sample];

    // Find the bucket that a uniform sample lies in
    int choice = binarySearchForMultinomial<T>(normDistPrefixSum, normDist,
                                               categories, uniform_random);

    dest[sample] = choice;
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

    // std::vector<T> sum_rows(num_distributions);
    // SumArrayCUDAKernel<T>(in_data, sum_rows,)

    VLOG(3) << "Print num_distributions " << num_distributions << "\n";

    VLOG(3) << "Print num_categories " << num_categories << "\n";

    VLOG(3) << "Print in_rank " << in_rank << "\n";

    framework::Tensor sum_rows_t;
    auto* sum_rows_data = sum_rows_t.mutable_data<T>({1}, ctx.GetPlace());
    // auto* sum_rows_data =
    // sum_rows_t->mutable_data<T>(framework::make_ddim({1}), ctx.GetPlace());

    auto& place = *ctx.template device_context<platform::CUDADeviceContext>()
                       .eigen_device();

    auto eigen_input = framework::EigenVector<T>::Flatten(*x);
    // auto eigen_sum_rows = framework::EigenVector<T>::From(sum_rows_t);
    auto eigen_sum_rows = framework::EigenScalar<T>::From(sum_rows_t);
    eigen_sum_rows.device(place) =
        eigen_input.sum(Eigen::DSizes<int, 1>(0))
            .eval()
            .reshape(Eigen::DSizes<int, 1>(sum_rows_t.dims()[0]));
    // eigen_sum_rows.device(place) =
    // eigen_input.sum().eval().reshape(Eigen::DSizes<int, 1>(1));

    dim3 grid(num_distributions);
    dim3 block(num_categories);

    // std::vector<T> in_data_norm(num_categories);
    framework::Tensor norm_probs_t;
    auto* norm_probs_data =
        norm_probs_t.mutable_data<T>({num_categories}, ctx.GetPlace());
    NormalizeProbability<
        T><<<grid, block, 0, ctx.cuda_device_context().stream()>>>(
        norm_probs_data, in_data, sum_rows_data);

    // num_distributions can only be 1.
    // std::vector<T> cumulative_probs(num_categories);
    framework::Tensor cumulative_probs_t;
    auto* cumulative_probs =
        cumulative_probs_t.mutable_data<T>({num_categories}, ctx.GetPlace());
    // T cumulative_probs[num_categories];
    int64_t size = num_categories;
    thrust::inclusive_scan(thrust::device, norm_probs_data,
                           norm_probs_data + num_categories, cumulative_probs);

    if (replacement) {
      dim3 block(128);
      // int grid_y = 1;
      dim3 grid((num_samples - 1) / block.x + 1);

      /*
      // std::vector<T> rng(num_samples);
      T rng[num_samples];
      std::uniform_real_distribution<T> dist(0, 1);
      auto gen_ptr = framework::DefaultCPUGenerator();
      auto engine = gen_ptr->GetCPUEngine();

      for (int s = 0; s < num_samples; s++) {
        rng[s] = dist(*engine);
      }
      */

      std::random_device rd;
      auto seed = rd();

      framework::Tensor rng_data_t;
      auto* rng_data =
          rng_data_t.mutable_data<T>({num_samples}, ctx.GetPlace());

      thrust::counting_iterator<unsigned int> index_sequence_begin(0);
      platform::Transform<platform::CUDADeviceContext> trans;
      auto* context = static_cast<const platform::CUDADeviceContext*>(
          &ctx.device_context());
      trans(*context, index_sequence_begin, index_sequence_begin + num_samples,
            rng_data, RandomGeneratorCudaFunctor<T>(seed));

      VLOG(3) << "Print enter\n";
      // VLOG(3) << "Print size in_data " <<
      // sizeof(in_data)/sizeof(in_data[num_categories-1]) << "\n";
      // VLOG(3) << "Print norm_probs_data0 " <<
      // sizeof(norm_probs_data[num_categories-1]) << "\n";

      sampleMultinomialWithReplacement<
          T><<<grid, block, 0, ctx.cuda_device_context().stream()>>>(
          rng_data, num_samples, out_data, num_distributions, num_categories,
          cumulative_probs, norm_probs_data);
    }

    // MultinomialCudaFunctor<T>(out_data, in_data, num_samples, replacement,
    //                    num_categories, num_distributions);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_CUDA_KERNEL(
    multinomial, ops::MultinomialOpKernel<plat::CUDADeviceContext, float>,
    ops::MultinomialOpKernel<plat::CUDADeviceContext, double>);
