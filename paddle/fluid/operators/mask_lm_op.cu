/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#define EIGEN_USE_GPU
#include <thrust/device_ptr.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/random.h>
#include <thrust/transform.h>
#include "paddle/fluid/operators/mask_lm_op.h"
#include "paddle/fluid/platform/float16.h"

namespace paddle {
namespace operators {

using framework::LoDTensor;

template <typename T>
__global__ void RandomGenerator(const int64_t n, const int seed,
                                const int64_t post_rnd_offset,
                                const int voc_size, const float masked_prob,
                                const T mask_id, const T* src, T* mask_data,
                                T* dst, int64_t* is_masked) {
  thrust::minstd_rand rng;
  rng.seed(seed);
  thrust::uniform_real_distribution<float> dist(0, 1);

  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  int step_size = 0;

  float fake_masked_prob = masked_prob * 0.1;
  float rand_masked_prob = masked_prob * 0.2;
  float rand_scale = rand_masked_prob - fake_masked_prob;

  T dest;
  for (; idx < n; idx += blockDim.x * gridDim.x) {
    T s = src[idx];
    is_masked[idx] = static_cast<int64_t>(1);

    // hack for null mask
    if (idx == post_rnd_offset) {
      mask_data[idx] = s;
      dst[idx] = mask_id;
      continue;
    }

    if (step_size == 0) {
      rng.discard(idx);
      step_size = blockDim.x * gridDim.x;
    } else {
      rng.discard(step_size);
    }

    float dist_rng = dist(rng);
    if (dist_rng < fake_masked_prob) {
      mask_data[idx] = s;
      dst[idx] = s;
      continue;
    }
    if (dist_rng < rand_masked_prob) {
      mask_data[idx] = s;
      dest = static_cast<T>(
          floor((dist_rng - fake_masked_prob) / rand_scale * voc_size));
      dst[idx] = dest;
      continue;
    }
    if (dist_rng < masked_prob) {
      mask_data[idx] = s;
      dst[idx] = mask_id;
      continue;
    }
    // else
    mask_data[idx] = static_cast<T>(-1);
    is_masked[idx] = static_cast<int64_t>(0);
    dst[idx] = s;
  }
}

template <typename T>
__global__ void SummarizeMask(const int64_t* is_masked, const T* mask_tmp,
                              int64_t n, T* mask_out, int64_t* mask_pos) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n && (is_masked[idx] < is_masked[idx + 1])) {
    mask_pos[is_masked[idx]] = idx;
    mask_out[is_masked[idx]] = mask_tmp[idx];
  }
}

// It seems that Eigen::Tensor::setRandom in GPU will SEGFAULT.
// Use std::random and thrust::random(thrust is a std library in CUDA) to
// implement uniform random.

template <typename Place, typename T>
class GPUMaskLMKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* x = context.Input<LoDTensor>("X");
    auto* y = context.Output<LoDTensor>("Out");
    auto* mask_out = context.Output<framework::Tensor>("Mask");
    auto* mask_pos = context.Output<framework::Tensor>("MaskPos");

    T mask_id = static_cast<T>(context.Attr<int>("mask_id"));
    float masked_prob = context.Attr<float>("masked_prob");
    int voc_size = context.Attr<int>("voc_size");

    auto* y_data = y->mutable_data<T>(context.GetPlace());

    std::random_device rnd;
    int seed =
        context.Attr<bool>("fix_seed") ? context.Attr<int>("seed") : rnd();

    // hack for null mask
    int64_t numel = x->numel();
    std::mt19937 gen(seed);
    std::uniform_int_distribution<> dis(0, numel - 1);
    int64_t post_rnd_offset = dis(gen);

    Tensor is_masked, mask_tmp;
    is_masked.Resize({numel + 1, 1});
    mask_tmp.Resize({numel, 1});
    is_masked.mutable_data<int64_t>(context.GetPlace());
    mask_tmp.mutable_data<T>(context.GetPlace());

    int threads = 512;
    int grid = (numel + threads - 1) / threads;
    auto stream = context.cuda_device_context().stream();
    RandomGenerator<T><<<grid, threads, 0, stream>>>(
        numel, seed, post_rnd_offset, voc_size, masked_prob, mask_id,
        x->data<T>(), mask_tmp.data<T>(), y_data, is_masked.data<int64_t>());

    thrust::device_ptr<int64_t> is_masked_ptr =
        thrust::device_pointer_cast(is_masked.data<int64_t>());
    thrust::exclusive_scan(is_masked_ptr, is_masked_ptr + numel + 1,
                           is_masked_ptr);

    int64_t out_len;
    platform::CUDAPlace place =
        boost::get<platform::CUDAPlace>(context.GetPlace());
    memory::Copy(platform::CPUPlace(), &out_len, place,
                 is_masked.data<int64_t>() + numel, sizeof(int64_t), stream);

    mask_out->Resize({out_len, 1});
    mask_pos->Resize({out_len, 1});
    mask_out->mutable_data<T>(context.GetPlace());
    mask_pos->mutable_data<int64_t>(context.GetPlace());

    SummarizeMask<<<grid, threads, 0, stream>>>(
        is_masked.data<int64_t>(), mask_tmp.data<T>(), numel,
        mask_out->data<T>(), mask_pos->data<int64_t>());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_CUDA_KERNEL(
    mask_lm, ops::GPUMaskLMKernel<plat::CUDADeviceContext, float>,
    ops::GPUMaskLMKernel<plat::CUDADeviceContext, plat::float16>,
    ops::GPUMaskLMKernel<plat::CUDADeviceContext, double>,
    ops::GPUMaskLMKernel<plat::CUDADeviceContext, uint8_t>,
    ops::GPUMaskLMKernel<plat::CUDADeviceContext, int>,
    ops::GPUMaskLMKernel<plat::CUDADeviceContext, int64_t>);
