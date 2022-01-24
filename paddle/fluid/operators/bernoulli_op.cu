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
#include <thrust/transform.h>

#include "paddle/fluid/framework/generator.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/operators/bernoulli_op.h"
#include "paddle/fluid/platform/transform.h"

namespace paddle {
namespace operators {
// it can be consistent with cpu when CUDAGenerator is provided.
template <typename T>
struct BernoulliCudaFunctor {
  unsigned int seed_;
  unsigned int offset_;
  __host__ __device__ BernoulliCudaFunctor(unsigned int seed,
                                           unsigned int offset)
      : seed_(seed), offset_(offset) {}

  __host__ __device__ T operator()(const unsigned int n, const T p) const {
    // NOTE(zhiqiu): currently, PADDLE_ENFORCE in cuda kernel may print several
    // lines of error messages if, and it should be refined.
    PADDLE_ENFORCE(p >= 0.0 && p <= 1.0,
                   "The probability should be >=0 and <= 1, but got %f", p);
    thrust::minstd_rand rng;
    rng.seed(seed_);
    thrust::uniform_real_distribution<T> dist(0.0, 1.0);
    rng.discard(n + offset_);
    return static_cast<T>(dist(rng) < p);
  }
};

template <typename T>
class BernoulliOpKernel<platform::CUDADeviceContext, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const auto x = ctx.Input<framework::Tensor>("X");
    auto out = ctx.Output<framework::Tensor>("Out");
    auto* in_data = x->data<T>();
    auto* out_data = out->mutable_data<T>(ctx.GetPlace());
    int64_t size = x->numel();

    int device_id = ctx.GetPlace().GetDeviceId();
    auto gen_cuda = framework::GetDefaultCUDAGenerator(device_id);
    auto seed_offset = gen_cuda->IncrementOffset(1);
    int64_t gen_offset = size * seed_offset.second;
    platform::Transform<platform::CUDADeviceContext> trans;
    thrust::counting_iterator<int64_t> index_sequence_begin(0);
    auto* context =
        static_cast<const platform::CUDADeviceContext*>(&ctx.device_context());

    trans(*context, index_sequence_begin, index_sequence_begin + size, in_data,
          out_data,
          BernoulliCudaFunctor<T>(static_cast<int64_t>(seed_offset.first),
                                  static_cast<int64_t>(gen_offset)));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_CUDA_KERNEL(
    bernoulli, ops::BernoulliOpKernel<plat::CUDADeviceContext, float>,
    ops::BernoulliOpKernel<plat::CUDADeviceContext, double>);
