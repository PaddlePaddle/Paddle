// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)

#ifndef _MSC_VER
#include <thrust/device_ptr.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/random.h>
#include <thrust/shuffle.h>
#endif

#include "paddle/fluid/operators/shuffle_batch_op.h"
#include "paddle/fluid/platform/for_range.h"

namespace paddle {
namespace operators {

template <typename T, bool kIsForward>
struct ReorderFunctor {
  ReorderFunctor(const T *x, const int64_t *shuffle_idx, T *y, int64_t stride)
      : x_(x), shuffle_idx_(shuffle_idx), y_(y), stride_(stride) {}

  HOSTDEVICE void operator()(int64_t idx) {
    auto reorder_idx = shuffle_idx_[idx / stride_] * stride_ + idx % stride_;
    if (kIsForward) {
      y_[idx] = x_[reorder_idx];
    } else {
      y_[reorder_idx] = x_[idx];
    }
  }

 private:
  const T *x_;
  const int64_t *shuffle_idx_;
  T *y_;
  int64_t stride_;
};

template <typename T>
class ShuffleBatchCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
#ifdef _MSC_VER
    PADDLE_THROW(platform::errors::Unimplemented(
        "GPU shuffle_batch is not supported on Windows yet"));
#else
    auto *x = ctx.Input<phi::DenseTensor>("X");
    auto *seed = ctx.Input<phi::DenseTensor>("Seed");
    auto *out = ctx.Output<phi::DenseTensor>("Out");
    auto *shuffleidx = ctx.Output<phi::DenseTensor>("ShuffleIdx");
    auto *seed_out = ctx.Output<phi::DenseTensor>("SeedOut");

    int64_t x_embed_size = x->dims()[x->dims().size() - 1];
    int64_t elem_size = 1;
    for (int i = 0; i < x->dims().size() - 1; i++) {
      elem_size *= x->dims()[i];
    }
    shuffleidx->Resize(phi::make_ddim({elem_size}));

    int64_t seed_int = 0;
    if (seed->IsInitialized()) {
      const auto &seed_place = seed->place();
      if (platform::is_gpu_place(seed_place)) {
        // NOTE: We have overwritten GetKernelTypeForVar, so seed_place would
        // not be CUDAPlace in practice. This case would only happen in Python
        // op_test framework.
        phi::DenseTensor tmp_tensor;
        framework::TensorCopySync(*seed, platform::CPUPlace(), &tmp_tensor);
        seed_int = *(tmp_tensor.data<int64_t>());
      } else {
        seed_int = *(seed->data<int64_t>());
      }
    } else {
      seed_int = ctx.Attr<int>("startup_seed");
    }

    auto *shuffleidx_data = shuffleidx->mutable_data<int64_t>(ctx.GetPlace());

    auto &dev_ctx = ctx.template device_context<phi::GPUContext>();
#ifdef PADDLE_WITH_CUDA
    const auto &exec_policy = thrust::cuda::par.on(dev_ctx.stream());
#else
    const auto &exec_policy = thrust::hip::par.on(dev_ctx.stream());
#endif
    thrust::random::default_random_engine engine(seed_int);
    thrust::counting_iterator<int64_t> cnt_iter(0);
    thrust::shuffle_copy(exec_policy,
                         cnt_iter,
                         cnt_iter + elem_size,
                         thrust::device_pointer_cast(shuffleidx_data),
                         engine);
    // TODO(zengjinle): for small data, direct cudaMemcpy may be better
    auto *x_data = x->data<T>();
    auto *out_data = out->mutable_data<T>(ctx.GetPlace());
    ReorderFunctor<T, true> functor(
        x_data, shuffleidx_data, out_data, x_embed_size);
    platform::ForRange<phi::GPUContext> for_range(dev_ctx,
                                                  elem_size * x_embed_size);
    for_range(functor);

    auto *seed_out_data = seed_out->mutable_data<int64_t>(phi::make_ddim({1}),
                                                          platform::CPUPlace());
    *seed_out_data = engine();
#endif
  }
};

template <typename T>
class ShuffleBatchGradCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
#ifdef _MSC_VER
    PADDLE_THROW(platform::errors::Unimplemented(
        "GPU shuffle_batch_grad is not supported on Windows yet"));
#else
    const auto *out_grad =
        ctx.Input<phi::DenseTensor>(framework::GradVarName("Out"));
    const auto *shuffleidx = ctx.Input<phi::DenseTensor>("ShuffleIdx");
    auto *x_grad = ctx.Output<phi::DenseTensor>(framework::GradVarName("X"));

    const auto *out_grad_data = out_grad->data<T>();
    const auto *shuffleidx_data = shuffleidx->data<int64_t>();
    auto *x_grad_data = x_grad->mutable_data<T>(ctx.GetPlace());
    auto x_embed_size = x_grad->dims()[x_grad->dims().size() - 1];
    ReorderFunctor<T, false> functor(
        out_grad_data, shuffleidx_data, x_grad_data, x_embed_size);
    auto &dev_ctx = ctx.template device_context<phi::GPUContext>();
    // TODO(zengjinle): for small data, direct cudaMemcpy may be better
    platform::ForRange<phi::GPUContext> for_range(dev_ctx, x_grad->numel());
    for_range(functor);
#endif
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(shuffle_batch,
                        ops::ShuffleBatchCUDAKernel<float>,
                        ops::ShuffleBatchCUDAKernel<double>,
                        ops::ShuffleBatchCUDAKernel<int32_t>,
                        ops::ShuffleBatchCUDAKernel<int64_t>);

REGISTER_OP_CUDA_KERNEL(shuffle_batch_grad,
                        ops::ShuffleBatchGradCUDAKernel<float>,
                        ops::ShuffleBatchGradCUDAKernel<double>,
                        ops::ShuffleBatchGradCUDAKernel<int32_t>,
                        ops::ShuffleBatchGradCUDAKernel<int64_t>);
#endif
