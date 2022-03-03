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
#ifdef __NVCC__
#include "cub/cub.cuh"
#endif
#ifdef __HIPCC__
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#endif
#include "paddle/fluid/memory/malloc.h"
#include "paddle/fluid/operators/math.h"
#include "paddle/fluid/operators/reduce_ops/reduce_op.cu.h"
#include "paddle/fluid/operators/sigmoid_cross_entropy_with_logits_op.h"
#include "paddle/fluid/platform/device/gpu/gpu_primitives.h"
#include "paddle/phi/core/hostdevice.h"
#include "paddle/phi/kernels/funcs/elementwise_base.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

#ifdef __HIPCC__
static constexpr int kNumCUDAThreads = 256;
#else
static constexpr int kNumCUDAThreads = 512;
#endif
static constexpr int kNumMaxinumNumBlocks = 4096;

static inline int NumBlocks(const int N) {
  return std::min((N + kNumCUDAThreads - 1) / kNumCUDAThreads,
                  kNumMaxinumNumBlocks);
}

template <typename T>
struct NonzeroFunctor {
  HOSTDEVICE explicit inline NonzeroFunctor() {}
  HOSTDEVICE inline T operator()(const T x) const {
    return static_cast<T>(static_cast<double>(x) != 0);
  }
};

template <typename T>
struct SigmoidFwdFunctor {
  T ignore_index_;
  T eps = static_cast<T>(1e-5);

  HOSTDEVICE inline SigmoidFwdFunctor(const T ignore_index)
      : ignore_index_(ignore_index) {}

  HOSTDEVICE inline phi::Array<T, 2> operator()(const T x, const T label) {
    T counts;
    T out_data;

    T diff = label - static_cast<T>(ignore_index_);
    if ((diff > -eps) && (diff < eps)) {
      out_data = static_cast<T>(0.);
      counts = 0;
    } else {
      T term1 = (x > 0) ? x : 0;
      T term2 = x * label;
      T term3 = real_log(static_cast<T>(1) + real_exp(static_cast<T>(-abs(x))));

      out_data = term1 - term2 + term3;
      counts = 1;
    }
    phi::Array<T, 2> outs;

    outs[0] = out_data;
    outs[1] = counts;
    return outs;
  }
};

template <typename T>
struct SigmoidBwdFunctor {
  T ignore_index_;
  T eps = static_cast<T>(1e-5);

  HOSTDEVICE inline SigmoidBwdFunctor(const T ignore_index)
      : ignore_index_(ignore_index) {}

  HOSTDEVICE inline phi::Array<T, 2> operator()(const T x, const T label,
                                                const T dout) {
    T counts;
    T dx_data;

    T diff = label - static_cast<T>(ignore_index_);
    if ((diff > -eps) && (diff < eps)) {
      dx_data = static_cast<T>(0.);
      counts = 0;
    } else {
      T simoid_x = static_cast<T>(1) / (static_cast<T>(1) + real_exp(-x));
      T diff = simoid_x - label;
      dx_data = dout * diff;
      counts = 1;
    }
    phi::Array<T, 2> outs;

    outs[0] = dx_data;
    outs[1] = counts;
    return outs;
  }
};

template <typename T>
struct DivFunctor {
  const T norm_;
  HOSTDEVICE inline DivFunctor(const T norm) : norm_(norm) {}

  HOSTDEVICE inline T operator()(T loss) {
    loss /= norm_;
    return loss;
  }
};

// Out = max(X, 0) - X * Labels + log(1 + exp(-abs(X)))
template <typename DeviceContext, typename T>
class GPUSigmoidCrossEntropyWithLogitsKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    const Tensor *X = context.Input<Tensor>("X");
    const Tensor *Labels = context.Input<Tensor>("Label");
    Tensor *Out = context.Output<Tensor>("Out");
    int ignore_index = context.Attr<int>("ignore_index");
    auto out_data = Out->mutable_data<T>(context.GetPlace());

    auto &dev_ctx = context.cuda_device_context();
    bool normalize = context.Attr<bool>("normalize");

    // Temporary memory
    Tensor *counts_tensor = new Tensor();
    counts_tensor->mutable_data<T>(context.GetPlace(),
                                   Labels->numel() * sizeof(T));
    counts_tensor->Resize(Out->dims());
    int limit = Out->numel();
    int blocks = NumBlocks(limit);
    int threads = kNumCUDAThreads;
    std::vector<const framework::Tensor *> ins = {X, Labels};
    std::vector<framework::Tensor *> outs = {Out, counts_tensor};
    auto functor = SigmoidFwdFunctor<T>(ignore_index);
    constexpr int Size = 2;
    phi::funcs::ElementwiseKernel<T, decltype(functor), Size>(dev_ctx, ins,
                                                              &outs, functor);
    if (normalize) {
      T *counts = counts_tensor->mutable_data<T>(context.GetPlace());
      Tensor *norm_tensor = new Tensor();
      norm_tensor->mutable_data<T>(context.GetPlace(), sizeof(T));
      auto dims = phi::vectorize(counts_tensor->dims());
      std::vector<int> reduce_dim = {};
      for (int i = 0; i < dims.size(); i++) {
        reduce_dim.push_back(i);
      }

      TensorReduceImpl<T, T, kps::AddFunctor, NonzeroFunctor<T>>(
          context.cuda_device_context(), *counts_tensor, norm_tensor,
          NonzeroFunctor<T>(), reduce_dim, dev_ctx.stream());
      T *norm = norm_tensor->mutable_data<T>(context.GetPlace());
      auto norm_cpu_mem = memory::Alloc(platform::CPUPlace(), sizeof(T));
      T *norm_cpu_ptr = reinterpret_cast<T *>(norm_cpu_mem->ptr());
      memory::Copy(platform::CPUPlace(), norm_cpu_ptr, dev_ctx.GetPlace(), norm,
                   sizeof(T), dev_ctx.stream());
      auto eps = static_cast<T>(1e-5);
      *norm_cpu_ptr = *norm_cpu_ptr > eps ? *norm_cpu_ptr : eps;

      std::vector<const framework::Tensor *> div_ins = {Out};
      std::vector<framework::Tensor *> div_outs = {Out};
      auto div_functor = DivFunctor<T>(*norm_cpu_ptr);
      phi::funcs::ElementwiseKernel<T>(dev_ctx, div_ins, &div_outs,
                                       div_functor);

      delete norm_tensor;
      delete counts_tensor;
    }
  }
};

// dX = sigmoid(X) - labels
template <typename DeviceContext, typename T>
class GPUSigmoidCrossEntropyWithLogitsGradKernel
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    const Tensor *X = context.Input<Tensor>("X");
    const Tensor *Labels = context.Input<Tensor>("Label");
    const Tensor *dOut = context.Input<Tensor>(framework::GradVarName("Out"));
    Tensor *dX = context.Output<Tensor>(framework::GradVarName("X"));
    auto dx_data = dX->mutable_data<T>(context.GetPlace());

    int ignore_index = context.Attr<int>("ignore_index");

    auto &dev_ctx = context.cuda_device_context();
    // Temporary memory
    Tensor *counts_tensor = new Tensor();
    counts_tensor->mutable_data<T>(context.GetPlace(),
                                   Labels->numel() * sizeof(T));
    counts_tensor->Resize(dX->dims());

    int limit = dX->numel();
    int blocks = NumBlocks(limit);
    int threads = kNumCUDAThreads;
    std::vector<const framework::Tensor *> ins = {X, Labels, dOut};
    std::vector<framework::Tensor *> outs = {dX, counts_tensor};
    auto functor = SigmoidBwdFunctor<T>(ignore_index);
    constexpr int Size = 2;
    phi::funcs::ElementwiseKernel<T, decltype(functor), Size>(dev_ctx, ins,
                                                              &outs, functor);
    bool normalize = context.Attr<bool>("normalize");
    if (normalize) {
      T *counts = counts_tensor->mutable_data<T>(context.GetPlace());
      Tensor *norm_tensor = new Tensor();
      norm_tensor->mutable_data<T>(context.GetPlace(), sizeof(T));
      auto dims = phi::vectorize(counts_tensor->dims());
      std::vector<int> reduce_dim = {};
      for (int i = 0; i < dims.size(); i++) {
        reduce_dim.push_back(i);
      }

      TensorReduceImpl<T, T, kps::AddFunctor, NonzeroFunctor<T>>(
          context.cuda_device_context(), *counts_tensor, norm_tensor,
          NonzeroFunctor<T>(), reduce_dim, dev_ctx.stream());
      T *norm = norm_tensor->mutable_data<T>(context.GetPlace());
      auto norm_cpu_mem = memory::Alloc(platform::CPUPlace(), sizeof(T));
      T *norm_cpu_ptr = reinterpret_cast<T *>(norm_cpu_mem->ptr());
      memory::Copy(platform::CPUPlace(), norm_cpu_ptr, dev_ctx.GetPlace(), norm,
                   sizeof(T), dev_ctx.stream());
      auto eps = static_cast<T>(1e-5);
      *norm_cpu_ptr = *norm_cpu_ptr > eps ? *norm_cpu_ptr : eps;

      std::vector<const framework::Tensor *> div_ins = {dX};
      std::vector<framework::Tensor *> div_outs = {dX};
      auto div_functor = DivFunctor<T>(*norm_cpu_ptr);
      phi::funcs::ElementwiseKernel<T>(dev_ctx, div_ins, &div_outs,
                                       div_functor);
      delete norm_tensor;
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(sigmoid_cross_entropy_with_logits,
                        ops::GPUSigmoidCrossEntropyWithLogitsKernel<
                            paddle::platform::CUDADeviceContext, float>,
                        ops::GPUSigmoidCrossEntropyWithLogitsKernel<
                            paddle::platform::CUDADeviceContext, double>);
REGISTER_OP_CUDA_KERNEL(sigmoid_cross_entropy_with_logits_grad,
                        ops::GPUSigmoidCrossEntropyWithLogitsGradKernel<
                            paddle::platform::CUDADeviceContext, float>,
                        ops::GPUSigmoidCrossEntropyWithLogitsGradKernel<
                            paddle::platform::CUDADeviceContext, double>);
