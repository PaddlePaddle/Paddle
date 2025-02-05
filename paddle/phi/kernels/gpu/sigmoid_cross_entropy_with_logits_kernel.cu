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

#include "paddle/phi/kernels/sigmoid_cross_entropy_with_logits_kernel.h"

#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/kernels/gpu/sigmoid_cross_entropy_with_logits.h"
#include "paddle/phi/kernels/scale_kernel.h"

namespace phi {

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
      T term3 = phi::funcs::real_log(
          static_cast<T>(1) + phi::funcs::real_exp(static_cast<T>(-abs(x))));

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
struct SigmoidFwdPosWeightFunctor {
  T ignore_index_;
  T eps = static_cast<T>(1e-5);

  HOSTDEVICE inline SigmoidFwdPosWeightFunctor(const T ignore_index)
      : ignore_index_(ignore_index) {}

  HOSTDEVICE inline phi::Array<T, 2> operator()(const T x,
                                                const T label,
                                                T pos_weight) {
    T counts;
    T out_data;

    T diff = label - static_cast<T>(ignore_index_);
    if ((diff > -eps) && (diff < eps)) {
      out_data = static_cast<T>(0.);
      counts = 0;
    } else {
      T max_val = x < 0 ? -x : 0;
      T term1 = (static_cast<T>(1.) - label) * x;
      T term2 = phi::funcs::real_log(phi::funcs::real_exp(-max_val) +
                                     phi::funcs::real_exp(-x - max_val));
      out_data = term1 + pos_weight * (term2 + max_val);

      counts = 1;
    }
    phi::Array<T, 2> outs;

    outs[0] = out_data;
    outs[1] = counts;
    return outs;
  }
};

template <typename T, typename Context>
void SigmoidCrossEntropyWithLogitsKernel(
    const Context &dev_ctx,
    const DenseTensor &x,
    const DenseTensor &label,
    const paddle::optional<DenseTensor> &pos_weight,
    bool normalize,
    int ignore_index,
    DenseTensor *out) {
  auto out_data = dev_ctx.template Alloc<T>(out);

  // Temporary memory
  DenseTensor *counts_tensor = new DenseTensor();

  int64_t out_dims = label.numel() * sizeof(T);
  counts_tensor->Resize({out_dims});
  dev_ctx.template Alloc<T>(counts_tensor);
  counts_tensor->Resize(out->dims());

  std::vector<DenseTensor *> outs = {out, counts_tensor};

  if (pos_weight.get_ptr() == nullptr) {
    std::vector<const DenseTensor *> ins = {&x, &label};
    auto functor = SigmoidFwdFunctor<T>(ignore_index);
    phi::funcs::ElementwiseKernel<T, decltype(functor), 2>(
        dev_ctx, ins, &outs, functor);
  } else {
    std::vector<const DenseTensor *> ins = {&x, &label, pos_weight.get_ptr()};
    auto functor = SigmoidFwdPosWeightFunctor<T>(ignore_index);
    phi::funcs::ElementwiseKernel<T, decltype(functor), 2>(
        dev_ctx, ins, &outs, functor);
  }
  if (normalize) {
    DenseTensor *norm_tensor = new DenseTensor();
    norm_tensor->Resize({sizeof(T)});
    dev_ctx.template Alloc<T>(norm_tensor);
    auto dims = common::vectorize(counts_tensor->dims());
    std::vector<int> reduce_dim = {};
    for (int i = 0; i < dims.size(); i++) {
      reduce_dim.push_back(i);
    }

    funcs::ReduceKernel<T, T, kps::AddFunctor, NonzeroFunctor<T>>(
        dev_ctx, *counts_tensor, norm_tensor, NonzeroFunctor<T>(), reduce_dim);
    T *norm = dev_ctx.template Alloc<T>(norm_tensor);
    auto norm_cpu_mem = phi::memory_utils::Alloc(phi::CPUPlace(), sizeof(T));
    T *norm_cpu_ptr = reinterpret_cast<T *>(norm_cpu_mem->ptr());
    memory_utils::Copy(phi::CPUPlace(),
                       norm_cpu_ptr,
                       dev_ctx.GetPlace(),
                       norm,
                       sizeof(T),
                       dev_ctx.stream());
    dev_ctx.Wait();
    auto eps = static_cast<T>(1e-5);
    *norm_cpu_ptr = *norm_cpu_ptr > eps ? *norm_cpu_ptr : eps;

    phi::ScaleKernel<T>(dev_ctx, *out, 1.0 / (*norm_cpu_ptr), 0.0f, false, out);

    delete norm_tensor;
  }
  delete counts_tensor;
}

}  // namespace phi

PD_REGISTER_KERNEL(sigmoid_cross_entropy_with_logits,
                   GPU,
                   ALL_LAYOUT,
                   phi::SigmoidCrossEntropyWithLogitsKernel,
                   float,
                   double) {}
