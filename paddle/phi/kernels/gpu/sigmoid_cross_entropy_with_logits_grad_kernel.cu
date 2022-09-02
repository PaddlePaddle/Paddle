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

#include "paddle/phi/kernels/sigmoid_cross_entropy_with_logits_grad_kernel.h"

#include "paddle/phi/kernels/gpu/sigmoid_cross_entropy_with_logits.h"

namespace phi {

template <typename T>
struct SigmoidBwdFunctor {
  T ignore_index_;
  T eps = static_cast<T>(1e-5);

  HOSTDEVICE inline SigmoidBwdFunctor(const T ignore_index)
      : ignore_index_(ignore_index) {}

  HOSTDEVICE inline phi::Array<T, 2> operator()(const T x,
                                                const T label,
                                                const T dout) {
    T counts;
    T dx_data;

    T diff = label - static_cast<T>(ignore_index_);
    if ((diff > -eps) && (diff < eps)) {
      dx_data = static_cast<T>(0.);
      counts = 0;
    } else {
      T simoid_x = static_cast<T>(1) /
                   (static_cast<T>(1) + paddle::operators::real_exp(-x));
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

template <typename T, typename Context>
void SigmoidCrossEntropyWithLogitsGradKernel(const Context &dev_ctx,
                                             const DenseTensor &x,
                                             const DenseTensor &label,
                                             const DenseTensor &out_grad,
                                             bool normalize,
                                             int ignore_index,
                                             DenseTensor *in_grad) {
  auto dx_data = dev_ctx.template Alloc<T>(in_grad);

  // Temporary memory
  DenseTensor *counts_tensor = new DenseTensor();

  int64_t out_dims = label.numel() * sizeof(T);
  counts_tensor->Resize({out_dims});
  dev_ctx.template Alloc<T>(counts_tensor);
  counts_tensor->Resize(in_grad->dims());

  std::vector<const DenseTensor *> ins = {&x, &label, &out_grad};
  std::vector<DenseTensor *> outs = {in_grad, counts_tensor};
  auto functor = SigmoidBwdFunctor<T>(ignore_index);
  phi::funcs::ElementwiseKernel<T, decltype(functor), 2>(
      dev_ctx, ins, &outs, functor);
  if (normalize) {
    DenseTensor *norm_tensor = new DenseTensor();
    norm_tensor->Resize({sizeof(T)});
    dev_ctx.template Alloc<T>(norm_tensor);
    auto dims = phi::vectorize(counts_tensor->dims());
    std::vector<int> reduce_dim = {};
    for (int i = 0; i < dims.size(); i++) {
      reduce_dim.push_back(i);
    }

    funcs::ReduceKernel<T, T, kps::AddFunctor, NonzeroFunctor<T>>(
        dev_ctx, *counts_tensor, norm_tensor, NonzeroFunctor<T>(), reduce_dim);
    T *norm = dev_ctx.template Alloc<T>(norm_tensor);
    auto norm_cpu_mem = paddle::memory::Alloc(phi::CPUPlace(), sizeof(T));
    T *norm_cpu_ptr = reinterpret_cast<T *>(norm_cpu_mem->ptr());
    paddle::memory::Copy(phi::CPUPlace(),
                         norm_cpu_ptr,
                         dev_ctx.GetPlace(),
                         norm,
                         sizeof(T),
                         dev_ctx.stream());
    dev_ctx.Wait();
    auto eps = static_cast<T>(1e-5);
    *norm_cpu_ptr = *norm_cpu_ptr > eps ? *norm_cpu_ptr : eps;

    std::vector<const DenseTensor *> div_ins = {in_grad};
    std::vector<DenseTensor *> div_outs = {in_grad};
    auto div_functor = DivFunctor<T>(*norm_cpu_ptr);
    phi::funcs::ElementwiseKernel<T>(dev_ctx, div_ins, &div_outs, div_functor);

    delete norm_tensor;
  }
  delete counts_tensor;
}

}  // namespace phi

PD_REGISTER_KERNEL(sigmoid_cross_entropy_with_logits_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::SigmoidCrossEntropyWithLogitsGradKernel,
                   float,
                   double) {}
