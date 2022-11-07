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
#pragma once
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/selected_rows.h"
#include "paddle/phi/kernels/funcs/lamb_functors.h"
#include "paddle/phi/kernels/funcs/selected_rows_functor.h"

namespace phi {
namespace sr {

template <typename T, typename MT, typename Context, bool IsMultiPrecision>
void ComputeRowImpl(const Context& dev_ctx,
                    const DenseTensor& param,
                    const SelectedRows& grad,
                    const DenseTensor& lr,
                    const DenseTensor& mom1,
                    const DenseTensor& mom2,
                    const DenseTensor& beta1_pow,
                    const DenseTensor& beta2_pow,
                    const paddle::optional<DenseTensor>& master_param_opt,
                    const paddle::optional<DenseTensor>& skip_update_opt,
                    float weight_decay_f,
                    float beta1_f,
                    float beta2_f,
                    float epsilon_f,
                    bool multi_precision,
                    DenseTensor* param_out,
                    DenseTensor* mom1_out,
                    DenseTensor* mom2_out,
                    DenseTensor* beta1_pow_out,
                    DenseTensor* beta2_pow_out,
                    DenseTensor* master_param_out);

template <typename T, typename Context>
void LambKernel(const Context& dev_ctx,
                const DenseTensor& param,
                const SelectedRows& grad,
                const DenseTensor& learning_rate,
                const DenseTensor& moment1,
                const DenseTensor& moment2,
                const DenseTensor& beta1_pow,
                const DenseTensor& beta2_pow,
                const paddle::optional<DenseTensor>& master_param,
                const paddle::optional<DenseTensor>& skip_update,
                float weight_decay,
                float beta1,
                float beta2,
                float epsilon,
                bool multi_precision,
                DenseTensor* param_out,
                DenseTensor* moment1_out,
                DenseTensor* moment2_out,
                DenseTensor* beta1_pow_out,
                DenseTensor* beta2_pow_out,
                DenseTensor* master_param_outs) {
  using MT = typename phi::dtype::MPTypeTrait<T>::Type;
  if (multi_precision) {
    ComputeRowImpl<T, MT, Context, true>(dev_ctx,
                                         param,
                                         grad,
                                         learning_rate,
                                         moment1,
                                         moment2,
                                         beta1_pow,
                                         beta2_pow,
                                         master_param,
                                         skip_update,
                                         weight_decay,
                                         beta1,
                                         beta2,
                                         epsilon,
                                         multi_precision,
                                         param_out,
                                         moment1_out,
                                         moment2_out,
                                         beta1_pow_out,
                                         beta2_pow_out,
                                         master_param_outs);
  } else {
    ComputeRowImpl<T, T, Context, false>(dev_ctx,
                                         param,
                                         grad,
                                         learning_rate,
                                         moment1,
                                         moment2,
                                         beta1_pow,
                                         beta2_pow,
                                         master_param,
                                         skip_update,
                                         weight_decay,
                                         beta1,
                                         beta2,
                                         epsilon,
                                         multi_precision,
                                         param_out,
                                         moment1_out,
                                         moment2_out,
                                         beta1_pow_out,
                                         beta2_pow_out,
                                         master_param_outs);
  }
}

template <typename T, typename MT, typename Context, bool IsMultiPrecision>
void ComputeRowImpl(const Context& dev_ctx,
                    const DenseTensor& param,
                    const SelectedRows& grad,
                    const DenseTensor& lr,
                    const DenseTensor& mom1,
                    const DenseTensor& mom2,
                    const DenseTensor& beta1_pow,
                    const DenseTensor& beta2_pow,
                    const paddle::optional<DenseTensor>& master_param_opt,
                    const paddle::optional<DenseTensor>& skip_update_opt,
                    float weight_decay_f,
                    float beta1_f,
                    float beta2_f,
                    float epsilon_f,
                    bool multi_precision,
                    DenseTensor* param_out,
                    DenseTensor* mom1_out,
                    DenseTensor* mom2_out,
                    DenseTensor* beta1_pow_out,
                    DenseTensor* beta2_pow_out,
                    DenseTensor* master_param_out) {
  if (!IsMultiPrecision) {
    constexpr auto kIsSameType = std::is_same<T, MT>::value;
    PADDLE_ENFORCE_EQ(
        kIsSameType,
        true,
        phi::errors::InvalidArgument(
            "When multi_precision=False, T and MT must be the same type."));
  }

  const auto* master_param =
      IsMultiPrecision ? master_param_opt.get_ptr() : nullptr;
  const auto* skip_update = skip_update_opt.get_ptr();
  const bool* skip_update_flag = skip_update && skip_update->IsInitialized()
                                     ? skip_update->data<bool>()
                                     : nullptr;
  if (skip_update_flag &&
      paddle::platform::is_cpu_place(skip_update->place()) &&
      (*skip_update_flag)) {
    return;
  }

  auto weight_decay = static_cast<MT>(weight_decay_f);
  auto beta1 = static_cast<MT>(beta1_f);
  auto beta2 = static_cast<MT>(beta2_f);
  auto epsilon = static_cast<MT>(epsilon_f);
  auto numel = param.numel();
  phi::funcs::ForRange<Context> for_range(dev_ctx, numel);
  DenseTensor trust_ratio_div;
  trust_ratio_div.Resize(param.dims());
  /*auto trust_ratio_div =
      ctx.AllocateTmpTensor<MT, DeviceContext>(param.dims(), dev_ctx);*/
  auto* trust_ratio_div_ptr = dev_ctx.template Alloc<MT>(&trust_ratio_div);

  const void* param_ptr = param.data();
  const void* master_param_ptr = master_param ? master_param->data() : nullptr;
  void* param_out_ptr = dev_ctx.template Alloc<T>(param_out);
  void* master_param_out_ptr =
      master_param_out ? dev_ctx.template Alloc<MT>(master_param_out) : nullptr;
  // Update moments
  bool should_update_beta_pow_later = false;
  const MT *beta1_pow_ptr = nullptr, *beta2_pow_ptr = nullptr;
  MT *beta1_pow_out_ptr = nullptr, *beta2_pow_out_ptr = nullptr;
  VLOG(10) << "Beta1Pow place: " << beta1_pow.place()
           << " , Beta2Pow place: " << beta2_pow.place();
  // Diff from here
  PADDLE_ENFORCE_EQ(
      IsMultiPrecision,
      false,
      phi::errors::Unimplemented("SelectedRows gradient is not supported when "
                                 "multi_precision=True."));
  constexpr bool kIsSameType = std::is_same<T, MT>::value;
  PADDLE_ENFORCE_EQ(
      kIsSameType,
      true,
      phi::errors::Unimplemented("SelectedRows gradient is not supported when "
                                 "multi_precision=True."));
  if (grad.rows().size() == 0) {
    VLOG(3) << "grad row size is 0!!";
    return;
  }

  std::vector<int64_t> cpu_rows(grad.rows().begin(), grad.rows().end());
  bool is_strict_sorted = true;
  for (size_t i = 1; i < cpu_rows.size(); ++i) {
    if (cpu_rows[i - 1] >= cpu_rows[i]) {
      is_strict_sorted = false;
      break;
    }
  }

  phi::SelectedRows tmp_grad_merge;
  const phi::SelectedRows* grad_merge_ptr;
  if (is_strict_sorted) {
    grad_merge_ptr = &grad;
  } else {
    // merge duplicated rows if any.
    // The rows of grad_merge have been sorted inside MergeAdd functor
    phi::funcs::scatter::MergeAdd<Context, T> merge_func;
    merge_func(dev_ctx, grad, &tmp_grad_merge, true);
    grad_merge_ptr = &tmp_grad_merge;
  }

  auto& grad_merge = *grad_merge_ptr;
  auto& grad_tensor = grad_merge.value();
  const T* grad_data = grad_tensor.template data<T>();
  auto* grad_merge_rows = &grad_merge.rows();
  paddle::framework::MixVector<int64_t> mixv_grad_merge_rows(grad_merge_rows);
  const int64_t* rows = mixv_grad_merge_rows.Data(dev_ctx.GetPlace());
  auto row_numel = grad_tensor.numel() / grad_merge.rows().size();
  if (paddle::platform::is_gpu_place(dev_ctx.GetPlace()) &&
      beta1_pow.place() == phi::CPUPlace() &&
      beta2_pow.place() == phi::CPUPlace()) {
    SparseLambMomentREGUpdateFunctor<T> moment_update_functor(
        static_cast<T>(weight_decay),
        static_cast<T>(beta1),
        static_cast<T>(beta2),
        static_cast<T>(epsilon),
        *beta1_pow.template data<T>(),
        *beta2_pow.template data<T>(),
        mom1.template data<T>(),
        dev_ctx.template Alloc<T>(mom1_out),
        mom2.template data<T>(),
        dev_ctx.template Alloc<T>(mom2_out),
        grad_data,
        param.template data<T>(),
        trust_ratio_div.template data<T>(),
        rows,
        row_numel,
        grad_merge.rows().size(),
        skip_update_flag);
    for_range(moment_update_functor);
    T* beta1_pow_out_data = dev_ctx.template HostAlloc<T>(beta1_pow_out);
    beta1_pow_out_data[0] =
        static_cast<T>(beta1) * beta1_pow.template data<T>()[0];
    T* beta2_pow_out_data = dev_ctx.template HostAlloc<T>(beta2_pow_out);
    beta2_pow_out_data[0] =
        static_cast<T>(beta2) * beta2_pow.template data<T>()[0];
  } else {
    beta1_pow_ptr = beta1_pow.template data<MT>();
    beta2_pow_ptr = beta2_pow.template data<MT>();
    beta1_pow_out_ptr = dev_ctx.template Alloc<MT>(beta1_pow_out);
    beta2_pow_out_ptr = dev_ctx.template Alloc<MT>(beta2_pow_out);
    should_update_beta_pow_later = true;
    SparseLambMomentMENUpdateFunctor<T> moment_update_functor(
        static_cast<T>(weight_decay),
        static_cast<T>(beta1),
        static_cast<T>(beta2),
        static_cast<T>(epsilon),
        reinterpret_cast<const T*>(beta1_pow_ptr),
        reinterpret_cast<const T*>(beta2_pow_ptr),
        mom1.template data<T>(),
        dev_ctx.template Alloc<T>(mom1_out),
        mom2.template data<T>(),
        dev_ctx.template Alloc<T>(mom2_out),
        grad_data,
        param.template data<T>(),
        trust_ratio_div.template data<T>(),
        rows,
        row_numel,
        grad_merge.rows().size(),
        skip_update_flag);
    for_range(moment_update_functor);
  }
  // Same from here
  // Update parameter
  // The code in the following part is exactly the same as that in
  // paddle/phi/kernels/impl/lamb_kernel_impl.h Please modify it together
  DenseTensor p_norm_t;
  p_norm_t.Resize(phi::make_ddim({1}));
  auto* p_norm_ptr = dev_ctx.template Alloc<MT>(&p_norm_t);

  DenseTensor trust_ratio_div_norm_t;
  trust_ratio_div_norm_t.Resize(phi::make_ddim({1}));
  auto* trust_ratio_div_norm_ptr =
      dev_ctx.template Alloc<MT>(&trust_ratio_div_norm_t);

  // TODO(zengjinle): remove the following Eigen operations when
  // *skip_update == true.
  paddle::memory::Buffer buffer(dev_ctx.GetPlace());
  phi::funcs::SquaredL2Norm(
      dev_ctx,
      reinterpret_cast<const MT*>(IsMultiPrecision ? master_param_ptr
                                                   : param_ptr),
      p_norm_ptr,
      numel,
      &buffer);
  phi::funcs::SquaredL2Norm(
      dev_ctx, trust_ratio_div_ptr, trust_ratio_div_norm_ptr, numel, &buffer);

  if (VLOG_IS_ON(1)) {
    const auto& name = "Param";
    auto pn = phi::funcs::ToVector(p_norm_ptr, 1, dev_ctx.GetPlace());
    auto tn =
        phi::funcs::ToVector(trust_ratio_div_norm_ptr, 1, dev_ctx.GetPlace());
    auto dtype = paddle::framework::DataTypeToString(
        paddle::framework::DataTypeTrait<T>::DataType());
    VLOG(1) << "Param " << dtype << " " << name << " pn = " << pn[0]
            << " , tn = " << tn[0];
  }

#define CALL_PADDLE_UPDATE_LAMB_PARAM_FUNC(__should_update_beta_pow)         \
  do {                                                                       \
    LambParamUpateFunctor<T, MT, IsMultiPrecision, __should_update_beta_pow> \
        param_update_functor(lr.template data<MT>(),                         \
                             static_cast<const T*>(param_ptr),               \
                             static_cast<const MT*>(master_param_ptr),       \
                             p_norm_ptr,                                     \
                             trust_ratio_div_ptr,                            \
                             trust_ratio_div_norm_ptr,                       \
                             static_cast<T*>(param_out_ptr),                 \
                             static_cast<MT*>(master_param_out_ptr),         \
                             skip_update_flag);                              \
    if (__should_update_beta_pow) {                                          \
      param_update_functor.SetBetaPows(beta1_pow_ptr,                        \
                                       beta2_pow_ptr,                        \
                                       beta1_pow_out_ptr,                    \
                                       beta2_pow_out_ptr,                    \
                                       beta1,                                \
                                       beta2);                               \
    }                                                                        \
    for_range(param_update_functor);                                         \
  } while (0)

  if (should_update_beta_pow_later) {
    CALL_PADDLE_UPDATE_LAMB_PARAM_FUNC(true);
  } else {
    CALL_PADDLE_UPDATE_LAMB_PARAM_FUNC(false);
  }

#undef CALL_PADDLE_UPDATE_LAMB_PARAM_FUNC
}

}  // namespace sr
}  // namespace phi
