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
#include "glog/logging.h"

#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/funcs/lamb_functors.h"

namespace phi {

template <typename T, typename MT, typename Context, bool IsMultiPrecision>
void ComputeImpl(const Context& dev_ctx,
                 const DenseTensor& param,
                 const DenseTensor& grad,
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
                const DenseTensor& grad,
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
    ComputeImpl<T, MT, Context, true>(dev_ctx,
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
    ComputeImpl<T, T, Context, false>(dev_ctx,
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
void ComputeImpl(const Context& dev_ctx,
                 const DenseTensor& param,
                 const DenseTensor& grad,
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
                 bool multi_precision UNUSED,
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
      skip_update->place().GetType() == phi::AllocationType::CPU &&
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

  if (dev_ctx.GetPlace().GetType() == phi::AllocationType::GPU &&
      beta1_pow.place() == phi::CPUPlace() &&
      beta2_pow.place() == phi::CPUPlace()) {
    LambMomentREGUpdateFunctor<T, IsMultiPrecision> moment_update_functor(
        weight_decay,
        beta1,
        beta2,
        epsilon,
        *beta1_pow.template data<MT>(),
        *beta2_pow.template data<MT>(),
        mom1.template data<MT>(),
        dev_ctx.template Alloc<MT>(mom1_out),
        mom2.template data<MT>(),
        dev_ctx.template Alloc<MT>(mom2_out),
        grad.template data<T>(),
        static_cast<const MT*>(IsMultiPrecision ? master_param_ptr : param_ptr),
        trust_ratio_div_ptr,
        skip_update_flag);
    for_range(moment_update_functor);
    MT* beta1_pow_out_data = dev_ctx.template HostAlloc<MT>(beta1_pow_out);
    beta1_pow_out_data[0] = beta1 * beta1_pow.template data<MT>()[0];
    MT* beta2_pow_out_data = dev_ctx.template HostAlloc<MT>(beta2_pow_out);
    beta2_pow_out_data[0] = beta2 * beta2_pow.template data<MT>()[0];
  } else {
    beta1_pow_ptr = beta1_pow.template data<MT>();
    beta2_pow_ptr = beta2_pow.template data<MT>();
    beta1_pow_out_ptr = dev_ctx.template Alloc<MT>(beta1_pow_out);
    beta2_pow_out_ptr = dev_ctx.template Alloc<MT>(beta2_pow_out);
    should_update_beta_pow_later = true;
    LambMomentMENUpdateFunctor<T, IsMultiPrecision> moment_update_functor(
        weight_decay,
        beta1,
        beta2,
        epsilon,
        static_cast<const MT*>(beta1_pow_ptr),
        static_cast<const MT*>(beta2_pow_ptr),
        mom1.template data<MT>(),
        dev_ctx.template Alloc<MT>(mom1_out),
        mom2.template data<MT>(),
        dev_ctx.template Alloc<MT>(mom2_out),
        grad.template data<T>(),
        static_cast<const MT*>(IsMultiPrecision ? master_param_ptr : param_ptr),
        trust_ratio_div_ptr,
        skip_update_flag);
    for_range(moment_update_functor);
  }

  // Same from here
  // Update parameter
  // The code in the following part is exactly the same as that in
  // paddle/phi/kernels/selected_rows/impl/lamb_kernel_impl.h Please modify it
  // together
  DenseTensor p_norm_t;
  p_norm_t.Resize(phi::make_ddim({1}));
  auto* p_norm_ptr = dev_ctx.template Alloc<MT>(&p_norm_t);

  DenseTensor trust_ratio_div_norm_t;
  trust_ratio_div_norm_t.Resize(phi::make_ddim({1}));
  auto* trust_ratio_div_norm_ptr =
      dev_ctx.template Alloc<MT>(&trust_ratio_div_norm_t);

  // TODO(zengjinle): remove the following Eigen operations when
  // *skip_update == true.
  memory_utils::Buffer buffer(dev_ctx.GetPlace());
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
    auto dtype = DataTypeToString(phi::CppTypeToDataType<T>::Type());
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

}  // namespace phi
