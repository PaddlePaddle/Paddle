/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#include <cstdint>
#include <type_traits>

#include "paddle/phi/kernels/scale_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/elementwise_base.h"
#include "paddle/phi/backends/gpu/musa/mudnn_helper.h"
#include "paddle/phi/kernels/funcs/tensor_formatter.h"
#include "paddle/phi/kernels/impl/fill_kernel_impl.h"

namespace phi {
using GPUDNNDataLayout = phi::backends::gpu::DataLayout;

template <typename DataT, typename ParamT>
struct ScaleFunctor {
  ParamT bias;
  ParamT scale;
  bool bias_after_scale;

  ScaleFunctor(ParamT scale_data, ParamT bias_data, bool is_bias_after_sacle)
      : bias(bias_data),
        scale(scale_data),
        bias_after_scale(is_bias_after_sacle) {}

  __device__ __forceinline__ DataT operator()(const DataT x) const {
    if (bias_after_scale) {
      return static_cast<DataT>(scale * static_cast<ParamT>(x) + bias);
    } else {
      return static_cast<DataT>(scale * (static_cast<ParamT>(x) + bias));
    }
  }
};

//use ScopedUnaryDescriptor twice (add and mul) will be slower than original implementation!
template <typename T, typename Context>
void ScaleKernel(const Context& dev_ctx,
                 const DenseTensor& x,
                 const Scalar& scale,
                 float bias,
                 bool bias_after_scale,
                 DenseTensor* out) {
  if(bias!=0.0f && scale.to<int64_t>()!=1){
    using MT = typename phi::dtype::MPTypeTrait<T>::Type;
    std::vector<const DenseTensor*> inputs;
    std::vector<DenseTensor*> outputs;
    inputs.emplace_back(&x);
    outputs.emplace_back(out);
    dev_ctx.template Alloc<T>(out);
    if (x.numel() <= 0 || (!x.IsInitialized())) {
      return;
    }
    phi::funcs::ElementwiseKernel<T>(
        dev_ctx,
        inputs,
        &outputs,
        ScaleFunctor<T, MT>(
            scale.to<MT>(), static_cast<MT>(bias), bias_after_scale));
    return;
  }
  dev_ctx.template Alloc<T>(out);
  backends::gpu::ScopedTensorDescriptor x_scoped_desc;
  backends::gpu::ScopedTensorDescriptor out_scoped_desc;
  auto musa_x=x_scoped_desc.descriptor_with_stride<T>(x, GPUDNNDataLayout::kNCHW, common::vectorize<int>(x.dims()));
  auto musa_out=out_scoped_desc.descriptor_with_stride<T>(*out, GPUDNNDataLayout::kNCHW, common::vectorize<int>(out->dims()));
  auto handle = dev_ctx.cudnn_handle();
  phi::backends::gpu::ScopedUnaryDescriptor un_desc;
  if(bias==0.0f){
    if constexpr (std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t> || std::is_same_v<T, int16_t> || std::is_same_v<T, int>|| std::is_same_v<T, int64_t>) {
      un_desc.desc_.SetAlpha(scale.to<int64_t>());
    }else if constexpr(std::is_same_v<T, float> || std::is_same_v<T, double> || std::is_same_v<T, phi::dtype::float16> || std::is_same_v<T, phi::dtype::bfloat16>){
      un_desc.desc_.SetAlpha(scale.to<double>());
    }else{
      auto __summary__ = phi::ErrorSummary("does not support");
      auto __message__ = ::paddle::string::Sprintf(
          "",
          __summary__.error_message());
      __THROW_ERROR_INTERNAL__(
          phi::ErrorSummary(__summary__.code(), std::move(__message__)));
    }
    un_desc.desc_.SetMode(::musa::dnn::Unary::Mode::MUL);
    un_desc.desc_.Run(*handle,musa_out,musa_x);
  }else if(scale.to<int64_t>()==1){
    if constexpr (std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t> || std::is_same_v<T, int16_t> || std::is_same_v<T, int>|| std::is_same_v<T, int64_t>) {
      un_desc.desc_.SetAlpha(static_cast<int64_t>(bias));
    }else if constexpr(std::is_same_v<T, float> || std::is_same_v<T, double> || std::is_same_v<T, phi::dtype::float16> || std::is_same_v<T, phi::dtype::bfloat16>){
      un_desc.desc_.SetAlpha(static_cast<double>(bias));
    }else{
      auto __summary__ = phi::ErrorSummary("does not support");
      auto __message__ = ::paddle::string::Sprintf(
          "",
          __summary__.error_message());
      __THROW_ERROR_INTERNAL__(
          phi::ErrorSummary(__summary__.code(), std::move(__message__)));
    }
    un_desc.desc_.SetMode(::musa::dnn::Unary::Mode::ADD);
    un_desc.desc_.Run(*handle,musa_out,musa_x);
  }
  return;
}

}  // namespace phi


PD_REGISTER_KERNEL(scale,
                   GPU,
                   ALL_LAYOUT,
                   phi::ScaleKernel,
                   bool,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int,
                   int64_t,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>
                   ) {}
