// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/layer_norm_kernel.h"

#include "paddle/phi/backends/onednn/onednn_reuse.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T>
class LayerNormOneDNNHandler
    : public phi::funcs::
          OneDNNHandlerNoCachingT<T, dnnl::layer_normalization_forward> {
 public:
  LayerNormOneDNNHandler(const std::vector<int64_t>& dims,
                         const float& epsilon,
                         const dnnl::normalization_flags& flags,
                         const bool& is_test,
                         const phi::DenseTensor* x,
                         const dnnl::engine engine,
                         Place cpu_place)
      : phi::funcs::OneDNNHandlerNoCachingT<T,
                                            dnnl::layer_normalization_forward>(
            engine, cpu_place) {
    const auto fwd_prop_kind = is_test ? dnnl::prop_kind::forward_inference
                                       : dnnl::prop_kind::forward_training;

    this->AcquireForwardPrimitiveDescriptor(
        fwd_prop_kind, x->mem_desc(), x->mem_desc(), epsilon, flags);
  }

  std::tuple<std::shared_ptr<dnnl::memory>, std::shared_ptr<dnnl::memory>>
  AcquireScaleShiftMemory(const phi::DenseTensor* scale,
                          const phi::DenseTensor* shift) {
    auto scale_memory = this->AcquireMemoryFromPrimitive(
        this->fwd_pd_->weights_desc(),
        phi::funcs::to_void_cast<float>(scale->data<float>()));
    auto shift_memory = this->AcquireMemoryFromPrimitive(
        this->fwd_pd_->weights_desc(),
        phi::funcs::to_void_cast<float>(shift->data<float>()));

    return std::make_tuple(scale_memory, shift_memory);
  }

  std::shared_ptr<dnnl::memory> AcquireMeanMemory(const OneDNNContext& dev_ctx,
                                                  phi::DenseTensor* mean) {
    float* mean_data = dev_ctx.template Alloc<float>(
        mean, this->fwd_pd_->mean_desc().get_size());
    return this->AcquireMemoryFromPrimitive(this->fwd_pd_->mean_desc(),
                                            mean_data);
  }

  std::shared_ptr<dnnl::memory> AcquireVarianceMemory(
      const OneDNNContext& dev_ctx, phi::DenseTensor* variance) {
    float* variance_data = dev_ctx.template Alloc<float>(
        variance, this->fwd_pd_->variance_desc().get_size());
    return this->AcquireMemoryFromPrimitive(this->fwd_pd_->variance_desc(),
                                            variance_data);
  }
};

template <typename T, typename Context>
void LayerNormKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     const paddle::optional<DenseTensor>& scale_opt,
                     const paddle::optional<DenseTensor>& bias_opt,
                     float epsilon,
                     int begin_norm_axis,
                     DenseTensor* y,
                     DenseTensor* mean,
                     DenseTensor* var) {
  bool is_test = dev_ctx.HasDnnAttr("is_test")
                     ? PADDLE_GET_CONST(bool, dev_ctx.GetDnnAttr("is_test"))
                     : false;

  const auto& onednn_engine = dev_ctx.GetEngine();

  auto src_tz = common::vectorize(x.dims());
  PADDLE_ENFORCE_EQ(begin_norm_axis,
                    (src_tz.size() - 1),
                    common::errors::InvalidArgument(
                        "MKL-DNN Layer Norm supports only last logical "
                        "axis:%d as begin_norm_axis.",
                        (src_tz.size() - 1)));

  const bool with_scaleshift = (scale_opt && bias_opt);
  dnnl::normalization_flags flags{};

  if (with_scaleshift) {
    flags |= dnnl::normalization_flags::use_scale |
             dnnl::normalization_flags::use_shift;
  }

  LayerNormOneDNNHandler<T> handler(
      src_tz, epsilon, flags, is_test, &x, onednn_engine, dev_ctx.GetPlace());

  auto src_memory = handler.AcquireSrcMemory(&x);
  auto dst_memory = handler.AcquireDstMemory(y);

  auto layer_norm_p = handler.AcquireForwardPrimitive();

  auto& astream = phi::OneDNNContext::tls().get_stream();
  std::unordered_map<int, dnnl::memory> args = {{DNNL_ARG_SRC, *src_memory},
                                                {DNNL_ARG_DST, *dst_memory}};

  if (!is_test) {
    auto mean_memory = handler.AcquireMeanMemory(dev_ctx, mean);
    auto variance_memory = handler.AcquireVarianceMemory(dev_ctx, var);

    args.insert({DNNL_ARG_MEAN, *mean_memory});
    args.insert({DNNL_ARG_VARIANCE, *variance_memory});
  }

  if (with_scaleshift) {
    auto scaleshift_mems = handler.AcquireScaleShiftMemory(scale_opt.get_ptr(),
                                                           bias_opt.get_ptr());
    args.insert({DNNL_ARG_SCALE, *(std::get<0>(scaleshift_mems))});
    args.insert({DNNL_ARG_SHIFT, *(std::get<1>(scaleshift_mems))});
  }

  layer_norm_p->execute(astream, args);
  astream.wait();

  y->set_mem_desc(dst_memory->get_desc());
}
}  // namespace phi

PD_REGISTER_KERNEL(layer_norm,
                   OneDNN,
                   ONEDNN,
                   phi::LayerNormKernel,
                   float,
                   phi::dtype::bfloat16) {
  kernel->OutputAt(1).SetDataType(phi::DataType::UNDEFINED);
  kernel->OutputAt(2).SetDataType(phi::DataType::UNDEFINED);
}
