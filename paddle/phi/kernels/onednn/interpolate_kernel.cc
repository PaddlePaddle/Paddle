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

#include "paddle/phi/kernels/interpolate_kernel.h"

#include "paddle/phi/backends/onednn/onednn_reuse.h"
#include "paddle/phi/core/compat/get_kerneltype_forvar_utils.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/interpolate_function.h"

namespace phi {

KernelKey InterpolateGetKernelTypeForVar(
    const GetKernelTypeForVarContext* ctx) {
  const std::string& var_name = ctx->GetVarName();
  const DenseTensor& tensor = ctx->GetTensor();
  const KernelKey& expected_kernel_type = ctx->GetKernelKey();
  const AttributeMap& attrs = ctx->GetAttrs();
  // Only input require reshaping, weights and
  // bias are having shape in NCHW order
  if ((expected_kernel_type.layout() == DataLayout::ONEDNN) &&
      (tensor.layout() != DataLayout::ONEDNN)) {
    auto it = attrs.find("data_layout");
    const std::string data_layout = PADDLE_GET_CONST(std::string, it->second);
    auto dl = common::StringToDataLayout(data_layout);
    // Some models may have intentionally set "AnyLayout" for pool
    // op. Treat this as NCHW (default data_format value)
    if (dl != DataLayout::kAnyLayout) {
      return KernelKey(tensor.place(), dl, expected_kernel_type.dtype());
    }
  }
  if (var_name == "OutSize" || var_name == "SizeTensor" ||
      var_name == "Scale") {
    return KernelKey(Backend::ALL_BACKEND,
                     expected_kernel_type.layout(),
                     expected_kernel_type.dtype());
  }
  return KernelKey(
      tensor.place(), tensor.layout(), expected_kernel_type.dtype());
}

namespace funcs {
template <typename T = float>
class InterpolateOneDNNHandler
    : public OneDNNHandlerNoCachingT<T, dnnl::resampling_forward> {
 public:
  InterpolateOneDNNHandler(const dnnl::algorithm algo,
                           const dnnl::engine engine,
                           Place cpu_place,
                           const DenseTensor* x,
                           DenseTensor* out)
      : OneDNNHandlerNoCachingT<T, dnnl::resampling_forward>(engine,
                                                             cpu_place) {
    const auto dst_tz = common::vectorize(out->dims());
    const auto dst_md = dnnl::memory::desc(
        dst_tz, OneDNNGetDataType<T>(), OneDNNMemoryFormat::any);
    this->AcquireForwardPrimitiveDescriptor(
        dnnl::prop_kind::forward_inference, algo, x->mem_desc(), dst_md);
  }
};
}  // namespace funcs

std::vector<int> ComputeOutputShape(
    const DenseTensor* x,
    const paddle::optional<DenseTensor>& out_size,
    const paddle::optional<std::vector<const DenseTensor*>>& size_tensor,
    const paddle::optional<DenseTensor>& scale_tensor,
    const std::string& data_layout,
    int out_d,
    int out_h,
    int out_w,
    const std::vector<float>& scale_attr) {
  const auto& in_dims = x->dims();
  const DDim in_dhw_dims = slice_ddim(in_dims, 2, in_dims.size());

  std::vector<int> out_dims;
  out_dims.reserve(5);
  if (in_dhw_dims.size() == 1) {
    out_dims.push_back(out_w);
  } else if (in_dhw_dims.size() == 2) {
    out_dims.push_back(out_h);
    out_dims.push_back(out_w);
  } else if (in_dhw_dims.size() == 3) {
    out_dims.push_back(out_d);
    out_dims.push_back(out_h);
    out_dims.push_back(out_w);
  }

  if (size_tensor && !size_tensor.get().empty()) {
    auto new_size = funcs::get_new_shape(size_tensor.get());
    if (new_size.size() == out_dims.size()) {
      out_dims = new_size;
    }
  } else if (out_size) {
    auto out_size_data =
        funcs::get_new_data_from_tensor<int>(out_size.get_ptr());
    if (out_size_data.size() == out_dims.size()) {
      out_dims = out_size_data;
    }
  } else {
    std::vector<float> scale;
    scale.reserve(3);
    if (scale_tensor) {
      auto scale_data =
          funcs::get_new_data_from_tensor<float>(scale_tensor.get_ptr());
      scale.resize(3, scale_data[0]);
      std::copy(scale_data.begin(), scale_data.end(), scale.begin());
    } else {
      if (!scale_attr.empty()) {
        scale.resize(3, scale_attr[0]);
        std::copy(scale_attr.begin(), scale_attr.end(), scale.begin());
      }
    }

    if (scale.size() == 3 && scale[0] > 0.0f && scale[1] > 0.0f &&
        scale[2] > 0.0f) {
      int j = 0;
      std::vector<int64_t> in_dhw_vec = common::vectorize(in_dhw_dims);
      std::transform(
          in_dhw_vec.begin(),
          in_dhw_vec.end(),
          out_dims.begin(),
          [&](int64_t i) -> int { return static_cast<int>(i * scale[j++]); });
    }
  }

  PADDLE_ENFORCE_GT(
      std::all_of(
          out_dims.begin(), out_dims.end(), [](int i) { return i > 0; }),
      0,
      errors::InvalidArgument("out_d, out_h, out_w of Op(interpolate) "
                              "should be greater than 0."));

  const std::vector<int64_t> nc_dims = {in_dims[0], in_dims[1]};
  out_dims.insert(out_dims.begin(), nc_dims.begin(), nc_dims.end());
  return out_dims;
}

template <typename T, typename Context>
void InterpolateKernel(
    const Context& dev_ctx,
    const DenseTensor& x,
    const paddle::optional<DenseTensor>& out_size,
    const paddle::optional<std::vector<const DenseTensor*>>& size_tensor,
    const paddle::optional<DenseTensor>& scale_tensor,
    const std::string& data_layout,
    int out_d,
    int out_h,
    int out_w,
    const std::vector<float>& scale,
    const std::string& interp_method,
    DenseTensor* out) {
  const auto& onednn_engine = dev_ctx.GetEngine();

  const dnnl::algorithm algo = (interp_method == "nearest")
                                   ? dnnl::algorithm::resampling_nearest
                                   : dnnl::algorithm::resampling_linear;

  const auto out_dims_vec = ComputeOutputShape(&x,
                                               out_size,
                                               size_tensor,
                                               scale_tensor,
                                               data_layout,
                                               out_d,
                                               out_h,
                                               out_w,
                                               scale);
  DDim dim_out = common::make_ddim(out_dims_vec);
  out->Resize(dim_out);

  funcs::InterpolateOneDNNHandler<T> handler(
      algo, onednn_engine, dev_ctx.GetPlace(), &x, out);

  auto src_memory_p = handler.AcquireSrcMemory(&x);
  auto dst_memory_p = handler.AcquireDstMemory(out);

  auto resampling_prim = handler.AcquireForwardPrimitive();
  const std::unordered_map<int, dnnl::memory> args = {
      {DNNL_ARG_SRC, *src_memory_p}, {DNNL_ARG_DST, *dst_memory_p}};
  auto& astream = OneDNNContext::tls().get_stream();

  resampling_prim->execute(astream, args);
  astream.wait();

  out->set_mem_desc(dst_memory_p->get_desc());
}

template <typename T, typename Context>
void BilinearInterpKernel(
    const Context& ctx,
    const DenseTensor& x,
    const paddle::optional<DenseTensor>& out_size,
    const paddle::optional<std::vector<const DenseTensor*>>& size_tensor,
    const paddle::optional<DenseTensor>& scale_tensor,
    const std::string& data_layout,
    int out_d,
    int out_h,
    int out_w,
    const std::vector<float>& scale,
    const std::string& interp_method,
    bool align_corners UNUSED,
    int align_mode UNUSED,
    DenseTensor* output) {
  InterpolateKernel<T, Context>(ctx,
                                x,
                                out_size,
                                size_tensor,
                                scale_tensor,
                                data_layout,
                                out_d,
                                out_h,
                                out_w,
                                scale,
                                interp_method,
                                output);
}

template <typename T, typename Context>
void LegacyBilinearInterpKernel(
    const Context& ctx,
    const DenseTensor& x,
    const paddle::optional<DenseTensor>& out_size,
    const paddle::optional<std::vector<const DenseTensor*>>& size_tensor,
    const paddle::optional<DenseTensor>& scale_tensor,
    const std::string& data_layout,
    int out_d,
    int out_h,
    int out_w,
    float scale,
    const std::string& interp_method,
    bool align_corners UNUSED,
    int align_mode UNUSED,
    DenseTensor* output) {
  const auto& dim_x = x.dims();
  std::vector<float> scale_vec;
  if (scale > 0) {
    for (int i = 0; i < dim_x.size() - 2; i++) {
      scale_vec.push_back(scale);
    }
  }
  InterpolateKernel<T, Context>(ctx,
                                x,
                                out_size,
                                size_tensor,
                                scale_tensor,
                                data_layout,
                                out_d,
                                out_h,
                                out_w,
                                scale_vec,
                                interp_method,
                                output);
}

template <typename T, typename Context>
void NearestInterpKernel(
    const Context& ctx,
    const DenseTensor& x,
    const paddle::optional<DenseTensor>& out_size,
    const paddle::optional<std::vector<const DenseTensor*>>& size_tensor,
    const paddle::optional<DenseTensor>& scale_tensor,
    const std::string& data_layout,
    int out_d,
    int out_h,
    int out_w,
    const std::vector<float>& scale,
    const std::string& interp_method,
    bool align_corners UNUSED,
    int align_mode UNUSED,
    DenseTensor* output) {
  InterpolateKernel<T, Context>(ctx,
                                x,
                                out_size,
                                size_tensor,
                                scale_tensor,
                                data_layout,
                                out_d,
                                out_h,
                                out_w,
                                scale,
                                interp_method,
                                output);
}

template <typename T, typename Context>
void LegacyNearestInterpKernel(
    const Context& ctx,
    const DenseTensor& x,
    const paddle::optional<DenseTensor>& out_size,
    const paddle::optional<std::vector<const DenseTensor*>>& size_tensor,
    const paddle::optional<DenseTensor>& scale_tensor,
    const std::string& data_layout,
    int out_d,
    int out_h,
    int out_w,
    float scale,
    const std::string& interp_method,
    bool align_corners UNUSED,
    int align_mode UNUSED,
    DenseTensor* output) {
  const auto& dim_x = x.dims();
  std::vector<float> scale_vec;
  if (scale > 0) {
    for (int i = 0; i < dim_x.size() - 2; i++) {
      scale_vec.push_back(scale);
    }
  }
  InterpolateKernel<T, Context>(ctx,
                                x,
                                out_size,
                                size_tensor,
                                scale_tensor,
                                data_layout,
                                out_d,
                                out_h,
                                out_w,
                                scale_vec,
                                interp_method,
                                output);
}
}  // namespace phi

PD_REGISTER_KERNEL(bilinear_interp,
                   OneDNN,
                   ONEDNN,
                   phi::BilinearInterpKernel,
                   float,
                   phi::dtype::bfloat16,
                   phi::dtype::float16) {
  kernel->get_kerneltype_forvar_fn_ = phi::InterpolateGetKernelTypeForVar;
}

PD_REGISTER_KERNEL(nearest_interp,
                   OneDNN,
                   ONEDNN,
                   phi::NearestInterpKernel,
                   float,
                   phi::dtype::bfloat16,
                   phi::dtype::float16,
                   int8_t,
                   uint8_t) {
  kernel->get_kerneltype_forvar_fn_ = phi::InterpolateGetKernelTypeForVar;
}
PD_REGISTER_KERNEL(legacy_bilinear_interp,
                   OneDNN,
                   ONEDNN,
                   phi::LegacyBilinearInterpKernel,
                   float,
                   phi::dtype::bfloat16,
                   phi::dtype::float16) {
  kernel->get_kerneltype_forvar_fn_ = phi::InterpolateGetKernelTypeForVar;
}
PD_REGISTER_KERNEL(legacy_nearest_interp,
                   OneDNN,
                   ONEDNN,
                   phi::LegacyNearestInterpKernel,
                   float,
                   phi::dtype::bfloat16,
                   phi::dtype::float16,
                   int8_t,
                   uint8_t) {
  kernel->get_kerneltype_forvar_fn_ = phi::InterpolateGetKernelTypeForVar;
}
