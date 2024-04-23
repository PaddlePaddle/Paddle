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

#include "paddle/phi/backends/onednn/onednn_reuse.h"
#include "paddle/phi/core/compat/convert_utils.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/errors.h"
#include "paddle/phi/core/expect.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/utils/data_type.h"

namespace phi {

using dnnl::memory;

template <typename T, typename DeviceContext>
class QuantOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<phi::DenseTensor>("Input");
    auto* out = ctx.Output<phi::DenseTensor>("Output");

    const auto quantization_scale = ctx.Attr<float>("Scale");
    const auto quantization_shift =
        static_cast<int32_t>(ctx.Attr<float>("Shift"));
    const bool with_scale = quantization_scale != 1.0f;
    const bool with_shift = quantization_shift != 0.0f;

    PADDLE_ENFORCE_NE(quantization_scale,
                      0.0f,
                      phi::errors::InvalidArgument(
                          "Quantization scale must be different than 0.0f"));
    PADDLE_ENFORCE(quantization_shift <= 255 && quantization_shift >= 0,
                   phi::errors::InvalidArgument(
                       "Quantization shift must be lower or equal to ",
                       "255 and greater or equal to 0, but got %f",
                       quantization_shift));

    auto& dev_ctx = ctx.template device_context<phi::OneDNNContext>();

    auto x_tz = common::vectorize<int64_t>(x->dims());

    const bool is_negative_input = ctx.Attr<bool>("is_negative_input");
    const bool bfloat16 = ctx.Attr<bool>("bfloat16");

    dnnl::primitive_attr attrs;
    static constexpr int32_t mask = 0;

    if (with_scale) {
      attrs.set_scales_mask(DNNL_ARG_SRC, mask);
    }

    if (with_shift) {
      attrs.set_zero_points_mask(DNNL_ARG_DST, mask);
    }

    auto x_type = phi::funcs::ToOneDNNDataType(x->dtype());
    DataType out_dtype;

    if (bfloat16) {
      out_dtype = DataType::BFLOAT16;
    } else if (is_negative_input && !with_shift) {
      out_dtype = DataType::INT8;
    } else {
      out_dtype = DataType::UINT8;
    }

    auto out_type = phi::funcs::ToOneDNNDataType(out_dtype);

    phi::funcs::ReorderOneDNNHandler reorder_handler(
        x_tz, x->dtype(), x_type, out_dtype, out_type, dev_ctx.GetEngine());

    auto reorder_src_memory_p = reorder_handler.AcquireSrcMemory(
        x->mem_desc(), phi::funcs::to_void_cast(x->data<T>()));
    auto reorder_dst_memory_p = reorder_handler.AcquireDstMemory(
        out, x->mem_desc(), dev_ctx.GetPlace());

    auto reorder_p = reorder_handler.AcquireReorder(
        reorder_dst_memory_p, reorder_src_memory_p, attrs);

    auto& astream = phi::OneDNNContext::tls().get_stream();

    auto scales_md = dnnl::memory::desc(
        {1}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::x);
    auto scales_mem =
        dnnl::memory(scales_md,
                     dev_ctx.GetEngine(),
                     phi::funcs::to_void_cast<float>(&quantization_scale));
    auto zero_points_md = dnnl::memory::desc(
        {1}, dnnl::memory::data_type::s32, dnnl::memory::format_tag::x);
    auto zero_points_mem =
        dnnl::memory(zero_points_md,
                     dev_ctx.GetEngine(),
                     phi::funcs::to_void_cast<int32_t>(&quantization_shift));

    std::unordered_map<int, dnnl::memory> reorder_args;
    reorder_args.insert({DNNL_ARG_SRC, *reorder_src_memory_p});
    reorder_args.insert({DNNL_ARG_DST, *reorder_dst_memory_p});
    if (with_scale) {
      reorder_args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, scales_mem});
    }
    if (with_shift) {
      reorder_args.insert(
          {DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_DST, zero_points_mem});
    }

    reorder_p->execute(astream, reorder_args);
    astream.wait();

    out->set_mem_desc(reorder_dst_memory_p->get_desc());
  }
};
}  // namespace phi

PD_REGISTER_KERNEL(quantize, OneDNN, ONEDNN, phi::QuantOpKernel, float) {}
