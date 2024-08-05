// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/dequantize_kernel.h"

#include "paddle/phi/backends/onednn/onednn_context.h"
#include "paddle/phi/backends/onednn/onednn_helper.h"
#include "paddle/phi/backends/onednn/onednn_reuse.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void DeQuantKernel(const Context& dev_ctx,
                   const DenseTensor& x,
                   const float quantization_scale,
                   const float quantization_shift,
                   DenseTensor* out) {
  PADDLE_ENFORCE(quantization_scale != 0.0f,
                 common::errors::InvalidArgument(
                     "Dequantization scale must be different than 0.0f"));

  const auto q_shift = static_cast<int32_t>(quantization_shift);
  PADDLE_ENFORCE_GE(q_shift,
                    0,
                    common::errors::InvalidArgument(
                        "Dequantization shift must be greater or equal to 0"));
  PADDLE_ENFORCE_LE(q_shift,
                    255,
                    common::errors::InvalidArgument(
                        "Dequantization shift must be lower or equal to 255"));

  const bool with_shift = q_shift != 0;

  auto x_tz = common::vectorize<int64_t>(x.dims());
  auto x_type = phi::funcs::ToOneDNNDataType(x.dtype());
  auto out_type = phi::funcs::ToOneDNNDataType(out->dtype());

  dnnl::primitive_attr attrs;
  static constexpr int32_t mask = 0;  // same shift and scale for whole tensor

  attrs.set_scales_mask(DNNL_ARG_DST, mask);

  if (with_shift) {
    attrs.set_zero_points_mask(DNNL_ARG_SRC, mask);
  }

  phi::funcs::ReorderOneDNNHandler reorder_handler(
      x_tz, x.dtype(), x_type, out->dtype(), out_type, dev_ctx.GetEngine());

  auto reorder_src_memory_p = reorder_handler.AcquireSrcMemory(
      x.mem_desc(), phi::funcs::to_void_cast(x.data<T>()));
  auto reorder_dst_memory_p =
      reorder_handler.AcquireDstMemory(out, x.mem_desc(), dev_ctx.GetPlace());

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
                   phi::funcs::to_void_cast<int32_t>(&q_shift));
  std::unordered_map<int, dnnl::memory> reorder_args;
  reorder_args.insert({DNNL_ARG_SRC, *reorder_src_memory_p});
  reorder_args.insert({DNNL_ARG_DST, *reorder_dst_memory_p});
  reorder_args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST, scales_mem});
  if (with_shift) {
    reorder_args.insert(
        {DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC, zero_points_mem});
  }
  reorder_p->execute(astream, reorder_args);
  astream.wait();

  out->set_mem_desc(reorder_dst_memory_p->get_desc());
}

}  // namespace phi

PD_REGISTER_KERNEL(dequantize,
                   OneDNN,
                   ONEDNN,
                   phi::DeQuantKernel,
                   uint8_t,
                   int8_t,
                   phi::dtype::bfloat16) {
  kernel->OutputAt(0).SetDataType(phi::DataType::FLOAT32);
}
