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

#include "paddle/phi/kernels/transpose_kernel.h"
#include "paddle/phi/backends/onednn/onednn_reuse.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void TransposeKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     const std::vector<int>& axis,
                     DenseTensor* out) {
  PADDLE_ENFORCE_EQ(
      dev_ctx.GetPlace().GetType(),
      AllocationType::CPU,
      errors::PreconditionNotMet("oneDNN Transpose kernel must use CPUPlace"));

  if (axis.size() == 1) {
    Copy<Context>(dev_ctx, x, x.place(), false, out);
    out->set_mem_desc(x.mem_desc());
    return;
  }

  auto x_vec_dims = vectorize(x.dims());
  auto x_type = funcs::ToOneDNNDataType(x.dtype());

  dnnl::primitive_attr attrs;
  const int32_t mask = 0;
  const auto quantization_scale =
      dev_ctx.HasDnnAttr("scale")
          ? PADDLE_GET_CONST(float, dev_ctx.GetDnnAttr("scale"))
          : 1.0f;
  const auto quantization_shift =
      dev_ctx.HasDnnAttr("shift")
          ? PADDLE_GET_CONST(float, dev_ctx.GetDnnAttr("shift"))
          : 0.0f;
  const auto output_data_type =
      dev_ctx.HasDnnAttr("output_data_type")
          ? PADDLE_GET_CONST(std::string,
                             dev_ctx.GetDnnAttr("output_data_type"))
          : "";
  const bool with_scale = quantization_scale != 1.0f;
  const bool with_shift = quantization_shift != 0.0f;

  if (with_scale) {
    attrs.set_output_scales(mask, {quantization_scale});
  }

  if (with_shift) {
    auto dst = output_data_type == "fp32" ? DNNL_ARG_SRC : DNNL_ARG_DST;
    attrs.set_zero_points(
        dst, mask, {static_cast<int32_t>(quantization_shift)});
  }

  DataType out_dtype;
  if (output_data_type == "bf16") {
    out_dtype = DataType::BFLOAT16;
  } else if (output_data_type == "int8") {
    out_dtype = DataType::INT8;
  } else if (output_data_type == "uint8") {
    out_dtype = DataType::UINT8;
  } else if (output_data_type == "fp32") {
    out_dtype = DataType::FLOAT32;
  } else {
    out_dtype = x.dtype();
  }
  auto out_type = phi::funcs::ToOneDNNDataType(out_dtype);

  funcs::ReorderOneDNNHandler reorder_handler(
      x_vec_dims, x.dtype(), x_type, out_dtype, out_type, dev_ctx.GetEngine());

  auto reorder_src_memory_p = reorder_handler.AcquireSrcMemory(
      x.mem_desc(), funcs::to_void_cast(x.data<T>()));

  auto fake_strides = funcs::FakeTransposeStrides(x_vec_dims, axis);
  auto dst_md = dnnl::memory::desc(x_vec_dims, out_type, fake_strides);
  auto reorder_dst_memory_p =
      reorder_handler.AcquireDstMemory(out, dst_md, dev_ctx.GetPlace());

  auto reorder_p = reorder_handler.AcquireReorder(
      reorder_dst_memory_p, reorder_src_memory_p, attrs);

  auto& astream = OneDNNContext::tls().get_stream();
  reorder_p->execute(astream, *reorder_src_memory_p, *reorder_dst_memory_p);
  astream.wait();

  // it is needed because oneDNN's permute axis understand axes order in
  // different way PaddlePaddle's transpose
  out->set_mem_desc(reorder_dst_memory_p->get_desc().permute_axes(
      funcs::TransposeToPermuteAxes(axis)));
}
}  // namespace phi

PD_REGISTER_KERNEL(transpose,
                   OneDNN,
                   ONEDNN,
                   phi::TransposeKernel,
                   float,
                   uint8_t,
                   int8_t,
                   phi::dtype::bfloat16) {}
