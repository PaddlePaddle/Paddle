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

#include "paddle/phi/backends/onednn/onednn_reuse.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/transpose_kernel.h"

namespace phi {

void SetInMemDescWithSqueeze2FuseSupport(
    const std::vector<int> fused_squeeze2_axes,
    DenseTensor* in,
    const dnnl::memory::desc& in_md) {
  const std::set<int64_t> squeeze2_axes_set(fused_squeeze2_axes.begin(),
                                            fused_squeeze2_axes.end());
  const std::vector<int64_t>& x_vec_dims = in_md.dims();
  std::vector<int64_t> squeezed_op_tz(
      x_vec_dims.size() - fused_squeeze2_axes.size(), 0);

  int j = 0;
  for (size_t i = 0; i < x_vec_dims.size(); ++i) {
    if (squeeze2_axes_set.count(i) ||
        squeeze2_axes_set.count(i - x_vec_dims.size())) {
      PADDLE_ENFORCE_EQ(
          x_vec_dims[i],
          1,
          errors::InvalidArgument(
              "Squeeze2 input dim %d should be equal to one, but get %d.",
              i,
              x_vec_dims[i]));
      continue;
    }
    squeezed_op_tz[j++] = x_vec_dims[i];
  }

  in->set_mem_desc(in_md.reshape(squeezed_op_tz));
  in->Resize(make_ddim(squeezed_op_tz));
}

template <typename T, typename Context>
void FusedTransposeKernel(const Context& dev_ctx,
                          const DenseTensor& x,
                          const std::vector<int>& axis,
                          const std::vector<int>& fused_squeeze2_axes,
                          const std::vector<int>& fused_unsqueeze2_axes,
                          const std::vector<int>& fused_reshape2_shape,
                          const float scale,
                          const float shift,
                          const std::string& output_data_type,
                          DenseTensor* out) {
  PADDLE_ENFORCE_EQ(
      dev_ctx.GetPlace().GetType(),
      AllocationType::CPU,
      errors::PreconditionNotMet("oneDNN Transpose kernel must use CPUPlace"));

  if (!(fused_squeeze2_axes.empty())) {
    SetInMemDescWithSqueeze2FuseSupport(
        fused_squeeze2_axes, const_cast<DenseTensor*>(&x), x.mem_desc());
  }

  if (axis.size() == 1) {
    Copy<Context>(dev_ctx, x, x.place(), false, out);
    out->set_mem_desc(x.mem_desc());
    return;
  }

  auto x_vec_dims = vectorize(x.dims());
  auto x_type = funcs::ToOneDNNDataType(x.dtype());

  dnnl::primitive_attr attrs;
  const int32_t mask = 0;

  if (scale != 1.0f) {
    attrs.set_output_scales(mask, {scale});
  }

  if (shift != 0.0f) {
    auto dst = output_data_type == "fp32" ? DNNL_ARG_SRC : DNNL_ARG_DST;
    attrs.set_zero_points(dst, mask, {static_cast<int32_t>(shift)});
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
  auto out_type = funcs::ToOneDNNDataType(out_dtype);

  funcs::ReorderOneDNNHandler reorder_handler(
      x_vec_dims, x.dtype(), x_type, out_dtype, out_type, dev_ctx.GetEngine());

  auto reorder_src_memory_p = reorder_handler.AcquireSrcMemory(
      x.mem_desc(), funcs::to_void_cast(x.data<T>()));

  // a trick is used here to fake transpose of out_md, so later it will be
  // "untransposed", leaving output data in plain format tag
  auto fake_strides = funcs::FakeTransposeStrides(x_vec_dims, axis);
  auto dst_md = dnnl::memory::desc(x_vec_dims, out_type, fake_strides);
  auto reorder_dst_memory_p =
      reorder_handler.AcquireDstMemory(out, dst_md, dev_ctx.GetPlace());

  auto reorder_p = reorder_handler.AcquireReorder(
      reorder_dst_memory_p, reorder_src_memory_p, attrs);

  auto& astream = OneDNNContext::tls().get_stream();
  reorder_p->execute(astream, *reorder_src_memory_p, *reorder_dst_memory_p);
  astream.wait();

  auto out_md = reorder_dst_memory_p->get_desc().permute_axes(
      funcs::TransposeToPermuteAxes(axis));

  if (!fused_unsqueeze2_axes.empty()) {
    funcs::SetOutMemDescWithUnsqueeze2FuseSupport(
        fused_unsqueeze2_axes, out, out_md);
  } else if (!fused_reshape2_shape.empty()) {
    funcs::SetOutMemDescWithReshape2FuseSupport(
        fused_reshape2_shape, out, out_md);
  } else if (!fused_squeeze2_axes.empty()) {
    out->set_mem_desc(out_md);
    out->Resize(make_ddim(out_md.dims()));
  } else {
    out->set_mem_desc(out_md);
  }
}
}  // namespace phi

PD_REGISTER_KERNEL(fused_transpose,
                   OneDNN,
                   ONEDNN,
                   phi::FusedTransposeKernel,
                   float,
                   uint8_t,
                   int8_t,
                   phi::dtype::bfloat16) {}
