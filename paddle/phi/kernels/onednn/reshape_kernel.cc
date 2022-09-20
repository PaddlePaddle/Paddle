/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/reshape_kernel.h"
#include "paddle/phi/backends/onednn/onednn_reuse.h"

#include "paddle/fluid/framework/convert_utils.h"
// #include "paddle/fluid/framework/tensor_util.h"

namespace phi {

phi::DDim ValidateShape(const std::vector<int64_t>& shape,
                                   const phi::DDim& in_dims) {
  const int64_t in_size = phi::product(in_dims);
  auto in_dims_vec = phi::vectorize(in_dims);
  bool all_positive = std::all_of(in_dims_vec.cbegin(),
                                  in_dims_vec.cend(),
                                  [](int64_t i) { return i > 0; });
  // only one dimension can be set to -1, whose size will be automatically
  // infered
  const int64_t unk_dim_val = -1;
  const int64_t copy_dim_val = 0;

  std::vector<int64_t> output_shape(shape.size(), 0);
  int64_t capacity = 1;
  int unk_dim_idx = -1;
  for (size_t i = 0; i < shape.size(); ++i) {
    if (shape[i] == unk_dim_val) {
      PADDLE_ENFORCE_EQ(
          unk_dim_idx,
          -1,
          phi::errors::InvalidArgument(
              "Only one dimension value of 'shape' in ReshapeOp can "
              "be -1. But received shape = [%s], shape[%d] is also -1.",
              phi::make_ddim(shape),
              i));
      unk_dim_idx = i;
    } else if (shape[i] == copy_dim_val) {
      PADDLE_ENFORCE_LT(
          static_cast<int>(i),
          in_dims.size(),
          phi::errors::InvalidArgument(
              "The index of 0 in `shape` must be less than "
              "the input tensor X's dimensions. "
              "But received shape = [%s], shape[%d] = 0, X's shape = [%s], "
              "X's dimensions = %d.",
              phi::make_ddim(shape),
              i,
              in_dims,
              in_dims.size()));
    } else {
      PADDLE_ENFORCE_GT(
          shape[i],
          0,
          phi::errors::InvalidArgument(
              "Each dimension value of 'shape' in ReshapeOp must not "
              "be negative except one unknown dimension. "
              "But received  shape = [%s], shape[%d] = %d.",
              phi::make_ddim(shape),
              i,
              shape[i]));
    }

    capacity *= (shape[i] ? shape[i] : in_dims[i]);
    output_shape[i] = (shape[i] ? static_cast<int64_t>(shape[i]) : in_dims[i]);
  }

  if (unk_dim_idx != -1) {
    if (all_positive) {
      // in_size < 0 and is un-determinate in compile time, skip the check,
      // for example, in_dims = [-1, 8, 1, 1], shape = [-1, 3, 8],
      // capacity = -24, in_size = -8, output_shape[0] = 0
      // the following check will fail.
      output_shape[unk_dim_idx] = -in_size / capacity;
      PADDLE_ENFORCE_EQ(
          output_shape[unk_dim_idx] * capacity,
          -in_size,
          phi::errors::InvalidArgument(
              "The 'shape' attribute in ReshapeOp is invalid. "
              "The input tensor X'size must be divisible by known "
              "capacity of 'shape'. "
              "But received X's shape = [%s], X's size = %d, "
              "'shape' is [%s], known capacity of 'shape' is %d.",
              in_dims,
              in_size,
              phi::make_ddim(shape),
              capacity));
    } else {
      output_shape[unk_dim_idx] = -1;
    }
  } else {
    if (all_positive) {
      PADDLE_ENFORCE_EQ(
          capacity,
          in_size,
          phi::errors::InvalidArgument(
              "The 'shape' in ReshapeOp is invalid. "
              "The input tensor X'size must be equal to the capacity of "
              "'shape'. "
              "But received X's shape = [%s], X's size = %d, 'shape' is "
              "[%s], the capacity of 'shape' is %d.",
              in_dims,
              in_size,
              phi::make_ddim(shape),
              capacity));
    }
  }
  return phi::make_ddim(output_shape);
}

dnnl::memory::format_tag getPlainFormatTag(const DenseTensor& tensor) {
  auto tensor_dims_size = tensor.dims().size();
  PADDLE_ENFORCE_EQ(
      tensor_dims_size <= 6 && tensor_dims_size >= 1,
      true,
      phi::errors::InvalidArgument(
          "Dims for squeeze_grad oneDNN op must be in range <1, 6>"));

  switch (tensor_dims_size) {
    case 1:
      return dnnl::memory::format_tag::a;
    case 2:
      return dnnl::memory::format_tag::ab;
    case 3:
      return dnnl::memory::format_tag::abc;
    case 4:
      return dnnl::memory::format_tag::abcd;
    case 5:
      return dnnl::memory::format_tag::abcde;
    default:
      return dnnl::memory::format_tag::abcdef;
  }
}

template <typename Context>
void ReshapeKernel(const Context& dev_ctx,
                   const DenseTensor& x,
                   const IntArray& shape,
                   DenseTensor* out) {
  phi::DDim x_dims = x.dims();
  phi::DDim out_dims = out->dims();
  out_dims = ValidateShape(shape.GetData(), x_dims);

  phi::funcs::ReorderOneDNNHandler reorder_handler(
      phi::vectorize(x_dims),
      x.dtype(),
      phi::funcs::ToOneDNNDataType(x.dtype()),
      dev_ctx.GetEngine());

  auto reorder_src_memory_p = reorder_handler.AcquireSrcMemory(
      x.mem_desc(), const_cast<void*>(x.data()));
  out->Resize(x_dims);  // to match x numel, format is changed later
  // reorder is done into a plain tag to allow usage with blocked formats
  auto reorder_dst_memory_p = reorder_handler.AcquireDstMemory(
      out, getPlainFormatTag(x), dev_ctx.GetPlace());
  auto reorder_p = reorder_handler.AcquireReorder(reorder_dst_memory_p,
                                                  reorder_src_memory_p);

  auto& astream = phi::OneDNNContext::tls().get_stream();
  reorder_p->execute(astream, *reorder_src_memory_p, *reorder_dst_memory_p);

  astream.wait();

  out->Resize(out_dims);
  out->set_mem_desc(reorder_dst_memory_p->get_desc().reshape(phi::vectorize(out_dims)));
}
}  // namespace phi

PD_REGISTER_KERNEL(reshape,
                   OneDNN,
                   ALL_LAYOUT,
                   phi::ReshapeKernel<phi::OneDNNContext>,
                   float,
                   phi::dtype::bfloat16) {}
PD_REGISTER_KERNEL(reshape,
                   OneDNN,
                   ALL_LAYOUT,
                   phi::ReshapeWithXShape<phi::OneDNNContext>,
                   float,
                   phi::dtype::bfloat16) {}