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

#include <iterator>  // NOLINT
#include "dnnl.hpp"  // NOLINT
#include "paddle/phi/backends/onednn/onednn_helper.h"
#include "paddle/phi/backends/onednn/onednn_reuse.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

using dnnl::memory;

namespace {

inline uint8_t clip_to_uint8(float x) {
  return std::max(0L, std::min(255L, std::lround(x)));
}

}  // namespace

template <typename T, typename Context>
void ReQuantOpKernel(const Context& dev_ctx,
                     const DenseTensor& input,
                     float scale_in,
                     float scale_out,
                     float shift_in,
                     float shift_out,
                     DenseTensor* out) {
  bool with_shift = shift_in != 0 || shift_out != 0;
  auto* output = out;

  PADDLE_ENFORCE_NE(
      scale_in,
      0.0f,
      common::errors::InvalidArgument("Scale of input cannot be 0.0"));
  PADDLE_ENFORCE_NE(
      scale_out,
      0.0f,
      common::errors::InvalidArgument("Scale of output cannot be 0.0"));
  if (shift_in != 0) {
    PADDLE_ENFORCE_EQ(
        input.dtype(),
        DataType::UINT8,
        common::errors::Unimplemented("Requantize does not support nonzero "
                                      "shift for signed input."));
  }

  auto src_tz = common::vectorize(input.dims());

  auto src_paddle_dt = input.dtype();
  auto dst_paddle_dt = with_shift ? DataType::UINT8 : src_paddle_dt;

  auto xstrides = input.mem_desc().get_strides();

  dnnl::primitive_attr attrs;
  int mask = 0;
  float reorder_scale = scale_in / scale_out;
  attrs.set_scales_mask(DNNL_ARG_DST, mask);
  auto scales_md = dnnl::memory::desc(
      {1}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::x);
  auto scales_mem =
      dnnl::memory(scales_md,
                   dev_ctx.GetEngine(),
                   phi::funcs::to_void_cast<float>(&reorder_scale));

  uint32_t reorder_shift =
      with_shift ? clip_to_uint8(shift_out - (1.0f / reorder_scale) * shift_in)
                 : 0;

  if (with_shift) {
    attrs.set_zero_points_mask(DNNL_ARG_DST, mask);
  }

  phi::funcs::ReorderOneDNNHandler reorder_handler(
      src_tz,
      src_paddle_dt,
      phi::funcs::ToOneDNNDataType(src_paddle_dt),
      dst_paddle_dt,
      phi::funcs::ToOneDNNDataType(dst_paddle_dt),
      dev_ctx.GetEngine());

  auto src_memory_p = reorder_handler.AcquireSrcMemory(
      input.mem_desc(), phi::funcs::to_void_cast(input.data<T>()));
  auto dst_memory_p = reorder_handler.AcquireDstMemory(
      output, src_tz, xstrides, dev_ctx.GetPlace());

  auto reorder_p =
      reorder_handler.AcquireReorder(dst_memory_p, src_memory_p, attrs);

  auto& astream = phi::OneDNNContext::tls().get_stream();

  auto zero_points_md = dnnl::memory::desc(
      {1}, dnnl::memory::data_type::s32, dnnl::memory::format_tag::x);
  auto zero_points_out_mem =
      dnnl::memory(zero_points_md, dev_ctx.GetEngine(), &reorder_shift);

  std::unordered_map<int, dnnl::memory> reorder_args;
  reorder_args.insert({DNNL_ARG_SRC, *src_memory_p});
  reorder_args.insert({DNNL_ARG_DST, *dst_memory_p});
  reorder_args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST, scales_mem});
  // shift for DST
  if (with_shift) {
    reorder_args.insert(
        {DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_DST, zero_points_out_mem});
  }

  reorder_p->execute(astream, reorder_args);
  astream.wait();

  output->set_mem_desc(dst_memory_p->get_desc());
}

}  // namespace phi

PD_REGISTER_KERNEL(requantize,
                   OneDNN,
                   ONEDNN,
                   phi::ReQuantOpKernel,
                   int8_t,
                   uint8_t,
                   phi::dtype::bfloat16) {}
