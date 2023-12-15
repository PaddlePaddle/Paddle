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

#include "paddle/phi/kernels/elementwise_add_grad_kernel.h"
#include "paddle/phi/kernels/elementwise_divide_grad_kernel.h"
#include "paddle/phi/kernels/elementwise_multiply_grad_kernel.h"
#include "paddle/phi/kernels/elementwise_subtract_grad_kernel.h"

#include "paddle/phi/backends/onednn/onednn_reuse.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {
namespace funcs {

inline std::vector<int64_t> CalculateBroadcastedDims(
    const phi::DenseTensor* x, const phi::DenseTensor* y) {
  const auto src_tz = common::vectorize(x->dims());
  const auto dst_tz = common::vectorize(y->dims());

  std::vector<int64_t> dst_tz_ex(src_tz.size(), 1);

  if (src_tz.size() == dst_tz.size()) {
    for (size_t i = 0; i < src_tz.size(); i++) {
      dst_tz_ex[i] = (src_tz[i] == dst_tz[i]) ? dst_tz[i] : 1;
    }
  } else {
    size_t j = 0;
    for (size_t i = 0; i < src_tz.size(); i++) {
      dst_tz_ex[i] = (src_tz[i] != dst_tz[j]) ? 1 : dst_tz[j++];
      if (j == dst_tz.size()) break;
    }
  }

  return dst_tz_ex;
}

inline void AddSubNonBroadcast(ReorderOneDNNHandler* reorder_handler,
                               phi::DenseTensor* grad_tensor,
                               const std::shared_ptr<dnnl::memory>& src_memory,
                               const std::shared_ptr<dnnl::memory>& dst_memory,
                               const dnnl::memory& scales_memory) {
  dnnl::primitive_attr reorder_attr;
  reorder_attr.set_scales_mask(DNNL_ARG_DST, 0);
  auto reorder_p =
      reorder_handler->AcquireReorder(dst_memory, src_memory, reorder_attr);

  std::unordered_map<int, dnnl::memory> args = {
      {DNNL_ARG_SRC, *src_memory},
      {DNNL_ARG_DST, *dst_memory},
      {DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST, scales_memory}};
  auto& astream = OneDNNContext::tls().get_stream();
  reorder_p->execute(astream, args);
}

template <typename T>
inline void BroadcastReduction(const Place& place,
                               const dnnl::engine& onednn_engine,
                               phi::DenseTensor* grad_tensor,
                               const phi::DenseTensor* dout,
                               const std::shared_ptr<dnnl::memory>& src_memory,
                               std::shared_ptr<dnnl::memory> dst_memory,
                               const std::vector<float>& scales,
                               const bool is_sub) {
  dnnl::primitive_attr broadcast_reduction_attr;

  // Broadcasting
  if (is_sub) {
    dnnl::post_ops po;
    po.append_eltwise(dnnl::algorithm::eltwise_linear, scales[0], 0);
    broadcast_reduction_attr.set_post_ops(po);
  }

  ReductionOneDNNHandler<T> reduction_handler(
      dnnl::algorithm::reduction_sum,
      0.0f,
      0.0f,
      onednn_engine,
      place,
      dout,
      grad_tensor,
      CalculateBroadcastedDims(dout, grad_tensor),
      broadcast_reduction_attr);
  dst_memory = reduction_handler.AcquireDstMemory(grad_tensor);

  auto reduction_p = reduction_handler.AcquireForwardPrimitive();
  auto astream = OneDNNContext::tls().get_stream();
  reduction_p->execute(astream,
                       {
                           {DNNL_ARG_SRC, *src_memory},
                           {DNNL_ARG_DST, *dst_memory},
                       });
  astream.wait();
  auto grad_shape = grad_tensor->dims().size() == 0
                        ? std::vector<int64_t>{1}
                        : common::vectorize<int64_t>(grad_tensor->dims());
  grad_tensor->set_mem_desc(dst_memory->get_desc().reshape(grad_shape));
}

}  // namespace funcs

template <typename T, dnnl::algorithm BINARY_OP>
void ElementwiseGradKernel(const OneDNNContext& dev_ctx,
                           const DenseTensor& x,
                           const DenseTensor& y,
                           const DenseTensor* out,
                           const DenseTensor& dout,
                           int axis,
                           DenseTensor* dx,
                           DenseTensor* dy) {
  const auto& onednn_engine = dev_ctx.GetEngine();
  // oneDNN's binary is optimized for broadcasting y into x, so in other case
  // we have to swap tensors to achieve optimal performance
  bool swap_x_y = false;
  auto* non_const_x = &x;
  auto* non_const_y = &y;
  if (x.numel() < y.numel()) {
    std::swap(non_const_x, non_const_y);
    std::swap(dx, dy);
    swap_x_y = true;
  }

  float scale{1.0};
  if (swap_x_y) {
    scale = (BINARY_OP == dnnl::algorithm::binary_add) ? 1 : -1;
  }

  auto tz = common::vectorize<int64_t>(dout.dims());

  funcs::ReorderOneDNNHandler reorder_handler(
      tz, dout.dtype(), funcs::ToOneDNNDataType(dout.dtype()), onednn_engine);

  auto reorder_src_memory = reorder_handler.AcquireSrcMemory(
      dout.mem_desc(), funcs::to_void_cast(dout.data<T>()));

  std::shared_ptr<dnnl::memory> dst_memory;
  std::shared_ptr<dnnl::memory> broadcast_src_memory = reorder_src_memory;

  auto& astream = OneDNNContext::tls().get_stream();
  auto scales_md = dnnl::memory::desc(
      {1}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::x);
  auto scales_mem = dnnl::memory(scales_md, onednn_engine);
  auto scale_memory_buf = static_cast<float*>(scales_mem.get_data_handle());
  *scale_memory_buf = scale;
  if (dx) {
    // elementwise_add & elementwise_sub
    if (BINARY_OP == dnnl::algorithm::binary_add ||
        BINARY_OP == dnnl::algorithm::binary_sub) {
      if (dout.dims() == dx->dims()) {
        dst_memory = reorder_handler.AcquireDstMemory(
            dx, dout.mem_desc(), dev_ctx.GetPlace());
        AddSubNonBroadcast(
            &reorder_handler, dx, reorder_src_memory, dst_memory, scales_mem);
      }
    } else {  // elementwise_mul & elementwise_div
      funcs::BinaryOneDNNHandler<T> binary_handler(BINARY_OP,
                                                   axis,
                                                   onednn_engine,
                                                   dev_ctx.GetPlace(),
                                                   &dout,
                                                   non_const_y,
                                                   dx,
                                                   1.0f,
                                                   1.0f,
                                                   1.0f,
                                                   false);

      const auto src_dout_memory = binary_handler.AcquireSrcMemory(&dout);
      const auto src_y_memory =
          binary_handler.AcquireSecondSrcMemory(non_const_y);
      dst_memory = binary_handler.AcquireDstMemory(dx);

      const auto binary_prim = binary_handler.AcquireForwardPrimitive();

      const std::unordered_map<int, dnnl::memory> args = {
          {DNNL_ARG_SRC_0, *src_dout_memory},
          {DNNL_ARG_SRC_1, *src_y_memory},
          {DNNL_ARG_DST, *dst_memory},
          {DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC_0, scales_mem},
          {DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC_1, scales_mem}};

      binary_prim->execute(astream, args);
    }
    astream.wait();

    if (dout.dims() != dx->dims()) {
      funcs::BroadcastReduction<T>(dev_ctx.GetPlace(),
                                   onednn_engine,
                                   dx,
                                   &dout,
                                   broadcast_src_memory,
                                   dst_memory,
                                   {scale},
                                   BINARY_OP == dnnl::algorithm::binary_sub);
    } else {
      dx->set_mem_desc(dst_memory->get_desc());
    }
  }

  if (dy) {
    // elementwise_add & elementwise_sub
    if (BINARY_OP == dnnl::algorithm::binary_add ||
        BINARY_OP == dnnl::algorithm::binary_sub) {
      if (dout.dims() == dy->dims()) {
        dst_memory = reorder_handler.AcquireDstMemory(
            dy, dout.mem_desc(), dev_ctx.GetPlace());
        AddSubNonBroadcast(
            &reorder_handler, dy, reorder_src_memory, dst_memory, scales_mem);
      }
    } else {  // elementwise_mul & elementwise_div
      std::unordered_map<int, dnnl::memory> args;
      std::shared_ptr<dnnl::binary> binary_prim;
      std::shared_ptr<dnnl::memory> post_op_memory;
      std::shared_ptr<dnnl::memory> src_0_memory;
      std::shared_ptr<dnnl::memory> src_1_memory;

      funcs::BinaryOneDNNHandler<T> binary_handler(dnnl::algorithm::binary_mul,
                                                   axis,
                                                   onednn_engine,
                                                   dev_ctx.GetPlace(),
                                                   &dout,
                                                   non_const_x,
                                                   nullptr,
                                                   1.0f,
                                                   1.0f,
                                                   1.0f,
                                                   false);

      src_1_memory = binary_handler.AcquireSecondSrcMemory(non_const_x);

      if (BINARY_OP == dnnl::algorithm::binary_div) {
        funcs::BinaryOneDNNHandler<T> post_op_binary_handler(
            dnnl::algorithm::binary_div,
            axis,
            onednn_engine,
            dev_ctx.GetPlace(),
            non_const_y,
            non_const_y,
            nullptr,
            1.0f,
            1.0f,
            1.0f,
            false);

        post_op_memory = post_op_binary_handler.AcquireSrcMemory(non_const_y);

        dnnl::post_ops po;
        po.append_binary(dnnl::algorithm::binary_div,
                         post_op_memory->get_desc());

        binary_handler =
            funcs::BinaryOneDNNHandler<T>(dnnl::algorithm::binary_mul,
                                          axis,
                                          onednn_engine,
                                          dev_ctx.GetPlace(),
                                          &dout,
                                          out,
                                          nullptr,
                                          -1.0f,
                                          1.0f,
                                          1.0f,
                                          false,
                                          po);

        src_1_memory = binary_handler.AcquireSecondSrcMemory(out);
      }

      src_0_memory = binary_handler.AcquireSrcMemory(&dout);

      const auto dst_dy_memory = (dout.dims() == dy->dims())
                                     ? binary_handler.AcquireDstMemory(dy)
                                     : binary_handler.AcquireDstMemory();

      binary_prim = binary_handler.AcquireForwardPrimitive();
      args = {{DNNL_ARG_SRC_0, *src_0_memory},
              {DNNL_ARG_SRC_1, *src_1_memory},
              {DNNL_ARG_DST, *dst_dy_memory},
              {DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC_0, scales_mem},
              {DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC_1, scales_mem}};

      if (BINARY_OP == dnnl::algorithm::binary_div)
        args.insert({DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1,
                     *post_op_memory});

      binary_prim->execute(astream, args);
      broadcast_src_memory = dst_dy_memory;
      dst_memory = dst_dy_memory;
    }
    astream.wait();

    if (dout.dims() != dy->dims()) {
      funcs::BroadcastReduction<T>(dev_ctx.GetPlace(),
                                   onednn_engine,
                                   dy,
                                   &dout,
                                   broadcast_src_memory,
                                   dst_memory,
                                   {scale},
                                   BINARY_OP == dnnl::algorithm::binary_sub);
    } else {
      dy->set_mem_desc(dst_memory->get_desc());
    }
  }
}

#define DEFINE_ONEDNN_ELEMENTWISE_GRAD_KERNEL(name, algorithm) \
  template <typename T, typename Context>                      \
  void name##GradKernel(const Context& dev_ctx,                \
                        const DenseTensor& x,                  \
                        const DenseTensor& y,                  \
                        const DenseTensor& dout,               \
                        int axis,                              \
                        DenseTensor* dx,                       \
                        DenseTensor* dy) {                     \
    ElementwiseGradKernel<T, algorithm>(                       \
        dev_ctx, x, y, nullptr, dout, axis, dx, dy);           \
  }

DEFINE_ONEDNN_ELEMENTWISE_GRAD_KERNEL(Add, dnnl::algorithm::binary_add)
DEFINE_ONEDNN_ELEMENTWISE_GRAD_KERNEL(Subtract, dnnl::algorithm::binary_sub)
DEFINE_ONEDNN_ELEMENTWISE_GRAD_KERNEL(Multiply, dnnl::algorithm::binary_mul)

template <typename T, typename Context>
void DivideGradKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      const DenseTensor& y,
                      const DenseTensor& out,
                      const DenseTensor& dout,
                      int axis,
                      DenseTensor* dx,
                      DenseTensor* dy) {
  ElementwiseGradKernel<T, dnnl::algorithm::binary_div>(
      dev_ctx, x, y, &out, dout, axis, dx, dy);
}
}  // namespace phi

PD_REGISTER_KERNEL(
    add_grad, OneDNN, ONEDNN, phi::AddGradKernel, float, phi::dtype::bfloat16) {
}

PD_REGISTER_KERNEL(subtract_grad,
                   OneDNN,
                   ONEDNN,
                   phi::SubtractGradKernel,
                   float,
                   phi::dtype::bfloat16) {}

PD_REGISTER_KERNEL(multiply_grad,
                   OneDNN,
                   ONEDNN,
                   phi::MultiplyGradKernel,
                   float,
                   phi::dtype::bfloat16) {}

PD_REGISTER_KERNEL(divide_grad,
                   OneDNN,
                   ONEDNN,
                   phi::DivideGradKernel,
                   float,
                   phi::dtype::bfloat16) {}
