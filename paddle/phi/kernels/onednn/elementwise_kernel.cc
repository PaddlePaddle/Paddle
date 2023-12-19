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

#include "paddle/phi/kernels/elementwise_add_kernel.h"
#include "paddle/phi/kernels/elementwise_divide_kernel.h"
#include "paddle/phi/kernels/elementwise_multiply_kernel.h"
#include "paddle/phi/kernels/elementwise_subtract_kernel.h"

#include "paddle/phi/backends/onednn/onednn_reuse.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

KernelKey ElementwiseGetKernelTypeForVar(
    const GetKernelTypeForVarContext* ctx) {
  const DenseTensor& tensor = ctx->GetTensor();
  const KernelKey& expected_kernel_type = ctx->GetKernelKey();
  // Only input require reshaping, weights and
  // bias are having shape in NCHW order
  if (expected_kernel_type.dtype() == phi::DataType::COMPLEX64 ||
      expected_kernel_type.dtype() == phi::DataType::COMPLEX128) {
    // only promote inputs’s types when contains complex input
    return phi::KernelKey(tensor.place(), tensor.layout(), tensor.dtype());
  } else {
    // When elementwise is first oneDNN op (there was some non oneDNN op
    // previously)
    // then we also need to rotate shape NHWC -> NCWH
    if ((expected_kernel_type.layout() == phi::DataLayout::ONEDNN) &&
        (tensor.layout() != phi::DataLayout::ONEDNN) &&
        phi::OneDNNContext::tls().get_cur_paddle_data_layout() ==
            phi::DataLayout::kNHWC) {
      return phi::KernelKey(
          tensor.place(), phi::DataLayout::kNHWC, expected_kernel_type.dtype());
    }
    return phi::KernelKey(
        tensor.place(), tensor.layout(), expected_kernel_type.dtype());
  }
}

template <typename T, dnnl::algorithm BINARY_OP>
void ElementwiseKernel(const OneDNNContext& dev_ctx,
                       const DenseTensor& x,
                       const DenseTensor& y,
                       int axis,
                       DenseTensor* out) {
  const auto& onednn_engine = dev_ctx.GetEngine();

  auto* non_const_x = &x;
  auto* non_const_y = &y;

  funcs::BinaryOneDNNHandler<T> handler(BINARY_OP,
                                        axis,
                                        onednn_engine,
                                        dev_ctx.GetPlace(),
                                        non_const_x,
                                        non_const_y,
                                        out,
                                        1.0f,
                                        1.0f,
                                        1.0f,
                                        true);

  // oneDNN's binary is optimized for broadcasting y into x, so in other case
  // we have to swap tensors to achieve optimal performance
  if (x.numel() < y.numel()) {
    std::swap(non_const_x, non_const_y);
  }

  const auto src_x_memory =
      handler.swin_case ? (x.numel() == y.numel()
                               ? handler.AcquireExtendSrcMemory(non_const_x, 0)
                               : handler.AcquireSrcMemory(non_const_x))
                        : handler.AcquireSrcMemory(non_const_x);

  const auto src_y_memory =
      handler.swin_case ? (x.numel() == y.numel()
                               ? handler.AcquireSecondSrcMemory(non_const_y)
                               : handler.AcquireExtendSrcMemory(non_const_y, 1))
                        : handler.AcquireSecondSrcMemory(non_const_y);

  // (jczaja) For Inplace src and dst should be the same memory object.
  // So x should share buffer with z. But UT mechanics is testing inplace
  // execution for this op not checking that x can be bradcasted to match in
  // shape y tensor.
  // This is wrong as when x is to be broadcasted then z(out) will match the
  // shape of y which is bigger than x. Hence if x is smaller in shape than z
  // and they share a buffer (of
  // shape x) then this buffer is not big enough to hold result of elementwise
  // operation.
  const bool reuse_x_memory = non_const_x->numel() == out->numel() &&
                              non_const_x->IsSharedBufferWith(*out);
  std::shared_ptr<dnnl::memory> dst_memory;

  if (reuse_x_memory) {
    dst_memory = src_x_memory;
    // NOTE(chenfeiyu): when the output reuses memory from other tensor rather
    // than allocate its own, it's still need to take care of its data type.
    // Unfortunately, paddle's operator only infers the output' shape, but not
    // the data type. Alloc<T> takes care of allocation and data type
    // normally, but if the memory is already allocated and there is no need
    // to re-allocate, it just set the data type. So this it added there to
    // get the right data type.
    dev_ctx.template Alloc<T>(out);
  } else {
    dst_memory = handler.AcquireDstMemory(out);
  }

  const auto binary_prim = handler.AcquireForwardPrimitive();

  auto& astream = OneDNNContext::tls().get_stream();

  std::unordered_map<int, dnnl::memory> args = {{DNNL_ARG_SRC_0, *src_x_memory},
                                                {DNNL_ARG_SRC_1, *src_y_memory},
                                                {DNNL_ARG_DST, *dst_memory}};

  if (handler.Has_SRC_0_Scale()) {
    args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC_0,
                 handler.Get_SRC_0_Scale_Memory()});
  }

  if (handler.Has_SRC_1_Scale()) {
    args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC_1,
                 handler.Get_SRC_1_Scale_Memory()});
  }

  binary_prim->execute(astream, args);
  astream.wait();

  auto out_md = dst_memory->get_desc();

  if (handler.use_broadcasting_hack) {
    auto dims = out_md.get_dims();
    dims.insert(dims.begin(), non_const_x->dims()[0]);
    dims[1] /= dims[0];
    out_md = out_md.reshape(dims);
  }

  out->set_mem_desc(out_md);
}

#define DEFINE_ONEDNN_ELEMENTWISE_KERNEL(name, algorithm)      \
  template <typename T, typename Context>                      \
  void name##RawKernel(const Context& dev_ctx,                 \
                       const DenseTensor& x,                   \
                       const DenseTensor& y,                   \
                       int axis,                               \
                       DenseTensor* out) {                     \
    ElementwiseKernel<T, algorithm>(dev_ctx, x, y, axis, out); \
  }                                                            \
  template <typename T, typename Context>                      \
  void name##Kernel(const Context& dev_ctx,                    \
                    const DenseTensor& x,                      \
                    const DenseTensor& y,                      \
                    DenseTensor* out) {                        \
    ElementwiseKernel<T, algorithm>(dev_ctx, x, y, -1, out);   \
  }

DEFINE_ONEDNN_ELEMENTWISE_KERNEL(Add, dnnl::algorithm::binary_add)
DEFINE_ONEDNN_ELEMENTWISE_KERNEL(Subtract, dnnl::algorithm::binary_sub)
DEFINE_ONEDNN_ELEMENTWISE_KERNEL(Multiply, dnnl::algorithm::binary_mul)
DEFINE_ONEDNN_ELEMENTWISE_KERNEL(Divide, dnnl::algorithm::binary_div)

}  // namespace phi

PD_REGISTER_KERNEL(add_raw,
                   OneDNN,
                   ONEDNN,
                   phi::AddRawKernel,
                   float,
                   phi::dtype::bfloat16,
                   int8_t,
                   uint8_t) {
  kernel->get_kerneltype_forvar_fn_ = phi::ElementwiseGetKernelTypeForVar;
}

PD_REGISTER_KERNEL(add,
                   OneDNN,
                   ONEDNN,
                   phi::AddKernel,
                   float,
                   phi::dtype::bfloat16,
                   int8_t,
                   uint8_t) {
  kernel->get_kerneltype_forvar_fn_ = phi::ElementwiseGetKernelTypeForVar;
}

PD_REGISTER_KERNEL(subtract_raw,
                   OneDNN,
                   ONEDNN,
                   phi::SubtractRawKernel,
                   float,
                   phi::dtype::bfloat16,
                   int8_t,
                   uint8_t) {
  kernel->get_kerneltype_forvar_fn_ = phi::ElementwiseGetKernelTypeForVar;
}

PD_REGISTER_KERNEL(subtract,
                   OneDNN,
                   ONEDNN,
                   phi::SubtractKernel,
                   float,
                   phi::dtype::bfloat16,
                   int8_t,
                   uint8_t) {
  kernel->get_kerneltype_forvar_fn_ = phi::ElementwiseGetKernelTypeForVar;
}

PD_REGISTER_KERNEL(multiply_raw,
                   OneDNN,
                   ONEDNN,
                   phi::MultiplyRawKernel,
                   float,
                   phi::dtype::bfloat16,
                   int8_t,
                   uint8_t) {
  kernel->get_kerneltype_forvar_fn_ = phi::ElementwiseGetKernelTypeForVar;
}

PD_REGISTER_KERNEL(multiply,
                   OneDNN,
                   ONEDNN,
                   phi::MultiplyKernel,
                   float,
                   phi::dtype::bfloat16,
                   int8_t,
                   uint8_t) {
  kernel->get_kerneltype_forvar_fn_ = phi::ElementwiseGetKernelTypeForVar;
}

PD_REGISTER_KERNEL(divide_raw,
                   OneDNN,
                   ONEDNN,
                   phi::DivideRawKernel,
                   float,
                   phi::dtype::bfloat16) {}

PD_REGISTER_KERNEL(
    divide, OneDNN, ONEDNN, phi::DivideKernel, float, phi::dtype::bfloat16) {
  kernel->get_kerneltype_forvar_fn_ = phi::ElementwiseGetKernelTypeForVar;
}
