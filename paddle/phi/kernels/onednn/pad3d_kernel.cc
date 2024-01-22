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

#include "paddle/phi/kernels/pad3d_kernel.h"

#include "paddle/phi/backends/onednn/onednn_reuse.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/onednn/pad_kernel_impl.h"

namespace phi {

KernelKey Pad3dGetKernelTypeForVar(const GetKernelTypeForVarContext* ctx) {
  const DenseTensor& tensor = ctx->GetTensor();
  const KernelKey& expected_kernel_type = ctx->GetKernelKey();
  const AttributeMap& attrs = ctx->GetAttrs();
#ifdef PADDLE_WITH_DNNL
  if ((expected_kernel_type.layout() == phi::DataLayout::ONEDNN) &&
      (tensor.layout() != phi::DataLayout::ONEDNN)) {
    auto it = attrs.find("data_format");
    const std::string data_format = PADDLE_GET_CONST(std::string, it->second);
    return phi::KernelKey(tensor.place(),
                          common::StringToDataLayout(data_format),
                          expected_kernel_type.dtype());
  }
#endif
  return phi::KernelKey(
      tensor.place(), tensor.layout(), expected_kernel_type.dtype());
}

bool Pad3dCheckIfOneDNNSupport(const KernelContext* ctx) {
  // only constant mode and non-blocked layouts are supported for oneDNN
  if (ctx->AttrAt<std::string>(1) == "constant" &&
      ctx->InputAt<phi::DenseTensor>(0).mem_desc().get_inner_nblks() == 0) {
    return true;
  }
  return false;
}

template <typename T, typename Context>
void Pad3dKernel(const Context& dev_ctx,
                 const DenseTensor& x,
                 const IntArray& paddings,
                 const std::string& mode UNUSED,
                 float pad_value,
                 const std::string& data_format UNUSED,
                 DenseTensor* out) {
  PadOpKernel<T, Context>(dev_ctx, x, paddings.GetData(), pad_value, out);
}
}  // namespace phi

PD_REGISTER_KERNEL(pad3d,
                   OneDNN,
                   ONEDNN,
                   phi::Pad3dKernel,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   float) {
  kernel->get_kerneltype_forvar_fn_ = phi::Pad3dGetKernelTypeForVar;
  kernel->check_if_onednn_kernel_support_ = phi::Pad3dCheckIfOneDNNSupport;
}
