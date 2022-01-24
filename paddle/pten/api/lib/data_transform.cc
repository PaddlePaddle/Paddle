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

#include "paddle/pten/api/lib/data_transform.h"

#include "paddle/pten/api/ext/dispatch.h"
#include "paddle/pten/api/lib/kernel_dispatch.h"
#include "paddle/pten/kernels/cast_kernel.h"
#include "paddle/pten/kernels/transfer_layout_kernel.h"

#include "paddle/fluid/framework/data_device_transform.h"

namespace paddle {
namespace experimental {

inline bool NeedTransformLayout(const DataLayout& l, const DataLayout& r) {
  bool ret =
      (l != DataLayout::ALL_LAYOUT && r != DataLayout::ALL_LAYOUT && l != r);
  return ret;
}

inline bool NeedTransformDataType(const DataType& l, const DataType& r) {
  return l != r;
}

inline pten::DenseTensor TransDataLayout(const pten::DenseTensor& tensor,
                                         DataLayout layout) {
  auto& pool = paddle::platform::DeviceContextPool::Instance();
  VLOG(3) << "DataLayoutTransform in, src_layout " << tensor.layout()
          << " dst_layout: " << layout;
  if (platform::is_cpu_place(tensor.place())) {
    auto* dev_ctx = static_cast<pten::CPUContext*>(pool.Get(tensor.place()));
    return pten::TransferLayout(*dev_ctx, tensor, layout);
  } else {
    PADDLE_THROW(platform::errors::PreconditionNotMet(
        "Unsupported data layout cast from CPU to GPU."));
  }
}

inline pten::DenseTensor TransDataType(const pten::DenseTensor& tensor,
                                       DataType dtype) {
  auto& pool = paddle::platform::DeviceContextPool::Instance();

  VLOG(3) << "DataTypeTransform in, src_dtype " << tensor.dtype()
          << " dst_dtype: " << dtype;

  pten::DenseTensor out(
      pten::make_intrusive<paddle::experimental::SharedStorage>(tensor.place()),
      {dtype, tensor.dims(), tensor.layout()});

  if (platform::is_cpu_place(tensor.place())) {
    auto* dev_ctx = static_cast<pten::CPUContext*>(pool.Get(tensor.place()));
    PD_VISIT_ALL_TYPES(tensor.dtype(), "CastDataType", ([&] {
                         pten::CastKernel<data_t>(dev_ctx, tensor, dtype, &out);
                       }));
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  } else if (platform::is_gpu_place(tensor.place())) {
    auto* dev_ctx = static_cast<pten::GPUContext*>(pool.Get(tensor.place()));
    PD_VISIT_ALL_TYPES(tensor.dtype(), "CastDataType", ([&] {
                         pten::CastKernel<data_t>(dev_ctx, tensor, dtype, &out);
                       }));
#endif
  } else {
    PADDLE_THROW(platform::errors::Unimplemented(
        "Place type is not supported when casting data type."));
  }
  return out;
}

std::shared_ptr<pten::TensorBase> PrepareData(
    const std::shared_ptr<pten::TensorBase>& input,
    const pten::TensorArgDef& target_args_def) {
  if (input->place() == pten::TransToFluidPlace(target_args_def.backend) &&
      !NeedTransformDataType(input->dtype(), target_args_def.dtype) &&
      !NeedTransformLayout(input->layout(), target_args_def.layout)) {
    return input;
  }

  pten::DenseTensor out = *(static_cast<pten::DenseTensor*>(input.get()));

  if (NeedTransformLayout(input->layout(), target_args_def.layout)) {
    out = TransDataLayout(out, target_args_def.layout);
  }

  if (NeedTransformDataType(input->dtype(), target_args_def.dtype)) {
    out = TransDataType(out, target_args_def.dtype);
  }

  if (!platform::is_same_place(
          out.place(), pten::TransToFluidPlace(target_args_def.backend))) {
    pten::DenseTensor result(
        pten::make_intrusive<paddle::experimental::SharedStorage>(
            pten::TransToFluidPlace(target_args_def.backend)),
        {out.dtype(), out.dims(), out.layout()});
    framework::TransDataDevice(
        out, pten::TransToFluidPlace(target_args_def.backend), &result);
    out = result;
  }
  return std::make_shared<pten::DenseTensor>(out);
}

}  // namespace experimental
}  // namespace paddle
