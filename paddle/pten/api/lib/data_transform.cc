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
#include "paddle/pten/backends/all_context.h"
#include "paddle/pten/kernels/cast_kernel.h"
#include "paddle/pten/kernels/transfer_layout_kernel.h"

#include "paddle/fluid/framework/data_device_transform.h"

namespace paddle {
namespace experimental {

inline bool NeedTransformDataType(const DataType& input,
                                  const DataType& target,
                                  const TransformFlag& transform_flag) {
  return input != target &&
         (transform_flag.need_trans_data_type() ||
          target == DataType::COMPLEX64 || target == DataType::COMPLEX128);
}

inline bool NeedTransformPlace(const paddle::platform::Place& input,
                               const Backend& target,
                               const TransformFlag& transform_flag) {
  bool ret = transform_flag.need_trans_backend() &&
             target != Backend::ALL_BACKEND &&
             !platform::is_same_place(input, pten::TransToPtenPlace(target));
  return ret;
}

inline bool NeedTransformLayout(const DataLayout& input,
                                const DataLayout& target,
                                const TransformFlag& transform_flag) {
  bool ret = transform_flag.need_trans_layout() &&
             (input != DataLayout::ALL_LAYOUT &&
              target != DataLayout::ALL_LAYOUT && input != target);
  return ret;
}

inline pten::DenseTensor TransDataLayout(const pten::DenseTensor& tensor,
                                         DataLayout layout) {
  auto& pool = paddle::platform::DeviceContextPool::Instance();
  VLOG(3) << "DataLayoutTransform src_layout: " << tensor.layout()
          << " dst_layout: " << layout;
  if (platform::is_cpu_place(tensor.place())) {
    auto* dev_ctx = static_cast<pten::CPUContext*>(pool.Get(tensor.place()));
    return pten::TransferLayout(*dev_ctx, tensor, layout);
  } else {
    PADDLE_THROW(pten::errors::PreconditionNotMet(
        "Unsupported data layout cast from CPU to GPU."));
  }
}

template <typename Context>
pten::DenseTensor CastDateType(const Context& dev_ctx,
                               const pten::DenseTensor& tensor,
                               DataType dtype) {
  switch (tensor.dtype()) {
    case DataType::FLOAT32:
      return pten::Cast<float>(dev_ctx, tensor, dtype);
    case DataType::FLOAT64:
      return pten::Cast<double>(dev_ctx, tensor, dtype);
    case DataType::INT32:
      return pten::Cast<int32_t>(dev_ctx, tensor, dtype);
    case DataType::INT64:
      return pten::Cast<int64_t>(dev_ctx, tensor, dtype);
    case DataType::FLOAT16:
      return pten::Cast<pten::dtype::float16>(dev_ctx, tensor, dtype);
    case DataType::BFLOAT16:
      return pten::Cast<pten::dtype::bfloat16>(dev_ctx, tensor, dtype);
    case DataType::BOOL:
      return pten::Cast<bool>(dev_ctx, tensor, dtype);
    case DataType::INT16:
      return pten::Cast<int16_t>(dev_ctx, tensor, dtype);
    case DataType::UINT8:
      return pten::Cast<uint8_t>(dev_ctx, tensor, dtype);
    default:
      PADDLE_THROW(pten::errors::Unimplemented(
          "Data type (%s) is not supported when casting data type.",
          tensor.dtype()));
  }
}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
pten::DenseTensor CastDateType(const pten::GPUContext& dev_ctx,
                               const pten::DenseTensor& tensor,
                               DataType dtype) {
  switch (tensor.dtype()) {
    case DataType::FLOAT32:
      return pten::Cast<float>(dev_ctx, tensor, dtype);
    case DataType::FLOAT64:
      return pten::Cast<double>(dev_ctx, tensor, dtype);
    case DataType::INT32:
      return pten::Cast<int32_t>(dev_ctx, tensor, dtype);
    case DataType::INT64:
      return pten::Cast<int64_t>(dev_ctx, tensor, dtype);
    case DataType::FLOAT16:
      return pten::Cast<pten::dtype::float16>(dev_ctx, tensor, dtype);
    case DataType::BOOL:
      return pten::Cast<bool>(dev_ctx, tensor, dtype);
    case DataType::INT16:
      return pten::Cast<int16_t>(dev_ctx, tensor, dtype);
    case DataType::UINT8:
      return pten::Cast<uint8_t>(dev_ctx, tensor, dtype);
    default:
      PADDLE_THROW(pten::errors::Unimplemented(
          "Data type (%s) is not supported when casting data type.",
          tensor.dtype()));
  }
}
#endif

inline pten::DenseTensor TransDataType(const pten::DenseTensor& tensor,
                                       DataType dtype) {
  auto& pool = paddle::platform::DeviceContextPool::Instance();

  VLOG(3) << "DataTypeTransform src_dtype: " << tensor.dtype()
          << " dst_dtype: " << dtype;

  pten::DenseTensor out(
      pten::make_intrusive<paddle::experimental::SharedStorage>(tensor.place()),
      {dtype, tensor.dims(), tensor.layout()});

  if (platform::is_cpu_place(tensor.place())) {
    auto* dev_ctx = static_cast<pten::CPUContext*>(pool.Get(tensor.place()));
    return CastDateType(*dev_ctx, tensor, dtype);
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  } else if (platform::is_gpu_place(tensor.place())) {
    auto* dev_ctx = static_cast<pten::GPUContext*>(pool.Get(tensor.place()));
    return CastDateType(*dev_ctx, tensor, dtype);
#endif
  } else {
    PADDLE_THROW(pten::errors::Unimplemented(
        "Place type is not supported when casting data type."));
  }
  return out;
}

pten::DenseTensor TransformData(const pten::DenseTensor& tensor,
                                const pten::TensorArgDef& target_args_def,
                                const TransformFlag& transform_flag) {
  pten::DenseTensor out = tensor;
  if (NeedTransformLayout(
          tensor.layout(), target_args_def.layout, transform_flag)) {
    out = TransDataLayout(out, target_args_def.layout);
  }

  if (NeedTransformDataType(
          tensor.dtype(), target_args_def.dtype, transform_flag)) {
    out = TransDataType(out, target_args_def.dtype);
  }

  if (NeedTransformPlace(
          out.place(), target_args_def.backend, transform_flag)) {
    pten::DenseTensor result(
        pten::make_intrusive<paddle::experimental::SharedStorage>(
            pten::TransToPtenPlace(target_args_def.backend)),
        {out.dtype(), out.dims(), out.layout()});
    framework::TransDataDevice(
        out, pten::TransToPtenPlace(target_args_def.backend), &result);
    out = result;
  }
  return out;
}

std::shared_ptr<pten::DenseTensor> PrepareData(
    const Tensor& input,
    const pten::TensorArgDef& target_args_def,
    const TransformFlag& transform_flag) {
  const auto& tensor_in = input.impl();
  if (!transform_flag.NeedTransform() || !tensor_in->initialized() ||
      (!NeedTransformPlace(
           tensor_in->place(), target_args_def.backend, transform_flag) &&
       !NeedTransformDataType(
           tensor_in->dtype(), target_args_def.dtype, transform_flag) &&
       !NeedTransformLayout(
           tensor_in->layout(), target_args_def.layout, transform_flag))) {
    return std::dynamic_pointer_cast<pten::DenseTensor>(tensor_in);
  }

  pten::DenseTensor out =
      TransformData(*(static_cast<pten::DenseTensor*>(tensor_in.get())),
                    target_args_def,
                    transform_flag);
  return std::make_shared<pten::DenseTensor>(out);
}

std::unique_ptr<std::vector<pten::DenseTensor>> PrepareData(
    const std::vector<Tensor>& inputs,
    const pten::TensorArgDef& target_args_def,
    const TransformFlag& transform_flag) {
  auto pt_tensors = std::make_unique<std::vector<pten::DenseTensor>>();
  pt_tensors->reserve(inputs.size());

  for (const auto& input : inputs) {
    const auto& tensor_in = input.impl();
    if (!transform_flag.NeedTransform() || !tensor_in->initialized() ||
        (!NeedTransformPlace(
             tensor_in->place(), target_args_def.backend, transform_flag) &&
         !NeedTransformDataType(
             tensor_in->dtype(), target_args_def.dtype, transform_flag) &&
         !NeedTransformLayout(
             tensor_in->layout(), target_args_def.layout, transform_flag))) {
      pt_tensors->emplace_back(
          *std::dynamic_pointer_cast<pten::DenseTensor>(tensor_in));
    } else {
      pt_tensors->emplace_back(
          TransformData(*(static_cast<pten::DenseTensor*>(tensor_in.get())),
                        target_args_def,
                        transform_flag));
    }
  }

  return std::move(pt_tensors);
}

}  // namespace experimental
}  // namespace paddle
