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

#include "paddle/phi/api/lib/data_transform.h"

#include <sstream>

#include "glog/logging.h"

#include "paddle/common/flags.h"
#include "paddle/phi/api/lib/kernel_dispatch.h"
#include "paddle/phi/api/lib/utils/allocator.h"
#include "paddle/phi/backends/context_pool.h"
#include "paddle/phi/core/distributed/auto_parallel/dist_tensor.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard/reshard_function.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard/reshard_function_registry.h"
#include "paddle/phi/core/distributed/auto_parallel/reshard/reshard_utils.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/core/visit_type.h"
#include "paddle/phi/kernels/cast_kernel.h"
#include "paddle/phi/kernels/contiguous_kernel.h"
#include "paddle/phi/kernels/transfer_layout_kernel.h"

PHI_DECLARE_bool(use_stride_kernel);

namespace paddle::experimental {

inline bool NeedTransformDataType(const DataType& input,
                                  const DataType& target,
                                  const TransformFlag& transform_flag) {
  return input != target &&
         (transform_flag.need_trans_data_type() ||
          ((target == DataType::COMPLEX64 || target == DataType::COMPLEX128) &&
           (input != DataType::INT32 && input != DataType::INT64 &&
            input != DataType::BOOL)));
}

inline bool NeedTransformLayout(const DataLayout& input,
                                const DataLayout& target,
                                const phi::Place& place,
                                const TransformFlag& transform_flag) {
  if (FLAGS_use_stride_kernel && target == DataLayout::STRIDED) {
    return false;
  }

  bool ret = transform_flag.need_trans_layout() &&
             (input != DataLayout::ALL_LAYOUT &&
              target != DataLayout::ALL_LAYOUT && input != target);
  if (place.GetType() == phi::AllocationType::GPU) {
    return false;
  }
  return ret;
}

inline bool NeedTransform2Contiguous(bool is_stride_kernel,
                                     bool is_contiguous) {
  return FLAGS_use_stride_kernel && !is_stride_kernel && !is_contiguous;
}

inline phi::DenseTensor TransDataLayout(const phi::DenseTensor& tensor,
                                        DataLayout layout) {
  auto& pool = phi::DeviceContextPool::Instance();
  VLOG(3) << "DataLayoutTransform src_layout: " << tensor.layout()
          << " dst_layout: " << layout;
  if (tensor.place().GetType() == phi::AllocationType::CPU) {
    auto* dev_ctx = static_cast<phi::CPUContext*>(pool.Get(tensor.place()));
    return phi::TransferLayout(*dev_ctx, tensor, layout);
  } else {
    PADDLE_THROW(common::errors::PreconditionNotMet(
        "Unsupported data layout cast from CPU to GPU."));
  }
  return tensor;
}

template <typename Context>
phi::DenseTensor CastDataType(const Context& dev_ctx,
                              const phi::DenseTensor& tensor,
                              DataType dtype) {
  switch (tensor.dtype()) {
    case DataType::FLOAT32:
      return phi::Cast<float>(dev_ctx, tensor, dtype);
    case DataType::FLOAT64:
      return phi::Cast<double>(dev_ctx, tensor, dtype);
    case DataType::INT32:
      return phi::Cast<int32_t>(dev_ctx, tensor, dtype);
    case DataType::INT64:
      return phi::Cast<int64_t>(dev_ctx, tensor, dtype);
    case DataType::FLOAT16:
      return phi::Cast<phi::dtype::float16>(dev_ctx, tensor, dtype);
    case DataType::BFLOAT16:
      return phi::Cast<phi::dtype::bfloat16>(dev_ctx, tensor, dtype);
    case DataType::BOOL:
      return phi::Cast<bool>(dev_ctx, tensor, dtype);
    case DataType::INT16:
      return phi::Cast<int16_t>(dev_ctx, tensor, dtype);
    case DataType::UINT8:
      return phi::Cast<uint8_t>(dev_ctx, tensor, dtype);
    default:
      PADDLE_THROW(common::errors::Unimplemented(
          "Data type (%s) is not supported when casting data type.",
          tensor.dtype()));
  }
}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
phi::DenseTensor CastDataType(const phi::GPUContext& dev_ctx,
                              const phi::DenseTensor& tensor,
                              DataType dtype) {
  switch (tensor.dtype()) {
    case DataType::FLOAT32:
      return phi::Cast<float>(dev_ctx, tensor, dtype);
    case DataType::FLOAT64:
      return phi::Cast<double>(dev_ctx, tensor, dtype);
    case DataType::INT32:
      return phi::Cast<int32_t>(dev_ctx, tensor, dtype);
    case DataType::INT64:
      return phi::Cast<int64_t>(dev_ctx, tensor, dtype);
    case DataType::FLOAT16:
      return phi::Cast<phi::dtype::float16>(dev_ctx, tensor, dtype);
    case DataType::BOOL:
      return phi::Cast<bool>(dev_ctx, tensor, dtype);
    case DataType::INT16:
      return phi::Cast<int16_t>(dev_ctx, tensor, dtype);
    case DataType::UINT8:
      return phi::Cast<uint8_t>(dev_ctx, tensor, dtype);
    default:
      PADDLE_THROW(common::errors::Unimplemented(
          "Data type (%s) is not supported when casting data type.",
          tensor.dtype()));
  }
}
#endif

#ifdef PADDLE_WITH_XPU
phi::DenseTensor CastDataType(const phi::XPUContext& dev_ctx,
                              const phi::DenseTensor& tensor,
                              DataType dtype) {
  switch (tensor.dtype()) {
    case DataType::FLOAT32:
      return phi::Cast<float>(dev_ctx, tensor, dtype);
    case DataType::FLOAT64:
      return phi::Cast<double>(dev_ctx, tensor, dtype);
    case DataType::INT32:
      return phi::Cast<int32_t>(dev_ctx, tensor, dtype);
    case DataType::INT64:
      return phi::Cast<int64_t>(dev_ctx, tensor, dtype);
    case DataType::FLOAT16:
      return phi::Cast<phi::dtype::float16>(dev_ctx, tensor, dtype);
    case DataType::BOOL:
      return phi::Cast<bool>(dev_ctx, tensor, dtype);
    case DataType::UINT8:
      return phi::Cast<uint8_t>(dev_ctx, tensor, dtype);
    default:
      PADDLE_THROW(common::errors::Unimplemented(
          "Data type (%s) is not supported when casting data type.",
          tensor.dtype()));
  }
}
#endif

inline phi::DenseTensor TransDataType(const phi::DenseTensor& tensor,
                                      DataType dtype) {
  auto& pool = phi::DeviceContextPool::Instance();

  VLOG(3) << "DataTypeTransform src_dtype: " << tensor.dtype()
          << " dst_dtype: " << dtype;

  DefaultAllocator alloc(tensor.place());
  phi::DenseTensor out(&alloc, {dtype, tensor.dims(), tensor.layout()});

  if (tensor.place().GetType() == phi::AllocationType::CPU) {
    auto* dev_ctx = static_cast<phi::CPUContext*>(pool.Get(tensor.place()));
    return CastDataType(*dev_ctx, tensor, dtype);
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  } else if (tensor.place().GetType() == phi::AllocationType::GPU) {
    auto* dev_ctx = static_cast<phi::GPUContext*>(pool.Get(tensor.place()));
    return CastDataType(*dev_ctx, tensor, dtype);
#endif
#ifdef PADDLE_WITH_XPU
  } else if (tensor.place().GetType() == phi::AllocationType::XPU) {
    auto* dev_ctx = static_cast<phi::XPUContext*>(pool.Get(tensor.place()));
    return CastDataType(*dev_ctx, tensor, dtype);
#endif
#ifdef PADDLE_WITH_CUSTOM_DEVICE
  } else if (tensor.place().GetType() == phi::AllocationType::CUSTOM) {
    phi::DenseTensor out;
    out.Resize(tensor.dims());
    auto* dev_ctx = static_cast<phi::CustomContext*>(pool.Get(tensor.place()));
    auto kernel_result =
        phi::KernelFactory::Instance().SelectKernelOrThrowError(
            "cast",
            {phi::TransToPhiBackend(tensor.place()),
             phi::DataLayout::ALL_LAYOUT,
             tensor.dtype()});
    using kernel_signature = void (*)(const phi::DeviceContext&,
                                      const phi::DenseTensor&,
                                      phi::DataType,
                                      phi::DenseTensor*);
    const auto& kernel = kernel_result.kernel;
    auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
    (*kernel_fn)(*dev_ctx, tensor, dtype, &out);
    return out;
#endif
  } else {
    PADDLE_THROW(common::errors::Unimplemented(
        "Place type is not supported when casting data type."));
  }
  return out;
}

inline phi::DenseTensor TransDataPlace(const phi::DenseTensor& tensor,
                                       Place dst_place) {
  VLOG(3) << "DeviceTransform in, src_place " << tensor.place()
          << " dst_place: " << dst_place;

  auto& pool = phi::DeviceContextPool::Instance();
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  // NOTE(yy): TransDataPlace should wait for computation of input.
  if (tensor.place().GetType() != phi::AllocationType::GPUPINNED) {
    pool.Get(tensor.place())->Wait();
    pool.Get(dst_place)->Wait();
  }
#endif

  // FIXME(zcd): TransDataPlace is used to transform data from GPU to CPU and
  // the enforced checkings have been done in GetDeviceContext, so the
  // `dev_ctx->Wait()` is necessary. But `dev_ctx->Wait()` will make the program
  // slow, especially when the number of elements is little, for example,
  // the elements of learning rate are one and it's CPU side.
  // One solution is to use a CUDA kernel to complete the copy operation when
  // the transforming is from CPU to GPU and the number of elements is little.
  // But the embarrassment is that this solution this solution makes training
  // slower.
  phi::DenseTensor out;
  phi::DeviceContext* dev_ctx = nullptr;
  if (dst_place.GetType() != AllocationType::CPU) {
    dev_ctx = pool.Get(dst_place);
  } else {
    dev_ctx = pool.Get(tensor.place());
  }
  phi::Copy(*dev_ctx, tensor, dst_place, true, &out);
  return out;
}

template <typename Context>
phi::DenseTensor TensorContiguous(const Context& dev_ctx,
                                  const phi::DenseTensor& tensor) {
  phi::DenseTensor dense_out;
  phi::MetaTensor meta_input(tensor);
  phi::MetaTensor meta_out(&dense_out);
  UnchangedInferMeta(meta_input, &meta_out);

  PD_VISIT_ALL_TYPES(tensor.dtype(), "TensorContiguous", ([&] {
                       phi::ContiguousKernel<data_t, Context>(
                           dev_ctx, tensor, &dense_out);
                     }));
  return dense_out;
}

phi::DenseTensor Trans2Contiguous(const phi::DenseTensor& tensor) {
  auto& pool = phi::DeviceContextPool::Instance();

  VLOG(3) << "Trans2Contiguous...";

  if (tensor.place().GetType() == phi::AllocationType::CPU) {
    auto* dev_ctx = static_cast<phi::CPUContext*>(pool.Get(tensor.place()));
    return TensorContiguous<phi::CPUContext>(*dev_ctx, tensor);
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  } else if (tensor.place().GetType() == phi::AllocationType::GPU) {
    auto* dev_ctx = static_cast<phi::GPUContext*>(pool.Get(tensor.place()));
    return TensorContiguous<phi::GPUContext>(*dev_ctx, tensor);
#endif
#ifdef PADDLE_WITH_XPU
  } else if (tensor.place().GetType() == phi::AllocationType::XPU) {
    auto* dev_ctx = static_cast<phi::XPUContext*>(pool.Get(tensor.place()));
    return TensorContiguous<phi::XPUContext>(*dev_ctx, tensor);
#endif
#ifdef PADDLE_WITH_CUSTOM_DEVICE
  } else if (tensor.place().GetType() == phi::AllocationType::CUSTOM) {
    auto* dev_ctx = static_cast<phi::CustomContext*>(pool.Get(tensor.place()));
    phi::DenseTensor dense_out;
    phi::MetaTensor meta_input(tensor);
    phi::MetaTensor meta_out(&dense_out);
    UnchangedInferMeta(meta_input, &meta_out);
    const phi::KernelKey& kernel_key = {phi::TransToPhiBackend(tensor.place()),
                                        phi::DataLayout::ALL_LAYOUT,
                                        tensor.dtype()};
    using kernel_signature = void (*)(
        const phi::DeviceContext&, const phi::DenseTensor&, phi::DenseTensor*);
    PD_VISIT_KERNEL("contiguous",
                    kernel_key,
                    kernel_signature,
                    false,
                    *dev_ctx,
                    tensor,
                    &dense_out);
    return dense_out;
#endif
  } else {
    PADDLE_THROW(common::errors::Unimplemented(
        "Place type is not supported when casting data type."));
  }

  return tensor;
}

void CheckAndTrans2Contiguous(phi::DenseTensor* tensor) {
  if (!tensor->meta().is_contiguous()) {
    phi::DenseTensor tmp = Trans2Contiguous(*tensor);
    tensor->ShareDataWith(tmp);
  }
}

phi::DenseTensor CheckAndTrans2NewContiguousTensor(
    const phi::DenseTensor& tensor) {
  if (!tensor.meta().is_contiguous()) {
    return Trans2Contiguous(tensor);
  }
  return tensor;
}

std::vector<phi::DenseTensor> CheckAndTrans2NewContiguousTensor(
    const std::vector<phi::DenseTensor>& tensor) {
  std::vector<phi::DenseTensor> out;
  out.reserve(tensor.size());
  for (auto& t : tensor) {
    out.emplace_back(CheckAndTrans2NewContiguousTensor(t));
  }
  return out;
}

phi::DenseTensor TransformData(const phi::DenseTensor& tensor,
                               const phi::TensorArgDef& target_args_def,
                               const TransformFlag& transform_flag,
                               bool is_stride_kernel) {
  phi::DenseTensor out = tensor;
  bool trans_layout = false;
  bool trans_dtype = false;

  if (NeedTransform2Contiguous(is_stride_kernel, out.meta().is_contiguous())) {
    out = Trans2Contiguous(out);
  }

  if (NeedTransformLayout(tensor.layout(),
                          target_args_def.layout,
                          tensor.place(),
                          transform_flag) &&
      tensor.dims().size() != 1) {
    if (NeedTransform2Contiguous(false, out.meta().is_contiguous())) {
      out = Trans2Contiguous(out);
    }
    out = TransDataLayout(out, target_args_def.layout);
    trans_layout = true;
  }

  if (NeedTransformDataType(
          tensor.dtype(), target_args_def.dtype, transform_flag)) {
    if (NeedTransform2Contiguous(false, out.meta().is_contiguous())) {
      out = Trans2Contiguous(out);
    }
    out = TransDataType(out, target_args_def.dtype);
    trans_dtype = true;
  }

  if (NeedTransformPlace(
          out.place(), target_args_def.backend, transform_flag)) {
    out = TransDataPlace(out, phi::TransToPhiPlace(target_args_def.backend));
    if (!trans_layout && !trans_dtype &&
        tensor.place().GetType() == AllocationType::GPUPINNED) {
      // Sharing buffer on GPUPINNED place is a special case due to historical
      // reasons, and it should not be implemented in this way from a
      // reasonable point of view, but because the performance of the previous
      // model depends on the inplace operation here, the model performance
      // will deteriorate after reverting to non-place impl, so it needs to be
      // retained here and need to use `const_cast`
      const_cast<phi::DenseTensor&>(tensor).ShareBufferWith(out);
    }
  }
  return out;
}

std::shared_ptr<phi::DenseTensor> PrepareData(
    const Tensor& input,
    const phi::TensorArgDef& target_args_def,
    const TransformFlag& transform_flag,
    bool is_stride_kernel) {
  const auto& tensor_in = input.impl();
  if (tensor_in) {
    phi::DenseTensor& dense_tensor =
        *static_cast<phi::DenseTensor*>(tensor_in.get());
    if (!transform_flag.NeedTransform() || !dense_tensor.initialized() ||
        (!NeedTransformPlace(
             dense_tensor.place(), target_args_def.backend, transform_flag) &&
         !NeedTransformDataType(
             dense_tensor.dtype(), target_args_def.dtype, transform_flag) &&
         !NeedTransformLayout(dense_tensor.layout(),
                              target_args_def.layout,
                              dense_tensor.place(),
                              transform_flag) &&
         !NeedTransform2Contiguous(is_stride_kernel,
                                   dense_tensor.meta().is_contiguous()))) {
      if (NeedTransform2Contiguous(is_stride_kernel,
                                   dense_tensor.meta().is_contiguous()) &&
          dense_tensor.initialized()) {
        phi::DenseTensor out = dense_tensor;
        out = Trans2Contiguous(out);
        return std::make_shared<phi::DenseTensor>(std::move(out));
      }
      return std::static_pointer_cast<phi::DenseTensor>(tensor_in);
    }
    phi::DenseTensor out = TransformData(
        dense_tensor, target_args_def, transform_flag, is_stride_kernel);
    return std::make_shared<phi::DenseTensor>(std::move(out));
  }
  return nullptr;
}

paddle::optional<phi::DenseTensor> PrepareData(
    const paddle::optional<Tensor>& input,
    const phi::TensorArgDef& target_args_def,
    const TransformFlag& transform_flag,
    bool is_stride_kernel) {
  if (input) {
    return {*PrepareData(
        *input, target_args_def, transform_flag, is_stride_kernel)};
  }
  return paddle::none;
}

std::unique_ptr<std::vector<phi::DenseTensor>> PrepareData(
    const std::vector<Tensor>& inputs,
    const phi::TensorArgDef& target_args_def,
    const TransformFlag& transform_flag,
    bool is_stride_kernel) {
  auto pt_tensors = std::make_unique<std::vector<phi::DenseTensor>>();
  pt_tensors->reserve(inputs.size());

  for (const auto& input : inputs) {
    const auto& tensor_in = input.impl();
    auto dense_tensor = std::dynamic_pointer_cast<phi::DenseTensor>(tensor_in);
    if (!transform_flag.NeedTransform() || !tensor_in->initialized() ||
        (!NeedTransformPlace(
             tensor_in->place(), target_args_def.backend, transform_flag) &&
         !NeedTransformDataType(
             tensor_in->dtype(), target_args_def.dtype, transform_flag) &&
         !NeedTransformLayout(tensor_in->layout(),
                              target_args_def.layout,
                              tensor_in->place(),
                              transform_flag) &&
         !(dense_tensor &&
           NeedTransform2Contiguous(is_stride_kernel,
                                    dense_tensor->meta().is_contiguous())))) {
      if (NeedTransform2Contiguous(is_stride_kernel,
                                   dense_tensor->meta().is_contiguous()) &&
          tensor_in->initialized()) {
        phi::DenseTensor out =
            *(static_cast<phi::DenseTensor*>(tensor_in.get()));
        out = Trans2Contiguous(out);
        pt_tensors->emplace_back(out);
      } else {
        pt_tensors->emplace_back(
            *std::dynamic_pointer_cast<phi::DenseTensor>(tensor_in));
      }
    } else {
      pt_tensors->emplace_back(
          TransformData(*(static_cast<phi::DenseTensor*>(tensor_in.get())),
                        target_args_def,
                        transform_flag,
                        is_stride_kernel));
    }
  }

  return pt_tensors;
}

paddle::optional<std::vector<phi::DenseTensor>> PrepareData(
    const paddle::optional<std::vector<Tensor>>& inputs,
    const phi::TensorArgDef& target_args_def,
    const TransformFlag& transform_flag,
    bool is_stride_kernel) {
  if (inputs) {
    return {*PrepareData(
        *inputs, target_args_def, transform_flag, is_stride_kernel)};
  }
  return paddle::none;
}

std::shared_ptr<phi::SelectedRows> PrepareDataForSelectedRows(
    const Tensor& input,
    const phi::TensorArgDef& target_args_def,
    const TransformFlag& transform_flag) {
  const auto& tensor_in = input.impl();
  if (tensor_in) {
    phi::SelectedRows& selected_rows =
        *static_cast<phi::SelectedRows*>(tensor_in.get());
    if ((!transform_flag.NeedTransform() || !selected_rows.initialized() ||
         (!NeedTransformPlace(selected_rows.place(),
                              target_args_def.backend,
                              transform_flag))) &&
        !NeedTransform2Contiguous(
            false, selected_rows.value().meta().is_contiguous())) {
      if (NeedTransform2Contiguous(
              false, selected_rows.value().meta().is_contiguous()) &&
          selected_rows.initialized()) {
        auto out_new = std::make_shared<phi::SelectedRows>(
            selected_rows.rows(), selected_rows.height());
        auto dense_out = Trans2Contiguous(selected_rows.value());
        *out_new->mutable_value() = dense_out;
        return out_new;
      }
      return std::static_pointer_cast<phi::SelectedRows>(tensor_in);
    }

    if (selected_rows.place().GetType() == AllocationType::GPUPINNED) {
      if (NeedTransform2Contiguous(
              false, selected_rows.value().meta().is_contiguous())) {
        auto dense_out = Trans2Contiguous(selected_rows.value());
        selected_rows.mutable_value()->ShareDataWith(dense_out);
      }
      if (transform_flag.NeedTransform() && selected_rows.initialized() &&
          NeedTransformPlace(
              selected_rows.place(), target_args_def.backend, transform_flag)) {
        auto dense_out =
            TransDataPlace(selected_rows.value(),
                           phi::TransToPhiPlace(target_args_def.backend));
        selected_rows.mutable_value()->ShareBufferWith(dense_out);
      }
      return std::static_pointer_cast<phi::SelectedRows>(tensor_in);
    } else {
      auto out_new = std::make_shared<phi::SelectedRows>(
          selected_rows.rows(), selected_rows.height());
      if (NeedTransform2Contiguous(
              false, selected_rows.value().meta().is_contiguous())) {
        auto dense_out = Trans2Contiguous(selected_rows.value());
        *out_new->mutable_value() = dense_out;
      }
      if (transform_flag.NeedTransform() && selected_rows.initialized() &&
          NeedTransformPlace(
              selected_rows.place(), target_args_def.backend, transform_flag)) {
        auto dense_out =
            TransDataPlace(selected_rows.value(),
                           phi::TransToPhiPlace(target_args_def.backend));
        *out_new->mutable_value() = dense_out;
      }
      return out_new;
    }
  }
  PADDLE_THROW(common::errors::InvalidArgument(
      "The impl() of input tensor is nullptr, it doesn't support for "
      "selected_rows data transform now."));
}

paddle::optional<phi::SelectedRows> PrepareDataForSelectedRows(
    const paddle::optional<Tensor>& input,
    const phi::TensorArgDef& target_args_def,
    const TransformFlag& transform_flag) {
  if (input) {
    return *PrepareDataForSelectedRows(*input, target_args_def, transform_flag);
  }
  return paddle::none;
}

std::shared_ptr<phi::SparseCooTensor> PrepareDataForSparseCooTensor(
    const Tensor& input) {
  const auto& tensor_in = input.impl();
  if (tensor_in) {
    phi::SparseCooTensor& sparse_tensor =
        *static_cast<phi::SparseCooTensor*>(tensor_in.get());
    if (sparse_tensor.indices().meta().is_contiguous() &&
        sparse_tensor.values().meta().is_contiguous()) {
      return std::static_pointer_cast<phi::SparseCooTensor>(tensor_in);
    }

    if (!sparse_tensor.indices().meta().is_contiguous()) {
      *sparse_tensor.mutable_indices() =
          Trans2Contiguous(sparse_tensor.indices());
    }

    if (!sparse_tensor.values().meta().is_contiguous()) {
      *sparse_tensor.mutable_values() =
          Trans2Contiguous(sparse_tensor.values());
    }
    return std::static_pointer_cast<phi::SparseCooTensor>(tensor_in);
  }
  PADDLE_THROW(common::errors::InvalidArgument(
      "The impl() of input tensor is nullptr, it doesn't support for "
      "SparseCooTensor data transform now."));
}

paddle::optional<phi::SparseCooTensor> PrepareDataForSparseCooTensor(
    const paddle::optional<Tensor>& input) {
  if (input) {
    return *PrepareDataForSparseCooTensor(*input);
  }
  return paddle::none;
}

std::shared_ptr<phi::SparseCsrTensor> PrepareDataForSparseCsrTensor(
    const Tensor& input) {
  const auto& tensor_in = input.impl();
  if (tensor_in) {
    phi::SparseCsrTensor& sparse_tensor =
        *static_cast<phi::SparseCsrTensor*>(tensor_in.get());
    if (sparse_tensor.crows().meta().is_contiguous() &&
        sparse_tensor.cols().meta().is_contiguous() &&
        sparse_tensor.values().meta().is_contiguous()) {
      return std::static_pointer_cast<phi::SparseCsrTensor>(tensor_in);
    }

    if (!sparse_tensor.crows().meta().is_contiguous()) {
      *sparse_tensor.mutable_crows() = Trans2Contiguous(sparse_tensor.crows());
    }

    if (!sparse_tensor.cols().meta().is_contiguous()) {
      *sparse_tensor.mutable_cols() = Trans2Contiguous(sparse_tensor.cols());
    }

    if (!sparse_tensor.values().meta().is_contiguous()) {
      *sparse_tensor.mutable_values() =
          Trans2Contiguous(sparse_tensor.values());
    }
    return std::static_pointer_cast<phi::SparseCsrTensor>(tensor_in);
  }
  PADDLE_THROW(common::errors::InvalidArgument(
      "The impl() of input tensor is nullptr, it doesn't support for "
      "SparseCsrTensor data transform now."));
}

paddle::optional<phi::SparseCsrTensor> PrepareDataForSparseCsrTensor(
    const paddle::optional<Tensor>& input) {
  if (input) {
    return *PrepareDataForSparseCsrTensor(*input);
  }
  return paddle::none;
}

std::shared_ptr<phi::DenseTensor> PrepareDataForDenseTensorInSparse(
    const Tensor& input) {
  const auto& tensor_in = input.impl();
  if (tensor_in) {
    phi::DenseTensor& dense_tensor =
        *static_cast<phi::DenseTensor*>(tensor_in.get());
    if (dense_tensor.meta().is_contiguous()) {
      return std::static_pointer_cast<phi::DenseTensor>(tensor_in);
    }

    return std::make_shared<phi::DenseTensor>(Trans2Contiguous(dense_tensor));
  }
  PADDLE_THROW(common::errors::InvalidArgument(
      "The impl() of input tensor is nullptr, it doesn't support for "
      "DenseTensor data transform now."));
}

paddle::optional<phi::DenseTensor> PrepareDataForDenseTensorInSparse(
    const paddle::optional<Tensor>& input) {
  if (input) {
    return *PrepareDataForDenseTensorInSparse(*input);
  }
  return paddle::none;
}
void TransDataBackend(const phi::DenseTensor* tensor,
                      Backend target_backend,
                      phi::DenseTensor* out) {
  if (tensor && tensor->initialized()) {
    *out = TransDataPlace(*tensor, phi::TransToPhiPlace(target_backend));
  }
}

void TransDataBackend(const std::vector<phi::DenseTensor*>& tensors,
                      Backend target_backend,
                      std::vector<phi::DenseTensor*> outs) {
  size_t n = tensors.size();
  for (size_t i = 0; i < n; ++i) {
    TransDataBackend(tensors[i], target_backend, outs[i]);
  }
}

void TransDataBackend(const phi::SelectedRows* tensor,
                      Backend target_backend,
                      phi::SelectedRows* out) {
  if (tensor) {
    TransDataBackend(&tensor->value(), target_backend, out->mutable_value());
  }
}

/* ------------------ for auto parallel ----------------------- */

static bool ReshardIsNeededWithPartial(
    const phi::distributed::TensorDistAttr& in_dist_attr,
    const phi::distributed::TensorDistAttr& out_dist_attr) {
  return (in_dist_attr.process_mesh() != out_dist_attr.process_mesh() ||
          in_dist_attr.dims_mapping() != out_dist_attr.dims_mapping() ||
          in_dist_attr.partial_status() != out_dist_attr.partial_status());
}

static bool ReshardIsNeeded(
    const phi::distributed::TensorDistAttr& in_dist_attr,
    const phi::distributed::TensorDistAttr& out_dist_attr) {
  return (in_dist_attr.process_mesh() != out_dist_attr.process_mesh() ||
          in_dist_attr.dims_mapping() != out_dist_attr.dims_mapping());
}

std::string ReshardDebugInfo(
    const phi::distributed::DistTensor& src_tensor,
    const phi::distributed::TensorDistAttr& dist_attr) {
  std::stringstream sstream;
  sstream << "reshard from {Global Shape: " << src_tensor.dims()
          << ", Local Shape: " << src_tensor.local_dims()
          << ", DistAttr: " << src_tensor.dist_attr()
          << "} to {DistAttr: " << dist_attr << "}";
  return sstream.str();
}

std::shared_ptr<phi::distributed::DistTensor> ReshardApiInputToKernelInput(
    phi::DeviceContext* dev_ctx,
    const Tensor& tensor,
    const phi::distributed::ArgDistAttr& dist_attr,
    const std::string& arg_name) {
  PADDLE_ENFORCE_EQ(
      paddle::holds_alternative<phi::distributed::TensorDistAttr>(dist_attr),
      true,
      common::errors::PreconditionNotMet("Arg must be a TensorDistAttr"));

  auto tensor_in = tensor.impl();
  const auto& tensor_dist_attr = paddle::get<0>(dist_attr);
  if (tensor_in) {
    phi::distributed::DistTensor* dist_tensor =
        static_cast<phi::distributed::DistTensor*>(tensor_in.get());
    if (ReshardIsNeededWithPartial(dist_tensor->dist_attr(),
                                   tensor_dist_attr)) {
      auto argument_name = (arg_name.empty() ? "tensor" : arg_name);
      auto tensor_name = (tensor.name().empty() ? "None" : tensor.name());
      VLOG(4) << "Reshard input: " << argument_name << "(" << tensor_name
              << ") " << ReshardDebugInfo(*dist_tensor, tensor_dist_attr);
      auto* func = phi::distributed::ChooseProperReshardFunction(
          *dist_tensor, tensor_dist_attr);
      return func->Eval(dev_ctx, *dist_tensor, tensor_dist_attr);
    }
    return std::static_pointer_cast<phi::distributed::DistTensor>(tensor_in);
  }
  return nullptr;
}

std::vector<std::shared_ptr<phi::distributed::DistTensor>>
ReshardApiInputToKernelInput(phi::DeviceContext* dev_ctx,
                             const std::vector<Tensor>& tensors,
                             const phi::distributed::ArgDistAttr& dist_attrs,
                             const std::string& arg_name) {
  PADDLE_ENFORCE_EQ(
      paddle::holds_alternative<std::vector<phi::distributed::TensorDistAttr>>(
          dist_attrs),
      true,
      common::errors::PreconditionNotMet(
          "Arg must be a vector of TensorDistAttr"));
  const auto& tensor_dist_attrs = PADDLE_GET_CONST(
      std::vector<phi::distributed::TensorDistAttr>, dist_attrs);

  PADDLE_ENFORCE_EQ(tensors.size(),
                    tensor_dist_attrs.size(),
                    common::errors::InvalidArgument(
                        "Tensor's size should be equal to dist_attrs' size."));

  std::vector<std::shared_ptr<phi::distributed::DistTensor>> out;
  for (size_t i = 0; i < tensors.size(); i++) {
    auto tensor_in = tensors[i].impl();
    auto dist_attr = tensor_dist_attrs[i];
    if (tensor_in) {
      phi::distributed::DistTensor* dist_tensor =
          static_cast<phi::distributed::DistTensor*>(tensor_in.get());
      VLOG(4) << "ReshardIsNeededWithPartial"
              << ReshardIsNeededWithPartial(dist_tensor->dist_attr(),
                                            dist_attr);
      if (ReshardIsNeededWithPartial(dist_tensor->dist_attr(), dist_attr)) {
        auto argument_name =
            (arg_name.empty() ? "tensor" : arg_name) + "_" + std::to_string(i);
        auto tensor_name =
            (tensors[i].name().empty() ? "None" : tensors[i].name());
        VLOG(4) << "Reshard input: " << argument_name << "(" << tensor_name
                << ") " << ReshardDebugInfo(*dist_tensor, dist_attr);
        auto* func = phi::distributed::ChooseProperReshardFunction(*dist_tensor,
                                                                   dist_attr);
        out.push_back(func->Eval(dev_ctx, *dist_tensor, dist_attr));
      } else {
        out.push_back(
            std::static_pointer_cast<phi::distributed::DistTensor>(tensor_in));
      }
    } else {
      out.push_back(nullptr);
    }
  }
  return out;
}

paddle::optional<std::shared_ptr<phi::distributed::DistTensor>>
ReshardApiInputToKernelInput(phi::DeviceContext* dev_ctx,
                             const paddle::optional<Tensor>& tensor,
                             const phi::distributed::ArgDistAttr& dist_attr,
                             const std::string& arg_name) {
  if (tensor) {
    VLOG(6) << "Optional ApiIn to Replicated KernelIn.";
    return paddle::make_optional<std::shared_ptr<phi::distributed::DistTensor>>(
        ReshardApiInputToKernelInput(dev_ctx, *tensor, dist_attr, arg_name));
  }
  return paddle::none;
}

paddle::optional<std::vector<std::shared_ptr<phi::distributed::DistTensor>>>
ReshardApiInputToKernelInput(
    phi::DeviceContext* dev_ctx,
    const paddle::optional<std::vector<Tensor>>& tensors,
    const phi::distributed::ArgDistAttr& dist_attrs,
    const std::string& arg_name) {
  if (tensors) {
    VLOG(6) << "Optional ApiIn to Replicated KernelIn.";
    return paddle::make_optional<
        std::vector<std::shared_ptr<phi::distributed::DistTensor>>>(
        ReshardApiInputToKernelInput(dev_ctx, *tensors, dist_attrs, arg_name));
  }
  return paddle::none;
}

void SetInplaceOutputCorrectDistAttr(
    phi::DeviceContext* dev_ctx,
    Tensor& tensor,  // NOLINT
    const phi::distributed::TensorDistAttr& dist_attr,
    bool use_general_spmd_rule) {
  auto tensor_in = tensor.impl();
  if (tensor_in) {
    phi::distributed::DistTensor* dist_tensor =
        static_cast<phi::distributed::DistTensor*>(tensor_in.get());
    if (dist_tensor->initialized()) {
      if (use_general_spmd_rule) {
        if (ReshardIsNeeded(dist_tensor->dist_attr(), dist_attr)) {
          VLOG(6) << "SetInplaceOutputCorrectDistAttr Reshard inplace output"
                  << " to origin dist_attr "
                  << ReshardDebugInfo(*dist_tensor, dist_attr);
          auto* func = phi::distributed::ChooseProperReshardFunction(
              *dist_tensor, dist_attr);
          func->Eval(dev_ctx, *dist_tensor, dist_attr, dist_tensor);
          return;
        }
      }
    }
    VLOG(6) << "SetInplaceOutputCorrectDistAttr for tensor " << tensor.name()
            << ", just set its dist_attr from " << dist_tensor->dist_attr()
            << " to " << dist_attr;
    dist_tensor->unsafe_set_dist_attr(dist_attr);
  }
}

void SetInplaceOutputCorrectDistAttr(
    phi::DeviceContext* dev_ctx,
    Tensor& tensor,  // NOLINT
    const phi::distributed::ArgDistAttr& dist_attr,
    bool use_general_spmd_rule) {
  PADDLE_ENFORCE_EQ(
      paddle::holds_alternative<phi::distributed::TensorDistAttr>(dist_attr),
      true,
      common::errors::PreconditionNotMet("Arg must be a TensorDistAttr"));
  SetInplaceOutputCorrectDistAttr(
      dev_ctx, tensor, paddle::get<0>(dist_attr), use_general_spmd_rule);
}

void SetInplaceOutputCorrectDistAttr(
    phi::DeviceContext* dev_ctx,
    std::vector<Tensor>& tensors,  // NOLINT
    const std::vector<phi::distributed::TensorDistAttr>& dist_attr,
    bool use_general_spmd_rule) {
  for (size_t i = 0; i < tensors.size(); i++) {
    auto tensor_in = tensors[i].impl();
    if (tensor_in) {
      phi::distributed::DistTensor* dist_tensor =
          static_cast<phi::distributed::DistTensor*>(tensor_in.get());
      if (dist_tensor->initialized()) {
        if (use_general_spmd_rule) {
          if (ReshardIsNeededWithPartial(dist_tensor->dist_attr(),
                                         dist_attr[i])) {
            VLOG(6) << "SetInplaceOutputCorrectDistAttr Reshard inplace output"
                    << " to origin dist_attr "
                    << ReshardDebugInfo(*dist_tensor, dist_attr[i]);
            auto* func = phi::distributed::ChooseProperReshardFunction(
                *dist_tensor, dist_attr[i]);
            func->Eval(dev_ctx, *dist_tensor, dist_attr[i], dist_tensor);
            continue;
          }
        }
      }
      VLOG(6) << "SetInplaceOutputCorrectDistAttr for tensor "
              << tensors[i].name() << ", just set its dist_attr from "
              << dist_tensor->dist_attr() << " to " << dist_attr[i];
      dist_tensor->unsafe_set_dist_attr(dist_attr[i]);
    }
  }
}

void SetInplaceOutputCorrectDistAttr(
    phi::DeviceContext* dev_ctx,
    std::vector<Tensor>& tensors,  // NOLINT
    const phi::distributed::ArgDistAttr& dist_attr,
    bool use_general_spmd_rule) {
  PADDLE_ENFORCE_EQ(
      paddle::holds_alternative<std::vector<phi::distributed::TensorDistAttr>>(
          dist_attr),
      true,
      common::errors::PreconditionNotMet(
          "Arg must be a vector of TensorDistAttr"));
  SetInplaceOutputCorrectDistAttr(
      dev_ctx, tensors, paddle::get<1>(dist_attr), use_general_spmd_rule);
}

void ReshardKernelOutputToApiOutput(
    phi::DeviceContext* dev_ctx,
    const std::shared_ptr<phi::distributed::DistTensor>& src_tensor,
    Tensor* dst_tensor,
    const std::string& arg_name) {
  if (dst_tensor) {
    auto tensor_out = dst_tensor->impl();
    PADDLE_ENFORCE_NE(
        tensor_out,
        nullptr,
        common::errors::InvalidArgument("The output tensor is nullptr."));
    phi::distributed::DistTensor* dist_tensor =
        static_cast<phi::distributed::DistTensor*>(tensor_out.get());
    dist_tensor->unsafe_set_dims(src_tensor->dims());
    if (ReshardIsNeeded(src_tensor->dist_attr(), dist_tensor->dist_attr())) {
      auto argument_name = (arg_name.empty() ? "tensor" : arg_name);
      auto tensor_name =
          (dst_tensor->name().empty() ? "None" : src_tensor->name());
      VLOG(4) << "Reshard output(bwd): " << argument_name << "(" << tensor_name
              << ") "
              << ReshardDebugInfo(*src_tensor, dist_tensor->dist_attr());
      auto* func = phi::distributed::ChooseProperReshardFunction(
          *src_tensor, dist_tensor->dist_attr());
      func->Eval(dev_ctx, *src_tensor, dist_tensor->dist_attr(), dist_tensor);
    } else {
      // TODO(chenweihang): add dist attr compare and default copy rule to
      // avoid add branch here
      // shallow copy dense tensor
      *dist_tensor->unsafe_mutable_value() = src_tensor->value();
      dist_tensor->unsafe_set_dist_attr(src_tensor->dist_attr());
    }
  } else {
    VLOG(3) << "The output tensor is nullptr when call "
               "ReshardKernelOutputToApiOutput.";
  }
}

void ReshardKernelOutputToApiOutput(
    phi::DeviceContext* dev_ctx,
    const std::vector<std::shared_ptr<phi::distributed::DistTensor>>&
        src_tensors,
    const std::vector<Tensor*>& dst_tensors,
    const std::string& arg_name) {
  PADDLE_ENFORCE_EQ(
      src_tensors.size(),
      dst_tensors.size(),
      common::errors::PreconditionNotMet(
          "src_tensors.size() [%d] and dst_tensors.size() [%d] not match",
          src_tensors.size(),
          dst_tensors.size()));
  auto size = src_tensors.size();
  for (size_t i = 0; i < size; i++) {
    ReshardKernelOutputToApiOutput(
        dev_ctx, src_tensors[i], dst_tensors[i], arg_name);
  }
}

std::shared_ptr<phi::distributed::DistTensor> PrepareDataForDistTensor(
    std::shared_ptr<phi::distributed::DistTensor> input,
    const phi::TensorArgDef& target_args_def,
    const TransformFlag& transform_flag,
    bool is_stride_kernel) {
  if (input) {
    phi::distributed::DistTensor* dist_tensor = input.get();
    const phi::DenseTensor& dense_tensor = dist_tensor->value();
    if (!transform_flag.NeedTransform() || !dense_tensor.initialized() ||
        (!NeedTransformPlace(
             dense_tensor.place(), target_args_def.backend, transform_flag) &&
         !NeedTransformDataType(
             dense_tensor.dtype(), target_args_def.dtype, transform_flag) &&
         !NeedTransformLayout(dense_tensor.layout(),
                              target_args_def.layout,
                              dense_tensor.place(),
                              transform_flag) &&
         !NeedTransform2Contiguous(is_stride_kernel,
                                   dense_tensor.meta().is_contiguous()))) {
      if (NeedTransform2Contiguous(is_stride_kernel,
                                   dense_tensor.meta().is_contiguous()) &&
          dense_tensor.initialized()) {
        auto dist_out = std::make_shared<phi::distributed::DistTensor>(
            dist_tensor->dims(), dist_tensor->dist_attr());
        auto* out = dist_out->unsafe_mutable_value();
        *out = Trans2Contiguous(dense_tensor);
        return dist_out;
      }
      return input;
    }
    // TODO(chenweihang): The global meta in DistTensor is not changed,
    // but the local meta in DenseTensor maybe changed, such as layout
    // change(NCHW->NHWC), so the new DistTensor's meta maybe not unified.
    VLOG(6) << "PrepareDataForDistTensor return transformed dist tensor";
    auto dist_out = std::make_shared<phi::distributed::DistTensor>(
        dist_tensor->dims(), dist_tensor->dist_attr());
    auto* out = dist_out->unsafe_mutable_value();
    *out = TransformData(
        dense_tensor, target_args_def, transform_flag, is_stride_kernel);
    return dist_out;
  }
  return nullptr;
}

std::vector<std::shared_ptr<phi::distributed::DistTensor>>
PrepareDataForDistTensor(
    std::vector<std::shared_ptr<phi::distributed::DistTensor>> input,
    const phi::TensorArgDef& target_args_def,
    const TransformFlag& transform_flag,
    bool is_stride_kernel) {
  std::vector<std::shared_ptr<phi::distributed::DistTensor>> out;
  for (const auto& tensor_in : input) {
    if (tensor_in) {
      phi::distributed::DistTensor* dist_tensor =
          static_cast<phi::distributed::DistTensor*>(tensor_in.get());
      const phi::DenseTensor& dense_tensor = dist_tensor->value();
      if (!transform_flag.NeedTransform() || !dense_tensor.initialized() ||
          (!NeedTransformPlace(
               dense_tensor.place(), target_args_def.backend, transform_flag) &&
           !NeedTransformDataType(
               dense_tensor.dtype(), target_args_def.dtype, transform_flag) &&
           !NeedTransformLayout(dense_tensor.layout(),
                                target_args_def.layout,
                                dense_tensor.place(),
                                transform_flag) &&
           !NeedTransform2Contiguous(is_stride_kernel,
                                     dense_tensor.meta().is_contiguous()))) {
        if (NeedTransform2Contiguous(is_stride_kernel,
                                     dense_tensor.meta().is_contiguous()) &&
            dense_tensor.initialized()) {
          phi::DenseTensor trans_in_tensor = Trans2Contiguous(dense_tensor);
          out.push_back(std::make_shared<phi::distributed::DistTensor>(
              std::make_shared<phi::DenseTensor>(trans_in_tensor),
              dist_tensor->dist_attr()));
        } else {
          out.push_back(std::static_pointer_cast<phi::distributed::DistTensor>(
              tensor_in));
        }
      } else {
        phi::DenseTensor trans_in_tensor = TransformData(
            dense_tensor, target_args_def, transform_flag, is_stride_kernel);
        // TODO(GhostScreaming): The global meta in DistTensor is not changed,
        // but the local meta in DenseTensor maybe changed, such as layout
        // change(NCHW->NHWC), so the new DistTensor's meta maybe not unified.
        VLOG(6) << "PrepareDataForDistTensor return transformed dist tensor";
        out.push_back(std::make_shared<phi::distributed::DistTensor>(
            std::make_shared<phi::DenseTensor>(trans_in_tensor),
            dist_tensor->dist_attr()));
      }
    } else {
      out.push_back(nullptr);
    }
  }
  return out;
}

paddle::optional<std::shared_ptr<phi::distributed::DistTensor>>
PrepareDataForDistTensor(
    paddle::optional<std::shared_ptr<phi::distributed::DistTensor>> input,
    const phi::TensorArgDef& target_args_def,
    const TransformFlag& transform_flag,
    bool is_stride_kernel) {
  if (input) {
    VLOG(6) << "PrepareDataForDistTensor for optional return transformed dist "
               "tensor";
    return paddle::make_optional<std::shared_ptr<phi::distributed::DistTensor>>(
        PrepareDataForDistTensor(
            *input, target_args_def, transform_flag, is_stride_kernel));
  }
  return paddle::none;
}

paddle::optional<std::vector<std::shared_ptr<phi::distributed::DistTensor>>>
PrepareDataForDistTensor(
    paddle::optional<std::vector<std::shared_ptr<phi::distributed::DistTensor>>>
        input,
    const phi::TensorArgDef& target_args_def,
    const TransformFlag& transform_flag,
    bool is_stride_kernel) {
  if (input) {
    VLOG(6) << "PrepareDataForDistTensor for optional vector return "
               "transformed dist "
               "tensor";
    return paddle::make_optional<
        std::vector<std::shared_ptr<phi::distributed::DistTensor>>>(
        PrepareDataForDistTensor(
            *input, target_args_def, transform_flag, is_stride_kernel));
  }
  return paddle::none;
}

}  // namespace paddle::experimental
