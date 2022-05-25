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

#include "paddle/phi/core/custom_phi_kernel_c_api.h"

#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/infermeta/unary.h"

inline PD_DataType ToPDDataType(paddle::experimental::DataType dtype) {
#define return_result(in, ret)             \
  case paddle::experimental::DataType::in: \
    return PD_DataType::ret
  switch (dtype) {
    return_result(UNDEFINED, UNDEFINED);
    return_result(FLOAT64, FLOAT64);
    return_result(FLOAT32, FLOAT32);
    return_result(FLOAT16, FLOAT16);
    return_result(BFLOAT16, BFLOAT16);
    return_result(INT64, INT64);
    return_result(INT32, INT32);
    return_result(INT16, INT16);
    return_result(INT8, INT8);
    return_result(UINT64, UINT64);
    return_result(UINT32, UINT32);
    return_result(UINT16, UINT16);
    return_result(UINT8, UINT8);
    return_result(BOOL, BOOL);
    default: {
      PADDLE_THROW(
          phi::errors::Unavailable("DataType %d is not supported.", dtype));
    }
  }
#undef return_result
}

inline paddle::experimental::DataType ToPhiDataType(PD_DataType dtype) {
#define return_result(in, ret) \
  case PD_DataType::in:        \
    return paddle::experimental::DataType::ret
  switch (dtype) {
    return_result(UNDEFINED, UNDEFINED);
    return_result(FLOAT64, FLOAT64);
    return_result(FLOAT32, FLOAT32);
    return_result(FLOAT16, FLOAT16);
    return_result(BFLOAT16, BFLOAT16);
    return_result(INT64, INT64);
    return_result(INT32, INT32);
    return_result(INT16, INT16);
    return_result(INT8, INT8);
    return_result(UINT64, UINT64);
    return_result(UINT32, UINT32);
    return_result(UINT16, UINT16);
    return_result(UINT8, UINT8);
    return_result(BOOL, BOOL);
    default: {
      PADDLE_THROW(
          phi::errors::Unavailable("DataType %d is not supported.", dtype));
      return paddle::experimental::DataType::UNDEFINED;
    }
  }
#undef return_result
}

inline PD_DataLayout ToPDDataLayout(paddle::experimental::DataLayout layout) {
#define return_result(in, ret)               \
  case paddle::experimental::DataLayout::in: \
    return PD_DataLayout::ret
  switch (layout) {
    return_result(ANY, ANY);
    return_result(NHWC, NHWC);
    return_result(NCHW, NCHW);
    return_result(NCDHW, NCDHW);
    return_result(NDHWC, NDHWC);
    default: {
      PADDLE_THROW(
          phi::errors::Unavailable("DataLayout %d is not supported.", layout));
      return PD_DataLayout::ANY;
    }
  }
#undef return_result
}

inline paddle::experimental::DataLayout ToPhiDataLayout(PD_DataLayout layout) {
#define return_result(in, ret) \
  case PD_DataLayout::in:      \
    return paddle::experimental::DataLayout::ret
  switch (layout) {
    return_result(ANY, ANY);
    return_result(NHWC, NHWC);
    return_result(NCHW, NCHW);
    return_result(NCDHW, NCDHW);
    return_result(NDHWC, NDHWC);
    default: {
      PADDLE_THROW(
          phi::errors::Unavailable("DataLayout %d is not supported.", layout));
      return paddle::experimental::DataLayout::ANY;
    }
  }
#undef return_result
}

void PD_KernelArgsParseFn(const phi::KernelKey& default_key,
                          phi::KernelArgsDef* args_def, size_t in_nargs,
                          PD_ArgumentType* in_args_type, size_t attr_nargs,
                          PD_ArgumentType* attr_args_type, size_t out_nargs,
                          PD_ArgumentType* out_args_type) {
  auto default_tensor_layout = phi::DataLayout::NCHW;
  if (default_key.layout() != phi::DataLayout::ANY) {
    default_tensor_layout = default_key.layout();
  }
  // inputs
  for (size_t i = 0; i < in_nargs; ++i) {
    auto arg_type = in_args_type[i];
    if (arg_type == PD_ArgumentType::PD_ARG_TYPE_CONTEXT) {
    } else if (arg_type == PD_ArgumentType::PD_ARG_TYPE_TENSOR) {
      args_def->AppendInput(default_key.backend(), default_tensor_layout,
                            default_key.dtype(),
                            std::type_index(typeid(const phi::DenseTensor&)));
    } else if (arg_type == PD_ArgumentType::PD_ARG_TYPE_OPTIONAL_TENSOR) {
      args_def->AppendInput(
          default_key.backend(), default_tensor_layout, default_key.dtype(),
          std::type_index(typeid(paddle::optional<const phi::DenseTensor&>)));
    } else if (arg_type == PD_ArgumentType::PD_ARG_TYPE_LIST_TENSOR) {
      args_def->AppendInput(
          default_key.backend(), default_tensor_layout, default_key.dtype(),
          std::type_index(typeid(const std::vector<const phi::DenseTensor*>&)));
    } else {
      PADDLE_THROW(phi::errors::Unavailable(
          "PD_ArgumentType %d is not supported.", arg_type));
    }
  }
  // attributes
  for (size_t i = 0; i < attr_nargs; ++i) {
    auto arg_type = attr_args_type[i];
    if (arg_type == PD_ArgumentType::PD_ARG_TYPE_BOOL) {
      args_def->AppendAttribute(phi::AttributeType::BOOL);
    } else if (arg_type == PD_ArgumentType::PD_ARG_TYPE_FLOAT32) {
      args_def->AppendAttribute(phi::AttributeType::FLOAT32);
    } else if (arg_type == PD_ArgumentType::PD_ARG_TYPE_FLOAT64) {
      args_def->AppendAttribute(phi::AttributeType::FLOAT64);
    } else if (arg_type == PD_ArgumentType::PD_ARG_TYPE_INT32) {
      args_def->AppendAttribute(phi::AttributeType::INT32);
    } else if (arg_type == PD_ArgumentType::PD_ARG_TYPE_INT64) {
      args_def->AppendAttribute(phi::AttributeType::INT64);
    } else if (arg_type == PD_ArgumentType::PD_ARG_TYPE_STRING) {
      args_def->AppendAttribute(phi::AttributeType::STRING);
    } else if (arg_type == PD_ArgumentType::PD_ARG_TYPE_SCALAR) {
      args_def->AppendAttribute(phi::AttributeType::SCALAR);
    } else if (arg_type == PD_ArgumentType::PD_ARG_TYPE_INT_ARRAY) {
      args_def->AppendAttribute(phi::AttributeType::INT_ARRAY);
    } else if (arg_type == PD_ArgumentType::PD_ARG_TYPE_DATA_TYPE) {
      args_def->AppendAttribute(phi::AttributeType::DATA_TYPE);
    } else if (arg_type == PD_ArgumentType::PD_ARG_TYPE_DATA_LAYOUT) {
      args_def->AppendAttribute(phi::AttributeType::DATA_LAYOUT);
    } else if (arg_type == PD_ArgumentType::PD_ARG_TYPE_PLACE) {
      args_def->AppendAttribute(phi::AttributeType::PLACE);
    } else if (arg_type == PD_ArgumentType::PD_ARG_TYPE_LIST_BOOL) {
      args_def->AppendAttribute(phi::AttributeType::BOOLS);
    } else if (arg_type == PD_ArgumentType::PD_ARG_TYPE_LIST_INT32) {
      args_def->AppendAttribute(phi::AttributeType::INT32S);
    } else if (arg_type == PD_ArgumentType::PD_ARG_TYPE_LIST_INT64) {
      args_def->AppendAttribute(phi::AttributeType::INT64S);
    } else if (arg_type == PD_ArgumentType::PD_ARG_TYPE_LIST_FLOAT32) {
      args_def->AppendAttribute(phi::AttributeType::FLOAT32S);
    } else if (arg_type == PD_ArgumentType::PD_ARG_TYPE_LIST_FLOAT64) {
      args_def->AppendAttribute(phi::AttributeType::FLOAT64S);
    } else if (arg_type == PD_ArgumentType::PD_ARG_TYPE_LIST_STRING) {
      args_def->AppendAttribute(phi::AttributeType::STRINGS);
    } else if (arg_type == PD_ArgumentType::PD_ARG_TYPE_LIST_SCALAR) {
      args_def->AppendAttribute(phi::AttributeType::SCALARS);
    } else {
      PADDLE_THROW(phi::errors::Unavailable(
          "PD_ArgumentType %d is not supported.", arg_type));
    }
  }
  // outputs
  for (size_t i = 0; i < out_nargs; ++i) {
    auto arg_type = out_args_type[i];
    if (arg_type == PD_ArgumentType::PD_ARG_TYPE_TENSOR) {
      args_def->AppendOutput(default_key.backend(), default_tensor_layout,
                             default_key.dtype(),
                             std::type_index(typeid(phi::DenseTensor*)));
    } else if (arg_type == PD_ArgumentType::PD_ARG_TYPE_LIST_TENSOR) {
      args_def->AppendOutput(
          default_key.backend(), default_tensor_layout, default_key.dtype(),
          std::type_index(typeid(std::vector<phi::DenseTensor*>)));
    } else {
      PADDLE_THROW(phi::errors::Unavailable(
          "PD_ArgumentType %d is not supported.", arg_type));
    }
  }
}

void PD_RegisterPhiKernel(const char* kernel_name_cstr,
                          const char* backend_cstr, PD_DataType pd_dtype,
                          PD_DataLayout pd_layout, size_t in_nargs,
                          PD_ArgumentType* in_args_type, size_t attr_nargs,
                          PD_ArgumentType* attr_args_type, size_t out_nargs,
                          PD_ArgumentType* out_args_type,
                          void (*fn)(PD_ExecutionContext*),
                          void* variadic_kernel_fn) {
  phi::KernelArgsDefFn args_def_fn = [](const phi::KernelKey& kernel_key,
                                        phi::Kernel* kernel) {};
  phi::KernelFn kernel_fn = [fn](phi::KernelContext* ctx) {
    fn(reinterpret_cast<PD_ExecutionContext*>(ctx));
  };
  std::string kernel_name(kernel_name_cstr);

  auto dtype = ToPhiDataType(pd_dtype);
  auto layout = ToPhiDataLayout(pd_layout);
  phi::KernelKey kernel_key(paddle::experimental::StringToBackend(backend_cstr),
                            layout, dtype);

  phi::Kernel kernel(kernel_fn, variadic_kernel_fn);
  PD_KernelArgsParseFn(kernel_key, kernel.mutable_args_def(), in_nargs,
                       in_args_type, attr_nargs, attr_args_type, out_nargs,
                       out_args_type);

  args_def_fn(kernel_key, &kernel);
  phi::KernelFactory::Instance().kernels()[kernel_name][kernel_key] = kernel;
}

PD_Context* PD_OriginGetContext(PD_ExecutionContext* ctx) {
  auto kernel_context = reinterpret_cast<phi::KernelContext*>(ctx);
  auto dev_ctx_type = kernel_context->GetDeviceContext<phi::DeviceContext>()
                          .GetPlace()
                          .GetType();
  if (dev_ctx_type == phi::AllocationType::CUSTOM) {
    return reinterpret_cast<PD_Context*>(const_cast<phi::CustomContext*>(
        &kernel_context->GetDeviceContext<phi::CustomContext>()));
  } else if (dev_ctx_type == phi::AllocationType::CPU) {
    return reinterpret_cast<PD_Context*>(const_cast<phi::CPUContext*>(
        &kernel_context->GetDeviceContext<phi::CPUContext>()));
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  } else if (dev_ctx_type == phi::AllocationType::GPU) {
    return reinterpret_cast<PD_Context*>(const_cast<phi::GPUContext*>(
        &kernel_context->GetDeviceContext<phi::GPUContext>()));
#endif
#ifdef PADDLE_WITH_XPU
  } else if (dev_ctx_type == phi::AllocationType::XPU) {
    return reinterpret_cast<PD_Context*>(const_cast<phi::XPUContext*>(
        &kernel_context->GetDeviceContext<phi::XPUContext>()));
#endif
  } else {
    PADDLE_THROW(phi::errors::Unavailable(
        "Only support Custom/CPU/GPU/XPU DeviceContext"));
  }
}

PD_Tensor* PD_OriginInputAt(PD_ExecutionContext* ctx, size_t index) {
  auto kernel_context = reinterpret_cast<phi::KernelContext*>(ctx);
  const std::pair<int, int>& range = kernel_context->InputRangeAt(index);
  return reinterpret_cast<PD_Tensor*>(const_cast<phi::DenseTensor*>(
      &kernel_context->InputAt<phi::DenseTensor>(range.first)));
}

PD_Tensor* PD_OriginOptionalInputAt(PD_ExecutionContext* ctx, size_t index) {
  auto kernel_context = reinterpret_cast<phi::KernelContext*>(ctx);
  const std::pair<int, int>& range = kernel_context->InputRangeAt(index);
  auto tensor = kernel_context->OptionalInputAt<phi::DenseTensor>(range.first);
  if (tensor.is_initialized()) {
    return reinterpret_cast<PD_Tensor*>(tensor.get_ptr());
  }
  return nullptr;
}

PD_List PD_OriginMultiInputAt(PD_ExecutionContext* ctx, size_t index) {
  auto kernel_context = reinterpret_cast<phi::KernelContext*>(ctx);
  const std::pair<int, int>& range = kernel_context->InputRangeAt(index);
  auto tensor_vec = kernel_context->InputsBetween<phi::DenseTensor>(
      range.first, range.second);
  PD_List list;
  list.size = tensor_vec.size();
  list.data = tensor_vec.data();
  return list;
}

PD_Tensor* PD_OriginOutputAt(PD_ExecutionContext* ctx, size_t index) {
  auto kernel_context = reinterpret_cast<phi::KernelContext*>(ctx);
  const std::pair<int, int>& range = kernel_context->OutputRangeAt(index);
  return reinterpret_cast<PD_Tensor*>(
      kernel_context->MutableOutputAt<phi::DenseTensor>(range.first));
}

PD_List PD_OriginMultiOutputAt(PD_ExecutionContext* ctx, size_t index) {
  auto kernel_context = reinterpret_cast<phi::KernelContext*>(ctx);
  const std::pair<int, int>& range = kernel_context->OutputRangeAt(index);
  auto tensor_vec = kernel_context->MutableOutputBetween<phi::DenseTensor>(
      range.first, range.second);
  PD_List list;
  list.size = tensor_vec.size();
  list.data = tensor_vec.data();
  return list;
}

bool PD_BoolAttrAt(PD_ExecutionContext* ctx, size_t index) {
  auto kernel_context = reinterpret_cast<phi::KernelContext*>(ctx);
  return kernel_context->AttrAt<bool>(index);
}

int32_t PD_Int32AttrAt(PD_ExecutionContext* ctx, size_t index) {
  auto kernel_context = reinterpret_cast<phi::KernelContext*>(ctx);
  return kernel_context->AttrAt<int32_t>(index);
}

int64_t PD_Int64AttrAt(PD_ExecutionContext* ctx, size_t index) {
  auto kernel_context = reinterpret_cast<phi::KernelContext*>(ctx);
  return kernel_context->AttrAt<int64_t>(index);
}

float PD_FloatAttrAt(PD_ExecutionContext* ctx, size_t index) {
  auto kernel_context = reinterpret_cast<phi::KernelContext*>(ctx);
  return kernel_context->AttrAt<float>(index);
}

double PD_DoubleAttrAt(PD_ExecutionContext* ctx, size_t index) {
  auto kernel_context = reinterpret_cast<phi::KernelContext*>(ctx);
  return kernel_context->AttrAt<double>(index);
}

PD_Scalar* PD_ScalarAttrAt(PD_ExecutionContext* ctx, size_t index) {
  auto kernel_context = reinterpret_cast<phi::KernelContext*>(ctx);
  return reinterpret_cast<PD_Scalar*>(
      const_cast<phi::Scalar*>(&kernel_context->AttrAt<phi::Scalar>(index)));
}

PD_IntArray* PD_IntArrayAttrAt(PD_ExecutionContext* ctx, size_t index) {
  auto kernel_context = reinterpret_cast<phi::KernelContext*>(ctx);
  return reinterpret_cast<PD_IntArray*>(const_cast<phi::IntArray*>(
      &kernel_context->AttrAt<phi::IntArray>(index)));
}

PD_List PD_ListBoolAttrAt(PD_ExecutionContext* ctx, size_t index) {
  PD_List list;
  auto kernel_context = reinterpret_cast<phi::KernelContext*>(ctx);
  const auto& cc_list = kernel_context->AttrAt<std::vector<bool>>(index);
  list.size = cc_list.size();
  auto data = reinterpret_cast<uint8_t*>(new uint8_t[cc_list.size()]);
  for (size_t i = 0; i < cc_list.size(); ++i) {
    data[i] = static_cast<uint8_t>(cc_list[i]);
  }
  list.data = data;
  return list;
}

PD_List PD_ListInt32AttrAt(PD_ExecutionContext* ctx, size_t index) {
  PD_List list;
  auto kernel_context = reinterpret_cast<phi::KernelContext*>(ctx);
  const auto& cc_list = kernel_context->AttrAt<std::vector<int32_t>>(index);
  list.size = cc_list.size();
  list.data = const_cast<int32_t*>(cc_list.data());
  return list;
}

PD_List PD_ListInt64AttrAt(PD_ExecutionContext* ctx, size_t index) {
  PD_List list;
  auto kernel_context = reinterpret_cast<phi::KernelContext*>(ctx);
  const auto& cc_list = kernel_context->AttrAt<std::vector<int64_t>>(index);
  list.size = cc_list.size();
  list.data = const_cast<int64_t*>(cc_list.data());
  return list;
}

PD_List PD_ListFloatAttrAt(PD_ExecutionContext* ctx, size_t index) {
  PD_List list;
  auto kernel_context = reinterpret_cast<phi::KernelContext*>(ctx);
  const auto& cc_list = kernel_context->AttrAt<std::vector<float>>(index);
  list.size = cc_list.size();
  list.data = const_cast<float*>(cc_list.data());
  return list;
}

PD_List PD_ListDoubleAttrAt(PD_ExecutionContext* ctx, size_t index) {
  PD_List list;
  auto kernel_context = reinterpret_cast<phi::KernelContext*>(ctx);
  const auto& cc_list = kernel_context->AttrAt<std::vector<double>>(index);
  list.size = cc_list.size();
  list.data = const_cast<double*>(cc_list.data());
  return list;
}

char* PD_StringAttrAt(PD_ExecutionContext* ctx, size_t index) {
  auto kernel_context = reinterpret_cast<phi::KernelContext*>(ctx);
  return const_cast<char*>(kernel_context->AttrAt<std::string>(index).data());
}

PD_List PD_ListStringAttrAt(PD_ExecutionContext* ctx, size_t index) {
  PD_List list;
  auto kernel_context = reinterpret_cast<phi::KernelContext*>(ctx);
  const auto& cc_list = kernel_context->AttrAt<std::vector<std::string>>(index);
  list.size = cc_list.size();
  auto data = new char*[list.size];
  for (size_t i = 0; i < list.size; ++i) {
    data[i] = const_cast<char*>(cc_list[i].data());
  }
  list.data = reinterpret_cast<void*>(data);
  return list;
}

PD_List PD_ListScalarAttrAt(PD_ExecutionContext* ctx, size_t index) {
  PD_List list;
  auto kernel_context = reinterpret_cast<phi::KernelContext*>(ctx);
  const auto& cc_list = kernel_context->AttrAt<std::vector<phi::Scalar>>(index);
  list.size = cc_list.size();
  auto data = new PD_Scalar*[list.size];
  for (size_t i = 0; i < list.size; ++i) {
    data[i] =
        const_cast<PD_Scalar*>(reinterpret_cast<const PD_Scalar*>(&cc_list[i]));
  }
  list.data = data;
  return list;
}

PD_Place* PD_PlaceAttrAt(PD_ExecutionContext* ctx, size_t index) {
  auto kernel_context = reinterpret_cast<phi::KernelContext*>(ctx);
  return reinterpret_cast<PD_Place*>(
      const_cast<phi::Place*>(&kernel_context->AttrAt<phi::Place>(index)));
}

PD_DataType PD_DataTypeAttrAt(PD_ExecutionContext* ctx, size_t index) {
  auto kernel_context = reinterpret_cast<phi::KernelContext*>(ctx);
  return ToPDDataType(kernel_context->AttrAt<phi::DataType>(index));
}

PD_DataLayout PD_DataLayoutAttrAt(PD_ExecutionContext* ctx, size_t index) {
  auto kernel_context = reinterpret_cast<phi::KernelContext*>(ctx);
  return ToPDDataLayout(kernel_context->AttrAt<phi::DataLayout>(index));
}

PD_Stream PD_GetStream(const PD_Context* ctx, PD_Status* status) {
  if (status) {
    if (!ctx) {
      *status = C_FAILED;
      return nullptr;
    }
    *status = C_SUCCESS;
  }
  auto dev_ctx_type =
      reinterpret_cast<const phi::CustomContext*>(ctx)->GetPlace().GetType();
  if (dev_ctx_type == phi::AllocationType::CUSTOM) {
    return reinterpret_cast<PD_Stream>(
        reinterpret_cast<const phi::CustomContext*>(ctx)->stream());
  } else if (dev_ctx_type == phi::AllocationType::CPU) {
    return nullptr;
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  } else if (dev_ctx_type == phi::AllocationType::GPU) {
    return reinterpret_cast<PD_Stream>(
        reinterpret_cast<const phi::GPUContext*>(ctx)->stream());
#endif
#ifdef PADDLE_WITH_XPU
  } else if (dev_ctx_type == phi::AllocationType::XPU) {
    return nullptr;
#endif
  } else {
    PADDLE_THROW(phi::errors::Unavailable(
        "Only support Custom/CPU/GPU/XPU DeviceContext"));
  }
}

void* PD_AllocateTensor(const PD_Context* ctx, PD_Tensor* tensor, size_t size,
                        PD_DataType dtype, PD_Status* status) {
  if (status) {
    if (!tensor) {
      *status = C_FAILED;
      return nullptr;
    }
    *status = C_SUCCESS;
  }

  auto dev_ctx = reinterpret_cast<const phi::DeviceContext*>(ctx);
  auto cc_tensor = reinterpret_cast<phi::DenseTensor*>(tensor);
  auto phi_dtype = ToPhiDataType(dtype);
  if (ctx) {
    return dev_ctx->Alloc(cc_tensor, phi_dtype, size);
  } else {
    auto place = phi::CPUPlace();
    return cc_tensor->mutable_data(place, phi_dtype, size);
  }
}

PD_DataType PD_GetTensorType(const PD_Tensor* tensor, PD_Status* status) {
  if (status) {
    if (!tensor) {
      *status = C_FAILED;
      return PD_DataType::UNDEFINED;
    }
    *status = C_SUCCESS;
  }
  auto cc_tensor = reinterpret_cast<const phi::DenseTensor*>(tensor);
  return ToPDDataType(cc_tensor->dtype());
}

PD_DataLayout PD_GetTensorLayout(const PD_Tensor* tensor, PD_Status* status) {
  if (status) {
    if (!tensor) {
      *status = C_FAILED;
      return PD_DataLayout::ALL_LAYOUT;
    }
    *status = C_SUCCESS;
  }
  auto cc_tensor = reinterpret_cast<const phi::DenseTensor*>(tensor);
  return ToPDDataLayout(cc_tensor->layout());
}

int64_t PD_GetTensorByteSize(const PD_Tensor* tensor, PD_Status* status) {
  if (status) {
    if (!tensor) {
      *status = C_FAILED;
      return 0;
    }
    *status = C_SUCCESS;
  }

  auto cc_tensor = reinterpret_cast<const phi::DenseTensor*>(tensor);
  return cc_tensor->memory_size();
}

void* PD_GetTensorData(const PD_Tensor* tensor, PD_Status* status) {
  if (status) {
    if (!tensor) {
      *status = C_FAILED;
      return nullptr;
    }
    *status = C_SUCCESS;
  }
  auto cc_tensor = reinterpret_cast<const phi::DenseTensor*>(tensor);
  return const_cast<void*>(cc_tensor->data());
}

int64_t PD_GetTensorElementCount(const PD_Tensor* tensor, PD_Status* status) {
  if (status) {
    if (!tensor) {
      *status = C_FAILED;
      return 0;
    }
    *status = C_SUCCESS;
  }

  auto cc_tensor = reinterpret_cast<const phi::DenseTensor*>(tensor);
  return cc_tensor->numel();
}

int64_t PD_GetTensorNumDims(const PD_Tensor* tensor, PD_Status* status) {
  if (status) {
    if (!tensor) {
      *status = C_FAILED;
      return 0;
    }
    *status = C_SUCCESS;
  }

  auto cc_tensor = reinterpret_cast<const phi::DenseTensor*>(tensor);
  return cc_tensor->dims().size();
}

int64_t PD_GetTensorDim(const PD_Tensor* tensor, size_t index,
                        PD_Status* status) {
  auto cc_tensor = reinterpret_cast<const phi::DenseTensor*>(tensor);

  if (status) {
    if (!tensor || index >= static_cast<size_t>(cc_tensor->dims().size())) {
      *status = C_FAILED;
      return 0;
    }
    *status = C_SUCCESS;
  }

  return cc_tensor->dims()[index];
}

void PD_GetTensorLoD(const PD_Tensor* tensor, PD_List* data, PD_List* offset,
                     PD_Status* status) {
  auto cc_tensor = reinterpret_cast<const phi::DenseTensor*>(tensor);

  if (status) {
    if (!tensor || !data || !offset) {
      *status = C_FAILED;
      return;
    }
    *status = C_SUCCESS;
  }

  auto lod = cc_tensor->lod();
  offset->size = lod.size() + 1;
  auto offset_data = new size_t[offset->size];
  offset->data = offset_data;
  offset_data[0] = 0;

  size_t sz = 0;
  for (size_t i = 0; i < lod.size(); ++i) {
    offset_data[i + 1] = lod[i].size() + offset_data[i];
    sz += lod[i].size();
  }

  auto data_ptr = new size_t[sz];
  data->data = data_ptr;
  data->size = sz;
  for (size_t i = 0; i < lod.size(); ++i) {
    memcpy(data_ptr, lod[i].data(), lod[i].size() * sizeof(size_t));
    data_ptr += lod[i].size();
  }
}

bool PD_TensorIsInitialized(const PD_Tensor* tensor, PD_Status* status) {
  if (status) {
    if (!tensor) {
      *status = C_FAILED;
      return false;
    }
    *status = C_SUCCESS;
  }

  auto cc_tensor = reinterpret_cast<const phi::DenseTensor*>(tensor);
  return cc_tensor->initialized();
}

bool PD_TensorIsValid(const PD_Tensor* tensor, PD_Status* status) {
  if (status) {
    if (!tensor) {
      *status = C_FAILED;
      return false;
    }
    *status = C_SUCCESS;
  }

  auto cc_tensor = reinterpret_cast<const phi::DenseTensor*>(tensor);
  return cc_tensor->valid();
}

void* PD_GetTensorHolder(const PD_Tensor* tensor, PD_Status* status) {
  if (status) {
    if (!tensor) {
      *status = C_FAILED;
      return nullptr;
    }
    *status = C_SUCCESS;
  }

  auto cc_tensor = reinterpret_cast<const phi::DenseTensor*>(tensor);
  return cc_tensor->Holder().get();
}

void PD_SetTensorDims(PD_Tensor* tensor, int64_t ndims, const int64_t* dims,
                      PD_Status* status) {
  if (status) {
    if (!tensor) {
      *status = C_FAILED;
      return;
    }
    *status = C_SUCCESS;
  }
  auto cc_tensor = reinterpret_cast<phi::DenseTensor*>(tensor);
  std::vector<int> shape(dims, dims + ndims);
  cc_tensor->Resize(phi::make_ddim(shape));
}

void PD_SetTensorType(PD_Tensor* tensor, PD_DataType dtype, PD_Status* status) {
  if (status) {
    if (!tensor) {
      *status = C_FAILED;
      return;
    }
    *status = C_SUCCESS;
  }

  auto cc_tensor = reinterpret_cast<phi::DenseTensor*>(tensor);
  cc_tensor->set_type(ToPhiDataType(dtype));
}

void PD_SetTensorLayout(PD_Tensor* tensor, PD_DataLayout layout,
                        PD_Status* status) {
  if (status) {
    if (!tensor) {
      *status = C_FAILED;
      return;
    }
    *status = C_SUCCESS;
  }

  auto cc_tensor = reinterpret_cast<phi::DenseTensor*>(tensor);
  cc_tensor->set_layout(ToPhiDataLayout(layout));
}

void PD_ResetTensorLoD(PD_Tensor* tensor, PD_List data, PD_List offset,
                       PD_Status* status) {
  if (status) {
    if (!tensor) {
      *status = C_FAILED;
      return;
    }
    *status = C_SUCCESS;
  }

  phi::LoD lod;
  auto offset_ptr = static_cast<size_t*>(offset.data);
  auto data_ptr = static_cast<size_t*>(data.data);

  for (size_t i = 0; i < offset.size - 1; ++i) {
    lod.emplace_back(data_ptr + offset_ptr[i], data_ptr + offset_ptr[i + 1]);
  }
  auto cc_tensor = reinterpret_cast<phi::DenseTensor*>(tensor);
  cc_tensor->ResetLoD(lod);
}

PD_Tensor* PD_NewTensor() {
  return reinterpret_cast<PD_Tensor*>(new phi::DenseTensor());
}

void PD_DeleteTensor(PD_Tensor* tensor) {
  auto cc_tensor = reinterpret_cast<phi::DenseTensor*>(tensor);
  delete cc_tensor;
}

void PD_ShareDataWith(PD_Tensor* dst, const PD_Tensor* src, PD_Status* status) {
  if (status) {
    if (!dst || !src) {
      *status = C_FAILED;
      return;
    }
    *status = C_SUCCESS;
  }

  auto cc_dst_tensor = reinterpret_cast<phi::DenseTensor*>(dst);
  auto cc_src_tensor = reinterpret_cast<const phi::DenseTensor*>(src);
  cc_dst_tensor->ShareDataWith(*cc_src_tensor);
}

void PD_ShareLoDWith(PD_Tensor* dst, const PD_Tensor* src, PD_Status* status) {
  if (status) {
    if (!dst || !src) {
      *status = C_FAILED;
      return;
    }
    *status = C_SUCCESS;
  }

  auto cc_dst_tensor = reinterpret_cast<phi::DenseTensor*>(dst);
  auto cc_src_tensor = const_cast<phi::DenseTensor*>(
      reinterpret_cast<const phi::DenseTensor*>(src));

  phi::MetaTensor meta_dst(cc_dst_tensor);
  const phi::MetaTensor meta_src(cc_src_tensor);
  meta_dst.share_lod(meta_src);
}

PD_DataType PD_GetScalarType(PD_Scalar* scalar) {
  auto cc_scalar = reinterpret_cast<phi::Scalar*>(scalar);
  return ToPDDataType(cc_scalar->dtype());
}

bool PD_GetScalarBoolData(PD_Scalar* scalar) {
  auto cc_scalar = reinterpret_cast<phi::Scalar*>(scalar);
  return cc_scalar->to<bool>();
}

int8_t PD_GetScalarInt8Data(PD_Scalar* scalar) {
  auto cc_scalar = reinterpret_cast<phi::Scalar*>(scalar);
  return cc_scalar->to<int8_t>();
}

int16_t PD_GetScalarInt16Data(PD_Scalar* scalar) {
  auto cc_scalar = reinterpret_cast<phi::Scalar*>(scalar);
  return cc_scalar->to<int16_t>();
}

int32_t PD_GetScalarInt32Data(PD_Scalar* scalar) {
  auto cc_scalar = reinterpret_cast<phi::Scalar*>(scalar);
  return cc_scalar->to<int32_t>();
}

int64_t PD_GetScalarInt64Data(PD_Scalar* scalar) {
  auto cc_scalar = reinterpret_cast<phi::Scalar*>(scalar);
  return cc_scalar->to<int64_t>();
}

uint8_t PD_GetScalarUInt8Data(PD_Scalar* scalar) {
  auto cc_scalar = reinterpret_cast<phi::Scalar*>(scalar);
  return cc_scalar->to<uint8_t>();
}

uint16_t PD_GetScalarUInt16Data(PD_Scalar* scalar) {
  auto cc_scalar = reinterpret_cast<phi::Scalar*>(scalar);
  return cc_scalar->to<uint16_t>();
}

uint32_t PD_GetScalarUInt32Data(PD_Scalar* scalar) {
  auto cc_scalar = reinterpret_cast<phi::Scalar*>(scalar);
  return cc_scalar->to<uint32_t>();
}

uint64_t PD_GetScalarUInt64Data(PD_Scalar* scalar) {
  auto cc_scalar = reinterpret_cast<phi::Scalar*>(scalar);
  return cc_scalar->to<uint64_t>();
}

float PD_GetScalarFloat32Data(PD_Scalar* scalar) {
  auto cc_scalar = reinterpret_cast<phi::Scalar*>(scalar);
  return cc_scalar->to<float>();
}

double PD_GetScalarFloat64Data(PD_Scalar* scalar) {
  auto cc_scalar = reinterpret_cast<phi::Scalar*>(scalar);
  return cc_scalar->to<double>();
}

PD_List PD_GetIntArrayData(PD_IntArray* int_array) {
  auto cc_int_array = reinterpret_cast<phi::IntArray*>(int_array);
  const auto& data = cc_int_array->GetData();
  PD_List list;
  list.size = data.size();
  list.data = const_cast<int64_t*>(data.data());
  return list;
}

size_t PD_GetIntArraySize(PD_IntArray* int_array) {
  auto cc_int_array = reinterpret_cast<phi::IntArray*>(int_array);
  return cc_int_array->size();
}

void PD_DeleteList(PD_List list) {
  auto data = reinterpret_cast<void**>(list.data);
  delete[] data;
}

void PD_DeleteUInt8List(PD_List list) {
  auto data = reinterpret_cast<uint8_t*>(list.data);
  delete[] data;
}

void PD_DeleteInt64List(PD_List list) {
  auto data = reinterpret_cast<int64_t*>(list.data);
  delete[] data;
}

void PD_DeleteInt32List(PD_List list) {
  auto data = reinterpret_cast<int32_t*>(list.data);
  delete[] data;
}

void PD_DeleteFloat64List(PD_List list) {
  auto data = reinterpret_cast<double*>(list.data);
  delete[] data;
}

void PD_DeleteFloat32List(PD_List list) {
  auto data = reinterpret_cast<float*>(list.data);
  delete[] data;
}

bool PD_PlaceIsHost(PD_Place* place) {
  auto cc_place = reinterpret_cast<phi::Place*>(place);
  return cc_place->GetType() == phi::AllocationType::CPU;
}

int8_t PD_PlaceGetDeviceId(PD_Place* place) {
  auto cc_place = reinterpret_cast<phi::Place*>(place);
  return cc_place->GetDeviceId();
}
