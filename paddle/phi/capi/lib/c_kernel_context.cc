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

#include "paddle/phi/capi/include/c_kernel_context.h"

#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/capi/include/common.h"
#include "paddle/phi/capi/include/type_utils.h"
#include "paddle/phi/core/kernel_context.h"

PD_DeviceContext* PD_KernelContextGetDeviceContext(PD_KernelContext* ctx) {
  auto kernel_context = reinterpret_cast<phi::KernelContext*>(ctx);
  auto dev_ctx_type = kernel_context->GetDeviceContext<phi::DeviceContext>()
                          .GetPlace()
                          .GetType();
  if (dev_ctx_type == phi::AllocationType::CUSTOM) {
    return reinterpret_cast<PD_DeviceContext*>(const_cast<phi::CustomContext*>(
        &kernel_context->GetDeviceContext<phi::CustomContext>()));
  } else if (dev_ctx_type == phi::AllocationType::CPU) {
    return reinterpret_cast<PD_DeviceContext*>(const_cast<phi::CPUContext*>(
        &kernel_context->GetDeviceContext<phi::CPUContext>()));
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  } else if (dev_ctx_type == phi::AllocationType::GPU) {
    return reinterpret_cast<PD_DeviceContext*>(const_cast<phi::GPUContext*>(
        &kernel_context->GetDeviceContext<phi::GPUContext>()));
#endif
#ifdef PADDLE_WITH_XPU
  } else if (dev_ctx_type == phi::AllocationType::XPU) {
    return reinterpret_cast<PD_DeviceContext*>(const_cast<phi::XPUContext*>(
        &kernel_context->GetDeviceContext<phi::XPUContext>()));
#endif
  } else {
    PADDLE_THROW(phi::errors::Unavailable(
        "Only support Custom/CPU/GPU/XPU DeviceContext"));
  }
}

PD_Tensor* PD_KernelContextInputAt(PD_KernelContext* ctx, size_t index) {
  auto kernel_context = reinterpret_cast<phi::KernelContext*>(ctx);
  const std::pair<int, int>& range = kernel_context->InputRangeAt(index);
  return reinterpret_cast<PD_Tensor*>(const_cast<phi::DenseTensor*>(
      &kernel_context->InputAt<phi::DenseTensor>(range.first)));
}

PD_List PD_KernelContextMultiInputAt(PD_KernelContext* ctx, size_t index) {
  auto kernel_context = reinterpret_cast<phi::KernelContext*>(ctx);
  const std::pair<int, int>& range = kernel_context->InputRangeAt(index);
  auto tensor_vec = kernel_context->InputsBetween<phi::DenseTensor>(
      range.first, range.second);
  PD_List list;
  list.size = tensor_vec.size();
  list.data = new void*[list.size];
  for (size_t i = 0; i < list.size; ++i) {
    (reinterpret_cast<void**>(list.data))[i] =
        reinterpret_cast<void*>(const_cast<phi::DenseTensor*>(tensor_vec[i]));
  }
  return list;
}

PD_Tensor* PD_KernelContextOutputAt(PD_KernelContext* ctx, size_t index) {
  auto kernel_context = reinterpret_cast<phi::KernelContext*>(ctx);
  const std::pair<int, int>& range = kernel_context->OutputRangeAt(index);
  return reinterpret_cast<PD_Tensor*>(
      kernel_context->MutableOutputAt<phi::DenseTensor>(range.first));
}

PD_List PD_KernelContextMultiOutputAt(PD_KernelContext* ctx, size_t index) {
  auto kernel_context = reinterpret_cast<phi::KernelContext*>(ctx);
  const std::pair<int, int>& range = kernel_context->OutputRangeAt(index);
  auto tensor_vec = kernel_context->MutableOutputBetween<phi::DenseTensor>(
      range.first, range.second);
  PD_List list;
  list.size = tensor_vec.size();
  list.data = new void*[list.size];
  for (size_t i = 0; i < list.size; ++i) {
    (reinterpret_cast<void**>(list.data))[i] =
        reinterpret_cast<void*>(tensor_vec[i]);
  }
  return list;
}

bool PD_KernelContextBoolAttrAt(PD_KernelContext* ctx, size_t index) {
  auto kernel_context = reinterpret_cast<phi::KernelContext*>(ctx);
  return kernel_context->AttrAt<bool>(index);
}

int32_t PD_KernelContextInt32AttrAt(PD_KernelContext* ctx, size_t index) {
  auto kernel_context = reinterpret_cast<phi::KernelContext*>(ctx);
  return kernel_context->AttrAt<int32_t>(index);
}

int64_t PD_KernelContextInt64AttrAt(PD_KernelContext* ctx, size_t index) {
  auto kernel_context = reinterpret_cast<phi::KernelContext*>(ctx);
  return kernel_context->AttrAt<int64_t>(index);
}

float PD_KernelContextFloatAttrAt(PD_KernelContext* ctx, size_t index) {
  auto kernel_context = reinterpret_cast<phi::KernelContext*>(ctx);
  return kernel_context->AttrAt<float>(index);
}

double PD_KernelContextDoubleAttrAt(PD_KernelContext* ctx, size_t index) {
  auto kernel_context = reinterpret_cast<phi::KernelContext*>(ctx);
  return kernel_context->AttrAt<double>(index);
}

PD_Scalar* PD_KernelContextScalarAttrAt(PD_KernelContext* ctx, size_t index) {
  auto kernel_context = reinterpret_cast<phi::KernelContext*>(ctx);
  return reinterpret_cast<PD_Scalar*>(
      const_cast<phi::Scalar*>(&kernel_context->AttrAt<phi::Scalar>(index)));
}

PD_IntArray* PD_KernelContextIntArrayAttrAt(PD_KernelContext* ctx,
                                            size_t index) {
  auto kernel_context = reinterpret_cast<phi::KernelContext*>(ctx);
  return reinterpret_cast<PD_IntArray*>(const_cast<phi::IntArray*>(
      &kernel_context->AttrAt<phi::IntArray>(index)));
}

PD_List PD_KernelContextListBoolAttrAt(PD_KernelContext* ctx, size_t index) {
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

PD_List PD_KernelContextListInt32AttrAt(PD_KernelContext* ctx, size_t index) {
  PD_List list;
  auto kernel_context = reinterpret_cast<phi::KernelContext*>(ctx);
  const auto& cc_list = kernel_context->AttrAt<std::vector<int32_t>>(index);
  list.size = cc_list.size();
  list.data = const_cast<int32_t*>(cc_list.data());
  return list;
}

PD_List PD_KernelContextListInt64AttrAt(PD_KernelContext* ctx, size_t index) {
  PD_List list;
  auto kernel_context = reinterpret_cast<phi::KernelContext*>(ctx);
  const auto& cc_list = kernel_context->AttrAt<std::vector<int64_t>>(index);
  list.size = cc_list.size();
  list.data = const_cast<int64_t*>(cc_list.data());
  return list;
}

PD_List PD_KernelContextListFloatAttrAt(PD_KernelContext* ctx, size_t index) {
  PD_List list;
  auto kernel_context = reinterpret_cast<phi::KernelContext*>(ctx);
  const auto& cc_list = kernel_context->AttrAt<std::vector<float>>(index);
  list.size = cc_list.size();
  list.data = const_cast<float*>(cc_list.data());
  return list;
}

PD_List PD_KernelContextListDoubleAttrAt(PD_KernelContext* ctx, size_t index) {
  PD_List list;
  auto kernel_context = reinterpret_cast<phi::KernelContext*>(ctx);
  const auto& cc_list = kernel_context->AttrAt<std::vector<double>>(index);
  list.size = cc_list.size();
  list.data = const_cast<double*>(cc_list.data());
  return list;
}

char* PD_KernelContextStringAttrAt(PD_KernelContext* ctx, size_t index) {
  auto kernel_context = reinterpret_cast<phi::KernelContext*>(ctx);
  return const_cast<char*>(kernel_context->AttrAt<std::string>(index).data());
}

PD_List PD_KernelContextListStringAttrAt(PD_KernelContext* ctx, size_t index) {
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

PD_List PD_KernelContextListScalarAttrAt(PD_KernelContext* ctx, size_t index) {
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

PD_Place* PD_KernelContextPlaceAttrAt(PD_KernelContext* ctx, size_t index) {
  auto kernel_context = reinterpret_cast<phi::KernelContext*>(ctx);
  return reinterpret_cast<PD_Place*>(
      const_cast<phi::Place*>(&kernel_context->AttrAt<phi::Place>(index)));
}

PD_DataType PD_KernelContextDataTypeAttrAt(PD_KernelContext* ctx,
                                           size_t index) {
  auto kernel_context = reinterpret_cast<phi::KernelContext*>(ctx);
  return phi::capi::ToPDDataType(kernel_context->AttrAt<phi::DataType>(index));
}

PD_DataLayout PD_KernelContextDataLayoutAttrAt(PD_KernelContext* ctx,
                                               size_t index) {
  auto kernel_context = reinterpret_cast<phi::KernelContext*>(ctx);
  return phi::capi::ToPDDataLayout(
      kernel_context->AttrAt<phi::DataLayout>(index));
}

// eager
const char* PD_StringAttr(void* attr) {
  auto* str = reinterpret_cast<std::string*>(attr);
  return str->c_str();
}

PD_DataType PD_DatatTypeAttr(void* attr) {
  auto* dtype = reinterpret_cast<phi::DataType*>(attr);
  return phi::capi::ToPDDataType(*dtype);
}

PD_DataLayout PD_DatatLayoutAttr(void* attr) {
  auto* layout = reinterpret_cast<phi::DataLayout*>(attr);
  return phi::capi::ToPDDataLayout(*layout);
}

PD_List PD_ListInt32Attr(void* attr) {
  PD_List list;
  const auto& cc_list = *reinterpret_cast<std::vector<int32_t>*>(attr);
  list.size = cc_list.size();
  list.data = const_cast<int32_t*>(cc_list.data());
  return list;
}

PD_List PD_ListInt64Attr(void* attr) {
  PD_List list;
  const auto& cc_list = *reinterpret_cast<std::vector<int64_t>*>(attr);
  list.size = cc_list.size();
  list.data = const_cast<int64_t*>(cc_list.data());
  return list;
}

PD_List PD_ListFloatAttr(void* attr) {
  PD_List list;
  const auto& cc_list = *reinterpret_cast<std::vector<float>*>(attr);
  list.size = cc_list.size();
  list.data = const_cast<float*>(cc_list.data());
  return list;
}

PD_List PD_ListDoubleAttr(void* attr) {
  PD_List list;
  const auto& cc_list = *reinterpret_cast<std::vector<double>*>(attr);
  list.size = cc_list.size();
  list.data = const_cast<double*>(cc_list.data());
  return list;
}

PD_List PD_ListScalarAttr(void* attr) {
  PD_List list;
  const auto& cc_list = *reinterpret_cast<std::vector<phi::Scalar>*>(attr);
  list.size = cc_list.size();
  auto data = new PD_Scalar*[list.size];
  for (size_t i = 0; i < list.size; ++i) {
    data[i] =
        const_cast<PD_Scalar*>(reinterpret_cast<const PD_Scalar*>(&cc_list[i]));
  }
  list.data = data;
  return list;
}

PD_List PD_ListStringAttr(void* attr) {
  PD_List list;
  const auto& cc_list = *reinterpret_cast<std::vector<std::string>*>(attr);
  list.size = cc_list.size();
  auto data = new char*[list.size];
  for (size_t i = 0; i < list.size; ++i) {
    data[i] = const_cast<char*>(cc_list[i].data());
  }
  list.data = reinterpret_cast<void*>(data);
  return list;
}

PD_List PD_ListBoolAttr(void* attr) {
  PD_List list;
  const auto& cc_list = *reinterpret_cast<std::vector<bool>*>(attr);
  list.size = cc_list.size();
  auto data = reinterpret_cast<uint8_t*>(new uint8_t[cc_list.size()]);
  for (size_t i = 0; i < cc_list.size(); ++i) {
    data[i] = static_cast<uint8_t>(cc_list[i]);
  }
  list.data = data;
  return list;
}

PD_REGISTER_CAPI(kernel_context);
