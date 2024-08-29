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

#include "paddle/phi/capi/include/c_kernel_registry.h"

#include "paddle/phi/capi/include/common.h"
#include "paddle/phi/capi/include/type_utils.h"
#include "paddle/phi/core/kernel_registry.h"

void PD_KernelArgsParseFn(const phi::KernelKey& default_key,
                          phi::KernelArgsDef* args_def,
                          size_t in_nargs,
                          PD_KernelArgumentType* in_args_type,
                          size_t attr_nargs,
                          PD_KernelArgumentType* attr_args_type,
                          size_t out_nargs,
                          PD_KernelArgumentType* out_args_type) {
  auto default_tensor_layout = phi::DataLayout::NCHW;
  if (default_key.layout() != phi::DataLayout::ANY) {
    default_tensor_layout = default_key.layout();
  }
  // inputs
  for (size_t i = 0; i < in_nargs; ++i) {
    auto arg_type = in_args_type[i];
    if (arg_type == PD_KernelArgumentType::PD_ARG_TYPE_CONTEXT) {
    } else if (arg_type == PD_KernelArgumentType::PD_ARG_TYPE_TENSOR) {
      args_def->AppendInput(default_key.backend(),
                            default_tensor_layout,
                            default_key.dtype(),
                            std::type_index(typeid(const phi::DenseTensor&)));
    } else if (arg_type == PD_KernelArgumentType::PD_ARG_TYPE_OPTIONAL_TENSOR) {
      args_def->AppendInput(
          default_key.backend(),
          default_tensor_layout,
          default_key.dtype(),
          std::type_index(typeid(const paddle::optional<phi::DenseTensor>&)));
    } else if (arg_type == PD_KernelArgumentType::PD_ARG_TYPE_LIST_TENSOR) {
      args_def->AppendInput(
          default_key.backend(),
          default_tensor_layout,
          default_key.dtype(),
          std::type_index(typeid(const std::vector<const phi::DenseTensor*>&)));
    } else if (arg_type ==
               PD_KernelArgumentType::PD_ARG_TYPE_OPTIONAL_MULTI_TENSOR) {
      args_def->AppendInput(
          default_key.backend(),
          default_tensor_layout,
          default_key.dtype(),
          std::type_index(typeid(
              const paddle::optional<std::vector<const phi::DenseTensor*>>&)));
    } else {
      PADDLE_THROW(common::errors::Unavailable(
          "PD_KernelArgumentType %d is not supported.", arg_type));
    }
  }
  // attributes
  for (size_t i = 0; i < attr_nargs; ++i) {
    auto arg_type = attr_args_type[i];
    if (arg_type == PD_KernelArgumentType::PD_ARG_TYPE_BOOL) {
      args_def->AppendAttribute(phi::AttributeType::BOOL);
    } else if (arg_type == PD_KernelArgumentType::PD_ARG_TYPE_FLOAT32) {
      args_def->AppendAttribute(phi::AttributeType::FLOAT32);
    } else if (arg_type == PD_KernelArgumentType::PD_ARG_TYPE_FLOAT64) {
      args_def->AppendAttribute(phi::AttributeType::FLOAT64);
    } else if (arg_type == PD_KernelArgumentType::PD_ARG_TYPE_INT32) {
      args_def->AppendAttribute(phi::AttributeType::INT32);
    } else if (arg_type == PD_KernelArgumentType::PD_ARG_TYPE_INT64) {
      args_def->AppendAttribute(phi::AttributeType::INT64);
    } else if (arg_type == PD_KernelArgumentType::PD_ARG_TYPE_STRING) {
      args_def->AppendAttribute(phi::AttributeType::STRING);
    } else if (arg_type == PD_KernelArgumentType::PD_ARG_TYPE_SCALAR) {
      args_def->AppendAttribute(phi::AttributeType::SCALAR);
    } else if (arg_type == PD_KernelArgumentType::PD_ARG_TYPE_INT_ARRAY) {
      args_def->AppendAttribute(phi::AttributeType::INT_ARRAY);
    } else if (arg_type == PD_KernelArgumentType::PD_ARG_TYPE_DATA_TYPE) {
      args_def->AppendAttribute(phi::AttributeType::DATA_TYPE);
    } else if (arg_type == PD_KernelArgumentType::PD_ARG_TYPE_DATA_LAYOUT) {
      args_def->AppendAttribute(phi::AttributeType::DATA_LAYOUT);
    } else if (arg_type == PD_KernelArgumentType::PD_ARG_TYPE_PLACE) {
      args_def->AppendAttribute(phi::AttributeType::PLACE);
    } else if (arg_type == PD_KernelArgumentType::PD_ARG_TYPE_LIST_BOOL) {
      args_def->AppendAttribute(phi::AttributeType::BOOLS);
    } else if (arg_type == PD_KernelArgumentType::PD_ARG_TYPE_LIST_INT32) {
      args_def->AppendAttribute(phi::AttributeType::INT32S);
    } else if (arg_type == PD_KernelArgumentType::PD_ARG_TYPE_LIST_INT64) {
      args_def->AppendAttribute(phi::AttributeType::INT64S);
    } else if (arg_type == PD_KernelArgumentType::PD_ARG_TYPE_LIST_FLOAT32) {
      args_def->AppendAttribute(phi::AttributeType::FLOAT32S);
    } else if (arg_type == PD_KernelArgumentType::PD_ARG_TYPE_LIST_FLOAT64) {
      args_def->AppendAttribute(phi::AttributeType::FLOAT64S);
    } else if (arg_type == PD_KernelArgumentType::PD_ARG_TYPE_LIST_STRING) {
      args_def->AppendAttribute(phi::AttributeType::STRINGS);
    } else if (arg_type == PD_KernelArgumentType::PD_ARG_TYPE_LIST_SCALAR) {
      args_def->AppendAttribute(phi::AttributeType::SCALARS);
    } else {
      PADDLE_THROW(common::errors::Unavailable(
          "PD_KernelArgumentType %d is not supported.", arg_type));
    }
  }
  // outputs
  for (size_t i = 0; i < out_nargs; ++i) {
    auto arg_type = out_args_type[i];
    if (arg_type == PD_KernelArgumentType::PD_ARG_TYPE_TENSOR) {
      args_def->AppendOutput(default_key.backend(),
                             default_tensor_layout,
                             default_key.dtype(),
                             std::type_index(typeid(phi::DenseTensor*)));
    } else if (arg_type == PD_KernelArgumentType::PD_ARG_TYPE_LIST_TENSOR) {
      args_def->AppendOutput(
          default_key.backend(),
          default_tensor_layout,
          default_key.dtype(),
          std::type_index(typeid(std::vector<phi::DenseTensor*>)));
    } else {
      PADDLE_THROW(common::errors::Unavailable(
          "PD_KernelArgumentType %d is not supported.", arg_type));
    }
  }
}

void PD_RegisterPhiKernel(const char* kernel_name_cstr,
                          const char* backend_cstr,
                          PD_DataType pd_dtype,
                          PD_DataLayout pd_layout,
                          size_t in_nargs,
                          PD_KernelArgumentType* in_args_type,
                          size_t attr_nargs,
                          PD_KernelArgumentType* attr_args_type,
                          size_t out_nargs,
                          PD_KernelArgumentType* out_args_type,
                          void (*args_def_fn)(const PD_KernelKey*, PD_Kernel*),
                          void (*fn)(PD_KernelContext*),
                          void* variadic_kernel_fn) {
  auto args_def_fn_wrapper = [args_def_fn](const phi::KernelKey& kernel_key,
                                           phi::Kernel* kernel) {
    args_def_fn(reinterpret_cast<const PD_KernelKey*>(&kernel_key),
                reinterpret_cast<PD_Kernel*>(kernel));
  };
  phi::KernelFn kernel_fn = [fn](phi::KernelContext* ctx) {
    fn(reinterpret_cast<PD_KernelContext*>(ctx));
  };
  std::string kernel_name(kernel_name_cstr);

  auto dtype = phi::capi::ToPhiDataType(pd_dtype);
  auto layout = phi::capi::ToPhiDataLayout(pd_layout);
  phi::KernelKey kernel_key(
      paddle::experimental::StringToBackend(backend_cstr), layout, dtype);

  phi::Kernel kernel(kernel_fn, variadic_kernel_fn);
  PD_KernelArgsParseFn(kernel_key,
                       kernel.mutable_args_def(),
                       in_nargs,
                       in_args_type,
                       attr_nargs,
                       attr_args_type,
                       out_nargs,
                       out_args_type);

  args_def_fn_wrapper(kernel_key, &kernel);
  phi::KernelFactory::Instance().kernels()[kernel_name][kernel_key] = kernel;
}

PD_REGISTER_CAPI(kernel_registry);
