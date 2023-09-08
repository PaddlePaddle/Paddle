// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include <cstring>
#include <string>
#include <type_traits>
#include <typeindex>
#include <typeinfo>
#include <vector>

#include "paddle/phi/core/custom_kernel.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/extended_tensor.h"
#include "paddle/phi/core/kernel_factory.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/kernel_utils.h"
#include "paddle/phi/core/macros.h"
#include "paddle/phi/core/type_defs.h"

namespace phi {

template <typename Return_, typename... Args_>
struct KernelArgsParseFunctor<Return_ (*)(Args_...)> {
  using Args = std::tuple<Args_...>;
  enum : std::size_t { Arity = sizeof...(Args_) };
  using Indices = std::make_index_sequence<Arity>;
  template <std::size_t Index>
  using Arg = typename std::tuple_element<Index, Args>::type;

  static void Parse(const KernelKey& default_key, KernelArgsDef* args_def) {
    // TODO(chenweihang): The fluid Tensor's default layout is NCHW,
    // it is not same as kernel's layout, we should fix this error on
    // fluid Tensor
    auto default_tensor_layout = phi::DataLayout::NCHW;
    if (default_key.layout() != phi::DataLayout::ANY) {
      default_tensor_layout = default_key.layout();
    }
    auto args_type = ParseArgType(Indices{});
    for (auto arg_type : args_type) {
      if (arg_type == std::type_index(typeid(const CPUContext&))
#if defined(PADDLE_WITH_DNNL)
          || arg_type == std::type_index(typeid(const OneDNNContext&))
#endif
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
          || arg_type == std::type_index(typeid(const GPUContext&))
#elif defined(PADDLE_WITH_XPU) && !defined(PADDLE_WITH_XPU_KP)
          || arg_type == std::type_index(typeid(const XPUContext&))
#elif defined(PADDLE_WITH_XPU) && defined(PADDLE_WITH_XPU_KP)
          || arg_type == std::type_index(typeid(const KPSContext&))
#endif
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
          || arg_type == std::type_index(typeid(const CustomContext&))) {
#else
      ) {
#endif
        // do nothing, skip context arg now
      } else if (arg_type == std::type_index(typeid(const DenseTensor&))) {
        args_def->AppendInput(default_key.backend(),
                              default_tensor_layout,
                              default_key.dtype(),
                              arg_type);
      } else if (arg_type == std::type_index(typeid(
                                 const paddle::optional<DenseTensor>&))) {
        args_def->AppendInput(default_key.backend(),
                              default_tensor_layout,
                              default_key.dtype(),
                              arg_type);
      } else if (arg_type ==
                 std::type_index(typeid(const paddle::optional<
                                        std::vector<const DenseTensor*>>&))) {
        args_def->AppendInput(default_key.backend(),
                              default_tensor_layout,
                              default_key.dtype(),
                              arg_type);
      } else if (arg_type == std::type_index(typeid(
                                 const paddle::optional<SelectedRows>&))) {
        args_def->AppendInput(default_key.backend(),
                              default_tensor_layout,
                              default_key.dtype(),
                              arg_type);
      } else if (arg_type == std::type_index(typeid(
                                 const std::vector<const DenseTensor*>&))) {
        args_def->AppendInput(default_key.backend(),
                              default_tensor_layout,
                              default_key.dtype(),
                              arg_type);
      } else if (arg_type ==
                 std::type_index(typeid(const phi::ExtendedTensor&))) {
        args_def->AppendInput(default_key.backend(),
                              default_tensor_layout,
                              default_key.dtype(),
                              arg_type);
      } else if (arg_type == std::type_index(typeid(
                                 const std::vector<const ExtendedTensor*>&))) {
        args_def->AppendInput(default_key.backend(),
                              default_tensor_layout,
                              default_key.dtype(),
                              arg_type);
      } else if (arg_type == std::type_index(typeid(
                                 const std::vector<const SelectedRows*>&))) {
        args_def->AppendInput(default_key.backend(),
                              default_tensor_layout,
                              default_key.dtype(),
                              arg_type);
      } else if (arg_type == std::type_index(typeid(
                                 const std::vector<const TensorBase*>&))) {
        args_def->AppendInput(default_key.backend(),
                              default_tensor_layout,
                              default_key.dtype(),
                              arg_type);
      } else if (arg_type == std::type_index(typeid(
                                 const std::vector<const TensorArray*>&))) {
        args_def->AppendInput(default_key.backend(),
                              default_tensor_layout,
                              default_key.dtype(),
                              arg_type);
      } else if (arg_type == std::type_index(typeid(const SelectedRows&))) {
        args_def->AppendInput(default_key.backend(),
                              default_tensor_layout,
                              default_key.dtype(),
                              arg_type);
      } else if (arg_type == std::type_index(typeid(const StringTensor&))) {
        args_def->AppendInput(default_key.backend(),
                              default_tensor_layout,
                              default_key.dtype(),
                              arg_type);
      } else if (arg_type == std::type_index(typeid(const SparseCooTensor&))) {
        args_def->AppendInput(default_key.backend(),
                              default_tensor_layout,
                              default_key.dtype(),
                              arg_type);
      } else if (arg_type == std::type_index(typeid(
                                 paddle::optional<const SparseCooTensor&>))) {
        args_def->AppendInput(default_key.backend(),
                              default_tensor_layout,
                              default_key.dtype(),
                              arg_type);
      } else if (arg_type == std::type_index(typeid(const SparseCsrTensor&))) {
        args_def->AppendInput(default_key.backend(),
                              default_tensor_layout,
                              default_key.dtype(),
                              arg_type);
      } else if (arg_type == std::type_index(typeid(
                                 paddle::optional<const SparseCsrTensor&>))) {
        args_def->AppendInput(default_key.backend(),
                              default_tensor_layout,
                              default_key.dtype(),
                              arg_type);
      } else if (arg_type == std::type_index(typeid(const TensorArray&))) {
        args_def->AppendInput(default_key.backend(),
                              default_tensor_layout,
                              default_key.dtype(),
                              arg_type);
      } else if (arg_type == std::type_index(typeid(DenseTensor*))) {
        args_def->AppendOutput(default_key.backend(),
                               default_tensor_layout,
                               default_key.dtype(),
                               arg_type);
      } else if (arg_type ==
                 std::type_index(typeid(std::vector<DenseTensor*>))) {
        args_def->AppendOutput(default_key.backend(),
                               default_tensor_layout,
                               default_key.dtype(),
                               arg_type);
      } else if (arg_type == std::type_index(typeid(SelectedRows*))) {
        args_def->AppendOutput(default_key.backend(),
                               default_tensor_layout,
                               default_key.dtype(),
                               arg_type);
      } else if (arg_type == std::type_index(typeid(TensorArray*))) {
        args_def->AppendOutput(default_key.backend(),
                               default_tensor_layout,
                               default_key.dtype(),
                               arg_type);
      } else if (arg_type == std::type_index(typeid(SparseCooTensor*))) {
        args_def->AppendOutput(default_key.backend(),
                               default_tensor_layout,
                               default_key.dtype(),
                               arg_type);
      } else if (arg_type == std::type_index(typeid(SparseCsrTensor*))) {
        args_def->AppendOutput(default_key.backend(),
                               default_tensor_layout,
                               default_key.dtype(),
                               arg_type);
      } else if (arg_type == std::type_index(typeid(StringTensor*))) {
        args_def->AppendOutput(default_key.backend(),
                               default_tensor_layout,
                               default_key.dtype(),
                               arg_type);
      } else if (arg_type == std::type_index(typeid(ExtendedTensor*))) {
        args_def->AppendOutput(default_key.backend(),
                               default_tensor_layout,
                               default_key.dtype(),
                               arg_type);
      } else if (arg_type == std::type_index(typeid(bool))) {
        args_def->AppendAttribute(AttributeType::BOOL);
      } else if (arg_type == std::type_index(typeid(int))) {
        args_def->AppendAttribute(AttributeType::INT32);
      } else if (arg_type == std::type_index(typeid(int64_t))) {
        args_def->AppendAttribute(AttributeType::INT64);
      } else if (arg_type == std::type_index(typeid(float))) {
        args_def->AppendAttribute(AttributeType::FLOAT32);
      } else if (arg_type == std::type_index(typeid(double))) {
        args_def->AppendAttribute(AttributeType::FLOAT64);
      } else if (arg_type == std::type_index(typeid(std::string))) {
        args_def->AppendAttribute(AttributeType::STRING);
      } else if (arg_type ==
                 std::type_index(typeid(const std::vector<bool>&))) {
        args_def->AppendAttribute(AttributeType::BOOLS);
      } else if (arg_type == std::type_index(typeid(const std::vector<int>&))) {
        args_def->AppendAttribute(AttributeType::INT32S);
      } else if (arg_type ==
                 std::type_index(typeid(const std::vector<int64_t>&))) {
        args_def->AppendAttribute(AttributeType::INT64S);
      } else if (arg_type ==
                 std::type_index(typeid(const std::vector<float>&))) {
        args_def->AppendAttribute(AttributeType::FLOAT32S);
      } else if (arg_type ==
                 std::type_index(typeid(const std::vector<double>&))) {
        args_def->AppendAttribute(AttributeType::FLOAT64S);
      } else if (arg_type ==
                 std::type_index(typeid(const std::vector<std::string>&))) {
        args_def->AppendAttribute(AttributeType::STRINGS);
      } else if (arg_type == std::type_index(typeid(const Scalar&))) {
        args_def->AppendAttribute(AttributeType::SCALAR);
      } else if (arg_type ==
                 std::type_index(typeid(const std::vector<Scalar>&))) {
        args_def->AppendAttribute(AttributeType::SCALARS);
      } else if (arg_type == std::type_index(typeid(const IntArray&))) {
        args_def->AppendAttribute(AttributeType::INT_ARRAY);
      } else if (arg_type == std::type_index(typeid(DataType))) {
        args_def->AppendAttribute(AttributeType::DATA_TYPE);
      } else if (arg_type == std::type_index(typeid(DataLayout))) {
        args_def->AppendAttribute(AttributeType::DATA_LAYOUT);
      } else if (arg_type == std::type_index(typeid(Place))) {
        args_def->AppendAttribute(AttributeType::PLACE);
      } else {
        PADDLE_THROW(phi::errors::Unavailable(
            "Unsupported kernel argument type `%s`.", arg_type.name()));
      }
    }
  }

 private:
  template <std::size_t... INDEX>
  static std::vector<std::type_index> ParseArgType(
      std::index_sequence<INDEX...>) {
    return {std::type_index(typeid(Arg<INDEX>))...};
  }
};

// NOTE: used for making a difference between inner or outer registration.

KernelRegistrar::KernelRegistrar(RegType reg_type,
                                 const char* kernel_name_cstr,
                                 const char* backend_cstr,
                                 DataLayout layout,
                                 DataType dtype,
                                 KernelArgsParseFn args_parse_fn,
                                 KernelArgsDefFn args_def_fn,
                                 KernelFn kernel_fn,
                                 void* variadic_kernel_fn) {
  ConstructKernel(reg_type,
                  kernel_name_cstr,
                  backend_cstr,
                  layout,
                  dtype,
                  args_parse_fn,
                  args_def_fn,
                  kernel_fn,
                  variadic_kernel_fn);
}

KernelRegistrar::KernelRegistrar(RegType reg_type,
                                 const char* kernel_name_cstr,
                                 const char* backend_cstr,
                                 DataLayout layout,
                                 KernelArgsParseFn args_parse_fn,
                                 KernelArgsDefFn args_def_fn,
                                 KernelFn kernel_fn,
                                 void* variadic_kernel_fn) {
  for (size_t dtype = static_cast<size_t>(DataType::BOOL);
       dtype != static_cast<size_t>(DataType::NUM_DATA_TYPES);
       dtype++) {
    // NOTE(zhiqiu): why skip these types, because fluid kernel has no kernel
    // of these type.
    if (dtype == static_cast<size_t>(DataType::UINT32) ||
        dtype == static_cast<size_t>(DataType::UINT64) ||
        dtype == static_cast<size_t>(DataType::UINT16)) {
      continue;
    }
    // NOTE(zhoushunjie): Only the strings kernels can support pstring dtype
    constexpr char strings_kernels_prefix[] = "strings_";
    if (dtype == static_cast<size_t>(DataType::PSTRING) &&
        strncmp(kernel_name_cstr,
                strings_kernels_prefix,
                strlen(strings_kernels_prefix))) {
      continue;
    }
    ConstructKernel(reg_type,
                    kernel_name_cstr,
                    backend_cstr,
                    layout,
                    static_cast<DataType>(dtype),
                    args_parse_fn,
                    args_def_fn,
                    kernel_fn,
                    variadic_kernel_fn);
  }
}

void KernelRegistrar::ConstructKernel(RegType reg_type,
                                      const char* kernel_name_cstr,
                                      const char* backend_cstr,
                                      DataLayout layout,
                                      DataType dtype,
                                      KernelArgsParseFn args_parse_fn,
                                      KernelArgsDefFn args_def_fn,
                                      KernelFn kernel_fn,
                                      void* variadic_kernel_fn) {
  std::string kernel_name(kernel_name_cstr);
  KernelKey kernel_key(
      paddle::experimental::StringToBackend(backend_cstr), layout, dtype);
  Kernel kernel(kernel_fn, variadic_kernel_fn);
  if (kernel.GetKernelRegisteredType() == KernelRegisteredType::FUNCTION) {
    args_parse_fn(kernel_key, kernel.mutable_args_def());
  }
  args_def_fn(kernel_key, &kernel);
  if (reg_type == RegType::INNER) {
    KernelFactory::Instance().kernels()[kernel_name][kernel_key] = kernel;
  } else {
    CustomKernelMap::Instance().RegisterCustomKernel(
        kernel_name, kernel_key, kernel);
  }
}

}  // namespace phi
