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

#pragma once

#include <iostream>
#include <string>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>
#include <vector>

#include "paddle/pten/api/ext/dll_decl.h"
#include "paddle/pten/api/ext/exception.h"
#include "paddle/pten/api/ext/op_meta_info.h"
#include "paddle/pten/api/include/tensor.h"
#include "paddle/pten/common/scalar.h"
#include "paddle/pten/common/scalar_array.h"
#include "paddle/utils/any.h"
#include "paddle/utils/small_vector.h"

#include "paddle/pten/common/data_type.h"

/**
 * Custom Kernel Info Define.
 *
 * Used to maintain custom kernel core information before registering.
 * Pten is working on exposing headers, custom kernel depends on them, and
 * we prefer outer users following pten-kernel-function-style and registering
 * macro. So, we have to re-implement some structs or class and functions to
 * make sure users' custom kernel functions can be registered to pten.
 *
 * TODO(Aganlengzi): We should upgrade following pten.
 */

namespace paddle {
namespace framework {
class PADDLE_API OpKernelInfoHelper;
}  // namespace framework

// TODO(Aganlengzi): Simple DeviceContext temporarily for stream getting
// before pten::DeviceContext is exposed.
class DeviceContext {
 public:
  DeviceContext() { stream_ = nullptr; }
  void set_stream(void* stream) { stream_ = stream; }
  void* stream() const { return stream_; }

 private:
  void* stream_;
};
class CPUContext : public DeviceContext {};

// TODO(Aganlengzi): Use paddle::Tensor before DenseTensor is exposed
using Tensor = paddle::experimental::Tensor;
using Scalar = pten::Scalar;
using ScalarArray = pten::ScalarArray;

// Record custom kernel core information
// We can not use pten::KernelFn directly, so users' custom kernel function
// is signatured to `CustomKernelFunc', notice that the first parameter is
// fixed to `const DeviceContext&'.
using CustomKernelFunc =
    void (*)(const DeviceContext& dev_ctx,
             const std::vector<Tensor>& inputs,
             const std::vector<std::vector<Tensor>>& vec_inputs,
             const std::vector<paddle::any>& attrs,
             std::vector<Tensor*>* outputs,
             std::vector<std::vector<Tensor*>>* vec_outputs);

////////////////////// Kernel Function (PD_PT_KERNEL) ////////////////////////
#define PD_SPECIALIZE_KernelCallHelper_FOR_DEV_CONTEXT(device_ctx)           \
  template <typename... Tail>                                                \
  struct CustomComputeCallHelper<const device_ctx&, Tail...> {               \
    template <int dev_ctx_idx,                                               \
              int in_idx,                                                    \
              int vec_in_idx,                                                \
              int attr_idx,                                                  \
              int out_idx,                                                   \
              int vec_out_idx,                                               \
              typename... PreviousArgs>                                      \
    static void Compute(const DeviceContext& dev_ctx,                        \
                        const std::vector<Tensor>& inputs,                   \
                        const std::vector<std::vector<Tensor>>& vec_inputs,  \
                        const std::vector<paddle::any>& attrs,               \
                        std::vector<Tensor*>* outputs,                       \
                        std::vector<std::vector<Tensor*>>* vec_outputs,      \
                        PreviousArgs... pargs) {                             \
      static_assert(in_idx == 0,                                             \
                    "Kernel's DeviceContext should appear before Inputs.");  \
      static_assert(vec_in_idx == 0,                                         \
                    "Kernel's DeviceContext should appear before Inputs.");  \
      static_assert(                                                         \
          attr_idx == 0,                                                     \
          "Kernel's DeviceContext should appear before Attributes.");        \
      static_assert(out_idx == 0,                                            \
                    "Kernel's DeviceContext should appear before Outputs."); \
      static_assert(vec_out_idx == 0,                                        \
                    "Kernel's DeviceContext should appear before Outputs."); \
      const device_ctx& arg = static_cast<const device_ctx&>(dev_ctx);       \
      CustomComputeCallHelper<Tail...>::template Compute<dev_ctx_idx + 1,    \
                                                         in_idx,             \
                                                         vec_in_idx,         \
                                                         attr_idx,           \
                                                         out_idx,            \
                                                         vec_out_idx>(       \
          dev_ctx,                                                           \
          inputs,                                                            \
          vec_inputs,                                                        \
          attrs,                                                             \
          outputs,                                                           \
          vec_outputs,                                                       \
          pargs...,                                                          \
          arg);                                                              \
    }                                                                        \
  }

#define PD_SPECIALIZE_KernelCallHelper_FOR_INPUT(tensor_type)               \
  template <typename... Tail>                                               \
  struct CustomComputeCallHelper<const tensor_type&, Tail...> {             \
    template <int dev_ctx_idx,                                              \
              int in_idx,                                                   \
              int vec_in_idx,                                               \
              int attr_idx,                                                 \
              int out_idx,                                                  \
              int vec_out_idx,                                              \
              typename... PreviousArgs>                                     \
    static void Compute(const DeviceContext& dev_ctx,                       \
                        const std::vector<Tensor>& inputs,                  \
                        const std::vector<std::vector<Tensor>>& vec_inputs, \
                        const std::vector<paddle::any>& attrs,              \
                        std::vector<Tensor*>* outputs,                      \
                        std::vector<std::vector<Tensor*>>* vec_outputs,     \
                        PreviousArgs... pargs) {                            \
      static_assert(attr_idx == 0,                                          \
                    "Kernel's Input should appear before Attributes.");     \
      static_assert(out_idx == 0,                                           \
                    "Kernel's Input should appear before Outputs.");        \
      static_assert(vec_out_idx == 0,                                       \
                    "Kernel's Input should appear before Outputs.");        \
      const Tensor& arg = inputs[in_idx];                                   \
      CustomComputeCallHelper<Tail...>::template Compute<dev_ctx_idx,       \
                                                         in_idx + 1,        \
                                                         vec_in_idx,        \
                                                         attr_idx,          \
                                                         out_idx,           \
                                                         vec_out_idx>(      \
          dev_ctx,                                                          \
          inputs,                                                           \
          vec_inputs,                                                       \
          attrs,                                                            \
          outputs,                                                          \
          vec_outputs,                                                      \
          pargs...,                                                         \
          arg);                                                             \
    }                                                                       \
  }

#define PD_SPECIALIZE_KernelCallHelper_FOR_MULTI_INPUT(tensor_type)          \
  template <typename... Tail>                                                \
  struct CustomComputeCallHelper<const std::vector<tensor_type>&, Tail...> { \
    template <int dev_ctx_idx,                                               \
              int in_idx,                                                    \
              int vec_in_idx,                                                \
              int attr_idx,                                                  \
              int out_idx,                                                   \
              int vec_out_idx,                                               \
              typename... PreviousArgs>                                      \
    static void Compute(const DeviceContext& dev_ctx,                        \
                        const std::vector<Tensor>& inputs,                   \
                        const std::vector<std::vector<Tensor>>& vec_inputs,  \
                        const std::vector<paddle::any>& attrs,               \
                        std::vector<Tensor*>* outputs,                       \
                        std::vector<std::vector<Tensor*>>* vec_outputs,      \
                        PreviousArgs... pargs) {                             \
      static_assert(attr_idx == 0,                                           \
                    "Kernel's Input should appear before Attributes.");      \
      static_assert(out_idx == 0,                                            \
                    "Kernel's Input should appear before Outputs.");         \
      static_assert(vec_out_idx == 0,                                        \
                    "Kernel's Input should appear before Outputs.");         \
      const std::vector<Tensor>& arg = vec_inputs[vec_in_idx];               \
      CustomComputeCallHelper<Tail...>::template Compute<dev_ctx_idx,        \
                                                         in_idx,             \
                                                         vec_in_idx + 1,     \
                                                         attr_idx,           \
                                                         out_idx,            \
                                                         vec_out_idx>(       \
          dev_ctx,                                                           \
          inputs,                                                            \
          vec_inputs,                                                        \
          attrs,                                                             \
          outputs,                                                           \
          vec_outputs,                                                       \
          pargs...,                                                          \
          arg);                                                              \
    }                                                                        \
  }

#define PD_SPECIALIZE_KernelCallHelper_FOR_ATTRIBUTE(attr_type)             \
  template <typename... Tail>                                               \
  struct CustomComputeCallHelper<attr_type, Tail...> {                      \
    template <int dev_ctx_idx,                                              \
              int in_idx,                                                   \
              int vec_in_idx,                                               \
              int attr_idx,                                                 \
              int out_idx,                                                  \
              int vec_out_idx,                                              \
              typename... PreviousArgs>                                     \
    static void Compute(const DeviceContext& dev_ctx,                       \
                        const std::vector<Tensor>& inputs,                  \
                        const std::vector<std::vector<Tensor>>& vec_inputs, \
                        const std::vector<paddle::any>& attrs,              \
                        std::vector<Tensor*>* outputs,                      \
                        std::vector<std::vector<Tensor*>>* vec_outputs,     \
                        PreviousArgs... pargs) {                            \
      static_assert(out_idx == 0,                                           \
                    "Kernel's Attributes should appear before Outputs.");   \
      static_assert(vec_out_idx == 0,                                       \
                    "Kernel's Attributes should appear before Outputs.");   \
      try {                                                                 \
        attr_type arg = paddle::any_cast<attr_type>(attrs[attr_idx]);       \
        return CustomComputeCallHelper<Tail...>::template Compute<          \
            dev_ctx_idx,                                                    \
            in_idx,                                                         \
            vec_in_idx,                                                     \
            attr_idx + 1,                                                   \
            out_idx,                                                        \
            vec_out_idx>(dev_ctx,                                           \
                         inputs,                                            \
                         vec_inputs,                                        \
                         attrs,                                             \
                         outputs,                                           \
                         vec_outputs,                                       \
                         pargs...,                                          \
                         arg);                                              \
      } catch (paddle::bad_any_cast&) {                                     \
        PD_THROW(                                                           \
            "Attribute cast error in custom operator. Expected " #attr_type \
            " value.");                                                     \
      }                                                                     \
    }                                                                       \
  }

#define PD_SPECIALIZE_KernelCallHelper_FOR_OUTPUT(tensor_type)              \
  template <typename... Tail>                                               \
  struct CustomComputeCallHelper<tensor_type*, Tail...> {                   \
    template <int dev_ctx_idx,                                              \
              int in_idx,                                                   \
              int vec_in_idx,                                               \
              int attr_idx,                                                 \
              int out_idx,                                                  \
              int vec_out_idx,                                              \
              typename... PreviousArgs>                                     \
    static void Compute(const DeviceContext& dev_ctx,                       \
                        const std::vector<Tensor>& inputs,                  \
                        const std::vector<std::vector<Tensor>>& vec_inputs, \
                        const std::vector<paddle::any>& attrs,              \
                        std::vector<Tensor*>* outputs,                      \
                        std::vector<std::vector<Tensor*>>* vec_outputs,     \
                        PreviousArgs... pargs) {                            \
      tensor_type* arg = (*outputs)[out_idx];                               \
      CustomComputeCallHelper<Tail...>::template Compute<dev_ctx_idx,       \
                                                         in_idx,            \
                                                         vec_in_idx,        \
                                                         attr_idx,          \
                                                         out_idx + 1,       \
                                                         vec_out_idx>(      \
          dev_ctx,                                                          \
          inputs,                                                           \
          vec_inputs,                                                       \
          attrs,                                                            \
          outputs,                                                          \
          vec_outputs,                                                      \
          pargs...,                                                         \
          arg);                                                             \
    }                                                                       \
  }

#define PD_SPECIALIZE_KernelCallHelper_FOR_MULTI_OUTPUT(tensor_type)        \
  template <typename... Tail>                                               \
  struct CustomComputeCallHelper<std::vector<tensor_type*>, Tail...> {      \
    template <int dev_ctx_idx,                                              \
              int in_idx,                                                   \
              int vec_in_idx,                                               \
              int attr_idx,                                                 \
              int out_idx,                                                  \
              int vec_out_idx,                                              \
              typename... PreviousArgs>                                     \
    static void Compute(const DeviceContext& dev_ctx,                       \
                        const std::vector<Tensor>& inputs,                  \
                        const std::vector<std::vector<Tensor>>& vec_inputs, \
                        const std::vector<paddle::any>& attrs,              \
                        std::vector<Tensor*>* outputs,                      \
                        std::vector<std::vector<Tensor*>>* vec_outputs,     \
                        PreviousArgs... pargs) {                            \
      std::vector<tensor_type*> arg = (*vec_outputs)[vec_out_idx];          \
      CustomComputeCallHelper<Tail...>::template Compute<dev_ctx_idx,       \
                                                         in_idx,            \
                                                         vec_in_idx,        \
                                                         attr_idx,          \
                                                         out_idx,           \
                                                         vec_out_idx + 1>(  \
          dev_ctx,                                                          \
          inputs,                                                           \
          vec_inputs,                                                       \
          attrs,                                                            \
          outputs,                                                          \
          vec_outputs,                                                      \
          pargs...,                                                         \
          arg);                                                             \
    }                                                                       \
  }

template <typename T>
struct PtenTypeTag {};

template <typename F, F f>
struct CustomKernelFuncImpl;

template <typename Return,
          typename DevCtx,
          typename... Args,
          Return (*impl_fn)(DevCtx, Args...)>
struct CustomKernelFuncImpl<Return (*)(DevCtx, Args...), impl_fn> {
  static void Compute(const DeviceContext& dev_ctx,
                      const std::vector<Tensor>& inputs,
                      const std::vector<std::vector<Tensor>>& vec_inputs,
                      const std::vector<paddle::any>& attrs,
                      std::vector<Tensor*>* outputs,
                      std::vector<std::vector<Tensor*>>* vec_outputs) {
    CustomComputeCallHelper<DevCtx, Args..., PtenTypeTag<int>>::
        template Compute<0, 0, 0, 0, 0, 0>(
            dev_ctx, inputs, vec_inputs, attrs, outputs, vec_outputs);
  }

  // NOTE: Tensor in args is paddle::Tensor but not DenseTensor
  static void VariadicCompute(const DeviceContext& dev_ctx, Args... args) {
    return impl_fn(static_cast<DevCtx>(dev_ctx), std::forward<Args>(args)...);
  }

 private:
  template <typename... RemainingArgs>
  struct CustomComputeCallHelper;

  /* DeviceContext Helpers */
  PD_SPECIALIZE_KernelCallHelper_FOR_DEV_CONTEXT(CPUContext);

  /* Input Helpers */
  PD_SPECIALIZE_KernelCallHelper_FOR_INPUT(Tensor);
  PD_SPECIALIZE_KernelCallHelper_FOR_MULTI_INPUT(Tensor);

  /* Attribute Helpers */
  PD_SPECIALIZE_KernelCallHelper_FOR_ATTRIBUTE(bool);
  PD_SPECIALIZE_KernelCallHelper_FOR_ATTRIBUTE(float);
  PD_SPECIALIZE_KernelCallHelper_FOR_ATTRIBUTE(double);
  PD_SPECIALIZE_KernelCallHelper_FOR_ATTRIBUTE(int);
  PD_SPECIALIZE_KernelCallHelper_FOR_ATTRIBUTE(int64_t);
  PD_SPECIALIZE_KernelCallHelper_FOR_ATTRIBUTE(pten::dtype::float16);
  PD_SPECIALIZE_KernelCallHelper_FOR_ATTRIBUTE(DataType);
  PD_SPECIALIZE_KernelCallHelper_FOR_ATTRIBUTE(const Scalar&);
  PD_SPECIALIZE_KernelCallHelper_FOR_ATTRIBUTE(const ScalarArray&);
  PD_SPECIALIZE_KernelCallHelper_FOR_ATTRIBUTE(const std::vector<int>&);
  PD_SPECIALIZE_KernelCallHelper_FOR_ATTRIBUTE(const std::vector<int64_t>&);

  /* Output Helpers */
  PD_SPECIALIZE_KernelCallHelper_FOR_OUTPUT(Tensor);
  PD_SPECIALIZE_KernelCallHelper_FOR_MULTI_OUTPUT(Tensor);

  // End: base template
  template <typename T>
  struct CustomComputeCallHelper<PtenTypeTag<T>> {
    template <int dev_ctx_idx,
              int in_idx,
              int vec_in_idx,
              int attr_idx,
              int out_idx,
              int vec_out_idx>
    static void Compute(const DeviceContext& dev_ctx,
                        const std::vector<Tensor>& inputs,
                        const std::vector<std::vector<Tensor>>& vec_inputs,
                        const std::vector<paddle::any>& attrs,
                        std::vector<Tensor*>* outputs,
                        std::vector<std::vector<Tensor*>>* vec_outputs,
                        DevCtx device_ctx,
                        Args... args) {
      return impl_fn(device_ctx, args...);
    }
  };
};

#define PD_PT_KERNEL(...) \
  ::paddle::CustomKernelFuncImpl<decltype(&__VA_ARGS__), &__VA_ARGS__>::Compute

#define PD_PT_VARIADIC_KERNEL(...)                            \
  reinterpret_cast<void*>(                                    \
      &::paddle::CustomKernelFuncImpl<decltype(&__VA_ARGS__), \
                                      &__VA_ARGS__>::VariadicCompute)

////////////////////// Op Kernel Info depended structs //////////////////////
// TODO(Aganlengzi): Re-define TensorArgDef and AttributeArgDef temporarily.
// TensorArgDef follows pten::TensorArgDef in kernel_factory.h, the
// difference is that custom_kernel needs extra `is_vector' to ensure we can
// deal with case like vector with only one element.
struct TensorArgDef {
  pten::Backend backend;
  pten::DataLayout layout;
  pten::DataType dtype;
  bool is_vector{false};

  TensorArgDef(pten::Backend in_backend,
               pten::DataLayout in_layout,
               pten::DataType in_dtype,
               bool is_vector = false)
      : backend(in_backend),
        layout(in_layout),
        dtype(in_dtype),
        is_vector(is_vector) {}

  TensorArgDef& SetBackend(pten::Backend in_backend) {
    backend = in_backend;
    return *this;
  }

  TensorArgDef& SetDataLayout(pten::DataLayout in_layout) {
    layout = in_layout;
    return *this;
  }

  TensorArgDef& SetDataType(pten::DataType in_dtype) {
    dtype = in_dtype;
    return *this;
  }
};

// AttributeArgDef follows pten::AttributeArgDef in kernel_factory.h
struct AttributeArgDef {
  std::type_index type_index;

  explicit AttributeArgDef(std::type_index type_index)
      : type_index(type_index) {}
};

////////////////////// Op Kernel Info //////////////////////
// OpKernelInfo stores all info parsed from user kernel function, includes:
// 0. op_name and kernel key(backend, data_layout and data_type)
// 1. unified custom kernel function
// 2. variadic kernel function(use paddle::Tensor)
// 3. args info and user defined change for specific arg
class PADDLE_API OpKernelInfo {
 public:
  explicit OpKernelInfo(const std::string& op_name,
                        pten::Backend backend,
                        pten::DataLayout data_layout,
                        pten::DataType data_type)
      : op_name_(op_name),
        backend_(backend),
        layout_(data_layout),
        dtype_(data_type) {}

  // format: PD_PT_KERNEL(...)
  OpKernelInfo& SetKernelFn(CustomKernelFunc&& func);
  // format: PD_PT_VARIADIC_KERNEL(...)
  OpKernelInfo& SetVariadicKernelFn(void* func);

  // for Args parsing and storing
  void AppendInput(pten::Backend backend,
                   pten::DataLayout layout,
                   pten::DataType dtype,
                   bool is_vector = false) {
    input_defs_.emplace_back(TensorArgDef(backend, layout, dtype, is_vector));
  }

  void AppendOutput(pten::Backend backend,
                    pten::DataLayout layout,
                    pten::DataType dtype,
                    bool is_vector = false) {
    output_defs_.emplace_back(TensorArgDef(backend, layout, dtype, is_vector));
  }

  void AppendAttribute(std::type_index type_index) {
    attribute_defs_.emplace_back(AttributeArgDef(type_index));
  }

  // for Args user-def function
  TensorArgDef& InputAt(size_t idx) { return input_defs_.at(idx); }
  TensorArgDef& OutputAt(size_t idx) { return output_defs_.at(idx); }

  const pten::Backend& GetBackend() const { return backend_; }
  const pten::DataLayout& GetDataLayout() const { return layout_; }
  const pten::DataType& GetDataType() const { return dtype_; }

 private:
  friend class framework::OpKernelInfoHelper;

  // 1. op info
  std::string op_name_;

  // 2. kernel key info
  pten::Backend backend_{pten::Backend::UNDEFINED};
  pten::DataLayout layout_{pten::DataLayout::UNDEFINED};
  pten::DataType dtype_{pten::DataType::UNDEFINED};

  // 3. args info
  paddle::SmallVector<TensorArgDef> input_defs_{{}};
  paddle::SmallVector<TensorArgDef> output_defs_{{}};
  paddle::SmallVector<AttributeArgDef> attribute_defs_{{}};

  // 4. func info
  CustomKernelFunc kernel_fn_{nullptr};
  void* variadic_kernel_fn_{nullptr};
};

////////////////////// Op Kernel Args Parser //////////////////////
// Define CustomKernelArgsParseFunctor for args parsing
// We have to store parsed info into OpKernelInfo before
// mapping to pten::KernelArgsDef in pten::Kernel
template <typename Func>
struct CustomKernelArgsParseFunctor;

template <typename Return_, typename... Args_>
struct CustomKernelArgsParseFunctor<Return_ (*)(Args_...)> {
  using Args = std::tuple<Args_...>;
  enum : std::size_t { Arity = sizeof...(Args_) };
  using Indices = std::make_index_sequence<Arity>;
  template <std::size_t Index>
  using Arg = typename std::tuple_element<Index, Args>::type;

  static void Parse(OpKernelInfo* op_kernel_info) {
    const pten::Backend& backend = op_kernel_info->GetBackend();
    const pten::DataLayout& layout = op_kernel_info->GetDataLayout();
    const pten::DataType& dtype = op_kernel_info->GetDataType();

    auto default_tensor_layout = pten::DataLayout::NCHW;
    if (layout != pten::DataLayout::ANY) {
      default_tensor_layout = layout;
    }
    auto args_type = ParseArgType(Indices{});
    for (auto arg_type : args_type) {
      if (arg_type == std::type_index(typeid(const CPUContext&))) {
        // do nothing, skip context arg now
      } else if (arg_type == std::type_index(typeid(const Tensor&))) {
        op_kernel_info->AppendInput(backend, default_tensor_layout, dtype);
      } else if (arg_type ==
                 std::type_index(typeid(const std::vector<Tensor>&))) {
        op_kernel_info->AppendInput(
            backend, default_tensor_layout, dtype, true);
      } else if (arg_type == std::type_index(typeid(Tensor*))) {
        op_kernel_info->AppendOutput(backend, default_tensor_layout, dtype);
      } else if (arg_type == std::type_index(typeid(std::vector<Tensor*>))) {
        op_kernel_info->AppendOutput(
            backend, default_tensor_layout, dtype, true);
      } else {
        op_kernel_info->AppendAttribute(arg_type);
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

#define PD_PT_ARGS_PARSE(...) \
  ::paddle::CustomKernelArgsParseFunctor<decltype(&__VA_ARGS__)>::Parse

//////////////// Op Kernel Info Map /////////////////
// all user custom kernels information are stored in this map
class PADDLE_API OpKernelInfoMap {
 public:
  static OpKernelInfoMap& Instance() {
    static OpKernelInfoMap g_custom_kernel_info_map;
    return g_custom_kernel_info_map;
  }

  std::vector<OpKernelInfo>& operator[](const std::string& name);

  const std::unordered_map<std::string, std::vector<OpKernelInfo>>& GetMap()
      const;

 private:
  OpKernelInfoMap() = default;
  std::unordered_map<std::string, std::vector<OpKernelInfo>> map_;

  PD_DISABLE_COPY_AND_ASSIGN(OpKernelInfoMap);
};

//////////////// Op Kernel Info Builder /////////////////
// format: PD_PT_ARGS_PARSE(...)
using CustomKernelArgsParseFn = void (*)(OpKernelInfo* op_kernel_info);
using CustomKernelArgsDefFn = void (*)(OpKernelInfo* kernel);

class PADDLE_API OpKernelInfoBuilder {
 public:
  explicit OpKernelInfoBuilder(std::string&& op_name,
                               pten::Backend backend,
                               pten::DataLayout data_layout,
                               pten::DataType data_type);

  OpKernelInfoBuilder& SetKernelFn(CustomKernelFunc func);
  OpKernelInfoBuilder& SetVariadicKernelFn(void* func);
  OpKernelInfoBuilder& ArgsParse(CustomKernelArgsParseFn func);
  OpKernelInfoBuilder& ArgsDef(CustomKernelArgsDefFn func);

 private:
  // op name
  std::string op_name_;

  // kernel key info
  pten::Backend backend_{pten::Backend::UNDEFINED};
  pten::DataLayout layout_{pten::DataLayout::UNDEFINED};
  pten::DataType dtype_{pten::DataType::UNDEFINED};

  // ref current info ptr
  OpKernelInfo* info_ptr_;
};
/////////////////////// Custom kernel register API /////////////////////////
// For inference: compile directly with framework
// Call after PD_REGISTER_KERNEL(...)
void RegisterAllCustomKernel();

// Using this api to load compiled custom kernel's dynamic library and
// register custom kernels
void LoadCustomKernelLib(const std::string& dso_name);

//////////////// Custom kernel register macro /////////////////////
// Refer to paddle/pten/core/kernel_registry.h, we can not use
// PT_REGISTER_KERNEL directly, common macros and functions are
// not ready for custom kernel now.
// Difference: custom_kernel stores all kernels' info into global
// g_custom_kernel_info_map before loading and registering into
// pten kernel management. Only providing PD_REGISTER_KERNEL which
// supports 2 template arguments.

#define PD_BACKEND(arg__) pten::Backend::arg__
#define PD_DATALAYOUT(arg__) pten::DataLayout::arg__
#define PD_DATATYPE(arg__) pten::DataType::arg__

#define PD_NARGS(...) _PD_NARGS((__VA_ARGS__, _PD_RESQ_N()))
#define _PD_NARGS(...) _PD_ARG_N(__VA_ARGS__)
#define _PD_ARG_N_EXPAND(                                                     \
    _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, N, ...) \
  N
#define _PD_ARG_N(args) _PD_ARG_N_EXPAND args
#define _PD_RESQ_N() 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0

#define PD_CONCATENATE(arg1, arg2) PD_CONCATENATE1(arg1, arg2)
#define PD_CONCATENATE1(arg1, arg2) PD_CONCATENATE2(arg1, arg2)
#define PD_CONCATENATE2(arg1, arg2) arg1##arg2

#define PD_EXPAND(x) x

#ifdef __COUNTER__
#define PD_ID __COUNTER__
#else
#define PD_ID __LINE__
#endif

#define PD_REGISTER_KERNEL(kernel_name, backend, layout, func, cpp_dtype, ...) \
  STATIC_ASSERT_GLOBAL_NAMESPACE(                                              \
      _reg_custom_kernel_ns_check_##kernel_name##_##backend##_##layout,        \
      "PD_REGISTER_KERNEL must be called in global namespace.");               \
  _PD_REGISTER_2TA_KERNEL(                                                     \
      kernel_name, backend, layout, func, cpp_dtype, ##__VA_ARGS__)

// WIN32 is not supported
#define _PD_REGISTER_2TA_KERNEL(                                              \
    kernel_name, backend, layout, meta_kernel_fn, cpp_dtype, ...)             \
  PD_KERNEL_INSTANTIATION(meta_kernel_fn, backend, cpp_dtype, ##__VA_ARGS__); \
  static void __PD_KERNEL_args_def_FN_##kernel_name##_##backend##_##layout(   \
      ::paddle::OpKernelInfo* kernel);                                        \
  PD_KERNEL_REGISTRAR_INIT(                                                   \
      kernel_name,                                                            \
      backend,                                                                \
      layout,                                                                 \
      &__PD_KERNEL_args_def_FN_##kernel_name##_##backend##_##layout,          \
      meta_kernel_fn,                                                         \
      cpp_dtype,                                                              \
      ##__VA_ARGS__);                                                         \
  void __PD_KERNEL_args_def_FN_##kernel_name##_##backend##_##layout(          \
      ::paddle::OpKernelInfo* kernel)

#define PD_KERNEL_INSTANTIATION(meta_kernel_fn, backend, cpp_dtype, ...) \
  _PD_KERNEL_INSTANTIATION(PD_NARGS(cpp_dtype, ##__VA_ARGS__),           \
                           meta_kernel_fn,                               \
                           backend,                                      \
                           cpp_dtype,                                    \
                           ##__VA_ARGS__)

#define _PD_KERNEL_INSTANTIATION(N, meta_kernel_fn, backend, cpp_dtype, ...) \
  PD_CONCATENATE(_PD_KERNEL_INSTANTIATION_, N)                               \
  (meta_kernel_fn, backend, cpp_dtype, ##__VA_ARGS__)

#define _PD_KERNEL_INSTANTIATION_1(meta_kernel_fn, backend, cpp_dtype, ...) \
  template decltype(meta_kernel_fn<cpp_dtype, ::paddle::backend##Context>)  \
      meta_kernel_fn<cpp_dtype, ::paddle::backend##Context>
#define _PD_KERNEL_INSTANTIATION_2(meta_kernel_fn, backend, cpp_dtype, ...) \
  template decltype(meta_kernel_fn<cpp_dtype, ::paddle::backend##Context>)  \
      meta_kernel_fn<cpp_dtype, ::paddle::backend##Context>;                \
  PD_EXPAND(_PD_KERNEL_INSTANTIATION_1(meta_kernel_fn, backend, ##__VA_ARGS__))
#define _PD_KERNEL_INSTANTIATION_3(meta_kernel_fn, backend, cpp_dtype, ...) \
  template decltype(meta_kernel_fn<cpp_dtype, ::paddle::backend##Context>)  \
      meta_kernel_fn<cpp_dtype, ::paddle::backend##Context>;                \
  PD_EXPAND(_PD_KERNEL_INSTANTIATION_2(meta_kernel_fn, backend, ##__VA_ARGS__))
#define _PD_KERNEL_INSTANTIATION_4(meta_kernel_fn, backend, cpp_dtype, ...) \
  template decltype(meta_kernel_fn<cpp_dtype, ::paddle::backend##Context>)  \
      meta_kernel_fn<cpp_dtype, ::paddle::backend##Context>;                \
  PD_EXPAND(_PD_KERNEL_INSTANTIATION_3(meta_kernel_fn, backend, ##__VA_ARGS__))
#define _PD_KERNEL_INSTANTIATION_5(meta_kernel_fn, backend, cpp_dtype, ...) \
  template decltype(meta_kernel_fn<cpp_dtype, ::paddle::backend##Context>)  \
      meta_kernel_fn<cpp_dtype, ::paddle::backend##Context>;                \
  PD_EXPAND(_PD_KERNEL_INSTANTIATION_4(meta_kernel_fn, backend, ##__VA_ARGS__))
#define _PD_KERNEL_INSTANTIATION_6(meta_kernel_fn, backend, cpp_dtype, ...) \
  template decltype(meta_kernel_fn<cpp_dtype, ::paddle::backend##Context>)  \
      meta_kernel_fn<cpp_dtype, ::paddle::backend##Context>;                \
  PD_EXPAND(_PD_KERNEL_INSTANTIATION_5(meta_kernel_fn, backend, ##__VA_ARGS__))
#define _PD_KERNEL_INSTANTIATION_7(meta_kernel_fn, backend, cpp_dtype, ...) \
  template decltype(meta_kernel_fn<cpp_dtype, ::paddle::backend##Context>)  \
      meta_kernel_fn<cpp_dtype, ::paddle::backend##Context>;                \
  PD_EXPAND(_PD_KERNEL_INSTANTIATION_6(meta_kernel_fn, backend, ##__VA_ARGS__))
#define _PD_KERNEL_INSTANTIATION_8(meta_kernel_fn, backend, cpp_dtype, ...) \
  template decltype(meta_kernel_fn<cpp_dtype, ::paddle::backend##Context>)  \
      meta_kernel_fn<cpp_dtype, ::paddle::backend##Context>;                \
  PD_EXPAND(_PD_KERNEL_INSTANTIATION_7(meta_kernel_fn, backend, ##__VA_ARGS__))
#define _PD_KERNEL_INSTANTIATION_9(meta_kernel_fn, backend, cpp_dtype, ...) \
  template decltype(meta_kernel_fn<cpp_dtype, ::paddle::backend##Context>)  \
      meta_kernel_fn<cpp_dtype, ::paddle::backend##Context>;                \
  PD_EXPAND(_PD_KERNEL_INSTANTIATION_8(meta_kernel_fn, backend, ##__VA_ARGS__))
#define _PD_KERNEL_INSTANTIATION_10(meta_kernel_fn, backend, cpp_dtype, ...) \
  template decltype(meta_kernel_fn<cpp_dtype, ::paddle::backend##Context>)   \
      meta_kernel_fn<cpp_dtype, ::paddle::backend##Context>;                 \
  PD_EXPAND(_PD_KERNEL_INSTANTIATION_9(meta_kernel_fn, backend, ##__VA_ARGS__))
#define _PD_KERNEL_INSTANTIATION_11(meta_kernel_fn, backend, cpp_dtype, ...) \
  template decltype(meta_kernel_fn<cpp_dtype, ::paddle::backend##Context>)   \
      meta_kernel_fn<cpp_dtype, ::paddle::backend##Context>;                 \
  PD_EXPAND(_PD_KERNEL_INSTANTIATION_10(meta_kernel_fn, backend, ##__VA_ARGS__))
#define _PD_KERNEL_INSTANTIATION_12(meta_kernel_fn, backend, cpp_dtype, ...) \
  template decltype(meta_kernel_fn<cpp_dtype, ::paddle::backend##Context>)   \
      meta_kernel_fn<cpp_dtype, ::paddle::backend##Context>;                 \
  PD_EXPAND(_PD_KERNEL_INSTANTIATION_11(meta_kernel_fn, backend, ##__VA_ARGS__))
#define _PD_KERNEL_INSTANTIATION_13(meta_kernel_fn, backend, cpp_dtype, ...) \
  template decltype(meta_kernel_fn<cpp_dtype, ::paddle::backend##Context>)   \
      meta_kernel_fn<cpp_dtype, ::paddle::backend##Context>;                 \
  PD_EXPAND(_PD_KERNEL_INSTANTIATION_12(meta_kernel_fn, backend, ##__VA_ARGS__))
#define _PD_KERNEL_INSTANTIATION_14(meta_kernel_fn, backend, cpp_dtype, ...) \
  template decltype(meta_kernel_fn<cpp_dtype, ::paddle::backend##Context>)   \
      meta_kernel_fn<cpp_dtype, ::paddle::backend##Context>;                 \
  PD_EXPAND(_PD_KERNEL_INSTANTIATION_13(meta_kernel_fn, backend, ##__VA_ARGS__))
#define _PD_KERNEL_INSTANTIATION_15(meta_kernel_fn, backend, cpp_dtype, ...) \
  template decltype(meta_kernel_fn<cpp_dtype, ::paddle::backend##Context>)   \
      meta_kernel_fn<cpp_dtype, ::paddle::backend##Context>;                 \
  PD_EXPAND(_PD_KERNEL_INSTANTIATION_14(meta_kernel_fn, backend, ##__VA_ARGS__))

#define PD_KERNEL_REGISTRAR_INIT(                                              \
    kernel_name, backend, layout, args_def_fn, meta_kernel_fn, cpp_dtype, ...) \
  _PD_KERNEL_REGISTRAR_INIT(PD_NARGS(cpp_dtype, ##__VA_ARGS__),                \
                            kernel_name,                                       \
                            backend,                                           \
                            layout,                                            \
                            args_def_fn,                                       \
                            meta_kernel_fn,                                    \
                            cpp_dtype,                                         \
                            ##__VA_ARGS__)

// clang-format off

/* The =pre-commit always treats this macro into the wrong format,
  and multi-line macros cannot be skipped with NOLINT.*/
#define _PD_KERNEL_REGISTRAR_INIT(N,              \
                                  kernel_name,    \
                                  backend,        \
                                  layout,         \
                                  args_def_fn,    \
                                  meta_kernel_fn, \
                                  cpp_dtype,      \
                                  ...)            \
  PD_CONCATENATE(_PD_KERNEL_REGISTRAR_INIT_, N) ( \
    kernel_name,                                  \
    backend,                                      \
    layout,                                       \
    PD_ID,                                        \
    args_def_fn,                                  \
    meta_kernel_fn,                               \
    cpp_dtype,                                    \
    ##__VA_ARGS__)

// clang-format on

#define _PD_KERNEL_REGISTRAR_INIT_1(kernel_name,                        \
                                    backend,                            \
                                    layout,                             \
                                    registrar_id,                       \
                                    args_def_fn,                        \
                                    meta_kernel_fn,                     \
                                    cpp_dtype,                          \
                                    ...)                                \
  static ::paddle::OpKernelInfoBuilder PD_CONCATENATE(                  \
      custom_kernel_info_##kernel_name##_##backend##_##layout##_,       \
      registrar_id) =                                                   \
      ::paddle::OpKernelInfoBuilder(                                    \
          #kernel_name,                                                 \
          PD_BACKEND(backend),                                          \
          PD_DATALAYOUT(layout),                                        \
          ::paddle::experimental::CppTypeToDataType<cpp_dtype>::Type()) \
          .SetKernelFn(PD_PT_KERNEL(                                    \
              meta_kernel_fn<cpp_dtype, ::paddle::backend##Context>))   \
          .SetVariadicKernelFn(PD_PT_VARIADIC_KERNEL(                   \
              meta_kernel_fn<cpp_dtype, ::paddle::backend##Context>))   \
          .ArgsParse(PD_PT_ARGS_PARSE(                                  \
              meta_kernel_fn<cpp_dtype, ::paddle::backend##Context>))   \
          .ArgsDef(args_def_fn);

#define _PD_KERNEL_REGISTRAR_INIT_2(kernel_name,                        \
                                    backend,                            \
                                    layout,                             \
                                    registrar_id,                       \
                                    args_def_fn,                        \
                                    meta_kernel_fn,                     \
                                    cpp_dtype,                          \
                                    ...)                                \
  static ::paddle::OpKernelInfoBuilder PD_CONCATENATE(                  \
      custom_kernel_info_##kernel_name##_##backend##_##layout##_,       \
      registrar_id) =                                                   \
      ::paddle::OpKernelInfoBuilder(                                    \
          #kernel_name,                                                 \
          PD_BACKEND(backend),                                          \
          PD_DATALAYOUT(layout),                                        \
          ::paddle::experimental::CppTypeToDataType<cpp_dtype>::Type()) \
          .SetKernelFn(PD_PT_KERNEL(                                    \
              meta_kernel_fn<cpp_dtype, ::paddle::backend##Context>))   \
          .SetVariadicKernelFn(PD_PT_VARIADIC_KERNEL(                   \
              meta_kernel_fn<cpp_dtype, ::paddle::backend##Context>))   \
          .ArgsParse(PD_PT_ARGS_PARSE(                                  \
              meta_kernel_fn<cpp_dtype, ::paddle::backend##Context>))   \
          .ArgsDef(args_def_fn);                                        \
  PD_EXPAND(_PD_KERNEL_REGISTRAR_INIT_1(kernel_name,                    \
                                        backend,                        \
                                        layout,                         \
                                        PD_ID,                          \
                                        args_def_fn,                    \
                                        meta_kernel_fn,                 \
                                        ##__VA_ARGS__))

#define _PD_KERNEL_REGISTRAR_INIT_3(kernel_name,                        \
                                    backend,                            \
                                    layout,                             \
                                    registrar_id,                       \
                                    args_def_fn,                        \
                                    meta_kernel_fn,                     \
                                    cpp_dtype,                          \
                                    ...)                                \
  static ::paddle::OpKernelInfoBuilder PD_CONCATENATE(                  \
      custom_kernel_info_##kernel_name##_##backend##_##layout##_,       \
      registrar_id) =                                                   \
      ::paddle::OpKernelInfoBuilder(                                    \
          #kernel_name,                                                 \
          PD_BACKEND(backend),                                          \
          PD_DATALAYOUT(layout),                                        \
          ::paddle::experimental::CppTypeToDataType<cpp_dtype>::Type()) \
          .SetKernelFn(PD_PT_KERNEL(                                    \
              meta_kernel_fn<cpp_dtype, ::paddle::backend##Context>))   \
          .SetVariadicKernelFn(PD_PT_VARIADIC_KERNEL(                   \
              meta_kernel_fn<cpp_dtype, ::paddle::backend##Context>))   \
          .ArgsParse(PD_PT_ARGS_PARSE(                                  \
              meta_kernel_fn<cpp_dtype, ::paddle::backend##Context>))   \
          .ArgsDef(args_def_fn);                                        \
  PD_EXPAND(_PD_KERNEL_REGISTRAR_INIT_2(kernel_name,                    \
                                        backend,                        \
                                        layout,                         \
                                        PD_ID,                          \
                                        args_def_fn,                    \
                                        meta_kernel_fn,                 \
                                        ##__VA_ARGS__))

#define _PD_KERNEL_REGISTRAR_INIT_4(kernel_name,                        \
                                    backend,                            \
                                    layout,                             \
                                    registrar_id,                       \
                                    args_def_fn,                        \
                                    meta_kernel_fn,                     \
                                    cpp_dtype,                          \
                                    ...)                                \
  static ::paddle::OpKernelInfoBuilder PD_CONCATENATE(                  \
      custom_kernel_info_##kernel_name##_##backend##_##layout##_,       \
      registrar_id) =                                                   \
      ::paddle::OpKernelInfoBuilder(                                    \
          #kernel_name,                                                 \
          PD_BACKEND(backend),                                          \
          PD_DATALAYOUT(layout),                                        \
          ::paddle::experimental::CppTypeToDataType<cpp_dtype>::Type()) \
          .SetKernelFn(PD_PT_KERNEL(                                    \
              meta_kernel_fn<cpp_dtype, ::paddle::backend##Context>))   \
          .SetVariadicKernelFn(PD_PT_VARIADIC_KERNEL(                   \
              meta_kernel_fn<cpp_dtype, ::paddle::backend##Context>))   \
          .ArgsParse(PD_PT_ARGS_PARSE(                                  \
              meta_kernel_fn<cpp_dtype, ::paddle::backend##Context>))   \
          .ArgsDef(args_def_fn);                                        \
  PD_EXPAND(_PD_KERNEL_REGISTRAR_INIT_3(kernel_name,                    \
                                        backend,                        \
                                        layout,                         \
                                        PD_ID,                          \
                                        args_def_fn,                    \
                                        meta_kernel_fn,                 \
                                        ##__VA_ARGS__))

#define _PD_KERNEL_REGISTRAR_INIT_5(kernel_name,                        \
                                    backend,                            \
                                    layout,                             \
                                    registrar_id,                       \
                                    args_def_fn,                        \
                                    meta_kernel_fn,                     \
                                    cpp_dtype,                          \
                                    ...)                                \
  static ::paddle::OpKernelInfoBuilder PD_CONCATENATE(                  \
      custom_kernel_info_##kernel_name##_##backend##_##layout##_,       \
      registrar_id) =                                                   \
      ::paddle::OpKernelInfoBuilder(                                    \
          #kernel_name,                                                 \
          PD_BACKEND(backend),                                          \
          PD_DATALAYOUT(layout),                                        \
          ::paddle::experimental::CppTypeToDataType<cpp_dtype>::Type()) \
          .SetKernelFn(PD_PT_KERNEL(                                    \
              meta_kernel_fn<cpp_dtype, ::paddle::backend##Context>))   \
          .SetVariadicKernelFn(PD_PT_VARIADIC_KERNEL(                   \
              meta_kernel_fn<cpp_dtype, ::paddle::backend##Context>))   \
          .ArgsParse(PD_PT_ARGS_PARSE(                                  \
              meta_kernel_fn<cpp_dtype, ::paddle::backend##Context>))   \
          .ArgsDef(args_def_fn);                                        \
  PD_EXPAND(_PD_KERNEL_REGISTRAR_INIT_4(kernel_name,                    \
                                        backend,                        \
                                        layout,                         \
                                        PD_ID,                          \
                                        args_def_fn,                    \
                                        meta_kernel_fn,                 \
                                        ##__VA_ARGS__))

#define _PD_KERNEL_REGISTRAR_INIT_6(kernel_name,                        \
                                    backend,                            \
                                    layout,                             \
                                    registrar_id,                       \
                                    args_def_fn,                        \
                                    meta_kernel_fn,                     \
                                    cpp_dtype,                          \
                                    ...)                                \
  static ::paddle::OpKernelInfoBuilder PD_CONCATENATE(                  \
      custom_kernel_info_##kernel_name##_##backend##_##layout##_,       \
      registrar_id) =                                                   \
      ::paddle::OpKernelInfoBuilder(                                    \
          #kernel_name,                                                 \
          PD_BACKEND(backend),                                          \
          PD_DATALAYOUT(layout),                                        \
          ::paddle::experimental::CppTypeToDataType<cpp_dtype>::Type()) \
          .SetKernelFn(PD_PT_KERNEL(                                    \
              meta_kernel_fn<cpp_dtype, ::paddle::backend##Context>))   \
          .SetVariadicKernelFn(PD_PT_VARIADIC_KERNEL(                   \
              meta_kernel_fn<cpp_dtype, ::paddle::backend##Context>))   \
          .ArgsParse(PD_PT_ARGS_PARSE(                                  \
              meta_kernel_fn<cpp_dtype, ::paddle::backend##Context>))   \
          .ArgsDef(args_def_fn);                                        \
  PD_EXPAND(_PD_KERNEL_REGISTRAR_INIT_5(kernel_name,                    \
                                        backend,                        \
                                        layout,                         \
                                        PD_ID,                          \
                                        args_def_fn,                    \
                                        meta_kernel_fn,                 \
                                        ##__VA_ARGS__))

#define _PD_KERNEL_REGISTRAR_INIT_7(kernel_name,                        \
                                    backend,                            \
                                    layout,                             \
                                    registrar_id,                       \
                                    args_def_fn,                        \
                                    meta_kernel_fn,                     \
                                    cpp_dtype,                          \
                                    ...)                                \
  static ::paddle::OpKernelInfoBuilder PD_CONCATENATE(                  \
      custom_kernel_info_##kernel_name##_##backend##_##layout##_,       \
      registrar_id) =                                                   \
      ::paddle::OpKernelInfoBuilder(                                    \
          #kernel_name,                                                 \
          PD_BACKEND(backend),                                          \
          PD_DATALAYOUT(layout),                                        \
          ::paddle::experimental::CppTypeToDataType<cpp_dtype>::Type()) \
          .SetKernelFn(PD_PT_KERNEL(                                    \
              meta_kernel_fn<cpp_dtype, ::paddle::backend##Context>))   \
          .SetVariadicKernelFn(PD_PT_VARIADIC_KERNEL(                   \
              meta_kernel_fn<cpp_dtype, ::paddle::backend##Context>))   \
          .ArgsParse(PD_PT_ARGS_PARSE(                                  \
              meta_kernel_fn<cpp_dtype, ::paddle::backend##Context>))   \
          .ArgsDef(args_def_fn);                                        \
  PD_EXPAND(_PD_KERNEL_REGISTRAR_INIT_6(kernel_name,                    \
                                        backend,                        \
                                        layout,                         \
                                        PD_ID,                          \
                                        args_def_fn,                    \
                                        meta_kernel_fn,                 \
                                        ##__VA_ARGS__))

#define _PD_KERNEL_REGISTRAR_INIT_8(kernel_name,                        \
                                    backend,                            \
                                    layout,                             \
                                    registrar_id,                       \
                                    args_def_fn,                        \
                                    meta_kernel_fn,                     \
                                    cpp_dtype,                          \
                                    ...)                                \
  static ::paddle::OpKernelInfoBuilder PD_CONCATENATE(                  \
      custom_kernel_info_##kernel_name##_##backend##_##layout##_,       \
      registrar_id) =                                                   \
      ::paddle::OpKernelInfoBuilder(                                    \
          #kernel_name,                                                 \
          PD_BACKEND(backend),                                          \
          PD_DATALAYOUT(layout),                                        \
          ::paddle::experimental::CppTypeToDataType<cpp_dtype>::Type()) \
          .SetKernelFn(PD_PT_KERNEL(                                    \
              meta_kernel_fn<cpp_dtype, ::paddle::backend##Context>))   \
          .SetVariadicKernelFn(PD_PT_VARIADIC_KERNEL(                   \
              meta_kernel_fn<cpp_dtype, ::paddle::backend##Context>))   \
          .ArgsParse(PD_PT_ARGS_PARSE(                                  \
              meta_kernel_fn<cpp_dtype, ::paddle::backend##Context>))   \
          .ArgsDef(args_def_fn);                                        \
  PD_EXPAND(_PD_KERNEL_REGISTRAR_INIT_7(kernel_name,                    \
                                        backend,                        \
                                        layout,                         \
                                        PD_ID,                          \
                                        args_def_fn,                    \
                                        meta_kernel_fn,                 \
                                        ##__VA_ARGS__))

#define _PD_KERNEL_REGISTRAR_INIT_9(kernel_name,                        \
                                    backend,                            \
                                    layout,                             \
                                    registrar_id,                       \
                                    args_def_fn,                        \
                                    meta_kernel_fn,                     \
                                    cpp_dtype,                          \
                                    ...)                                \
  static ::paddle::OpKernelInfoBuilder PD_CONCATENATE(                  \
      custom_kernel_info_##kernel_name##_##backend##_##layout##_,       \
      registrar_id) =                                                   \
      ::paddle::OpKernelInfoBuilder(                                    \
          #kernel_name,                                                 \
          PD_BACKEND(backend),                                          \
          PD_DATALAYOUT(layout),                                        \
          ::paddle::experimental::CppTypeToDataType<cpp_dtype>::Type()) \
          .SetKernelFn(PD_PT_KERNEL(                                    \
              meta_kernel_fn<cpp_dtype, ::paddle::backend##Context>))   \
          .SetVariadicKernelFn(PD_PT_VARIADIC_KERNEL(                   \
              meta_kernel_fn<cpp_dtype, ::paddle::backend##Context>))   \
          .ArgsParse(PD_PT_ARGS_PARSE(                                  \
              meta_kernel_fn<cpp_dtype, ::paddle::backend##Context>))   \
          .ArgsDef(args_def_fn);                                        \
  PD_EXPAND(_PD_KERNEL_REGISTRAR_INIT_8(kernel_name,                    \
                                        backend,                        \
                                        layout,                         \
                                        PD_ID,                          \
                                        args_def_fn,                    \
                                        meta_kernel_fn,                 \
                                        ##__VA_ARGS__))

#define _PD_KERNEL_REGISTRAR_INIT_10(kernel_name,                       \
                                     backend,                           \
                                     layout,                            \
                                     registrar_id,                      \
                                     args_def_fn,                       \
                                     meta_kernel_fn,                    \
                                     cpp_dtype,                         \
                                     ...)                               \
  static ::paddle::OpKernelInfoBuilder PD_CONCATENATE(                  \
      custom_kernel_info_##kernel_name##_##backend##_##layout##_,       \
      registrar_id) =                                                   \
      ::paddle::OpKernelInfoBuilder(                                    \
          #kernel_name,                                                 \
          PD_BACKEND(backend),                                          \
          PD_DATALAYOUT(layout),                                        \
          ::paddle::experimental::CppTypeToDataType<cpp_dtype>::Type()) \
          .SetKernelFn(PD_PT_KERNEL(                                    \
              meta_kernel_fn<cpp_dtype, ::paddle::backend##Context>))   \
          .SetVariadicKernelFn(PD_PT_VARIADIC_KERNEL(                   \
              meta_kernel_fn<cpp_dtype, ::paddle::backend##Context>))   \
          .ArgsParse(PD_PT_ARGS_PARSE(                                  \
              meta_kernel_fn<cpp_dtype, ::paddle::backend##Context>))   \
          .ArgsDef(args_def_fn);                                        \
  PD_EXPAND(_PD_KERNEL_REGISTRAR_INIT_9(kernel_name,                    \
                                        backend,                        \
                                        layout,                         \
                                        PD_ID,                          \
                                        args_def_fn,                    \
                                        meta_kernel_fn,                 \
                                        ##__VA_ARGS__))

#define _PD_KERNEL_REGISTRAR_INIT_11(kernel_name,                       \
                                     backend,                           \
                                     layout,                            \
                                     registrar_id,                      \
                                     args_def_fn,                       \
                                     meta_kernel_fn,                    \
                                     cpp_dtype,                         \
                                     ...)                               \
  static ::paddle::OpKernelInfoBuilder PD_CONCATENATE(                  \
      custom_kernel_info_##kernel_name##_##backend##_##layout##_,       \
      registrar_id) =                                                   \
      ::paddle::OpKernelInfoBuilder(                                    \
          #kernel_name,                                                 \
          PD_BACKEND(backend),                                          \
          PD_DATALAYOUT(layout),                                        \
          ::paddle::experimental::CppTypeToDataType<cpp_dtype>::Type()) \
          .SetKernelFn(PD_PT_KERNEL(                                    \
              meta_kernel_fn<cpp_dtype, ::paddle::backend##Context>))   \
          .SetVariadicKernelFn(PD_PT_VARIADIC_KERNEL(                   \
              meta_kernel_fn<cpp_dtype, ::paddle::backend##Context>))   \
          .ArgsParse(PD_PT_ARGS_PARSE(                                  \
              meta_kernel_fn<cpp_dtype, ::paddle::backend##Context>))   \
          .ArgsDef(args_def_fn);                                        \
  PD_EXPAND(_PD_KERNEL_REGISTRAR_INIT_10(kernel_name,                   \
                                         backend,                       \
                                         layout,                        \
                                         PD_ID,                         \
                                         args_def_fn,                   \
                                         meta_kernel_fn,                \
                                         ##__VA_ARGS__))

#define _PD_KERNEL_REGISTRAR_INIT_12(kernel_name,                       \
                                     backend,                           \
                                     layout,                            \
                                     registrar_id,                      \
                                     args_def_fn,                       \
                                     meta_kernel_fn,                    \
                                     cpp_dtype,                         \
                                     ...)                               \
  static ::paddle::OpKernelInfoBuilder PD_CONCATENATE(                  \
      custom_kernel_info_##kernel_name##_##backend##_##layout##_,       \
      registrar_id) =                                                   \
      ::paddle::OpKernelInfoBuilder(                                    \
          #kernel_name,                                                 \
          PD_BACKEND(backend),                                          \
          PD_DATALAYOUT(layout),                                        \
          ::paddle::experimental::CppTypeToDataType<cpp_dtype>::Type()) \
          .SetKernelFn(PD_PT_KERNEL(                                    \
              meta_kernel_fn<cpp_dtype, ::paddle::backend##Context>))   \
          .SetVariadicKernelFn(PD_PT_VARIADIC_KERNEL(                   \
              meta_kernel_fn<cpp_dtype, ::paddle::backend##Context>))   \
          .ArgsParse(PD_PT_ARGS_PARSE(                                  \
              meta_kernel_fn<cpp_dtype, ::paddle::backend##Context>))   \
          .ArgsDef(args_def_fn);                                        \
  PD_EXPAND(_PD_KERNEL_REGISTRAR_INIT_11(kernel_name,                   \
                                         backend,                       \
                                         layout,                        \
                                         PD_ID,                         \
                                         args_def_fn,                   \
                                         meta_kernel_fn,                \
                                         ##__VA_ARGS__))

#define _PD_KERNEL_REGISTRAR_INIT_13(kernel_name,                       \
                                     backend,                           \
                                     layout,                            \
                                     registrar_id,                      \
                                     args_def_fn,                       \
                                     meta_kernel_fn,                    \
                                     cpp_dtype,                         \
                                     ...)                               \
  static ::paddle::OpKernelInfoBuilder PD_CONCATENATE(                  \
      custom_kernel_info_##kernel_name##_##backend##_##layout##_,       \
      registrar_id) =                                                   \
      ::paddle::OpKernelInfoBuilder(                                    \
          #kernel_name,                                                 \
          PD_BACKEND(backend),                                          \
          PD_DATALAYOUT(layout),                                        \
          ::paddle::experimental::CppTypeToDataType<cpp_dtype>::Type()) \
          .SetKernelFn(PD_PT_KERNEL(                                    \
              meta_kernel_fn<cpp_dtype, ::paddle::backend##Context>))   \
          .SetVariadicKernelFn(PD_PT_VARIADIC_KERNEL(                   \
              meta_kernel_fn<cpp_dtype, ::paddle::backend##Context>))   \
          .ArgsParse(PD_PT_ARGS_PARSE(                                  \
              meta_kernel_fn<cpp_dtype, ::paddle::backend##Context>))   \
          .ArgsDef(args_def_fn);                                        \
  PD_EXPAND(_PD_KERNEL_REGISTRAR_INIT_12(kernel_name,                   \
                                         backend,                       \
                                         layout,                        \
                                         PD_ID,                         \
                                         args_def_fn,                   \
                                         meta_kernel_fn,                \
                                         ##__VA_ARGS__))

#define _PD_KERNEL_REGISTRAR_INIT_14(kernel_name,                       \
                                     backend,                           \
                                     layout,                            \
                                     registrar_id,                      \
                                     args_def_fn,                       \
                                     meta_kernel_fn,                    \
                                     cpp_dtype,                         \
                                     ...)                               \
  static ::paddle::OpKernelInfoBuilder PD_CONCATENATE(                  \
      custom_kernel_info_##kernel_name##_##backend##_##layout##_,       \
      registrar_id) =                                                   \
      ::paddle::OpKernelInfoBuilder(                                    \
          #kernel_name,                                                 \
          PD_BACKEND(backend),                                          \
          PD_DATALAYOUT(layout),                                        \
          ::paddle::experimental::CppTypeToDataType<cpp_dtype>::Type()) \
          .SetKernelFn(PD_PT_KERNEL(                                    \
              meta_kernel_fn<cpp_dtype, ::paddle::backend##Context>))   \
          .SetVariadicKernelFn(PD_PT_VARIADIC_KERNEL(                   \
              meta_kernel_fn<cpp_dtype, ::paddle::backend##Context>))   \
          .ArgsParse(PD_PT_ARGS_PARSE(                                  \
              meta_kernel_fn<cpp_dtype, ::paddle::backend##Context>))   \
          .ArgsDef(args_def_fn);                                        \
  PD_EXPAND(_PD_KERNEL_REGISTRAR_INIT_13(kernel_name,                   \
                                         backend,                       \
                                         layout,                        \
                                         PD_ID,                         \
                                         args_def_fn,                   \
                                         meta_kernel_fn,                \
                                         ##__VA_ARGS__))

#define _PD_KERNEL_REGISTRAR_INIT_15(kernel_name,                       \
                                     backend,                           \
                                     layout,                            \
                                     registrar_id,                      \
                                     args_def_fn,                       \
                                     meta_kernel_fn,                    \
                                     cpp_dtype,                         \
                                     ...)                               \
  static ::paddle::OpKernelInfoBuilder PD_CONCATENATE(                  \
      custom_kernel_info_##kernel_name##_##backend##_##layout##_,       \
      registrar_id) =                                                   \
      ::paddle::OpKernelInfoBuilder(                                    \
          #kernel_name,                                                 \
          PD_BACKEND(backend),                                          \
          PD_DATALAYOUT(layout),                                        \
          ::paddle::experimental::CppTypeToDataType<cpp_dtype>::Type()) \
          .SetKernelFn(PD_PT_KERNEL(                                    \
              meta_kernel_fn<cpp_dtype, ::paddle::backend##Context>))   \
          .SetVariadicKernelFn(PD_PT_VARIADIC_KERNEL(                   \
              meta_kernel_fn<cpp_dtype, ::paddle::backend##Context>))   \
          .ArgsParse(PD_PT_ARGS_PARSE(                                  \
              meta_kernel_fn<cpp_dtype, ::paddle::backend##Context>))   \
          .ArgsDef(args_def_fn);                                        \
  PD_EXPAND(_PD_KERNEL_REGISTRAR_INIT_14(kernel_name,                   \
                                         backend,                       \
                                         layout,                        \
                                         PD_ID,                         \
                                         args_def_fn,                   \
                                         meta_kernel_fn,                \
                                         ##__VA_ARGS__))
}  // namespace paddle
