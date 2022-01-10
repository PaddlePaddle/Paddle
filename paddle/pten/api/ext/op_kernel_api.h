/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/utils/any.h"
#include "paddle/utils/small_vector.h"

using DevContext = void*;

namespace paddle {
namespace framework {
class PADDLE_API OpKernelInfoHelper;
}  // namespace framework

// Record Op kernel core function
using PtenKernelFunc =
    void (*)(const DevContext& dev_ctx,
             const std::vector<Tensor>& inputs,
             const std::vector<std::vector<Tensor>>& vec_inputs,
             const std::vector<paddle::any>& attrs,
             std::vector<Tensor*>* outputs,
             std::vector<std::vector<Tensor*>>* vec_outputs);

////////////////////// Kernel Function (PD_PTEN_KERNEL) ////////////////////////

// can not use PT_KERNEL directly without exposing KernelContext...
#define PD_SPECIALIZE_KernelCallHelper_FOR_DEV_CONTEXT(device_ctx)           \
  template <typename... Tail>                                                \
  struct PtenComputeCallHelper<const device_ctx&, Tail...> {                 \
    template <int dev_ctx_idx,                                               \
              int in_idx,                                                    \
              int vec_in_idx,                                                \
              int attr_idx,                                                  \
              int out_idx,                                                   \
              int vec_out_idx,                                               \
              typename... PreviousArgs>                                      \
    static void Compute(const DevContext& dev_ctx,                           \
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
      const device_ctx& arg = dev_ctx;                                       \
      PtenComputeCallHelper<Tail...>::template Compute<dev_ctx_idx + 1,      \
                                                       in_idx,               \
                                                       vec_in_idx,           \
                                                       attr_idx,             \
                                                       out_idx,              \
                                                       vec_out_idx>(         \
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
  struct PtenComputeCallHelper<const tensor_type&, Tail...> {               \
    template <int dev_ctx_idx,                                              \
              int in_idx,                                                   \
              int vec_in_idx,                                               \
              int attr_idx,                                                 \
              int out_idx,                                                  \
              int vec_out_idx,                                              \
              typename... PreviousArgs>                                     \
    static void Compute(const DevContext& dev_ctx,                          \
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
      PtenComputeCallHelper<Tail...>::template Compute<dev_ctx_idx,         \
                                                       in_idx + 1,          \
                                                       vec_in_idx,          \
                                                       attr_idx,            \
                                                       out_idx,             \
                                                       vec_out_idx>(        \
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

#define PD_SPECIALIZE_KernelCallHelper_FOR_MULTI_INPUT(tensor_type)         \
  template <typename... Tail>                                               \
  struct PtenComputeCallHelper<const std::vector<tensor_type>&, Tail...> {  \
    template <int dev_ctx_idx,                                              \
              int in_idx,                                                   \
              int vec_in_idx,                                               \
              int attr_idx,                                                 \
              int out_idx,                                                  \
              int vec_out_idx,                                              \
              typename... PreviousArgs>                                     \
    static void Compute(const DevContext& dev_ctx,                          \
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
      const std::vector<Tensor>& arg = vec_inputs[vec_in_idx];              \
      PtenComputeCallHelper<Tail...>::template Compute<dev_ctx_idx,         \
                                                       in_idx,              \
                                                       vec_in_idx + 1,      \
                                                       attr_idx,            \
                                                       out_idx,             \
                                                       vec_out_idx>(        \
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

#define PD_SPECIALIZE_KernelCallHelper_FOR_ATTRIBUTE(attr_type)               \
  template <typename... Tail>                                                 \
  struct PtenComputeCallHelper<attr_type, Tail...> {                          \
    template <int dev_ctx_idx,                                                \
              int in_idx,                                                     \
              int vec_in_idx,                                                 \
              int attr_idx,                                                   \
              int out_idx,                                                    \
              int vec_out_idx,                                                \
              typename... PreviousArgs>                                       \
    static void Compute(const DevContext& dev_ctx,                            \
                        const std::vector<Tensor>& inputs,                    \
                        const std::vector<std::vector<Tensor>>& vec_inputs,   \
                        const std::vector<paddle::any>& attrs,                \
                        std::vector<Tensor*>* outputs,                        \
                        std::vector<std::vector<Tensor*>>* vec_outputs,       \
                        PreviousArgs... pargs) {                              \
      static_assert(out_idx == 0,                                             \
                    "Kernel's Attributes should appear before Outputs.");     \
      static_assert(vec_out_idx == 0,                                         \
                    "Kernel's Attributes should appear before Outputs.");     \
      try {                                                                   \
        attr_type arg = paddle::any_cast<attr_type>(attrs[attr_idx]);         \
        return PtenComputeCallHelper<Tail...>::template Compute<dev_ctx_idx,  \
                                                                in_idx,       \
                                                                vec_in_idx,   \
                                                                attr_idx + 1, \
                                                                out_idx,      \
                                                                vec_out_idx>( \
            dev_ctx,                                                          \
            inputs,                                                           \
            vec_inputs,                                                       \
            attrs,                                                            \
            outputs,                                                          \
            vec_outputs,                                                      \
            pargs...,                                                         \
            arg);                                                             \
      } catch (paddle::bad_any_cast&) {                                       \
        PD_THROW(                                                             \
            "Attribute cast error in custom operator. Expected " #attr_type   \
            " value.");                                                       \
      }                                                                       \
    }                                                                         \
  }

#define PD_SPECIALIZE_KernelCallHelper_FOR_OUTPUT(tensor_type)                \
  template <typename... Tail>                                                 \
  struct PtenComputeCallHelper<tensor_type*, Tail...> {                       \
    template <int dev_ctx_idx,                                                \
              int in_idx,                                                     \
              int vec_in_idx,                                                 \
              int attr_idx,                                                   \
              int out_idx,                                                    \
              int vec_out_idx,                                                \
              typename... PreviousArgs>                                       \
    static void Compute(const DevContext& dev_ctx,                            \
                        const std::vector<Tensor>& inputs,                    \
                        const std::vector<std::vector<Tensor>>& vec_inputs,   \
                        const std::vector<paddle::any>& attrs,                \
                        std::vector<Tensor*>* outputs,                        \
                        std::vector<std::vector<Tensor*>>* vec_outputs,       \
                        PreviousArgs... pargs) {                              \
      tensor_type* arg = (*outputs)[out_idx];                                 \
      std::cout << "[CUSTOM PTEN KERNEL] " << arg << std::endl;               \
      std::cout << "[CUSTOM PTEN KERNEL] " << arg->impl().get() << std::endl; \
      PtenComputeCallHelper<Tail...>::template Compute<dev_ctx_idx,           \
                                                       in_idx,                \
                                                       vec_in_idx,            \
                                                       attr_idx,              \
                                                       out_idx + 1,           \
                                                       vec_out_idx>(          \
          dev_ctx,                                                            \
          inputs,                                                             \
          vec_inputs,                                                         \
          attrs,                                                              \
          outputs,                                                            \
          vec_outputs,                                                        \
          pargs...,                                                           \
          arg);                                                               \
    }                                                                         \
  }

#define PD_SPECIALIZE_KernelCallHelper_FOR_MULTI_OUTPUT(tensor_type)        \
  template <typename... Tail>                                               \
  struct PtenComputeCallHelper<std::vector<tensor_type*>, Tail...> {        \
    template <int dev_ctx_idx,                                              \
              int in_idx,                                                   \
              int vec_in_idx,                                               \
              int attr_idx,                                                 \
              int out_idx,                                                  \
              int vec_out_idx,                                              \
              typename... PreviousArgs>                                     \
    static void Compute(const DevContext& dev_ctx,                          \
                        const std::vector<Tensor>& inputs,                  \
                        const std::vector<std::vector<Tensor>>& vec_inputs, \
                        const std::vector<paddle::any>& attrs,              \
                        std::vector<Tensor*>* outputs,                      \
                        std::vector<std::vector<Tensor*>>* vec_outputs,     \
                        PreviousArgs... pargs) {                            \
      std::vector<tensor_type*> arg = (*vec_outputs)[vec_out_idx];          \
      PtenComputeCallHelper<Tail...>::template Compute<dev_ctx_idx,         \
                                                       in_idx,              \
                                                       vec_in_idx,          \
                                                       attr_idx,            \
                                                       out_idx,             \
                                                       vec_out_idx + 1>(    \
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
struct PtenKernelFuncImpl;

template <typename Return,
          typename DevCtx,
          typename... Args,
          Return (*impl_fn)(DevCtx, Args...)>
struct PtenKernelFuncImpl<Return (*)(DevCtx, Args...), impl_fn> {
  static void Compute(
      const DevContext& dev_ctx,
      const std::vector<paddle::Tensor>& inputs,
      const std::vector<std::vector<paddle::Tensor>>& vec_inputs,
      const std::vector<paddle::any>& attrs,
      std::vector<paddle::Tensor*>* outputs,
      std::vector<std::vector<paddle::Tensor*>>* vec_outputs) {
    PtenComputeCallHelper<DevCtx, Args..., PtenTypeTag<int>>::
        template Compute<0, 0, 0, 0, 0, 0>(
            dev_ctx, inputs, vec_inputs, attrs, outputs, vec_outputs);
  }

  // not finished: pass directly is not OK: DenseTensor -> paddle::Tensor
  static void VariadicCompute(const DevContext& dev_ctx, Args... args) {
    return impl_fn(static_cast<DevCtx>(dev_ctx), std::forward<Args>(args)...);
  }

 private:
  template <typename... RemainingArgs>
  struct PtenComputeCallHelper;

  /* DeviceContext Helpers */
  PD_SPECIALIZE_KernelCallHelper_FOR_DEV_CONTEXT(DevContext);

  /* Input Helpers */
  PD_SPECIALIZE_KernelCallHelper_FOR_INPUT(Tensor);
  PD_SPECIALIZE_KernelCallHelper_FOR_MULTI_INPUT(Tensor);
  // TODO(chenweihang): adapt SelectedRows
  // PT_SPECIALIZE_KernelCallHelper_FOR_INPUT(SelectedRowsTensor);

  /* Attribute Helpers */
  PD_SPECIALIZE_KernelCallHelper_FOR_ATTRIBUTE(bool);
  PD_SPECIALIZE_KernelCallHelper_FOR_ATTRIBUTE(float);
  PD_SPECIALIZE_KernelCallHelper_FOR_ATTRIBUTE(double);
  PD_SPECIALIZE_KernelCallHelper_FOR_ATTRIBUTE(int);
  PD_SPECIALIZE_KernelCallHelper_FOR_ATTRIBUTE(int64_t);
  PD_SPECIALIZE_KernelCallHelper_FOR_ATTRIBUTE(paddle::platform::float16);
  // PD_SPECIALIZE_KernelCallHelper_FOR_ATTRIBUTE(const Scalar&);
  PD_SPECIALIZE_KernelCallHelper_FOR_ATTRIBUTE(DataType);
  PD_SPECIALIZE_KernelCallHelper_FOR_ATTRIBUTE(const std::vector<int64_t>&);
  // PD_SPECIALIZE_KernelCallHelper_FOR_ATTRIBUTE(const ScalarArray&);
  PD_SPECIALIZE_KernelCallHelper_FOR_ATTRIBUTE(const std::vector<int>&);

  /* Output Helpers */
  PD_SPECIALIZE_KernelCallHelper_FOR_OUTPUT(Tensor);
  PD_SPECIALIZE_KernelCallHelper_FOR_MULTI_OUTPUT(Tensor);
  // TODO(chenweihang): adapt SelectedRows
  // PT_SPECIALIZE_KernelCallHelper_FOR_OUTPUT(SelectedRowsTensor);

  // end: base template
  template <typename T>
  struct PtenComputeCallHelper<PtenTypeTag<T>> {
    template <int dev_ctx_idx,
              int in_idx,
              int vec_in_idx,
              int attr_idx,
              int out_idx,
              int vec_out_idx>
    static void Compute(const DevContext& dev_ctx,
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

#define PD_PTEN_KERNEL(...) \
  ::paddle::PtenKernelFuncImpl<decltype(&__VA_ARGS__), &__VA_ARGS__>::Compute

#define PD_PTEN_VARIADIC_KERNEL(...)                        \
  reinterpret_cast<void*>(                                  \
      &::paddle::PtenKernelFuncImpl<decltype(&__VA_ARGS__), \
                                    &__VA_ARGS__>::VariadicCompute)

class OpKernelInfo;
using PtenKernelArgsParseFn = void (*)(OpKernelInfo* op_kernel_info);
using PtenKernelArgsDefFn = void (*)(OpKernelInfo* op_kernel_info);

// for TensorArgDef
struct PtenTensorArgDef {
  pten::Backend backend;
  pten::DataLayout layout;
  pten::DataType dtype;

  PtenTensorArgDef(pten::Backend in_backend,
                   pten::DataLayout in_layout,
                   pten::DataType in_dtype)
      : backend(in_backend), layout(in_layout), dtype(in_dtype) {}
};

// for AttributeArgDef
struct PtenAttributeArgDef {
  std::type_index type_index;

  explicit PtenAttributeArgDef(std::type_index type_index)
      : type_index(type_index) {}
};

////////////////////// Op Kernel Info //////////////////////
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

  // format: PD_KERNEL(...)
  OpKernelInfo& SetKernelFn(KernelFunc&& func);
  // format: PD_PTEN_KERNEL(...)
  OpKernelInfo& SetPtenKernelFn(PtenKernelFunc&& func);
  // void*
  OpKernelInfo& SetPtenVariadicFn(void* func);
  // format: PD_PTEN_ARGS_PARSE(...)
  OpKernelInfo& SetPtenArgsParseFn(PtenKernelArgsParseFn&& func);
  // user define
  OpKernelInfo& SetPtenArgsDefFn(PtenKernelArgsDefFn&& func);

  pten::Backend& GetBackend() { return backend_; }
  pten::DataLayout& GetDataLayout() { return layout_; }
  pten::DataType& GetDataType() { return dtype_; }

  void AppendInput(pten::Backend backend,
                   pten::DataLayout layout,
                   pten::DataType dtype) {
    input_defs_.emplace_back(PtenTensorArgDef(backend, layout, dtype));
  }

  void AppendOutput(pten::Backend backend,
                    pten::DataLayout layout,
                    pten::DataType dtype) {
    output_defs_.emplace_back(PtenTensorArgDef(backend, layout, dtype));
  }

  void AppendAttribute(std::type_index type_index) {
    attribute_defs_.emplace_back(PtenAttributeArgDef(type_index));
  }

  const paddle::SmallVector<PtenTensorArgDef>& input_defs() const {
    return input_defs_;
  }

  const paddle::SmallVector<PtenTensorArgDef>& output_defs() const {
    return output_defs_;
  }

  const paddle::SmallVector<PtenAttributeArgDef>& attribute_defs() const {
    return attribute_defs_;
  }

  // user_args_def_func may use?
  paddle::SmallVector<PtenTensorArgDef>& input_defs() { return input_defs_; }
  paddle::SmallVector<PtenTensorArgDef>& output_defs() { return output_defs_; }
  paddle::SmallVector<PtenAttributeArgDef>& attribute_defs() {
    return attribute_defs_;
  }

 private:
  friend class framework::OpKernelInfoHelper;

  // 1. op info
  std::string op_name_;

  // 2. kernel key info
  pten::Backend backend_{pten::Backend::UNDEFINED};
  pten::DataLayout layout_{pten::DataLayout::UNDEFINED};
  pten::DataType dtype_{pten::DataType::UNDEFINED};

  // 3. args info
  PtenKernelArgsParseFn pten_kernel_args_parse_fn_{nullptr};
  paddle::SmallVector<PtenTensorArgDef> input_defs_{{}};
  paddle::SmallVector<PtenTensorArgDef> output_defs_{{}};
  paddle::SmallVector<PtenAttributeArgDef> attribute_defs_{{}};
  PtenKernelArgsDefFn pten_kernel_args_def_fn_{nullptr};

  // 4. func info
  KernelFunc kernel_fn_{nullptr};           // for fluid kernel func call
  PtenKernelFunc pten_kernel_fn_{nullptr};  // for pten kernel func call
  void* variadic_fn_{nullptr};              // for pten variadic func call
};

////////////////////// Op Kernel Args Parser //////////////////////

template <typename Func>
struct PtenKernelArgsParseFunctor;

template <typename Return_, typename... Args_>
struct PtenKernelArgsParseFunctor<Return_ (*)(Args_...)> {
  using Args = std::tuple<Args_...>;
  enum : std::size_t { Arity = sizeof...(Args_) };
  using Indices = std::make_index_sequence<Arity>;
  template <std::size_t Index>
  using Arg = typename std::tuple_element<Index, Args>::type;

  static void Parse(OpKernelInfo* op_kernel_info) {
    // TODO(chenweihang): The fluid Tensor's default layout is NCHW,
    // it is not same as kernel's layout, we should fix this error on
    // fluid Tensor
    pten::Backend& backend = op_kernel_info->GetBackend();
    pten::DataLayout& layout = op_kernel_info->GetDataLayout();
    pten::DataType& dtype = op_kernel_info->GetDataType();

    auto default_tensor_layout = pten::DataLayout::NCHW;
    if (layout != pten::DataLayout::ANY) {
      default_tensor_layout = layout;
    }
    auto args_type = ParseArgType(Indices{});
    for (auto arg_type : args_type) {
      if (arg_type == std::type_index(typeid(const DevContext&))) {
        // do nothing, skip context arg now
      } else if (arg_type == std::type_index(typeid(const Tensor&))) {
        op_kernel_info->AppendInput(backend, default_tensor_layout, dtype);
      } else if (arg_type ==
                 std::type_index(typeid(const std::vector<Tensor>&))) {
        op_kernel_info->AppendInput(backend, default_tensor_layout, dtype);
      } else if (arg_type == std::type_index(typeid(Tensor*))) {
        op_kernel_info->AppendOutput(backend, default_tensor_layout, dtype);
      } else if (arg_type == std::type_index(typeid(std::vector<Tensor*>))) {
        op_kernel_info->AppendOutput(backend, default_tensor_layout, dtype);
      } else {
        // Attribute deal with
        // TODO(chenweihang): now here allow any types of attribute, maybe
        // should add limits here
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

#define PD_PTEN_ARGS_PARSE(...) \
  ::paddle::PtenKernelArgsParseFunctor<decltype(&__VA_ARGS__)>::Parse

//////////////// Op Kernel Info Map /////////////////

class PADDLE_API OpKernelInfoMap {
 public:
  // this function's impl should keep in header file.
  // if move to cc file, meta info can not be added
  // into map
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

class PADDLE_API OpKernelInfoBuilder {
 public:
  explicit OpKernelInfoBuilder(std::string&& op_name,
                               pten::Backend backend,
                               pten::DataLayout data_layout,
                               pten::DataType data_type);
  OpKernelInfoBuilder& SetKernelFn(KernelFunc func);
  OpKernelInfoBuilder& SetPtenKernelFn(PtenKernelFunc func);
  OpKernelInfoBuilder& SetPtenVariadicFn(void* func);
  OpKernelInfoBuilder& SetPtenArgsParseFn(PtenKernelArgsParseFn func);
  OpKernelInfoBuilder& SetPtenArgsDefFn(PtenKernelArgsDefFn func);

 private:
  // Forward Op name
  std::string op_name_;

  // kernel key info
  pten::Backend backend_{pten::Backend::UNDEFINED};
  pten::DataLayout layout_{pten::DataLayout::UNDEFINED};
  pten::DataType dtype_{pten::DataType::UNDEFINED};

  // ref current info ptr
  OpKernelInfo* info_ptr_;
};

#define PD_BACKEND(arg__) pten::Backend::arg__
#define PD_DATALAYOUT(arg__) pten::DataLayout::arg__
#define PD_DATATYPE(arg__) pten::DataType::arg__

#define PD_REG_KERNEL(op_name, dev, layout, dtype, func)                     \
  STATIC_ASSERT_GLOBAL_NAMESPACE(                                            \
      __reg_kernel__##op_name##_##dev##_##layout##_##dtype,                  \
      "PD_REG_KERNEL must be called in global namespace.");                  \
  void __PD_PTEN_USER_args_def_FN_##op_name##_##dev##_##layout(              \
      ::paddle::OpKernelInfo* op_kernel_info);                               \
  static ::paddle::OpKernelInfoBuilder                                       \
      __op_kernel_info_##op_name##_##dev##_##layout##_##dtype =              \
          ::paddle::OpKernelInfoBuilder(#op_name,                            \
                                        PD_BACKEND(dev),                     \
                                        PD_DATALAYOUT(layout),               \
                                        PD_DATATYPE(dtype))                  \
              .SetPtenKernelFn(PD_PTEN_KERNEL(func))                         \
              .SetPtenArgsParseFn(PD_PTEN_ARGS_PARSE(func))                  \
              .SetPtenVariadicFn(PD_PTEN_VARIADIC_KERNEL(func))              \
              .SetPtenArgsDefFn(                                             \
                  &__PD_PTEN_USER_args_def_FN_##op_name##_##dev##_##layout); \
  void __PD_PTEN_USER_args_def_FN_##op_name##_##dev##_##layout(              \
      ::paddle::OpKernelInfo* op_kernel_info)

}  // namespace paddle

#ifdef __cplusplus
extern "C" {
#endif

paddle::OpKernelInfoMap& PD_GetOpKernelInfoMap();

#ifdef __cplusplus
}  // end extern "C"
#endif
