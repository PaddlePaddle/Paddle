//   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/backends/custom/custom_context.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/phi/common/scalar_array.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/kernel_context.h"
#include "paddle/phi/core/selected_rows.h"
#include "paddle/phi/core/sparse_coo_tensor.h"
#include "paddle/phi/core/sparse_csr_tensor.h"
#include "paddle/phi/core/string_tensor.h"
#include "paddle/phi/core/type_defs.h"

namespace phi {

// PD_KERNEL has been used by custom op api
#define PHI_KERNEL(...) \
  ::phi::KernelImpl<decltype(&__VA_ARGS__), &__VA_ARGS__>::Compute

#define PHI_VARIADIC_KERNEL(...)                                     \
  reinterpret_cast<void*>(&::phi::KernelImpl<decltype(&__VA_ARGS__), \
                                             &__VA_ARGS__>::VariadicCompute)

#define PD_SPECIALIZE_KernelCallHelper_FOR_DEVICE_CONTEXT(dev_ctx)           \
  template <typename... Tail>                                                \
  struct KernelCallHelper<const dev_ctx&, Tail...> {                         \
    template <int dev_ctx_idx,                                               \
              int in_idx,                                                    \
              int attr_idx,                                                  \
              int out_idx,                                                   \
              typename... PreviousArgs>                                      \
    static void Compute(KernelContext* ctx, PreviousArgs&... pargs) {        \
      static_assert(in_idx == 0,                                             \
                    "Kernel's DeviceContext should appear before Inputs.");  \
      static_assert(                                                         \
          attr_idx == 0,                                                     \
          "Kernel's DeviceContext should appear before Attributes.");        \
      static_assert(out_idx == 0,                                            \
                    "Kernel's DeviceContext should appear before Outputs."); \
      const dev_ctx& arg = ctx->GetDeviceContext<dev_ctx>();                 \
      KernelCallHelper<Tail...>::                                            \
          template Compute<dev_ctx_idx + 1, in_idx, attr_idx, out_idx>(      \
              ctx, pargs..., arg);                                           \
    }                                                                        \
  }

#define PD_SPECIALIZE_KernelCallHelper_FOR_INPUT(tensor_type)           \
  template <typename... Tail>                                           \
  struct KernelCallHelper<const tensor_type&, Tail...> {                \
    template <int dev_ctx_idx,                                          \
              int in_idx,                                               \
              int attr_idx,                                             \
              int out_idx,                                              \
              typename... PreviousArgs>                                 \
    static void Compute(KernelContext* ctx, PreviousArgs&... pargs) {   \
      static_assert(attr_idx == 0,                                      \
                    "Kernel's Input should appear before Attributes."); \
      static_assert(out_idx == 0,                                       \
                    "Kernel's Input should appear before Outputs.");    \
      const std::pair<int, int> range = ctx->InputRangeAt(in_idx);      \
      const tensor_type& arg = ctx->InputAt<tensor_type>(range.first);  \
      KernelCallHelper<Tail...>::                                       \
          template Compute<dev_ctx_idx, in_idx + 1, attr_idx, out_idx>( \
              ctx, pargs..., arg);                                      \
    }                                                                   \
  }

#define PD_SPECIALIZE_KernelCallHelper_FOR_OPTIONAL_INPUT(tensor_type)     \
  template <typename... Tail>                                              \
  struct KernelCallHelper<paddle::optional<const tensor_type&>, Tail...> { \
    template <int dev_ctx_idx,                                             \
              int in_idx,                                                  \
              int attr_idx,                                                \
              int out_idx,                                                 \
              typename... PreviousArgs>                                    \
    static void Compute(KernelContext* ctx, PreviousArgs&... pargs) {      \
      static_assert(attr_idx == 0,                                         \
                    "Kernel's Input should appear before Attributes.");    \
      static_assert(out_idx == 0,                                          \
                    "Kernel's Input should appear before Outputs.");       \
      const std::pair<int, int> range = ctx->InputRangeAt(in_idx);         \
      auto arg = ctx->OptionalInputAt<tensor_type>(range.first);           \
      KernelCallHelper<Tail...>::                                          \
          template Compute<dev_ctx_idx, in_idx + 1, attr_idx, out_idx>(    \
              ctx, pargs..., arg);                                         \
    }                                                                      \
  }

#define PD_SPECIALIZE_KernelCallHelper_FOR_MULTI_INPUT(tensor_type)          \
  template <typename... Tail>                                                \
  struct KernelCallHelper<const std::vector<const tensor_type*>&, Tail...> { \
    template <int dev_ctx_idx,                                               \
              int in_idx,                                                    \
              int attr_idx,                                                  \
              int out_idx,                                                   \
              typename... PreviousArgs>                                      \
    static void Compute(KernelContext* ctx, PreviousArgs&... pargs) {        \
      static_assert(attr_idx == 0,                                           \
                    "Kernel's Input should appear before Attributes.");      \
      static_assert(out_idx == 0,                                            \
                    "Kernel's Input should appear before Outputs.");         \
      const std::pair<int, int> range = ctx->InputRangeAt(in_idx);           \
      std::vector<const tensor_type*> arg = std::move(                       \
          ctx->InputsBetween<tensor_type>(range.first, range.second));       \
      KernelCallHelper<Tail...>::                                            \
          template Compute<dev_ctx_idx, in_idx + 1, attr_idx, out_idx>(      \
              ctx, pargs..., arg);                                           \
    }                                                                        \
  }

#define PD_SPECIALIZE_KernelCallHelper_FOR_ATTRIBUTE(attr_type)           \
  template <typename... Tail>                                             \
  struct KernelCallHelper<attr_type, Tail...> {                           \
    template <int dev_ctx_idx,                                            \
              int in_idx,                                                 \
              int attr_idx,                                               \
              int out_idx,                                                \
              typename... PreviousArgs>                                   \
    static void Compute(KernelContext* ctx, PreviousArgs&... pargs) {     \
      static_assert(out_idx == 0,                                         \
                    "Kernel's Attributes should appear before Outputs."); \
      attr_type arg = ctx->AttrAt<attr_type>(attr_idx);                   \
      KernelCallHelper<Tail...>::                                         \
          template Compute<dev_ctx_idx, in_idx, attr_idx + 1, out_idx>(   \
              ctx, pargs..., arg);                                        \
    }                                                                     \
  }

#define PD_SPECIALIZE_KernelCallHelper_FOR_OUTPUT(tensor_type)           \
  template <typename... Tail>                                            \
  struct KernelCallHelper<tensor_type*, Tail...> {                       \
    template <int dev_ctx_idx,                                           \
              int in_idx,                                                \
              int attr_idx,                                              \
              int out_idx,                                               \
              typename... PreviousArgs>                                  \
    static void Compute(KernelContext* ctx, PreviousArgs&... pargs) {    \
      const std::pair<int, int> range = ctx->OutputRangeAt(out_idx);     \
      tensor_type* arg = ctx->MutableOutputAt<tensor_type>(range.first); \
      KernelCallHelper<Tail...>::                                        \
          template Compute<dev_ctx_idx, in_idx, attr_idx, out_idx + 1>(  \
              ctx, pargs..., arg);                                       \
    }                                                                    \
  }

#define PD_SPECIALIZE_KernelCallHelper_FOR_MULTI_OUTPUT(tensor_type)          \
  template <typename... Tail>                                                 \
  struct KernelCallHelper<std::vector<tensor_type*>, Tail...> {               \
    template <int dev_ctx_idx,                                                \
              int in_idx,                                                     \
              int attr_idx,                                                   \
              int out_idx,                                                    \
              typename... PreviousArgs>                                       \
    static void Compute(KernelContext* ctx, PreviousArgs&... pargs) {         \
      const std::pair<int, int> range = ctx->OutputRangeAt(out_idx);          \
      std::vector<tensor_type*> arg = std::move(                              \
          ctx->MutableOutputBetween<tensor_type>(range.first, range.second)); \
      KernelCallHelper<Tail...>::                                             \
          template Compute<dev_ctx_idx, in_idx, attr_idx, out_idx + 1>(       \
              ctx, pargs..., arg);                                            \
    }                                                                         \
  }

template <typename T>
struct TypeTag {};

template <typename Fn, Fn fn>
struct KernelImpl;

template <typename Return,
          typename DevCtx,
          typename... Args,
          Return (*kernel_fn)(DevCtx, Args...)>
struct KernelImpl<Return (*)(DevCtx, Args...), kernel_fn> {
  static void Compute(KernelContext* ctx) {
    KernelCallHelper<DevCtx,
                     Args...,
                     TypeTag<int>>::template Compute<0, 0, 0, 0>(ctx);
  }

  static void VariadicCompute(const DeviceContext& dev_ctx, Args... args) {
    return kernel_fn(static_cast<DevCtx>(dev_ctx), std::forward<Args>(args)...);
  }

 private:
  template <typename... RemainingArgs>
  struct KernelCallHelper;

  /* DeviceContext Helpers */

  PD_SPECIALIZE_KernelCallHelper_FOR_DEVICE_CONTEXT(CPUContext);
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  PD_SPECIALIZE_KernelCallHelper_FOR_DEVICE_CONTEXT(GPUContext);
#endif
#ifdef PADDLE_WITH_XPU
  PD_SPECIALIZE_KernelCallHelper_FOR_DEVICE_CONTEXT(XPUContext);
#endif
#ifdef PADDLE_WITH_CUSTOM_DEVICE
  PD_SPECIALIZE_KernelCallHelper_FOR_DEVICE_CONTEXT(CustomContext);
#endif

  /* Input Helpers */

  PD_SPECIALIZE_KernelCallHelper_FOR_INPUT(DenseTensor);
  PD_SPECIALIZE_KernelCallHelper_FOR_OPTIONAL_INPUT(DenseTensor);
  PD_SPECIALIZE_KernelCallHelper_FOR_OPTIONAL_INPUT(SelectedRows);
  PD_SPECIALIZE_KernelCallHelper_FOR_MULTI_INPUT(DenseTensor);
  PD_SPECIALIZE_KernelCallHelper_FOR_INPUT(SelectedRows);

  PD_SPECIALIZE_KernelCallHelper_FOR_INPUT(SparseCooTensor);
  PD_SPECIALIZE_KernelCallHelper_FOR_OPTIONAL_INPUT(SparseCooTensor);
  PD_SPECIALIZE_KernelCallHelper_FOR_MULTI_INPUT(SparseCooTensor);

  PD_SPECIALIZE_KernelCallHelper_FOR_INPUT(SparseCsrTensor);
  PD_SPECIALIZE_KernelCallHelper_FOR_OPTIONAL_INPUT(SparseCsrTensor);
  PD_SPECIALIZE_KernelCallHelper_FOR_MULTI_INPUT(SparseCsrTensor);

  PD_SPECIALIZE_KernelCallHelper_FOR_INPUT(StringTensor);
  PD_SPECIALIZE_KernelCallHelper_FOR_OPTIONAL_INPUT(StringTensor);
  PD_SPECIALIZE_KernelCallHelper_FOR_MULTI_INPUT(StringTensor);

  /* Attribute Helpers */

  PD_SPECIALIZE_KernelCallHelper_FOR_ATTRIBUTE(bool);
  PD_SPECIALIZE_KernelCallHelper_FOR_ATTRIBUTE(float);
  PD_SPECIALIZE_KernelCallHelper_FOR_ATTRIBUTE(double);
  PD_SPECIALIZE_KernelCallHelper_FOR_ATTRIBUTE(int);
  PD_SPECIALIZE_KernelCallHelper_FOR_ATTRIBUTE(int64_t);
  PD_SPECIALIZE_KernelCallHelper_FOR_ATTRIBUTE(phi::dtype::float16);
  PD_SPECIALIZE_KernelCallHelper_FOR_ATTRIBUTE(const Scalar&);
  PD_SPECIALIZE_KernelCallHelper_FOR_ATTRIBUTE(DataType);
  PD_SPECIALIZE_KernelCallHelper_FOR_ATTRIBUTE(DataLayout);
  PD_SPECIALIZE_KernelCallHelper_FOR_ATTRIBUTE(Place);
  PD_SPECIALIZE_KernelCallHelper_FOR_ATTRIBUTE(const std::vector<int64_t>&);
  PD_SPECIALIZE_KernelCallHelper_FOR_ATTRIBUTE(const ScalarArray&);
  PD_SPECIALIZE_KernelCallHelper_FOR_ATTRIBUTE(const std::vector<int>&);
  PD_SPECIALIZE_KernelCallHelper_FOR_ATTRIBUTE(const std::string&);
  PD_SPECIALIZE_KernelCallHelper_FOR_ATTRIBUTE(const std::vector<bool>&);
  PD_SPECIALIZE_KernelCallHelper_FOR_ATTRIBUTE(const std::vector<float>&);
  PD_SPECIALIZE_KernelCallHelper_FOR_ATTRIBUTE(const std::vector<double>&);
  PD_SPECIALIZE_KernelCallHelper_FOR_ATTRIBUTE(const std::vector<std::string>&);
  PD_SPECIALIZE_KernelCallHelper_FOR_ATTRIBUTE(const std::vector<Scalar>&);

  /* Output Helpers */

  PD_SPECIALIZE_KernelCallHelper_FOR_OUTPUT(DenseTensor);
  PD_SPECIALIZE_KernelCallHelper_FOR_MULTI_OUTPUT(DenseTensor);
  PD_SPECIALIZE_KernelCallHelper_FOR_OUTPUT(SelectedRows);

  PD_SPECIALIZE_KernelCallHelper_FOR_OUTPUT(SparseCooTensor);
  PD_SPECIALIZE_KernelCallHelper_FOR_MULTI_OUTPUT(SparseCooTensor);

  PD_SPECIALIZE_KernelCallHelper_FOR_OUTPUT(SparseCsrTensor);
  PD_SPECIALIZE_KernelCallHelper_FOR_MULTI_OUTPUT(SparseCsrTensor);

  PD_SPECIALIZE_KernelCallHelper_FOR_OUTPUT(StringTensor);
  PD_SPECIALIZE_KernelCallHelper_FOR_MULTI_OUTPUT(StringTensor);
  /* End case */
  template <typename T>
  struct KernelCallHelper<TypeTag<T>> {
    template <int dev_ctx_idx, int in_idx, int attr_idx, int out_idx>
    static void Compute(KernelContext* ctx, DevCtx dev_ctx, Args&... args) {
      static_assert(dev_ctx_idx > 0,
                    "Kernel should pass DeviceContext as argument.");
      static_assert(out_idx > 0, "Kernel should have output argument.");
      // TODO(chenweihang): check dev_ctx, in, attr, out number
      return kernel_fn(dev_ctx, args...);
    }
  };
};

}  // namespace phi
