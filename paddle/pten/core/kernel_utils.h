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

#include "paddle/pten/backends/all_context.h"
#include "paddle/pten/common/scalar.h"
#include "paddle/pten/common/scalar_array.h"
#include "paddle/pten/core/dense_tensor.h"
#include "paddle/pten/core/kernel_context.h"
#include "paddle/pten/core/kernel_def.h"

// See Note [ Why still include the fluid headers? ]
#include "paddle/fluid/platform/enforce.h"

namespace pten {

#define PT_KERNEL(...) \
  ::pten::KernelImpl<decltype(&__VA_ARGS__), &__VA_ARGS__>::Compute

#define PT_VARIADIC_KERNEL(...)                                       \
  reinterpret_cast<void*>(&::pten::KernelImpl<decltype(&__VA_ARGS__), \
                                              &__VA_ARGS__>::VariadicCompute)

#define PT_SPECIALIZE_KernelCallHelper_FOR_DEVICE_CONTEXT(dev_ctx)           \
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

#define PT_SPECIALIZE_KernelCallHelper_FOR_INPUT(tensor_type)           \
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

#define PT_SPECIALIZE_KernelCallHelper_FOR_MULTI_INPUT(tensor_type)        \
  template <typename... Tail>                                              \
  struct KernelCallHelper<const std::vector<tensor_type>&, Tail...> {      \
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
      std::vector<tensor_type> arg = std::move(                            \
          ctx->MoveInputsBetween<tensor_type>(range.first, range.second)); \
      KernelCallHelper<Tail...>::                                          \
          template Compute<dev_ctx_idx, in_idx + 1, attr_idx, out_idx>(    \
              ctx, pargs..., arg);                                         \
    }                                                                      \
  }

#define PT_SPECIALIZE_KernelCallHelper_FOR_ATTRIBUTE(attr_type)           \
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

#define PT_SPECIALIZE_KernelCallHelper_FOR_OUTPUT(tensor_type)           \
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

#define PT_SPECIALIZE_KernelCallHelper_FOR_MULTI_OUTPUT(tensor_type)          \
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

  PT_SPECIALIZE_KernelCallHelper_FOR_DEVICE_CONTEXT(CPUContext);
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  PT_SPECIALIZE_KernelCallHelper_FOR_DEVICE_CONTEXT(GPUContext);
#endif
#ifdef PADDLE_WITH_XPU
  PT_SPECIALIZE_KernelCallHelper_FOR_DEVICE_CONTEXT(XPUContext);
#endif

  /* Input Helpers */

  PT_SPECIALIZE_KernelCallHelper_FOR_INPUT(DenseTensor);
  PT_SPECIALIZE_KernelCallHelper_FOR_MULTI_INPUT(DenseTensor);
  // TODO(chenweihang): adapt SelectedRows
  // PT_SPECIALIZE_KernelCallHelper_FOR_INPUT(SelectedRowsTensor);

  /* Attribute Helpers */

  PT_SPECIALIZE_KernelCallHelper_FOR_ATTRIBUTE(bool);
  PT_SPECIALIZE_KernelCallHelper_FOR_ATTRIBUTE(float);
  PT_SPECIALIZE_KernelCallHelper_FOR_ATTRIBUTE(double);
  PT_SPECIALIZE_KernelCallHelper_FOR_ATTRIBUTE(int);
  PT_SPECIALIZE_KernelCallHelper_FOR_ATTRIBUTE(int64_t);
  PT_SPECIALIZE_KernelCallHelper_FOR_ATTRIBUTE(paddle::platform::float16);
  PT_SPECIALIZE_KernelCallHelper_FOR_ATTRIBUTE(const Scalar&);
  PT_SPECIALIZE_KernelCallHelper_FOR_ATTRIBUTE(DataType);
  PT_SPECIALIZE_KernelCallHelper_FOR_ATTRIBUTE(const std::vector<int64_t>&);
  PT_SPECIALIZE_KernelCallHelper_FOR_ATTRIBUTE(const ScalarArray&);
  PT_SPECIALIZE_KernelCallHelper_FOR_ATTRIBUTE(const std::vector<int>&);

  /* Output Helpers */

  PT_SPECIALIZE_KernelCallHelper_FOR_OUTPUT(DenseTensor);
  PT_SPECIALIZE_KernelCallHelper_FOR_MULTI_OUTPUT(DenseTensor);
  // TODO(chenweihang): adapt SelectedRows
  // PT_SPECIALIZE_KernelCallHelper_FOR_OUTPUT(SelectedRowsTensor);

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

}  // namespace pten
