/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. */

#pragma once

#include <memory>
#include <tuple>
#include <type_traits>
#include "paddle/fluid/operators/jitkernels/kernel_base.h"
#include "paddle/fluid/operators/jitkernels/kernels.h"
#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace operators {
namespace jitkernels {

// make_unique is supported from c++14
template <typename T, typename... Args>
inline std::unique_ptr<T> make_unique(Args&&... args) {
  static_assert(!std::is_array<T>::value, "T must not be array");
  return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

template <typename PlaceType, bool IsEnd, size_t I, typename... KernelImpls>
struct JitKernelRegistrarFunctor;

template <typename PlaceType, size_t I, typename... KernelImpls>
struct JitKernelRegistrarFunctor<PlaceType, true, I, KernelImpls...> {
  void operator()(KernelType kt) const {}
};

template <typename PlaceType, size_t I, typename... KernelImpls>
struct JitKernelRegistrarFunctor<PlaceType, false, I, KernelImpls...> {
  using KERNEL_IMPL_TYPE =
      typename std::tuple_element<I, std::tuple<KernelImpls...>>::type;

  void operator()(KernelType kt) const {
    KernelKey kkey(kt, PlaceType());
    KernelPool().Instance().Insert(
        kkey, std::move(make_unique<const KERNEL_IMPL_TYPE>()));
    constexpr auto size = std::tuple_size<std::tuple<KernelImpls...>>::value;
    JitKernelRegistrarFunctor<PlaceType, I + 1 == size, I + 1, KernelImpls...>
        func;
    func(kt);
  }
};

template <typename PlaceType, typename... KernelImpls>
class JitKernelRegistrar {
 public:
  explicit JitKernelRegistrar(KernelType kt) {
    JitKernelRegistrarFunctor<PlaceType, false, 0, KernelImpls...> func;
    func(kt);
  }
};

#define STATIC_ASSERT_JITKERNEL_GLOBAL_NAMESPACE(uniq_name, msg)              \
  struct __test_global_namespace_##uniq_name##__ {};                          \
  static_assert(std::is_same<::__test_global_namespace_##uniq_name##__,       \
                             __test_global_namespace_##uniq_name##__>::value, \
                msg)

// kernel_type: should be in paddle::operators::jitkernels::KernelType
// place_type: should be one of CPUPlace and GPUPlace in paddle::platform
#define REGISTER_KERNEL_MORE(kernel_type, impl_type, place_type, ...)        \
  STATIC_ASSERT_JITKERNEL_GLOBAL_NAMESPACE(                                  \
      __reg_jitkernel_##kernel_type##_##impl_type##_##place_type,            \
      "REGISTER_KERNEL_MORE must be called in global namespace");            \
  static ::paddle::operators::jitkernels::JitKernelRegistrar<                \
      ::paddle::platform::place_type, __VA_ARGS__>                           \
      __jit_kernel_registrar_##kernel_type##_##impl_type##_##place_type##__( \
          ::paddle::operators::jitkernels::KernelType::kernel_type)
// TODO(TJ): Add Touch and use me

#define REGISTER_JITKERNEL_MORE(kernel_type, impl_type, ...) \
  REGISTER_KERNEL_MORE(kernel_type, impl_type, CPUPlace, __VA_ARGS__)

#define REGISTER_GPUKERNEL_MORE(kernel_type, impl_type, ...) \
  REGISTER_KERNEL_MORE(kernel_type, impl_type, GPUPlace, __VA_ARGS__)

/*
REGISTER_JITKERNEL_JITCODE(vmul, JitKernelCode<vmul, int>);

// refer must be only one and at least one
REGISTER_JITKERNEL_REFER(vmul, VMul);  // Refer need support dtype

// you can register more implementations and the condition when use it
REGISTER_JITKERNEL_MORE(vmul, mkl::VMUL<float>, UseMe<float>, mkl::VMUL<double>,
                        UseMe<double>)

#define STATIC_ASSERT_PASS_GLOBAL_NAMESPACE(uniq_name, msg)                   \
  struct __test_global_namespace_##uniq_name##__ {};                          \
  static_assert(std::is_same<::__test_global_namespace_##uniq_name##__,       \
                             __test_global_namespace_##uniq_name##__>::value, \
                msg)

// Register a new pass that can be applied on the IR.
#define REGISTER_PASS(pass_type, pass_class)                 \
  STATIC_ASSERT_PASS_GLOBAL_NAMESPACE(                       \
      __reg_pass__##pass_type,                               \
      "REGISTER_PASS must be called in global namespace");   \
  static ::paddle::framework::ir::PassRegistrar<pass_class>  \
      __pass_registrar_##pass_type##__(#pass_type);          \
  int TouchPassRegistrar_##pass_type() {                     \
    __pass_registrar_##pass_type##__.Touch();                \
    return 0;                                                \
  }                                                          \
  static ::paddle::framework::ir::PassRegistrar<pass_class>& \
      __pass_tmp_registrar_##pass_type##__ UNUSED =          \
          __pass_registrar_##pass_type##__

#define USE_PASS(pass_type)                           \
  STATIC_ASSERT_PASS_GLOBAL_NAMESPACE(                \
      __use_pass_itself_##pass_type,                  \
      "USE_PASS must be called in global namespace"); \
  extern int TouchPassRegistrar_##pass_type();        \
  static int use_pass_itself_##pass_type##_ UNUSED =  \
      TouchPassRegistrar_##pass_type()
*/

}  // namespace jitkernels
}  // namespace operators
}  // namespace paddle
