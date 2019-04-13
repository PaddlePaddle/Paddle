// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
#include <memory>
#include <string>
#include <unordered_map>
#include "paddle/fluid/lite/core/kernel.h"
#include "paddle/fluid/lite/core/op_lite.h"
#include "paddle/fluid/lite/core/target_wrapper.h"
#include "paddle/fluid/lite/utils/all.h"

namespace paddle {
namespace lite {

using KernelFunc = std::function<void()>;
using KernelFuncCreator = std::function<std::unique_ptr<KernelFunc>()>;

class LiteOpRegistry final : public Factory<OpLite> {
 public:
  static LiteOpRegistry &Global() {
    static auto *x = new LiteOpRegistry;
    return *x;
  }

 private:
  LiteOpRegistry() = default;
};

template <typename OpClass>
class OpLiteRegistor : public Registor<OpClass> {
 public:
  OpLiteRegistor(const std::string &op_type)
      : Registor<OpClass>([&] {
          LiteOpRegistry::Global().Register(
              op_type, [op_type]() -> std::unique_ptr<OpLite> {
                return std::unique_ptr<OpLite>(new OpClass(op_type));
              });
        }) {}
};

template <TargetType Target, PrecisionType Precision>
using KernelRegistryForTarget = Factory<OpKernel<Target, Precision>>;

class KernelRegistry final {
 public:
  using any_kernel_registor_t =
      variant<KernelRegistryForTarget<TARGET(kCUDA), PRECISION(kFloat)> *,  //
              KernelRegistryForTarget<TARGET(kCUDA), PRECISION(kInt8)> *,   //
              KernelRegistryForTarget<TARGET(kX86), PRECISION(kFloat)> *,   //
              KernelRegistryForTarget<TARGET(kX86), PRECISION(kInt8)> *,    //
              KernelRegistryForTarget<TARGET(kHost), PRECISION(kFloat)> *   //
              >;

  KernelRegistry();

  static KernelRegistry &Global();

  template <TargetType Target, PrecisionType Precision>
  void Register(const std::string &name,
                typename KernelRegistryForTarget<Target, Precision>::creator_t
                    &&creator) {
    using kernel_registor_t = KernelRegistryForTarget<Target, Precision>;
    registries_[GetKernelOffset<Target, Precision>()]
        .template get<kernel_registor_t *>()
        ->Register(name, std::move(creator));
  }

  template <TargetType Target, PrecisionType Precision>
  std::unique_ptr<KernelBase> Create(const std::string &op_type) {
    using kernel_registor_t = KernelRegistryForTarget<Target, Precision>;
    return registries_[GetKernelOffset<Target, Precision>()]
        .template get<kernel_registor_t *>()
        ->Create(op_type);
  }

  std::unique_ptr<KernelBase> Create(const std::string &op_type,
                                     TargetType target,
                                     PrecisionType precision);

  // Get a kernel registry offset in all the registries.
  template <TargetType Target, PrecisionType Precision>
  static constexpr int GetKernelOffset() {
    return kNumTargets * static_cast<int>(Target) + static_cast<int>(Precision);
  }

  std::string DebugString() const {
    std::stringstream ss;

    ss << "KernelCreator<host, float>:" << std::endl;
    ss << registries_[GetKernelOffset<TARGET(kHost), PRECISION(kFloat)>()]
              .get<
                  KernelRegistryForTarget<TARGET(kHost), PRECISION(kFloat)> *>()
              ->DebugString();
    ss << std::endl;
    return ss.str();
  }

 private:
  mutable std::array<any_kernel_registor_t, kNumTargets * kNumPrecisions>
      registries_;
};

template <TargetType target, PrecisionType precision, typename KernelType>
class KernelRegistor : public lite::Registor<KernelType> {
 public:
  KernelRegistor(const std::string op_type)
      : Registor<KernelType>([&] {
          LOG(INFO) << "Register kernel " << op_type << " for "
                    << TargetToStr(target) << " " << PrecisionToStr(precision);
          KernelRegistry::Global().Register<target, precision>(
              op_type, [&]() -> std::unique_ptr<KernelType> {
                return std::unique_ptr<KernelType>(new KernelType);
              });
        }) {}
};

}  // namespace lite
}  // namespace paddle

// Operator registry
#define LITE_OP_REGISTER_INSTANCE(op_type__) op_type__##__registry__instance__
#define LITE_OP_REGISTER_FAKE(op_type__) op_type__##__registry__
#define REGISTER_LITE_OP(op_type__, OpClass)                              \
  static paddle::lite::OpLiteRegistor<OpClass> LITE_OP_REGISTER_INSTANCE( \
      op_type__)(#op_type__);                                             \
  int touch_op_##op_type__() {                                            \
    return LITE_OP_REGISTER_INSTANCE(op_type__).Touch();                  \
  }

#define USE_LITE_OP(op_type__)                                   \
  extern int touch_op_##op_type__();                             \
  int LITE_OP_REGISTER_FAKE(op_type__) __attribute__((unused)) = \
      touch_op_##op_type__();

// Kernel registry
#define LITE_KERNEL_REGISTER(op_type__, target__, precision__) \
  op_type__##__##target__##__##precision__##__registor__
#define LITE_KERNEL_REGISTER_INSTANCE(op_type__, target__, precision__) \
  op_type__##__##target__##__##precision__##__registor__instance__
#define LITE_KERNEL_REGISTER_FAKE(op_type__, target__, precision__) \
  LITE_KERNEL_REGISTER_INSTANCE(op_type__, target__, precision__)

#define REGISTER_LITE_KERNEL(op_type__, target__, precision__, KernelClass)   \
  static paddle::lite::KernelRegistor<TARGET(target__),                       \
                                      PRECISION(precision__), KernelClass>    \
      LITE_KERNEL_REGISTER_INSTANCE(op_type__, target__,                      \
                                    precision__)(#op_type__);                 \
  static KernelClass LITE_KERNEL_INSTANCE(op_type__, target__, precision__);  \
  int touch_##op_type__##target__##precision__() {                            \
    LITE_KERNEL_INSTANCE(op_type__, target__, precision__).Touch();           \
    return 0;                                                                 \
  }                                                                           \
  static bool op_type__##target__##precision__##param_register                \
      __attribute__((unused)) = paddle::lite::ParamTypeRegistry::NewInstance< \
          TARGET(target__), PRECISION(precision__)>(#op_type__)

#define USE_LITE_KERNEL(op_type__, target__, precision__)         \
  extern int touch_##op_type__##target__##precision__();          \
  int LITE_KERNEL_REGISTER_FAKE(op_type__, target__, precision__) \
      __attribute__((unused)) = touch_##op_type__##target__##precision__();

#define LITE_KERNEL_INSTANCE(op_type__, target__, precision__) \
  op_type__##target__##precision__
