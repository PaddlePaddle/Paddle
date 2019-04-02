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
              op_type, []() -> std::unique_ptr<OpLite> {
                return std::unique_ptr<OpLite>(new OpClass);
              });
        }) {}
};

template <TargetType Target, PrecisionType Precision>
using KernelRegistryForTarget = Factory<OpKernel<Target, Precision>>;

class KernelRegistry final {
 public:
  using any_kernel_registor_t = variant<
      KernelRegistryForTarget<TargetType::kCUDA, PrecisionType::kFloat> *,  //
      KernelRegistryForTarget<TargetType::kCUDA, PrecisionType::kInt8> *,   //
      KernelRegistryForTarget<TargetType::kX86, PrecisionType::kFloat> *,   //
      KernelRegistryForTarget<TargetType::kX86, PrecisionType::kInt8> *,    //
      KernelRegistryForTarget<TargetType::kARM, PrecisionType::kFloat> *,   //
      KernelRegistryForTarget<TargetType::kHost, PrecisionType::kFloat> *   //
      >;

  KernelRegistry() {
/*
using kernel_target_t =
    KernelRegistryForTarget<TARGET(kCUDA), PRECISION(kFloat)>;
registries_[0].set<kernel_target_t *>(
    &KernelRegistryForTarget<TARGET(kCUDA), PRECISION(kFloat)>::Global());
    */
#define INIT_FOR(target__, precision__)                                      \
  registries_[KernelRegistry::GetKernelOffset<TARGET(target__),              \
                                              PRECISION(precision__)>()]     \
      .set<KernelRegistryForTarget<TARGET(target__), PRECISION(precision__)> \
               *>(&KernelRegistryForTarget<TARGET(target__),                 \
                                           PRECISION(precision__)>::Global());
    // Currently, just register 2 kernel targets.
    INIT_FOR(kARM, kFloat);
    INIT_FOR(kHost, kFloat);
#undef INIT_FOR
  }

  static KernelRegistry &Global() {
    static auto *x = new KernelRegistry;
    return *x;
  }

  template <TargetType Target, PrecisionType Precision>
  void Register(const std::string &name,
                typename KernelRegistryForTarget<Target, Precision>::creator_t
                    &&creator) {
    using kernel_registor_t = KernelRegistryForTarget<Target, Precision>;
    registries_[GetKernelOffset<Target, Precision>()]
        .template get<kernel_registor_t *>()
        ->Register(name, std::move(creator));
  }

  // Get a kernel registry offset in all the registries.
  template <TargetType Target, PrecisionType Precision>
  static constexpr int GetKernelOffset() {
    return kNumTargets * static_cast<int>(Target) + static_cast<int>(Precision);
  }

 private:
  std::array<any_kernel_registor_t, kNumTargets * kNumPrecisions> registries_;
};

template <TargetType target, PrecisionType precision, typename KernelType>
class KernelRegistor : public lite::Registor<KernelType> {
 public:
  KernelRegistor(const std::string op_type)
      : Registor<KernelType>([&] {
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
      op_type__)(#op_type__);

#define USE_LITE_OP(op_type__)                     \
  int LITE_OP_REGISTER_FAKE(op_type__)((unused)) = \
      LITE_OP_REGISTER_INSTANCE(op_type__).Touch();

// Kernel registry
#define LITE_KERNEL_REGISTER(op_type__, target__, precision__) \
  op_type__##target__##precision__##__registor__
#define LITE_KERNEL_REGISTER_INSTANCE(op_type__, target__, precision__) \
  op_type__##target__##precision__##__registor__instance__
#define LITE_KERNEL_REGISTER_FAKE(op_type__, target__, precision__) \
  LITE_KERNEL_REGISTER_INSTANCE(op_type__, target__, precision__)##__fake__

#define REGISTER_LITE_KERNEL(op_type__, target__, precision__, KernelClass) \
  static paddle::lite::KernelRegistor<TARGET(target__),                     \
                                      PRECISION(precision__), KernelClass>  \
      LITE_KERNEL_REGISTER_INSTANCE(op_type__, target__,                    \
                                    precision__)(#op_type__);

#define USE_LITE_KERNEL(op_type__, target__, precision__)                     \
  int LITE_KERNEL_REGISTER_FAKE(op_type__, target__, precision__)((unused)) = \
      LITE_KERNEL_REGISTER(op_type__, target__, precision__).Touch();
