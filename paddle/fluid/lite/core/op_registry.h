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
#include <list>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include "paddle/fluid/lite/api/paddle_lite_factory_helper.h"
#include "paddle/fluid/lite/core/kernel.h"
#include "paddle/fluid/lite/core/op_lite.h"
#include "paddle/fluid/lite/core/target_wrapper.h"
#include "paddle/fluid/lite/utils/all.h"

using LiteType = paddle::lite::Type;

namespace paddle {
namespace lite {

using KernelFunc = std::function<void()>;
using KernelFuncCreator = std::function<std::unique_ptr<KernelFunc>()>;
class LiteOpRegistry final : public Factory<OpLite, std::shared_ptr<OpLite>> {
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
  explicit OpLiteRegistor(const std::string &op_type)
      : Registor<OpClass>([&] {
          LiteOpRegistry::Global().Register(
              op_type, [op_type]() -> std::unique_ptr<OpLite> {
                return std::unique_ptr<OpLite>(new OpClass(op_type));
              });
        }) {}
};

template <TargetType Target, PrecisionType Precision, DataLayoutType Layout>
using KernelRegistryForTarget =
    Factory<KernelLite<Target, Precision, Layout>, std::unique_ptr<KernelBase>>;

class KernelRegistry final {
 public:
  using any_kernel_registor_t =
      variant<KernelRegistryForTarget<TARGET(kCUDA), PRECISION(kFloat),
                                      DATALAYOUT(kNCHW)> *,  //
              KernelRegistryForTarget<TARGET(kCUDA), PRECISION(kInt8),
                                      DATALAYOUT(kNCHW)> *,  //
              KernelRegistryForTarget<TARGET(kX86), PRECISION(kFloat),
                                      DATALAYOUT(kNCHW)> *,  //
              KernelRegistryForTarget<TARGET(kX86), PRECISION(kInt64),
                                      DATALAYOUT(kNCHW)> *,  //
              KernelRegistryForTarget<TARGET(kX86), PRECISION(kInt8),
                                      DATALAYOUT(kNCHW)> *,  //
              KernelRegistryForTarget<TARGET(kHost), PRECISION(kFloat),
                                      DATALAYOUT(kNCHW)> *,  //
              KernelRegistryForTarget<TARGET(kHost), PRECISION(kAny),
                                      DATALAYOUT(kAny)> *,  //
              KernelRegistryForTarget<TARGET(kCUDA), PRECISION(kAny),
                                      DATALAYOUT(kAny)> *,  //
              KernelRegistryForTarget<TARGET(kARM), PRECISION(kAny),
                                      DATALAYOUT(kAny)> *,  //
              KernelRegistryForTarget<TARGET(kARM), PRECISION(kFloat),
                                      DATALAYOUT(kNCHW)> *,  //
              KernelRegistryForTarget<TARGET(kARM), PRECISION(kInt8),
                                      DATALAYOUT(kNCHW)> *,  //
              KernelRegistryForTarget<TARGET(kOpenCL), PRECISION(kFloat),
                                      DATALAYOUT(kNCHW)> *,  //
              KernelRegistryForTarget<TARGET(kOpenCL), PRECISION(kInt8),
                                      DATALAYOUT(kNCHW)> *  //
              >;

  KernelRegistry();

  static KernelRegistry &Global();

  template <TargetType Target, PrecisionType Precision, DataLayoutType Layout>
  void Register(const std::string &name,
                typename KernelRegistryForTarget<Target, Precision,
                                                 Layout>::creator_t &&creator) {
    /*VLOG(3) << "register for " << TargetToStr(Target) << ":"
            << PrecisionToStr(Precision) << "//"
            << GetKernelOffset<Target, Precision, Layout>();*/
    using kernel_registor_t =
        KernelRegistryForTarget<Target, Precision, Layout>;
    auto &varient = registries_[GetKernelOffset<Target, Precision, Layout>()];
    auto *reg = varient.template get<kernel_registor_t *>();
    CHECK(reg) << "Can not be empty of " << name;
    reg->Register(name, std::move(creator));
  }

  template <TargetType Target, PrecisionType Precision = PRECISION(kFloat),
            DataLayoutType Layout = DATALAYOUT(kNCHW)>
  std::list<std::unique_ptr<KernelBase>> Create(const std::string &op_type) {
    using kernel_registor_t =
        KernelRegistryForTarget<Target, Precision, Layout>;
    return registries_[GetKernelOffset<Target, Precision, Layout>()]
        .template get<kernel_registor_t *>()
        ->Creates(op_type);
  }

  std::list<std::unique_ptr<KernelBase>> Create(const std::string &op_type,
                                                TargetType target,
                                                PrecisionType precision,
                                                DataLayoutType layout);

  // Get a kernel registry offset in all the registries.
  template <TargetType Target, PrecisionType Precision, DataLayoutType Layout>
  static int GetKernelOffset() {
    CHECK_LT(static_cast<int>(Target), static_cast<int>(TARGET(NUM)));
    CHECK_LT(static_cast<int>(Precision), static_cast<int>(PRECISION(NUM)));
    CHECK_LT(static_cast<int>(Layout), static_cast<int>(DATALAYOUT(NUM)));
    return static_cast<int>(Target) * static_cast<int>(PRECISION(NUM)) *
               static_cast<int>(DATALAYOUT(NUM)) +                            //
           static_cast<int>(Precision) * static_cast<int>(DATALAYOUT(NUM)) +  //
           static_cast<int>(Layout);
  }

  std::string DebugString() const {
    std::stringstream ss;
    ss << "KernelCreator<host, float>:" << std::endl;
    constexpr TargetType tgt = TARGET(kHost);
    constexpr PrecisionType dt = PRECISION(kFloat);
    constexpr DataLayoutType lt = DATALAYOUT(kNCHW);
    constexpr DataLayoutType kany = DATALAYOUT(kAny);
    using kernel_registor_t = KernelRegistryForTarget<tgt, dt, lt>;
    auto *reg = registries_[GetKernelOffset<tgt, dt, kany>()]
                    .template get<kernel_registor_t *>();
    ss << reg->DebugString() << std::endl;
    return ss.str();
  }

 private:
  mutable std::vector<any_kernel_registor_t> registries_;
};

template <TargetType target, PrecisionType precision, DataLayoutType layout,
          typename KernelType>
class KernelRegistor : public lite::Registor<KernelType> {
 public:
  KernelRegistor(const std::string &op_type, const std::string &alias)
      : Registor<KernelType>([=] {
          /*VLOG(3) << "Register kernel " << op_type << " for "
                  << TargetToStr(target) << " " << PrecisionToStr(precision)
                  << " " << DataLayoutToStr(layout) << " alias " << alias;*/
          KernelRegistry::Global().Register<target, precision, layout>(
              op_type, [=]() -> std::unique_ptr<KernelType> {
                std::unique_ptr<KernelType> x(new KernelType);
                x->set_op_type(op_type);
                x->set_alias(alias);
                return x;
              });
        }) {}
};

}  // namespace lite
}  // namespace paddle

// Operator registry
#define LITE_OP_REGISTER_INSTANCE(op_type__) op_type__##__registry__instance__
#define REGISTER_LITE_OP(op_type__, OpClass)                              \
  static paddle::lite::OpLiteRegistor<OpClass> LITE_OP_REGISTER_INSTANCE( \
      op_type__)(#op_type__);                                             \
  int touch_op_##op_type__() {                                            \
    return LITE_OP_REGISTER_INSTANCE(op_type__).Touch();                  \
  }

// Kernel registry
#define LITE_KERNEL_REGISTER(op_type__, target__, precision__) \
  op_type__##__##target__##__##precision__##__registor__
#define LITE_KERNEL_REGISTER_INSTANCE(op_type__, target__, precision__, \
                                      layout__, alias__)                \
  op_type__##__##target__##__##precision__##__registor__instance__##alias__
#define LITE_KERNEL_REGISTER_FAKE(op_type__, target__, precision__, alias__) \
  LITE_KERNEL_REGISTER_INSTANCE(op_type__, target__, precision__, alias__)

#define REGISTER_LITE_KERNEL(op_type__, target__, precision__, layout__,      \
                             KernelClass, alias__)                            \
  static paddle::lite::KernelRegistor<TARGET(target__),                       \
                                      PRECISION(precision__),                 \
                                      DATALAYOUT(layout__), KernelClass>      \
      LITE_KERNEL_REGISTER_INSTANCE(op_type__, target__, precision__,         \
                                    layout__, alias__)(#op_type__, #alias__); \
  static KernelClass LITE_KERNEL_INSTANCE(op_type__, target__, precision__,   \
                                          layout__, alias__);                 \
  int touch_##op_type__##target__##precision__##layout__##alias__() {         \
    LITE_KERNEL_INSTANCE(op_type__, target__, precision__, layout__, alias__) \
        .Touch();                                                             \
    return 0;                                                                 \
  }                                                                           \
  static bool LITE_KERNEL_PARAM_INSTANCE(op_type__, target__, precision__,    \
                                         layout__, alias__)                   \
      __attribute__((unused)) = paddle::lite::ParamTypeRegistry::NewInstance< \
          TARGET(target__), PRECISION(precision__), DATALAYOUT(layout__)>(    \
          #op_type__ "/" #alias__)

#define LITE_KERNEL_INSTANCE(op_type__, target__, precision__, layout__, \
                             alias__)                                    \
  op_type__##target__##precision__##layout__##alias__
#define LITE_KERNEL_PARAM_INSTANCE(op_type__, target__, precision__, layout__, \
                                   alias__)                                    \
  op_type__##target__##precision__##layout__##alias__##param_register
