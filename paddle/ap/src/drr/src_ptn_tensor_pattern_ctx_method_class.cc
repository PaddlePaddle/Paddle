// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/ap/include/drr/src_ptn_tensor_pattern_ctx_method_class.h"

namespace ap::drr {

struct SrcPtnTensorPatternCtx {
  using This = SrcPtnTensorPatternCtx;
  using ObjT = drr::tSrcPtn<drr::TensorPatternCtx>;
  using Self = ObjT;
  using Helper = OpTensorPatternCtxHelper;

  static adt::Result<axpr::Value> ToString(
      const axpr::Value& self_val, const std::vector<axpr::Value>& args) {
    ADT_LET_CONST_REF(self, self_val.template CastTo<Self>());
    std::ostringstream ss;
    const void* ptr = self.value().__adt_rc_shared_ptr_raw_ptr();
    ss << "<" << drr::Type<Self>{}.Name() << " object at " << ptr << ">";
    return ss.str();
  }

  static adt::Result<axpr::Value> Hash(const axpr::Value& self_val,
                                       const std::vector<axpr::Value>& args) {
    ADT_LET_CONST_REF(self, self_val.template CastTo<Self>());
    const void* ptr = self.value().__adt_rc_shared_ptr_raw_ptr();
    return reinterpret_cast<int64_t>(ptr);
  }

  static adt::Result<axpr::Value> GetAttr(
      const axpr::Value& self_val, const std::vector<axpr::Value>& args) {
    return This::GetOrCreateTensor(self_val, args);
  }

  static adt::Result<axpr::Value> GetOrCreateTensor(
      const axpr::Value& self_val, const std::vector<axpr::Value>& args) {
    ADT_CHECK(args.size() == 1);
    const auto& arg = args.at(0);
    ADT_LET_CONST_REF(self, self_val.template CastTo<Self>());
    if (arg.template Has<adt::Nothing>()) {
      return adt::Nothing{};
    }
    ADT_LET_CONST_REF(tensor_name, arg.template CastTo<std::string>());

    const auto& opt_ir_value =
        Helper{}.GetIrValueByUid(self.value(), tensor_name);
    if (opt_ir_value.HasError()) {
      UnboundIrValue<drr::Node> unbound_ir_value{tensor_name,
                                                 self.value().shared_ptr()};
      return DrrValueHelper{}.CastToAxprValue(unbound_ir_value);
    }
    const auto& ir_value = opt_ir_value.GetOkValue();
    const auto& drr_value = ir_value.Match(
        [](const auto& impl) -> DrrValue { return SrcPtn(impl); });
    return DrrValueHelper{}.CastToAxprValue(drr_value);
  }
};

axpr::TypeImpl<axpr::BuiltinClassInstance<axpr::Value>>
GetSrcPtnTensorPatternCtxClass() {
  using Impl = drr::SrcPtnTensorPatternCtx;
  using TT = drr::Type<drr::tSrcPtn<drr::TensorPatternCtx>>;
  static auto cls(
      axpr::MakeBuiltinClass<axpr::Value>(TT{}.Name(), [&](const auto& Define) {
        Define("__str__", &Impl::ToString);
        Define("__hash__", &Impl::Hash);
        Define("__getattr__", &Impl::GetAttr);
        Define("get_or_create_tensor", &Impl::GetOrCreateTensor);
      }));
  using Self = typename Impl::Self;
  return axpr::MakeGlobalNaiveClassOps<Self>(cls);
}

}  // namespace ap::drr
