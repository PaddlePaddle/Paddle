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

#include "paddle/ap/include/drr/src_ptn_op_pattern_ctx_method_class.h"
#include <set>
#include "paddle/ap/include/drr/drr_pass_type_helper.h"

namespace ap::drr {

struct SrcPtnOpPatternCtxMethodClass {
  using This = SrcPtnOpPatternCtxMethodClass;
  using ObjT = tSrcPtn<OpPatternCtx>;
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

  static adt::Result<axpr::Value> SetAttr(
      const axpr::Value& self_val, const std::vector<axpr::Value>& args) {
    ADT_CHECK(args.size() == 2);
    const auto& arg = args.at(0);
    ADT_LET_CONST_REF(attr_name, arg.template CastTo<std::string>());
    ADT_CHECK(!IsBasicAttrName(attr_name)) << adt::errors::AttributeError{
        "can't set attribute '" + attr_name + "'"};
    return MakeAndRegisterUnboundIrOp(self_val, args);
  }

  static adt::Result<axpr::Value> MakeAndRegisterUnboundIrOp(
      const axpr::Value& self_val, const std::vector<axpr::Value>& args) {
    ADT_LET_CONST_REF(self, self_val.template CastTo<Self>());
    ADT_CHECK(args.size() == 2);
    ADT_LET_CONST_REF(op_uid, args.at(0).template CastTo<std::string>());
    const auto& drr_value = DrrValueHelper{}.CastFromAxprValue(args.at(1));
    const auto& opt_ir_op = drr_value.DrrValueMatch(
        [&](const tSrcPtn<PackedIrOpDeclare<drr::Node>>& op)
            -> adt::Result<IrOp> {
          return UnboundPackedIrOp<drr::Node>{op.value(), op_uid};
        },
        [&](const OptPackedIrOpDeclare<drr::Node>& op) -> adt::Result<IrOp> {
          return UnboundOptPackedIrOp<drr::Node>{op, op_uid};
        },
        [&](const tSrcPtn<NativeIrOpDeclare<drr::Node>>& op)
            -> adt::Result<IrOp> {
          return UnboundNativeIrOp<drr::Node>{op.value(), op_uid};
        },
        [&](const auto&) -> adt::Result<IrOp> {
          return adt::errors::TypeError{
              std::string() +
              "only 'SrcPtnPackedIrOpDeclare' and 'SrcPtnNativeIrOpDeclare' "
              "supported for op name binding. '" +
              axpr::GetTypeName(args.at(1)) + "' were given."};
        });
    ADT_LET_CONST_REF(ir_op, opt_ir_op);
    bool has_ir_op = Helper{}.HasIrOpByUid(self.value(), op_uid);
    if (has_ir_op) {
      ADT_RETURN_IF_ERR(
          Helper{}.CheckIrOpNameByUid(self.value(), op_uid, ir_op));
    } else {
      Helper{}.SetIrOpByUid(self.value(), op_uid, ir_op);
    }
    return adt::Nothing{};
  }

  static adt::Result<axpr::Value> GetAttr(
      const axpr::Value& self_val, const std::vector<axpr::Value>& args) {
    ADT_CHECK(args.size() == 1);
    const auto& arg = args.at(0);
    ADT_LET_CONST_REF(self, self_val.template CastTo<Self>());
    ADT_LET_CONST_REF(attr_name, arg.template CastTo<std::string>());
    ADT_CHECK(!IsBasicAttrName(attr_name)) << adt::errors::RuntimeError{
        std::string() + "Dead code encounterred. attr_name: " + attr_name};
    ADT_LET_CONST_REF(ir_op, Helper{}.GetIrOpByUid(self.value(), attr_name));
    const auto& drr_value = ir_op.Match(
        [](const NativeIrOp<drr::Node>& impl) -> DrrValue { return impl; },
        [](const PackedIrOp<drr::Node>& impl) -> DrrValue { return impl; },
        [](const OptPackedIrOp<drr::Node>& impl) -> DrrValue { return impl; },
        [](const UnboundOptPackedIrOp<drr::Node>& impl) -> DrrValue {
          return impl;
        },
        [](const UnboundNativeIrOp<drr::Node>& x) -> DrrValue {
          return SrcPtn(x);
        },
        [](const UnboundPackedIrOp<drr::Node>& x) -> DrrValue {
          return SrcPtn(x);
        });
    return DrrValueHelper{}.CastToAxprValue(drr_value);
  }

  static adt::Result<axpr::Value> DeclareApTrivialFusionOp(
      const axpr::Value& self_val, const std::vector<axpr::Value>& args) {
    ADT_LET_CONST_REF(self, self_val.template CastTo<Self>());
    std::optional<axpr::Function<axpr::SerializableValue>> opt_func;
    if (args.size() == 1) {
      ADT_LET_CONST_REF(
          func,
          args.at(0)
              .template CastTo<axpr::Function<axpr::SerializableValue>>());
      opt_func = func;
    } else {
      ADT_CHECK(args.size() == 0) << adt::errors::TypeError{
          std::string() +
          "SrcPtnOpPatternCtx.ap_trivial_fusion_op takes 1 or 0 arguments. "
          "but " +
          std::to_string(args.size()) + " were given."};
    }
    auto ptr = std::make_shared<SrcPtnPackedIrOpDeclareData>();
    ptr->inner_source_pattern_func = opt_func;
    std::shared_ptr<PackedIrOpDeclareData> op_declare_data{ptr};
    PackedIrOpDeclare<drr::Node> op_declare{
        "ap_trivial_fusion_op", self.value().shared_ptr(), op_declare_data};
    return DrrValueHelper{}.CastToAxprValue(SrcPtn(op_declare));
  }

  static adt::Result<axpr::Value> DeclareOptionalApTrivialFusionOp(
      const axpr::Value& self_val, const std::vector<axpr::Value>& args) {
    ADT_LET_CONST_REF(self, self_val.template CastTo<Self>());
    ADT_LET_CONST_REF(drr_ctx, adt::WeakPtrLock(self.value()->drr_ctx));
    ADT_CHECK(
        DrrPassTypeHelper{}.SupportOptionalPackedOp(drr_ctx->drr_pass_type));
    ADT_CHECK(args.size() == 0)
        << adt::errors::TypeError{std::string() +
                                  "SrcPtnOpPatternCtx.optional_ap_trivial_"
                                  "fusion_op takes 0 arguments. but " +
                                  std::to_string(args.size()) + " were given."};
    OptPackedIrOpDeclare<drr::Node> op_declare{
        "ap_trivial_fusion_op", self.value().shared_ptr(), std::nullopt};
    return DrrValueHelper{}.CastToAxprValue(op_declare);
  }

  static adt::Result<axpr::Value> DeclareNativeIrOp(
      const axpr::Value& self_val, const std::vector<axpr::Value>& args) {
    ADT_LET_CONST_REF(self, self_val.template CastTo<Self>());
    ADT_CHECK(args.size() == 1) << adt::errors::TypeError{
        std::string() +
        "SrcPtnOpPatternCtx.ap_native_op takes 1 arguments. but " +
        std::to_string(args.size()) + " were given."};
    ADT_LET_CONST_REF(op_name, args.at(0).template CastTo<std::string>());
    NativeIrOpDeclare<drr::Node> op_declare{op_name, self.value().shared_ptr()};
    return DrrValueHelper{}.CastToAxprValue(SrcPtn(op_declare));
  }

  static bool IsBasicAttrName(const std::string& attr_name) {
    const auto& attr_getters = AttrGetters();
    return attr_getters.count(attr_name) > 0;
  }

  static const std::set<std::string>& AttrGetters() {
    static const std::set<std::string> set{
        "ap_trivial_fusion_op",
        "optional_ap_trivial_fusion_op",
        "ap_native_op",
    };
    return set;
  }
};

axpr::TypeImpl<axpr::BuiltinClassInstance<axpr::Value>>
GetSrcPtnOpPatternCtxClass() {
  using Impl = drr::SrcPtnOpPatternCtxMethodClass;
  using TT = drr::Type<drr::tSrcPtn<drr::OpPatternCtx>>;
  static auto cls(
      axpr::MakeBuiltinClass<axpr::Value>(TT{}.Name(), [&](const auto& Define) {
        Define("ap_trivial_fusion_op", &Impl::DeclareApTrivialFusionOp);
        Define("optional_ap_trivial_fusion_op",
               &Impl::DeclareOptionalApTrivialFusionOp);
        Define("ap_native_op", &Impl::DeclareNativeIrOp);
        Define("__str__", &Impl::ToString);
        Define("__hash__", &Impl::Hash);
        Define("__getattr__", &Impl::GetAttr);
        Define("__setattr__", &Impl::SetAttr);
      }));
  using Self = typename Impl::Self;
  return axpr::MakeGlobalNaiveClassOps<Self>(cls);
}

}  // namespace ap::drr
