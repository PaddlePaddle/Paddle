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

#include "paddle/ap/include/drr/res_ptn_op_pattern_ctx_method_class.h"
#include "paddle/ap/include/axpr/callable_helper.h"

namespace ap::drr {

struct ResPtnOpPatternCtxMethodClass {
  using This = ResPtnOpPatternCtxMethodClass;
  using ObjT = tResPtn<OpPatternCtx>;
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
    ADT_CHECK(!This{}.IsBasicAttrName(attr_name))
        << adt::errors::AttributeError{"can't set attribute '" + attr_name +
                                       "'"};
    return MakeAndRegisterUnboundIrOp(self_val, args);
  }

  static adt::Result<axpr::Value> MakeAndRegisterUnboundIrOp(
      const axpr::Value& self_val, const std::vector<axpr::Value>& args) {
    ADT_LET_CONST_REF(self, self_val.template CastTo<Self>());
    ADT_CHECK(args.size() == 2);
    ADT_LET_CONST_REF(op_uid, args.at(0).template CastTo<std::string>());
    const auto& drr_value = DrrValueHelper{}.CastFromAxprValue(args.at(1));
    const auto& opt_ir_op = drr_value.DrrValueMatch(
        [&](const tResPtn<PackedIrOpDeclare<drr::Node>>& op)
            -> adt::Result<IrOp> {
          return UnboundPackedIrOp<drr::Node>{op.value(), op_uid};
        },
        [&](const tResPtn<NativeIrOpDeclare<drr::Node>>& op)
            -> adt::Result<IrOp> {
          return UnboundNativeIrOp<drr::Node>{op.value(), op_uid};
        },
        [&](const OptPackedIrOpDeclare<drr::Node>&) -> adt::Result<IrOp> {
          return adt::errors::TypeError{
              std::string() +
              "only 'ResPtnPackedIrOpDeclare' and 'ResPtnNativeIrOpDeclare' "
              "supported for op name binding. '" +
              axpr::GetTypeName(args.at(1)) + "' were given."};
        },
        [&](const auto&) -> adt::Result<IrOp> {
          return adt::errors::TypeError{
              std::string() +
              "only 'ResPtnPackedIrOpDeclare' and 'ResPtnNativeIrOpDeclare' "
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
    ADT_CHECK(!This{}.IsBasicAttrName(attr_name)) << adt::errors::RuntimeError{
        std::string() + "Dead code encounterred. attr_name: " + attr_name};
    ADT_LET_CONST_REF(ir_op, Helper{}.GetIrOpByUid(self.value(), attr_name));
    const auto& convert_result = ir_op.Match(
        [](const NativeIrOp<drr::Node>& impl) -> adt::Result<DrrValue> {
          return impl;
        },
        [](const PackedIrOp<drr::Node>& impl) -> adt::Result<DrrValue> {
          return impl;
        },
        [](const OptPackedIrOp<drr::Node>& impl) -> adt::Result<DrrValue> {
          return adt::errors::KeyError{
              std::string() +
              "OptPackedIrOp is not supported in result pattern."};
        },
        [](const UnboundNativeIrOp<drr::Node>& x) -> adt::Result<DrrValue> {
          return ResPtn(x);
        },
        [](const UnboundPackedIrOp<drr::Node>& x) -> adt::Result<DrrValue> {
          return ResPtn(x);
        },
        [](const UnboundOptPackedIrOp<drr::Node>& x) -> adt::Result<DrrValue> {
          return adt::errors::KeyError{
              std::string() +
              "UnboundOptPackedIrOp is not supported in result pattern."};
        });
    ADT_LET_CONST_REF(drr_value, convert_result);
    return DrrValueHelper{}.CastToAxprValue(drr_value);
  }

  static adt::Result<axpr::Value> StaticDeclareApPatternFusionOp(
      const axpr::Value& self_val, const std::vector<axpr::Value>& args) {
    return This{}.DeclareApPatternFusionOp(self_val, args);
  }

  adt::Result<axpr::Value> DeclareApPatternFusionOp(
      const axpr::Value& self_val, const std::vector<axpr::Value>& args) {
    ADT_LET_CONST_REF(self, self_val.template CastTo<Self>());
    ADT_CHECK(args.size() == 1) << adt::errors::TypeError{
        std::string() +
        "ResPtnOpPatternCtx.ap_pattern_fusion_op takes 1 arguments. but " +
        std::to_string(args.size()) + " were given."};
    ADT_LET_CONST_REF(kernel_define_lambda, CheckCallable(args.at(0)))
        << adt::errors::TypeError{std::string() +
                                  "argument 1 of o.ap_pattern_fusion_op should "
                                  "be a function_code object."};
    auto data =
        std::make_shared<ResPtnPackedIrOpDeclareData>(kernel_define_lambda);
    PackedIrOpDeclare<drr::Node> op_declare{
        "ap_pattern_fusion_op", self.value().shared_ptr(), data};
    return DrrValueHelper{}.CastToAxprValue(ResPtn(op_declare));
  }

  static adt::Result<axpr::Value> StaticDeclareApNativeOp(
      const axpr::Value& self_val, const std::vector<axpr::Value>& args) {
    return This{}.DeclareApNativeOp(self_val, args);
  }

  adt::Result<axpr::Value> DeclareApNativeOp(
      const axpr::Value& self_val, const std::vector<axpr::Value>& args) {
    ADT_LET_CONST_REF(self, self_val.template CastTo<Self>());
    ADT_CHECK(args.size() == 1) << adt::errors::TypeError{
        std::string() +
        "ResPtnOpPatternCtx.ap_native_op takes 1 arguments. but " +
        std::to_string(args.size()) + " were given."};
    ADT_LET_CONST_REF(op_name, args.at(0).template CastTo<std::string>())
        << adt::errors::TypeError{std::string() +
                                  "argument 1 of o.ap_native_op should "
                                  "be a str."};
    NativeIrOpDeclare<drr::Node> op_declare{op_name, self.value().shared_ptr()};
    return DrrValueHelper{}.CastToAxprValue(ResPtn(op_declare));
  }

  adt::Result<axpr::Value> CheckCallable(const axpr::Value& val) {
    ADT_CHECK(axpr::CallableHelper{}.IsCallable(val)) << adt::errors::TypeError{
        std::string() +
        "the argument 1 of ResPtnOpPatternCtx.ap_pattern_fusion_op() should be "
        "callable"};
    return val;
  }

  static adt::Result<axpr::Value> DeclareNativeIrOp(
      const axpr::Value& self_val, const std::vector<axpr::Value>& args) {
    ADT_LET_CONST_REF(self, self_val.template CastTo<Self>());
    ADT_CHECK(args.size() == 1) << adt::errors::TypeError{
        std::string() +
        "ResPtnOpPatternCtx.ap_native_op takes 1 arguments. but " +
        std::to_string(args.size()) + " were given."};
    ADT_LET_CONST_REF(op_name, args.at(0).template CastTo<std::string>());
    NativeIrOpDeclare<drr::Node> op_declare{op_name, self.value().shared_ptr()};
    return DrrValueHelper{}.CastToAxprValue(ResPtn(op_declare));
  }

  bool IsBasicAttrName(const std::string& attr_name) {
    const auto& attr_getters = AttrGetters();
    return attr_getters.count(attr_name) > 0;
  }

  const std::set<std::string>& AttrGetters() {
    static const std::set<std::string> set{
        "ap_pattern_fusion_op",
    };
    return set;
  }
};

axpr::TypeImpl<axpr::BuiltinClassInstance<axpr::Value>>
GetResPtnOpPatternCtxClass() {
  using Impl = drr::ResPtnOpPatternCtxMethodClass;
  using TT = drr::Type<drr::tResPtn<drr::OpPatternCtx>>;
  static auto cls(
      axpr::MakeBuiltinClass<axpr::Value>(TT{}.Name(), [&](const auto& Define) {
        Define("__str__", &Impl::ToString);
        Define("__hash__", &Impl::Hash);
        Define("__getattr__", &Impl::GetAttr);
        Define("__setattr__", &Impl::SetAttr);
        Define("ap_pattern_fusion_op", &Impl::StaticDeclareApPatternFusionOp);
        Define("ap_native_op", &Impl::StaticDeclareApNativeOp);
      }));
  using Self = typename Impl::Self;
  return axpr::MakeGlobalNaiveClassOps<Self>(cls);
}

}  // namespace ap::drr
