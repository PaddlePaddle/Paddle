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

#include "paddle/ap/include/drr/res_ptn_unbound_native_ir_op_method_class.h"
#include "paddle/ap/include/axpr/callable_helper.h"
#include "paddle/ap/include/drr/drr_pass_type_helper.h"

namespace ap::drr {

struct ResPtnUnboundNativeIrOpMethodClass {
  using This = ResPtnUnboundNativeIrOpMethodClass;
  using Self = tResPtn<UnboundNativeIrOp<drr::Node>>;

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

  using Helper = OpTensorPatternCtxHelper;

  static adt::Result<axpr::Value> StaticCall(
      const axpr::Value& self_val, const std::vector<axpr::Value>& args) {
    ADT_LET_CONST_REF(self, self_val.template CastTo<Self>());
    return This{}.Call(self, args);
  }

  adt::Result<axpr::Value> Call(const Self& self,
                                const std::vector<axpr::Value>& args) {
    ADT_CHECK(args.size() == 2) << adt::errors::TypeError{
        std::string() +
        "ResPtnUnboundNativeIrOp.__call__ takes 2 arguments. but " +
        std::to_string(args.size()) + " were given."};
    ADT_LET_CONST_REF(input_vals,
                      args.at(0).template CastTo<adt::List<axpr::Value>>())
        << adt::errors::TypeError{
               std::string() +
               "the first argument of ResPtnUnboundNativeIrOp.__call__ should "
               "be a list."};
    adt::List<NativeIrValue<drr::Node>> inputs;
    inputs->reserve(input_vals->size());
    for (const auto& input_val : *input_vals) {
      ADT_LET_CONST_REF(
          input, input_val.template CastTo<tResPtn<NativeIrValue<drr::Node>>>())
          << adt::errors::TypeError{
                 std::string() +
                 "unsupported operand types for "
                 "ResPtnUnboundNativeIrOp.__call__ inputs: '" +
                 axpr::GetTypeName(input_val) + "'."};
      inputs->emplace_back(input.value());
    }
    ADT_LET_CONST_REF(output_vals,
                      args.at(1).template CastTo<adt::List<axpr::Value>>())
        << adt::errors::TypeError{
               std::string() +
               "the second argument of ResPtnUnboundNativeIrOp.__call__ should "
               "be a list."};
    adt::List<NativeIrValue<drr::Node>> outputs;
    outputs->reserve(output_vals->size());
    for (const auto& output_val : *output_vals) {
      ADT_LET_CONST_REF(valid_output,
                        ResPtnValidOutIrValue::CastFromAxprValue(output_val))
          << adt::errors::TypeError{
                 std::string() +
                 "unsupported operand types for "
                 "ResPtnUnboundNativeIrOp.__call__ outputs: '" +
                 axpr::GetTypeName(output_val) + "'."};
      using RetT = adt::Result<NativeIrValue<drr::Node>>;
      ADT_LET_CONST_REF(
          output,
          valid_output.Match(
              [&](const UnboundIrValue<drr::Node>& impl) -> RetT {
                return Helper{}.GetNativeIrValueByUnboundIrValue(impl);
              },
              [&](const tResPtn<NativeIrValue<drr::Node>>& impl) -> RetT {
                return impl.value();
              }));
      outputs->emplace_back(output);
    }
    ADT_RETURN_IF_ERR(CheckNoRedundantTensorNames(inputs, outputs));
    ADT_LET_CONST_REF(native_op,
                      Helper{}.GetNativeIrOpByUnboundNativeIrOp(self.value()));
    ADT_RETURN_IF_ERR(
        Helper{}.ConnectIrOpAndIrValue(native_op, inputs, outputs));
    return adt::Nothing{};
  }

  adt::Result<adt::Ok> CheckNoRedundantTensorNames(
      const adt::List<NativeIrValue<drr::Node>>& inputs,
      const adt::List<NativeIrValue<drr::Node>>& outputs) {
    std::unordered_set<std::string> existed_names;
    for (const auto& output : *outputs) {
      ADT_CHECK(existed_names.emplace(output->name).second)
          << adt::errors::TypeError{std::string() + "redundant tensor name '" +
                                    output->name + "' detected."};
    }
    return adt::Ok{};
  }

  static adt::Result<axpr::Value> SetAttr(
      const axpr::Value& self_val, const std::vector<axpr::Value>& args) {
    ADT_LET_CONST_REF(self, self_val.template CastTo<Self>());
    ADT_CHECK(args.size() == 2);
    ADT_LET_CONST_REF(attr_name, args.at(0).template CastTo<std::string>());
    const auto& attr_val = args.at(1);
    ADT_LET_CONST_REF(support_reifying, This{}.SupportReifying(self));
    if (support_reifying) {
      ADT_RETURN_IF_ERR(
          attr_val.template CastTo<axpr::Function<axpr::SerializableValue>>())
          << adt::errors::TypeError{
                 std::string() +
                 "an attribute of ResPtnNativeIrOp of abstract_drr_pass_type "
                 "should be a serializable `Function`(not a " +
                 axpr::GetTypeName(attr_val) +
                 "). op_name: " + self.value()->op_declare->op_name +
                 ", attr_name: " + attr_name};
    } else {
      ADT_CHECK(axpr::CallableHelper{}.IsCallable(attr_val))
          << adt::errors::TypeError{std::string() +
                                    "an attribute of ResPtnNativeIrOp should "
                                    "be a callable getter. op_name: " +
                                    self.value()->op_declare->op_name +
                                    ", attr_name: " + attr_name};
    }
    auto* attr_map = self.value()->op_declare->attr_map.shared_ptr().get();
    attr_map->Set(attr_name, attr_val);
    return adt::Nothing{};
  }

  adt::Result<bool> SupportReifying(const Self& self) const {
    ADT_LET_CONST_REF(
        op_pattern_ctx,
        adt::WeakPtrLock(self.value()->op_declare->op_pattern_ctx));
    ADT_LET_CONST_REF(drr_ctx, adt::WeakPtrLock(op_pattern_ctx->drr_ctx));
    return DrrPassTypeHelper{}.SupportReifying(drr_ctx->drr_pass_type);
  }
};

axpr::TypeImpl<axpr::BuiltinClassInstance<axpr::Value>>
GetResPtnUnboundNativeIrOpClass() {
  using TT = drr::Type<drr::tResPtn<drr::UnboundNativeIrOp<drr::Node>>>;
  using Impl = ResPtnUnboundNativeIrOpMethodClass;
  static auto cls(
      axpr::MakeBuiltinClass<axpr::Value>(TT{}.Name(), [&](const auto& Define) {
        Define("__str__", &Impl::ToString);
        Define("__hash__", &Impl::Hash);
        Define("__call__", &Impl::StaticCall);
        Define("__setattr__", &Impl::SetAttr);
      }));
  using Self = typename Impl::Self;
  return axpr::MakeGlobalNaiveClassOps<Self>(cls);
}

}  // namespace ap::drr
