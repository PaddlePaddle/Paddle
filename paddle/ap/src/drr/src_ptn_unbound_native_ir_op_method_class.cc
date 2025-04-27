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

#include "paddle/ap/include/drr/src_ptn_unbound_native_ir_op_method_class.h"

namespace ap::drr {

using SrcPtnValidIrValueImpl =
    std::variant<NativeIrValue<drr::Node>, UnboundIrValue<drr::Node>>;

struct SrcPtnValidIrValue : public SrcPtnValidIrValueImpl {
  using SrcPtnValidIrValueImpl::SrcPtnValidIrValueImpl;

  ADT_DEFINE_VARIANT_METHODS(SrcPtnValidIrValueImpl);

  const std::string& name() const {
    return Match([](const auto& ir_value) -> const std::string& {
      return ir_value->name;
    });
  }
};

struct SrcPtnUnboundNativeIrOp {
  using This = SrcPtnUnboundNativeIrOp;
  using Self = tSrcPtn<UnboundNativeIrOp<drr::Node>>;

  static adt::Result<axpr::Value> GetAttr(
      const axpr::Value& self_val, const std::vector<axpr::Value>& args) {
    ADT_LET_CONST_REF(self, self_val.template CastTo<Self>());
    ADT_CHECK(args.size() == 1);
    ADT_LET_CONST_REF(attr_name, args.at(0).template CastTo<std::string>());
    const auto& op_declare = self.value()->op_declare;
    const auto& attr_map = op_declare->attr_map;
    ADT_CHECK(attr_map->Has(attr_name)) << adt::errors::AttributeError{
        std::string() + "SrcPtnUnboundNativeIrOp '" + op_declare->op_name +
        "' has no attribute '" + attr_name + "'"};
    return attr_map->Get(attr_name);
  }

  static adt::Result<axpr::Value> SetAttr(
      const axpr::Value& self_val, const std::vector<axpr::Value>& args) {
    ADT_LET_CONST_REF(self, self_val.template CastTo<Self>());
    ADT_CHECK(args.size() == 2);
    ADT_LET_CONST_REF(attr_name, args.at(0).template CastTo<std::string>());
    const auto& attr_val = args.at(1);
    auto* attr_map = self.value()->op_declare->attr_map.shared_ptr().get();
    attr_map->Set(attr_name, attr_val);
    return adt::Nothing{};
  }

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
        "SrcPtnUnboundNativeIrOp.__call__ takes 2 arguments. but " +
        std::to_string(args.size()) + " were given."};
    ADT_LET_CONST_REF(input_vals,
                      args.at(0).template CastTo<adt::List<axpr::Value>>())
        << adt::errors::TypeError{
               std::string() +
               "the first argument of SrcPtnUnboundNativeIrOp.__call__ should "
               "be a list."};
    adt::List<SrcPtnValidIrValue> inputs;
    inputs->reserve(input_vals->size());
    for (const auto& input_val : *input_vals) {
      ADT_LET_CONST_REF(input, CastToSrcPtnValidIrValue(input_val));
      inputs->emplace_back(input);
    }
    ADT_LET_CONST_REF(output_vals,
                      args.at(1).template CastTo<adt::List<axpr::Value>>())
        << adt::errors::TypeError{
               std::string() +
               "the second argument of SrcPtnUnboundNativeIrOp.__call__ should "
               "be a list."};
    adt::List<UnboundIrValue<drr::Node>> outputs;
    outputs->reserve(output_vals->size());
    for (const auto& output_val : *output_vals) {
      ADT_LET_CONST_REF(
          output, output_val.template CastTo<UnboundIrValue<drr::Node>>());
      outputs->emplace_back(output);
    }
    ADT_RETURN_IF_ERR(CheckNoRedundantTensorNames(inputs, outputs));
    ADT_LET_CONST_REF(native_inputs, ConvertInputs(inputs));
    ADT_LET_CONST_REF(native_outputs, ConvertOutputs(outputs));
    ADT_LET_CONST_REF(native_op,
                      Helper{}.GetNativeIrOpByUnboundNativeIrOp(self.value()));
    ADT_RETURN_IF_ERR(Helper{}.ConnectIrOpAndIrValue(
        native_op, native_inputs, native_outputs));
    return adt::Nothing{};
  }

  adt::Result<adt::List<NativeIrValue<drr::Node>>> ConvertInputs(
      const adt::List<SrcPtnValidIrValue>& inputs) {
    adt::List<NativeIrValue<drr::Node>> ret_inputs;
    ret_inputs->reserve(inputs->size());
    using Native = NativeIrValue<drr::Node>;
    for (const auto& input : *inputs) {
      const auto& opt_ret_input = input.Match(
          [&](const NativeIrValue<drr::Node>& ir_value) -> adt::Result<Native> {
            return ir_value;
          },
          [&](const UnboundIrValue<drr::Node>& ir_value)
              -> adt::Result<Native> {
            return Helper{}.GetNativeIrValueByUnboundIrValue(ir_value);
          });
      ADT_LET_CONST_REF(ret_input, opt_ret_input);
      ret_inputs->emplace_back(ret_input);
    }
    return ret_inputs;
  }

  adt::Result<adt::List<NativeIrValue<drr::Node>>> ConvertOutputs(
      const adt::List<UnboundIrValue<drr::Node>>& outputs) {
    adt::List<NativeIrValue<drr::Node>> ret_outputs;
    ret_outputs->reserve(outputs->size());
    for (const auto& output : *outputs) {
      ADT_LET_CONST_REF(ret_output,
                        Helper{}.GetNativeIrValueByUnboundIrValue(output));
      ret_outputs->emplace_back(ret_output);
    }
    return ret_outputs;
  }

  adt::Result<adt::Ok> CheckNoRedundantTensorNames(
      const adt::List<SrcPtnValidIrValue>& inputs,
      const adt::List<UnboundIrValue<drr::Node>>& outputs) {
    std::unordered_set<std::string> existed_names;
    for (const auto& input : *inputs) {
      existed_names.insert(input.name());
    }
    for (const auto& output : *outputs) {
      ADT_CHECK(existed_names.emplace(output->name).second)
          << adt::errors::TypeError{std::string() + "redundant tensor name '" +
                                    output->name + "' detected."};
    }
    return adt::Ok{};
  }

  adt::Result<SrcPtnValidIrValue> CastToSrcPtnValidIrValue(
      const axpr::Value& arg) {
    DrrValueHelper helper{};
    return helper.CastFromAxprValue(arg).DrrValueMatch(
        [&](const tSrcPtn<NativeIrValue<drr::Node>>& value)
            -> adt::Result<SrcPtnValidIrValue> { return value.value(); },
        [&](const UnboundIrValue<drr::Node>& value)
            -> adt::Result<SrcPtnValidIrValue> { return value; },
        [&](const auto&) -> adt::Result<SrcPtnValidIrValue> {
          return adt::errors::TypeError{
              std::string() +
              "unsupported operand types for "
              "SrcPtnUnboundNativeIrOp.__call__: " +
              axpr::GetTypeName(arg) +
              ". only 'SrcPtnNativeIrValue' and 'UnboundIrValue' supported. "};
        });
  }
};

axpr::TypeImpl<axpr::BuiltinClassInstance<axpr::Value>>
GetSrcPtnUnboundNativeIrOpClass() {
  using Impl = drr::SrcPtnUnboundNativeIrOp;
  using TT = drr::Type<drr::tSrcPtn<drr::UnboundNativeIrOp<drr::Node>>>;
  static auto cls(
      axpr::MakeBuiltinClass<axpr::Value>(TT{}.Name(), [&](const auto& Define) {
        Define("__str__", &Impl::ToString);
        Define("__hash__", &Impl::Hash);
        Define("__call__", &Impl::StaticCall);
        Define("__getattr__", &Impl::GetAttr);
        Define("__setattr__", &Impl::SetAttr);
      }));
  using Self = typename Impl::Self;
  return axpr::MakeGlobalNaiveClassOps<Self>(cls);
}

}  // namespace ap::drr
