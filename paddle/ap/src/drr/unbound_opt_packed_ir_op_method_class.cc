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

#include "paddle/ap/include/drr/unbound_opt_packed_ir_op_method_class.h"

namespace ap::drr {

struct UnboundOptPackedIrOpMethodClass {
  using This = UnboundOptPackedIrOpMethodClass;
  using Self = UnboundOptPackedIrOp<drr::Node>;

  using DrrNode = drr::Node;
  using DrrNativeIrValue = drr::NativeIrValue<DrrNode>;
  using DrrPackedIrValue = drr::PackedIrValue<DrrNode>;

  static adt::Result<axpr::Value> ToString(
      const axpr::Value& self_val, const std::vector<axpr::Value>& args) {
    ADT_LET_CONST_REF(self, self_val.template CastTo<Self>());
    const void* ptr = self.__adt_rc_shared_ptr_raw_ptr();
    std::ostringstream ss;
    ss << "<" << drr::Type<Self>{}.Name() << " object at " << ptr << ">";
    return ss.str();
  }

  static adt::Result<axpr::Value> Hash(const axpr::Value& self_val,
                                       const std::vector<axpr::Value>& args) {
    ADT_LET_CONST_REF(self, self_val.template CastTo<Self>());
    const void* ptr = self.__adt_rc_shared_ptr_raw_ptr();
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
        "UnboundOptPackedIrOp.__call__ takes 2 arguments. but " +
        std::to_string(args.size()) + " were given."};
    ADT_LET_CONST_REF(input_vals,
                      args.at(0).template CastTo<adt::List<axpr::Value>>())
        << adt::errors::TypeError{
               std::string() +
               "the first argument of UnboundOptPackedIrOp.__call__ should "
               "be a list."};
    adt::List<SrcPtnValidInIrValue> inputs;
    inputs->reserve(input_vals->size());
    for (const auto& input_val : *input_vals) {
      ADT_LET_CONST_REF(input, CastToSrcPtnValidInIrValue(input_val));
      inputs->emplace_back(input);
    }
    ADT_LET_CONST_REF(output_vals,
                      args.at(1).template CastTo<adt::List<axpr::Value>>())
        << adt::errors::TypeError{
               std::string() +
               "the second argument of UnboundOptPackedIrOp.__call__ should "
               "be a list."};
    adt::List<SrcPtnValidOutIrValue> outputs;
    outputs->reserve(output_vals->size());
    for (const auto& output_val : *output_vals) {
      ADT_LET_CONST_REF(output, CastToSrcPtnValidOutIrValue(output_val));
      outputs->emplace_back(output);
    }
    ADT_RETURN_IF_ERR(CheckNoRedundantTensorNames(inputs, outputs));
    ADT_LET_CONST_REF(packed_inputs, ConvertInputs(inputs));
    {
      ADT_LET_CONST_REF(num_packed_ir_value_inputs,
                        GetNumIrValues<DrrPackedIrValue>(packed_inputs));
      ADT_CHECK(num_packed_ir_value_inputs <= 1) << adt::errors::TypeError{
          std::string() +
          "UnboundOptPackedIrOp.__call__(): only 0 or 1 packed ir value inputs "
          "are supported. " +
          std::to_string(num_packed_ir_value_inputs) + " inputs were given."};
    }
    {
      ADT_LET_CONST_REF(num_native_ir_value_inputs,
                        GetNumIrValues<DrrNativeIrValue>(packed_inputs));
      ADT_CHECK(num_native_ir_value_inputs == 1) << adt::errors::TypeError{
          std::string() +
          "UnboundOptPackedIrOp.__call__(): only sole native ir value input "
          "are supported,  but" +
          std::to_string(num_native_ir_value_inputs) +
          " native ir value inputs were given."};
    }
    ADT_LET_CONST_REF(packed_outputs, ConvertOutputs(outputs));
    {
      ADT_LET_CONST_REF(
          num_packed_ir_value_outputs,
          this->template GetNumIrValues<DrrPackedIrValue>(packed_outputs));
      ADT_CHECK(num_packed_ir_value_outputs <= 1) << adt::errors::TypeError{
          std::string() +
          "UnboundOptPackedIrOp.__call__(): only 0 or 1 packed ir value "
          "outputs are supported. " +
          std::to_string(num_packed_ir_value_outputs) + " outputs were given."};
    }
    {
      ADT_LET_CONST_REF(
          num_native_ir_value_outputs,
          this->template GetNumIrValues<DrrNativeIrValue>(packed_outputs));
      ADT_CHECK(num_native_ir_value_outputs == 1) << adt::errors::TypeError{
          std::string() +
          "UnboundOptPackedIrOp.__call__(): only sole native ir value output "
          "are supported,  but" +
          std::to_string(num_native_ir_value_outputs) +
          " native ir value outputs were given."};
    }
    ADT_LET_CONST_REF(opt_packed_op,
                      Helper{}.GetOptPackedIrOpByUnboundOptPackedIrOp(self));
    ADT_RETURN_IF_ERR(Helper{}.ConnectIrOpAndIrValue(
        opt_packed_op, packed_inputs, packed_outputs));
    return adt::Nothing{};
  }

  template <typename DrrNodeT>
  adt::Result<std::size_t> GetNumIrValues(
      const adt::List<IrValue>& ir_values) const {
    std::size_t count = 0;
    for (const auto& ir_value : *ir_values) {
      count += ir_value.template Has<DrrNodeT>();
    }
    return count;
  }

  adt::Result<adt::List<IrValue>> ConvertInputs(
      const adt::List<SrcPtnValidInIrValue>& inputs) {
    adt::List<IrValue> ret_inputs;
    ret_inputs->reserve(inputs->size());
    using IrVal = IrValue;
    for (const auto& input : *inputs) {
      const auto& opt_ret_input = input.Match(
          [&](const NativeIrValue<drr::Node>& ir_value) -> adt::Result<IrVal> {
            return ir_value;
          },
          [&](const PackedIrValue<drr::Node>& ir_value) -> adt::Result<IrVal> {
            return ir_value;
          },
          [&](const UnboundIrValue<drr::Node>& ir_value) -> adt::Result<IrVal> {
            ADT_LET_CONST_REF(
                ret, Helper{}.GetNativeIrValueByUnboundIrValue(ir_value));
            return ret;
          },
          [&](const UnboundPackedIrValue<drr::Node>& ir_value)
              -> adt::Result<IrVal> {
            ADT_LET_CONST_REF(
                ret, Helper{}.GetPackedIrValueByUnboundPackedIrValue(ir_value));
            return ret;
          });
      ADT_LET_CONST_REF(ret_input, opt_ret_input);
      ret_inputs->emplace_back(ret_input);
    }
    return ret_inputs;
  }

  adt::Result<adt::List<IrValue>> ConvertOutputs(
      const adt::List<SrcPtnValidOutIrValue>& outputs) {
    adt::List<IrValue> ret_outputs;
    using IrVal = IrValue;
    ret_outputs->reserve(outputs->size());
    for (const auto& output : *outputs) {
      const auto& opt_ret_output = output.Match(
          [&](const UnboundIrValue<drr::Node>& ir_value) -> adt::Result<IrVal> {
            ADT_LET_CONST_REF(
                ret, Helper{}.GetNativeIrValueByUnboundIrValue(ir_value));
            return ret;
          },
          [&](const UnboundPackedIrValue<drr::Node>& ir_value)
              -> adt::Result<IrVal> {
            ADT_LET_CONST_REF(
                ret, Helper{}.GetPackedIrValueByUnboundPackedIrValue(ir_value));
            return ret;
          });
      ADT_LET_CONST_REF(ret_output, opt_ret_output);
      ret_outputs->emplace_back(ret_output);
    }
    return ret_outputs;
  }

  adt::Result<adt::Ok> CheckNoRedundantTensorNames(
      const adt::List<SrcPtnValidInIrValue>& inputs,
      const adt::List<SrcPtnValidOutIrValue>& outputs) {
    std::unordered_set<std::string> existed_names;
    for (const auto& input : *inputs) {
      existed_names.insert(input.name());
    }
    for (const auto& output : *outputs) {
      ADT_CHECK(existed_names.emplace(output.name()).second)
          << adt::errors::TypeError{std::string() + "redundant tensor name '" +
                                    output.name() + "' detected."};
    }
    return adt::Ok{};
  }

  adt::Result<SrcPtnValidInIrValue> CastToSrcPtnValidInIrValue(
      const axpr::Value& arg) {
    DrrValueHelper helper{};
    return helper.CastFromAxprValue(arg).DrrValueMatch(
        [&](const tSrcPtn<NativeIrValue<drr::Node>>& value)
            -> adt::Result<SrcPtnValidInIrValue> {
          return SrcPtnValidInIrValue{value.value()};
        },
        [&](const tSrcPtn<PackedIrValue<drr::Node>>& value)
            -> adt::Result<SrcPtnValidInIrValue> {
          return SrcPtnValidInIrValue{value.value()};
        },
        [&](const UnboundIrValue<drr::Node>& value)
            -> adt::Result<SrcPtnValidInIrValue> { return value; },
        [&](const UnboundPackedIrValue<drr::Node>& value)
            -> adt::Result<SrcPtnValidInIrValue> { return value; },
        [&](const auto&) -> adt::Result<SrcPtnValidInIrValue> {
          return adt::errors::TypeError{
              std::string() +
              "unsupported operand types for the first arguments of "
              "UnboundOptPackedIrOp.__call__: " +
              axpr::GetTypeName(arg) +
              ". only 'SrcPtnPackedIrValue' and 'UnboundIrValue' supported. "};
        });
  }

  adt::Result<SrcPtnValidOutIrValue> CastToSrcPtnValidOutIrValue(
      const axpr::Value& arg) {
    DrrValueHelper helper{};
    return helper.CastFromAxprValue(arg).DrrValueMatch(
        [&](const UnboundIrValue<drr::Node>& value)
            -> adt::Result<SrcPtnValidOutIrValue> { return value; },
        [&](const UnboundPackedIrValue<drr::Node>& value)
            -> adt::Result<SrcPtnValidOutIrValue> { return value; },
        [&](const auto&) -> adt::Result<SrcPtnValidOutIrValue> {
          return adt::errors::TypeError{
              std::string() +
              "unsupported operand types for the second arguments of "
              "UnboundOptPackedIrOp.__call__: " +
              axpr::GetTypeName(arg) +
              ". only 'SrcPtnPackedIrValue' and 'UnboundIrValue' supported. "};
        });
  }
};

axpr::TypeImpl<axpr::BuiltinClassInstance<axpr::Value>>
GetUnboundOptPackedIrOpClass() {
  using Impl = UnboundOptPackedIrOpMethodClass;
  using TT = drr::Type<UnboundOptPackedIrOp<drr::Node>>;
  static auto cls(
      axpr::MakeBuiltinClass<axpr::Value>(TT{}.Name(), [&](const auto& Define) {
        Define("__str__", &Impl::ToString);
        Define("__hash__", &Impl::Hash);
      }));
  using Self = typename Impl::Self;
  return axpr::MakeGlobalNaiveClassOps<Self>(cls);
}

}  // namespace ap::drr
