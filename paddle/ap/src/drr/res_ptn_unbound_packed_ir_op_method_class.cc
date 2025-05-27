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

#include "paddle/ap/include/drr/res_ptn_unbound_packed_ir_op_method_class.h"

namespace ap::drr {

struct ResPtnUnboundPackedIrOp {
  using This = ResPtnUnboundPackedIrOp;
  using Self = tResPtn<UnboundPackedIrOp<drr::Node>>;

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
        "ResPtnUnboundPackedIrOp.__call__ takes 2 arguments. but " +
        std::to_string(args.size()) + " were given."};
    ADT_LET_CONST_REF(input_vals,
                      args.at(0).template CastTo<adt::List<axpr::Value>>())
        << adt::errors::TypeError{
               std::string() +
               "the first argument of ResPtnUnboundPackedIrOp.__call__ should "
               "be a list."};
    adt::List<IrValue> inputs;
    inputs->reserve(input_vals->size());
    for (const auto& input_val : *input_vals) {
      ADT_LET_CONST_REF(input, CastToIrValue(input_val));
      inputs->emplace_back(input);
    }
    ADT_LET_CONST_REF(output_vals,
                      args.at(1).template CastTo<adt::List<axpr::Value>>())
        << adt::errors::TypeError{
               std::string() +
               "the second argument of ResPtnUnboundPackedIrOp.__call__ should "
               "be a list."};
    adt::List<IrValue> outputs;
    outputs->reserve(output_vals->size());
    for (const auto& output_val : *output_vals) {
      ADT_LET_CONST_REF(output, CastToIrValue(output_val));
      outputs->emplace_back(output);
    }
    ADT_RETURN_IF_ERR(CheckNoRedundantTensorNames(inputs, outputs));
    ADT_LET_CONST_REF(packed_op,
                      Helper{}.GetPackedIrOpByUnboundPackedIrOp(self.value()));
    ADT_RETURN_IF_ERR(
        Helper{}.ConnectIrOpAndIrValue(packed_op, inputs, outputs));
    return adt::Nothing{};
  }

  adt::Result<adt::Ok> CheckNoRedundantTensorNames(
      const adt::List<IrValue>& inputs, const adt::List<IrValue>& outputs) {
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

  adt::Result<IrValue> CastToIrValue(const axpr::Value& arg) {
    DrrValueHelper helper{};
    return helper.CastFromAxprValue(arg).DrrValueMatch(
        [&](const tResPtn<NativeIrValue<drr::Node>>& value)
            -> adt::Result<IrValue> { return value.value(); },
        [&](const tResPtn<PackedIrValue<drr::Node>>& value)
            -> adt::Result<IrValue> { return value.value(); },
        [&](const auto&) -> adt::Result<IrValue> {
          return adt::errors::TypeError{std::string() +
                                        "unsupported operand types for "
                                        "ResPtnUnboundPackedIrOp.__call__: " +
                                        axpr::GetTypeName(arg) +
                                        ". only 'ResPtnNativeIrValue' and "
                                        "'ResPtnPackedIrValue' supported. "};
        });
  }
};

axpr::TypeImpl<axpr::BuiltinClassInstance<axpr::Value>>
GetResPtnUnboundPackedIrOpClass() {
  using Impl = drr::ResPtnUnboundPackedIrOp;
  using TT = drr::Type<drr::tResPtn<drr::UnboundPackedIrOp<drr::Node>>>;
  static auto cls(
      axpr::MakeBuiltinClass<axpr::Value>(TT{}.Name(), [&](const auto& Define) {
        Define("__str__", &Impl::ToString);
        Define("__hash__", &Impl::Hash);
        Define("__call__", &Impl::StaticCall);
      }));
  using Self = typename Impl::Self;
  return axpr::MakeGlobalNaiveClassOps<Self>(cls);
}

}  // namespace ap::drr
