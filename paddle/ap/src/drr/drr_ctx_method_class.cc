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

#include "paddle/ap/include/drr/drr_ctx_method_class.h"
#include "paddle/ap/include/axpr/callable_helper.h"

namespace ap::drr {

struct DrrCtxMethodClass {
  using This = DrrCtxMethodClass;
  using Self = drr::DrrCtx;

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

  static adt::Result<axpr::Value> StaticInitPassName(
      axpr::InterpreterBase<axpr::Value>* interpreter,
      const axpr::Value& self_val,
      const std::vector<axpr::Value>& args) {
    ADT_LET_CONST_REF(self, self_val.template CastTo<Self>());
    ADT_CHECK(args.size() == 1);
    ADT_LET_CONST_REF(pass_name, args.at(0).template CastTo<std::string>())
        << adt::errors::TypeError{
               std::string() +
               "DrrCtx.init_pass_name() missing str typed argument 1"};
    self.shared_ptr()->pass_name = pass_name;
    return adt::Nothing{};
  }

  static adt::Result<axpr::Value> StaticSetDrrPassType(
      axpr::InterpreterBase<axpr::Value>* interpreter,
      const axpr::Value& self_val,
      const std::vector<axpr::Value>& args) {
    ADT_LET_CONST_REF(self, self_val.template CastTo<Self>());
    ADT_CHECK(args.size() == 1);
    ADT_LET_CONST_REF(drr_pass_type, args.at(0).template CastTo<std::string>())
        << adt::errors::TypeError{
               std::string() +
               "DrrCtx.set_drr_pass_type() missing str typed argument 1"};
    if (drr_pass_type == "abstract_drr_pass_type") {
      self.shared_ptr()->drr_pass_type = drr::AbstractDrrPassType{};
    } else if (drr_pass_type == "reified_drr_pass_type") {
      self.shared_ptr()->drr_pass_type = drr::ReifiedDrrPassType{};
    } else if (drr_pass_type == "access_topo_drr_pass_type") {
      self.shared_ptr()->drr_pass_type = drr::AccessTopoDrrPassType{};
    } else {
      return adt::errors::TypeError{
          std::string() + "invalid drr_pass_type '" + drr_pass_type +
          "'. valid drr pass types "
          "abstract_drr_pass_type/reified_drr_pass_type/"
          "access_topo_drr_pass_type "};
    }
    return adt::Nothing{};
  }

  static adt::Result<axpr::Value> StaticInitSourcePattern(
      axpr::InterpreterBase<axpr::Value>* interpreter,
      const axpr::Value& self_val,
      const std::vector<axpr::Value>& args) {
    ADT_LET_CONST_REF(self, self_val.template CastTo<Self>());
    ADT_CHECK(!self->source_pattern_ctx.has_value());
    ADT_CHECK(args.size() == 1);
    const auto& def_source_pattern = args.at(0);
    auto node_arena = std::make_shared<graph::NodeArena<drr::Node>>();
    SourcePatternCtx source_pattern_ctx{
        node_arena,
        OpPatternCtx{
            node_arena, std::map<std::string, IrOp>{}, self.shared_ptr()},
        TensorPatternCtx{
            node_arena, std::map<std::string, IrValue>{}, self.shared_ptr()}};
    self.shared_ptr()->source_pattern_ctx = source_pattern_ctx;
    DrrValueHelper helper{};
    ADT_RETURN_IF_ERR(interpreter->InterpretCall(
        def_source_pattern,
        {helper.CastToAxprValue(SrcPtn(source_pattern_ctx->op_pattern_ctx)),
         helper.CastToAxprValue(
             SrcPtn(source_pattern_ctx->tensor_pattern_ctx))}));
    return adt::Nothing{};
  }

  static adt::Result<axpr::Value> StaticInitConstraintFunc(
      axpr::InterpreterBase<axpr::Value>* interpreter,
      const axpr::Value& self_val,
      const std::vector<axpr::Value>& args) {
    ADT_LET_CONST_REF(self, self_val.template CastTo<Self>());
    ADT_CHECK(args.size() == 1);
    ADT_CHECK(axpr::CallableHelper{}.IsCallable(args.at(0)))
        << adt::errors::TypeError{
               std::string() +
               "the argument 1 of DrrCtx.init_constraint_func() should be a "
               "callable object"};
    self.shared_ptr()->constraint_func = args.at(0);
    return adt::Nothing{};
  }

  static adt::Result<axpr::Value> StaticInitResultPattern(
      axpr::InterpreterBase<axpr::Value>* interpreter,
      const axpr::Value& self_val,
      const std::vector<axpr::Value>& args) {
    ADT_LET_CONST_REF(self, self_val.template CastTo<Self>());
    ADT_CHECK(!self->result_pattern_ctx.has_value());
    ADT_CHECK(args.size() == 1);
    const auto& def_result_pattern = args.at(0);
    auto node_arena = std::make_shared<graph::NodeArena<drr::Node>>();
    ResultPatternCtx result_pattern_ctx{
        node_arena,
        OpPatternCtx{
            node_arena, std::map<std::string, IrOp>{}, self.shared_ptr()},
        TensorPatternCtx{
            node_arena, std::map<std::string, IrValue>{}, self.shared_ptr()},
        self->source_pattern_ctx.value()};
    self.shared_ptr()->result_pattern_ctx = result_pattern_ctx;
    DrrValueHelper helper{};
    ADT_RETURN_IF_ERR(interpreter->InterpretCall(
        def_result_pattern,
        {helper.CastToAxprValue(ResPtn(result_pattern_ctx->op_pattern_ctx)),
         helper.CastToAxprValue(
             ResPtn(result_pattern_ctx->tensor_pattern_ctx))}));
    return adt::Nothing{};
  }
};

struct TypeImplDrrCtxMethodClass {
  using This = TypeImplDrrCtxMethodClass;
  using Self = drr::Type<DrrCtx>;

  static adt::Result<axpr::Value> StaticConstruct(
      axpr::InterpreterBase<axpr::Value>* interpreter,
      const axpr::Value& instance_val,
      const std::vector<axpr::Value>& args) {
    return This{}.Construct(interpreter, instance_val, args);
  }

  adt::Result<axpr::Value> Construct(
      axpr::InterpreterBase<axpr::Value>* interpreter,
      const axpr::Value& instance_val,
      const std::vector<axpr::Value>& packed_args_val) {
    ADT_LET_CONST_REF(
        empty_self,
        instance_val
            .template CastTo<axpr::BuiltinClassInstance<axpr::Value>>());
    DrrCtx self{interpreter->circlable_ref_list()};
    if (packed_args_val.size() == 0) {
      return empty_self.type.New(self);
    }
    DrrValueHelper helper{};
    const auto& packed_args = axpr::CastToPackedArgs(packed_args_val);
    const auto& [args, kwargs] = *packed_args;
    ADT_CHECK(args->size() == 0) << adt::errors::TypeError{
        "the constructor of DrrCtx takes keyword arguments only."};
    {
      ADT_LET_CONST_REF(def_source_pattern, kwargs->Get("source_pattern"));
      auto node_arena = std::make_shared<graph::NodeArena<drr::Node>>();
      SourcePatternCtx source_pattern_ctx{
          node_arena,
          OpPatternCtx{
              node_arena, std::map<std::string, IrOp>{}, self.shared_ptr()},
          TensorPatternCtx{
              node_arena, std::map<std::string, IrValue>{}, self.shared_ptr()}};
      self.shared_ptr()->source_pattern_ctx = source_pattern_ctx;
      ADT_RETURN_IF_ERR(interpreter->InterpretCall(
          def_source_pattern,
          {helper.CastToAxprValue(SrcPtn(source_pattern_ctx->op_pattern_ctx)),
           helper.CastToAxprValue(
               SrcPtn(source_pattern_ctx->tensor_pattern_ctx))}));
    }
    {
      ADT_LET_CONST_REF(def_result_pattern, kwargs->Get("result_pattern"));
      auto node_arena = std::make_shared<graph::NodeArena<drr::Node>>();
      ResultPatternCtx result_pattern_ctx{
          node_arena,
          OpPatternCtx{
              node_arena, std::map<std::string, IrOp>{}, self.shared_ptr()},
          TensorPatternCtx{
              node_arena, std::map<std::string, IrValue>{}, self.shared_ptr()},
          self->source_pattern_ctx.value()};
      self.shared_ptr()->result_pattern_ctx = result_pattern_ctx;
      ADT_RETURN_IF_ERR(interpreter->InterpretCall(
          def_result_pattern,
          {helper.CastToAxprValue(ResPtn(result_pattern_ctx->op_pattern_ctx)),
           helper.CastToAxprValue(
               ResPtn(result_pattern_ctx->tensor_pattern_ctx))}));
    }
    return empty_self.type.New(self);
  }
};

axpr::TypeImpl<axpr::BuiltinClassInstance<axpr::Value>> GetDrrCtxClass() {
  using Impl = drr::DrrCtxMethodClass;
  using TImpl = TypeImplDrrCtxMethodClass;
  using TT = drr::Type<drr::DrrCtx>;
  static auto cls(
      axpr::MakeBuiltinClass<axpr::Value>(TT{}.Name(), [&](const auto& Define) {
        Define("__init__", &TImpl::StaticConstruct);
        Define("set_drr_pass_type", &Impl::StaticSetDrrPassType);
        Define("init_pass_name", &Impl::StaticInitPassName);
        Define("init_source_pattern", &Impl::StaticInitSourcePattern);
        Define("init_constraint_func", &Impl::StaticInitConstraintFunc);
        Define("init_result_pattern", &Impl::StaticInitResultPattern);
        Define("__str__", &Impl::ToString);
        Define("__hash__", &Impl::Hash);
      }));
  using Self = typename Impl::Self;
  return axpr::MakeGlobalNaiveClassOps<Self>(cls);
}

}  // namespace ap::drr
