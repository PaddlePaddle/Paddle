// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/ap/include/paddle/pass/ir_helper_method_class.h"
#include "paddle/ap/include/axpr/module_mgr.h"
#include "paddle/ap/include/axpr/to_string.h"
#include "paddle/ap/include/drr/drr_graph_descriptor.h"
#include "paddle/ap/include/drr/drr_node_descriptor.h"
#include "paddle/ap/include/paddle/pir_graph_descriptor.h"
#include "paddle/ap/include/paddle/pir_node_descriptor.h"

namespace ap::paddle {

struct PirHelperMethodClass {
  using This = PirHelperMethodClass;
  using GraphMatchCtx = ir_match::GraphMatchCtx<PirNode>;

  static adt::Result<axpr::Value> CreatePassManager(
      const axpr::Value&, const std::vector<axpr::Value>& args) {
    ADT_CHECK(args.size() == 0);
    auto* ctx = ::pir::IrContext::Instance();
    ctx->GetOrRegisterDialect<ap::dialect::OperatorDialect>();
    PassManager pass_manager{std::make_shared<::pir::PassManager>(ctx, 3)};
    return GetPirPassManagerClass().New(pass_manager);
  }

  static adt::Result<axpr::Value> CreateAccessTopoDrrPass(
      axpr::InterpreterBase<axpr::Value>* interpreter,
      const axpr::Value&,
      const std::vector<axpr::Value>& args) {
    ADT_CHECK(args.size() == 1) << adt::errors::TypeError{
        std::string() + "create_ap_drr_pass() takes 1 arguments, but " +
        std::to_string(args.size()) + " were given"};
    std::optional<std::unique_ptr<pir::Pass>> opt_pass;
    if (args.at(0).template CastableTo<std::string>()) {
      ADT_LET_CONST_REF(drr_pass_tag_name,
                        args.at(0).template CastTo<std::string>());
      opt_pass =
          ap::paddle::CreateAccessTopoDrrPass(interpreter->circlable_ref_list(),
                                              drr_pass_tag_name,
                                              /*steps_limit=*/std::nullopt);
    } else {
      opt_pass = ap::paddle::CreateCustomAccessTopoDrrPass(
          interpreter->circlable_ref_list(),
          args.at(0),
          /*steps_limit=*/std::nullopt,
          /*mut_matched_pattern_as_programs=*/adt::Nothing{});
    }
    if (!opt_pass.has_value()) {
      return adt::Nothing{};
    }
    Pass pass{std::move(opt_pass.value())};
    return GetPirPassClass().New(pass);
  }

  static adt::Result<axpr::Value> CreateAccessTopoDrrOneStepPass(
      axpr::InterpreterBase<axpr::Value>* interpreter,
      const axpr::Value&,
      const std::vector<axpr::Value>& packed_args_val) {
    const auto [args, kwargs] = *axpr::CastToPackedArgs(packed_args_val);
    ADT_CHECK(args->size() == 1) << adt::errors::TypeError{
        std::string() + "create_ap_drr_pass() takes 1 arguments, but " +
        std::to_string(args->size()) + " were given"};
    std::optional<std::unique_ptr<pir::Pass>> opt_pass;
    if (args->at(0).template CastableTo<std::string>()) {
      ADT_LET_CONST_REF(drr_pass_tag_name,
                        args->at(0).template CastTo<std::string>());
      opt_pass =
          ap::paddle::CreateAccessTopoDrrPass(interpreter->circlable_ref_list(),
                                              drr_pass_tag_name,
                                              /*steps_limit=*/1);
    } else {
      std::optional<axpr::Value> matched_pattern_mut_list{
          kwargs->OptGet("matched_pattern_mut_list")};
      if (!matched_pattern_mut_list.has_value()) {
        matched_pattern_mut_list = adt::Nothing{};
      }
      opt_pass = ap::paddle::CreateCustomAccessTopoDrrPass(
          interpreter->circlable_ref_list(),
          args->at(0),
          /*steps_limit=*/1,
          /*mut_matched_pattern_as_programs=*/matched_pattern_mut_list.value());
    }
    if (!opt_pass.has_value()) {
      return adt::Nothing{};
    }
    Pass pass{std::move(opt_pass.value())};
    return GetPirPassClass().New(pass);
  }

  static adt::Result<axpr::Value> CreateDeadCodeEliminationPass(
      const axpr::Value&, const std::vector<axpr::Value>& args) {
    ADT_CHECK(args.size() == 0) << adt::errors::TypeError{
        std::string() + "create_dce_pass() takes 0 arguments, but " +
        std::to_string(args.size()) + " were given"};
    Pass pass{pir::CreateDeadCodeEliminationPass()};
    return GetPirPassClass().New(pass);
  }

  static adt::Result<axpr::Value> CopyFusedOpsToProgram(
      const axpr::Value&, const std::vector<axpr::Value>& packed_args_val) {
    const auto [args, kwargs] = *axpr::CastToPackedArgs(packed_args_val);
    ADT_CHECK(args->size() == 1) << adt::errors::TypeError{
        std::string() + "copy_fused_ops_to_program() takes 1 arguments, but " +
        std::to_string(args->size()) + " were given"};
    ADT_LET_CONST_REF(pir_node, PirNodeHelper{}.CastFromAxprValue(args->at(0)))
        << adt::errors::TypeError{
               std::string() +
               "the first argument of copy_fused_ops_to_program() must be a "
               "PackedIrOp/OptPackedIrOp (not " +
               axpr::GetTypeName(args->at(0)) + ")"};
    ADT_LET_CONST_REF(tensor_match_ctx_val, kwargs->Get("tensor_match_ctx"))
        << adt::errors::TypeError{
               std::string() +
               "copy_fused_ops_to_program() need keyword argument "
               "'tensor_match_ctx' of 'TensorMatchCtx' type "};
    ADT_LET_CONST_REF(tensor_match_ctx,
                      tensor_match_ctx_val
                          .template CastTo<ir_match::TensorMatchCtx<PirNode>>())
        << adt::errors::TypeError{
               std::string() +
               "copy_fused_ops_to_program() need keyword argument "
               "'tensor_match_ctx' of 'TensorMatchCtx' type "};
    std::unordered_map<pir::Value, std::string> map;
    ADT_RETURN_IF_ERR(
        This{}.InitPirValue2Name(&map, tensor_match_ctx, pir_node));
    auto NameGetter = [&](pir::Value value) -> adt::Result<const std::string*> {
      const auto& iter = map.find(value);
      ADT_CHECK(iter != map.end());
      return &iter->second;
    };
    using RetT = adt::Result<axpr::Value>;
    return pir_node.Match(
        [&](const PackedIrOp& packed_ir_op) -> RetT {
          return This{}.CopyPackedIrOpBlockToProgram(packed_ir_op, NameGetter);
        },
        [&](const auto&) -> RetT {
          return adt::errors::TypeError{
              std::string() +
              "the first argument of copy_fused_ops_to_program() must be a "
              "PackedIrOp (not " +
              axpr::GetTypeName(args->at(0)) + ")"};
        });
  }

  adt::Result<adt::Ok> InitPirValue2Name(
      std::unordered_map<pir::Value, std::string>* map,
      const ir_match::TensorMatchCtx<PirNode>& tensor_match_ctx,
      const PirNode& pir_node) {
    ADT_LET_CONST_REF(ir_match_ctx,
                      adt::WeakPtrLock(tensor_match_ctx->ir_mtach_ctx));
    const auto& graph_match_ctx = ir_match_ctx->graph_match_ctx;
    ADT_LET_CONST_REF(drr_graph_node,
                      graph_match_ctx->GetMatchedSmallGraphNode(pir_node));
    ADT_LET_CONST_REF(drr_node, drr_graph_node.Get());
    ADT_RETURN_IF_ERR(CheckIsOpNode(drr_node));
    using Ok = adt::Result<adt::Ok>;
    auto DoEachNameAndIrValue = [&](const std::string& name,
                                    pir::Value val) -> Ok {
      if (!map->emplace(val, name).second) {
        ADT_CHECK(map->at(val) == name) << adt::errors::ValueError{
            std::string() + "InitPirValue2Name() failed. old_name: " +
            map->at(val) + ", new_name: " + name};
      }
      return adt::Ok{};
    };
    ADT_RETURN_IF_ERR(VisitInputNameAndPirValue(
        graph_match_ctx, drr_node, DoEachNameAndIrValue));
    return adt::Ok{};
  }

  template <typename YieldT>
  adt::Result<adt::Ok> VisitInputNameAndPirValue(
      const GraphMatchCtx& graph_match_ctx,
      const drr::Node& drr_node,
      const YieldT& Yield) {
    using Ok = adt::Result<adt::Ok>;
    auto DoEach = [&](const auto& drr_graph_node) -> Ok {
      ADT_LET_CONST_REF(upstreams_of_upstream, drr_graph_node.UpstreamNodes());
      ADT_LET_CONST_REF(input_graph_node, upstreams_of_upstream.Sole());
      ADT_LET_CONST_REF(input_drr_node, input_graph_node.Get());
      ADT_RETURN_IF_ERR(
          VisitNameAndPirValue(graph_match_ctx, input_drr_node, Yield));
      return adt::Ok{};
    };
    ADT_LET_CONST_REF(upstreams, drr_node.node().UpstreamNodes());
    ADT_RETURN_IF_ERR(upstreams.VisitNodes(DoEach));
    return adt::Ok{};
  }

  template <typename YieldT>
  adt::Result<adt::Ok> VisitNameAndPirValue(
      const GraphMatchCtx& graph_match_ctx,
      const drr::Node& drr_node,
      const YieldT& Yield) {
    using Ok = adt::Result<adt::Ok>;
    return drr_node.Match(
        [&](const drr::NativeIrValue<drr::Node>& impl) -> Ok {
          const auto& node = impl->node;
          std::size_t i = 0;
          auto DoEach = [&](const PirNode& pir_node) -> Ok {
            ADT_LET_CONST_REF(
                native_ir_value,
                pir_node.template TryGet<ap::paddle::NativeIrValue>());
            ADT_CHECK(i++ == 0);
            ADT_RETURN_IF_ERR(Yield(impl->name, native_ir_value.value));
            return adt::Ok{};
          };
          ADT_RETURN_IF_ERR(
              graph_match_ctx->VisitBigGraphIrValueNode(node, DoEach));
          return adt::Ok{};
        },
        [&](const drr::PackedIrValue<drr::Node>& impl) -> Ok {
          const auto& node = impl->node;
          std::size_t i = 0;
          auto DoEach = [&](const PirNode& pir_node) -> Ok {
            ADT_LET_CONST_REF(
                native_ir_value,
                pir_node.template TryGet<ap::paddle::NativeIrValue>());
            const auto& name = impl->name + "[" + std::to_string(i++) + "]";
            ADT_RETURN_IF_ERR(Yield(name, native_ir_value.value));
            return adt::Ok{};
          };
          ADT_RETURN_IF_ERR(
              graph_match_ctx->VisitBigGraphIrValueNode(node, DoEach));
          return adt::Ok{};
        },
        [&](const auto&) -> Ok {
          return adt::errors::TypeError{
              "copy_fused_ops_to_program() failed. the inputs of DrrPackedIrOp "
              "should be a DrrNativeIrValue or DrrPackedIrValue"};
        });
  }

  adt::Result<adt::Ok> CheckIsOpNode(const drr::Node& drr_node) {
    using Ok = adt::Result<adt::Ok>;
    return drr_node.Match(
        [&](const drr::PackedIrOp<drr::Node>&) -> Ok { return adt::Ok{}; },
        [&](const drr::OptPackedIrOp<drr::Node>&) -> Ok { return adt::Ok{}; },
        [&](const auto& impl) -> Ok {
          return adt::errors::TypeError{
              std::string() +
              "the argument 1 of ir_helper.copy_fused_ops_to_program() should "
              "be a PackedIrOp/RefIrOp"};
        });
  }

  template <typename NameGetterT>
  adt::Result<axpr::Value> CopyPackedIrOpBlockToProgram(
      const PackedIrOp& packed_ir_op, const NameGetterT& NameGetter) {
    auto* block = packed_ir_op.fusion_op.block();
    pir::IrContext* ctx = ::pir::IrContext::Instance();
    auto new_program = std::make_shared<::pir::Program>(ctx);
    auto clone_options = ::pir::CloneOptions::All();
    pir::IrMapping ir_mapping;
    ADT_RETURN_IF_ERR(InitIrMapping(NameGetter,
                                    pir::GetUsedExternalValue(*block),
                                    &ir_mapping,
                                    new_program->block()));
    for (const auto& op : *block) {
      auto* new_op = op.Clone(ir_mapping, clone_options);
      new_program->block()->push_back(new_op);
    }
    ADT_RETURN_IF_ERR(
        CloneSymbolicShapes(packed_ir_op.fusion_op->GetParentProgram(),
                            new_program.get(),
                            ir_mapping));
    Program ap_program{new_program};
    return GetPirProgramClass().New(ap_program);
  }

  adt::Result<adt::Ok> CloneSymbolicShapes(pir::Program* new_program,
                                           pir::Program* old_program,
                                           const pir::IrMapping& ir_mapping) {
    auto* new_shape_analysis =
        &::pir::ShapeAnalysisManager::Instance().Get(new_program);
    auto* old_shape_analysis =
        &::pir::ShapeAnalysisManager::Instance().Get(old_program);
    for (const auto& [old_value, new_value] : ir_mapping.GetMap<pir::Value>()) {
      new_shape_analysis->SetShapeOrDataForValue(
          new_value, old_shape_analysis->GetShapeOrDataForValue(old_value));
    }
    return adt::Ok{};
  }

  template <typename NameGetterT>
  adt::Result<adt::Ok> InitIrMapping(const NameGetterT& NameGetter,
                                     const std::vector<pir::Value>& free_values,
                                     pir::IrMapping* ir_mapping,
                                     pir::Block* block) {
    int i = 0;
    pir::Builder builder(pir::IrContext::Instance(), block);
    for (const auto& free_value : free_values) {
      ADT_LET_CONST_REF(name, NameGetter(free_value));
      ADT_CHECK(free_value.type().isa<pir::DenseTensorType>());
      const auto& type = free_value.type().dyn_cast<pir::DenseTensorType>();
      const auto& dims = ::common::vectorize(type.dims());
      auto phi_type = ::paddle::dialect::TransToPhiDataType(type.dtype());
      auto op = builder.Build<::paddle::dialect::DataOp>(
          *name, dims, phi_type, phi::Place());
      ir_mapping->Add(free_value, op->result(0));
    }
    return adt::Ok{};
  }

  static adt::Result<axpr::Value> Match(
      axpr::InterpreterBase<axpr::Value>* interpreter,
      const axpr::Value& self_val,
      const std::vector<axpr::Value>& args) {
    ADT_CHECK(args.size() == 2) << adt::errors::TypeError{
        std::string() + "PirHelper.match() takes 2 arguments, but " +
        std::to_string(args.size()) + " were given"};
    ADT_LET_CONST_REF(program, args.at(0).template CastTo<Program>())
        << adt::errors::TypeError{std::string() +
                                  "the argument 1 of PirHelper.match() should "
                                  "b a PirProgram (not " +
                                  axpr::GetTypeName(args.at(0)) + ")"};
    ADT_CHECK(axpr::CallableHelper{}.IsCallable(args.at(1)))
        << adt::errors::TypeError{std::string() +
                                  "the argument 2 of PirHelper.match() should "
                                  "be callable object (not " +
                                  axpr::GetTypeName(args.at(1)) + ")"};
    std::vector<axpr::Value> src_ptn_func_args{std::string("fake_pass"),
                                               args.at(1)};
    ADT_LET_CONST_REF(lambda, This{}.GetDrrCtxMaker());
    axpr::Function<axpr::SerializableValue> function{lambda, std::nullopt};
    ADT_LET_CONST_REF(drr_ctx,
                      ap::paddle::ApDrrHelper{interpreter->circlable_ref_list()}
                          .InterpretDrrCtxMaker(function, src_ptn_func_args));
    ADT_CHECK(drr_ctx->source_pattern_ctx.has_value());
    ap::paddle::PackedIrOpInnerSourcePatternHelper src_pattern_helper{};
    ADT_LET_CONST_REF(
        opt_graph_match_ctx,
        src_pattern_helper.Match(program->pir_program->block(),
                                 drr_ctx->source_pattern_ctx.value()));
    return opt_graph_match_ctx.has_value();
  }

  adt::Result<axpr::Lambda<axpr::CoreExpr>> GetDrrCtxMaker() {
    using LambdaT = adt::Result<axpr::Lambda<axpr::CoreExpr>>;
    static LambdaT lambda([]() -> LambdaT {
      auto GetBody = [&](auto& ctx) -> axpr::AnfExpr {
        auto& drr_ctx = ctx.Var("DrrCtx").Call();
        drr_ctx.Attr("init_pass_name").Call(ctx.Var("pass_name"));
        drr_ctx.Attr("init_source_pattern").Call(ctx.Var("src_ptn_func"));
        return drr_ctx;
      };
      axpr::LambdaExprBuilder lmbd;
      const auto& anf_expr =
          lmbd.Lambda({"pass_name", "src_ptn_func"}, GetBody);
      const auto& core_expr = axpr::ConvertAnfExprToCoreExpr(anf_expr);
      ADT_LET_CONST_REF(
          atomic, core_expr.template TryGet<axpr::Atomic<axpr::CoreExpr>>());
      return atomic.template TryGet<axpr::Lambda<axpr::CoreExpr>>();
    }());
    return lambda;
  }
};

void ForceLinkIrTools() {
  // Do nothing.
}

REGISTER_AP_BUILTIN_MODULE("ir_tools", [](auto* m) {
  using Impl = PirHelperMethodClass;
  m->Def("create_pass_manager", &Impl::CreatePassManager);
  m->Def("create_access_topo_drr_pass", &Impl::CreateAccessTopoDrrPass);
  m->Def("create_access_topo_drr_one_step_pass",
         &Impl::CreateAccessTopoDrrOneStepPass);
  m->Def("create_dce_pass", &Impl::CreateDeadCodeEliminationPass);
  m->Def("copy_fused_ops_to_program", &Impl::CopyFusedOpsToProgram);
  m->Def("match", &Impl::Match);
});

}  // namespace ap::paddle
