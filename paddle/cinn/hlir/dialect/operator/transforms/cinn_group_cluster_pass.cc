// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/cinn/hlir/dialect/operator/transforms/cinn_group_cluster_pass.h"

#include <unordered_map>

#include "paddle/cinn/adt/generate_map_expr.h"
#include "paddle/cinn/hlir/dialect/operator/ir/manual_op.h"
#include "paddle/cinn/hlir/dialect/operator/ir/op_attribute.h"
#include "paddle/cinn/hlir/dialect/operator/ir/op_dialect.h"
#include "paddle/cinn/hlir/dialect/runtime/ir/runtime_dialect.h"
#include "paddle/cinn/hlir/framework/pir_compiler.h"
#include "paddle/cinn/runtime/flags.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/pir/core/program.h"
#include "paddle/pir/pass/pass_registry.h"
#include "paddle/pir/pattern_rewrite/frozen_rewrite_pattern_set.h"
// #include "paddle/cinn/hlir/dialect/operator/transforms/group_merge/op_with_group_merge_util.h"
// #include "paddle/fluid/pir/transforms/sub_graph_detector.h"



namespace cinn {
namespace dialect {
namespace ir {

// std::unordered_set<pir::Value> GetInnerGeneValue( const std::vector<pir::Operation *>& op_list )
// {
//   std::unordered_set<pir::Value> inner_values;

//   for( auto op : op_list )
//   {
//     for( size_t i = 0; i < op->num_results(); ++i )
//     {
//       inner_values.insert( op->result(i));
//     }
//   }

//   return inner_values;
// }

// class GroupOpClusterPattern : public pir::OpRewritePattern<cinn::dialect::GroupOp> {
//  public:
//   GroupOpClusterPattern(
//       ::pir::IrContext* context)
//       : pir::OpRewritePattern<cinn::dialect::GroupOp>(context)  {}

//   bool MatchAndRewrite(cinn::dialect::GroupOp group_op,
//                        pir::PatternRewriter& rewriter) const override {
//     ::pir::IrContext* ctx = ::pir::IrContext::Instance();
//     auto target = cinn::common::DefaultNVGPUTarget();
//     auto* program = group_op->GetParentProgram();
//     VLOG(4) << "Before GroupOpPattern: " << *program;
    
//     auto inner_values = GetInnerGeneValue( group_op.ops() );

//     std::queue<::pir::Operation*> op_queue;
    
//     std::unordered_map<::pir::Operation*, std::vector<::pir::Operation*> > op_path;


//     auto op_list = group_op.ops();

//     std::vector< std::vector<::pir::Operation*>  > first_stage_output;

//     std::unordered_set<::pir::Operation*> yield_output_ops;
//     auto yield_op = op_list.back();
//     for( size_t i = 0; i < yield_op->num_operands(); ++i )
//     {
//       yield_output_ops.insert( yield_op->operand_source(i).defining_op() );
//     }
    
//     for( auto* op : op_list)
//     {
      
//       for( size_t i = 0; i < op->num_operands(); ++i)
//       {
//         if( !inner_values.count( op->operand_source(i)))
//         {
//           continue;
//         }
        
//         auto pre_op = op->operand_source(i).defining_op();

//         if( cinn::hlir::framework::pir::CompatibleInfo::OpKind(*pre_op) == cinn::hlir::framework::kReduction )
//         {
//           continue;
//         }
//         // get pre op all op list
//         op_path[ op ].insert( op_path[op].end(), op_path[pre_op].begin(), op_path[pre_op].end() );
        
//       }

//       op_path[op].push_back( op ); 

//       if( yield_output_ops.count(op) || cinn::hlir::framework::pir::CompatibleInfo::OpKind(*op) == cinn::hlir::framework::kReduction)
//       {
//         first_stage_output.push_back( op_path[op ]);
//       }
//     }

//     // for( auto & group_ops : first_stage_output)
//     // {
//     //   ::pir::ReplaceWithGroupOp(rewriter.block(), group_ops);
//     // }


//     return true;
//   }

// //  private:
// //   std::shared_ptr<pir::ShapeConstraintIRAnalysis> shape_analysis_{nullptr};
// };


CinnGroupClustergPass::CinnGroupClustergPass()
    : pir::PatternRewritePass("cinn_group_cluster_pass", 1) {}

  pir::RewritePatternSet CinnGroupClustergPass::InitializePatterns(pir::IrContext* context) {
    // context->GetOrRegisterDialect<cinn::dialect::RuntimeDialect>();
    // context->GetOrRegisterDialect<cinn::dialect::OperatorDialect>();
    // context->GetOrRegisterDialect<paddle::dialect::KernelDialect>();

    pir::RewritePatternSet ps(context);
    // ps.Add<GroupOpClusterPattern>(context);

    return ps;
  }

  bool CinnGroupClustergPass::CanApplyOn(pir::Operation* op) const {
    return op->isa<pir::ModuleOp>() && op->num_regions() > 0;
  }



// std::unique_ptr<::pir::Pass> CreateCinnGroupClusterPass() {
//   return std::make_unique<CinnGroupClustergPass>();
// }

}  // namespace ir
}  // namespace dialect
}  // namespace cinn

// REGISTER_IR_PASS(cinn_group_cluster_pass, CinnGroupClustergPass);