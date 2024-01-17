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

#include "paddle/cinn/hlir/dialect/operator/transforms/cinn_group_cluster_pass.h"

#include "paddle/cinn/hlir/dialect/operator/ir/cinn_op.h"
#include "paddle/cinn/hlir/framework/pir/utils.h"
#include "paddle/common/ddim.h"
#include "paddle/fluid/pir/dialect/operator/ir/manual_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/drr/api/match_context.h"
#include "paddle/pir/core/builtin_dialect.h"
#include "paddle/pir/pass/pass.h"
#include "paddle/pir/pattern_rewrite/pattern_applicator.h"
#include "paddle/pir/pattern_rewrite/pattern_match.h"
#include "paddle/pir/pattern_rewrite/pattern_rewrite_driver.h"
#include "paddle/cinn/hlir/dialect/operator/ir/manual_op.h"
#include "paddle/fluid/pir/transforms/sub_graph_detector.h"
#include "paddle/pir/dialect/control_flow/ir/cf_dialect.h"
#include "paddle/pir/dialect/control_flow/ir/cf_op.h"

namespace cinn {
namespace dialect {
namespace ir {

namespace {

std::unordered_set<pir::Value> GetInnerGeneValue( const std::vector<pir::Operation *>& op_list )
{
  std::unordered_set<pir::Value> inner_values;

  for( auto op : op_list )
  {
    for( size_t i = 0; i < op->num_results(); ++i )
    {
      inner_values.insert( op->result(i));
    }
  }

  return inner_values;
}

struct GroupClusterNode
{
    std::vector<::pir::Operation*> ops;


    std::unordered_set<::pir::Value> GetOutsideInput()
    {
      
      std::unordered_set<pir::Value> outside_ops;
      auto block_inner_output = GetInnerGeneValue( ops);
      

      for (auto & op : ops) {
        for (size_t i = 0; i < op->num_operands(); ++i) {
          if (!block_inner_output.count(op->operand_source(i)) &&
              !outside_ops.count(op->operand_source(i))) {
            outside_ops.insert(op->operand_source(i));
        }
      }
      return outside_ops;
    }

    void GenerateOutputValue( const std::unordered_set<::pir::Value>& outside_need_value)
    {
      for( auto & op : ops)
      {
        for( size_t i = 0; i < op->num_results(); ++i )
        {
          if( outside_need_value.count( op->result(i) ) )
          {
            output_value.push_back( op->result(i) );
          }
        } 
      }
    }

private:
    std::vector<::pir::Value> output_value;
};

::pir::Operation* ReplaceWithGroupOp(::pir::Block* block,
                        const ::pir::GroupOpsVec& group_ops, 
                        ::pir::Operation* insert_op) 
{
  ::pir::IrContext* ctx = ::pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<cinn::dialect::OperatorDialect>();
  ctx->GetOrRegisterDialect<::pir::ControlFlowDialect>();
  ::pir::Builder builder = ::pir::Builder(ctx, block);
  // step 1: Ensure the insert point and create GroupOp here.
  auto* last_op = group_ops.back();
  builder.SetInsertionPointAfter(insert_op);
  std::vector<pir::Type> output_types;
  std::vector<pir::Value> outputs = ::pir::AnalysisOutputs(group_ops);

  ::pir::IrMapping ir_mapping;
  for (auto& value : outputs) {
    output_types.emplace_back(value.type());   
  }
  // step 2: Replace the old op with GroupOp.
  auto new_group_op = builder.Build<cinn::dialect::GroupOp>(output_types);
  pir::Block* group_block = new_group_op.block();

  
  ::pir::CloneOptions clone_options(false, true, false);

   std::stringstream ss;
   ::pir::IrPrinter printer(ss);
   
  for (auto op : group_ops) {
    printer.PrintOperation( op );
    ss << "\n";
  }

  std::cerr << "program \n" << ss.str() << std::endl;

  for (auto op : group_ops) {
    std::cerr << "!! name " << op->name() << std::endl; 
    auto new_op = op->Clone(ir_mapping, clone_options);

        
    group_block->insert(group_block->end(), new_op); 
  }

  // step 3: Replace outputs of inner ops
  std::vector<pir::OpResult> group_outs = new_group_op->results();
  std::unordered_set<pir::Operation*> inner_ops(group_ops.begin(),
                                                group_ops.end());
  for (size_t i = 0; i < outputs.size(); ++i) {
    outputs[i].ReplaceUsesWithIf(group_outs[i],
                                 [&inner_ops](pir::OpOperand op) {
                                   return !inner_ops.count(op.owner());
                                 });
  }

  // step 4: Insert YieldOp for outputs
  builder.SetInsertionPointToBlockEnd(group_block);
  builder.Build<::pir::YieldOp>(outputs);

  return new_group_op;
}




std::vector< GroupClusterNode > GroupSplit( ::pir::Operation* input_op) 
{
    std::cerr << "group op cluster!!!!!!!!!!!!!\n";

    cinn::dialect::GroupOp group_op = input_op->dyn_cast<cinn::dialect::GroupOp>();
    ::pir::IrContext* ctx = ::pir::IrContext::Instance();
    auto target = cinn::common::DefaultNVGPUTarget();      
    
    auto inner_values = GetInnerGeneValue( group_op.ops() );
       
    std::unordered_map<::pir::Operation*, GroupClusterNode > op_path;


    auto op_list = group_op.ops();

    std::vector< GroupClusterNode  > first_stage_output;

    std::unordered_set<::pir::Operation*> yield_output_ops;
    auto yield_op = op_list.back();
    for( size_t i = 0; i < yield_op->num_operands(); ++i )
    {
      yield_output_ops.insert( yield_op->operand_source(i).defining_op() );
    }
    
    for ( auto* op : op_list)
    {
      if( op->name () == "cf.yield" )
      {
        continue;
      }

      auto& op_list = op_path[op].ops;
      for( size_t i = 0; i < op->num_operands(); ++i)
      {
        if( !inner_values.count( op->operand_source(i)))
        {
          continue;
        }
        
        auto pre_op = op->operand_source(i).defining_op();

        if( cinn::hlir::framework::pir::CompatibleInfo::OpKind(*pre_op) == cinn::hlir::framework::kReduction )
        {
          continue;
        }
        // get pre op all op list
        
        op_list.insert( op_list.end(), op_path[pre_op].ops.begin(), op_path[pre_op].ops.end() );
        
      }

      op_list.push_back( op ); 

      if( yield_output_ops.count(op) || cinn::hlir::framework::pir::CompatibleInfo::OpKind(*op) == cinn::hlir::framework::kReduction)
      {
        first_stage_output.push_back( op_path[op]);
      }
    }

    return first_stage_output;

  }

}  // namespace


class CinnGroupClusterPass : public pir::Pass {
 public:
  CinnGroupClusterPass() : pir::Pass("cinn_group_cluster_pass", /*opt_level=*/1) {}

  void Run(pir::Operation* op) override {
    auto module_op = op->dyn_cast<pir::ModuleOp>();
    IR_ENFORCE(module_op, "build_cinn_pass should run on module op.");
    auto& block = module_op.block();

    // std::vector<GroupOpsVec> groups =
    //     ::pir::SubgraphDetector(&block, IsSupportCinn)();
    // AddStatistics(groups.size());
    // for (auto& group_ops : groups) {
    //   VLOG(4) << "current group_ops.size(): " << group_ops.size();
    //   ::pir::ReplaceWithGroupOp(&block, group_ops);
    // }
    //for( auto& op : block)
    for( auto it = block.begin(); it != block.end(); )
    {
      auto base_it = it;
      
      ++it;
      if( base_it->isa<cinn::dialect::GroupOp>() )
      {
        auto split_res = GroupSplit( base_it );

        size_t index = 0;
        std::unordered_map<pir::Operation*, size_t> op2id;
         cinn::dialect::GroupOp group_op = base_it->dyn_cast<cinn::dialect::GroupOp>();
        for (auto op1 : group_op.ops() ) {
          op2id[op1] = index++;
        }

        ::pir::Operation* insert_point = base_it;
        for (auto& node : split_res) {


        std::vector<pir::Operation*> tmp_ops(node.ops.begin(),
                                         node.ops.end());
          
          std::sort(tmp_ops.begin(),
                    tmp_ops.end(),
                    [&op2id](pir::Operation* a, pir::Operation* b) {
                      return op2id.at(a) < op2id.at(b);
                    });
          
          for( auto op1 : tmp_ops )
          {
            std::cerr << "name  " << op1->name() << std::endl;
          }
          std::cerr << "fin!!!!!!!!!!! \n";
          insert_point = ReplaceWithGroupOp(&block, tmp_ops, insert_point);

          // ++it;
        }

        
        // auto yield_op = group_op.ops().back();
        // yield_op->Erase();

        base_it->Erase();
      } 
    }
  }

  bool CanApplyOn(pir::Operation* op) const override {
    return op->isa<pir::ModuleOp>() && op->num_regions() > 0;
  }
};

std::unique_ptr<pir::Pass> CreateCinnGroupClusterPass() {
  return std::make_unique<CinnGroupClusterPass>();
}

}  // namespace ir
}  // namespace dialect
}  // namespace cinn
