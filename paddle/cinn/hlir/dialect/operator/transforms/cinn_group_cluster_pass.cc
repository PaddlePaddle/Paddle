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
#include "paddle/pir/core/builtin_dialect.h"
#include "paddle/pir/pass/pass.h"
#include "paddle/pir/pattern_rewrite/pattern_applicator.h"
#include "paddle/pir/pattern_rewrite/pattern_match.h"
#include "paddle/pir/pattern_rewrite/pattern_rewrite_driver.h"
#include "paddle/cinn/hlir/dialect/operator/ir/manual_op.h"
#include "paddle/fluid/pir/transforms/sub_graph_detector.h"
#include "paddle/pir/dialect/control_flow/ir/cf_dialect.h"
#include "paddle/pir/dialect/control_flow/ir/cf_op.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/group_merge/op_with_group_merge_util.h"

namespace cinn {
namespace dialect {
namespace ir {

namespace {

using cinn::hlir::framework::pir::ScheduleInfoNode;

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

std::unordered_set<::pir::Value> GetListOutsideInput( const std::vector<::pir::Operation*>& ops)
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
      }
      return outside_ops;
}


struct GroupClusterNode
{
    std::vector<::pir::Operation*> ops;
    cinn::hlir::framework::OpPatternKind group_kind{cinn::hlir::framework::kElementWise};
    std::vector<int64_t> reduce_axis;
    std::vector<int64_t> loop_ranges;

    std::unordered_map<::pir::Operation*, std::vector<ScheduleInfoNode> > alignment_schedule_info;
    
    std::unordered_set<::pir::Value> GetOutsideInput()
    {
      
      return GetListOutsideInput(ops);
    }

    std::string DebugStr()
    {
        std::stringstream ss;
      ::pir::IrPrinter printer(ss);
      
      ss << "type " << group_kind << "\n";
      ss << "loop range\t";
      
      for( auto d : loop_ranges)
      {
        ss << ", " << d;
      }
      ss << "\n";
      ss << "reduce axis \t";
      for( auto d : reduce_axis)
      {
        ss << ", " << d;
      }
      ss << "\n";
      
      for (auto op : ops) {
        printer.PrintOperation( op );
        if( alignment_schedule_info.count(op) )
        {
          for( auto& node : alignment_schedule_info.at(op) ) 
          {
            ss << node.DebugStr();
          }
        }
        ss << "\n";
      }

      return ss.str();
    }

    void GenerateOutputValue( const std::unordered_set<::pir::Value>& outside_need_value)
    {
      output_value.clear();
      for( auto & op : ops)
      {
        if( op->name() == "cf.yield")
        {
          continue;
        }
 
        std::unordered_set<::pir::Value> inserted_val;
        for( size_t i = 0; i < op->num_results(); ++i )
        {
          if( outside_need_value.count( op->result(i) ) )
          {
            if( ! inserted_val.count( op->result(i)))
            {
              output_value.push_back( op->result(i) );

              inserted_val.insert( op->result(i));
            }
          }
        } 
      }
    }

    void MergeNode( const GroupClusterNode& node, const ScheduleInfoNode& sch_node)
    {
      std::unordered_set<::pir::Operation*> inner_ops( ops.begin(), ops.end() );

      if( sch_node.type != "")
      {
        // all the data need add sch node
        for( auto it = alignment_schedule_info.begin(); it != alignment_schedule_info.end(); ++it )
        {
          it->second.push_back( sch_node );
        }
      }
      for( auto op : node.ops)
      {
        if( ! inner_ops.count( op ))
        {
          ops.push_back( op );  
          // copy align info
          if( node.alignment_schedule_info.count( op ) )
          {
            alignment_schedule_info[op] = node.alignment_schedule_info.at(op);        
          }
        }
      }
      

      if( group_kind < node.group_kind )
      {
        group_kind = node.group_kind;
      }

      if ( node.group_kind ==  cinn::hlir::framework::kReduction)
      {
        reduce_axis = node.reduce_axis;
        loop_ranges = node.loop_ranges;
      }

    }

    std::vector<::pir::Value> output_value;
};

::pir::Operation* ReplaceWithGroupOp(::pir::Block* block,
                        const ::pir::GroupOpsVec& group_ops, 
                        const std::vector<::pir::Value>& output_value,
                        const std::unordered_map<::pir::Operation*, std::vector<ScheduleInfoNode> >& alignment_schedule_info,
                        ::pir::Operation* insert_op,
                        ::pir::IrMapping* ir_mapping ) 
{
  ::pir::IrContext* ctx = ::pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<cinn::dialect::OperatorDialect>();
  ctx->GetOrRegisterDialect<::pir::ControlFlowDialect>();
  ::pir::Builder builder = ::pir::Builder(ctx, block);
  // step 1: Ensure the insert point and create GroupOp here.
  auto* last_op = group_ops.back();
  builder.SetInsertionPointAfter(insert_op);
  std::vector<pir::Type> output_types;
  //std::vector<pir::Value> outputs = ::pir::AnalysisOutputs(group_ops);

 //  ::pir::IrMapping ir_mapping;
  for (auto& value : output_value) {
    output_types.emplace_back(value.type());   
  }
  // step 2: Replace the old op with GroupOp.
  auto new_group_op = builder.Build<cinn::dialect::GroupOp>(output_types, alignment_schedule_info);
  pir::Block* group_block = new_group_op.block();

  
  ::pir::CloneOptions clone_options(false, true, false);

  //  std::stringstream ss;
  //  ::pir::IrPrinter printer(ss);
   
  // for (auto op : group_ops) {
  //   printer.PrintOperation( op );
  //   ss << "\n";
  // }

  // std::cerr << "program \n" << ss.str() << std::endl;

  for (auto op : group_ops) {
    auto new_op = op->Clone(*ir_mapping, clone_options);
        
    group_block->insert(group_block->end(), new_op); 
  }

  // step 3: Replace outputs of inner ops
  std::vector<pir::OpResult> group_outs = new_group_op->results();
  std::unordered_set<pir::Operation*> inner_ops(group_ops.begin(),
                                                group_ops.end());

  std::vector<::pir::Value> new_output;
  for( size_t i = 0; i < output_value.size(); ++i )
  {
    new_output.push_back( ir_mapping->Lookup<::pir::Value>( output_value[i]));
  }
  builder.SetInsertionPointToBlockEnd(group_block);
  builder.Build<::pir::YieldOp>(new_output);

  return new_group_op;
}

bool CanFuse( const GroupClusterNode& first, const GroupClusterNode& second, ScheduleInfoNode* sch_node )
{
  if( ( first.group_kind == cinn::hlir::framework::kReduction && second.group_kind == cinn::hlir::framework::kElementWise)
    || (first.group_kind == cinn::hlir::framework::kReduction && second.group_kind == cinn::hlir::framework::kBroadcast) )
  {
    if( first.loop_ranges == second.loop_ranges )
    {
      return true;
    }
    std::set<int64_t> reduce_axis;
    for( auto axis : first.reduce_axis)
    {
      if( axis < 0 )
      {
        axis += first.loop_ranges.size();
      }

      reduce_axis.insert( axis);
    }
    if( (first.loop_ranges.size() != second.loop_ranges.size())  && (first.loop_ranges.size() != second.loop_ranges.size() + first.reduce_axis.size( )) )
    {
      return false;
    }
    size_t second_index = 0;
    for( size_t i = 0;  i < first.loop_ranges.size(); ++i )
    {
      if( ! reduce_axis.count( i ))
      {
        if( first.loop_ranges[i] != second.loop_ranges[ second_index++ ])
        {
          return false;
        }
      }
      else
      {
        if(first.loop_ranges.size() == second.loop_ranges.size())
        {          
           if ( ( second.loop_ranges[ second_index++ ] != 1 ) )
           {
             return false;
           }
        }
      }

    }

    sch_node->type = "broadcast";
    sch_node->axis_info = first.reduce_axis;
    sch_node->factor_info = first.loop_ranges;
    return true;
  }
  
  return ( first.loop_ranges == second.loop_ranges);
}


std::vector< GroupClusterNode > GroupSplit( ::pir::Operation* input_op) 
{
  
    cinn::dialect::GroupOp group_op = input_op->dyn_cast<cinn::dialect::GroupOp>();
    ::pir::IrContext* ctx = ::pir::IrContext::Instance();
    auto target = cinn::common::DefaultNVGPUTarget();      
    
    auto inner_values = GetInnerGeneValue( group_op.GetOperators() );
       
    std::unordered_map<::pir::Operation*, GroupClusterNode > op_path;


    auto op_list = group_op.GetOperators();

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

      auto& cluster_node =  op_path[op];
      auto& op_list = cluster_node.ops;
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

        cluster_node.alignment_schedule_info = op_path[pre_op].alignment_schedule_info;
        cluster_node.group_kind = op_path[pre_op].group_kind;
        cluster_node.loop_ranges = op_path[pre_op].loop_ranges;
      }

      // process cluster node
      if( cinn::hlir::framework::pir::CompatibleInfo::OpKind(*op) == cinn::hlir::framework::kReduction )
      { 
        // set reduce axis and loop range
        cluster_node.reduce_axis = cinn::dialect::ir::GetVectorAttr(op, "dim");
        cluster_node.loop_ranges = phi::vectorize( op->operand_source(0)
                   .type()
                   .dyn_cast<paddle::dialect::DenseTensorType>()
                   .dims() );
         cluster_node.group_kind = cinn::hlir::framework::kReduction;
      }
      else if ( cinn::hlir::framework::pir::CompatibleInfo::OpKind(*op) == cinn::hlir::framework::kElementWise )
      {
        if ( cluster_node.group_kind == cinn::hlir::framework::kElementWise )
        {
          if( op->name() != "cinn_op.reshape")
          {
           cluster_node.loop_ranges = phi::vectorize( op->result(0)
                   .type()
                   .dyn_cast<paddle::dialect::DenseTensorType>()
                   .dims() );
          }
        }
      }
      else if( cinn::hlir::framework::pir::CompatibleInfo::OpKind(*op) == cinn::hlir::framework::kBroadcast )
      {
        if ( cluster_node.group_kind == cinn::hlir::framework::kElementWise )
        {
         
          cluster_node.loop_ranges = phi::vectorize( op->result(0)
                    .type()
                    .dyn_cast<paddle::dialect::DenseTensorType>()
                    .dims() );                    
        }

        // all the op in cluster node must add broadcast 
        ScheduleInfoNode sch_node;
        sch_node.type = "brodacast";
        sch_node.axis_info = cinn::dialect::ir::GetVectorAttr( op, "broadcast_axes");
        sch_node.factor_info = cinn::dialect::ir::GetVectorAttr( op, "out_shape");

        for( auto op : op_list)
        {
          cluster_node.alignment_schedule_info[op].push_back( sch_node );
        }

        cluster_node.group_kind = cinn::hlir::framework::kBroadcast;
      }
      else
      {
        throw std::runtime_error("not support op kind yet");
      }

      op_list.push_back( op ); 

      if( yield_output_ops.count(op) || cinn::hlir::framework::pir::CompatibleInfo::OpKind(*op) == cinn::hlir::framework::kReduction)
      {
        first_stage_output.push_back( op_path[op]);
      }
    }



    // stage 2 merge
    // for now we merge node in same pass
    // only for vertial fuse    
   
    std::unordered_set<::pir::Value> all_needed_values;
      for( auto & node : first_stage_output )
      {
          auto node_outside_input = node.GetOutsideInput();
          all_needed_values.insert( node_outside_input.begin(), node_outside_input.end() );
      }

    
    
    std::unordered_map<::pir::Value, size_t> out_value_to_node_id;
    for( size_t i = 0; i < first_stage_output.size(); ++i )
    {
      auto & node = first_stage_output[i];
        node.GenerateOutputValue(all_needed_values);

        

      for( auto out_val : node.output_value )
        out_value_to_node_id[ out_val ] = i;
    }

    // sort the id   
    if( first_stage_output.size() <= 1 )
    {
      return first_stage_output;
    }

    std::set<int> fused_index;
    std::vector< GroupClusterNode  >  second_stage_output;
    for( int i = first_stage_output.size() - 1; i >= 0; --i )
    { 
      if( fused_index.count( i ))
      {
        
        continue;
      }
      auto& node = first_stage_output[i];
      auto node_outside_input = node.GetOutsideInput();

      GroupClusterNode new_node = node;
      
      for( auto in_val : node_outside_input )
      {
        // get pre id
        if( out_value_to_node_id.count( in_val ) )
        {
         
          auto pre_id = out_value_to_node_id.at( in_val );
          
          // can new_node merge with pre_id node
          auto& pre_node = first_stage_output[pre_id];
          
          ScheduleInfoNode sch_node;
          auto can_fuse =  CanFuse( pre_node, new_node, &sch_node);

          if( can_fuse )
          {
            // merge pre node to new_node
            new_node.MergeNode( pre_node, sch_node );
           
            fused_index.insert( pre_id);
          }
          else
          {
            second_stage_output.insert( second_stage_output.begin(), pre_node);
          }

        }
      }
      second_stage_output.insert( second_stage_output.end(), new_node);
     

    }
      
    // stage 3
    
    std::vector< GroupClusterNode  >  third_stage_output;
    auto reset_node = second_stage_output;
    while( true )
    {
      GroupClusterNode new_node = reset_node[0];
    
      std::vector< GroupClusterNode  > temp;
      for( size_t i = 1; i < reset_node.size(); ++i)
      {
        auto& pre_node = reset_node[i];
        ScheduleInfoNode sch_node;
        auto can_fuse =  CanFuse( new_node, pre_node, &sch_node);
        if( can_fuse )
        {
          new_node.MergeNode( pre_node, sch_node );
          
        }
        else
        {
          temp.push_back( pre_node );
        }
      }

      third_stage_output.push_back( new_node );


      if( temp.size() == 0 )
      {
        break;
      }

      temp.swap( reset_node );

    }

    
    return third_stage_output;

  }

}  // namespace


class CinnGroupClusterPass : public pir::Pass {
 public:
  CinnGroupClusterPass() : pir::Pass("cinn_group_cluster_pass", /*opt_level=*/1) {}

  void Run(pir::Operation* op) override {
    auto module_op = op->dyn_cast<pir::ModuleOp>();
    IR_ENFORCE(module_op, "build_cinn_pass should run on module op.");
    auto& block = module_op.block();

    for( auto it = block.begin(); it != block.end(); )
    {
      auto base_it = it;
      
      ++it;
      if( base_it->isa<cinn::dialect::GroupOp>() )
      {
        ::pir::IrMapping ir_mapping;
        cinn::dialect::GroupOp group_op = base_it->dyn_cast<cinn::dialect::GroupOp>();

        auto group_outside_input = GetListOutsideInput( group_op.GetOperators());
        for( auto val : group_outside_input)
        {
          ir_mapping.Add( val, val);
        }

        auto split_res = GroupSplit( base_it );
        // need sort split res

        std::unordered_set<::pir::Value> all_ouput_values;
        for( auto & node : split_res )
        {
            auto node_outside_input = node.GetOutsideInput();
            all_ouput_values.insert( node_outside_input.begin(), node_outside_input.end() );
        }

        size_t index = 0;
        std::unordered_map<pir::Operation*, size_t> op2id;
         
        for (auto op1 : group_op.GetOperators() ) {
          op2id[op1] = index++;
        }

        auto yield_op = group_op.GetOperators().back();
        for( size_t i = 0; i < yield_op->num_operands(); ++i )
        {
          all_ouput_values.insert( yield_op->operand_source(i));
        }

        ::pir::Operation* insert_point = base_it;
        for (auto& node : split_res) {

          node.GenerateOutputValue( all_ouput_values );
          std::vector<pir::Operation*> tmp_ops(node.ops.begin(),
                                         node.ops.end());
          
          std::sort(tmp_ops.begin(),
                    tmp_ops.end(),
                    [&op2id](pir::Operation* a, pir::Operation* b) {
                      return op2id.at(a) < op2id.at(b);
                    });
          
          auto node_outside_input = node.GetOutsideInput();
         
          

          insert_point = ReplaceWithGroupOp(&block, tmp_ops, node.output_value, node.alignment_schedule_info, insert_point, &ir_mapping);

          for(size_t i = 0; i < node.output_value.size(); ++i)
          {
            ir_mapping.Add( node.output_value[i], insert_point->result(i));
          } 
          
          std::unordered_set<::pir::Value> local_outs( node.output_value.begin(), node.output_value.end() );

          int local_index = 0;
          for(  size_t i = 0; i < yield_op->num_operands(); ++i )
          {
            if( local_outs.count( yield_op->operand_source(i)) )
            {
              base_it->result(i).ReplaceAllUsesWith(  insert_point->result( local_index ++ ));
            }
          }
        }

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
