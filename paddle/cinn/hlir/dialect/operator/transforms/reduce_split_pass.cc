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

#include "paddle/cinn/hlir/dialect/operator/transforms/add_broadcast_to_elementwise_pass.h"

#include "paddle/cinn/hlir/dialect/operator/ir/cinn_op.h"
#include "paddle/cinn/hlir/dialect/operator/ir/manual_op.h"
#include "paddle/cinn/hlir/framework/pir/utils.h"
#include "paddle/common/ddim.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/pir/include/core/builtin_dialect.h"
#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pattern_rewrite/frozen_rewrite_pattern_set.h"
#include "paddle/pir/include/pattern_rewrite/pattern_applicator.h"
#include "paddle/pir/include/pattern_rewrite/pattern_match.h"
#include "paddle/pir/include/pattern_rewrite/pattern_rewrite_driver.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/group_merge/op_with_group_merge_util.h"

namespace cinn {
namespace dialect {
namespace ir {

class ReduceSplitPattern
    : public pir::OpRewritePattern<cinn::dialect::ReduceSumOp> {
 public:
  using pir::OpRewritePattern<cinn::dialect::ReduceSumOp>::OpRewritePattern;

  bool MatchAndRewrite(cinn::dialect::ReduceSumOp sum_op,
                       pir::PatternRewriter& rewriter) const override {
    auto reduce_dim = cinn::dialect::ir::GetVectorAttr(sum_op, "dim");
    auto input_dim = sum_op->operand_source(0).type().dyn_cast<pir::DenseTensorType>().dims();

    size_t reduce_num = 1;
    for( auto axis : reduce_dim)
    {
      if( axis < 0)
      {
        axis += input_dim.size();
      }
      if( input_dim[axis] < 0 )
      {
        return false;
      }

      reduce_num *= input_dim[axis];
    }

    if( reduce_num > 4096 )
    {
      // reduce split
      // leave last 8 num for now
      int split_axis = -1;
      int first_8_numel = 1;
     
      std::vector<int64_t> split_factor;
      for( auto axis : reduce_dim)
      {
        if( axis < 0)
        {
          axis += input_dim.size();
        }
        first_8_numel *= input_dim[axis];
        if(first_8_numel == 8 )
        {
          split_axis = axis;
          break;
        }else if ( first_8_numel > 8 )
        {
          split_axis = axis;
          if( first_8_numel % 8 == 0)
          {
             auto second_fac = first_8_numel / 8;
             auto first_fac = input_dim[axis] / second_fac;
             split_factor.push_back( first_fac );
             split_factor.push_back( second_fac );
          }
          else
          {
            throw std::runtime_error("not support not divide by 8 yet");
          }
           
          break;
        }

      }

      std::vector<int> output_shape;

      auto input_x =   sum_op.operand_source(0);
      if( split_factor.size() == 2 )
      {
        // build output shape
        for( int i = 0; i < input_dim.size(); ++i )
        {
          if( i != split_axis)
          {
            output_shape.push_back( input_dim[i]);
          }
          else
          {
            output_shape.push_back( split_factor.front() );
            output_shape.push_back( split_factor.back() );
          }
        }

        for( auto& d : output_shape)
        {
          std::cerr << "out shape " << d << std::endl;
        }

        input_x = rewriter.Build<cinn::dialect::ReshapeOp>( sum_op.operand_source(0), output_shape ).result(0);
      }

      std::vector<int64_t> first_reduce_axis;
      std::vector<int64_t> second_reduce_axis; 

      std::cerr << "split axis " <<  split_axis << std::endl;

      for( auto axis : reduce_dim)
      {
        if( axis < 0)
        {
          axis += input_dim.size();
        }

        if( axis < split_axis )
        {
          second_reduce_axis.push_back( axis);
        }
        else if( axis == split_axis )
        {
          second_reduce_axis.push_back( axis );
          if( split_factor.size() == 2 )
          {
            first_reduce_axis.push_back( axis + 1 );
          }
        }
        else
        {
          if( split_factor.size() == 2 )
          {
             first_reduce_axis.push_back( axis + 1); 
          }
          else
          {
            first_reduce_axis.push_back( axis );
          }
          
        }

      }

      for( auto& d : first_reduce_axis)
      {
        std::cerr << "dd1   " << d << std::endl;
      }

      for( auto& d : second_reduce_axis)
      {
        std::cerr << "dd2   " << d << std::endl;
      }

      bool orig_keep_dim = sum_op.attribute("keep_dim").dyn_cast<pir::BoolAttribute>().data();

      auto first_reduce_out = rewriter.Build<cinn::dialect::ReduceSumOp>( input_x, first_reduce_axis, orig_keep_dim ).result(0);

      std::cerr << "first reduce out " << first_reduce_out.type().dyn_cast<paddle::dialect::DenseTensorType>().dims() << std::endl;
      auto second_reduce_out = rewriter.Build<cinn::dialect::ReduceSumOp>( first_reduce_out, second_reduce_axis, orig_keep_dim ).result(0);


      if( orig_keep_dim && split_factor.size() == 2)
      {
        // need reshape 
        std::vector<int> final_out_shape = phi::vectorize<int>( input_dim);
        for( auto axis : reduce_dim)
        {
          if(  axis < 0 )
          {
            axis += input_dim.size();
          }

          final_out_shape[axis]= 1;
        }
        for( auto& d : final_out_shape)
        {
          std::cerr << "out shape " << d << std::endl;
        }
        
        second_reduce_out = rewriter.Build<cinn::dialect::ReshapeOp>( second_reduce_out, final_out_shape ).result(0);

        for( auto& d : final_out_shape)
        {
          std::cerr << "out shape " << d << std::endl;
        }
      }

      rewriter.ReplaceAllUsesWith(sum_op.result(0),
                                  second_reduce_out);
      rewriter.EraseOp(sum_op);
      return true;
    }

    return false;
  }
};

class ReduceSplitPass : public pir::PatternRewritePass {
 public:
  ReduceSplitPass()
      : pir::PatternRewritePass("reduce_split_pass", 1) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext* context) override {
    pir::RewritePatternSet ps(context);
    ps.Add<ReduceSplitPattern>(context);
    

    return ps;
  }

  bool CanApplyOn(pir::Operation* op) const override {
    return op->num_regions() > 0;
  }
};

std::unique_ptr<pir::Pass> CreateReduceSplitPass() {
  return std::make_unique<ReduceSplitPass>();
}

}  // namespace ir
}  // namespace dialect
}  // namespace cinn
