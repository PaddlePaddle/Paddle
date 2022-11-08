#pragma once

#include "paddle/fluid/framework/ir/fuse_pass_base.h"

namespace paddle {
namespace framework {
namespace ir {

class Graph;

class TrtSkipLayerNormMixedPrecisionPass : public FusePassBase {
 public:
  TrtSkipLayerNormMixedPrecisionPass() {
    AddOpCompat(OpCompat("elementwise_add"))
        .AddInput("X")
        .IsTensor()
        .End()
        .AddInput("Y")
        .IsTensor()
        .End()
        .AddOutput("Out")
        .IsTensor()
        .End()
        .AddAttr("axis")
        .IsIntIn({0, -1})
        .End();

    AddOpCompat(OpCompat("layer_norm"))
        .AddInput("X")
        .IsTensor()
        .End()
        .AddInput("Scale")
        .IsTensor()
        .End()
        .AddInput("Bias")
        .IsTensor()
        .End()
        .AddOutput("Y")
        .IsTensor()
        .End()
        .AddOutput("Mean")
        .IsTensor()
        .End()
        .AddOutput("Variance")
        .IsTensor()
        .End()
        .AddAttr("epsilon")
        .IsNumGE(0.0f)
        .IsNumLE(0.001f)
        .End()
        .AddAttr("begin_norm_axis")
        .IsNumGT(0)
        .End();
  }

  virtual ~TrtSkipLayerNormMixedPrecisionPass() {}

 protected:
  void ApplyImpl(ir::Graph* graph) const override;
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle 
