#pragma once

#include "paddle/fluid/framework/ir/param_opt_pass_base.h"

namespace paddle {
namespace framework {
namespace ir {

class AttentionLSTMFusePass : public ParamOptPassBase {
 protected:
  void RegisterParamOperations() const override;
  void Operate(Graph *graph, Scope *scope) const override;

};

}  // namespace ir
}  // namespace framework
}  // namespace paddle
