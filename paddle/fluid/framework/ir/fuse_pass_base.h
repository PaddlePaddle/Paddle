#pragma once

#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/pass.h"
#include "paddle/fluid/framework/scope.h"

namespace paddle {
namespace framework {
namespace ir {

static const char kParamScopeAttr[] = "param_scope";

class FusePassBase : public Pass {
 public:
  void Init(Graph* graph) const { graph_ = graph; }

  Scope* param_scope() {
    PADDLE_ENFORCE(graph_->Has(kParamScopeAttr));
    return graph_->Get<framework::Scope*>(kParamScopeAttr);
  }

  virtual ~FusePassBase() {}

 protected:
  mutable Graph* graph_;
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle
