#include "paddle/framework/recurrent_network_op.h"

namespace paddle {
namespace framework {

void RecurrentOp::Run(OpRunContext* contex) const {
  auto scope = contex->scope;

  Variable* net = scope->GetVariable(net_name_);
  if (net == nullptr) {
    CreateStepNet(scope);
    net = scope->GetVariable(net_name_);
  }
  PADDLE_ENFORCE(net, "failed to create step net");

  CreateScopes(scope);
  SegmentInputs(scope);
  PrepareMemories(scope);

  Variable* step_scopes = scope->GetVariable(step_scopes_name_);
  PADDLE_ENFORCE(step_scopes, "failed to get scopes");
  // forward
  for (Scope* step_scope : *step_scopes->GetMutable<std::vector<Scope*>>()) {
    net->GetMutable<PlainNet>()->Run(step_scope);
  }

  // prepare outputs
  ConcateOutputs(scope);
}

}  // namespace framework
}  // namespace paddle
