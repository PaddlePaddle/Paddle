#include "paddle/framework/recurrent_network_op.h"
#include "paddle/framework/tensor.h"

namespace paddle {
namespace framework {

void RecurrentOp::Run(OpRunContext* contex) const {
  auto scope = contex->scope;

  if (!scope->HasVariable(net_name_)) {
    CreateStepNet(scope);
  }
  Variable* net = scope->GetVariable(net_name_);
  PADDLE_ENFORCE(net, "failed to get step net");

  CreateScopes(scope);
  SegmentInputs(scope);
  CreateMemories(scope);

  Variable* step_scopes = scope->GetVariable(step_scopes_name_);
  PADDLE_ENFORCE(step_scopes, "failed to get step scopes");
  // forward
  auto dims = Input(scope, 0)->GetMutable<Tensor>()->dims();
  size_t seq_len = dims[1];
  auto& scopes = *step_scopes->GetMutable<std::vector<Scope*>>();
  for (size_t step_id = 0; step_id < seq_len; step_id++) {
    Scope* step_scope = scopes[step_id];
    // TODO replace memorys' copy with reference
    // copy pre-memory
    for (const auto& attr : memory_attrs_) {
      Variable* pre_memory_var = step_scope->CreateVariable(attr.pre_var);
      // copy boot_var to current memory in first step
      if (step_id == 0) {
        Variable* boot_var = step_scope->GetVariable(attr.boot_var);
        *pre_memory_var->GetMutable<Tensor>() = *boot_var->GetMutable<Tensor>();
        // copy varible of memory in previous scope to current pre-memory
      } else {
        Variable* pre_state_var = scopes[step_id - 1]->GetVariable(attr.var);
        *pre_memory_var->GetMutable<Tensor>() =
            *pre_state_var->GetMutable<Tensor>();
      }
    }

    net->GetMutable<PlainNet>()->Run(step_scope);
  }

  // prepare outputs
  ConcateOutputs(scope);
}

void RecurrentOp::CreateScopes(Scope* scope) const {
  auto dims = Input(scope, 0)->GetMutable<Tensor>()->dims();
  size_t seq_len = dims[1];
  Variable* scopes_var = scope->GetVariable(step_scopes_name_);
  // auto step_scopes =
  // scopes_var->GetMutable<std::vector<std::shared_ptr<Scope>>>();
  auto step_scopes = scopes_var->GetMutable<std::vector<Scope*>>();
  // TODO Only two scopes are needed for inference, this case will be supported
  // later.
  if (seq_len > step_scopes->size()) {
    for (size_t i = step_scopes->size(); i < seq_len; ++i) {
      // step_scopes->push_back(std::make_shared<Scope>(
      // std::shared_ptr<Scope>(scope)));
      step_scopes->push_back(new Scope(std::shared_ptr<Scope>(scope)));
    }
  }
}

void RecurrentOp::CreateStepNet(Scope* scope) const {
  Variable* var = scope->CreateVariable(net_name_);
  auto step_net = GetAttr<std::string>("step_net");
  // get the step net proto from the string.
  // PADDLE_ENFORCE(
  //   google::protobuf::TextFormat::ParseFromString(step_net,
  //   &step_net_desc_));
  // this is a fake net, it will be rewrite after the network has been merged.
  var->Reset<PlainNet>(new PlainNet(step_net));
}

void RecurrentOp::CreateMemories(Scope* scope) const {
  Variable* scopes_var = scope->CreateVariable(step_scopes_name_);
  auto scopes = scopes_var->GetMutable<std::vector<Scope*>>();
  PADDLE_ENFORCE(!scopes->empty(), "step scopes should be created before.");

  PADDLE_ENFORCE(!memory_attrs_.empty(),
                 "memory attributes should be provided.");
  for (size_t i = 0; i < scopes->size(); i++) {
    for (const auto& attr : memory_attrs_) {
      // check boot var exists
      PADDLE_ENFORCE(scope->HasVariable(attr.boot_var),
                     "boot var %s not in context scope", attr.boot_var);
      // create the memory in this scope
      scope->CreateVariable(attr.var);
      // create pre-memory in this scope
      scope->CreateVariable(attr.pre_var);
      // TODO reference pre-memory to the memory in previous scope if Variance
      // supports reference
    }
  }
}

}  // namespace framework
}  // namespace paddle
