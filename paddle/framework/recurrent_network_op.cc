#include "paddle/framework/recurrent_network_op.h"
#include "paddle/framework/tensor.h"

namespace paddle {
namespace framework {

// fake op implementations
namespace fake {
class FcOp : public OperatorBase {
 public:
  FcOp(NetDesc& net_desc) : name_(net_desc.name) {}

  virtual void InferShape(const Scope* scope) const override {
    LOG(INFO) << "fc InferShape";
  }

  virtual void Run(OpRunContext* contex) const override {
    LOG(INFO) << "fc Run";
  }

 private:
  std::string name_;
};

class SGDOptimizerOp : public OperatorBase {
 public:
  FcOp(NetDesc& net_desc) : name_(net_desc.name) {}

  virtual void InferShape(const Scope* scope) const override {
    LOG(INFO) << "optimizer InferShape";
  }

  virtual void Run(OpRunContext* contex) const override {
    LOG(INFO) << "optimizer Run";
  }

 private:
  std::string name_;
};
};  // namespace fake

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
  auto& scopes = *step_scopes->GetMutable<std::vector<Scope*>>();
  for (size_t step_id = 0; step_id < scopes.size(); step_id++) {
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
