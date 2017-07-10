#include <glog/logging.h>
#include <cstring>

#include "paddle/framework/recurrent_network_op.h"
#include "paddle/framework/tensor.h"

namespace paddle {
namespace framework {

// fake op implementations
namespace fake {
class FcOp : public OperatorBase {
 public:
  FcOp(const OpDesc& desc) {}

  virtual void InferShape(const Scope* scope) const override {
    LOG(INFO) << "fc InferShape";
  }

  virtual void Run(OpRunContext* contex) const override {
    LOG(INFO) << "fc Run";
  }

 private:
  std::string name_;
};
};  // namespace fake

void PlainNet::AddOp(const OpDesc& desc) {
  if (desc.type() == "fc") {
    ops_.emplace_back(new fake::FcOp(desc));
  }
}

void RecurrentOp::Run(OpRunContext* contex) const {
  auto scope = contex->scope;

  if (!scope->HasVariable(net_name_)) {
    CreateStepNet(scope);
  }
  Variable* net = scope->GetVariable(net_name_);
  PADDLE_ENFORCE(net, "failed to get step net");

  CreateScopes(scope);
  SegmentInputs(scope);

  Variable* step_scopes = scope->GetVariable(step_scopes_name_);
  PADDLE_ENFORCE(step_scopes, "failed to get step scopes");
  // forward
  auto dims = Input(scope, 0)->GetMutable<Tensor>()->dims();
  size_t seq_len = dims[1];
  auto& scopes = *step_scopes->GetMutable<std::vector<Scope*>>();
  for (size_t step_id = 0; step_id < seq_len; step_id++) {
    Scope* step_scope = scopes[step_id];
    // TODO replace memorys' copy with reference
    LinkMemories(scope, scopes, step_id);

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
  NetDesc desc;
  desc.name_ = "rnn_step_net";
  var->Reset<PlainNet>(new PlainNet(desc));
  // TODO add op descs
}

void RecurrentOp::LinkMemories(Scope* scope, std::vector<Scope*>& step_scopes,
                               size_t step) const {
  PADDLE_ENFORCE(step < step_scopes.size(),
                 "step [%d] out of range of step scopes' size [%d]", step,
                 step_scopes.size());
  // copy boot memory
  for (auto& attr : memory_attrs_) {
    Scope* step_scope = step_scopes[step];

    Tensor* boot_tensor{nullptr};
    Variable* memory_var = step_scope->CreateVariable(attr.pre_var);
    if (step == 0) {
      PADDLE_ENFORCE(scope->HasVariable(attr.boot_var),
                     "memory [%s]'s boot variable [%s] not exists", attr.var,
                     attr.boot_var);
      // update memory's ddim
      boot_tensor = scope->CreateVariable(attr.boot_var)->GetMutable<Tensor>();
      attr.dims = boot_tensor->dims();
    }

    // copy from boot memory
    // TODO support more device
    float* memory_tensor_val =
        memory_var->GetMutable<Tensor>()->mutable_data<float>(
            attr.dims, platform::CPUPlace());
    if (step == 0) {
      PADDLE_ENFORCE(boot_tensor, "boot_tensor should be retrieved before");
      // copy from boot memory
      std::memcpy(memory_tensor_val, boot_tensor->data<float>(),
                  product(attr.dims));
    } else {
      // copy from previous step scope's memory to this scope's `pre-memory`
      Tensor* pre_step_memory =
          step_scopes[step - 1]->GetVariable(attr.var)->GetMutable<Tensor>();
      std::memcpy(memory_tensor_val, pre_step_memory->data<float>(),
                  product(attr.dims));
    }
  }
}

}  // namespace framework
}  // namespace paddle
