#include "paddle/operators/switch_op.h"

namespace paddle {
namespace operators {

// namespace if_else{


void CondOp::Init() override {
}

void InferShape(const std::shared_ptr<Scope>& scope) const override {
  subnet_t = GetAttr<std::string>("subnet_t");
  subnet_f = GetAttr<std::string>("subnet_f");

  // Create two Nets
  // I use the same style as Recurrent_op, but does it create the net?
  // can be called like
  Variable* net_t = scope.FindVar(subnet_t);
  Variable* net_f = scope.FindVar(subnet_f);

  net_op_t = scope.FindVar(net_t)->GetMutable<NetOp>();
  net_op_f = scope.FindVar(net_f)->GetMutable<NetOp>();

  // Create two scopes
  scope_t = scope.NewScope();
  scope_f = scope.NewScope();

  // check cond of size (batch_size), type bool
  net_op_t->InferShape(scope_t);
  net_op_f->InferShape(scope_f);

  // check net_op_t and net_op_f of exactly same shape?
}

void IfElseOp::Run(const std::shared_ptr<Scope>& scope,
                   const platform::DeviceContext& dev_ctx) const {
  /* step 1: create two subnets and scopes, supposed done in Infershape() */

  /* step 2: get true and false index */
  cond = Input(name.cond);
  // get condition tensor
  auto cond_tensor = scope.get<Tensor>(cond);
  // tensor to cpu, whatever device it used to be in
  cond_cpu.CopyFrom(cond_tensor, platform::CPUPlace());

  size_t batch_size = cond_cpu.dims()[0];

  // keep index of true and false to slice, clear them first before each batch
  true_index.clear();
  false_index.clear();
  	
  // get a DDim type variable dims, check dimension
  auto dims = input0.dims();
  for(int i=0; i<dims; i++) {
    if (cond_cpu->data[i])
      true_index.push_back(i);
    else
      false_index.push_back(i);
  }

  // turn true_index and false_index to tensors
  Tensor* true_index_tensor = new Tensor(true_index);
  Tensor* false_index_tensor = new Tensor(false_index);

  /* Step 3: Gather */
  { // True Scope
    // Create new stuff
    for (auto& input : net_op_t->inputs_) {
      scope_t.NewVar(input);
      if (input.type() != PARAMETER) { // gather and slice required
        // Get Tensor and gather
        Tensor* input_gather_ = scope_t.FindVar(input)->GetMutable<Tensor>();
        Tensor* input_full_ = scope.FindVar(input)->GetMutable<Tensor>();
        input_gather_ = Gather(input_full_, true_index_tensor);
      }
    }

    for (auto& output : net_op->outputs_) {
      scope_t.NewVar(output);
    }
      
    net_op_t.Run();
  }

  { // False Scope
    // Create new stuff
    for (auto& input : net_op_f->inputs_) {
      scope_f.NewVar(input);
      if (input.type() != PARAMETER) { // gather and slice required
        // Get Tensor and gather
        Tensor* input_gather_ = scope_f.FindVar(input)->GetMutable<Tensor>();
        Tensor* input_full_ = scope.FindVar(input)->GetMutable<Tensor>();
        input_gather_ = Gather(input_full_, false_index_tensor);
      }
    }

    for (auto& output : net_op->outputs_) {
      scope_t.NewVar(output);
    }

    net_op_f.Run();
  }

  /* Merge Output Together by scatter update */
  for (auto& ouput : outputs_) {
    Tensor* output_t = scope_t->FindVar(output)->GetMutable<Tensor>();
    Tensor* output_f = scope_f->FindVar(output)->GetMutable<Tensor>();
    Tensor* output_tensor = scope->FindVar(output)->GetMutable<Tensor>();
    Scatter(output_t, output_tensor, true_index_tensor);
    Scatter(output_f, output_tensor, false_index_tensor);
  }
}

}  // namespace operators
}  // namespace paddle

REGISTER_OP(ifelse_op,
            paddle::operators::IfElseOp,
            paddle::operators::RecurrentAlgorithmProtoAndCheckerMaker);
