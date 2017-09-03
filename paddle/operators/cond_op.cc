#include "paddle/operators/switch_op.h"

namespace paddle {
namespace operators {

void CondOp::InferShape(const std::shared_ptr<Scope>& scope) const {
  // Create two Nets
  // Create two scopes
  for (int i = 0; i < 2; ++i)
    sub_scope.push_back(scope.NewScope());

  for (int i = 0; i < 2; ++i)
    sub_net_op_[i].InferShape(sub_scope[i]);

  for (int i = 0; i < 2; ++i)
    tensor_index = new Tensor();

  for (int i = 0; i < 2; ++i)
    _index.push_back(vector<int>());
  
  for (int i = 0; i < 2; ++i)
  {
    // for (auto& input : net_op_[i]->Inputs()) {
    for (auto& input : GetAttr<std::vector<std::string>>("True_inputs")) {
      auto var_name = input.second;
      // Create a new tensor in sub-scope for input-type tensor
      sub_scope[i]->NewVar(var_name)->GetMutable<Tensor>();
    }
  }
}

class CondOpProtoAndCheckerMaker : public OpProtoAndCheckerMaker {
public:
  CondOpProtoAndCheckerMaker(OpProto *proto, OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("Cond", "The condition, which is a bool vector");
    AddInput("Xs", "Inputs of Subnets");
    AddAttr<std::vector<std::string>>("sub_inputs", "Inputs of the Whole Op, net op and so forth");
    AddAttr<std::vector<std::string>>("sub_outputs", "True Outputs needs merge");
    AddOutput("Outs", "The output of cond op");

    AddComment(R"DOC(
Sample dependent Cond Operator:
The equation is: Out[i] = subnet_t[i], if Cond[i] == true
Out[i] = subnet_t[i], if Cond[i] == false
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OP_WITHOUT_GRADIENT(cond_op,
            paddle::operators::CondOp,
            paddle::operators::CondOpProtoAndCheckerMaker);

