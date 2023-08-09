#pragma once

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/var_type.h"
#include "paddle/fluid/framework/new_executor/standalone_executor.h"



namespace paddle {
namespace operators {

class StaticPyLayerOp : public framework::OperatorBase {
    public:
        StaticPyLayerOp(const std::string &type,
                        const framework::VariableNameMap &inputs,
                        const framework::VariableNameMap &outputs,
                        const framework::AttributeMap &attrs)
                    : framework::OperatorBase(type, inputs, outputs, attrs) {}

    static const char kInputs[];
    static const char kOutputs[];
    static const char kScope[];
    static const char kSkipEagerDeletionVars[];
};


class StaticPyLayerForwardOpProtoMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput(StaticPyLayerOp::kInputs, "The input variables of the sub-block.")
        .AsDuplicable();
    AddOutput(StaticPyLayerOp::kOutputs, "The output variables of the sub-block.")
        .AsDuplicable();
    // TODO: Must Use std::vector here ? 
    AddOutput(StaticPyLayerOp::kScope,
              "(std::vector<Scope*>) The scope of static pylayer block.");
    AddAttr<framework::BlockDesc *>(
        "forward_block", "The step block of conditional block operator");
    AddAttr<framework::BlockDesc *>(
        "backward_block", "The backward block of conditional block operator");
    AddComment(R"DOC(StaticPyLayer operator

TO-DO: added by luqi


)DOC");
  }
};


}   // namespace operators
}   // namespace paddle