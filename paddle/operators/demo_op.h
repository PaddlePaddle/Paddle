#pragma once

#include "paddle/framework/op_registry.h"

using namespace paddle::framework;

namespace paddle {
namespace operators {

class CosineOp : public OperatorWithKernel {
 public:
  void Run(const OpRunContext *context) const override {
    printf("%s\n", DebugString().c_str());
  }
};

class CosineOpProtoAndCheckerMaker : public OpProtoAndCheckerMaker {
 public:
  CosineOpProtoAndCheckerMaker(OpProto *proto, OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("input", "input of cosine op");
    AddOutput("output", "output of cosine op");
    AddAttr<float>("scale", "scale of cosine op")
        .SetDefault(1.0)
        .LargerThan(0.0);
    AddType("cos");
    AddComment("This is cos op");
  }
};

REGISTER_OP(CosineOp, CosineOpProtoAndCheckerMaker, cos_sim)

class MyTestOp : public OperatorWithKernel {
 public:
  void Run(const OpRunContext *context) const override {
    printf("%s\n", DebugString().c_str());
  }
};

class MyTestOpProtoAndCheckerMaker : public OpProtoAndCheckerMaker {
 public:
  MyTestOpProtoAndCheckerMaker(OpProto *proto, OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("input", "input of cosine op");
    AddOutput("output", "output of cosine op");
    auto my_checker = [](int i) {
      PADDLE_ENFORCE(i % 2 == 0, "'test_attr' must be even!");
    };
    AddAttr<int>("test_attr", "a simple test attribute")
        .AddCustomChecker(my_checker);
    AddType("my_test_op");
    AddComment("This is my_test op");
  }
};

REGISTER_OP(MyTestOp, MyTestOpProtoAndCheckerMaker, my_test_op)

}  // namespace operators
}  // namespace operators
