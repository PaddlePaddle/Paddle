#include "paddle/framework/op_registry.h"
#include <gtest/gtest.h>

namespace pd = paddle::framework;

namespace paddle {
namespace framework {
class CosineOp : public OperatorBase {
 public:
  using OperatorBase::OperatorBase;
  void Run(const Scope& scope,
           const platform::DeviceContext& dev_ctx) const override {}
};

class CosineOpProtoAndCheckerMaker : public OpProtoAndCheckerMaker {
 public:
  CosineOpProtoAndCheckerMaker(OpProto* proto, OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("input", "input of cosine op");
    AddOutput("output", "output of cosine op");
    AddAttr<float>("scale", "scale of cosine op")
        .SetDefault(1.0)
        .GreaterThan(0.0);
    AddComment("This is cos op");
  }
};

class MyTestOp : public OperatorBase {
 public:
  using OperatorBase::OperatorBase;
  void Run(const Scope& scope,
           const platform::DeviceContext& dev_ctx) const override {}
};

class MyTestOpProtoAndCheckerMaker : public OpProtoAndCheckerMaker {
 public:
  MyTestOpProtoAndCheckerMaker(OpProto* proto, OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("input", "input of cosine op").AsDuplicable();
    AddOutput("output", "output of cosine op").AsIntermediate();
    auto my_checker = [](int i) {
      PADDLE_ENFORCE(i % 2 == 0, "'test_attr' must be even!");
    };
    AddAttr<int>("test_attr", "a simple test attribute")
        .AddCustomChecker(my_checker);
    AddComment("This is my_test op");
  }
};
}  // namespace framework
}  // namespace paddle

static void BuildVar(const std::string& param_name,
                     std::initializer_list<const char*> arguments,
                     paddle::framework::OpDesc::Var* var) {
  var->set_parameter(param_name);
  for (auto& arg_name : arguments) {
    var->add_arguments(arg_name);
  }
}
REGISTER_OP_WITHOUT_GRADIENT(cos_sim, paddle::framework::CosineOp,
                             paddle::framework::CosineOpProtoAndCheckerMaker);
REGISTER_OP_WITHOUT_GRADIENT(my_test_op, paddle::framework::MyTestOp,
                             paddle::framework::MyTestOpProtoAndCheckerMaker);

TEST(OpRegistry, CreateOp) {
  paddle::framework::OpDesc op_desc;
  op_desc.set_type("cos_sim");
  BuildVar("input", {"aa"}, op_desc.add_inputs());
  BuildVar("output", {"bb"}, op_desc.add_outputs());

  float scale = 3.3;
  auto attr = op_desc.mutable_attrs()->Add();
  attr->set_name("scale");
  attr->set_type(paddle::framework::AttrType::FLOAT);
  attr->set_f(scale);

  auto op = paddle::framework::OpRegistry::CreateOp(op_desc);
  paddle::framework::Scope scope;
  paddle::platform::CPUDeviceContext dev_ctx;
  op->Run(scope, dev_ctx);
  float scale_get = op->Attr<float>("scale");
  ASSERT_EQ(scale_get, scale);
}

TEST(OpRegistry, IllegalAttr) {
  paddle::framework::OpDesc op_desc;
  op_desc.set_type("cos_sim");
  BuildVar("input", {"aa"}, op_desc.add_inputs());
  BuildVar("output", {"bb"}, op_desc.add_outputs());

  auto attr = op_desc.mutable_attrs()->Add();
  attr->set_name("scale");
  attr->set_type(paddle::framework::AttrType::FLOAT);
  attr->set_f(-2.0);

  bool caught = false;
  try {
    paddle::framework::OpRegistry::CreateOp(op_desc);
  } catch (paddle::platform::EnforceNotMet err) {
    caught = true;
    std::string msg = "larger_than check fail";
    const char* err_msg = err.what();
    for (size_t i = 0; i < msg.length(); ++i) {
      ASSERT_EQ(err_msg[i], msg[i]);
    }
  }
  ASSERT_TRUE(caught);
}

TEST(OpRegistry, DefaultValue) {
  paddle::framework::OpDesc op_desc;
  op_desc.set_type("cos_sim");
  BuildVar("input", {"aa"}, op_desc.add_inputs());
  BuildVar("output", {"bb"}, op_desc.add_outputs());

  ASSERT_TRUE(op_desc.IsInitialized());

  auto op = paddle::framework::OpRegistry::CreateOp(op_desc);
  paddle::framework::Scope scope;
  paddle::platform::CPUDeviceContext dev_ctx;
  op->Run(scope, dev_ctx);
  ASSERT_EQ(op->Attr<float>("scale"), 1.0);
}

TEST(OpRegistry, CustomChecker) {
  paddle::framework::OpDesc op_desc;
  op_desc.set_type("my_test_op");
  BuildVar("input", {"ii"}, op_desc.add_inputs());
  BuildVar("output", {"oo"}, op_desc.add_outputs());

  // attr 'test_attr' is not set
  bool caught = false;
  try {
    paddle::framework::OpRegistry::CreateOp(op_desc);
  } catch (paddle::platform::EnforceNotMet err) {
    caught = true;
    std::string msg = "Attribute 'test_attr' is required!";
    const char* err_msg = err.what();
    for (size_t i = 0; i < msg.length(); ++i) {
      ASSERT_EQ(err_msg[i], msg[i]);
    }
  }
  ASSERT_TRUE(caught);

  // set 'test_attr' set to an illegal value
  auto attr = op_desc.mutable_attrs()->Add();
  attr->set_name("test_attr");
  attr->set_type(paddle::framework::AttrType::INT);
  attr->set_i(3);
  caught = false;
  try {
    paddle::framework::OpRegistry::CreateOp(op_desc);
  } catch (paddle::platform::EnforceNotMet err) {
    caught = true;
    std::string msg = "'test_attr' must be even!";
    const char* err_msg = err.what();
    for (size_t i = 0; i < msg.length(); ++i) {
      ASSERT_EQ(err_msg[i], msg[i]);
    }
  }
  ASSERT_TRUE(caught);

  // set 'test_attr' set to a legal value
  op_desc.mutable_attrs()->Clear();
  attr = op_desc.mutable_attrs()->Add();
  attr->set_name("test_attr");
  attr->set_type(paddle::framework::AttrType::INT);
  attr->set_i(4);
  auto op = paddle::framework::OpRegistry::CreateOp(op_desc);
  paddle::platform::CPUDeviceContext dev_ctx;
  paddle::framework::Scope scope;
  op->Run(scope, dev_ctx);
  int test_attr = op->Attr<int>("test_attr");
  ASSERT_EQ(test_attr, 4);
}

class CosineOpComplete : public paddle::framework::CosineOp {
 public:
  DEFINE_OP_CONSTRUCTOR(CosineOpComplete, paddle::framework::CosineOp);
  DEFINE_OP_CLONE_METHOD(CosineOpComplete);
};

TEST(OperatorRegistrar, Test) {
  using namespace paddle::framework;
  OperatorRegistrar<CosineOpComplete, CosineOpProtoAndCheckerMaker> reg("cos");
}
