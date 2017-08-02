#include "paddle/framework/op_registry.h"
#include <gtest/gtest.h>

namespace pd = paddle::framework;

namespace paddle {
namespace framework {
class CosineOp : public OperatorBase {
 public:
  void Run(const Scope& scope,
           const platform::DeviceContext& dev_ctx) const override {}
  void InferShape(const Scope& scope) const override {}
};

class CosineOpProtoAndCheckerMaker : public OpProtoAndCheckerMaker {
 public:
  CosineOpProtoAndCheckerMaker(OpProto* proto, OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("input", "input of cosine op");
    AddOutput("output", "output of cosine op");
    AddAttr<float>("scale", "scale of cosine op")
        .SetDefault(1.0)
        .LargerThan(0.0);
    AddComment("This is cos op");
  }
};

class MyTestOp : public OperatorBase {
 public:
  void InferShape(const Scope& scope) const override {}
  void Run(const Scope& scope,
           const platform::DeviceContext& dev_ctx) const override {}
};

class MyTestOpProtoAndCheckerMaker : public OpProtoAndCheckerMaker {
 public:
  MyTestOpProtoAndCheckerMaker(OpProto* proto, OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("input", "input of cosine op").SetMultiple();
    AddOutput("output", "output of cosine op").SetTemporary();
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

REGISTER_OP(cos_sim, paddle::framework::CosineOp,
            paddle::framework::CosineOpProtoAndCheckerMaker);
REGISTER_OP(my_test_op, paddle::framework::MyTestOp,
            paddle::framework::MyTestOpProtoAndCheckerMaker);

TEST(OpRegistry, CreateOp) {
  paddle::framework::OpDesc op_desc;
  op_desc.set_type("cos_sim");
  op_desc.add_inputs("aa");
  op_desc.add_outputs("bb");

  float scale = 3.3;
  auto attr = op_desc.mutable_attrs()->Add();
  attr->set_name("scale");
  attr->set_type(paddle::framework::AttrType::FLOAT);
  attr->set_f(scale);

  std::shared_ptr<paddle::framework::OperatorBase> op =
      paddle::framework::OpRegistry::CreateOp(op_desc);
  paddle::framework::Scope scope;
  paddle::platform::CPUDeviceContext dev_ctx;
  op->Run(scope, dev_ctx);
  float scale_get = op->GetAttr<float>("scale");
  ASSERT_EQ(scale_get, scale);
}

TEST(OpRegistry, IllegalAttr) {
  paddle::framework::OpDesc op_desc;
  op_desc.set_type("cos_sim");
  op_desc.add_inputs("aa");
  op_desc.add_outputs("bb");

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
  op_desc.add_inputs("aa");
  op_desc.add_outputs("bb");

  ASSERT_TRUE(op_desc.IsInitialized());

  std::shared_ptr<paddle::framework::OperatorBase> op =
      paddle::framework::OpRegistry::CreateOp(op_desc);
  paddle::framework::Scope scope;
  paddle::platform::CPUDeviceContext dev_ctx;
  op->Run(scope, dev_ctx);
  ASSERT_EQ(op->GetAttr<float>("scale"), 1.0);
}

static void SetInputFormat(paddle::framework::OpDesc* desc) {
  auto attr = desc->add_attrs();
  attr->set_name("input_format");
  attr->set_type(paddle::framework::INTS);
  attr->mutable_ints()->Add(0);
  attr->mutable_ints()->Add(1);
}

TEST(OpRegistry, CustomChecker) {
  paddle::framework::OpDesc op_desc;
  op_desc.set_type("my_test_op");
  op_desc.add_inputs("ii");
  op_desc.add_outputs("oo");
  SetInputFormat(&op_desc);

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
  SetInputFormat(&op_desc);
  auto op = paddle::framework::OpRegistry::CreateOp(op_desc);
  paddle::platform::CPUDeviceContext dev_ctx;
  paddle::framework::Scope scope;
  op->Run(scope, dev_ctx);
  int test_attr = op->GetAttr<int>("test_attr");
  ASSERT_EQ(test_attr, 4);
}

class TestAttrProtoMaker : public pd::OpProtoAndCheckerMaker {
 public:
  TestAttrProtoMaker(pd::OpProto* proto, pd::OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddAttr<float>("scale", "scale of test op");
    AddAttr<float>("scale", "scale of test op");
  }
};

TEST(ProtoMaker, DuplicatedAttr) {
  pd::OpProto op_proto;
  pd::OpAttrChecker op_checker;
  auto proto_maker = TestAttrProtoMaker(&op_proto, &op_checker);
  ASSERT_THROW(proto_maker.Validate(), paddle::platform::EnforceNotMet);
}

class TestInOutProtoMaker : public pd::OpProtoAndCheckerMaker {
 public:
  TestInOutProtoMaker(pd::OpProto* proto, pd::OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("input", "input of test op");
    AddInput("input", "input of test op");
  }
};

TEST(ProtoMaker, DuplicatedInOut) {
  pd::OpProto op_proto;
  pd::OpAttrChecker op_checker;
  auto proto_maker = TestInOutProtoMaker(&op_proto, &op_checker);
  ASSERT_THROW(proto_maker.Validate(), paddle::platform::EnforceNotMet);
}
