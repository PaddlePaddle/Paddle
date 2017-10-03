#include "paddle/operators/dynamic_recurrent_op.h"

#include <gtest/gtest.h>
#include "paddle/framework/op_desc.h"
#include "paddle/framework/op_registry.h"

namespace paddle {
namespace operators {

static void BuildVar(const std::string& param_name,
                     std::initializer_list<const char*> arguments,
                     paddle::framework::OpDesc::Var* var) {
  var->set_parameter(param_name);
  for (auto& arg_name : arguments) {
    var->add_arguments(arg_name);
  }
}

class DynamicRecurrentOpTestHelper : public ::testing::Test {
 protected:
  const rnn::ArgumentName argname = DynamicRecurrentOp::kArgName;

  virtual void SetUp() override {
    // create op
    paddle::framework::OpDesc op_desc;
    op_desc.set_type("dynamic_recurrent_op");

    BuildVar(argname.inlinks, {"in0"}, op_desc.add_inputs());
    BuildVar(argname.boot_memories, {"boot_mem"}, op_desc.add_inputs());
    BuildVar(argname.step_scopes, {"step_scopes"}, op_desc.add_outputs());
    BuildVar(argname.outlinks, {"out0"}, op_desc.add_outputs());

    // set pre-memories
    auto pre_memories = op_desc.mutable_attrs()->Add();
    pre_memories->set_name(argname.pre_memories);
    pre_memories->set_type(paddle::framework::AttrType::STRINGS);
    auto pre_memories_item = pre_memories->add_strings();
    *pre_memories_item = "mem@pre";

    // set memories
    auto memories = op_desc.mutable_attrs()->Add();
    memories->set_name(argname.memories);
    memories->set_type(paddle::framework::AttrType::STRINGS);
    auto memories_item = memories->add_strings();
    *memories_item = "mem";

    paddle::platform::CPUDeviceContext device_context;
    paddle::framework::Scope scope;

    op = paddle::framework::OpRegistry::CreateOp(op_desc);
  }

 protected:
  std::unique_ptr<framework::OperatorBase> op;
};

TEST_F(DynamicRecurrentOpTestHelper, init) {}

}  // namespace test
}  // namespace paddle
