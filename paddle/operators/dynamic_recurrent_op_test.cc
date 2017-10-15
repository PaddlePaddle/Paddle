#include "paddle/operators/dynamic_recurrent_op.h"

#include <gtest/gtest.h>

#include "paddle/framework/ddim.h"
#include "paddle/framework/lod_tensor.h"
#include "paddle/framework/op_desc.h"
#include "paddle/framework/op_registry.h"
#include "paddle/operators/net_op.h"

namespace paddle {
namespace operators {

using framework::Scope;
using framework::TensorArray;
using framework::LoDTensor;
using framework::Variable;

class TestOp : public framework::OperatorBase {
 public:
  using framework::OperatorBase::OperatorBase;
  DEFINE_OP_CLONE_METHOD(TestOp);
  void Run(const Scope& scope,
           const platform::DeviceContext& dev_ctx) const override {}
};

void OpDescNewVar(const std::string& param_name,
                  std::initializer_list<const char*> arguments,
                  paddle::framework::OpDesc::Var* var) {
  var->set_parameter(param_name);
  for (auto& arg_name : arguments) {
    var->add_arguments(arg_name);
  }
}

// create a LoD tensor in scope with specific dims
LoDTensor* CreateVar(Scope& scope, std::string name, framework::DDim dims,
                     const platform::Place& place) {
  auto* var = scope.Var(name);
  auto* tensor = var->GetMutable<LoDTensor>();
  tensor->Resize(dims);
  tensor->mutable_data<float>(place);
  return tensor;
}

class RNNAlgorithmTestHelper : public ::testing::Test {
 protected:
  const rnn::ArgumentName argname = RNNAlgorithm::kArgNames[0];

  virtual void SetUp() override {
    CreateGlobalVariables();

    auto op_desc = CreateOpDesc();
    op = paddle::framework::OpRegistry::CreateOp(op_desc);
    dop = &(dynamic_cast<DynamicRecurrentOp*>(op.get())->rnn);
    InitCacheManually();
    InitStepNet();
  }

  framework::OpDesc CreateOpDesc() {
    // create op
    paddle::framework::OpDesc op_desc;
    op_desc.set_type("dynamic_recurrent");

    OpDescNewVar(argname.inlinks, {"in0"}, op_desc.add_inputs());
    OpDescNewVar(argname.initial_states, {"boot_mem"}, op_desc.add_inputs());
    OpDescNewVar(argname.step_scopes, {"step_scopes"}, op_desc.add_outputs());
    OpDescNewVar(argname.outlinks, {"out0"}, op_desc.add_outputs());

    // set pre-states
    auto pre_memories = op_desc.mutable_attrs()->Add();
    pre_memories->set_name(argname.ex_states);
    pre_memories->set_type(paddle::framework::AttrType::STRINGS);
    auto pre_memories_item = pre_memories->add_strings();
    *pre_memories_item = "mem@pre";

    // set states
    auto memories = op_desc.mutable_attrs()->Add();
    memories->set_name(argname.states);
    memories->set_type(paddle::framework::AttrType::STRINGS);
    auto memories_item = memories->add_strings();
    *memories_item = "mem";
    return op_desc;
  }

  void CreateGlobalVariables() {
    platform::CPUPlace place;
    scope.Var("step_scopes");
    CreateVar(scope, "boot_mem", framework::make_ddim({10, 20}), place);
    CreateVar(scope, "out0", framework::make_ddim({10, 20}), place);
    auto* in0 = CreateVar(scope, "in0", framework::make_ddim({10, 8}), place);
    // 10 instanes with 4 sentences, length is 4, 3, 2, 1 respectively.
    framework::LoD in0_lod(1);
    for (int x : std::vector<int>{0, 4, 7, 9, 10}) {
      in0_lod[0].push_back(x);
    }
    in0->set_lod(in0_lod);
    in0->Resize(framework::make_ddim({10, 8}));
    // set the content, each sentence content is seqid.batchid
    // the seqid starts from 0
    int start = 0;
    for (size_t seqid = 0; seqid < in0_lod.size() - 1; seqid++) {
      for (size_t batchid = 0;
           batchid < in0_lod[0][seqid + 1] - in0_lod[0][seqid]; batchid++) {
        float v = seqid + batchid * 0.1;

        for (size_t dim = 0; dim < 8; dim++) {
          in0->data<float>()[start * 8 + dim] = v;
        }
        start++;
      }
    }
  }

  void InitCacheManually() {
    dop->cache_.Init(RNNAlgorithm::kArgNames[0], *op, scope, &device_context,
                     &dop->arg_);
  }

  void InitStepNet() {
    std::unique_ptr<framework::OperatorBase> stepnet{new NetOp};
    dynamic_cast<NetOp*>(stepnet.get())
        ->AppendOp(std::unique_ptr<TestOp>(new TestOp(
            "test", {{"inputs", {"in0"}}, {"initial_states", {"boot_mem"}}},
            {{"outputs", {"out0"}}, {"step_scopes", {"step_scopes"}}}, {})));
    dop->SetStepUnit(std::move(stepnet));
  }

 protected:
  RNNAlgorithm* dop;
  std::unique_ptr<framework::OperatorBase> op;
  paddle::platform::CPUDeviceContext device_context;
  paddle::framework::Scope scope;
};

TEST_F(RNNAlgorithmTestHelper, CreateCache) {
  const rnn::Argument& arg = dop->arg_;
  ASSERT_EQ(arg.inlinks.size(), 1UL);
  ASSERT_EQ(arg.outlinks.size(), 1UL);
}

TEST_F(RNNAlgorithmTestHelper, SplitInputs) {
  dop->SplitInputs();
  auto& in0_ta = dop->step_inputs_["in0"];
  ASSERT_EQ(in0_ta.size(), 4UL);

  const auto& batch0 = in0_ta.Read(0);
  const auto& batch1 = in0_ta.Read(1);
  const auto& batch2 = in0_ta.Read(2);
  const auto& batch3 = in0_ta.Read(3);
  EXPECT_EQ(batch0.dims()[0], 4);
  EXPECT_EQ(batch1.dims()[0], 3);
  EXPECT_EQ(batch2.dims()[0], 2);
  EXPECT_EQ(batch3.dims()[0], 1);
}

TEST_F(RNNAlgorithmTestHelper, CreateScopes) {
  dop->SplitInputs();
  dop->CreateScopes();
  ASSERT_EQ(dop->cache_.num_steps, 4UL);
  ASSERT_EQ(dop->cache_.scopes->size(), 4UL);
}

TEST_F(RNNAlgorithmTestHelper, WriteStepInputs) {
  dop->SplitInputs();
  dop->CreateScopes();
  dop->WriteStepInputs();

  for (size_t step = 0; step < dop->cache_.num_steps; step++) {
    auto& scope = dop->cache_.GetScope(step);
    for (auto name : std::vector<std::string>({"in0"})) {
      ASSERT_TRUE(scope.FindVar(name) != nullptr);
    }
  }
}

TEST_F(RNNAlgorithmTestHelper, WriteStepOutputs) {
  dop->SplitInputs();
  dop->CreateScopes();
  dop->WriteStepInputs();
  dop->WriteStepOutputs();

  for (size_t step = 0; step < dop->cache_.num_steps; step++) {
    auto& scope = dop->cache_.GetScope(step);
    for (auto name : std::vector<std::string>({"out0"})) {
      ASSERT_TRUE(scope.FindVar(name));
    }
  }
}

TEST_F(RNNAlgorithmTestHelper, ConcatOutputs) {
  // Let's leave this test to python unittest.
}

TEST_F(RNNAlgorithmTestHelper, InitStates) {
  dop->SetComputeMode(RNNAlgorithm::ComputeMode::kForward);
  dop->SplitInputs();
  dop->CreateScopes();
  dop->WriteStepInputs();
  dop->WriteStepOutputs();
  dop->InitStates();

  for (size_t step = 0; step < dop->cache_.num_steps; step++) {
    auto& scope = dop->cache_.GetScope(step);
    auto state = scope.FindVar("mem");
    ASSERT_TRUE(state != nullptr);

    auto* pre_state = scope.FindVar("mem@pre");
    ASSERT_TRUE(pre_state != nullptr);

    auto* boot_state = scope.FindVar("boot_mem");
    ASSERT_TRUE(boot_state != nullptr);
  }
}

}  // operators
}  // namespace paddle
