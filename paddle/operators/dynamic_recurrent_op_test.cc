#include "paddle/operators/dynamic_recurrent_op.h"

#include <gtest/gtest.h>

#include "paddle/framework/ddim.h"
#include "paddle/framework/lod_tensor.h"
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
    CreateGlobalVariables();

    auto op_desc = CreateOpDesc();
    op = paddle::framework::OpRegistry::CreateOp(op_desc);
    dop = dynamic_cast<DynamicRecurrentOp*>(op.get());
    InitCacheManually();
  }

  framework::OpDesc CreateOpDesc() {
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
    return op_desc;
  }

  void CreateGlobalVariables() {
    auto* in0 = scope.NewVar("in0");
    auto* boot_mem = scope.NewVar("boot_mem");
    // auto* step_scopes =
    scope.NewVar("step_scopes");
    // auto* out0 =
    scope.NewVar("out0");

    platform::CPUPlace place;

    auto* in0_lod_tensor = in0->GetMutable<LoDTensor>();
    // 10 instanes with 4 sentences, length is 4, 3, 2, 1 respectively.
    framework::LoD in0_lod({{0, 4, 7, 9, 10}});
    in0_lod_tensor->set_lod(in0_lod);
    in0_lod_tensor->Resize(framework::make_ddim({10, 8}));
    // set the content, each sentence content is seqid.batchid
    // the seqid starts from 0
    int start = 0;
    for (size_t seqid = 0; seqid < in0_lod.size() - 1; seqid++) {
      for (size_t batchid = 0;
           batchid < in0_lod[0][seqid + 1] - in0_lod[0][seqid]; batchid++) {
        float v = seqid + batchid * 0.1;

        for (size_t dim = 0; dim < 8; dim++) {
          in0_lod_tensor->data<float>()[start * 8 + dim] = v;
        }
        start++;
      }
    }

    in0_lod_tensor->mutable_data<float>(place);

    auto common_ddim = framework::make_ddim({6, 8});
    auto* boot_mem_lod_tensor = boot_mem->GetMutable<LoDTensor>();
    boot_mem_lod_tensor->Resize(common_ddim);
    boot_mem_lod_tensor->mutable_data<float>(place);
  }

  void InitCacheManually() {
    dop->cache_.Init(DynamicRecurrentOp::kArgName, *dop, scope, &dop->arg_);
  }

 protected:
  DynamicRecurrentOp* dop;
  std::unique_ptr<framework::OperatorBase> op;
  paddle::platform::CPUDeviceContext device_context;
  paddle::framework::Scope scope;
};

TEST_F(DynamicRecurrentOpTestHelper, CreateCache) {
  const rnn::Argument& arg = dop->arg_;
  ASSERT_EQ(arg.inlinks.size(), 1UL);
  ASSERT_EQ(arg.outlinks.size(), 1UL);
}

TEST_F(DynamicRecurrentOpTestHelper, SplitInputs) {
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

TEST_F(DynamicRecurrentOpTestHelper, CreateScopes) {
  dop->SplitInputs();
  dop->CreateScopes();
  ASSERT_EQ(dop->cache_.num_steps, 4UL);
  ASSERT_EQ(dop->cache_.scopes->size(), 4UL);
}

TEST_F(DynamicRecurrentOpTestHelper, WriteStepInputs) {
  dop->SplitInputs();
  dop->CreateScopes();
  dop->WriteStepInputs();

  for (size_t step = 0; step < dop->cache_.num_steps; step++) {
    auto& scope = dop->cache_.GetScope(step);
    for (auto name : std::vector<std::string>({"in0", "mem", "mem@pre"})) {
      ASSERT_NE(scope.FindVar(name), nullptr);
    }
  }
}

TEST_F(DynamicRecurrentOpTestHelper, WriteStepOutputs) {
  dop->SplitInputs();
  dop->CreateScopes();
  dop->WriteStepInputs();
  dop->WriteStepOutputs();

  for (size_t step = 0; step < dop->cache_.num_steps; step++) {
    auto& scope = dop->cache_.GetScope(step);
    for (auto name : std::vector<std::string>({"out0"})) {
      ASSERT_NE(scope.FindVar(name), nullptr);
    }
  }
}

TEST_F(DynamicRecurrentOpTestHelper, ConcatOutputs) {
  // Let's leave this test to python unittest.
}

TEST_F(DynamicRecurrentOpTestHelper, InitStates) {
  dop->SplitInputs();
  dop->CreateScopes();
  dop->WriteStepInputs();
  dop->WriteStepOutputs();
  dop->InitStates();

  for (size_t step = 0; step < dop->cache_.num_steps; step++) {
    auto& scope = dop->cache_.GetScope(step);
    auto state = scope.FindVar("mem")->Get<LoDTensor>();
    auto pre_state = scope.FindVar("mem@pre")->Get<LoDTensor>();
    auto boot_state = scope.FindVar("boot_mem")->Get<LoDTensor>();

    if (step == 0) {
      // check pre_state is a reference of boot_state
      ASSERT_EQ(boot_state.data<float>(), pre_state.data<float>());
    }
  }
}

}  // operators
}  // namespace paddle
