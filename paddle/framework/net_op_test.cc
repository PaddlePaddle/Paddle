#include <gtest/gtest.h>
#include <paddle/framework/net.h>
#include <paddle/framework/op_registry.h>
#include <paddle/framework/operator.h>

USE_OP(add_two);
USE_OP(mul);
USE_OP(sigmoid);
USE_OP(softmax);

namespace paddle {
namespace framework {

static int infer_shape_cnt = 0;
static int run_cnt = 0;

class TestOp : public OperatorBase {
 public:
  void InferShapeImpl(
      const paddle::framework::InferShapeContext& ctx) const override {
    ++infer_shape_cnt;
  }
  void Run(const std::shared_ptr<framework::Scope>& scope,
           const paddle::platform::DeviceContext& dev_ctx) const override {
    ++run_cnt;
  }
};

template <typename T>
void AssertSameVectorWithoutOrder(const std::vector<T>& expected,
                                  const std::vector<T>& actual) {
  ASSERT_EQ(expected.size(), actual.size());
  std::unordered_set<T> expected_set;
  for (auto& tmp : expected) {
    expected_set.insert(tmp);
  }
  for (auto& act : actual) {
    ASSERT_NE(expected_set.end(), expected_set.find(act));
  }
}

TEST(OpKernel, all) {
  auto net = std::make_shared<PlainNet>();
  ASSERT_NE(net, nullptr);

  auto op1 = std::make_shared<TestOp>();
  op1->inputs_ = {"x", "w1", "b1"};
  op1->outputs_ = {"y"};
  net->AddOp(op1);

  auto op2 = std::make_shared<TestOp>();
  op2->inputs_ = {"y", "w2", "b2"};
  op2->outputs_ = {"z"};
  net->AddOp(op2);

  net->CompleteAddOp();
  AssertSameVectorWithoutOrder({"x", "w1", "b1", "w2", "b2"}, net->inputs_);
  AssertSameVectorWithoutOrder({"y", "z"}, net->outputs_);
  auto tmp_idx_iter = net->attrs_.find("temporary_index");
  ASSERT_NE(net->attrs_.end(), tmp_idx_iter);
  auto& tmp_idx = boost::get<std::vector<int>>(tmp_idx_iter->second);
  ASSERT_EQ(1UL, tmp_idx.size());
  ASSERT_EQ("y", net->outputs_[tmp_idx[0]]);

  auto scope = std::make_shared<Scope>();
  platform::CPUDeviceContext dev_ctx;

  net->InferShape(scope);
  net->Run(scope, dev_ctx);
  ASSERT_EQ(2, infer_shape_cnt);
  ASSERT_EQ(2, run_cnt);
  ASSERT_THROW(net->AddOp(op2), std::runtime_error);
}
TEST(AddBackwardOp, TestGradOp) {
  auto net = std::make_shared<PlainNet>();
  ASSERT_NE(net, nullptr);
  net->AddOp(framework::OpRegistry::CreateOp("mul", {"X", "Y"}, {"Out"}, {}));
  net->AddOp(
      framework::OpRegistry::CreateOp("add_two", {"X", "Y"}, {"Out"}, {}));
  net->AddOp(framework::OpRegistry::CreateOp("add_two", {"X", "Y"}, {""}, {}));
  auto grad_ops = AddBackwardOp(net);
  for (auto& op : grad_ops->ops_) {
    op->DebugString();
  }
}

// TODO(zhihong): add fc grad without registering.
// TEST(AddBackwardOp, TestNoGradOp) {
//   auto net = std::make_shared<PlainNet>();
//   ASSERT_NE(net, nullptr);
//   net->AddOp(framework::OpRegistry::CreateOp("fc", {"X", "W", "b"}, {"Y"},
//   {})); auto grad_ops = AddBackwardOp(net); for (auto& op : grad_ops->ops_) {
//     op->DebugString();
//   }
// }

}  // namespace framework
}  // namespace paddle
