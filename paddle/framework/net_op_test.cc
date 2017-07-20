#include <gtest/gtest.h>
#include <paddle/framework/net.h>
#include <paddle/framework/op_registry.h>
#include <paddle/framework/operator.h>

namespace pd = paddle::framework;

static int infer_shape_cnt = 0;
static int run_cnt = 0;

class TestOp : public pd::OperatorBase {
 public:
  void InferShape(const paddle::framework::ScopePtr& scope) const override {
    ++infer_shape_cnt;
  }
  void Run(const paddle::framework::ScopePtr& scope,
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
  auto net = std::make_shared<paddle::framework::PlainNet>();
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

  auto scope = std::make_shared<pd::Scope>();
  paddle::platform::CPUDeviceContext dev_ctx;

  net->InferShape(scope);
  net->Run(scope, dev_ctx);
  ASSERT_EQ(2, infer_shape_cnt);
  ASSERT_EQ(2, run_cnt);

  ASSERT_THROW(net->AddOp(op2), std::runtime_error);
}
