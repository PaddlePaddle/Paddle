#include "paddle/operators/net_op.h"

#include <gtest/gtest.h>

namespace paddle {
namespace operators {
using Scope = framework::Scope;
using DeviceContext = platform::DeviceContext;

static int infer_shape_cnt = 0;
static int run_cnt = 0;

class TestOp : public framework::OperatorBase {
 public:
  void InferShape(const Scope& scope) const override { ++infer_shape_cnt; }
  void Run(const Scope& scope,
           const platform::DeviceContext& dev_ctx) const override {
    ++run_cnt;
  }
};

class EmptyOp : public framework::OperatorBase {
 public:
  void InferShape(const Scope& scope) const override {}
  void Run(const Scope& scope, const DeviceContext& dev_ctx) const override {}
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
  auto net = std::make_shared<NetOp>();
  ASSERT_NE(net, nullptr);

  auto op1 = std::make_shared<TestOp>();
  op1->inputs_ = {{"X", {"x"}}, {"W", {"w1"}}, {"b", {"b1"}}};
  op1->outputs_ = {{"Out", {"y"}}};
  net->AddOp(op1);

  auto op2 = std::make_shared<TestOp>();
  op2->inputs_ = {{"X", {"y"}}, {"W", {"w2"}}, {"b", {"b2"}}};
  op2->outputs_ = {{"Out", {"z"}}};
  net->AddOp(op2);

  net->CompleteAddOp();
  AssertSameVectorWithoutOrder({"x", "w1", "b1", "w2", "b2"},
                               net->inputs_.at(NetOp::kAll));
  AssertSameVectorWithoutOrder({"y", "z"}, net->outputs_.at(NetOp::kAll));

  auto final_outs = net->OutputVars(false);

  ASSERT_EQ(final_outs.size(), 1UL);
  ASSERT_EQ(final_outs[0], "z");
}

TEST(NetOp, insert_op) {
  NetOp net;
  auto op1 = std::make_shared<EmptyOp>();
  op1->inputs_ = {{"X", {"x"}}, {"W", {"w1"}}, {"b", {"b1"}}};
  op1->outputs_ = {{"Out", {"y"}}};
  net.AddOp(op1);
  net.InsertOp(0, op1);
  ASSERT_EQ(2UL, net.ops_.size());
  net.InsertOp(2, op1);
  ASSERT_EQ(3UL, net.ops_.size());
}

}  // namespace operators
}  // namespace paddle
