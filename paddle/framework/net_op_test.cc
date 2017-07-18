#include <gtest/gtest.h>
#include <paddle/framework/net.h>
#include <paddle/framework/op_registry.h>
#include <paddle/framework/operator.h>

namespace paddle {
namespace framework {

static int infer_shape_cnt = 0;
static int run_cnt = 0;

class TestOp : public OperatorBase {
 public:
  void InferShape(const ScopePtr& scope) const override { ++infer_shape_cnt; }
  void Run(const ScopePtr& scope,
           const platform::DeviceContext& dev_ctx) const override {
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

class PlainNetTest : public testing::Test {
  virtual void SetUp() {
    net_ = std::make_shared<PlainNet>();
    ASSERT_NE(net_, nullptr);

    auto op1 = std::make_shared<TestOp>();
    op1->inputs_ = {"x", "w1", "b1"};
    op1->outputs_ = {"y"};
    net_->AddOp(op1);

    auto op2 = std::make_shared<TestOp>();
    op2->inputs_ = {"y", "w2", "b2"};
    op2->outputs_ = {"z"};
    net_->AddOp(op2);
    net_->CompleteAddOp();
  }

  virtual void TearDown() {}

  void TestOpKernel() {
    AssertSameVectorWithoutOrder({"x", "w1", "b1", "w2", "b2"}, net_->inputs_);
    AssertSameVectorWithoutOrder({"y", "z"}, net_->outputs_);
    auto tmp_idx_iter = net_->attrs_.find("temporary_index");
    ASSERT_NE(net_->attrs_.end(), tmp_idx_iter);
    auto& tmp_idx = boost::get<std::vector<int>>(tmp_idx_iter->second);
    ASSERT_EQ(1UL, tmp_idx.size());
    ASSERT_EQ("y", net_->outputs_[tmp_idx[0]]);

    auto scope = std::make_shared<Scope>();
    platform::CPUDeviceContext dev_ctx;

    net_->InferShape(scope);
    net_->Run(scope, dev_ctx);
    ASSERT_EQ(2, infer_shape_cnt);
    ASSERT_EQ(2, run_cnt);

    ASSERT_THROW(net_->AddOp(op2), EnforceNotMet);
  }

  void TestAddBackwardOp() {
    auto grad_ops = AddBackwardOp(net_);
    for (auto& op : grad_ops->ops_) {
      op->DebugString();
    }
  }

 private:
  std::shared_ptr<PlainNet> net_;
};

TEST(OpKernel, all) {
  PlainNetTest net;
  net->TestOpKernel();
}

TEST(AddBackwardOp, TestAddBackwardOp) {
  PlainNetTest net;
  net->TestAddBackwardOp();
}

}  // namespace framework
}  // namespace paddle
