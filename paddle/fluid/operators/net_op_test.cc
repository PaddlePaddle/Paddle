//  Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include "paddle/fluid/operators/net_op.h"

#include <gtest/gtest.h>

namespace paddle {
namespace operators {
using Scope = framework::Scope;
using DeviceContext = platform::DeviceContext;

static int run_cnt = 0;

class TestOp : public framework::OperatorBase {
 public:
  using framework::OperatorBase::OperatorBase;
  DEFINE_OP_CLONE_METHOD(TestOp);

 private:
  void RunImpl(const Scope& scope,
               const platform::Place& place) const override {
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
  auto net = std::make_shared<NetOp>();
  ASSERT_NE(net, nullptr);

  net->AppendOp(std::unique_ptr<TestOp>(
      new TestOp("test", {{"X", {"x"}}, {"W", {"w1"}}, {"b", {"b1"}}},
                 {{"Out", {"y"}}}, framework::AttributeMap{})));
  net->AppendOp(std::unique_ptr<TestOp>(
      new TestOp("test", {{"X", {"y"}}, {"W", {"w2"}}, {"b", {"b2"}}},
                 {{"Out", {"z"}}}, framework::AttributeMap{})));

  net->CompleteAddOp();
  AssertSameVectorWithoutOrder({"x", "w1", "b1", "w2", "b2"},
                               net->Inputs(NetOp::kAll));
  AssertSameVectorWithoutOrder({"y", "z"}, net->Outputs(NetOp::kAll));

  auto final_outs = net->OutputVars(false);

  ASSERT_EQ(final_outs.size(), 1UL);
  ASSERT_EQ(final_outs[0], "z");
}

TEST(NetOp, insert_op) {
  NetOp net;
  auto op1 = std::unique_ptr<framework::NOP>(
      new framework::NOP("empty", {{"X", {"x"}}, {"W", {"w1"}}, {"b", {"b1"}}},
                         {{"Out", {"y"}}}, framework::AttributeMap{}));
  net.AppendOp(*op1);
  net.InsertOp(0, *op1);
  ASSERT_EQ(2UL, net.ops_.size());
  net.InsertOp(2, std::move(op1));
  ASSERT_EQ(3UL, net.ops_.size());
}

TEST(NetOp, Clone) {
  NetOp net;
  net.AppendOp(std::unique_ptr<framework::NOP>(new framework::NOP{
      "empty", framework::VariableNameMap{}, framework::VariableNameMap{},
      framework::AttributeMap{}}));
  net.AppendOp(std::unique_ptr<framework::NOP>(new framework::NOP{
      "empty2", framework::VariableNameMap{}, framework::VariableNameMap{},
      framework::AttributeMap{}}));
  net.CompleteAddOp(true);
  auto new_net_op = net.Clone();
  ASSERT_NE(new_net_op, nullptr);
  ASSERT_TRUE(new_net_op->IsNetOp());
  auto* new_net = static_cast<NetOp*>(new_net_op.get());
  ASSERT_EQ(2UL, new_net->ops_.size());
  ASSERT_EQ(new_net->ops_[0]->Type(), "empty");
  ASSERT_EQ(new_net->ops_[1]->Type(), "empty2");
}

}  // namespace operators
}  // namespace paddle
