/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/feed_fetch_type.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"

namespace paddle {
namespace operators {
class FeedOp : public framework::OperatorBase {
 public:
  FeedOp(const std::string &type, const framework::VariableNameMap &inputs,
         const framework::VariableNameMap &outputs,
         const framework::AttributeMap &attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

 private:
  void RunImpl(const framework::Scope &scope,
               const platform::Place &place) const override {
    OP_INOUT_CHECK(HasInputs("X"), "Input", "X", "Feed");
    OP_INOUT_CHECK(HasOutputs("Out"), "Output", "Out", "Feed");

    auto feed_var_name = Input("X");
    auto *feed_var = scope.FindVar(feed_var_name);
    PADDLE_ENFORCE_NOT_NULL(
        feed_var,
        platform::errors::NotFound(
            "Input varibale(%s) cannot be found in scope for operator 'Feed'.",
            feed_var_name));

    auto out_name = this->Output("Out");
    auto *out_var = scope.FindVar(out_name);
    PADDLE_ENFORCE_NOT_NULL(
        out_var,
        platform::errors::NotFound(
            "Output variable(%s) cannot be found in scope for operator 'Feed'",
            out_name));

    auto col = Attr<int>("col");
    PADDLE_ENFORCE_GE(col, 0,
                      platform::errors::InvalidArgument(
                          "Expected the column index (the attribute 'col' of "
                          "operator 'Feed') of current feeding variable to be "
                          "no less than 0. But received column index = %d.",
                          col));

    VLOG(3) << "Feed variable " << feed_var_name << "'s " << col
            << " column to variable " << out_name;

    auto &feed_list = feed_var->Get<framework::FeedList>();
    PADDLE_ENFORCE_LT(
        static_cast<size_t>(col), feed_list.size(),
        platform::errors::InvalidArgument(
            "The column index of current feeding variable is expected to be "
            "less than the length of feeding list. But received column index = "
            "%d, the length of feeding list = %d",
            col, feed_list.size()));

    auto &feed_item = feed_list.at(static_cast<size_t>(col));
    auto *out_item = out_var->GetMutable<framework::FeedType>();

    if (platform::is_same_place(feed_item.place(), place)) {
      out_item->ShareDataWith(feed_item);
    } else {
      auto *dev_ctx = platform::DeviceContextPool::Instance().Get(place);
      framework::TensorCopy(feed_item, place, *dev_ctx, out_item);
    }
    out_item->set_lod(feed_item.lod());
  }
};

class FeedOpInfoMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "(vector<LoDTensor>) A feeding list of LoDTensor, which may have "
             "different dimension and data type.");
    AddOutput("Out",
              "(LoDTensor) The LoDTensor which is a copy of the col-th feeding "
              "object.");
    AddAttr<int>("col", "(int) The column index of current feeding object.");
    AddComment(R"DOC(
Feed Operator.

It should not be configured by users directly.

)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OPERATOR(
    feed, paddle::operators::FeedOp,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>,
    paddle::operators::FeedOpInfoMaker);
