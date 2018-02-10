/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/framework/feed_fetch_type.h"
#include "paddle/framework/op_registry.h"
#include "paddle/framework/operator.h"

namespace paddle {
namespace operators {
class FeedOp : public framework::OperatorBase {
 public:
  FeedOp(const std::string &type, const framework::VariableNameMap &inputs,
         const framework::VariableNameMap &outputs,
         const framework::AttributeMap &attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}
  void Run(const framework::Scope &scope,
           const platform::Place &place) const override {
    auto feed_var_name = Input("X");
    auto *feed_var = scope.FindVar(feed_var_name);

    PADDLE_ENFORCE(feed_var != nullptr,
                   "Cannot find feed_var in scope, feed_var_name is %s",
                   feed_var_name);

    auto out_name = this->Output("Out");
    auto *out_var = scope.FindVar(out_name);
    PADDLE_ENFORCE(out_var != nullptr,
                   "Cannot find out_var in scope, out_var_name is %s",
                   out_name);

    auto col = Attr<int>("col");

    VLOG(3) << "Feed Var " << feed_var_name << "'s " << col << " column to var "
            << out_name;

    auto &feed_list = feed_var->Get<framework::FeedFetchList>();
    auto &feed_item = feed_list.at(static_cast<size_t>(col));
    auto *out_item = out_var->GetMutable<framework::FeedFetchType>();

    // get device context from pool
    platform::DeviceContextPool &pool = platform::DeviceContextPool::Instance();
    auto &dev_ctx = *pool.Get(place);

    if (platform::is_same_place(feed_item.place(), place)) {
      out_item->ShareDataWith(feed_item);
    } else {
      framework::Copy(feed_item, place, dev_ctx, out_item);
    }
    out_item->set_lod(feed_item.lod());
  }
};

class FeedOpInfoMaker : public framework::OpProtoAndCheckerMaker {
 public:
  FeedOpInfoMaker(OpProto *proto, OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "The input of feed op");
    AddOutput("Out", "The output of feed op");
    AddAttr<int>("col", "(int) The column of feed");
    AddComment(R"DOC(
Feed Operator.

It should not be configured by users directly.

)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OPERATOR(feed, paddle::operators::FeedOp,
                  paddle::framework::EmptyGradOpMaker,
                  paddle::operators::FeedOpInfoMaker);
