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

namespace paddle {
namespace operators {

class FetchOp : public framework::OperatorBase {
 public:
  FetchOp(const std::string &type, const framework::VariableNameMap &inputs,
          const framework::VariableNameMap &outputs,
          const framework::AttributeMap &attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

  void Run(const framework::Scope &scope,
           const platform::DeviceContext &dev_ctx) const override {
    auto fetch_var_name = Input("X");
    auto *fetch_var = scope.FindVar(fetch_var_name);
    PADDLE_ENFORCE(fetch_var != nullptr,
                   "Cannot find fetch variable in scope, fetch_var_name is %s",
                   fetch_var_name);

    auto out_name = this->Output("Out");
    auto *out_var = scope.FindVar(out_name);
    PADDLE_ENFORCE(out_var != nullptr,
                   "Cannot find out_var in scope, out_var_name is %s",
                   out_name);

    auto col = static_cast<size_t>(Attr<int>("col"));

    auto *fetch_list = out_var->GetMutable<framework::FeedFetchList>();
    auto &src_item = fetch_var->Get<framework::FeedFetchType>();

    if (col >= fetch_list->size()) {
      fetch_list->resize(col + 1);
    }
    auto &dst_item = fetch_list->at(col);

    // FIXME(yuyang18): Should we assume the fetch operator always generate
    // CPU outputs?
    CopyFrom(src_item, platform::CPUPlace(), dev_ctx, &dst_item);
    dev_ctx.Wait();
    dst_item.set_lod(src_item.lod());

    VLOG(3) << "Fetch variable " << fetch_var_name << " to " << out_name;
  }
};

class FetchOpInfoMaker : public framework::OpProtoAndCheckerMaker {
 public:
  FetchOpInfoMaker(framework::OpProto *proto,
                   framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "The input of fetch op");
    AddOutput("Out", "The output of fetch op");
    AddAttr<int>("col", "(int) The column of fetch");
    AddComment(R"DOC(
Fetch Operator.

It should not be configured by users directly.

)DOC");
  }
};
}  // namespace operators
}  // namespace paddle

REGISTER_OPERATOR(fetch, paddle::operators::FetchOp,
                  paddle::framework::EmptyGradOpMaker,
                  paddle::operators::FetchOpInfoMaker);
