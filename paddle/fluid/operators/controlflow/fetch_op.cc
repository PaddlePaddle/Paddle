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

#include "paddle/fluid/framework/data_layout_transform.h"
#include "paddle/fluid/framework/feed_fetch_type.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/profiler.h"

namespace paddle {
namespace operators {
void TransDataLayout(const framework::LoDTensor &src_var,
                     std::string fetch_var_name,
                     framework::LoDTensor *dst_var) {
  if (src_var.IsInitialized() && src_var.numel() > 0) {
#ifdef PADDLE_WITH_MKLDNN
    if (src_var.layout() == framework::DataLayout::kMKLDNN) {
      framework::Tensor out;
      framework::innerTransDataLayoutFromMKLDNN(
          src_var.layout(),
          fetch_var_name == framework::GradVarName("Filter")
              ? framework::DataLayout::kNCHW
              : paddle::platform::get_cur_paddle_data_layout(),
          src_var, &out, platform::CPUPlace());
      TensorCopySync(out, platform::CPUPlace(), dst_var);
    } else {
      TensorCopySync(src_var, platform::CPUPlace(), dst_var);
    }
#else
    TensorCopySync(src_var, platform::CPUPlace(), dst_var);
#endif
  } else {
    dst_var->clear();
    dst_var->Resize({0});
  }
  dst_var->set_lod(src_var.lod());
}

class FetchOp : public framework::OperatorBase {
 public:
  FetchOp(const std::string &type, const framework::VariableNameMap &inputs,
          const framework::VariableNameMap &outputs,
          const framework::AttributeMap &attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

 private:
  void RunImpl(const framework::Scope &scope,
               const platform::Place &place) const override {
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
    if (out_var->IsType<framework::FeedFetchList>()) {
      auto *fetch_list = out_var->GetMutable<framework::FeedFetchList>();
      if (col >= fetch_list->size()) {
        fetch_list->resize(col + 1);
      }
      auto *dst_item = &(fetch_list->at(col));
      auto &src_item = fetch_var->Get<framework::FeedFetchType>();
      TransDataLayout(src_item, fetch_var_name, dst_item);
    } else {
      auto *fetch_list = out_var->GetMutable<framework::FetchVarList>();
      if (col >= fetch_list->size()) {
        fetch_list->resize(col + 1);
      }
      if (fetch_var->IsType<framework::LoDTensor>()) {
        auto &src_item = fetch_var->Get<framework::LoDTensor>();
        auto *dst_item =
            &(boost::get<framework::LoDTensor>(fetch_list->at(col)));
        TransDataLayout(src_item, fetch_var_name, dst_item);
      } else if (fetch_var->IsType<framework::LoDTensorArray>()) {
        auto &src_item = fetch_var->Get<framework::LoDTensorArray>();
        auto &item = fetch_list->at(col);
        std::vector<framework::LoDTensor> temp;
        temp.resize(src_item.size());
        item = temp;
        framework::LoDTensorArray *dst_item =
            &(boost::get<framework::LoDTensorArray>(item));
        for (size_t i = 0; i < src_item.size(); i++) {
          TransDataLayout(src_item[i], fetch_var_name, &(dst_item->at(i)));
        }
      }
    }
  }
};

class FetchOpInfoMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
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

REGISTER_OPERATOR(
    fetch, paddle::operators::FetchOp,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>,
    paddle::operators::FetchOpInfoMaker);
