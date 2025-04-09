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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/raw_tensor.h"
#include "paddle/phi/core/tensor_utils.h"

namespace paddle {
namespace framework {
class OpDesc;
class Scope;
template <typename T>
class EmptyGradOpMaker;
}  // namespace framework
namespace imperative {
class OpBase;
}  // namespace imperative
}  // namespace paddle

namespace paddle {
namespace operators {

const framework::FeedType& CheckAndGetFeedItem(const phi::ExtendedTensor& x,
                                               int col) {
  PADDLE_ENFORCE_GE(col,
                    0,
                    common::errors::InvalidArgument(
                        "Expected the column index (the attribute 'col' of "
                        "operator 'Feed') of current feeding variable to be "
                        "no less than 0. But received column index = %d.",
                        col));
  auto feed_list = static_cast<const paddle::framework::FeedList*>(&x);
  PADDLE_ENFORCE_LT(
      static_cast<size_t>(col),
      feed_list->size(),
      common::errors::InvalidArgument(
          "The column index of current feeding variable is expected to be "
          "less than the length of feeding list. But received column index = "
          "%d, the length of feeding list = %d",
          col,
          feed_list->size()));

  return feed_list->at(static_cast<size_t>(col));
}

class FeedOp : public framework::OperatorWithKernel {
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "feed");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "feed");
    if (ctx->IsRuntime()) {
      framework::Variable* x_var =
          PADDLE_GET(framework::Variable*, ctx->GetInputVarPtrs("X")[0]);
      framework::Variable* out_var =
          PADDLE_GET(framework::Variable*, ctx->GetOutputVarPtrs("Out")[0]);

      auto& x = x_var->Get<framework::FeedList>();
      int col = ctx->Attrs().Get<int>("col");
      const auto& feed_item = CheckAndGetFeedItem(x, col);

      auto& feed_tensor = feed_item;
      phi::DenseTensor* out_tensor = out_var->GetMutable<phi::DenseTensor>();
      phi::DenseTensorMeta meta = out_tensor->meta();
      meta.dims = feed_tensor.dims();
      meta.dtype = feed_tensor.dtype();
      meta.layout = feed_tensor.layout();
      meta.legacy_lod = feed_tensor.lod();
      meta.strides = feed_tensor.strides();
      if (meta.strides.size() == -1) {
        meta.strides = meta.calc_strides(meta.dims);
      }
      out_tensor->set_meta(meta);
    }
  }

 protected:
  phi::KernelKey GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    const framework::Variable* x_var = ctx.InputVar("X");
    auto& x = x_var->Get<framework::FeedList>();
    int col = ctx.Attr<int>("col");
    auto& feed_item = x[col];

    framework::proto::VarType::Type expected_data_type;
    expected_data_type = framework::TransToProtoVarType(feed_item.dtype());

    return phi::KernelKey(expected_data_type, ctx.GetPlace());
  }
};

class FeedOpInfoMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "(vector<phi::DenseTensor>) "
             "A feeding list of phi::DenseTensor, which may have "
             "different dimension and data type.");
    AddOutput("Out",
              "(phi::DenseTensor) The phi::DenseTensor which is a copy "
              "of the col-th feeding "
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

// TODO(YuanRisheng): Maybe we need design a new registry macro for
// registering device independent kernels.

REGISTER_OPERATOR(
    feed,
    paddle::operators::FeedOp,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>,
    paddle::operators::FeedOpInfoMaker);
