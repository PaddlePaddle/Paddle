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
#include "paddle/fluid/framework/raw.h"
#include "paddle/phi/core/kernel_registry.h"
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

// FeedVariableVisitor is to feed the variable data
// according to data type (phi::DenseTensor or  Strings).
template <typename T>
class FeedVariableVisitor {
 public:
  explicit FeedVariableVisitor(T* out_var, const platform::Place& place)
      : place_(place) {
    out_var_.SetData(out_var);
  }

  void operator()(const phi::DenseTensor& in_tensor) const {
    phi::DenseTensor* out_tensor = &(out_var_.Get<phi::DenseTensor>());
    if (platform::is_same_place(in_tensor.place(), place_)) {
      out_tensor->ShareDataWith(in_tensor);
#ifdef PADDLE_WITH_IPU
    } else if (platform::is_ipu_place(place_)) {
      // For ipu, both in_tensor and out_tensor are allocated on cpu,
      // PopART will copy tensor from host automatically,
      // no TensorCopy() is required here.
      out_tensor->ShareDataWith(in_tensor);
#endif
    } else {
      platform::DeviceContext* context =
          platform::DeviceContextPool::Instance().Get(place_);
      framework::TensorCopy(in_tensor, place_, *context, out_tensor);
    }
    out_tensor->set_lod(in_tensor.lod());
  }

  void operator()(const framework::Strings& in_str) const {
    framework::Strings* out_str = &(out_var_.Get<framework::Strings>());
    out_str->resize(in_str.size());
    *out_str = in_str;
  }

  void operator()(const phi::SparseCooTensor& in_tensor) const {
    phi::SparseCooTensor* out_tensor = &(out_var_.Get<phi::SparseCooTensor>());
    if (platform::is_same_place(in_tensor.place(), place_)) {
      *out_tensor = in_tensor;
    } else {
      platform::DeviceContext* context =
          platform::DeviceContextPool::Instance().Get(place_);

      phi::DenseTensor indices, values;
      framework::TensorCopy(in_tensor.indices(), place_, *context, &indices);
      framework::TensorCopy(in_tensor.values(), place_, *context, &values);
      out_tensor->SetMember(indices, values, in_tensor.meta());
    }
  }

 private:
  framework::Raw out_var_;
  const platform::Place& place_;
};

const framework::FeedType& CheckAndGetFeedItem(const phi::ExtendedTensor& x,
                                               int col) {
  PADDLE_ENFORCE_GE(col,
                    0,
                    platform::errors::InvalidArgument(
                        "Expected the column index (the attribute 'col' of "
                        "operator 'Feed') of current feeding variable to be "
                        "no less than 0. But received column index = %d.",
                        col));
  auto feed_list = static_cast<const paddle::framework::FeedList*>(&x);
  PADDLE_ENFORCE_LT(
      static_cast<size_t>(col),
      feed_list->size(),
      platform::errors::InvalidArgument(
          "The column index of current feeding variable is expected to be "
          "less than the length of feeding list. But received column index = "
          "%d, the length of feeding list = %d",
          col,
          feed_list->size()));

  return feed_list->at(static_cast<size_t>(col));
}

template <typename T, typename Context>
void FeedDenseTensorKernel(const Context& dev_ctx,
                           const phi::ExtendedTensor& x,
                           int col,
                           phi::DenseTensor* out) {
  PADDLE_ENFORCE_NOT_NULL(
      out,
      platform::errors::NotFound(
          "Output cannot be found in scope for operator 'Feed'"));
  const auto& feed_item = CheckAndGetFeedItem(x, col);
  FeedVariableVisitor<phi::DenseTensor> visitor(out, dev_ctx.GetPlace());
  paddle::visit(visitor, feed_item);
}

template <typename T, typename Context>
void FeedSparseCooTensorKernel(const Context& dev_ctx,
                               const phi::ExtendedTensor& x,
                               int col,
                               phi::SparseCooTensor* out) {
  PADDLE_ENFORCE_NOT_NULL(
      out,
      platform::errors::NotFound(
          "Output cannot be found in scope for operator 'Feed'"));
  const auto& feed_item = CheckAndGetFeedItem(x, col);
  FeedVariableVisitor<phi::SparseCooTensor> visitor(out, dev_ctx.GetPlace());
  paddle::visit(visitor, feed_item);
}

template <typename T, typename Context>
void FeedStringsKernel(const Context& dev_ctx,
                       const phi::ExtendedTensor& x,
                       int col,
                       phi::ExtendedTensor* out) {
  PADDLE_ENFORCE_NOT_NULL(
      out,
      platform::errors::NotFound(
          "Output cannot be found in scope for operator 'Feed'"));
  const auto& feed_item = CheckAndGetFeedItem(x, col);
  auto strs_out = static_cast<framework::Strings*>(out);
  FeedVariableVisitor<framework::Strings> visitor(strs_out, dev_ctx.GetPlace());
  paddle::visit(visitor, feed_item);
}

class FeedOp : public framework::OperatorWithKernel {
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {}

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(framework::proto::VarType::FP32,
                                   ctx.GetPlace());
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

REGISTER_OPERATOR(
    feed,
    paddle::operators::FeedOp,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>,
    paddle::operators::FeedOpInfoMaker);

PD_REGISTER_KERNEL(feed_dense_tensor,
                   CPU,
                   ALL_LAYOUT,
                   paddle::operators::FeedDenseTensorKernel,
                   float,
                   double,
                   int8_t,
                   uint8_t,
                   int,
                   int64_t,
                   bool,
                   paddle::platform::bfloat16,
                   paddle::platform::complex<float>,
                   paddle::platform::complex<double>,
                   paddle::platform::float16,
                   int16_t) {}
PD_REGISTER_KERNEL(feed_sparse_coo_tensor,
                   CPU,
                   ALL_LAYOUT,
                   paddle::operators::FeedSparseCooTensorKernel,
                   float,
                   double,
                   int8_t,
                   uint8_t,
                   int,
                   int64_t,
                   bool,
                   paddle::platform::bfloat16,
                   paddle::platform::complex<float>,
                   paddle::platform::complex<double>,
                   paddle::platform::float16,
                   int16_t) {}
PD_REGISTER_KERNEL(feed_strings,
                   CPU,
                   ALL_LAYOUT,
                   paddle::operators::FeedStringsKernel,
                   float,
                   double,
                   int8_t,
                   uint8_t,
                   int,
                   int64_t,
                   bool,
                   paddle::platform::bfloat16,
                   paddle::platform::complex<float>,
                   paddle::platform::complex<double>,
                   paddle::platform::float16,
                   int16_t) {}
