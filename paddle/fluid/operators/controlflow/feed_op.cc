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
<<<<<<< HEAD
#include "paddle/fluid/framework/raw_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
=======
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
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

<<<<<<< HEAD
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

template <typename Context>
void FeedDenseTensorKernel(const Context& dev_ctx,
                           const phi::ExtendedTensor& x,
                           int col,
                           phi::DenseTensor* out) {
  PADDLE_ENFORCE_NOT_NULL(
      out,
      platform::errors::NotFound(
          "Output cannot be found in scope for operator 'Feed'"));
  const auto& feed_item = CheckAndGetFeedItem(x, col);
  const auto& in_tensor = paddle::get<phi::DenseTensor>(feed_item);
  const auto& place = dev_ctx.GetPlace();
  if (platform::is_same_place(in_tensor.place(), place)) {
    out->ShareDataWith(in_tensor);
  } else {
    framework::TensorCopy(in_tensor, place, dev_ctx, out);
  }

  out->set_lod(in_tensor.lod());
}

template <typename Context>
void FeedSparseCooTensorKernel(const Context& dev_ctx,
                               const phi::ExtendedTensor& x,
                               int col,
                               phi::SparseCooTensor* out) {
  PADDLE_ENFORCE_NOT_NULL(
      out,
      platform::errors::NotFound(
          "Output cannot be found in scope for operator 'Feed'"));
  const auto& feed_item = CheckAndGetFeedItem(x, col);
  const auto& in_tensor = paddle::get<phi::SparseCooTensor>(feed_item);
  const auto& place = dev_ctx.GetPlace();
  if (platform::is_same_place(in_tensor.place(), place)) {
    *out = in_tensor;
  } else {
    phi::DenseTensor indices, values;
    framework::TensorCopy(in_tensor.indices(), place, dev_ctx, &indices);
    framework::TensorCopy(in_tensor.values(), place, dev_ctx, &values);
    out->SetMember(indices, values, in_tensor.meta());
  }
}

template <typename Context>
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
  const auto& in_str = paddle::get<framework::Strings>(feed_item);
  strs_out->resize(in_str.size());
  *strs_out = in_str;
}

class FeedOp : public framework::OperatorWithKernel {
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "feed");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "feed");
    if (ctx->IsRuntime()) {
      framework::Variable* x_var =
          PADDLE_GET(framework::Variable*, ctx->GetInputVarPtrs("X")[0]);
      auto& x = x_var->Get<framework::FeedList>();
      int col = ctx->Attrs().Get<int>("col");
      auto& feed_item = x[col];
      if (feed_item.index() == 0) {
        const auto& feed_item = CheckAndGetFeedItem(x, col);
        auto& feed_tensor = PADDLE_GET_CONST(phi::DenseTensor, feed_item);
        ctx->SetOutputDim("Out", feed_tensor.dims());
      } else if (feed_item.index() == 1) {
        auto& feed_str = PADDLE_GET_CONST(framework::Strings, feed_item);
        framework::Variable* out_var =
            PADDLE_GET(framework::Variable*, ctx->GetOutputVarPtrs("Out")[0]);
        out_var->GetMutable<framework::Strings>()->resize(feed_str.size());
      } else {
        auto& feed_sparse_tensor =
            PADDLE_GET_CONST(phi::SparseCooTensor, feed_item);
        framework::Variable* out_var =
            PADDLE_GET(framework::Variable*, ctx->GetOutputVarPtrs("Out")[0]);
        out_var->GetMutable<phi::SparseCooTensor>()->set_meta(
            feed_sparse_tensor.meta());
        out_var->GetMutable<phi::SparseCooTensor>()->SetCoalesced(
            feed_sparse_tensor.coalesced());
        out_var->GetMutable<phi::SparseCooTensor>()->SetIndicesDict(
            feed_sparse_tensor.GetIndicesDict());
      }
    }
  }

 protected:
  phi::KernelKey GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return phi::KernelKey(framework::proto::VarType::FP32, ctx.GetPlace());
=======
// FeedVariableVisitor is to feed the variable data
// according to data type (LoDTensor or  Strings).
class FeedVariableVisitor {
 public:
  explicit FeedVariableVisitor(framework::Variable *out_var,
                               const platform::Place &place)
      : out_var_(out_var), place_(place) {}

  void operator()(const framework::LoDTensor &in_tensor) const {
    framework::LoDTensor *out_tensor =
        out_var_->GetMutable<framework::LoDTensor>();
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
      platform::DeviceContext *context =
          platform::DeviceContextPool::Instance().Get(place_);
      framework::TensorCopy(in_tensor, place_, *context, out_tensor);
    }
    out_tensor->set_lod(in_tensor.lod());
  }

  void operator()(const framework::Strings &in_str) const {
    framework::Strings *out_str = out_var_->GetMutable<framework::Strings>();
    out_str->resize(in_str.size());
    *out_str = in_str;
  }

  void operator()(const phi::SparseCooTensor &in_tensor) const {
    phi::SparseCooTensor *out_tensor =
        out_var_->GetMutable<phi::SparseCooTensor>();
    if (platform::is_same_place(in_tensor.place(), place_)) {
      *out_tensor = in_tensor;
    } else {
      platform::DeviceContext *context =
          platform::DeviceContextPool::Instance().Get(place_);

      phi::DenseTensor indices, values;
      framework::TensorCopy(in_tensor.indices(), place_, *context, &indices);
      framework::TensorCopy(in_tensor.values(), place_, *context, &values);
      out_tensor->SetMember(indices, values, in_tensor.meta());
    }
  }

 private:
  framework::Variable *out_var_;
  const platform::Place &place_;
};

class FeedOp : public framework::OperatorBase {
 public:
  FeedOp(const std::string &type,
         const framework::VariableNameMap &inputs,
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
    PADDLE_ENFORCE_GE(col,
                      0,
                      platform::errors::InvalidArgument(
                          "Expected the column index (the attribute 'col' of "
                          "operator 'Feed') of current feeding variable to be "
                          "no less than 0. But received column index = %d.",
                          col));

    VLOG(3) << "Feed variable " << feed_var_name << "'s " << col
            << " column to variable " << out_name;

    auto &feed_list = feed_var->Get<framework::FeedList>();
    PADDLE_ENFORCE_LT(
        static_cast<size_t>(col),
        feed_list.size(),
        platform::errors::InvalidArgument(
            "The column index of current feeding variable is expected to be "
            "less than the length of feeding list. But received column index = "
            "%d, the length of feeding list = %d",
            col,
            feed_list.size()));

    auto &feed_item = feed_list.at(static_cast<size_t>(col));

    FeedVariableVisitor visitor(out_var, place);
    paddle::visit(visitor, feed_item);
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
  }
};

class FeedOpInfoMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
<<<<<<< HEAD
             "(vector<phi::DenseTensor>) "
             "A feeding list of phi::DenseTensor, which may have "
             "different dimension and data type.");
    AddOutput("Out",
              "(phi::DenseTensor) The phi::DenseTensor which is a copy "
=======
             "(vector<LoDTensor>) "
             "A feeding list of LoDTensor, which may have "
             "different dimension and data type.");
    AddOutput("Out",
              "(LoDTensor) The LoDTensor which is a copy "
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
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

<<<<<<< HEAD
// TODO(YuanRisheng): Maybe we need design a new registry macro for
// registering device independent kernels.

=======
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
REGISTER_OPERATOR(
    feed,
    paddle::operators::FeedOp,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>,
    paddle::operators::FeedOpInfoMaker);
<<<<<<< HEAD

PD_REGISTER_GENERAL_KERNEL(
    feed_dense_tensor,
    CPU,
    ALL_LAYOUT,
    paddle::operators::FeedDenseTensorKernel<phi::CPUContext>,
    ALL_DTYPE) {}

PD_REGISTER_GENERAL_KERNEL(
    feed_sparse_coo_tensor,
    CPU,
    ALL_LAYOUT,
    paddle::operators::FeedSparseCooTensorKernel<phi::CPUContext>,
    ALL_DTYPE) {}

PD_REGISTER_GENERAL_KERNEL(
    feed_strings,
    CPU,
    ALL_LAYOUT,
    paddle::operators::FeedStringsKernel<phi::CPUContext>,
    ALL_DTYPE) {}

#if defined(PADDLE_WITH_MKLDNN)
PD_REGISTER_GENERAL_KERNEL(
    feed_dense_tensor,
    OneDNN,
    ALL_LAYOUT,
    paddle::operators::FeedDenseTensorKernel<phi::OneDNNContext>,
    ALL_DTYPE) {}
PD_REGISTER_GENERAL_KERNEL(
    feed_sparse_coo_tensor,
    OneDNN,
    ALL_LAYOUT,
    paddle::operators::FeedSparseCooTensorKernel<phi::OneDNNContext>,
    ALL_DTYPE) {}
PD_REGISTER_GENERAL_KERNEL(
    feed_strings,
    OneDNN,
    ALL_LAYOUT,
    paddle::operators::FeedStringsKernel<phi::OneDNNContext>,
    ALL_DTYPE) {}
#endif

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
PD_REGISTER_GENERAL_KERNEL(
    feed_dense_tensor,
    GPU,
    ALL_LAYOUT,
    paddle::operators::FeedDenseTensorKernel<phi::GPUContext>,
    ALL_DTYPE) {}
PD_REGISTER_GENERAL_KERNEL(
    feed_sparse_coo_tensor,
    GPU,
    ALL_LAYOUT,
    paddle::operators::FeedSparseCooTensorKernel<phi::GPUContext>,
    ALL_DTYPE) {}
PD_REGISTER_GENERAL_KERNEL(
    feed_strings,
    GPU,
    ALL_LAYOUT,
    paddle::operators::FeedStringsKernel<phi::GPUContext>,
    ALL_DTYPE) {}
#elif defined(PADDLE_WITH_XPU)
PD_REGISTER_GENERAL_KERNEL(
    feed_dense_tensor,
    XPU,
    ALL_LAYOUT,
    paddle::operators::FeedDenseTensorKernel<phi::XPUContext>,
    ALL_DTYPE) {}
PD_REGISTER_GENERAL_KERNEL(
    feed_sparse_coo_tensor,
    XPU,
    ALL_LAYOUT,
    paddle::operators::FeedSparseCooTensorKernel<phi::XPUContext>,
    ALL_DTYPE) {}
PD_REGISTER_GENERAL_KERNEL(
    feed_strings,
    XPU,
    ALL_LAYOUT,
    paddle::operators::FeedStringsKernel<phi::XPUContext>,
    ALL_DTYPE) {}
#elif defined(PADDLE_WITH_ASCEND_CL)
PD_REGISTER_GENERAL_KERNEL(
    feed_dense_tensor,
    npu,
    ALL_LAYOUT,
    paddle::operators::FeedDenseTensorKernel<phi::CustomContext>,
    ALL_DTYPE) {}
PD_REGISTER_GENERAL_KERNEL(
    feed_sparse_coo_tensor,
    npu,
    ALL_LAYOUT,
    paddle::operators::FeedSparseCooTensorKernel<phi::CustomContext>,
    ALL_DTYPE) {}
PD_REGISTER_GENERAL_KERNEL(
    feed_strings,
    npu,
    ALL_LAYOUT,
    paddle::operators::FeedStringsKernel<phi::CustomContext>,
    ALL_DTYPE) {}
#elif defined(PADDLE_WITH_MLU)
PD_REGISTER_GENERAL_KERNEL(
    feed_dense_tensor,
    CustomMLU,
    ALL_LAYOUT,
    paddle::operators::FeedDenseTensorKernel<phi::CustomContext>,
    ALL_DTYPE) {}
PD_REGISTER_GENERAL_KERNEL(
    feed_sparse_coo_tensor,
    CustomMLU,
    ALL_LAYOUT,
    paddle::operators::FeedSparseCooTensorKernel<phi::CustomContext>,
    ALL_DTYPE) {}
PD_REGISTER_GENERAL_KERNEL(
    feed_strings,
    CustomMLU,
    ALL_LAYOUT,
    paddle::operators::FeedStringsKernel<phi::CustomContext>,
    ALL_DTYPE) {}
#endif
#ifdef PADDLE_WITH_CUSTOM_DEVICE
namespace paddle {
namespace operators {
template void FeedDenseTensorKernel<phi::CustomContext>(
    const phi::CustomContext& dev_ctx,
    const phi::ExtendedTensor& x,
    int col,
    phi::DenseTensor* out);
}  // namespace operators
}  // namespace paddle
#endif
=======
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
