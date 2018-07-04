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
#include "paddle/fluid/memory/memcpy.h"

namespace paddle {
namespace operators {

using LoD = framework::LoD;

class MergeLoDTensorOp : public framework::OperatorBase {
 public:
  MergeLoDTensorOp(const std::string &type,
                   const framework::VariableNameMap &inputs,
                   const framework::VariableNameMap &outputs,
                   const framework::AttributeMap &attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

 private:
  void RunImpl(const framework::Scope &scope,
               const platform::Place &dev_place) const override {
    // get device context from pool
    platform::DeviceContextPool &pool = platform::DeviceContextPool::Instance();
    auto &dev_ctx = *pool.Get(dev_place);

    auto &x = scope.FindVar(Input("X"))->Get<framework::LoDTensor>();
    auto &mask = scope.FindVar(Input("Mask"))->Get<framework::LoDTensor>();
    auto &in_true = scope.FindVar(Input("InTrue"))->Get<framework::LoDTensor>();
    auto &in_false =
        scope.FindVar(Input("InFalse"))->Get<framework::LoDTensor>();
    auto *out =
        scope.FindVar(Output("Out"))->GetMutable<framework::LoDTensor>();
    auto level = static_cast<size_t>(Attr<int>("level"));

    auto &mask_dim = mask.dims();

    std::unique_ptr<framework::LoDTensor> cpu_mask{new framework::LoDTensor()};
    if (platform::is_cpu_place(mask.place())) {
      cpu_mask->ShareDataWith(mask);
    } else if (platform::is_gpu_place(mask.place())) {
#ifdef PADDLE_WITH_CUDA
      framework::TensorCopy(mask, platform::CPUPlace(), dev_ctx,
                            cpu_mask.get());
#else
      PADDLE_THROW("Not supported GPU, Please compile WITH_GPU option");
#endif
    }
    auto *mask_data = cpu_mask->data<bool>();

    int rank = in_true.dims().size();
    platform::Place place = in_true.place();
    std::type_index data_type = in_true.type();
    framework::DDim in_true_dims =
        framework::slice_ddim(in_true.dims(), 1, rank);

    int64_t batch_size = in_true.dims()[0] + in_false.dims()[0];

    auto in_true_dim_vec = framework::vectorize(in_true_dims);
    in_true_dim_vec.insert(in_true_dim_vec.begin(), batch_size);

    framework::DDim out_dims = framework::make_ddim(in_true_dim_vec);
    out->Resize(out_dims);
    out->mutable_data(place, data_type);

    auto *out_lod = out->mutable_lod();
    out_lod->clear();
    size_t out_offset = 0;

    // Build LoDTensor `out`

    size_t in_true_idx = 0;
    size_t in_false_idx = 0;
    for (size_t i = 0; i < static_cast<size_t>(mask_dim[0]); i++) {
      const framework::LoDTensor *input = nullptr;
      size_t *in_idx = nullptr;
      if (static_cast<int>(mask_data[i]) == 0) {
        input = &in_false;
        in_idx = &in_false_idx;
      } else {
        input = &in_true;
        in_idx = &in_true_idx;
      }
      auto lod_and_offset = framework::GetSubLoDAndAbsoluteOffset(
          input->lod(), *in_idx, (*in_idx) + 1, 0);
      auto &lod_length = lod_and_offset.first;

      framework::AppendLoD(out_lod, lod_length);

      size_t start_offset = lod_and_offset.second.first;
      size_t end_offset = lod_and_offset.second.second;

      PADDLE_ENFORCE_GE(end_offset, start_offset);
      size_t len = end_offset - start_offset;
      if (len == 0) {
        continue;
      }
      auto slice = out->Slice(out_offset, out_offset + len);
      framework::TensorCopy(input->Slice(start_offset, end_offset), place,
                            dev_ctx, &slice);
      out_offset += len;
      (*in_idx) += 1;
    }

    for (size_t i = 0; i < level; i++) {
      out_lod->insert(out_lod->begin(), x.lod()[i]);
    }
  }
};

class MergeLoDTensorOpProtoMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "The input LoDTensor, contains complete lod information to "
             "construct the output");
    AddInput("Mask", "A bool column vector which mask the input");
    AddInput("InTrue", "The True branch to be merged");
    AddInput("InFalse", "The False branch to be merged");
    AddOutput("Out", "The merged output LoDTensor");
    AddAttr<int>("level", "(int) the specific lod level to rank.")
        .SetDefault(0)
        .EqualGreaterThan(0);
    AddComment(
        R"DOC(
        Merge True and False branches of LoDTensor into a single Output,
        with a mask at certain lod level. X is used to obtain complete
        lod information. Please refer to SplitLoDTensorOp.)DOC");
  }
};

class MergeLoDTensorInferShape : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext *context) const override {
    PADDLE_ENFORCE(context->HasInput("X"),
                   "MergeLoDTensorOp must has input X.");
    PADDLE_ENFORCE(context->HasInput("Mask"),
                   "MergeLoDTensorOp must has input Mask.");
    PADDLE_ENFORCE(context->HasInput("InTrue"),
                   "MergeLoDTensorOp must has input InTrue.");
    PADDLE_ENFORCE(context->HasInput("InFalse"),
                   "MergeLoDTensorOp must has input InFalse.");
    PADDLE_ENFORCE(context->HasOutput("Out"),
                   "MergeLoDTensorOp must has output Out");

    auto mask_dim = context->GetInputDim("Mask");
    PADDLE_ENFORCE_EQ(mask_dim.size(), 2);
    PADDLE_ENFORCE_EQ(mask_dim[1], 1);

    context->SetOutputDim("Out", context->GetInputDim("InTrue"));
  }
};

class MergeLoDTensorGradMaker : public framework::SingleGradOpDescMaker {
 public:
  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;

 protected:
  std::unique_ptr<framework::OpDesc> Apply() const override {
    auto *grad_op = new framework::OpDesc();
    grad_op->SetType("split_lod_tensor");
    grad_op->SetInput("X", OutputGrad("Out"));
    grad_op->SetInput("Mask", Input("Mask"));
    grad_op->SetOutput("OutTrue", InputGrad("InTrue"));
    grad_op->SetOutput("OutFalse", InputGrad("InFalse"));
    grad_op->SetAttrMap(Attrs());
    return std::unique_ptr<framework::OpDesc>(grad_op);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(merge_lod_tensor, ops::MergeLoDTensorOp,
                  ops::MergeLoDTensorOpProtoMaker,
                  ops::MergeLoDTensorInferShape, ops::MergeLoDTensorGradMaker);
