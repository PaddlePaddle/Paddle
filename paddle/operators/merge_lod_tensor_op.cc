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

#include "paddle/framework/op_registry.h"
#include "paddle/memory/memcpy.h"

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
  void Run(const framework::Scope &scope,
           const platform::DeviceContext &dev_ctx) const override {
    auto &x = scope.FindVar(Input("X"))->Get<framework::LoDTensor>();
    auto &mask = scope.FindVar(Input("Mask"))->Get<framework::LoDTensor>();
    auto &in_true = scope.FindVar(Input("InTrue"))->Get<framework::LoDTensor>();
    auto &in_false =
        scope.FindVar(Input("InFalse"))->Get<framework::LoDTensor>();
    auto *out =
        scope.FindVar(Output("Out"))->GetMutable<framework::LoDTensor>();
    auto level = static_cast<size_t>(Attr<int>("level"));

    auto &mask_dim = mask.dims();
    auto *mask_data = mask.data<bool>();

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

    for (size_t i = 0; i < static_cast<size_t>(mask_dim[0]); i++) {
      const framework::LoDTensor *input = nullptr;
      if (static_cast<int>(mask_data[i]) == 0) {
        input = &in_false;
      } else {
        input = &in_true;
      }
      auto lod_and_offset =
          framework::GetSubLoDAndAbsoluteOffset(input->lod(), i, i + 1, 0);
      auto &lod_length = lod_and_offset.first;

      framework::AppendLoD(out_lod, lod_length);

      size_t start_offset = lod_and_offset.second.first;
      size_t end_offset = lod_and_offset.second.second;

      PADDLE_ENFORCE_GE(end_offset, start_offset);
      size_t len = end_offset - start_offset;
      if (len == 0) {
        continue;
      }
      out->Slice(out_offset, out_offset + len)
          .CopyFrom(input->Slice(start_offset, end_offset), place, dev_ctx);
      out_offset += len;
    }

    for (int i = 0; i < level; i++) {
      out_lod->insert(out_lod->begin(), x.lod()[i]);
    }
  }
};

class MergeLoDTensorOpProtoMaker : public framework::OpProtoAndCheckerMaker {
 public:
  MergeLoDTensorOpProtoMaker(framework::OpProto *proto,
                             framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "");
    AddInput("Mask", "");
    AddInput("InTrue", "");
    AddInput("InFalse", "");
    AddOutput("Out", "");
    AddAttr<int>("level", "(int) the specific lod level to rank.")
        .SetDefault(0)
        .EqualGreaterThan(0);
    AddComment(
        R"DOC()DOC");
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

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(merge_lod_tensor, ops::MergeLoDTensorOp,
                  ops::MergeLoDTensorOpProtoMaker,
                  ops::MergeLoDTensorInferShape,
                  paddle::framework::EmptyGradOpMaker);
