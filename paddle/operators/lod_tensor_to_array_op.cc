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
#include "paddle/framework/lod_rank_table.h"
#include "paddle/framework/lod_tensor_array.h"
#include "paddle/framework/op_registry.h"

namespace paddle {
namespace operators {

class LoDTensorToArrayOp : public framework::OperatorBase {
 public:
  LoDTensorToArrayOp(const std::string &type,
                     const framework::VariableNameMap &inputs,
                     const framework::VariableNameMap &outputs,
                     const framework::AttributeMap &attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}
  void Run(const framework::Scope &scope,
           const platform::DeviceContext &dev_ctx) const override {
    auto x = scope.FindVar(Input("X"))->Get<framework::LoDTensor>();
    auto x_dim = x->dims();
    auto x_dim_vec = framework::vectorize(x_dim);

    auto rank_table =
        scope.FindVar(Input("RankTable"))->Get<framework::LoDRankTable>();
    auto *out =
        scope.FindVar(Output("Out"))->GetMutable<framework::LoDTensorArray>();

    auto items = rank_table.items();
    auto max_seq_len = items[0].length;
    auto table_height = items.size();

    auto rank_level = rank_table.coarse_lod().size();
    auto x_level = x.lod().size();

    out->resize(max_seq_len);
    auto place = ctx.GetPlace();

    // out InferShape
    for (size_t i = 0; i < max_seq_len; i++) {
      size_t height = 0;
      framework::LoD out_lod;
      out_lod.resize(x_level - rank_level - 1);
      for (size_t j = 0; j < table_height; j++) {
        if (i < items[j].length) {
          for (size_t k = 0; k < x_level - out_level - 1; k++) {
            out_lod[k] = x[k + out_level + 1]
          }
          height++;
        }
      }
      x_dim_vec[0] = height;
      out[i].Resize(framework::make_ddim(x_dim_vec));
      out[i].mutable_data(place, x.type());
    }

    // out CopyFrom
    for (size_t i = 0; i < max_seq_len; i++) {
      size_t out_slice_idx = 0;
      for (size_t j = 0; j < table_height; j++) {
        size_t input_slice_idx = items[j].index + items[j].length + i;
        if (i < items[j].length) {
          out[i]
              .Slice(out_slice_idx, out_slice_idx + 1)
              .CopyFrom(x.Slice(input_slice_idx, input_slice_idx + 1), place,
                        ctx.device_context());
          out_slice_idx++;
        }
      }
    }
  }
};

class LoDTensorToArrayOpProtoMaker : public framework::OpProtoAndCheckerMaker {
 public:
  LoDTensorToArrayOpProtoMaker(framework::OpProto *proto,
                               framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "");
    AddInput("RankTable", "");
    AddOutput("Out", "");
    AddComment("");
  }
};

class LoDTensorToArrayInferShape : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext *context) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of LoDTensorToArrayOp should not be null.");
    PADDLE_ENFORCE(
        ctx->HasInput("RankTable"),
        "Input(RankTable) of LoDTensorToArrayOp should not be null.");

    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of LoDTensorToArrayOp should not be null.");
  }
};

class LoDTensorToArrayInferVarType : public framework::VarTypeInference {
 public:
  void operator()(const framework::OpDescBind &op_desc,
                  framework::BlockDescBind *block) const override {
    for (auto &out_var : op_desc.Output("Out")) {
      block->Var(out_var)->SetType(framework::VarDesc::LOD_TENSOR_ARRAY);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(lod_tensor_to_array, ops::LoDTensorToArrayOp,
                  ops::LoDTensorToArrayOpProtoMaker,
                  ops::LoDTensorToArrayInferShape,
                  ops::LoDTensorToArrayInferVarType);
