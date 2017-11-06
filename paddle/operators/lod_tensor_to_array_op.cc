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
#include "paddle/framework/lod_tensor.h"
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
    auto x_dim = x.dims();
    auto x_dim_vec = framework::vectorize(x_dim);

    auto rank_table =
        scope.FindVar(Input("RankTable"))->Get<framework::LoDRankTable>();
    auto out = *(
        scope.FindVar(Output("Out"))->GetMutable<framework::LoDTensorArray>());

    auto items = rank_table.items();
    auto max_seq_len = items[0].length;
    auto table_height = items.size();

    auto rank_level = rank_table.coarse_lod().size() + 1;

    out.resize(max_seq_len);
    auto place = dev_ctx.GetPlace();

    // set out[i] lod
    for (size_t i = 0; i < max_seq_len; i++) {
      framework::LoD lod;
      lod.resize(rank_level);
      for (size_t j = 0; j < table_height; j++) {
        std::vector<std::vector<size_t>> lod_length;
        size_t start_offset;
        size_t start_idx = x.lod()[rank_level - 1][items[j].index] + i;
        if (i < items[j].length) {
          framework::GetFineGrainedLoDLength2(x.lod(), start_idx, start_idx + 1,
                                              rank_level, &lod_length,
                                              &start_offset);
          framework::AppendLoD(&lod, lod_length);
        }
      }
      out[i].set_lod(lod);
    }

    /*
    for (auto &lod_tensor : out) {
      auto lod = lod_tensor.lod();
      for (auto i : lod[0]) {
        std::cout << i << " ";
      }
      std::cout << std::endl;
    }
    */

    // set out[i] shape
    for (size_t i = 0; i < out.size(); i++) {
      auto lod = out[i].lod();
      x_dim_vec[0] = lod.back().back();
      out[i].Resize(framework::make_ddim(x_dim_vec));
      out[i].mutable_data(place, x.type());
    }

    // out CopyFrom
    for (size_t i = 0; i < max_seq_len; i++) {
      for (size_t j = 0; j < table_height; j++) {
        std::vector<std::vector<size_t>> lod_length;
        size_t start_offset;
        size_t start_idx = x.lod()[rank_level - 1][items[j].index] + i;
        if (i < items[j].length) {
          framework::GetFineGrainedLoDLength2(x.lod(), start_idx, start_idx + 1,
                                              rank_level, &lod_length,
                                              &start_offset);
          /*
          LOG(INFO) << start_offset;
          LOG(INFO) << start_offset + lod_length.back().back();

          LOG(INFO) << out[i].lod().back()[j];
          LOG(INFO) << out[i].lod().back()[j + 1];
          */
          out[i]
              .Slice(out[i].lod().back()[j], out[i].lod().back()[j + 1])
              .CopyFrom(x.Slice(start_offset,
                                start_offset + lod_length.back().back()),
                        place, dev_ctx);
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
    PADDLE_ENFORCE(context->HasInput("X"),
                   "Input(X) of LoDTensorToArrayOp should not be null.");
    PADDLE_ENFORCE(
        context->HasInput("RankTable"),
        "Input(RankTable) of LoDTensorToArrayOp should not be null.");

    PADDLE_ENFORCE(context->HasOutput("Out"),
                   "Output(Out) of LoDTensorToArrayOp should not be null.");

    auto x_dim = context->GetInputDim("X");
    // The first dim of each LoDTensor in Output can only be set at run-time.;
    // We still have to Resize each LoDTensor in Output.
    context->SetOutputDim("Out", x_dim);
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
