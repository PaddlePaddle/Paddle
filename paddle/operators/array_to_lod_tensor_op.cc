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
#include <numeric>
#include "paddle/framework/lod_rank_table.h"
#include "paddle/framework/lod_tensor_array.h"
#include "paddle/framework/op_registry.h"
#include "paddle/memory/memcpy.h"

namespace paddle {
namespace operators {

class ArrayToLoDTensorOp : public framework::OperatorBase {
 public:
  ArrayToLoDTensorOp(const std::string &type,
                     const framework::VariableNameMap &inputs,
                     const framework::VariableNameMap &outputs,
                     const framework::AttributeMap &attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}
  void Run(const framework::Scope &scope,
           const platform::DeviceContext &dev_ctx) const override {
    auto &x = scope.FindVar(Input("X"))->Get<framework::LoDTensorArray>();
    auto &rank_table =
        scope.FindVar(Input("RankTable"))->Get<framework::LoDRankTable>();
    auto *out =
        scope.FindVar(Output("Out"))->GetMutable<framework::LoDTensor>();

    // Check dims, place and data type of input's elements and infer output's
    // dim
    PADDLE_ENFORCE(!x.empty(), "There's no element in the input array.");
    int rank = x[0].dims().size();
    platform::Place place = x[0].place();
    std::type_index data_type = x[0].type();
    framework::DDim ins_dims = framework::slice_ddim(x[0].dims(), 1, rank);
    int64_t batch_size = x[0].dims()[0];
    for (size_t i = 1; i < x.size(); ++i) {
      PADDLE_ENFORCE_EQ(framework::slice_ddim(x[i].dims(), 1, rank), ins_dims,
                        "The dimension of the %zu'th element in LoDTensorArray "
                        "differs from previous ones.",
                        i);
      PADDLE_ENFORCE(platform::places_are_same_class(x[i].place(), place),
                     "The place class of the %zu'th element in LoDTensorArray "
                     "differs from previous ones.",
                     i);
      PADDLE_ENFORCE(x[i].type() == data_type,
                     "The date type of the %zu'th element in LoDTensorArray "
                     "differs from previous ones.",
                     i);
      batch_size += x[i].dims()[0];
    }
    auto ins_dim_vec = framework::vectorize(ins_dims);
    ins_dim_vec.insert(ins_dim_vec.begin(), batch_size);
    framework::DDim out_dims = framework::make_ddim(ins_dim_vec);
    out->Resize(out_dims);
    out->mutable_data(place, data_type);

    auto &table_items = rank_table.items();
    std::vector<size_t> table_item_idx(table_items.size());
    // table_item_idx = range(table_items_idx.size())
    std::iota(table_item_idx.begin(), table_item_idx.end(), 0);
    std::sort(table_item_idx.begin(), table_item_idx.end(),
              [&](size_t a, size_t b) {
                return table_items[a].index < table_items[b].index;
              });

    // Build LoDTensor `out`
    framework::LoD *out_lod = out->mutable_lod();
    out_lod->clear();
    size_t out_offset = 0;
    auto prefix_lod = rank_table.coarse_lod();
    prefix_lod.emplace_back();
    auto &cur_level_lod = prefix_lod.back();
    cur_level_lod.push_back(0);
    for (size_t idx : table_item_idx) {
      cur_level_lod.push_back(cur_level_lod.back() + table_items[idx].length);
      for (size_t x_idx = 0; x_idx < table_items[idx].length; ++x_idx) {
        std::vector<std::vector<size_t>> lod_length;
        size_t start_offset;
        size_t end_offset;
        framework::GetFineGrainedLoDLength2(x[x_idx].lod(), idx, idx + 1, 0,
                                            &lod_length, &start_offset,
                                            &end_offset);
        VLOG(10) << "idx=" << idx << " x_idx=" << x_idx << " [" << start_offset
                 << ", " << end_offset << "]";
        // Append lod
        framework::AppendLoD(out_lod, lod_length);
        // Copy data
        size_t len = end_offset - start_offset;
        if (len == 0) {
          continue;
        }
        out->Slice(out_offset, out_offset + len)
            .CopyFrom(x[x_idx].Slice(start_offset, end_offset), place, dev_ctx);
        out_offset += len;
      }
    }
    out_lod->insert(out_lod->begin(), prefix_lod.begin(), prefix_lod.end());
  }
};

class ArrayToLoDTensorOpProtoMaker : public framework::OpProtoAndCheckerMaker {
 public:
  ArrayToLoDTensorOpProtoMaker(framework::OpProto *proto,
                               framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X",
             "(std::vector<LodTensor>) A vector of tensors that is going to "
             "be casted to a big LoDTensor.");
    AddInput("RankTable",
             "(LoDRankTable) RankTable provides the coarse lod infomation to "
             "build the output LoDTensor. See "
             "'paddle/framework/lod_rank_table.h' for more details.");
    AddOutput("Out", "(LoDTensor) The LoDTensor formed by input tensor array.");
    AddComment(
        R"DOC(This Op build a big LoDTensor from a std::vector<LoDTensor> 
          and a LoDRankTable. It is supposed to be used in getting dynamic RNN's
          outputs back to a normal LoDTensor. The std::vector<LoDTensor> 
          would be the output of RNN Op and the LoDRankTable would be build 
          with RNN's input.)DOC");
  }
};

class ArrayToLoDTensorInferShape : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext *context) const override {
    PADDLE_ENFORCE(context->HasInput("X"),
                   "ArrayToLoDTensorOp must has input X.");
    PADDLE_ENFORCE(context->HasInput("RankTable"),
                   "ArrayToLoDTensorOp must has input RankTable.");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(array_to_lod_tensor, ops::ArrayToLoDTensorOp,
                  ops::ArrayToLoDTensorOpProtoMaker,
                  ops::ArrayToLoDTensorInferShape);
