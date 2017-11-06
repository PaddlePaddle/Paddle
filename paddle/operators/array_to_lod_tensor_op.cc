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
    int64_t batch_size = 0;
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

    auto &table_items = rank_table.items();
    std::vector<size_t> table_item_idx(table_items.size());
    std::iota(std::begin(table_item_idx), std::end(table_item_idx), 0);
    std::sort(table_item_idx.begin(), table_item_idx.end(),
              [&](const size_t &a, const size_t &b) {
                return table_items[a].index < table_items[b].index;
              });

    // Build LoDTensor `out`
    uintptr_t dst_ptr =
        reinterpret_cast<uintptr_t>(out->mutable_data(place, data_type));
    framework::LoD *out_lod = out->mutable_lod();
    out_lod->clear();
    for (const size_t &idx : table_item_idx) {
      size_t seq_len = table_items[idx].length;
      for (size_t x_idx = 0; x_idx < seq_len; ++x_idx) {
        std::vector<std::vector<size_t>> lod_length;
        size_t start_offset;
        framework::GetFineGrainedLoDLength(x[x_idx].lod(), idx, idx + 1,
                                           &lod_length, &start_offset);
        // Append lod
        framework::AppendLoD(out_lod, lod_length);
        // Copy data
        size_t type_sz = framework::SizeOfType(data_type);
        uintptr_t src_ptr = reinterpret_cast<uintptr_t>(x[x_idx].data<void>()) +
                            type_sz * start_offset;
        size_t cpy_len =
            type_sz * (lod_length.back().back() - lod_length.back().front());
        if (platform::is_cpu_place(place)) {
          auto cpu_place = boost::get<platform::CPUPlace>(place);
          memory::Copy(cpu_place, reinterpret_cast<void *>(dst_ptr), cpu_place,
                       reinterpret_cast<const void *>(src_ptr), cpy_len);
        } else if (platform::is_gpu_place(place)) {
#ifdef PADDLE_WITH_CUDA
          auto gpu_place = boost::get<platform::GPUPlace>(place);
          memory::Copy(
              gpu_place, reinterpret_cast<void *>(dst_ptr), gpu_place,
              reinterpret_cast<const void *>(src_ptr), cpy_len,
              reinterpret_cast<const platform::CUDADeviceContext &>(dev_ctx)
                  .stream());
#else
          PADDLE_THROW("GPU not supported");
#endif
        }

        dst_ptr += cpy_len;
      }
    }
    out_lod->insert(out_lod->begin(), rank_table.coarse_lod().begin(),
                    rank_table.coarse_lod().end());
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
