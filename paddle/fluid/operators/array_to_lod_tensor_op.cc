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
#include <paddle/fluid/operators/math/concat_and_split.h>
#include <numeric>

#include "paddle/fluid/framework/lod_rank_table.h"
#include "paddle/fluid/framework/lod_tensor_array.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/memory/memcpy.h"
#include "paddle/fluid/platform/device_context.h"

namespace paddle {
namespace operators {

using LoD = framework::LoD;

struct ArrayToLoDFunctor;
template <typename DeviceContext>
struct ArrayToLoDFunctorImpl {
  const ArrayToLoDFunctor *prev_functor_;
  DeviceContext *dev_ctx_;

  template <typename T>
  void apply();
};

struct ArrayToLoDFunctor : public boost::static_visitor<void> {
  std::vector<framework::Tensor> in;
  mutable framework::Tensor *out;

  template <typename Place>
  void operator()(Place place) const {
    auto &pool = platform::DeviceContextPool::Instance();
    if (std::is_same<Place, platform::CPUPlace>::value) {
      Apply(static_cast<platform::CPUDeviceContext *>(pool.Get(place)));
    } else {
#ifdef PADDLE_WITH_CUDA
      Apply(static_cast<platform::CUDADeviceContext *>(pool.Get(place)));
#else
      PADDLE_THROW("Fluid is not compiled with CUDA");
#endif
    }
  }

  template <typename DeviceContext>
  void Apply(DeviceContext *dev_ctx) const {
    ArrayToLoDFunctorImpl<DeviceContext> functor;
    functor.dev_ctx_ = dev_ctx;
    functor.prev_functor_ = this;
    framework::VisitDataType(out->type(), functor);
  }
};

template <typename DeviceContext>
template <typename T>
void ArrayToLoDFunctorImpl<DeviceContext>::apply() {
  math::ConcatFunctor<DeviceContext, T> func;
  func(*dev_ctx_, prev_functor_->in, 0, prev_functor_->out);
}

class ArrayToLoDTensorOp : public framework::OperatorBase {
 public:
  ArrayToLoDTensorOp(const std::string &type,
                     const framework::VariableNameMap &inputs,
                     const framework::VariableNameMap &outputs,
                     const framework::AttributeMap &attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

 private:
  void RunImpl(const framework::Scope &scope,
               const platform::Place &dev_place) const override {
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
    auto data_type = x[0].type();
    int64_t batch_size = x[0].dims()[0];
    framework::DDim ins_dims = rank > 1
                                   ? framework::slice_ddim(x[0].dims(), 1, rank)
                                   : framework::make_ddim({0});
    for (size_t i = 1; i < x.size(); ++i) {
      auto ins_i_dims = rank > 1 ? framework::slice_ddim(x[i].dims(), 1, rank)
                                 : framework::make_ddim({0});
      PADDLE_ENFORCE_EQ(ins_i_dims, ins_dims,
                        "The dimension of the %zu'th element in LoDTensorArray "
                        "differs from previous ones.",
                        i);
      PADDLE_ENFORCE(x[i].place() == place,
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
    auto prefix_lod = rank_table.coarse_lod();
    prefix_lod.emplace_back();
    auto &cur_level_lod = prefix_lod.back();
    cur_level_lod.push_back(0);
    ArrayToLoDFunctor functor;
    for (size_t idx : table_item_idx) {
      cur_level_lod.push_back(cur_level_lod.back() + table_items[idx].length);
      PADDLE_ENFORCE_LE(table_items[idx].length, x.size());
      for (size_t x_idx = 0; x_idx < table_items[idx].length; ++x_idx) {
        auto lod_and_offset = framework::GetSubLoDAndAbsoluteOffset(
            x[x_idx].lod(), idx, idx + 1, 0);

        auto &lod_length = lod_and_offset.first;
        framework::AppendLoD(out_lod, lod_length);

        size_t start_offset = lod_and_offset.second.first;
        size_t end_offset = lod_and_offset.second.second;
        VLOG(10) << "idx=" << idx << " x_idx=" << x_idx << " ["
                 << ", " << end_offset << "]";
        // Copy data
        PADDLE_ENFORCE_GE(end_offset, start_offset);
        size_t len = end_offset - start_offset;
        if (len == 0) {
          continue;
        }
        functor.in.emplace_back(x[x_idx].Slice(start_offset, end_offset));
      }
    }
    functor.out = out;
    platform::VisitPlace(place, functor);
    out_lod->insert(out_lod->begin(), prefix_lod.begin(), prefix_lod.end());
  }
};

class ArrayToLoDTensorOpProtoMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
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
    context->SetOutputDim("Out", context->GetInputDim("X"));
  }
};

class ArrayToLoDTensorGradMaker : public framework::SingleGradOpDescMaker {
 public:
  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;

 protected:
  std::unique_ptr<framework::OpDesc> Apply() const override {
    auto *grad_op = new framework::OpDesc();
    grad_op->SetType("lod_tensor_to_array");
    grad_op->SetInput("X", OutputGrad("Out"));
    grad_op->SetInput("RankTable", Input("RankTable"));
    grad_op->SetOutput("Out", InputGrad("X"));
    grad_op->SetAttrMap(Attrs());
    return std::unique_ptr<framework::OpDesc>(grad_op);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(array_to_lod_tensor, ops::ArrayToLoDTensorOp,
                  ops::ArrayToLoDTensorOpProtoMaker,
                  ops::ArrayToLoDTensorInferShape,
                  ops::ArrayToLoDTensorGradMaker);
