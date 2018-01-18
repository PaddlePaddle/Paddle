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
#include "paddle/operators/math/math_function.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;

class BipartiteMatchOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("DisMat"),
                   "Input(DisMat) of BipartiteMatch should not be null.");

    auto dims = ctx->GetInputDim("DisMat");
    PADDLE_ENFORCE_EQ(dims.size(), 2, "The rank of Input(DisMat) must be 2.");

    ctx->SetOutputDim("ColToRowMatchIndices", dims);
    ctx->SetOutputDim("ColToRowMatchDis", dims);
  }
};

template <typename T>
class BipartiteMatchKernel : public framework::OpKernel<T> {
 public:
  // The match_indices must be initialized to -1 at first.
  // The match_dis must be initialized to 0 at first.
  void BipartiteMatch(const Tensor& dis, int* match_indices,
                      T* match_dis) const {
    int64_t row = dis.dims()[0];
    int64_t col = dis.dims()[1];
    auto* dis_data = dis.data<T>();
    std::vector<int> row_pool;
    for (int i = 0; i < row; ++i) {
      row_pool.push_back(i);
    }
    while (row_pool.size() > 0) {
      int max_idx = -1;
      int max_row_idx = -1;
      T max_dis = -1;
      for (int64_t j = 0; j < col; ++j) {
        if (match_indices[j] != -1) {
          continue;
        }
        for (int k = 0; k < row_pool.size(); ++k) {
          int m = row_pool[k];
          // distance is 0 between m-th row and j-th column
          if (dis_data[m * col + j] < 1e-6) {
            continue;
          }
          if (dis_data[m * col + j] > max_dis) {
            max_idx = j;
            max_row_idx = m;
            max_dis = dis_data[m * col + j];
          }
        }
      }
      if (max_idx == -1) {
        // Cannot find good match.
        break;
      } else {
        PADDLE_ENFORCE_EQ(match_indices[max_idx], -1);
        match_indices[max_idx] = max_row_idx;
        match_dis[max_idx] = max_dis;
        // Erase the row index.
        row_pool.erase(
            std::find(row_pool.begin(), row_pool.end(), max_row_idx));
      }
    }
  }

  void Compute(const framework::ExecutionContext& context) const override {
    auto* dis_mat = context.Input<LoDTensor>("DisMat");
    auto* match_indices = context.Output<Tensor>("ColToRowMatchIndices");
    auto* match_dis = context.Output<Tensor>("ColToRowMatchDis");

    auto& dev_ctx = context.device_context<platform::CPUDeviceContext>();

    auto col = dis_mat->dims()[1];

    int64_t n = dis_mat->lod().size() == 0
                    ? 1
                    : static_cast<int64_t>(dis_mat->lod().back().size() - 1);
    match_indices->mutable_data<int>({n, col}, context.GetPlace());
    match_dis->mutable_data<T>({n, col}, context.GetPlace());

    math::SetConstant<platform::CPUDeviceContext, int> iset;
    iset(dev_ctx, match_indices, static_cast<int>(-1));
    math::SetConstant<platform::CPUDeviceContext, T> tset;
    tset(dev_ctx, match_dis, static_cast<T>(0));

    int* indices = match_indices->data<int>();
    T* dis = match_dis->data<T>();
    if (n == 1) {
      BipartiteMatch(*dis_mat, indices, dis);
    } else {
      auto lod = dis_mat->lod().back();
      for (size_t i = 0; i < lod.size() - 1; ++i) {
        Tensor one_ins = dis_mat->Slice(lod[i], lod[i + 1]);
        BipartiteMatch(one_ins, indices + i * col, dis + i * col);
      }
    }
  }
};

class BipartiteMatchOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  BipartiteMatchOpMaker(OpProto* proto, OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput(
        "DisMat",
        "(LoDTensor or Tensor) this input is a 2-D LoDTensor with shape "
        "[K, M]. It is pair-wise distance matrix between the entities "
        "represented by each row and each column. For example, assumed one "
        "entity is A with shape [K], another entity is B with shape [M]. The "
        "DisMat[i][j] is the distance between A[i] and B[j]. The bigger "
        "the distance is, the more similar the pairs are. Please note, "
        "This tensor can contain LoD information to represent a batch of "
        "inputs. One instance of this batch can contain different numbers of "
        "entities.");
    AddOutput("ColToRowMatchIndices",
              "(Tensor) A 2-D Tensor with shape [N, M] in int type. "
              "N is the batch size. If ColToRowMatchIndices[i][j] is -1, it "
              "means B[j] does not match any entity in i-th instance. "
              "Otherwise, it means B[j] is matched to row "
              "RowToColMatchIndices[i][j] in i-th instance. The row number of "
              "i-th instance is saved in RowToColMatchIndices[i][j].");
    AddOutput("ColToRowMatchDis",
              "(Tensor) A 2-D Tensor with shape [N, M] in float type. "
              "N is batch size. If ColToRowMatchIndices[i][j] is -1, "
              "ColToRowMatchDis[i][j] is also -1.0. Otherwise, assumed "
              "RowToColMatchIndices[i][j] = d, and the row offsets of each "
              "instance are called LoD. Then "
              "ColToRowMatchDis[i][j] = DisMat[d+LoD[i]][j]");
    AddComment(R"DOC(
This operator is a greedy bipartite matching algorithm, which is used to
obtain the matching with the (greedy) maximum distance based on the input
distance matrix. There are two outputs to save matched indices and distance.
And this operator only calculate matched indices from column to row.
A simple description, this algothrim matched the best (maximum distance)
row entity to the column entity and the matched indices are not duplicated
in each row of ColToRowMatchIndices. If the column entity is not matched
any row entity, set -1 in ColToRowMatchIndices.

Please note that the input DisMat can be LoDTensor (with LoD) or Tensor.
If LoDTensor with LoD, the height of ColToRowMatchIndices is batch size.
If Tensor, the height of ColToRowMatchIndices is 1.

)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(bipartite_match, ops::BipartiteMatchOp,
                  ops::BipartiteMatchOpMaker,
                  paddle::framework::EmptyGradOpMaker);
REGISTER_OP_CPU_KERNEL(bipartite_match, ops::BipartiteMatchKernel<float>,
                       ops::BipartiteMatchKernel<double>);
