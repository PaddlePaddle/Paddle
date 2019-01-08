// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/for_range.h"

#ifdef __NVCC__
#include <thrust/device_vector.h>
#include "paddle/fluid/framework/array.h"
#endif

namespace paddle {
namespace operators {

class StackOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE_GT(ctx->Inputs("X").size(), 0,
                      "Number of Inputs(X) must be larger than 0");
    PADDLE_ENFORCE(ctx->HasOutput("Y"), "Output(Y) must exist.");

    auto input_dims = ctx->GetInputsDim("X");
    for (size_t i = 1; i < input_dims.size(); ++i) {
      PADDLE_ENFORCE_EQ(input_dims[i], input_dims[0],
                        "Dims of all Inputs(X) must be the same");
    }

    // Only lod of X[0] would be shared with Y
    ctx->ShareLoD("X", /*->*/ "Y");

    int axis = ctx->Attrs().Get<int>("axis");
    int rank = input_dims[0].size();
    PADDLE_ENFORCE(
        axis >= -(rank + 1) && axis < rank + 1,
        "Attr(axis) must be inside [-(rank+1), rank+1), where rank = %d", rank);
    if (axis < 0) axis += (rank + 1);

    auto vec = framework::vectorize2int(input_dims[0]);
    vec.insert(vec.begin() + axis, input_dims.size());
    ctx->SetOutputDim("Y", framework::make_ddim(vec));
  }
};

class StackOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "The input of stack op.").AsDuplicable();
    AddOutput("Y", "The output of stack op.");
    AddAttr<int>("axis",
                 "The axis along which all of the Inputs(X) should be stacked.")
        .SetDefault(0);
    AddComment(R"DOC(
      Stack Operator.

      Stack all of the Inputs(X) into one tensor along Attr(axis). The dims of all Inputs(X) must be the same.
    )DOC");
  }
};

template <typename VecXType, typename T>
struct StackFunctor {
  HOSTDEVICE StackFunctor(const VecXType &x, T *y, int n, int post)
      : x_(x), y_(y), n_(n), post_(post) {}

  HOSTDEVICE void operator()(int idx) {
    int i = idx / (n_ * post_);
    int which_x = idx / post_ - i * n_;
    int x_index = i * post_ + idx % post_;
    y_[idx] = x_[which_x][x_index];
  }

 private:
  VecXType x_;
  T *y_;
  int n_;
  int post_;
};

template <typename VecDxType, typename T>
struct StackGradFunctor {
  HOSTDEVICE StackGradFunctor(const VecDxType &dx, const T *dy, int n, int post)
      : dx_(dx), dy_(dy), n_(n), post_(post) {}

  HOSTDEVICE void operator()(int idx) {
    int i = idx / (n_ * post_);
    int which_x = idx / post_ - i * n_;
    int x_index = i * post_ + idx % post_;
    dx_[which_x][x_index] = dy_[idx];
  }

 private:
  VecDxType dx_;
  const T *dy_;
  int n_;
  int post_;
};

template <typename DeviceContext, typename VecXType, typename T>
static inline void StackFunctorForRange(const DeviceContext &ctx,
                                        const VecXType &x, T *y, int total_num,
                                        int n, int post) {
  platform::ForRange<DeviceContext> for_range(ctx, total_num);
  for_range(StackFunctor<VecXType, T>(x, y, n, post));
}

template <typename DeviceContext, typename VecDxType, typename T>
static inline void StackGradFunctorForRange(const DeviceContext &ctx,
                                            const VecDxType &dx, const T *dy,
                                            int total_num, int n, int post) {
  platform::ForRange<DeviceContext> for_range(ctx, total_num);
  for_range(StackGradFunctor<VecDxType, T>(dx, dy, n, post));
}

template <typename DeviceContext, typename T>
class StackKernel : public framework::OpKernel<T> {
  using Tensor = framework::LoDTensor;

 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto x = ctx.MultiInput<Tensor>("X");
    auto *y = ctx.Output<Tensor>("Y");

    int axis = ctx.Attr<int>("axis");
    if (axis < 0) axis += (x[0]->dims().size() + 1);

    int n = static_cast<int>(x.size());
    auto *y_data = y->mutable_data<T>(ctx.GetPlace());
    std::vector<const T *> x_datas(n);
    for (int i = 0; i < n; i++) x_datas[i] = x[i]->data<T>();

    int pre = 1, post = 1;
    auto &dim = x[0]->dims();
    for (auto i = 0; i < axis; ++i) pre *= dim[i];
    for (auto i = axis; i < dim.size(); ++i) post *= dim[i];

#ifdef __NVCC__
    int total_num = pre * n * post;
    auto &dev_ctx = ctx.template device_context<DeviceContext>();

    thrust::device_vector<const T *> device_x_vec(x_datas);
    auto x_data_arr = device_x_vec.data().get();

    StackFunctorForRange(dev_ctx, x_data_arr, y_data, total_num, n, post);

    // Wait() must be called because device_x_vec may be destructed before
    // kernel ends
    dev_ctx.Wait();
#else
    auto x_data_arr = x_datas.data();

    size_t x_offset = 0;
    size_t y_offset = 0;
    for (int i = 0; i < pre; i++) {
      for (int j = 0; j < n; j++) {
        std::memcpy(y_data + y_offset, x_data_arr[j] + x_offset,
                    post * sizeof(T));
        y_offset += post;
      }
      x_offset += post;
    }
#endif
  }
};

class StackOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Y")),
                   "Input(Y@Grad) must exist.");

    int axis = ctx->Attrs().Get<int>("axis");
    auto dy_dim = ctx->GetInputDim(framework::GradVarName("Y"));
    int rank = dy_dim.size();
    PADDLE_ENFORCE(axis >= -rank && axis < rank,
                   "Attr(axis) must be inside [-rank, rank), where rank = %d",
                   rank);
    if (axis < 0) axis += rank;

    PADDLE_ENFORCE_EQ(ctx->Outputs(framework::GradVarName("X")).size(),
                      static_cast<size_t>(dy_dim[axis]),
                      "Number of Outputs(X@Grad) is wrong");
    auto vec = framework::vectorize2int(dy_dim);
    vec.erase(vec.begin() + axis);
    ctx->SetOutputsDim(
        framework::GradVarName("X"),
        std::vector<framework::DDim>(dy_dim[axis], framework::make_ddim(vec)));
  }
};

class StackGradOpDescMaker : public framework::SingleGradOpDescMaker {
 public:
  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;

 protected:
  std::unique_ptr<framework::OpDesc> Apply() const override {
    std::unique_ptr<framework::OpDesc> op(new framework::OpDesc());
    op->SetType("stack_grad");
    op->SetInput(framework::GradVarName("Y"), OutputGrad("Y"));
    op->SetOutput(framework::GradVarName("X"), InputGrad("X", false));
    op->SetAttrMap(Attrs());
    return op;
  }
};

template <typename DeviceContext, typename T>
class StackGradKernel : public framework::OpKernel<T> {
  using Tensor = framework::LoDTensor;

 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto *dy = ctx.Input<Tensor>(framework::GradVarName("Y"));
    auto dx = ctx.MultiOutput<Tensor>(framework::GradVarName("X"));
    int axis = ctx.Attr<int>("axis");
    if (axis < 0) axis += dy->dims().size();

    int n = dy->dims()[axis];
    std::vector<T *> dx_datas(n);  // NOLINT
    for (int i = 0; i < n; i++) {
      dx_datas[i] = dx[i]->mutable_data<T>(ctx.GetPlace());
    }
    auto dy_data = dy->data<T>();

    int pre = 1;
    for (int i = 0; i < axis; ++i) pre *= dy->dims()[i];
    int total_num = dy->numel();
    int post = total_num / (n * pre);

    auto &dev_ctx = ctx.template device_context<DeviceContext>();
#ifdef __NVCC__
    thrust::device_vector<T *> device_dx_vec(dx_datas);
    auto dx_data_arr = device_dx_vec.data().get();
#else
    auto dx_data_arr = dx_datas.data();
#endif
    StackGradFunctorForRange(dev_ctx, dx_data_arr, dy_data, total_num, n, post);
#ifdef __NVCC__
    // Wait() must be called because device_dx_vec may be destructed before
    // kernel ends
    dev_ctx.Wait();
#endif
  }
};

}  // namespace operators
}  // namespace paddle
