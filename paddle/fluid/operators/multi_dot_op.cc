/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <algorithm>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/operators/strided_memcpy.h"
#include "paddle/fluid/operators/utils.h"

namespace paddle {
namespace operators {
using Tensor = framework::Tensor;

/**
 * @brief compute the output shape and check the input shape valid or not
 */
inline framework::DDim ComputeAndCheckShape(
    const bool is_runtime, const std::vector<framework::DDim>& inputs_dims) {
  const size_t n = inputs_dims.size();
  auto first_dim = inputs_dims[0];

  bool is_vector = false;
  framework::DDim out_dim;

  PADDLE_ENFORCE_LT(
      first_dim.size(), static_cast<size_t>(3),
      platform::errors::InvalidArgument(
          "multi_dot: the first input tensor must be 1D or 2D but got[%d]!",
          static_cast<int>(first_dim.size())));

  // If the first tensor is 1D of size n view it as a row vector (1, n)
  if (first_dim.size() == 1) {
    first_dim = framework::make_ddim({1, static_cast<int>(first_dim[0])});
    is_vector = true;
  }

  auto last_dim = inputs_dims[n - 1];
  PADDLE_ENFORCE_LT(
      last_dim.size(), static_cast<size_t>(3),
      platform::errors::InvalidArgument(
          "the last input tensor of multi_dot must be 1D or 2D but got[%d]!",
          static_cast<int>(first_dim.size())));

  // If the last tensor is 1D of size n view it as a column vector (n, 1)
  if (last_dim.size() == 1) {
    last_dim = framework::make_ddim({static_cast<int>(last_dim[0]), 1});
    out_dim = is_vector ? framework::make_ddim({1})
                        : framework::make_ddim({first_dim[0]});
  } else {
    out_dim = is_vector ? framework::make_ddim({last_dim[1]})
                        : framework::make_ddim({first_dim[0], last_dim[1]});
  }

  auto width = first_dim[1];
  for (size_t i = 1; i < n - 1; i++) {
    PADDLE_ENFORCE_EQ(inputs_dims[i].size(), static_cast<size_t>(2),
                      platform::errors::InvalidArgument(
                          "the input tensor of multi_dot op must be 2D."));

    const auto& tmp_dim = inputs_dims[i];
    PADDLE_ENFORCE_EQ(
        tmp_dim[0], width,
        platform::errors::InvalidArgument(
            "the input matrix does not meet the multiplication requirements."));
    width = tmp_dim[1];
  }

  PADDLE_ENFORCE_EQ(
      last_dim[0], width,
      platform::errors::InvalidArgument(
          "the input matrix does not meet the multiplication requirements."));

  return out_dim;
}

template <typename DeviceContext, typename T>
inline framework::Tensor MatMul(const framework::ExecutionContext& ctx,
                                const framework::Tensor& matrix_a,
                                const framework::Tensor& matrix_b,
                                const framework::DDim& a_dim,
                                const framework::DDim& b_dim) {
  auto place = ctx.GetPlace();
  auto blas = math::GetBlas<DeviceContext, T>(ctx);

  framework::Tensor matrix_c;
  framework::DDim c_dim = framework::make_ddim({a_dim[0], b_dim[1]});
  matrix_c.Resize(c_dim);
  matrix_c.mutable_data<T>(place);

  auto mat_dim_a = math::CreateMatrixDescriptor(a_dim, 0, false);
  auto mat_dim_b = math::CreateMatrixDescriptor(b_dim, 0, false);
  const T alpha = static_cast<T>(1.0);
  blas.MatMul(matrix_a, mat_dim_a, matrix_b, mat_dim_b, alpha, &matrix_c, T(0));
  return matrix_c;
}

/**
 * @brief Recursively calculate matrix multiplication according to the optimal
 * order
 * Let k = order[i,j], then ins[i...j] = ins[i...k] * ins[k+1 ...j]
 *
 * @param
 * ins: the input tensors
 * ins_dims: the shape of ins after reshape
 * order: the optimal order
 * i: the left of sub chain
 * j: the righe of sub chain
 * save_result: set true by backward
 * results: save the intermediate result during backward
 */
template <typename DeviceContext, typename T>
inline framework::Tensor MatChainMul(
    const framework::ExecutionContext& ctx,
    const std::vector<const framework::Tensor*>& ins,
    const std::vector<framework::DDim>& ins_dims,
    const std::vector<uint64_t>& order, const uint64_t i, const uint64_t j,
    const bool save_result, std::vector<framework::Tensor>* results) {
  if (i == j) {
    return *ins[i];
  }

  const auto A = MatChainMul<DeviceContext, T>(ctx, ins, ins_dims, order, i,
                                               order[i * ins.size() + j],
                                               save_result, results);
  framework::DDim a_dim = A.dims();
  if (i == order[i * ins.size() + j]) {
    a_dim = ins_dims[i];
  }

  const auto B = MatChainMul<DeviceContext, T>(ctx, ins, ins_dims, order,
                                               order[i * ins.size() + j] + 1, j,
                                               save_result, results);
  framework::DDim b_dim = B.dims();
  if (j == order[i * ins.size() + j] + 1) {
    b_dim = ins_dims[j];
  }

  auto result = MatMul<DeviceContext, T>(ctx, A, B, a_dim, b_dim);
  if (save_result) {
    (*results)[i * ins.size() + j] = result;
  }
  return result;
}

/**
 * @brief get the optimal order
 */
std::vector<uint64_t> GetOrder(const std::vector<const framework::Tensor*>& ins,
                               const std::vector<framework::DDim>& ins_dims) {
  auto n = ins.size();
  // p: save the ins shape, the ins[i] shape is (p[i], p[i+1])
  std::vector<uint64_t> p(n + 1);
  for (uint64_t i = 0; i < n; i++) {
    p[i] = ins_dims[i][0];
  }
  p[n] = ins_dims[n - 1][1];

  // m[i, j]: save the lowest cost for multiplying ins[i...j]
  std::vector<uint64_t> m(n * n, 0);
  // define ins[i...j] means multiplying matrices from ins[i] to ins[j]
  // order[i, j] = k, this means that ins[i...k] and ins[k...j] fist and then
  // multiply the resulting matrices is the optimal order for ins[i...j]
  std::vector<uint64_t> order(n * n);
  for (uint64_t l = 1; l < n; l++) {
    for (uint64_t i = 0; i < n - l; i++) {
      auto j = i + l;
      m[i * n + j] = 0xffffffff;
      for (uint64_t k = i; k < j; k++) {
        uint64_t q =
            m[i * n + k] + m[(k + 1) * n + j] + p[i] * p[k + 1] * p[j + 1];
        if (q < m[i * n + j]) {
          m[i * n + j] = q;
          order[i * n + j] = k;
        }
      }
    }
  }
  return order;
}

template <typename DeviceContext, typename T>
static inline framework::Tensor MultiDotMatChainOrder(
    const framework::ExecutionContext& ctx,
    const std::vector<const framework::Tensor*>& ins,
    const std::vector<framework::DDim>& ins_dims, const bool save_result,
    std::vector<framework::Tensor>* results) {
  auto order = GetOrder(ins, ins_dims);
  return MatChainMul<DeviceContext, T>(ctx, ins, ins_dims, order, 0,
                                       ins.size() - 1, save_result, results);
}

inline void GetDims(const std::vector<const framework::Tensor*>& ins,
                    std::vector<framework::DDim>* ins_dims) {
  const auto n = ins.size();
  for (size_t i = 0; i < n; i++) {
    (*ins_dims)[i] = ins[i]->dims();
    if (i == 0 && (*ins_dims)[i].size() == 1) {
      (*ins_dims)[i] = framework::make_ddim({1, (*ins_dims)[i][0]});
    } else if (i == n - 1 && (*ins_dims)[i].size() == 1) {
      (*ins_dims)[i] = framework::make_ddim({(*ins_dims)[i][0], 1});
    }
  }
}

class MultiDotOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "The input tensors of multi_dot operator.").AsDuplicable();
    AddOutput("Out", "The output tensor of multi_dot operator");
    AddComment(R"DOC(
Compute the dot product of two or more arrays in a single function call, while automatically selecting the fastest evaluation order.

multi_dot chains MatMul and uses optimal parenthesization of the matrices [1] [2]. Depending on the shapes of the matrices, this can speed up the multiplication a lot.

If the first argument is 1-D it is treated as a row vector. If the last argument is 1-D it is treated as a column vector. The other arguments must be 2-D.
      )DOC");
  }
};

class MultiDotOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInputs("X"), "Input", "X", "multi_dot");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "multi_dot");

    auto inputs_dims = ctx->GetInputsDim("X");

    const size_t inputs_num = inputs_dims.size();
    PADDLE_ENFORCE_GT(
        inputs_num, static_cast<size_t>(1),
        platform::errors::InvalidArgument(
            "The number of input tensors in multi_dot op should > 1."));
    auto out_dims = ComputeAndCheckShape(ctx->IsRuntime(), inputs_dims);
    ctx->SetOutputDim("Out", out_dims);
    ctx->ShareLoD("X", "Out");
  }
};

/**
 * 1. there are only 2 matrices: direct matrix multiplication A*B
 * 2. there are only 3 matrices: calculate the cost of (A*B)*C and A*(B*C),
 *  choose the least cost order for calculation
 * 3. more than 3 matrices: call MultiDotMatChainOrder
 */
template <typename DeviceContext, typename T>
class MultiDotKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto ins = ctx.MultiInput<framework::Tensor>("X");
    auto* out = ctx.Output<framework::Tensor>("Out");

    auto place = ctx.GetPlace();
    out->mutable_data<T>(place);

    auto blas = math::GetBlas<DeviceContext, T>(ctx);

    auto n = ins.size();
    std::vector<framework::DDim> ins_dims(n);
    GetDims(ins, &ins_dims);

    const T scale = static_cast<T>(1.0);
    if (n == 2) {
      auto mat_dim_a = math::CreateMatrixDescriptor(ins_dims[0], 0, false);
      auto mat_dim_b = math::CreateMatrixDescriptor(ins_dims[1], 0, false);
      blas.MatMul(*ins[0], mat_dim_a, *ins[1], mat_dim_b, scale, out, T(0));
    } else if (n == 3) {
      const auto Ma = ins_dims[0][0];
      const auto Ka = ins_dims[0][1];
      const auto Nb = ins_dims[1][1];
      const auto Nc = ins_dims[2][1];
      const uint64_t cost1 = Ma * Nb * (Ka + Nc);
      const uint64_t cost2 = Ka * Nc * (Nb + Ma);
      auto mat_dim_a = math::CreateMatrixDescriptor(ins_dims[0], 0, false);
      auto mat_dim_b = math::CreateMatrixDescriptor(ins_dims[1], 0, false);
      auto mat_dim_c = math::CreateMatrixDescriptor(ins_dims[2], 0, false);
      if (cost1 < cost2) {
        framework::Tensor tmp_out;
        tmp_out.mutable_data<T>(place, Ma * Nb * sizeof(T));
        framework::DDim tmp_dim = framework::make_ddim({Ma, Nb});
        blas.MatMul(*ins[0], mat_dim_a, *ins[1], mat_dim_b, scale, &tmp_out,
                    T(0));
        auto mat_dim_tmp = math::CreateMatrixDescriptor(tmp_dim, 0, false);
        blas.MatMul(tmp_out, mat_dim_tmp, *ins[2], mat_dim_c, scale, out, T(0));
      } else {
        framework::Tensor tmp_out;
        tmp_out.mutable_data<T>(place, Ka * Nc * sizeof(T));
        framework::DDim tmp_dim = framework::make_ddim({Ka, Nc});
        blas.MatMul(*ins[1], mat_dim_b, *ins[2], mat_dim_c, scale, &tmp_out,
                    T(0));
        auto mat_dim_tmp = math::CreateMatrixDescriptor(tmp_dim, 0, false);
        blas.MatMul(*ins[0], mat_dim_a, tmp_out, mat_dim_tmp, scale, out, T(0));
      }
    } else {
      std::vector<framework::Tensor> results;
      const auto tmp = MultiDotMatChainOrder<DeviceContext, T>(
          ctx, ins, ins_dims, false, &results);
      auto out_dim = out->dims();
      *out = tmp;
      out->Resize(out_dim);
    }
  }
};

class MultiDotOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInputs("X"), "Input", "X", "multi_dot");
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Out")), "Input",
                   "Out@GRAD", "multi_dot");

    auto in_x = "X";
    auto out_x_g_n = framework::GradVarName(in_x);
    auto ins_dims = ctx->GetInputsDim(in_x);
    ctx->SetOutputsDim(out_x_g_n, ins_dims);
    ctx->ShareAllLoD(in_x, out_x_g_n);
  }
};

template <typename DeviceContext, typename T>
class MultiDotGradKernel : public framework::OpKernel<T> {
 public:
  /**
   * @brief calculate dA and dB
   * dA = dout * transpose(B)
   * dB = transpose(A) * dout
   */
  void CalcGrad(const framework::ExecutionContext& ctx,
                const framework::Tensor& dout, const framework::Tensor& A,
                const framework::Tensor& B, const framework::DDim& dout_dim,
                const framework::DDim& a_dim, const framework::DDim& b_dim,
                framework::Tensor* dA, framework::Tensor* dB) const {
    auto mat_dim_dout = math::CreateMatrixDescriptor(dout_dim, 0, false);
    auto mat_dim_a = math::CreateMatrixDescriptor(a_dim, 0, true);
    auto mat_dim_b = math::CreateMatrixDescriptor(b_dim, 0, true);
    T alpha = static_cast<T>(1.0);
    auto blas = math::GetBlas<DeviceContext, T>(ctx);
    blas.MatMul(A, mat_dim_a, dout, mat_dim_dout, alpha, dB, T(0));
    blas.MatMul(dout, mat_dim_dout, B, mat_dim_b, alpha, dA, T(0));
  }

  /**
   * @brief calculate multi matrix multiplication grad by a chain order
   * @param
   * dout: the grad of multi matrix multiplication out
   * dx: the out grad of inputs
   * ins: the input tensors
   * ins_dims: the shape of ins after reshape
   * order: the optimal order
   * i: the left of sub chain
   * j: the righe of sub chain
   * results: the intermediate result of farward
   */
  void MatChainMulGrad(const framework::ExecutionContext& ctx,
                       const framework::Tensor& dout,
                       std::vector<framework::Tensor*>* dx,
                       const std::vector<const framework::Tensor*>& ins,
                       const framework::DDim& dout_dim,
                       const std::vector<framework::DDim>& ins_dims,
                       const std::vector<uint64_t>& order, const uint64_t i,
                       const uint64_t j,
                       const std::vector<framework::Tensor>& results) const {
    if (i == j) {
      *((*dx)[i]) = dout;
      return;
    }

    const auto n = ins.size();
    const auto right = order[i * n + j];
    const auto left = order[i * n + j] + 1;
    // get the multi result of left sub chain
    const auto* A = &results[i * n + right];
    framework::DDim a_dim = A->dims();
    if (i == right) {
      A = ins[i];
      a_dim = ins_dims[i];
    }
    // get the multi result of right sub chain
    const auto* B = &results[left * n + j];
    framework::DDim b_dim = B->dims();
    if (left == j) {
      B = ins[j];
      b_dim = ins_dims[j];
    }
    framework::Tensor dA, dB;
    dA.Resize({dout_dim[0], b_dim[0]});
    dB.Resize({a_dim[1], dout_dim[1]});
    dA.mutable_data<T>(ctx.GetPlace());
    dB.mutable_data<T>(ctx.GetPlace());

    CalcGrad(ctx, dout, *A, *B, dout_dim, a_dim, b_dim, &dA, &dB);
    MatChainMulGrad(ctx, dA, dx, ins, dA.dims(), ins_dims, order, i, right,
                    results);
    MatChainMulGrad(ctx, dB, dx, ins, dB.dims(), ins_dims, order, left, j,
                    results);
  }

  void MultiDotGradMatChainOrder(
      const framework::ExecutionContext& ctx, const framework::Tensor& dout,
      const std::vector<const framework::Tensor*>& ins,
      const framework::DDim& dout_dim,
      const std::vector<framework::DDim>& ins_dims,
      std::vector<framework::Tensor*>* dx) const {
    auto order = GetOrder(ins, ins_dims);
    auto n = ins.size();
    std::vector<framework::Tensor> results(n * n);
    MatChainMul<DeviceContext, T>(ctx, ins, ins_dims, order, 0, n - 1, true,
                                  &results);
    MatChainMulGrad(ctx, dout, dx, ins, dout_dim, ins_dims, order, 0, n - 1,
                    results);
  }

  void Compute(const framework::ExecutionContext& ctx) const {
    auto ins = ctx.MultiInput<framework::Tensor>("X");
    auto dout = *ctx.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto dx = ctx.MultiOutput<framework::Tensor>(framework::GradVarName("X"));

    auto blas = math::GetBlas<DeviceContext, T>(ctx);
    auto place = ctx.GetPlace();

    const auto n = ins.size();
    for (size_t i = 0; i < n; i++) {
      dx[i]->mutable_data<T>(place);
    }

    std::vector<framework::DDim> ins_dims(n);
    GetDims(ins, &ins_dims);

    framework::DDim dout_dim = dout.dims();
    if (ins[0]->dims().size() == 1 && ins[n - 1]->dims().size() == 1) {
      dout_dim = framework::make_ddim({1, 1});
    } else if (ins[0]->dims().size() == 1) {
      if (dout_dim.size() == 1) {
        dout_dim = framework::make_ddim({1, dout_dim[0]});
      }
    } else if (ins[n - 1]->dims().size() == 1) {
      if (dout_dim.size() == 1) {
        dout_dim = framework::make_ddim({dout_dim[0], 1});
      }
    }

    T alpha = static_cast<T>(1);
    auto mat_dim_dout = math::CreateMatrixDescriptor(dout_dim, 0, false);
    if (n == 2) {
      CalcGrad(ctx, dout, *ins[0], *ins[1], dout_dim, ins_dims[0], ins_dims[1],
               dx[0], dx[1]);
    } else if (n == 3) {
      const auto Ma = ins_dims[0][0];
      const auto Ka = ins_dims[0][1];
      const auto Nb = ins_dims[1][1];
      const auto Nc = ins_dims[2][1];
      const uint64_t cost1 = Ma * Nb * (Ka + Nc);
      const uint64_t cost2 = Ka * Nc * (Nb + Ma);
      auto mat_dim_a = math::CreateMatrixDescriptor(ins_dims[0], 0, false);
      auto mat_dim_b = math::CreateMatrixDescriptor(ins_dims[1], 0, false);
      auto mat_dim_c = math::CreateMatrixDescriptor(ins_dims[2], 0, false);
      if (cost1 < cost2) {
        framework::Tensor tmp_out, tmp_dout;
        tmp_out.Resize({Ma, Nb});
        tmp_out.mutable_data<T>(place);
        tmp_dout.Resize({mat_dim_dout.height_, Nb});
        tmp_dout.mutable_data<T>(place);
        blas.MatMul(*ins[0], mat_dim_a, *ins[1], mat_dim_b, alpha, &tmp_out,
                    T(0));
        CalcGrad(ctx, dout, tmp_out, *ins[2], dout_dim, tmp_out.dims(),
                 ins_dims[2], &tmp_dout, dx[2]);
        CalcGrad(ctx, tmp_dout, *ins[0], *ins[1], tmp_dout.dims(), ins_dims[0],
                 ins_dims[1], dx[0], dx[1]);
      } else {
        framework::Tensor tmp_out, tmp_dout;
        tmp_out.Resize({Ka, Nc});
        tmp_out.mutable_data<T>(place);
        tmp_dout.Resize({Ka, mat_dim_dout.width_});
        tmp_dout.mutable_data<T>(place);
        blas.MatMul(*ins[1], mat_dim_b, *ins[2], mat_dim_c, alpha, &tmp_out,
                    T(0));
        CalcGrad(ctx, dout, *ins[0], tmp_out, dout_dim, ins_dims[0],
                 tmp_dout.dims(), dx[0], &tmp_dout);
        CalcGrad(ctx, tmp_dout, *ins[1], *ins[2], tmp_dout.dims(), ins_dims[1],
                 ins_dims[2], dx[1], dx[2]);
      }
    } else {
      MultiDotGradMatChainOrder(ctx, dout, ins, dout_dim, ins_dims, &dx);
      if (ins[n - 1]->dims().size() == 1) {
        dx[n - 1]->Resize({dx[n - 1]->dims()[0]});
      }
    }
  }
};

template <typename T>
class MultiDotOpGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("multi_dot_grad");
    op->SetInput("X", this->Input("X"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X", false));
  }
};
template <typename T>
class MultiDotOpDoubleGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> grad_op) const override {
    grad_op->SetType("multi_dot");
    grad_op->SetInput("X", this->Input(("X")));
    grad_op->SetInput("DOut", this->Input(framework::GradVarName("Out")));
    grad_op->SetOutput("DDx", this->OutputGrad(framework::GradVarName("X")));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(multi_dot, ops::MultiDotOp, ops::MultiDotOpMaker,
                  ops::MultiDotOpGradMaker<paddle::framework::OpDesc>,
                  ops::MultiDotOpGradMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(multi_dot_grad, ops::MultiDotOpGrad,
                  ops::MultiDotOpDoubleGradMaker<paddle::framework::OpDesc>,
                  ops::MultiDotOpDoubleGradMaker<paddle::imperative::OpBase>);

REGISTER_OP_CPU_KERNEL(
    multi_dot, ops::MultiDotKernel<paddle::platform::CPUDeviceContext, double>,
    ops::MultiDotKernel<paddle::platform::CPUDeviceContext, float>);
REGISTER_OP_CPU_KERNEL(
    multi_dot_grad,
    ops::MultiDotGradKernel<paddle::platform::CPUDeviceContext, double>,
    ops::MultiDotGradKernel<paddle::platform::CPUDeviceContext, float>);

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
REGISTER_OP_CUDA_KERNEL(
    multi_dot, ops::MultiDotKernel<paddle::platform::CUDADeviceContext, float>,
    ops::MultiDotKernel<paddle::platform::CUDADeviceContext, double>,
    ops::MultiDotKernel<paddle::platform::CUDADeviceContext,
                        paddle::platform::float16>);
REGISTER_OP_CUDA_KERNEL(
    multi_dot_grad,
    ops::MultiDotGradKernel<paddle::platform::CUDADeviceContext, float>,
    ops::MultiDotGradKernel<paddle::platform::CUDADeviceContext, double>,
    ops::MultiDotGradKernel<paddle::platform::CUDADeviceContext,
                            paddle::platform::float16>);
#endif
