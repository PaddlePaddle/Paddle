// Copyright (c) 2018 PaddlePaddle Authors. Any Rights Reserved.
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

#include <algorithm>
#include <set>
#include <string>
#include <vector>

#include "paddle/fluid/framework/data_type_transform.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/operators/reduce_ops/reduce_op_function.h"
#include "paddle/phi/kernels/funcs/math_function.h"
// only can include the headers in paddle/phi/api dirs
#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/framework/phi_utils.h"
#include "paddle/phi/kernels/cpu/reduce.h"

#if defined(__HIPCC__) || defined(__NVCC__) || defined(__xpu__)
#include "paddle/phi/kernels/gpu/reduce.h"
#include "paddle/phi/kernels/gpu/reduce_grad.h"
#endif

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/unary.h"
namespace paddle {
namespace framework {
class OpDesc;
template <typename T>
class EmptyGradOpMaker;
}  // namespace framework
namespace imperative {
class OpBase;
}  // namespace imperative
}  // namespace paddle

namespace paddle {
namespace operators {

namespace ops = paddle::operators;

class ReduceOpUseInputPlace : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "ReduceBaseOp");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "ReduceBaseOp");
    auto x_dims = ctx->GetInputDim("X");
    auto x_rank = x_dims.size();
    auto dims = ctx->Attrs().Get<std::vector<int>>("dim");
    PADDLE_ENFORCE_GT(dims.size(),
                      0,
                      platform::errors::InvalidArgument(
                          "The input dim dimensions of ReduceBaseOp "
                          "should be greater than 0. But received the dim "
                          "dimesions of Reduce = %d.",
                          dims.size()));

    for (size_t i = 0; i < dims.size(); ++i) {
      PADDLE_ENFORCE_LT(dims[i],
                        x_rank,
                        platform::errors::InvalidArgument(
                            "The reduce dim index %d should be in the "
                            "range [-dimension(X), dimension(X)] "
                            "which dimesion = %d. But received dim index = %d.",
                            i,
                            x_rank,
                            dims[i]));
      PADDLE_ENFORCE_GE(dims[i],
                        -x_rank,
                        platform::errors::InvalidArgument(
                            "The reduce dim index %d should be in the "
                            "range [-dimension(X), dimension(X)] "
                            "which dimesion = %d. But received dim index = %d.",
                            i,
                            x_rank,
                            dims[i]));
      if (dims[i] < 0) dims[i] = x_rank + dims[i];
    }
    sort(dims.begin(), dims.end());
    bool reduce_all = ctx->Attrs().Get<bool>("reduce_all");
    bool keep_dim = ctx->Attrs().Get<bool>("keep_dim");
    if (reduce_all) {
      if (keep_dim)
        ctx->SetOutputDim("Out",
                          phi::make_ddim(std::vector<int64_t>(x_rank, 1)));
      else
        ctx->SetOutputDim("Out", {1});
    } else {
      auto dims_vector = vectorize(x_dims);
      if (keep_dim) {
        for (size_t i = 0; i < dims.size(); ++i) {
          dims_vector[dims[i]] = 1;
        }
      } else {
        const int kDelFlag = -2;
        for (size_t i = 0; i < dims.size(); ++i) {
          dims_vector[dims[i]] = kDelFlag;
        }
        dims_vector.erase(
            remove(dims_vector.begin(), dims_vector.end(), kDelFlag),
            dims_vector.end());
      }
      if (!keep_dim && dims_vector.size() == 0) {
        dims_vector.push_back(1);
      }
      auto out_dims = phi::make_ddim(dims_vector);
      ctx->SetOutputDim("Out", out_dims);
      if (dims.size() > 0 && dims[0] != 0) {
        // Only pass LoD when not reducing on the first dim.
        ctx->ShareLoD("X", /*->*/ "Out");
      }
    }
  }

  // oneDNN's reduction kernel is optimized only for reducing throughout the
  // most outer dims, so in case of another type of reduction, it would be
  // better to fallback to native implementation
  static bool HasOptimizedOneDNNKernel(const framework::ExecutionContext& ctx) {
    // native reduce kernels don't support bf16
    // so oneDNN kernel is enforced in that case
    if (ctx.Input<phi::DenseTensor>("X")->dtype() == phi::DataType::BFLOAT16)
      return true;

    if (!ctx.HasAttr("dim") || !ctx.HasAttr("reduce_all")) {
      return false;
    }

    auto reduce_dims = ctx.Attr<std::vector<int>>("dim");
    const bool reduce_all = ctx.Attr<bool>("reduce_all");
    int ndims = ctx.Input<phi::DenseTensor>("X")->dims().size();

    if (reduce_all) {
      return true;
    }

    for (size_t i = 0; i < reduce_dims.size(); ++i) {
      if (reduce_dims[i] < 0) reduce_dims[i] = ndims + reduce_dims[i];
    }
    sort(reduce_dims.begin(), reduce_dims.end());
    for (size_t i = 0; i < reduce_dims.size(); ++i) {
      if (reduce_dims[reduce_dims.size() - i - 1] !=
          static_cast<int>(ndims - i - 1)) {
        return false;
      }
    }

    return true;
  }

  phi::KernelKey GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    phi::KernelKey kt = OperatorWithKernel::GetExpectedKernelType(ctx);
    kt.set_backend(
        phi::TransToPhiBackend(ctx.Input<phi::DenseTensor>("X")->place()));
    return kt;
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

DECLARE_INFER_SHAPE_FUNCTOR(reduce_any,
                            ReduceAnyInferShapeFunctor,
                            PD_INFER_META(phi::ReduceInferMetaBase));

class ReduceAnyOpMaker : public paddle::framework::OpProtoAndCheckerMaker {
 public:
  void Make() final {
    AddInput("X",
             "(Tensor) The input tensor. Tensors with rank at most 6 are "
             "supported.");
    AddOutput("Out", "(Tensor) The result tensor.");
    AddAttr<std::vector<int>>(
        "dim",
        "(list<int>, default {0}) The dimensions to reduce. "
        "Must be in the range [-rank(input), rank(input)). "
        "If `dim[i] < 0`, the dims[i] to reduce is `rank + dims[i]`. "
        "Note that reducing on the first dim will make the LoD info lost.")
        .SetDefault({0})
        .SupportTensor();
    AddAttr<bool>("keep_dim",
                  "(bool, default false) "
                  "If true, retain the reduced dimension with length 1.")
        .SetDefault(false);
    AddAttr<bool>("reduce_all",
                  "(bool, default false) "
                  "If true, output a scalar reduced along all dimensions.")
        .SetDefault(false);
    AddAttr<int>("in_dtype",
                 "(int, default -1)"
                 "The dtype of input, default value is -1, the user could not "
                 "set this value.")
        .SetDefault(-1);
    AddAttr<int>(
        "out_dtype",
        "(int, default -1)"
        "The dtype of output, default value is -1, the dtype is same as intput")
        .SetDefault(-1);
    AddComment(paddle::string::Sprintf(R"DOC(
%s Operator.

This operator computes the %s of input tensor along the given dimension.
The result tensor has 1 fewer dimension than the input unless keep_dim is true.
If reduce_all is true, just reduce along all dimensions and output a scalar.

)DOC",
                                       GetOpType(),
                                       GetName()));
  }

 protected:
  virtual std::string GetName() const { return "reduce_any"; }
  virtual std::string GetOpType() const { return "Reduce reduce_any"; }
};

// kernel's device type is decided by input tensor place, to be consistent with
// compare and logical ops
REGISTER_OPERATOR(
    reduce_any,
    ops::ReduceOpUseInputPlace,
    ReduceAnyOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>,
    ReduceAnyInferShapeFunctor);
