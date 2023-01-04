/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/detection/iou_similarity_op.h"
#include "paddle/fluid/operators/mlu/mlu_baseop.h"

namespace paddle {
namespace operators {

template <typename T>
struct IouFunction {
 public:
  explicit IouFunction(const framework::ExecutionContext& ctx) : ctx(ctx) {
    place = ctx.GetPlace();
  }
  void Transpose(const phi::DenseTensor* x,
                 phi::DenseTensor* y,
                 const std::vector<int>& axis) {
    //  y should be init first
    TransposeFromMLUTensor<T>(ctx, axis, x, y, false /*need_reshape_or_alloc*/);
  }
  void Add(const phi::DenseTensor* x,
           const phi::DenseTensor* y,
           phi::DenseTensor* z) {
    //  y should be init first
    MLUCnnlTensorDesc x_desc(*x);
    MLUCnnlTensorDesc y_desc(*y);
    MLUCnnlTensorDesc z_desc(*z);

    MLUCnnlOpTensorDesc add_op_desc(
        CNNL_OP_TENSOR_ADD, ToCnnlDataType<T>(), CNNL_NOT_PROPAGATE_NAN);
    MLUCnnl::OpTensor(ctx,
                      add_op_desc.get(),
                      x_desc.get(),
                      GetBasePtr(x),
                      y_desc.get(),
                      GetBasePtr(y),
                      z_desc.get(),
                      GetBasePtr(z),
                      ToCnnlDataType<T>());
  }

  void Sub(const phi::DenseTensor* x,
           const phi::DenseTensor* y,
           phi::DenseTensor* z) {
    //  y should be init first
    MLUCnnlTensorDesc x_desc(*x);
    MLUCnnlTensorDesc y_desc(*y);
    MLUCnnlTensorDesc z_desc(*z);

    MLUCnnlOpTensorDesc sub_op_desc(
        CNNL_OP_TENSOR_SUB, ToCnnlDataType<T>(), CNNL_NOT_PROPAGATE_NAN);
    MLUCnnl::OpTensor(ctx,
                      sub_op_desc.get(),
                      x_desc.get(),
                      GetBasePtr(x),
                      y_desc.get(),
                      GetBasePtr(y),
                      z_desc.get(),
                      GetBasePtr(z),
                      ToCnnlDataType<T>());
  }
  void Mul(const phi::DenseTensor* x,
           const phi::DenseTensor* y,
           phi::DenseTensor* z) {
    //  z should be init first
    MLUCnnlTensorDesc x_desc(*x);
    MLUCnnlTensorDesc y_desc(*y);
    MLUCnnlTensorDesc z_desc(*z);

    MLUCnnlOpTensorDesc mul_op_desc(
        CNNL_OP_TENSOR_MUL, ToCnnlDataType<T>(), CNNL_NOT_PROPAGATE_NAN);
    MLUCnnl::OpTensor(ctx,
                      mul_op_desc.get(),
                      x_desc.get(),
                      GetBasePtr(x),
                      y_desc.get(),
                      GetBasePtr(y),
                      z_desc.get(),
                      GetBasePtr(z),
                      ToCnnlDataType<T>());
  }
  void DivNoNan(const phi::DenseTensor* x,
                const phi::DenseTensor* y,
                phi::DenseTensor* z) {
    //  z should be init first
    MLUCnnlTensorDesc x_desc(*x);
    MLUCnnlTensorDesc y_desc(*y);
    MLUCnnlTensorDesc z_desc(*z);

    cnnlComputationPreference_t prefer = CNNL_COMPUTATION_FAST;

    MLUCnnl::DivNoNan(ctx,
                      prefer,
                      x_desc.get(),
                      GetBasePtr(x),
                      y_desc.get(),
                      GetBasePtr(y),
                      z_desc.get(),
                      GetBasePtr(z));
  }
  void Adds(const phi::DenseTensor* x, float scalar, phi::DenseTensor* y) {
    //  y should be init first
    MLUCnnlTensorDesc x_desc(*x);
    MLUCnnlTensorDesc y_desc(*y);
    float alpha = 1.0;
    float beta = scalar;
    MLUCnnl::Transform(ctx,
                       &alpha,
                       &beta,
                       x_desc.get(),
                       GetBasePtr(x),
                       y_desc.get(),
                       GetBasePtr(y));
  }
  void Maximum(const phi::DenseTensor* x,
               const phi::DenseTensor* y,
               phi::DenseTensor* z) {
    //  z should be init first
    MLUCnnlTensorDesc x_desc(*x);
    MLUCnnlTensorDesc y_desc(*y);
    MLUCnnlTensorDesc z_desc(*z);

    MLUCnnl::Maximum(ctx,
                     x_desc.get(),
                     GetBasePtr(x),
                     y_desc.get(),
                     GetBasePtr(y),
                     z_desc.get(),
                     GetBasePtr(z));
  }
  void Minimum(const phi::DenseTensor* x,
               const phi::DenseTensor* y,
               phi::DenseTensor* z) {
    //  z should be init first
    MLUCnnlTensorDesc x_desc(*x);
    MLUCnnlTensorDesc y_desc(*y);
    MLUCnnlTensorDesc z_desc(*z);

    MLUCnnl::Minimum(ctx,
                     x_desc.get(),
                     GetBasePtr(x),
                     y_desc.get(),
                     GetBasePtr(y),
                     z_desc.get(),
                     GetBasePtr(z));
  }

 private:
  platform::Place place;
  const framework::ExecutionContext& ctx;
};

template <typename T>
class IouSimilarityMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<phi::DenseTensor>("X");
    auto* y = ctx.Input<phi::DenseTensor>("Y");
    bool normalized = ctx.Attr<bool>("box_normalized");
    auto* out = ctx.Output<phi::DenseTensor>("Out");

    auto _type = x->dtype();
    auto place = ctx.GetPlace();

    IouFunction<T> F(ctx);

    auto N = x->dims()[0];
    auto M = y->dims()[0];

    out->mutable_data<T>({N, M}, place);
    phi::DenseTensor xt(_type);
    phi::DenseTensor yt(_type);
    xt.mutable_data<T>({4, N}, place);
    yt.mutable_data<T>({4, M}, place);
    std::vector<int> vec_trans = {1, 0};
    F.Transpose(x, &xt, vec_trans);
    F.Transpose(y, &yt, vec_trans);
    phi::DenseTensor xmin1 = xt.Slice(0, 1);
    phi::DenseTensor ymin1 = xt.Slice(1, 2);
    phi::DenseTensor xmax1 = xt.Slice(2, 3);
    phi::DenseTensor ymax1 = xt.Slice(3, 4);
    phi::DenseTensor xmin2 = yt.Slice(0, 1);
    phi::DenseTensor ymin2 = yt.Slice(1, 2);
    phi::DenseTensor xmax2 = yt.Slice(2, 3);
    phi::DenseTensor ymax2 = yt.Slice(3, 4);
    xmin1.Resize({N, 1});
    ymin1.Resize({N, 1});
    xmax1.Resize({N, 1});
    ymax1.Resize({N, 1});
    xmin2.Resize({1, M});
    ymin2.Resize({1, M});
    xmax2.Resize({1, M});
    ymax2.Resize({1, M});

    phi::DenseTensor w1(_type);
    phi::DenseTensor h1(_type);
    phi::DenseTensor w2(_type);
    phi::DenseTensor h2(_type);
    phi::DenseTensor area1(_type);
    phi::DenseTensor area2(_type);
    w1.mutable_data<T>({N, 1}, place);
    h1.mutable_data<T>({N, 1}, place);
    w2.mutable_data<T>({1, M}, place);
    h2.mutable_data<T>({1, M}, place);
    area1.mutable_data<T>({N, 1}, place);
    area2.mutable_data<T>({1, M}, place);
    F.Sub(&xmax1, &xmin1, &w1);
    F.Sub(&ymax1, &ymin1, &h1);
    F.Sub(&xmax2, &xmin2, &w2);
    F.Sub(&ymax2, &ymin2, &h2);
    if (!normalized) {
      F.Adds(&w1, 1.0f, &w1);
      F.Adds(&h1, 1.0f, &h1);
      F.Adds(&w2, 1.0f, &w2);
      F.Adds(&h2, 1.0f, &h2);
    }
    F.Mul(&w1, &h1, &area1);
    F.Mul(&w2, &h2, &area2);

    phi::DenseTensor inter_xmax(_type);
    phi::DenseTensor inter_ymax(_type);
    phi::DenseTensor inter_xmin(_type);
    phi::DenseTensor inter_ymin(_type);
    inter_xmax.mutable_data<T>({N, M}, place);
    inter_ymax.mutable_data<T>({N, M}, place);
    inter_xmin.mutable_data<T>({N, M}, place);
    inter_ymin.mutable_data<T>({N, M}, place);
    F.Minimum(&xmax1, &xmax2, &inter_xmax);
    F.Minimum(&ymax1, &ymax2, &inter_ymax);
    F.Maximum(&xmin1, &xmin2, &inter_xmin);
    F.Maximum(&ymin1, &ymin2, &inter_ymin);

    phi::DenseTensor inter_w(_type);
    phi::DenseTensor inter_h(_type);
    inter_w.mutable_data<T>({N, M}, place);
    inter_h.mutable_data<T>({N, M}, place);
    F.Sub(&inter_xmax, &inter_xmin, &inter_w);
    F.Sub(&inter_ymax, &inter_ymin, &inter_h);

    if (!normalized) {
      F.Adds(&inter_w, 1.0f, &inter_w);
      F.Adds(&inter_h, 1.0f, &inter_h);
    }
    phi::DenseTensor zeros(_type);
    zeros.mutable_data<T>({1}, place);
    FillMLUTensorWithHostValue<T>(ctx, static_cast<T>(0), &zeros);
    F.Maximum(&inter_w, &zeros, &inter_w);
    F.Maximum(&inter_h, &zeros, &inter_h);

    F.Mul(&inter_w, &inter_h, out);
    phi::DenseTensor union_area(_type);
    union_area.mutable_data<T>({N, M}, place);
    F.Add(&area1, &area2, &union_area);
    F.Sub(&union_area, out, &union_area);
    F.DivNoNan(out, &union_area, out);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_MLU_KERNEL(iou_similarity,
                       ops::IouSimilarityMLUKernel<float>,
                       ops::IouSimilarityMLUKernel<plat::float16>);
