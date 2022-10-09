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

#include "paddle/fluid/operators/detection/iou_similarity_op.h"
#include "paddle/fluid/platform/device/npu/npu_op_runner.h"

namespace paddle {
namespace operators {

using Tensor = phi::DenseTensor;

template <typename T>
struct IouFunction {
 public:
  explicit IouFunction(const framework::ExecutionContext& ctx) : ctx(ctx) {
    place = ctx.GetPlace();
    stream = ctx.template device_context<paddle::platform::NPUDeviceContext>()
                 .stream();
  }
  void Transpose(const phi::DenseTensor* x,
                 phi::DenseTensor* y,
                 const std::vector<int>& axis) {
    //  y should be init first
    const auto& runner =
        NpuOpRunner("TransposeD", {*x}, {*y}, {{"perm", axis}});
    runner.Run(stream);
  }
  void Add(const phi::DenseTensor* x,
           const phi::DenseTensor* y,
           phi::DenseTensor* z) {
    //  y should be init first
    const auto& runner = NpuOpRunner("AddV2", {*x, *y}, {*z}, {});
    runner.Run(stream);
  }
  void Sub(const phi::DenseTensor* x,
           const phi::DenseTensor* y,
           phi::DenseTensor* z) {
    //  y should be init first
    const auto& runner = NpuOpRunner("Sub", {*x, *y}, {*z}, {});
    runner.Run(stream);
  }
  void Mul(const phi::DenseTensor* x,
           const phi::DenseTensor* y,
           phi::DenseTensor* z) {
    //  y should be init first
    const auto& runner = NpuOpRunner("Mul", {*x, *y}, {*z}, {});
    runner.Run(stream);
  }
  void DivNoNan(const phi::DenseTensor* x,
                const phi::DenseTensor* y,
                phi::DenseTensor* z) {
    //  y should be init first
    const auto& runner = NpuOpRunner("DivNoNan", {*x, *y}, {*z}, {});
    runner.Run(stream);
  }
  void Adds(const phi::DenseTensor* x, float scalar, phi::DenseTensor* y) {
    //  y should be init first
    const auto& runner = NpuOpRunner("Adds", {*x}, {*y}, {{"value", scalar}});
    runner.Run(stream);
  }
  void Maximum(const phi::DenseTensor* x,
               const phi::DenseTensor* y,
               phi::DenseTensor* z) {
    //  z should be init first
    const auto& runner = NpuOpRunner("Maximum", {*x, *y}, {*z}, {});
    runner.Run(stream);
  }
  void Minimum(const phi::DenseTensor* x,
               const phi::DenseTensor* y,
               phi::DenseTensor* z) {
    //  z should be init first
    const auto& runner = NpuOpRunner("Minimum", {*x, *y}, {*z}, {});
    runner.Run(stream);
  }

 private:
  platform::Place place;
  aclrtStream stream;
  const framework::ExecutionContext& ctx;
};

template <typename T>
class IouSimilarityNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<framework::LoDTensor>("X");
    auto* y = ctx.Input<phi::DenseTensor>("Y");
    bool normalized = ctx.Attr<bool>("box_normalized");
    auto* out = ctx.Output<framework::LoDTensor>("Out");

    auto _type = x->dtype();
    auto place = ctx.GetPlace();

    IouFunction<T> F(ctx);

    auto N = x->dims()[0];
    auto M = y->dims()[0];

    out->mutable_data<T>({N, M}, place);
    Tensor xt(_type);
    Tensor yt(_type);
    xt.mutable_data<T>({4, N}, place);
    yt.mutable_data<T>({4, M}, place);
    std::vector<int> vec_trans = {1, 0};
    F.Transpose(x, &xt, vec_trans);
    F.Transpose(y, &yt, vec_trans);
    Tensor xmin1 = xt.Slice(0, 1);
    Tensor ymin1 = xt.Slice(1, 2);
    Tensor xmax1 = xt.Slice(2, 3);
    Tensor ymax1 = xt.Slice(3, 4);
    Tensor xmin2 = yt.Slice(0, 1);
    Tensor ymin2 = yt.Slice(1, 2);
    Tensor xmax2 = yt.Slice(2, 3);
    Tensor ymax2 = yt.Slice(3, 4);
    xmin1.Resize({N, 1});
    ymin1.Resize({N, 1});
    xmax1.Resize({N, 1});
    ymax1.Resize({N, 1});
    xmin2.Resize({1, M});
    ymin2.Resize({1, M});
    xmax2.Resize({1, M});
    ymax2.Resize({1, M});

    Tensor w1(_type);
    Tensor h1(_type);
    Tensor w2(_type);
    Tensor h2(_type);
    Tensor area1(_type);
    Tensor area2(_type);
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

    Tensor inter_xmax(_type);
    Tensor inter_ymax(_type);
    Tensor inter_xmin(_type);
    Tensor inter_ymin(_type);
    inter_xmax.mutable_data<T>({N, M}, place);
    inter_ymax.mutable_data<T>({N, M}, place);
    inter_xmin.mutable_data<T>({N, M}, place);
    inter_ymin.mutable_data<T>({N, M}, place);
    F.Minimum(&xmax1, &xmax2, &inter_xmax);
    F.Minimum(&ymax1, &ymax2, &inter_ymax);
    F.Maximum(&xmin1, &xmin2, &inter_xmin);
    F.Maximum(&ymin1, &ymin2, &inter_ymin);

    Tensor inter_w(_type);
    Tensor inter_h(_type);
    inter_w.mutable_data<T>({N, M}, place);
    inter_h.mutable_data<T>({N, M}, place);
    F.Sub(&inter_xmax, &inter_xmin, &inter_w);
    F.Sub(&inter_ymax, &inter_ymin, &inter_h);

    if (!normalized) {
      F.Adds(&inter_w, 1.0f, &inter_w);
      F.Adds(&inter_h, 1.0f, &inter_h);
    }
    Tensor zeros(_type);
    zeros.mutable_data<T>({1}, place);
    FillNpuTensorWithConstant<T>(&zeros, static_cast<T>(0));
    F.Maximum(&inter_w, &zeros, &inter_w);
    F.Maximum(&inter_h, &zeros, &inter_h);

    F.Mul(&inter_w, &inter_h, out);
    Tensor union_area(_type);
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

REGISTER_OP_NPU_KERNEL(iou_similarity,
                       ops::IouSimilarityNPUKernel<float>,
                       ops::IouSimilarityNPUKernel<plat::float16>);
