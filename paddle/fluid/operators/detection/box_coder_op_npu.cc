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

#include "paddle/fluid/operators/detection/box_coder_op.h"
#include <vector>
#include "paddle/fluid/operators/npu_op_runner.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename T>
struct BoxCoderFunction {
 public:
  explicit BoxCoderFunction(const framework::ExecutionContext& ctx) : ctx(ctx) {
    place = ctx.GetPlace();
    stream = ctx.template device_context<paddle::platform::NPUDeviceContext>()
                 .stream();
  }
  Tensor Adds(const Tensor& x, float scalar) {
    Tensor y;
    y.mutable_data<T>(x.dims(), place);
    const auto& runner = NpuOpRunner("Adds", {x}, {y}, {{"value", scalar}});
    runner.Run(stream);
    return y;
  }
  Tensor Muls(const Tensor& x, float scalar) {
    Tensor y;
    y.mutable_data<T>(x.dims(), place);
    const auto& runner = NpuOpRunner("Muls", {x}, {y}, {{"value", scalar}});
    runner.Run(stream);
    return y;
  }
  Tensor Mul(const Tensor& x, const Tensor& y) {
    Tensor z;
    z.mutable_data<T>(x.dims(), place);
    const auto& runner = NpuOpRunner("Mul", {x, y}, {z}, {});
    runner.Run(stream);
    return z;
  }
  Tensor SubWithBroadCast(const Tensor& x, const Tensor& y,
                          const framework::DDim& shape) {
    Tensor z;
    z.mutable_data<T>(shape, place);
    const auto& runner = NpuOpRunner("Sub", {x, y}, {z}, {});
    runner.Run(stream);
    return z;
  }
  void DivWithBroadCastVoid(const Tensor& x, const Tensor& y,
                            const framework::DDim& shape, Tensor* z) {
    z->mutable_data<T>(shape, place);
    const auto& runner = NpuOpRunner("Div", {x, y}, {*z}, {});
    runner.Run(stream);
  }
  Tensor DivWithBroadCast(const Tensor& x, const Tensor& y,
                          const framework::DDim& shape) {
    Tensor z;
    DivWithBroadCastVoid(x, y, shape, &z);
    return z;
  }
  Tensor Abs(const Tensor& x) {
    Tensor y;
    y.mutable_data<T>(x.dims(), place);
    const auto& runner = NpuOpRunner("Abs", {x}, {y}, {});
    runner.Run(stream);
    return y;
  }
  Tensor Log(const Tensor& x) {
    Tensor t_x_m1 = Adds(x, -1);
    Tensor y;
    y.mutable_data<T>(x.dims(), place);
    const auto& runner = NpuOpRunner("Log1p", {t_x_m1}, {y}, {});
    runner.Run(stream);
    return y;
  }
  Tensor Exp(const Tensor& x) {
    Tensor y;
    y.mutable_data<T>(x.dims(), place);
    const auto& runner = NpuOpRunner("Exp", {x}, {y}, {});
    runner.Run(stream);
    return y;
  }
  Tensor Dot(const Tensor& x, const Tensor& y) {
    auto dim_x = x.dims();
    auto dim_y = y.dims();
    PADDLE_ENFORCE_EQ(
        dim_x.size(), 2,
        platform::errors::InvalidArgument(
            "x should be a 2-dim tensor, but got %d-dim.", dim_x.size()));
    PADDLE_ENFORCE_EQ(
        dim_y.size(), 2,
        platform::errors::InvalidArgument(
            "y should be a 2-dim tensor, but got %d-dim.", dim_y.size()));
    PADDLE_ENFORCE_EQ(
        dim_x[1], dim_y[0],
        platform::errors::InvalidArgument("Expect dim_x[1] == dim_y[0], but "
                                          "got dim_x[1] = %d, dim_y[0] = %d.",
                                          dim_x[1], dim_y[0]));
    Tensor z;
    z.mutable_data<T>({dim_x[0], dim_y[1]}, place);
    const auto& runner = NpuOpRunner("MatMul", {x, y}, {z}, {});
    runner.Run(stream);
    return z;
  }
  void ConcatVoid(const std::vector<Tensor>& inputs,
                  const framework::DDim& shape_out, int axis, Tensor* output) {
    output->mutable_data<T>(shape_out, place);
    const auto& runner = NpuOpRunner(
        "ConcatD", {inputs}, {*output},
        {{"concat_dim", axis}, {"N", static_cast<int>(inputs.size())}});
    runner.Run(stream);
  }
  Tensor Concat(const std::vector<Tensor>& inputs,
                const framework::DDim& shape_out, int axis) {
    Tensor output;
    ConcatVoid(inputs, shape_out, axis, &output);
    return output;
  }

 private:
  platform::Place place;
  aclrtStream stream;
  const framework::ExecutionContext& ctx;
};

template <typename T>
void Vector2Tensor(const framework::ExecutionContext& ctx,
                   const std::vector<T>& vec, const framework::DDim& ddim,
                   Tensor* tsr) {
  framework::TensorFromVector<T>(vec, tsr);
  ctx.template device_context<paddle::platform::NPUDeviceContext>().Wait();
  tsr->Resize(ddim);
}

template <typename T>
void BoxCoderEnc(const framework::ExecutionContext& ctx, const Tensor* tb,
                 const Tensor* pb, const Tensor* pbv, const bool norm,
                 const std::vector<float>& variance, Tensor* out) {
  auto M = tb->dims()[0];
  auto N = pb->dims()[0];
  auto shape_0 = framework::make_ddim({4, 2});
  Tensor m_diff;
  Tensor m_aver;
  std::vector<T> vec_diff = {static_cast<T>(-1), static_cast<T>(0),
                             static_cast<T>(0),  static_cast<T>(-1),
                             static_cast<T>(1),  static_cast<T>(0),
                             static_cast<T>(0),  static_cast<T>(1)};
  std::vector<T> vec_aver = {static_cast<T>(0.5), static_cast<T>(0),
                             static_cast<T>(0),   static_cast<T>(0.5),
                             static_cast<T>(0.5), static_cast<T>(0),
                             static_cast<T>(0),   static_cast<T>(0.5)};
  Vector2Tensor<T>(ctx, vec_diff, shape_0, &m_diff);
  Vector2Tensor<T>(ctx, vec_aver, shape_0, &m_aver);

  BoxCoderFunction<T> F(ctx);
  Tensor pb_xy = F.Adds(F.Dot(*pb, m_aver), (norm ? 0 : 0.5));
  Tensor pb_wh = F.Adds(F.Dot(*pb, m_diff), (norm ? 0 : 1));
  Tensor tb_xy = F.Dot(*tb, m_aver);
  Tensor tb_wh = F.Adds(F.Dot(*tb, m_diff), (norm ? 0 : 1));

  pb_xy.Resize({1, M, 2});
  pb_wh.Resize({1, M, 2});
  tb_xy.Resize({N, 1, 2});
  tb_wh.Resize({N, 1, 2});

  auto shape_half = framework::make_ddim({N, M, 2});
  auto shape_full = framework::make_ddim({N, M, 4});

  Tensor out_xy_0 = F.DivWithBroadCast(
      F.SubWithBroadCast(tb_xy, pb_xy, shape_half), pb_wh, shape_half);
  Tensor out_wh_0 = F.Log(F.Abs(F.DivWithBroadCast(tb_wh, pb_wh, shape_half)));
  Tensor out_0 = F.Concat({out_xy_0, out_wh_0}, shape_full, 2);

  if (pbv) {
    F.DivWithBroadCastVoid(out_0, *pbv, shape_full, out);
  } else {
    Tensor t_var;
    std::vector<T> vec_var(4);
    for (auto i = 0; i < 4; i++) {
      vec_var[i] = static_cast<T>(variance[i]);
    }
    Vector2Tensor(ctx, vec_var, framework::make_ddim({1, 1, 4}), &t_var);
    F.DivWithBroadCastVoid(out_0, t_var, shape_full, out);
  }
}

template <typename T>
void BoxCoderDec(const framework::ExecutionContext& ctx, const Tensor* tb,
                 const Tensor* pb, const Tensor* pbv, const bool norm,
                 const std::vector<float>& variance, int axis, Tensor* out) {
  auto M = tb->dims()[0];
  auto N = pb->dims()[0];
  LOG(INFO) << M << N;

  auto shape_0 = framework::make_ddim({4, 1});
  std::vector<T> vec_diff_w = {static_cast<T>(-1), static_cast<T>(0),
                               static_cast<T>(1), static_cast<T>(0)};
  std::vector<T> vec_diff_h = {static_cast<T>(0), static_cast<T>(-1),
                               static_cast<T>(0), static_cast<T>(1)};
  std::vector<T> vec_aver_x = {static_cast<T>(0.5), static_cast<T>(0),
                               static_cast<T>(0.5), static_cast<T>(0)};
  std::vector<T> vec_aver_y = {static_cast<T>(0), static_cast<T>(0.5),
                               static_cast<T>(0), static_cast<T>(0.5)};
  Tensor m_diff_w, m_diff_h;
  Tensor m_aver_x, m_aver_y;

  Vector2Tensor<T>(ctx, vec_diff_w, shape_0, &m_diff_w);
  Vector2Tensor<T>(ctx, vec_diff_h, shape_0, &m_diff_h);
  Vector2Tensor<T>(ctx, vec_aver_x, shape_0, &m_aver_x);
  Vector2Tensor<T>(ctx, vec_aver_y, shape_0, &m_aver_y);

  BoxCoderFunction<T> F(ctx);

  Tensor pb_x = F.Adds(F.Dot(*pb, m_aver_x), (norm ? 0 : 0.5));
  Tensor pb_y = F.Adds(F.Dot(*pb, m_aver_y), (norm ? 0 : 0.5));
  Tensor pb_w = F.Adds(F.Dot(*pb, m_diff_w), (norm ? 0 : 1));
  Tensor pb_h = F.Adds(F.Dot(*pb, m_diff_h), (norm ? 0 : 1));

  auto pb_reshape = axis == 0 ? framework::make_ddim({1, M, 1})
                              : framework::make_ddim({M, 1, 1});
  pb_x.Resize(pb_reshape);
  pb_y.Resize(pb_reshape);
  pb_w.Resize(pb_reshape);
  pb_x.Resize(pb_reshape);

  if (pbv) {
    Tensor pbv_t = F.Adds(*pbv, 0);
    if (axis == 0) {
      pbv_t.Resize({1, M, 4});
    } else {
      pbv_t.Resize({M, 1, 4});
    }
    // Tensor tb_x =
  } else {
  }
}

template <typename T>
class BoxCoderNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* prior_box = ctx.Input<Tensor>("PriorBox");
    auto* prior_box_var = ctx.Input<Tensor>("PriorBoxVar");
    auto* target_box = ctx.Input<framework::LoDTensor>("TargetBox");
    auto* output_box = ctx.Output<Tensor>("OutputBox");
    std::vector<float> variance = ctx.Attr<std::vector<float>>("variance");
    const int axis = ctx.Attr<int>("axis");

    auto code_type = GetBoxCodeType(ctx.Attr<std::string>("code_type"));
    bool normalized = ctx.Attr<bool>("box_normalized");

    if (code_type == BoxCodeType::kEncodeCenterSize) {
      BoxCoderEnc<T>(ctx, target_box, prior_box, prior_box_var, normalized,
                     variance, output_box);
    } else {
      BoxCoderDec<T>(ctx, target_box, prior_box, prior_box_var, normalized,
                     variance, axis, output_box);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_NPU_KERNEL(box_coder, ops::BoxCoderNPUKernel<float>,
                       ops::BoxCoderNPUKernel<plat::float16>);
