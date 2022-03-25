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

#include "paddle/fluid/operators/interpolate_v2_op.h"
#include "paddle/fluid/platform/device/npu/npu_op_runner.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using DataLayout = framework::DataLayout;
using DDim = framework::DDim;
using fp16 = paddle::platform::float16;

template <typename T>
struct InterpolateFunction {
 public:
  explicit InterpolateFunction(const framework::ExecutionContext& ctx)
      : ctx(ctx) {
    place = ctx.GetPlace();
    stream = ctx.template device_context<paddle::platform::NPUDeviceContext>()
                 .stream();
    t0.mutable_data<float>({1}, place);
    t1.mutable_data<float>({1}, place);
    tn.mutable_data<float>({1}, place);
    FillNpuTensorWithConstant<float>(&t0, static_cast<float>(0));
    FillNpuTensorWithConstant<float>(&t1, static_cast<float>(1));
  }
  void Arange(int n, Tensor* x) {
    FillNpuTensorWithConstant<float>(&tn, static_cast<float>(n));
    const auto& runner = NpuOpRunner("Range", {t0, tn, t1}, {*x}, {});
    runner.Run(stream);
  }
  void ReduceSum(const Tensor* x, Tensor* y, const std::vector<int>& dim,
                 bool keep_dims = true) {
    const auto& runner = NpuOpRunner("ReduceSumD", {*x}, {*y},
                                     {{"axes", dim}, {"keep_dims", keep_dims}});
    runner.Run(stream);
  }
  void Add(const Tensor* x, const Tensor* y, Tensor* z) {
    const auto& runner = NpuOpRunner("AddV2", {*x, *y}, {*z}, {});
    runner.Run(stream);
  }
  void Adds(const Tensor* x, float scalar, Tensor* y) {
    const auto& runner = NpuOpRunner("Adds", {*x}, {*y}, {{"value", scalar}});
    runner.Run(stream);
  }
  void Mul(const Tensor* x, const Tensor* y, Tensor* z) {
    const auto& runner = NpuOpRunner("Mul", {*x, *y}, {*z}, {});
    runner.Run(stream);
  }
  void Sub(const Tensor* x, const Tensor* y, Tensor* z) {
    const auto& runner = NpuOpRunner("Sub", {*x, *y}, {*z}, {});
    runner.Run(stream);
  }
  void Cast(const Tensor* x, Tensor* y) {
    auto dst_dtype =
        ConvertToNpuDtype(framework::TransToProtoVarType(y->dtype()));
    const auto& runner = NpuOpRunner(
        "Cast", {*x}, {*y}, {{"dst_type", static_cast<int>(dst_dtype)}});
    runner.Run(stream);
  }
  void Gather(const Tensor* x, const Tensor* indices, const int axis,
              Tensor* y) {
    const auto& runner =
        NpuOpRunner("GatherV2D", {*x, *indices}, {*y}, {{"axis", axis}});
    runner.Run(stream);
  }
  void GatherGrad(const Tensor* gy, const Tensor* indices, const int axis,
                  Tensor* gx) {
    //  1  gy swapaxis: axis & 0
    int len = (gy->dims()).size();
    std::vector<int> axis_swap(len);
    for (int i = 0; i < len; i++) {
      axis_swap[i] = i;
    }
    axis_swap[0] = axis;
    axis_swap[axis] = 0;
    auto y_new_shape = gy->dims();
    auto yt = y_new_shape[axis];
    y_new_shape[axis] = y_new_shape[0];
    y_new_shape[0] = yt;
    Tensor gy_t;
    gy_t.mutable_data<T>(y_new_shape, place);
    Transpose(gy, &gy_t, axis_swap);
    //  2  scatter
    auto x_new_shape = gx->dims();
    auto xt = x_new_shape[axis];
    x_new_shape[axis] = x_new_shape[0];
    x_new_shape[0] = xt;
    Tensor gx_zero, gx_t;
    gx_zero.mutable_data<T>(x_new_shape, place);
    gx_t.mutable_data<T>(x_new_shape, place);
    FillNpuTensorWithConstant<T>(&gx_zero, static_cast<T>(0));
    gx_zero.Resize(x_new_shape);
    Scatter(&gx_zero, indices, &gy_t, &gx_t);
    //  3  gx swapaxis: axis, 0
    Transpose(&gx_t, gx, axis_swap);
  }
  void Scatter(const Tensor* x, const Tensor* index, const Tensor* updates,
               Tensor* y) {
    const auto& runner =
        NpuOpRunner("TensorScatterAdd", {*x, *index, *updates}, {*y}, {});
    runner.Run(stream);
  }
  void Transpose(const Tensor* x, Tensor* y, const std::vector<int>& axis) {
    const auto& runner =
        NpuOpRunner("TransposeD", {*x}, {*y}, {{"perm", axis}});
    runner.Run(stream);
  }
  void Muls(const Tensor* x, float scalar, Tensor* y) {
    const auto& runner = NpuOpRunner("Muls", {*x}, {*y}, {{"value", scalar}});
    runner.Run(stream);
  }
  void Maximum(const Tensor* x, const Tensor* y, Tensor* z) {
    const auto& runner = NpuOpRunner("Maximum", {*x, *y}, {*z}, {});
    runner.Run(stream);
  }
  void Minimum(const Tensor* x, const Tensor* y, Tensor* z) {
    const auto& runner = NpuOpRunner("Minimum", {*x, *y}, {*z}, {});
    runner.Run(stream);
  }
  void Floor(const Tensor* x, Tensor* y) {
    const auto& runner = NpuOpRunner("Floor", {*x}, {*y}, {});
    runner.Run(stream);
  }

 private:
  platform::Place place;
  aclrtStream stream;
  const framework::ExecutionContext& ctx;
  Tensor t0;
  Tensor t1;
  Tensor tn;
};

template <>
void InterpolateFunction<fp16>::Arange(int n, Tensor* x) {
  Tensor x_fp32(experimental::DataType::FLOAT32);
  x_fp32.mutable_data<float>(x->dims(), place);
  FillNpuTensorWithConstant<float>(&tn, static_cast<float>(n));
  const auto& runner = NpuOpRunner("Range", {t0, tn, t1}, {x_fp32}, {});
  runner.Run(stream);
  Cast(&x_fp32, x);
}

void InterpolateParamCompute(const float scale_h, const float scale_w,
                             const bool align_corners, const int align_mode,
                             const DataLayout& data_layout, const DDim& indim,
                             const DDim& outdim, int* axis_h, int* axis_w,
                             int* in_h, int* in_w, int* out_h, int* out_w,
                             float* ratio_h, float* ratio_w) {
  if (data_layout == DataLayout::kNCHW) {
    *axis_h = 2;
    *axis_w = 3;
  } else {
    *axis_h = 1;
    *axis_w = 2;
  }
  *out_h = outdim[*axis_h];
  *out_w = outdim[*axis_w];
  *in_h = indim[*axis_h];
  *in_w = indim[*axis_w];
  *ratio_h = 0.0f;
  *ratio_w = 0.0f;
  if (*out_h > 1) {
    *ratio_h =
        align_corners
            ? static_cast<float>(*in_h - 1) / (*out_h - 1)
            : (scale_h > 0 ? 1 / scale_h : static_cast<float>(*in_h) / *out_h);
  }
  if (*out_w > 1) {
    *ratio_w =
        align_corners
            ? static_cast<float>(*in_w - 1) / (*out_w - 1)
            : (scale_w > 0 ? 1 / scale_w : static_cast<float>(*in_w) / *out_w);
  }
}

template <typename T>
void BilinearParamTensorCompute(const framework::ExecutionContext& ctx,
                                const DataLayout& data_layout, int in_h,
                                int in_w, int out_h, int out_w, bool align_cond,
                                float ratio_h, float ratio_w, Tensor* h0,
                                Tensor* h1, Tensor* w0, Tensor* w1,
                                Tensor* coef_h0, Tensor* coef_h1,
                                Tensor* coef_w0, Tensor* coef_w1) {
  InterpolateFunction<T> F(ctx);
  auto place = ctx.GetPlace();
  Tensor _h0, _w0;
  _h0.mutable_data<T>({out_h}, place);
  _w0.mutable_data<T>({out_w}, place);
  F.Arange(out_h, &_h0);
  F.Arange(out_w, &_w0);
  if (align_cond) {
    F.Adds(&_h0, static_cast<float>(0.5), &_h0);
    F.Adds(&_w0, static_cast<float>(0.5), &_w0);
    F.Muls(&_h0, ratio_h, &_h0);
    F.Muls(&_w0, ratio_w, &_w0);
    F.Adds(&_h0, static_cast<float>(-0.5), &_h0);
    F.Adds(&_w0, static_cast<float>(-0.5), &_w0);
  } else {
    F.Muls(&_h0, ratio_h, &_h0);
    F.Muls(&_w0, ratio_w, &_w0);
  }

  Tensor zero_t;
  Tensor one_t;
  zero_t.mutable_data<T>({1}, place);
  one_t.mutable_data<T>({1}, place);
  FillNpuTensorWithConstant<T>(&zero_t, static_cast<T>(0));
  FillNpuTensorWithConstant<T>(&one_t, static_cast<T>(1));
  F.Maximum(&_h0, &zero_t, &_h0);
  F.Maximum(&_w0, &zero_t, &_w0);

  Tensor _h0_floor, _w0_floor;
  _h0_floor.mutable_data<T>({out_h}, place);
  _w0_floor.mutable_data<T>({out_w}, place);
  F.Floor(&_h0, &_h0_floor);
  F.Floor(&_w0, &_w0_floor);
  F.Cast(&_h0_floor, h0);
  F.Cast(&_w0_floor, w0);

  Tensor one_int;
  one_int.mutable_data<int>({1}, place);
  FillNpuTensorWithConstant<int>(&one_int, static_cast<int>(1));
  F.Add(h0, &one_int, h1);
  F.Add(w0, &one_int, w1);
  Tensor t_max_h, t_max_w;
  t_max_h.mutable_data<int>({1}, place);
  t_max_w.mutable_data<int>({1}, place);
  FillNpuTensorWithConstant<int>(&t_max_h, static_cast<int>(in_h - 1));
  FillNpuTensorWithConstant<int>(&t_max_w, static_cast<int>(in_w - 1));
  F.Minimum(h1, &t_max_h, h1);
  F.Minimum(w1, &t_max_w, w1);

  F.Sub(&_h0, &_h0_floor, coef_h1);
  F.Sub(&_w0, &_w0_floor, coef_w1);
  F.Sub(&one_t, coef_h1, coef_h0);
  F.Sub(&one_t, coef_w1, coef_w0);

  if (data_layout == DataLayout::kNCHW) {
    coef_h0->Resize({out_h, 1});
    coef_h1->Resize({out_h, 1});
  } else {
    coef_h0->Resize({out_h, 1, 1});
    coef_h1->Resize({out_h, 1, 1});
    coef_w0->Resize({out_w, 1});
    coef_w1->Resize({out_w, 1});
  }
}

template <typename T>
void BilinearFwdNpu(const framework::ExecutionContext& ctx, const Tensor* input,
                    Tensor* output, const float scale_h, const float scale_w,
                    const bool align_corners, const int align_mode,
                    const DataLayout& data_layout) {
  InterpolateFunction<T> F(ctx);
  auto place = ctx.GetPlace();
  auto outdim = output->dims();
  auto indim = input->dims();

  int axis_h, axis_w;
  int out_h, out_w, in_h, in_w;
  float ratio_h, ratio_w;
  InterpolateParamCompute(scale_h, scale_w, align_corners, align_mode,
                          data_layout, indim, outdim, &axis_h, &axis_w, &in_h,
                          &in_w, &out_h, &out_w, &ratio_h, &ratio_w);

  Tensor h0, h1, w0, w1;
  h0.mutable_data<int>({out_h}, place);
  h1.mutable_data<int>({out_h}, place);
  w0.mutable_data<int>({out_w}, place);
  w1.mutable_data<int>({out_w}, place);
  Tensor coef_h0, coef_h1, coef_w0, coef_w1;
  coef_h0.mutable_data<T>({out_h}, place);
  coef_h1.mutable_data<T>({out_h}, place);
  coef_w0.mutable_data<T>({out_w}, place);
  coef_w1.mutable_data<T>({out_w}, place);
  bool align_cond = align_mode == 0 && !align_corners;
  BilinearParamTensorCompute<T>(ctx, data_layout, in_h, in_w, out_h, out_w,
                                align_cond, ratio_h, ratio_w, &h0, &h1, &w0,
                                &w1, &coef_h0, &coef_h1, &coef_w0, &coef_w1);

  Tensor input_gather_h0, input_gather_h1;
  auto dim_gather_h = indim;
  dim_gather_h[axis_h] = out_h;
  input_gather_h0.mutable_data<T>(dim_gather_h, place);
  input_gather_h1.mutable_data<T>(dim_gather_h, place);

  F.Gather(input, &h0, axis_h, &input_gather_h0);
  F.Gather(input, &h1, axis_h, &input_gather_h1);

  F.Mul(&input_gather_h0, &coef_h0, &input_gather_h0);
  F.Mul(&input_gather_h1, &coef_h1, &input_gather_h1);
  Tensor out_x4;
  out_x4.mutable_data<T>({4, outdim[0], outdim[1], outdim[2], outdim[3]},
                         place);
  Tensor input_gather_h0_w0 = out_x4.Slice(0, 1);
  Tensor input_gather_h0_w1 = out_x4.Slice(1, 2);
  Tensor input_gather_h1_w0 = out_x4.Slice(2, 3);
  Tensor input_gather_h1_w1 = out_x4.Slice(3, 4);
  F.Gather(&input_gather_h0, &w0, axis_w, &input_gather_h0_w0);
  F.Gather(&input_gather_h0, &w1, axis_w, &input_gather_h0_w1);
  F.Gather(&input_gather_h1, &w0, axis_w, &input_gather_h1_w0);
  F.Gather(&input_gather_h1, &w1, axis_w, &input_gather_h1_w1);
  F.Mul(&input_gather_h0_w0, &coef_w0, &input_gather_h0_w0);
  F.Mul(&input_gather_h0_w1, &coef_w1, &input_gather_h0_w1);
  F.Mul(&input_gather_h1_w0, &coef_w0, &input_gather_h1_w0);
  F.Mul(&input_gather_h1_w1, &coef_w1, &input_gather_h1_w1);
  F.ReduceSum(&out_x4, output, std::vector<int>{0}, false);
}

template <typename T>
void BilinearBwdNpu(const framework::ExecutionContext& ctx, const Tensor* gout,
                    Tensor* gin, const float scale_h, const float scale_w,
                    const bool align_corners, const int align_mode,
                    const DataLayout& data_layout) {
  InterpolateFunction<T> F(ctx);
  auto place = ctx.GetPlace();
  auto outdim = gout->dims();
  auto indim = gin->dims();

  int axis_h, axis_w;
  int out_h, out_w, in_h, in_w;
  float ratio_h, ratio_w;
  InterpolateParamCompute(scale_h, scale_w, align_corners, align_mode,
                          data_layout, indim, outdim, &axis_h, &axis_w, &in_h,
                          &in_w, &out_h, &out_w, &ratio_h, &ratio_w);

  Tensor h0, h1, w0, w1;
  h0.mutable_data<int>({out_h}, place);
  h1.mutable_data<int>({out_h}, place);
  w0.mutable_data<int>({out_w}, place);
  w1.mutable_data<int>({out_w}, place);
  Tensor coef_h0, coef_h1, coef_w0, coef_w1;
  coef_h0.mutable_data<T>({out_h}, place);
  coef_h1.mutable_data<T>({out_h}, place);
  coef_w0.mutable_data<T>({out_w}, place);
  coef_w1.mutable_data<T>({out_w}, place);
  bool align_cond = align_mode == 0 && !align_corners;
  BilinearParamTensorCompute<T>(ctx, data_layout, in_h, in_w, out_h, out_w,
                                align_cond, ratio_h, ratio_w, &h0, &h1, &w0,
                                &w1, &coef_h0, &coef_h1, &coef_w0, &coef_w1);

  Tensor gy_w0, gy_w1;
  gy_w0.mutable_data<T>(outdim, place);
  gy_w1.mutable_data<T>(outdim, place);
  F.Mul(gout, &coef_w0, &gy_w0);
  F.Mul(gout, &coef_w1, &gy_w1);

  auto dim_gather_h = indim;
  dim_gather_h[axis_h] = out_h;
  Tensor g_gather_w0, g_gather_w1;
  g_gather_w0.mutable_data<T>(dim_gather_h, place);
  g_gather_w1.mutable_data<T>(dim_gather_h, place);
  w0.Resize({out_w, 1});
  w1.Resize({out_w, 1});
  F.GatherGrad(&gy_w0, &w0, axis_w, &g_gather_w0);
  F.GatherGrad(&gy_w1, &w1, axis_w, &g_gather_w1);

  F.Add(&g_gather_w0, &g_gather_w1, &g_gather_w0);
  F.Mul(&g_gather_w0, &coef_h1, &g_gather_w1);
  F.Mul(&g_gather_w0, &coef_h0, &g_gather_w0);

  Tensor gx_0, gx_1;
  gx_0.mutable_data<T>(indim, place);
  gx_1.mutable_data<T>(indim, place);
  h0.Resize({out_h, 1});
  h1.Resize({out_h, 1});
  F.GatherGrad(&g_gather_w0, &h0, axis_h, &gx_0);
  F.GatherGrad(&g_gather_w1, &h1, axis_h, &gx_1);

  F.Add(&gx_0, &gx_1, gin);
}

template <typename DeviceContext, typename T>
class InterpolateV2NPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* input = ctx.Input<Tensor>("X");
    auto* output = ctx.Output<Tensor>("Out");

    auto input_dims = input->dims();
    PADDLE_ENFORCE_EQ(input_dims.size(), 4UL,
                      platform::errors::External(
                          "NPU Interpolate Kernel only support 4-D Tensor."));

    const std::string data_layout_str = ctx.Attr<std::string>("data_layout");
    const DataLayout data_layout =
        framework::StringToDataLayout(data_layout_str);
    int n, c, in_d, in_h, in_w;
    ExtractNCDWH(input_dims, data_layout, &n, &c, &in_d, &in_h, &in_w);

    auto interp_method = ctx.Attr<std::string>("interp_method");
    bool align_corners = ctx.Attr<bool>("align_corners");

    // To-do(qili93): need to support align_corners = true case, try ReSizeD
    PADDLE_ENFORCE_EQ(
        align_corners, false,
        platform::errors::InvalidArgument(
            "NPU Interpolate Kernel has diff when align_corners is true."));

    int out_h = ctx.Attr<int>("out_h");
    int out_w = ctx.Attr<int>("out_w");
    float scale_h = -1;
    float scale_w = -1;

    // Priority: SizeTensor > OutSize > Scale > scale > out_h & out_w
    auto list_new_shape_tensor =
        ctx.MultiInput<framework::Tensor>("SizeTensor");
    if (list_new_shape_tensor.size() > 0) {
      std::vector<int32_t> output_h(1);
      std::vector<int32_t> output_w(1);
      auto dev_ctx =
          platform::DeviceContextPool::Instance().Get(ctx.GetPlace());
      framework::TensorToVector(*list_new_shape_tensor[0], *dev_ctx, &output_h);
      framework::TensorToVector(*list_new_shape_tensor[1], *dev_ctx, &output_w);
      out_h = output_h[0];
      out_w = output_w[0];
    } else if (ctx.HasInput("OutSize")) {
      auto out_size = ctx.Input<Tensor>("OutSize");
      auto out_size_data = get_new_data_from_tensor<int>(out_size);
      out_h = out_size_data[0];
      out_w = out_size_data[1];
    } else {
      auto scale_tensor = ctx.Input<Tensor>("Scale");
      auto scale = ctx.Attr<std::vector<float>>("scale");
      if (scale_tensor != nullptr) {
        auto scale_data = get_new_data_from_tensor<float>(scale_tensor);
        if (scale_data.size() > 1) {
          scale_h = scale_data[0];
          scale_w = scale_data[1];
        } else {
          scale_h = scale_data[0];
          scale_w = scale_data[0];
        }
        PADDLE_ENFORCE_EQ(
            scale_w > 0, true,
            platform::errors::InvalidArgument(
                "The scale_w in input 'Scale' Tensor of Operator(interpolate) "
                "should be greater than 0, but received value is %d.",
                scale_w));
        PADDLE_ENFORCE_EQ(
            scale_h > 0, true,
            platform::errors::InvalidArgument(
                "The scale_h in input 'Scale' Tensor of Operator(interpolate) "
                "should be greater than 0, but received value is %d.",
                scale_h));
      } else {
        if (scale.size() > 1) {
          scale_h = scale[0];
          scale_w = scale[1];

          PADDLE_ENFORCE_EQ(
              scale_w > 0, true,
              platform::errors::InvalidArgument(
                  "The scale_w in Attr(scale) of Operator(interpolate) "
                  "should be greater than 0, but received value is %d.",
                  scale_w));
          PADDLE_ENFORCE_EQ(
              scale_h > 0, true,
              platform::errors::InvalidArgument(
                  "The scale_h in Attr(scale) of Operator(interpolate) "
                  "should be greater than 0, but received value is %d.",
                  scale_h));
        }
      }
      if (scale_h > 0. && scale_w > 0.) {
        out_h = static_cast<int>(in_h * scale_h);
        out_w = static_cast<int>(in_w * scale_w);
      }
    }
    PADDLE_ENFORCE_GT(out_h, 0,
                      platform::errors::InvalidArgument(
                          "out_h in Attr(out_shape) of Op(interpolate) "
                          "should be greater than 0."));
    PADDLE_ENFORCE_GT(out_w, 0,
                      platform::errors::InvalidArgument(
                          "out_w in Attr(out_shape) of Op(interpolate) "
                          "should be greater than 0."));
    framework::DDim dim_out;
    if (data_layout == DataLayout::kNCHW) {
      dim_out = {n, c, out_h, out_w};
    } else {
      dim_out = {n, out_h, out_w, c};
    }
    output->mutable_data<T>(dim_out, ctx.GetPlace());

    if (in_h == out_h && in_w == out_w) {
      framework::TensorCopy(*input, ctx.GetPlace(), output);
      return;
    }

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();

    // To-do(qili93): need to support bilineare, try ResizeD
    // Add bilineare by zhulei
    if ("nearest" == interp_method) {
      NpuOpRunner runner;
      runner.SetType("ResizeNearestNeighborV2")
          .AddInput(*input)
          .AddInput(std::vector<int32_t>{out_h, out_w})
          .AddOutput(*output)
          .AddAttr("align_corners", align_corners)
          .AddAttr("half_pixel_centers", false);
      runner.Run(stream);
    } else if ("bilinear" == interp_method) {
      int align_mode = ctx.Attr<int>("align_mode");
      BilinearFwdNpu<T>(ctx, input, output, scale_h, scale_w, align_corners,
                        align_mode, data_layout);
    }
  }
};

template <typename DeviceContext, typename T>
class InterpolateV2NPUGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* input = ctx.Input<Tensor>("X");
    auto* output_grad = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* input_grad = ctx.Output<Tensor>(framework::GradVarName("X"));

    const std::string data_layout_str = ctx.Attr<std::string>("data_layout");
    const DataLayout data_layout =
        framework::StringToDataLayout(data_layout_str);
    int n, c, in_d, in_h, in_w;
    ExtractNCDWH(input->dims(), data_layout, &n, &c, &in_d, &in_h, &in_w);

    auto interp_method = ctx.Attr<std::string>("interp_method");
    bool align_corners = ctx.Attr<bool>("align_corners");

    // To-do(qili93): need to support align_corners = true case, try ReSizeD
    PADDLE_ENFORCE_EQ(
        align_corners, false,
        platform::errors::InvalidArgument(
            "NPU Interpolate Kernel has diff when align_corners is true."));

    int out_h = ctx.Attr<int>("out_h");
    int out_w = ctx.Attr<int>("out_w");
    float scale_h = -1;
    float scale_w = -1;

    // Priority: SizeTensor > OutSize > Scale > scale > out_h & out_w
    auto list_new_size_tensor = ctx.MultiInput<framework::Tensor>("SizeTensor");
    if (list_new_size_tensor.size() > 0) {
      std::vector<int32_t> output_h(1);
      std::vector<int32_t> output_w(1);
      auto dev_ctx =
          platform::DeviceContextPool::Instance().Get(ctx.GetPlace());
      framework::TensorToVector(*list_new_size_tensor[0], *dev_ctx, &output_h);
      framework::TensorToVector(*list_new_size_tensor[1], *dev_ctx, &output_w);
      out_h = output_h[0];
      out_w = output_w[0];
    } else if (ctx.HasInput("OutSize")) {
      auto out_size = ctx.Input<Tensor>("OutSize");
      auto out_size_data = get_new_data_from_tensor<int>(out_size);
      out_h = out_size_data[0];
      out_w = out_size_data[1];
    } else {
      auto scale_tensor = ctx.Input<Tensor>("Scale");
      auto scale = ctx.Attr<std::vector<float>>("scale");
      if (scale_tensor != nullptr) {
        auto scale_data = get_new_data_from_tensor<float>(scale_tensor);
        if (scale_data.size() > 1) {
          scale_h = scale_data[0];
          scale_w = scale_data[1];
        } else {
          scale_w = scale_data[0];
          scale_h = scale_data[0];
        }
        PADDLE_ENFORCE_EQ(
            scale_w > 0, true,
            platform::errors::InvalidArgument(
                "The scale_w in input 'Scale' Tensor of Operator(interpolate) "
                "should be greater than 0, but received value is %d.",
                scale_w));
        PADDLE_ENFORCE_EQ(
            scale_h > 0, true,
            platform::errors::InvalidArgument(
                "The scale_h in input 'Scale' Tensor of Operator(interpolate) "
                "should be greater than 0, but received value is %d.",
                scale_h));
      } else {
        if (scale.size() > 1) {
          scale_h = scale[0];
          scale_w = scale[1];
          PADDLE_ENFORCE_EQ(
              scale_w > 0, true,
              platform::errors::InvalidArgument(
                  "The scale_w in Attr(scale) of Operator(interpolate) "
                  "should be greater than 0, but received value is %d.",
                  scale_w));
          PADDLE_ENFORCE_EQ(
              scale_h > 0, true,
              platform::errors::InvalidArgument(
                  "The scale_h in Attr(scale) of Operator(interpolate) "
                  "should be greater than 0, but received value is %d.",
                  scale_h));
        }
      }
      if (scale_h > 0. && scale_w > 0.) {
        out_h = static_cast<int>(in_h * scale_h);
        out_w = static_cast<int>(in_w * scale_w);
      }
    }

    framework::DDim dim_grad;
    if (data_layout == DataLayout::kNCHW) {
      dim_grad = {n, c, in_h, in_w};
    } else {
      dim_grad = {n, in_h, in_w, c};
    }

    input_grad->mutable_data<T>(dim_grad, ctx.GetPlace());

    if (in_h == out_h && in_w == out_w) {
      framework::TensorCopy(*output_grad, ctx.GetPlace(), input_grad);
      return;
    }

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();

    // To-do(qili93): need to support bilineare, try ResizeGradD
    if ("nearest" == interp_method) {
      NpuOpRunner runner;
      runner.SetType("ResizeNearestNeighborV2Grad")
          .AddInput(*output_grad)
          .AddInput(std::vector<int32_t>{in_h, in_w})
          .AddOutput(*input_grad)
          .AddAttr("align_corners", align_corners)
          .AddAttr("half_pixel_centers", false);
      runner.Run(stream);
    } else if ("bilinear" == interp_method) {
      int align_mode = ctx.Attr<int>("align_mode");
      BilinearBwdNpu<T>(ctx, output_grad, input_grad, scale_h, scale_w,
                        align_corners, align_mode, data_layout);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_NPU_KERNEL(
    nearest_interp_v2,
    ops::InterpolateV2NPUKernel<plat::NPUDeviceContext, float>,
    ops::InterpolateV2NPUKernel<plat::NPUDeviceContext, plat::float16>);

REGISTER_OP_NPU_KERNEL(
    nearest_interp_v2_grad,
    ops::InterpolateV2NPUGradKernel<plat::NPUDeviceContext, float>,
    ops::InterpolateV2NPUGradKernel<plat::NPUDeviceContext, plat::float16>);

REGISTER_OP_NPU_KERNEL(
    bilinear_interp_v2,
    ops::InterpolateV2NPUKernel<plat::NPUDeviceContext, float>,
    ops::InterpolateV2NPUKernel<plat::NPUDeviceContext, plat::float16>);

REGISTER_OP_NPU_KERNEL(
    bilinear_interp_v2_grad,
    ops::InterpolateV2NPUGradKernel<plat::NPUDeviceContext, float>,
    ops::InterpolateV2NPUGradKernel<plat::NPUDeviceContext, plat::float16>);
