/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the Licnse. */

#include "paddle/fluid/operators/batch_norm_op.h"
#include "paddle/fluid/operators/npu_op_runner.h"

#if defined(PADDLE_WITH_ASCEND_CL)
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/hccl_helper.h"
#endif

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename T>
std::string outputVector(const std::vector<T> vec) {
  std::ostringstream oss;
  // for (auto ele : vec) oss << ele << ' ';
  for (size_t i = 0; i < vec.size() && i < 10; ++i) {
    oss << vec[i] << ' ';
  }
  return oss.str();
}
template <typename T>
void PrintTensor(const framework::Tensor &src,
                 const framework::ExecutionContext &ctx) {
  std::vector<T> vec(src.numel());
  TensorToVector(src, ctx.device_context(), &vec);
  LOG(WARNING) << "vec: " << outputVector<T>(vec);
}

template <typename T>
void training_or_inference(
    const framework::ExecutionContext &ctx, const aclrtStream &stream,
    const platform::Place &place, const DataLayout &layout,
    const bool &test_mode, const int &N, const int &C, const int &H,
    const int &W, const float epsilon, const float &momentum, const Tensor *x,
    const Tensor *common_mean, const Tensor *common_var, const Tensor *scale,
    const Tensor *bias, const Tensor *mean, const Tensor *variance,
    Tensor *mean_out, Tensor *variance_out, Tensor *saved_mean,
    Tensor *saved_variance, Tensor *y) {
  std::vector<int> axes;
  if (layout == framework::DataLayout::kNCHW) {
    axes = {0, 2, 3};
  } else if (layout == framework::DataLayout::kNHWC) {
    axes = {0, 1, 2};
  }

  std::vector<int> multiples;
  if (layout == framework::DataLayout::kNCHW)
    multiples = {N, 1, H, W};
  else if (layout == framework::DataLayout::kNHWC)
    multiples = {N, H, W, 1};

  Tensor mean_tile_1;
  {
    mean_tile_1.Resize({C});
    mean_tile_1.mutable_data<float>(place);

    TensorCopySync(*common_mean, place, &mean_tile_1);
    LOG(WARNING) << "mean_tile_1: ";
    PrintTensor<float>(mean_tile_1, ctx);

    if (layout == framework::DataLayout::kNCHW)
      mean_tile_1.Resize({1, C, 1, 1});
    else if (layout == framework::DataLayout::kNHWC)
      mean_tile_1.Resize({1, 1, 1, C});
    LOG(WARNING) << "mean_tile_1: ";
    PrintTensor<float>(mean_tile_1, ctx);
  }

  Tensor mean_tile;
  {
    framework::NPUAttributeMap attr_input = {{"multiples", multiples}};

    mean_tile.Resize(x->dims());
    mean_tile.mutable_data<float>(place);

    const auto &runner =
        NpuOpRunner("TileD", {mean_tile_1}, {mean_tile}, attr_input);
    runner.Run(stream);

    LOG(WARNING) << "mean_tile: ";
    PrintTensor<float>(mean_tile, ctx);
  }

  Tensor var_tile_1;
  {
    var_tile_1.Resize({C});
    var_tile_1.mutable_data<float>(place);

    TensorCopySync(*common_var, place, &var_tile_1);
    LOG(WARNING) << "var_tile_1: ";
    PrintTensor<float>(var_tile_1, ctx);

    if (layout == framework::DataLayout::kNCHW)
      var_tile_1.Resize({1, C, 1, 1});
    else if (layout == framework::DataLayout::kNHWC)
      var_tile_1.Resize({1, 1, 1, C});
    LOG(WARNING) << "var_tile_1: ";
    PrintTensor<float>(var_tile_1, ctx);
  }

  Tensor var_tile;
  {
    framework::NPUAttributeMap attr_input = {{"multiples", multiples}};

    var_tile.Resize(x->dims());
    var_tile.mutable_data<float>(place);

    const auto &runner =
        NpuOpRunner("TileD", {var_tile_1}, {var_tile}, attr_input);
    runner.Run(stream);

    LOG(WARNING) << "var_tile: ";
    PrintTensor<float>(var_tile, ctx);
  }

  Tensor var_tile_add_epsilon;
  {
    framework::NPUAttributeMap attr_input = {{"value", epsilon}};

    var_tile_add_epsilon.Resize(x->dims());
    var_tile_add_epsilon.mutable_data<float>(place);

    const auto &runner =
        NpuOpRunner("Adds", {var_tile}, {var_tile_add_epsilon}, attr_input);
    runner.Run(stream);

    LOG(WARNING) << "var_tile_add_epsilon: ";
    PrintTensor<float>(var_tile_add_epsilon, ctx);
  }

  Tensor var_tile_add_epsilon_sqrt;
  {
    var_tile_add_epsilon_sqrt.Resize(x->dims());
    var_tile_add_epsilon_sqrt.mutable_data<float>(place);

    const auto &runner = NpuOpRunner("Sqrt", {var_tile_add_epsilon},
                                     {var_tile_add_epsilon_sqrt}, {});
    runner.Run(stream);

    LOG(WARNING) << "var_tile_add_epsilon_sqrt: ";
    PrintTensor<float>(var_tile_add_epsilon_sqrt, ctx);
  }

  Tensor x_sub_mean;
  {
    LOG(WARNING) << "x: ";
    PrintTensor<T>(*x, ctx);

    if (x->type() == framework::proto::VarType::FP16) {
      Tensor x_tmp;
      {
        auto dst_dtype = ConvertToNpuDtype(framework::proto::VarType::FP32);
        framework::NPUAttributeMap attr_input = {
            {"dst_type", static_cast<int>(dst_dtype)}};

        x_tmp.Resize(x->dims());
        x_tmp.mutable_data<float>(place);

        const auto &runner = NpuOpRunner("Cast", {*x}, {x_tmp}, attr_input);
        runner.Run(stream);

        LOG(WARNING) << "x_tmp: ";
        PrintTensor<float>(x_tmp, ctx);
      }

      {
        x_sub_mean.Resize(x->dims());
        x_sub_mean.mutable_data<float>(place);

        const auto &runner =
            NpuOpRunner("Sub", {x_tmp, mean_tile}, {x_sub_mean}, {});
        runner.Run(stream);

        LOG(WARNING) << "x_sub_mean: ";
        PrintTensor<float>(x_sub_mean, ctx);
      }
    } else {
      x_sub_mean.Resize(x->dims());
      x_sub_mean.mutable_data<float>(place);

      const auto &runner =
          NpuOpRunner("Sub", {*x, mean_tile}, {x_sub_mean}, {});
      runner.Run(stream);

      LOG(WARNING) << "x_sub_mean: ";
      PrintTensor<float>(x_sub_mean, ctx);
    }
  }

  Tensor normalized;
  {
    normalized.Resize(x->dims());
    normalized.mutable_data<float>(place);

    const auto &runner = NpuOpRunner(
        "Div", {x_sub_mean, var_tile_add_epsilon_sqrt}, {normalized}, {});
    runner.Run(stream);

    LOG(WARNING) << "normalized: ";
    PrintTensor<float>(normalized, ctx);
  }

  Tensor scale_tile_1;
  {
    scale_tile_1.Resize({C});
    scale_tile_1.mutable_data<float>(place);

    TensorCopySync(*scale, place, &scale_tile_1);
    LOG(WARNING) << "scale_tile_1: ";
    PrintTensor<float>(scale_tile_1, ctx);

    if (layout == framework::DataLayout::kNCHW)
      scale_tile_1.Resize({1, C, 1, 1});
    else if (layout == framework::DataLayout::kNHWC)
      scale_tile_1.Resize({1, 1, 1, C});
    LOG(WARNING) << "scale_tile_1: ";
    PrintTensor<float>(scale_tile_1, ctx);
  }

  Tensor scale_tile;
  {
    framework::NPUAttributeMap attr_input = {{"multiples", multiples}};

    scale_tile.Resize(x->dims());
    scale_tile.mutable_data<float>(place);

    const auto &runner =
        NpuOpRunner("TileD", {scale_tile_1}, {scale_tile}, attr_input);
    runner.Run(stream);

    LOG(WARNING) << "scale_tile: ";
    PrintTensor<float>(scale_tile, ctx);
  }

  Tensor normalized_mul_scale;
  {
    normalized_mul_scale.Resize(x->dims());
    normalized_mul_scale.mutable_data<float>(place);

    const auto &runner = NpuOpRunner("Mul", {normalized, scale_tile},
                                     {normalized_mul_scale}, {});
    runner.Run(stream);

    LOG(WARNING) << "normalized_mul_scale: ";
    PrintTensor<float>(normalized_mul_scale, ctx);
  }

  Tensor bias_tile_1;
  {
    bias_tile_1.Resize({C});
    bias_tile_1.mutable_data<float>(place);

    TensorCopySync(*bias, place, &bias_tile_1);
    LOG(WARNING) << "bias_tile_1: ";
    PrintTensor<float>(bias_tile_1, ctx);

    if (layout == framework::DataLayout::kNCHW)
      bias_tile_1.Resize({1, C, 1, 1});
    else if (layout == framework::DataLayout::kNHWC)
      bias_tile_1.Resize({1, 1, 1, C});
    LOG(WARNING) << "bias_tile_1: ";
    PrintTensor<float>(bias_tile_1, ctx);
  }

  Tensor bias_tile;
  {
    framework::NPUAttributeMap attr_input = {{"multiples", multiples}};

    bias_tile.Resize(x->dims());
    bias_tile.mutable_data<float>(place);

    const auto &runner =
        NpuOpRunner("TileD", {bias_tile_1}, {bias_tile}, attr_input);
    runner.Run(stream);

    LOG(WARNING) << "bias_tile: ";
    PrintTensor<float>(bias_tile, ctx);
  }

  // at last, we get y
  {
    if (x->type() == framework::proto::VarType::FP16) {
      Tensor y_tmp;
      {
        y_tmp.Resize(y->dims());
        y_tmp.mutable_data<float>(place);

        const auto &runner =
            NpuOpRunner("Add", {normalized_mul_scale, bias_tile}, {y_tmp}, {});
        runner.Run(stream);

        LOG(WARNING) << "y_tmp: ";
        PrintTensor<float>(y_tmp, ctx);
      }

      {
        auto dst_dtype = ConvertToNpuDtype(framework::proto::VarType::FP16);
        framework::NPUAttributeMap attr_input = {
            {"dst_type", static_cast<int>(dst_dtype)}};

        y->mutable_data<T>(place);

        const auto &runner = NpuOpRunner("Cast", {y_tmp}, {*y}, attr_input);
        runner.Run(stream);

        LOG(WARNING) << "y: ";
        PrintTensor<T>(*y, ctx);
      }

    } else {
      y->mutable_data<T>(place);

      const auto &runner =
          NpuOpRunner("Add", {normalized_mul_scale, bias_tile}, {*y}, {});
      runner.Run(stream);

      LOG(WARNING) << "y: ";
      PrintTensor<T>(*y, ctx);
    }
  }

  if (!test_mode) {
    Tensor ones;
    {
      ones.Resize({C});
      ones.mutable_data<float>(place);

      FillNpuTensorWithConstant<float>(&ones, 1);

      // Or
      // const auto &runner =
      //     NpuOpRunner("OnesLike", {*variance}, {ones}, {});
      // runner.Run(stream);

      LOG(WARNING) << "ones: ";
      PrintTensor<float>(ones, ctx);
    }

    // cacl mean_out
    {
      Tensor momentum_mul_mean;
      {
        framework::NPUAttributeMap attr_input = {{"value", momentum}};

        momentum_mul_mean.Resize({C});
        momentum_mul_mean.mutable_data<float>(place);

        const auto &runner = NpuOpRunner("Muls", {*common_mean},
                                         {momentum_mul_mean}, attr_input);
        runner.Run(stream);

        LOG(WARNING) << "momentum_mul_mean: ";
        PrintTensor<float>(momentum_mul_mean, ctx);
      }

      Tensor saved_mean_mul_1_sub_momentum;
      {
        framework::NPUAttributeMap attr_input = {{"value", 1 - momentum}};

        saved_mean_mul_1_sub_momentum.Resize({C});
        saved_mean_mul_1_sub_momentum.mutable_data<float>(place);

        const auto &runner =
            NpuOpRunner("Muls", {*common_mean}, {saved_mean_mul_1_sub_momentum},
                        attr_input);
        runner.Run(stream);

        LOG(WARNING) << "saved_mean_mul_1_sub_momentum: ";
        PrintTensor<float>(saved_mean_mul_1_sub_momentum, ctx);
      }

      mean_out->mutable_data<float>(place);

      const auto &runner =
          NpuOpRunner("Add", {saved_mean_mul_1_sub_momentum, momentum_mul_mean},
                      {*mean_out}, {});
      runner.Run(stream);

      LOG(WARNING) << "mean_out: ";
      PrintTensor<float>(*mean_out, ctx);
    }

    // cacl variance_out
    {
      Tensor momentum_mul_var;
      {
        framework::NPUAttributeMap attr_input = {{"value", momentum}};

        momentum_mul_var.Resize({C});
        momentum_mul_var.mutable_data<float>(place);

        const auto &runner =
            NpuOpRunner("Muls", {*common_var}, {momentum_mul_var}, attr_input);
        runner.Run(stream);

        LOG(WARNING) << "momentum_mul_var: ";
        PrintTensor<float>(momentum_mul_var, ctx);
      }

      Tensor var_ref_mul_1_sub_momentum;
      {
        framework::NPUAttributeMap attr_input = {{"value", 1 - momentum}};

        var_ref_mul_1_sub_momentum.Resize({C});
        var_ref_mul_1_sub_momentum.mutable_data<float>(place);

        const auto &runner = NpuOpRunner(
            "Muls", {*common_var}, {var_ref_mul_1_sub_momentum}, attr_input);
        runner.Run(stream);

        LOG(WARNING) << "var_ref_mul_1_sub_momentum: ";
        PrintTensor<float>(var_ref_mul_1_sub_momentum, ctx);
      }

      variance_out->mutable_data<float>(place);

      const auto &runner =
          NpuOpRunner("Add", {var_ref_mul_1_sub_momentum, momentum_mul_var},
                      {*variance_out}, {});
      runner.Run(stream);

      LOG(WARNING) << "variance_out: ";
      PrintTensor<float>(*variance_out, ctx);
    }

    // cacl saved_variance
    {
      Tensor var_ref_add_epsilon;
      {
        framework::NPUAttributeMap attr_input = {{"value", epsilon}};

        var_ref_add_epsilon.Resize({C});
        var_ref_add_epsilon.mutable_data<float>(place);

        const auto &runner = NpuOpRunner("Adds", {*common_var},
                                         {var_ref_add_epsilon}, attr_input);
        runner.Run(stream);

        LOG(WARNING) << "var_ref_add_epsilon: ";
        PrintTensor<float>(var_ref_add_epsilon, ctx);
      }

      Tensor var_ref_add_epsilon_sqrt;
      {
        var_ref_add_epsilon_sqrt.Resize({C});
        var_ref_add_epsilon_sqrt.mutable_data<float>(place);

        const auto &runner = NpuOpRunner("Sqrt", {var_ref_add_epsilon},
                                         {var_ref_add_epsilon_sqrt}, {});
        runner.Run(stream);

        LOG(WARNING) << "var_ref_add_epsilon_sqrt: ";
        PrintTensor<float>(var_ref_add_epsilon_sqrt, ctx);
      }

      saved_variance->mutable_data<float>(place);

      const auto &runner = NpuOpRunner("Div", {ones, var_ref_add_epsilon_sqrt},
                                       {*saved_variance}, {});
      runner.Run(stream);

      LOG(WARNING) << "saved_variance: ";
      PrintTensor<float>(*saved_variance, ctx);
    }
  }
}

template <typename DeviceContext, typename T>
class SyncBatchNormNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    LOG(WARNING) << "SyncBatchNormNPUKernel";
    LOG(WARNING) << "op type: " << ctx.Type();

    // double epsilon = static_cast<double>(ctx.Attr<float>("epsilon"));
    const float epsilon = ctx.Attr<float>("epsilon");
    float momentum = ctx.Attr<float>("momentum");
    const bool is_test = ctx.Attr<bool>("is_test");
    const std::string layout_str = ctx.Attr<std::string>("data_layout");
    const DataLayout layout = framework::StringToDataLayout(layout_str);
    const bool use_global_stats = ctx.Attr<bool>("use_global_stats");
    const bool trainable_stats = ctx.Attr<bool>("trainable_statistics");

    LOG(WARNING) << "epsilon: " << epsilon;
    LOG(WARNING) << "momentum: " << momentum;
    LOG(WARNING) << "is_test: " << is_test;
    LOG(WARNING) << "layout_str: " << layout_str;
    LOG(WARNING) << "layout: " << layout;
    LOG(WARNING) << "use_global_stats: " << use_global_stats;
    LOG(WARNING) << "trainable_stats: " << trainable_stats;

    PADDLE_ENFORCE_EQ(use_global_stats, false,
                      platform::errors::InvalidArgument(
                          "sync_batch_norm doesn't support "
                          "to set use_global_stats True. Please use batch_norm "
                          "in this case."));

    const auto *x = ctx.Input<Tensor>("X");
    auto *y = ctx.Output<Tensor>("Y");

    const auto *scale = ctx.Input<Tensor>("Scale");
    const auto *bias = ctx.Input<Tensor>("Bias");

    const auto *mean = ctx.Input<Tensor>("Mean");
    const auto *variance = ctx.Input<Tensor>("Variance");

    // moving mean/variance
    auto *mean_out = ctx.Output<Tensor>("MeanOut");
    auto *variance_out = ctx.Output<Tensor>("VarianceOut");

    auto *saved_mean = ctx.Output<Tensor>("SavedMean");
    auto *saved_variance = ctx.Output<Tensor>("SavedVariance");

    LOG(WARNING) << "mean: " << mean;
    LOG(WARNING) << "mean_out: " << mean_out;
    LOG(WARNING) << "variance: " << variance;
    LOG(WARNING) << "variance_out: " << variance_out;

    LOG(WARNING) << "x dims: " << x->dims();
    LOG(WARNING) << "x numel: " << x->numel();
    LOG(WARNING) << "y dims: " << y->dims();
    LOG(WARNING) << "y numel: " << y->numel();
    LOG(WARNING) << "scale dims: " << scale->dims();
    LOG(WARNING) << "scale numel: " << scale->numel();
    LOG(WARNING) << "bias dims: " << bias->dims();
    LOG(WARNING) << "bias numel: " << bias->numel();
    LOG(WARNING) << "mean dims: " << mean->dims();
    LOG(WARNING) << "mean numel: " << mean->numel();
    LOG(WARNING) << "variance dims: " << variance->dims();
    LOG(WARNING) << "variance numel: " << variance->numel();
    LOG(WARNING) << "mean_out dims: " << mean_out->dims();
    LOG(WARNING) << "mean_out numel: " << mean_out->numel();
    LOG(WARNING) << "variance_out dims: " << variance_out->dims();
    LOG(WARNING) << "variance_out numel: " << variance_out->numel();
    LOG(WARNING) << "saved_mean dims: " << saved_mean->dims();
    LOG(WARNING) << "saved_mean numel: " << saved_mean->numel();
    LOG(WARNING) << "saved_variance dims: " << saved_variance->dims();
    LOG(WARNING) << "saved_variance numel: " << saved_variance->numel();

    LOG(WARNING) << "Input Tensor | x: ";
    PrintTensor<float>(*x, ctx);
    LOG(WARNING) << "Input Tensor | scale: ";
    PrintTensor<float>(*scale, ctx);
    LOG(WARNING) << "Input Tensor | bias: ";
    PrintTensor<float>(*bias, ctx);
    LOG(WARNING) << "Input Tensor | mean: ";
    PrintTensor<float>(*mean, ctx);
    LOG(WARNING) << "Input Tensor | variance: ";
    PrintTensor<float>(*variance, ctx);

    const auto &x_dims = x->dims();
    PADDLE_ENFORCE_GE(x_dims.size(), 2,
                      platform::errors::InvalidArgument(
                          "The Input dim size should be larger than 1."));
    PADDLE_ENFORCE_LE(x_dims.size(), 5,
                      platform::errors::InvalidArgument(
                          "The Input dim size should be less than 6."));

    int N, C, H, W, D;
    ExtractNCWHD(x_dims, layout, &N, &C, &H, &W, &D);

    LOG(WARNING) << "N: " << N;
    LOG(WARNING) << "C: " << C;
    LOG(WARNING) << "H: " << H;
    LOG(WARNING) << "W: " << W;
    LOG(WARNING) << "D: " << D;

    int x_numel = x->numel();

    // const T *x_data = x->data<T>();
    // const auto *scale_data = scale->data<float>();
    // const auto *bias_data = bias->data<float>();
    auto *mean_data = mean->data<float>();
    auto *mean_out_data = mean_out->data<float>();

    LOG(WARNING) << "mean_data: " << mean_data;
    LOG(WARNING) << "mean_out_data: " << mean_out_data;

    auto place = ctx.GetPlace();

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();

    std::vector<int> axes;
    if (layout == framework::DataLayout::kNCHW) {
      axes = {0, 2, 3};
    } else if (layout == framework::DataLayout::kNHWC) {
      axes = {0, 1, 2};
    }

    bool test_mode = is_test && (!trainable_stats);
    if (test_mode) {  // inference
      LOG(WARNING) << "inference";

      // cacl saved_mean
      saved_mean->mutable_data<float>(place);
      // saved_mean->ShareDataWith(*mean);
      TensorCopySync(*mean, place, saved_mean);
      LOG(WARNING) << "saved_mean: ";
      PrintTensor<float>(*saved_mean, ctx);

      // cacl saved_variance
      saved_variance->mutable_data<float>(place);
      // saved_variance->ShareDataWith(*variance);
      TensorCopySync(*variance, place, saved_variance);
      LOG(WARNING) << "saved_variance: ";
      PrintTensor<float>(*saved_variance, ctx);

      // cacl y
      training_or_inference<T>(ctx, stream, place, layout, test_mode, N, C, H,
                               W, epsilon, momentum, x, mean, variance, scale,
                               bias, mean, variance, NULL, NULL, NULL, NULL, y);

    } else {  // training
      LOG(WARNING) << "training";

      // if MomentumTensor is set, use MomentumTensor value, momentum
      // is only used in this training branch
      if (ctx.HasInput("MomentumTensor")) {
        const auto *mom_tensor = ctx.Input<Tensor>("MomentumTensor");
        Tensor mom_cpu;
        TensorCopySync(*mom_tensor, platform::CPUPlace(), &mom_cpu);
        momentum = mom_cpu.data<float>()[0];

        LOG(WARNING) << "momentum: " << momentum;
      }

      // cacl saved_mean and var_ref
      Tensor var_ref;
      {
        var_ref.Resize({C});
        var_ref.mutable_data<float>(place);
      }
      {
        // cacl saved_mean
        {
          LOG(WARNING) << "x: ";
          PrintTensor<T>(*x, ctx);

          if (x->type() == framework::proto::VarType::FP16) {
            Tensor x_tmp;
            {
              auto dst_dtype =
                  ConvertToNpuDtype(framework::proto::VarType::FP32);
              framework::NPUAttributeMap attr_input = {
                  {"dst_type", static_cast<int>(dst_dtype)}};

              x_tmp.Resize(x->dims());
              x_tmp.mutable_data<float>(place);

              const auto &runner =
                  NpuOpRunner("Cast", {*x}, {x_tmp}, attr_input);
              runner.Run(stream);

              LOG(WARNING) << "x_tmp: ";
              PrintTensor<float>(x_tmp, ctx);
            }

            {
              saved_mean->mutable_data<float>(place);

              framework::NPUAttributeMap attr_input = {{"keep_dims", false},
                                                       {"axes", axes}};

              const auto &runner = NpuOpRunner("ReduceMeanD", {x_tmp},
                                               {*saved_mean}, attr_input);

              runner.Run(stream);

              LOG(WARNING) << "saved_mean: ";
              PrintTensor<float>(*saved_mean, ctx);
            }
          } else {
            saved_mean->mutable_data<float>(place);

            framework::NPUAttributeMap attr_input = {{"keep_dims", false},
                                                     {"axes", axes}};

            const auto &runner =
                NpuOpRunner("ReduceMeanD", {*x}, {*saved_mean}, attr_input);

            runner.Run(stream);

            LOG(WARNING) << "saved_mean: ";
            PrintTensor<float>(*saved_mean, ctx);
          }
        }

        // cacl var_ref
        {
          Tensor x_square;
          // cacl x_square
          {
            x_square.Resize(x->dims());
            x_square.mutable_data<float>(place);

            if (x->type() == framework::proto::VarType::FP16) {
              Tensor x_tmp;
              {
                auto dst_dtype =
                    ConvertToNpuDtype(framework::proto::VarType::FP32);
                framework::NPUAttributeMap attr_input = {
                    {"dst_type", static_cast<int>(dst_dtype)}};

                x_tmp.Resize(x->dims());
                x_tmp.mutable_data<float>(place);

                const auto &runner =
                    NpuOpRunner("Cast", {*x}, {x_tmp}, attr_input);
                runner.Run(stream);

                LOG(WARNING) << "x_tmp: ";
                PrintTensor<float>(x_tmp, ctx);
              }

              {
                const auto &runner =
                    NpuOpRunner("Square", {x_tmp}, {x_square}, {});
                runner.Run(stream);

                LOG(WARNING) << "x_square: ";
                PrintTensor<float>(x_square, ctx);
              }
            } else {
              const auto &runner = NpuOpRunner("Square", {*x}, {x_square}, {});
              runner.Run(stream);

              LOG(WARNING) << "x_square: ";
              PrintTensor<float>(x_square, ctx);
            }
          }

          Tensor x_square_sum;
          // cacl x_square_sum
          {
            framework::NPUAttributeMap attr_input = {{"keep_dims", false},
                                                     {"axes", axes}};

            x_square_sum.Resize({C});
            x_square_sum.mutable_data<float>(place);

            const auto &runner = NpuOpRunner("ReduceSumD", {x_square},
                                             {x_square_sum}, attr_input);
            runner.Run(stream);

            LOG(WARNING) << "x_square_sum: ";
            PrintTensor<float>(x_square_sum, ctx);
          }

          Tensor x_square_sum_mean;
          // cacl x_square_sum_mean
          {
            framework::NPUAttributeMap attr_input = {
                {"value", 1.0f * C / x_numel}};

            x_square_sum_mean.Resize({C});
            x_square_sum_mean.mutable_data<float>(place);

            const auto &runner = NpuOpRunner("Muls", {x_square_sum},
                                             {x_square_sum_mean}, attr_input);
            runner.Run(stream);

            LOG(WARNING) << "x_square_sum_mean: ";
            PrintTensor<float>(x_square_sum_mean, ctx);
          }

          Tensor mean_square;
          // cacl mean_square
          {
            mean_square.Resize(mean->dims());
            mean_square.mutable_data<float>(place);

            const auto &runner =
                NpuOpRunner("Square", {*mean}, {mean_square}, {});
            runner.Run(stream);

            LOG(WARNING) << "mean_square: ";
            PrintTensor<float>(mean_square, ctx);
          }

          // cacl var_ref
          {
            const auto &runner = NpuOpRunner(
                "Sub", {x_square_sum_mean, mean_square}, {var_ref}, {});
            runner.Run(stream);

            LOG(WARNING) << "var_ref: ";
            PrintTensor<float>(var_ref, ctx);
          }
        }
      }

      int count = platform::GetSelectedNPUDevices().size();
      LOG(WARNING) << "count: " << count;

      if (count >= 1) {
        LOG(WARNING) << "before hccl | saved_mean: ";
        PrintTensor<float>(*saved_mean, ctx);

        LOG(WARNING) << "before hccl | var_ref: ";
        PrintTensor<float>(var_ref, ctx);

        // auto &dev_ctx = reinterpret_cast<const platform::NPUDeviceContext &>(
        //     ctx.device_context());
        // auto *comm = dev_ctx.hccl_comm();
        // LOG(WARNING) << "comm: " << comm;

        auto comm = paddle::platform::HCCLCommContext::Instance().Get(1, place);
        LOG(WARNING) << "comm: " << comm;

        float device_counts = 1.0;
        if (comm) {
          int dtype = platform::ToHCCLDataType(mean_out->type());

          // In-place operation
          LOG(WARNING) << "before hccl | device_counts: " << device_counts;
          PADDLE_ENFORCE_NPU_SUCCESS(platform::dynload::HcclAllReduce(
              &device_counts, &device_counts, 1,
              static_cast<HcclDataType>(dtype), HCCL_REDUCE_SUM, comm->comm(),
              stream));
          LOG(WARNING) << "after hccl | device_counts: " << device_counts;

          // In-place operation
          PADDLE_ENFORCE_NPU_SUCCESS(platform::dynload::HcclAllReduce(
              saved_mean->data<float>(), saved_mean->data<float>(), C,
              static_cast<HcclDataType>(dtype), HCCL_REDUCE_SUM, comm->comm(),
              stream));
          PADDLE_ENFORCE_NPU_SUCCESS(platform::dynload::HcclAllReduce(
              var_ref.data<float>(), var_ref.data<float>(), C,
              static_cast<HcclDataType>(dtype), HCCL_REDUCE_SUM, comm->comm(),
              stream));
        }

        LOG(WARNING) << "after hccl | saved_mean: ";
        PrintTensor<float>(*saved_mean, ctx);

        LOG(WARNING) << "after hccl | var_ref: ";
        PrintTensor<float>(var_ref, ctx);

        // mean saved_mean
        {
          framework::NPUAttributeMap attr_input = {
              {"value", 1.0f / device_counts}};

          const auto &runner =
              NpuOpRunner("Muls", {*saved_mean}, {*saved_mean}, attr_input);
          runner.Run(stream);

          LOG(WARNING) << "after | saved_mean: ";
          PrintTensor<float>(*saved_mean, ctx);
        }

        // mean var_ref
        {
          framework::NPUAttributeMap attr_input = {
              {"value", 1.0f / device_counts}};

          const auto &runner =
              NpuOpRunner("Muls", {var_ref}, {var_ref}, attr_input);
          runner.Run(stream);

          LOG(WARNING) << "after | var_ref: ";
          PrintTensor<float>(var_ref, ctx);
        }
      }

      training_or_inference<T>(ctx, stream, place, layout, test_mode, N, C, H,
                               W, epsilon, momentum, x, saved_mean, &var_ref,
                               scale, bias, mean, variance, mean_out,
                               variance_out, saved_mean, saved_variance, y);
    }

    LOG(WARNING) << "Output Tensor | y: ";
    PrintTensor<float>(*y, ctx);
    LOG(WARNING) << "Output Tensor | mean_out: ";
    PrintTensor<float>(*mean_out, ctx);
    LOG(WARNING) << "Output Tensor | variance_out: ";
    PrintTensor<float>(*variance_out, ctx);
    LOG(WARNING) << "Output Tensor | saved_mean: ";
    PrintTensor<float>(*saved_mean, ctx);
    LOG(WARNING) << "Output Tensor | saved_variance: ";
    PrintTensor<float>(*saved_variance, ctx);
  }
};

template <typename DeviceContext, typename T>
class SyncBatchNormNPUGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    LOG(WARNING) << "SyncBatchNormNPUGradKernel";
    LOG(WARNING) << "op type: " << ctx.Type();

    // double epsilon = static_cast<double>(ctx.Attr<float>("epsilon"));
    float epsilon = ctx.Attr<float>("epsilon");
    const std::string layout_str = ctx.Attr<std::string>("data_layout");
    const DataLayout layout = framework::StringToDataLayout(layout_str);

    LOG(WARNING) << "epsilon: " << epsilon;
    LOG(WARNING) << "layout_str: " << layout_str;
    LOG(WARNING) << "layout: " << layout;

    const auto *d_y = ctx.Input<Tensor>(framework::GradVarName("Y"));
    const auto *scale = ctx.Input<Tensor>("Scale");
    const auto *bias = ctx.Input<Tensor>("Bias");

    // init output
    auto *d_x = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto *d_scale = ctx.Output<Tensor>(framework::GradVarName("Scale"));
    auto *d_bias = ctx.Output<Tensor>(framework::GradVarName("Bias"));

    const auto *saved_mean = ctx.Input<Tensor>("SavedMean");
    const auto *saved_var = ctx.Input<Tensor>("SavedVariance");

    LOG(WARNING) << "saved_mean: ";
    PrintTensor<float>(*saved_mean, ctx);

    // sync_batch_norm with inplace as false will take X as grad input, which
    // is same as cuDNN batch_norm backward calculation, batch_norm
    // with inplace as true only take Y as input and X should be calculate
    // by inverse operation of batch_norm on Y
    const Tensor *x;
    bool is_inplace;
    if (ctx.HasInput("Y")) {
      x = ctx.Input<Tensor>("Y");
      is_inplace = true;
    } else {
      x = ctx.Input<Tensor>("X");
      is_inplace = false;
    }
    LOG(WARNING) << "is_inplace: " << is_inplace;

    LOG(WARNING) << "d_y dims: " << d_y->dims();
    LOG(WARNING) << "d_y numel: " << d_y->numel();
    LOG(WARNING) << "scale dims: " << scale->dims();
    LOG(WARNING) << "scale numel: " << scale->numel();
    LOG(WARNING) << "bias dims: " << bias->dims();
    LOG(WARNING) << "bias numel: " << bias->numel();

    if (d_x) {
      LOG(WARNING) << "d_x dims: " << d_x->dims();
      LOG(WARNING) << "d_x numel: " << d_x->numel();
    }
    if (d_scale) {
      LOG(WARNING) << "d_scale dims: " << d_scale->dims();
      LOG(WARNING) << "d_scale numel: " << d_scale->numel();
    }
    if (d_bias) {
      LOG(WARNING) << "d_bias dims: " << d_bias->dims();
      LOG(WARNING) << "d_bias numel: " << d_bias->numel();
    }

    LOG(WARNING) << "saved_mean dims: " << saved_mean->dims();
    LOG(WARNING) << "saved_mean numel: " << saved_mean->numel();
    LOG(WARNING) << "saved_var dims: " << saved_var->dims();
    LOG(WARNING) << "saved_var numel: " << saved_var->numel();

    LOG(WARNING) << "x dims: " << x->dims();
    LOG(WARNING) << "x numel: " << x->numel();

    int N, C, H, W, D;
    ExtractNCWHD(x->dims(), layout, &N, &C, &H, &W, &D);

    LOG(WARNING) << "N: " << N;
    LOG(WARNING) << "C: " << C;
    LOG(WARNING) << "H: " << H;
    LOG(WARNING) << "W: " << W;
    LOG(WARNING) << "D: " << D;

    int x_numel = x->numel();

    auto place = ctx.GetPlace();

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();

    std::vector<int> axes;
    if (layout == framework::DataLayout::kNCHW) {
      axes = {0, 2, 3};
    } else if (layout == framework::DataLayout::kNHWC) {
      axes = {0, 1, 2};
    }

    std::vector<int> multiples;
    if (layout == framework::DataLayout::kNCHW)
      multiples = {N, 1, H, W};
    else if (layout == framework::DataLayout::kNHWC)
      multiples = {N, H, W, 1};

    // cacl saved_mean and var_ref
    Tensor saved_mean_tmp;
    {
      saved_mean_tmp.Resize({C});
      saved_mean_tmp.mutable_data<float>(place);
    }
    Tensor var_ref;
    {
      var_ref.Resize({C});
      var_ref.mutable_data<float>(place);
    }
    {
      // cacl saved_mean_tmp
      {
        framework::NPUAttributeMap attr_input = {{"keep_dims", false},
                                                 {"axes", axes}};

        const auto &runner =
            NpuOpRunner("ReduceMeanD", {*x}, {saved_mean_tmp}, attr_input);

        runner.Run(stream);

        LOG(WARNING) << "saved_mean_tmp: ";
        PrintTensor<float>(saved_mean_tmp, ctx);
      }

      // cacl var_ref
      {
        Tensor x_square;
        // cacl x_square
        {
          x_square.Resize(x->dims());
          x_square.mutable_data<T>(place);

          const auto &runner = NpuOpRunner("Square", {*x}, {x_square}, {});
          runner.Run(stream);

          LOG(WARNING) << "x_square: ";
          PrintTensor<T>(x_square, ctx);
        }

        Tensor x_square_sum;
        // cacl x_square_sum
        {
          framework::NPUAttributeMap attr_input = {{"keep_dims", false},
                                                   {"axes", axes}};

          x_square_sum.Resize({C});
          x_square_sum.mutable_data<T>(place);

          const auto &runner =
              NpuOpRunner("ReduceSumD", {x_square}, {x_square_sum}, attr_input);
          runner.Run(stream);

          LOG(WARNING) << "x_square_sum: ";
          PrintTensor<T>(x_square_sum, ctx);
        }

        Tensor x_square_sum_mean;
        // cacl x_square_sum_mean
        {
          framework::NPUAttributeMap attr_input = {
              {"value", 1.0f * C / x_numel}};

          x_square_sum_mean.Resize({C});
          x_square_sum_mean.mutable_data<T>(place);

          const auto &runner = NpuOpRunner("Muls", {x_square_sum},
                                           {x_square_sum_mean}, attr_input);
          runner.Run(stream);

          LOG(WARNING) << "x_square_sum_mean: ";
          PrintTensor<T>(x_square_sum_mean, ctx);
        }

        Tensor mean_square;
        // cacl mean_square
        {
          mean_square.Resize({C});
          mean_square.mutable_data<T>(place);

          const auto &runner =
              NpuOpRunner("Square", {saved_mean_tmp}, {mean_square}, {});
          runner.Run(stream);

          LOG(WARNING) << "mean_square: ";
          PrintTensor<T>(mean_square, ctx);
        }

        // cacl var_ref
        {
          const auto &runner = NpuOpRunner(
              "Sub", {x_square_sum_mean, mean_square}, {var_ref}, {});
          runner.Run(stream);

          LOG(WARNING) << "var_ref: ";
          PrintTensor<float>(var_ref, ctx);
        }
      }
    }

    Tensor saved_mean_tile_1;
    {
      saved_mean_tile_1.Resize({C});
      saved_mean_tile_1.mutable_data<float>(place);

      TensorCopySync(*saved_mean, place, &saved_mean_tile_1);
      LOG(WARNING) << "saved_mean_tile_1: ";
      PrintTensor<float>(saved_mean_tile_1, ctx);

      if (layout == framework::DataLayout::kNCHW)
        saved_mean_tile_1.Resize({1, C, 1, 1});
      else if (layout == framework::DataLayout::kNHWC)
        saved_mean_tile_1.Resize({1, 1, 1, C});
      LOG(WARNING) << "saved_mean_tile_1: ";
      PrintTensor<float>(saved_mean_tile_1, ctx);
    }

    Tensor saved_mean_tile;
    {
      framework::NPUAttributeMap attr_input = {{"multiples", multiples}};

      saved_mean_tile.Resize(x->dims());
      saved_mean_tile.mutable_data<float>(place);

      const auto &runner = NpuOpRunner("TileD", {saved_mean_tile_1},
                                       {saved_mean_tile}, attr_input);
      runner.Run(stream);

      LOG(WARNING) << "saved_mean_tile: ";
      PrintTensor<float>(saved_mean_tile, ctx);
    }

    Tensor x_sub_saved_mean;
    {
      x_sub_saved_mean.Resize(x->dims());
      x_sub_saved_mean.mutable_data<float>(place);

      const auto &runner =
          NpuOpRunner("Sub", {*x, saved_mean_tile}, {x_sub_saved_mean}, {});
      runner.Run(stream);

      LOG(WARNING) << "x_sub_saved_mean: ";
      PrintTensor<float>(x_sub_saved_mean, ctx);
    }

    Tensor var_ref_tile_1;
    {
      var_ref_tile_1.Resize({C});
      var_ref_tile_1.mutable_data<float>(place);

      TensorCopySync(var_ref, place, &var_ref_tile_1);
      LOG(WARNING) << "var_ref_tile_1: ";
      PrintTensor<float>(var_ref_tile_1, ctx);

      if (layout == framework::DataLayout::kNCHW)
        var_ref_tile_1.Resize({1, C, 1, 1});
      else if (layout == framework::DataLayout::kNHWC)
        var_ref_tile_1.Resize({1, 1, 1, C});
      LOG(WARNING) << "var_ref_tile_1: ";
      PrintTensor<float>(var_ref_tile_1, ctx);
    }

    Tensor var_ref_tile;
    {
      framework::NPUAttributeMap attr_input = {{"multiples", multiples}};

      var_ref_tile.Resize(x->dims());
      var_ref_tile.mutable_data<float>(place);

      const auto &runner =
          NpuOpRunner("TileD", {var_ref_tile_1}, {var_ref_tile}, attr_input);
      runner.Run(stream);

      LOG(WARNING) << "var_ref_tile: ";
      PrintTensor<float>(var_ref_tile, ctx);
    }

    Tensor var_ref_tile_add_epsilon;
    {
      framework::NPUAttributeMap attr_input = {{"value", epsilon}};

      var_ref_tile_add_epsilon.Resize(x->dims());
      var_ref_tile_add_epsilon.mutable_data<float>(place);

      const auto &runner = NpuOpRunner("Adds", {var_ref_tile},
                                       {var_ref_tile_add_epsilon}, attr_input);
      runner.Run(stream);

      LOG(WARNING) << "var_ref_tile_add_epsilon: ";
      PrintTensor<float>(var_ref_tile_add_epsilon, ctx);
    }

    Tensor var_ref_tile_add_epsilon_sqrt;
    {
      var_ref_tile_add_epsilon_sqrt.Resize(x->dims());
      var_ref_tile_add_epsilon_sqrt.mutable_data<float>(place);

      const auto &runner = NpuOpRunner("Sqrt", {var_ref_tile_add_epsilon},
                                       {var_ref_tile_add_epsilon_sqrt}, {});
      runner.Run(stream);

      LOG(WARNING) << "var_ref_tile_add_epsilon_sqrt: ";
      PrintTensor<float>(var_ref_tile_add_epsilon_sqrt, ctx);
    }

    Tensor dy_mul_x_sub_mean;
    {
      dy_mul_x_sub_mean.Resize(x->dims());
      dy_mul_x_sub_mean.mutable_data<float>(place);

      const auto &runner =
          NpuOpRunner("Mul", {*d_y, x_sub_saved_mean}, {dy_mul_x_sub_mean}, {});
      runner.Run(stream);

      LOG(WARNING) << "dy_mul_x_sub_mean: ";
      PrintTensor<float>(dy_mul_x_sub_mean, ctx);
    }

    // cacl d_x
    // d_x = scale * (dy_sub_dy_mean - x_sub_saved_mean * dy_mul_x_sub_mean_mean
    // / var_ref_tile_add_epsilon) / var_ref_tile_add_epsilon_sqrt
    if (d_x) {
      Tensor dy_mean;
      {
        framework::NPUAttributeMap attr_input = {{"keep_dims", false},
                                                 {"axes", axes}};

        dy_mean.Resize({C});
        dy_mean.mutable_data<float>(place);

        const auto &runner =
            NpuOpRunner("ReduceMeanD", {*d_y}, {dy_mean}, attr_input);
        runner.Run(stream);

        LOG(WARNING) << "dy_mean: ";
        PrintTensor<float>(dy_mean, ctx);
      }

      Tensor dy_mean_tile_1;
      {
        dy_mean_tile_1.Resize({C});
        dy_mean_tile_1.mutable_data<float>(place);

        TensorCopySync(dy_mean, place, &dy_mean_tile_1);
        LOG(WARNING) << "dy_mean_tile_1: ";
        PrintTensor<float>(dy_mean_tile_1, ctx);

        if (layout == framework::DataLayout::kNCHW)
          dy_mean_tile_1.Resize({1, C, 1, 1});
        else if (layout == framework::DataLayout::kNHWC)
          dy_mean_tile_1.Resize({1, 1, 1, C});
        LOG(WARNING) << "mean_tile_1: ";
        PrintTensor<float>(dy_mean_tile_1, ctx);
      }

      Tensor dy_mean_tile;
      {
        framework::NPUAttributeMap attr_input = {{"multiples", multiples}};

        dy_mean_tile.Resize(x->dims());
        dy_mean_tile.mutable_data<float>(place);

        const auto &runner =
            NpuOpRunner("TileD", {dy_mean_tile_1}, {dy_mean_tile}, attr_input);
        runner.Run(stream);

        LOG(WARNING) << "dy_mean_tile: ";
        PrintTensor<float>(dy_mean_tile, ctx);
      }

      Tensor dy_sub_dy_mean;
      {
        dy_sub_dy_mean.Resize(x->dims());
        dy_sub_dy_mean.mutable_data<float>(place);

        const auto &runner =
            NpuOpRunner("Sub", {*d_y, dy_mean_tile}, {dy_sub_dy_mean}, {});
        runner.Run(stream);

        LOG(WARNING) << "dy_sub_dy_mean: ";
        PrintTensor<float>(dy_sub_dy_mean, ctx);
      }

      Tensor dy_mul_x_sub_mean_mean;
      {
        framework::NPUAttributeMap attr_input = {{"keep_dims", false},
                                                 {"axes", axes}};

        dy_mul_x_sub_mean_mean.Resize({C});
        dy_mul_x_sub_mean_mean.mutable_data<float>(place);

        const auto &runner = NpuOpRunner("ReduceMeanD", {dy_mul_x_sub_mean},
                                         {dy_mul_x_sub_mean_mean}, attr_input);
        runner.Run(stream);

        LOG(WARNING) << "dy_mul_x_sub_mean_mean: ";
        PrintTensor<float>(dy_mul_x_sub_mean_mean, ctx);
      }

      Tensor dy_mul_x_sub_mean_mean_tile_1;
      {
        dy_mul_x_sub_mean_mean_tile_1.Resize({C});
        dy_mul_x_sub_mean_mean_tile_1.mutable_data<float>(place);

        TensorCopySync(dy_mul_x_sub_mean_mean, place,
                       &dy_mul_x_sub_mean_mean_tile_1);
        LOG(WARNING) << "dy_mul_x_sub_mean_mean_tile_1: ";
        PrintTensor<float>(dy_mul_x_sub_mean_mean_tile_1, ctx);

        if (layout == framework::DataLayout::kNCHW)
          dy_mul_x_sub_mean_mean_tile_1.Resize({1, C, 1, 1});
        else if (layout == framework::DataLayout::kNHWC)
          dy_mul_x_sub_mean_mean_tile_1.Resize({1, 1, 1, C});
        LOG(WARNING) << "dy_mul_x_sub_mean_mean_tile_1: ";
        PrintTensor<float>(dy_mul_x_sub_mean_mean_tile_1, ctx);
      }

      Tensor dy_mul_x_sub_mean_mean_tile;
      {
        framework::NPUAttributeMap attr_input = {{"multiples", multiples}};

        dy_mul_x_sub_mean_mean_tile.Resize(x->dims());
        dy_mul_x_sub_mean_mean_tile.mutable_data<float>(place);

        const auto &runner =
            NpuOpRunner("TileD", {dy_mul_x_sub_mean_mean_tile_1},
                        {dy_mul_x_sub_mean_mean_tile}, attr_input);
        runner.Run(stream);

        LOG(WARNING) << "dy_mul_x_sub_mean_mean_tile: ";
        PrintTensor<float>(dy_mul_x_sub_mean_mean_tile, ctx);
      }

      // (x - mean) * np.mean(dy * (x - mean), axis=axis)
      // x_sub_saved_mean * dy_mul_x_sub_mean_mean_tile
      Tensor tmp1;
      {
        tmp1.Resize(x->dims());
        tmp1.mutable_data<float>(place);

        const auto &runner = NpuOpRunner(
            "Mul", {x_sub_saved_mean, dy_mul_x_sub_mean_mean_tile}, {tmp1}, {});
        runner.Run(stream);

        LOG(WARNING) << "tmp1: ";
        PrintTensor<float>(tmp1, ctx);
      }

      // (x - mean) * np.mean(dy * (x - mean), axis=axis) / (var + epsilon)
      // tmp1 / (var + epsilon)
      // tmp1 / var_ref_tile_add_epsilon
      Tensor tmp2;
      {
        tmp2.Resize(x->dims());
        tmp2.mutable_data<float>(place);

        const auto &runner =
            NpuOpRunner("Div", {tmp1, var_ref_tile_add_epsilon}, {tmp2}, {});
        runner.Run(stream);

        LOG(WARNING) << "tmp2: ";
        PrintTensor<float>(tmp2, ctx);
      }

      // dy - np.mean(dy, axis) - (x - mean) * np.mean(dy * (x - mean), axis) /
      // (var + epsilon)
      // dy_sub_dy_mean - tmp2
      Tensor tmp3;
      {
        tmp3.Resize(x->dims());
        tmp3.mutable_data<float>(place);

        const auto &runner =
            NpuOpRunner("Sub", {dy_sub_dy_mean, tmp2}, {tmp3}, {});
        runner.Run(stream);

        LOG(WARNING) << "tmp3: ";
        PrintTensor<float>(tmp3, ctx);
      }

      Tensor scale_tile_1;
      {
        scale_tile_1.Resize({C});
        scale_tile_1.mutable_data<float>(place);

        TensorCopySync(*scale, place, &scale_tile_1);
        LOG(WARNING) << "scale_tile_1: ";
        PrintTensor<float>(scale_tile_1, ctx);

        if (layout == framework::DataLayout::kNCHW)
          scale_tile_1.Resize({1, C, 1, 1});
        else if (layout == framework::DataLayout::kNHWC)
          scale_tile_1.Resize({1, 1, 1, C});
        LOG(WARNING) << "scale_tile_1: ";
        PrintTensor<float>(scale_tile_1, ctx);
      }

      Tensor scale_tile;
      {
        framework::NPUAttributeMap attr_input = {{"multiples", multiples}};

        scale_tile.Resize(x->dims());
        scale_tile.mutable_data<float>(place);

        const auto &runner =
            NpuOpRunner("TileD", {scale_tile_1}, {scale_tile}, attr_input);
        runner.Run(stream);

        LOG(WARNING) << "scale_tile: ";
        PrintTensor<float>(scale_tile, ctx);
      }

      // scale * (dy - np.mean(dy, axis) - (x - mean) * np.mean(dy * (x - mean),
      // axis) / (var + epsilon))
      // scale * tmp3
      Tensor dx_1;
      {
        dx_1.Resize(x->dims());
        dx_1.mutable_data<float>(place);

        const auto &runner = NpuOpRunner("Mul", {scale_tile, tmp3}, {dx_1}, {});
        runner.Run(stream);

        LOG(WARNING) << "dx_1: ";
        PrintTensor<float>(dx_1, ctx);
      }

      // dx_1 / var_ref_tile_add_epsilon_sqrt
      {
        d_x->Resize(x->dims());
        d_x->mutable_data<float>(place);

        const auto &runner = NpuOpRunner(
            "Div", {dx_1, var_ref_tile_add_epsilon_sqrt}, {*d_x}, {});
        runner.Run(stream);

        LOG(WARNING) << "d_x: ";
        PrintTensor<float>(*d_x, ctx);
      }
    }

    // cacl d_scale
    if (d_scale) {
      Tensor d_scale_2;
      {
        d_scale_2.Resize(x->dims());
        d_scale_2.mutable_data<float>(place);

        const auto &runner = NpuOpRunner(
            "Div", {dy_mul_x_sub_mean, var_ref_tile_add_epsilon_sqrt},
            {d_scale_2}, {});
        runner.Run(stream);

        LOG(WARNING) << "d_scale_2: ";
        PrintTensor<float>(d_scale_2, ctx);
      }

      {
        framework::NPUAttributeMap attr_input = {{"keep_dims", false},
                                                 {"axes", axes}};

        d_scale->mutable_data<float>(place);

        const auto &runner =
            NpuOpRunner("ReduceSumD", {d_scale_2}, {*d_scale}, attr_input);
        runner.Run(stream);

        LOG(WARNING) << "d_scale: ";
        PrintTensor<float>(*d_scale, ctx);
      }
    }

    // cacl d_bias
    if (d_bias) {
      framework::NPUAttributeMap attr_input = {{"keep_dims", false},
                                               {"axes", axes}};

      d_bias->mutable_data<float>(place);

      const auto &runner =
          NpuOpRunner("ReduceSumD", {*d_y}, {*d_bias}, attr_input);
      runner.Run(stream);

      LOG(WARNING) << "d_bias: ";
      PrintTensor<float>(*d_bias, ctx);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_NPU_KERNEL(
    sync_batch_norm, ops::SyncBatchNormNPUKernel<plat::NPUDeviceContext, float>,
    ops::SyncBatchNormNPUKernel<plat::NPUDeviceContext, plat::float16>);
REGISTER_OP_NPU_KERNEL(
    sync_batch_norm_grad,
    ops::SyncBatchNormNPUGradKernel<plat::NPUDeviceContext, float>,
    ops::SyncBatchNormNPUGradKernel<plat::NPUDeviceContext, plat::float16>);
