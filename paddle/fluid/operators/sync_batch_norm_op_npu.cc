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
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/device/npu/hccl_helper.h"
#include "paddle/fluid/platform/device/npu/npu_op_runner.h"

namespace paddle {
namespace operators {

template <typename T>
void training_or_inference(const framework::ExecutionContext &ctx,
                           const aclrtStream &stream,
                           const platform::Place &place,
                           const DataLayout &layout,
                           const bool &test_mode,
                           const int &N,
                           const int &C,
                           const int &H,
                           const int &W,
                           const float epsilon,
                           const float &momentum,
                           const phi::DenseTensor *common_mean,
                           const phi::DenseTensor *common_var,
                           const phi::DenseTensor *x,
                           const phi::DenseTensor *scale,
                           const phi::DenseTensor *bias,
                           const phi::DenseTensor *mean,
                           const phi::DenseTensor *variance,
                           phi::DenseTensor *mean_out,
                           phi::DenseTensor *variance_out,
                           phi::DenseTensor *saved_mean,
                           phi::DenseTensor *saved_variance,
                           phi::DenseTensor *y) {
  std::vector<int> axes;
  if (layout == phi::DataLayout::kNCHW) {
    axes = {0, 2, 3};
  } else if (layout == phi::DataLayout::kNHWC) {
    axes = {0, 1, 2};
  }

  std::vector<int> multiples;
  if (layout == phi::DataLayout::kNCHW)
    multiples = {N, 1, H, W};
  else if (layout == phi::DataLayout::kNHWC)
    multiples = {N, H, W, 1};

  phi::DenseTensor common_mean_tile_1;
  {
    common_mean_tile_1.Resize({C});
    common_mean_tile_1.mutable_data<float>(place);
    paddle::framework::TensorCopySync(*common_mean, place, &common_mean_tile_1);
    if (layout == phi::DataLayout::kNCHW)
      common_mean_tile_1.Resize({1, C, 1, 1});
    else if (layout == phi::DataLayout::kNHWC)
      common_mean_tile_1.Resize({1, 1, 1, C});
  }

  phi::DenseTensor common_mean_tile;
  {
    framework::NPUAttributeMap attr_input = {{"multiples", multiples}};
    common_mean_tile.Resize(x->dims());
    common_mean_tile.mutable_data<float>(place);
    const auto &runner = NpuOpRunner(
        "TileD", {common_mean_tile_1}, {common_mean_tile}, attr_input);
    runner.Run(stream);
  }

  phi::DenseTensor common_var_tile_1;
  {
    common_var_tile_1.Resize({C});
    common_var_tile_1.mutable_data<float>(place);
    paddle::framework::TensorCopySync(*common_var, place, &common_var_tile_1);
    if (layout == phi::DataLayout::kNCHW)
      common_var_tile_1.Resize({1, C, 1, 1});
    else if (layout == phi::DataLayout::kNHWC)
      common_var_tile_1.Resize({1, 1, 1, C});
  }

  phi::DenseTensor common_var_tile;
  {
    framework::NPUAttributeMap attr_input = {{"multiples", multiples}};
    common_var_tile.Resize(x->dims());
    common_var_tile.mutable_data<float>(place);
    const auto &runner = NpuOpRunner(
        "TileD", {common_var_tile_1}, {common_var_tile}, attr_input);
    runner.Run(stream);
  }

  phi::DenseTensor common_var_tile_add_epsilon;
  {
    framework::NPUAttributeMap attr_input = {{"value", epsilon}};
    common_var_tile_add_epsilon.Resize(x->dims());
    common_var_tile_add_epsilon.mutable_data<float>(place);
    const auto &runner = NpuOpRunner(
        "Adds", {common_var_tile}, {common_var_tile_add_epsilon}, attr_input);
    runner.Run(stream);
  }

  phi::DenseTensor common_var_tile_add_epsilon_sqrt;
  {
    common_var_tile_add_epsilon_sqrt.Resize(x->dims());
    common_var_tile_add_epsilon_sqrt.mutable_data<float>(place);
    const auto &runner = NpuOpRunner("Sqrt",
                                     {common_var_tile_add_epsilon},
                                     {common_var_tile_add_epsilon_sqrt},
                                     {});
    runner.Run(stream);
  }

  phi::DenseTensor x_sub_common_mean;
  {
    x_sub_common_mean.Resize(x->dims());
    x_sub_common_mean.mutable_data<float>(place);
    const auto &runner =
        NpuOpRunner("Sub", {*x, common_mean_tile}, {x_sub_common_mean}, {});
    runner.Run(stream);
  }

  phi::DenseTensor normalized;
  {
    normalized.Resize(x->dims());
    normalized.mutable_data<float>(place);
    const auto &runner =
        NpuOpRunner("Div",
                    {x_sub_common_mean, common_var_tile_add_epsilon_sqrt},
                    {normalized},
                    {});
    runner.Run(stream);
  }

  phi::DenseTensor scale_tile_1;
  {
    scale_tile_1.Resize({C});
    scale_tile_1.mutable_data<float>(place);
    paddle::framework::TensorCopySync(*scale, place, &scale_tile_1);
    if (layout == phi::DataLayout::kNCHW)
      scale_tile_1.Resize({1, C, 1, 1});
    else if (layout == phi::DataLayout::kNHWC)
      scale_tile_1.Resize({1, 1, 1, C});
  }

  phi::DenseTensor scale_tile;
  {
    framework::NPUAttributeMap attr_input = {{"multiples", multiples}};
    scale_tile.Resize(x->dims());
    scale_tile.mutable_data<float>(place);
    const auto &runner =
        NpuOpRunner("TileD", {scale_tile_1}, {scale_tile}, attr_input);
    runner.Run(stream);
  }

  phi::DenseTensor normalized_mul_scale;
  {
    normalized_mul_scale.Resize(x->dims());
    normalized_mul_scale.mutable_data<float>(place);
    const auto &runner = NpuOpRunner(
        "Mul", {normalized, scale_tile}, {normalized_mul_scale}, {});
    runner.Run(stream);
  }

  phi::DenseTensor bias_tile_1;
  {
    bias_tile_1.Resize({C});
    bias_tile_1.mutable_data<float>(place);
    paddle::framework::TensorCopySync(*bias, place, &bias_tile_1);
    if (layout == phi::DataLayout::kNCHW)
      bias_tile_1.Resize({1, C, 1, 1});
    else if (layout == phi::DataLayout::kNHWC)
      bias_tile_1.Resize({1, 1, 1, C});
  }

  phi::DenseTensor bias_tile;
  {
    framework::NPUAttributeMap attr_input = {{"multiples", multiples}};
    bias_tile.Resize(x->dims());
    bias_tile.mutable_data<float>(place);
    const auto &runner =
        NpuOpRunner("TileD", {bias_tile_1}, {bias_tile}, attr_input);
    runner.Run(stream);
  }

  // calculate y
  {
    y->mutable_data<T>(place);
    const auto &runner =
        NpuOpRunner("Add", {normalized_mul_scale, bias_tile}, {*y}, {});
    runner.Run(stream);
  }

  if (!test_mode) {
    phi::DenseTensor ones;
    {
      ones.Resize({C});
      ones.mutable_data<float>(place);
      FillNpuTensorWithConstant<float>(&ones, 1);
    }

    // cacl mean_out
    {
      phi::DenseTensor common_mean_mul_1_sub_momentum;
      {
        framework::NPUAttributeMap attr_input = {{"value", 1 - momentum}};
        common_mean_mul_1_sub_momentum.Resize({C});
        common_mean_mul_1_sub_momentum.mutable_data<float>(place);
        const auto &runner = NpuOpRunner("Muls",
                                         {*common_mean},
                                         {common_mean_mul_1_sub_momentum},
                                         attr_input);
        runner.Run(stream);
      }

      phi::DenseTensor mean_mul_momentum;
      {
        framework::NPUAttributeMap attr_input = {{"value", momentum}};
        mean_mul_momentum.Resize({C});
        mean_mul_momentum.mutable_data<float>(place);
        const auto &runner =
            NpuOpRunner("Muls", {*mean}, {mean_mul_momentum}, attr_input);
        runner.Run(stream);
      }

      mean_out->mutable_data<float>(place);

      const auto &runner =
          NpuOpRunner("Add",
                      {common_mean_mul_1_sub_momentum, mean_mul_momentum},
                      {*mean_out},
                      {});
      runner.Run(stream);
    }

    // cacl variance_out
    {
      phi::DenseTensor momentum_mul_var;
      {
        framework::NPUAttributeMap attr_input = {{"value", momentum}};
        momentum_mul_var.Resize({C});
        momentum_mul_var.mutable_data<float>(place);
        const auto &runner =
            NpuOpRunner("Muls", {*variance}, {momentum_mul_var}, attr_input);
        runner.Run(stream);
      }

      phi::DenseTensor var_ref_mul_1_sub_momentum;
      {
        framework::NPUAttributeMap attr_input = {{"value", 1 - momentum}};
        var_ref_mul_1_sub_momentum.Resize({C});
        var_ref_mul_1_sub_momentum.mutable_data<float>(place);
        const auto &runner = NpuOpRunner(
            "Muls", {*common_var}, {var_ref_mul_1_sub_momentum}, attr_input);
        runner.Run(stream);
      }

      variance_out->mutable_data<float>(place);

      const auto &runner =
          NpuOpRunner("Add",
                      {var_ref_mul_1_sub_momentum, momentum_mul_var},
                      {*variance_out},
                      {});
      runner.Run(stream);
    }

    // cacl saved_variance
    {
      phi::DenseTensor var_ref_add_epsilon;
      {
        framework::NPUAttributeMap attr_input = {{"value", epsilon}};
        var_ref_add_epsilon.Resize({C});
        var_ref_add_epsilon.mutable_data<float>(place);
        const auto &runner = NpuOpRunner(
            "Adds", {*common_var}, {var_ref_add_epsilon}, attr_input);
        runner.Run(stream);
      }

      phi::DenseTensor var_ref_add_epsilon_sqrt;
      {
        var_ref_add_epsilon_sqrt.Resize({C});
        var_ref_add_epsilon_sqrt.mutable_data<float>(place);
        const auto &runner = NpuOpRunner(
            "Sqrt", {var_ref_add_epsilon}, {var_ref_add_epsilon_sqrt}, {});
        runner.Run(stream);
      }

      saved_variance->mutable_data<float>(place);

      const auto &runner = NpuOpRunner(
          "Div", {ones, var_ref_add_epsilon_sqrt}, {*saved_variance}, {});
      runner.Run(stream);
    }
  }
}

template <typename DeviceContext, typename T>
class SyncBatchNormNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    const float epsilon = ctx.Attr<float>("epsilon");
    float momentum = ctx.Attr<float>("momentum");
    const bool is_test = ctx.Attr<bool>("is_test");
    const std::string layout_str = ctx.Attr<std::string>("data_layout");
    const DataLayout layout = phi::StringToDataLayout(layout_str);
    const bool use_global_stats = ctx.Attr<bool>("use_global_stats");
    const bool trainable_stats = ctx.Attr<bool>("trainable_statistics");

    PADDLE_ENFORCE_EQ(use_global_stats,
                      false,
                      platform::errors::InvalidArgument(
                          "sync_batch_norm doesn't support "
                          "to set use_global_stats True. Please use batch_norm "
                          "in this case."));

    const auto *x = ctx.Input<phi::DenseTensor>("X");
    auto *y = ctx.Output<phi::DenseTensor>("Y");
    const auto *scale = ctx.Input<phi::DenseTensor>("Scale");
    const auto *bias = ctx.Input<phi::DenseTensor>("Bias");
    const auto *mean = ctx.Input<phi::DenseTensor>("Mean");
    const auto *variance = ctx.Input<phi::DenseTensor>("Variance");
    auto *mean_out = ctx.Output<phi::DenseTensor>("MeanOut");
    auto *variance_out = ctx.Output<phi::DenseTensor>("VarianceOut");
    auto *saved_mean = ctx.Output<phi::DenseTensor>("SavedMean");
    auto *saved_variance = ctx.Output<phi::DenseTensor>("SavedVariance");

    const auto &x_dims = x->dims();
    PADDLE_ENFORCE_EQ(x_dims.size(),
                      4,
                      platform::errors::InvalidArgument(
                          "The input tensor X's dimension must equal to 4. But "
                          "received X's shape = [%s], X's dimension = [%d].",
                          x_dims,
                          x_dims.size()));

    int N, C, H, W, D;
    phi::funcs::ExtractNCWHD(x_dims, layout, &N, &C, &H, &W, &D);

    int x_numel = x->numel();
    auto place = ctx.GetPlace();
    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();

    std::vector<int> axes;
    if (layout == phi::DataLayout::kNCHW) {
      axes = {0, 2, 3};
    } else if (layout == phi::DataLayout::kNHWC) {
      axes = {0, 1, 2};
    }

    bool test_mode = is_test && (!trainable_stats);
    if (test_mode) {  // inference
      // cacl saved_mean
      saved_mean->mutable_data<float>(place);
      paddle::framework::TensorCopySync(*mean, place, saved_mean);

      // cacl saved_variance
      saved_variance->mutable_data<float>(place);
      paddle::framework::TensorCopySync(*variance, place, saved_variance);

      // cacl y
      training_or_inference<T>(ctx,
                               stream,
                               place,
                               layout,
                               test_mode,
                               N,
                               C,
                               H,
                               W,
                               epsilon,
                               momentum,
                               mean,
                               variance,
                               x,
                               scale,
                               bias,
                               mean,
                               variance,
                               NULL,
                               NULL,
                               NULL,
                               NULL,
                               y);

    } else {  // training
      if (ctx.HasInput("MomentumTensor")) {
        const auto *mom_tensor = ctx.Input<phi::DenseTensor>("MomentumTensor");
        phi::DenseTensor mom_cpu;
        paddle::framework::TensorCopySync(
            *mom_tensor, platform::CPUPlace(), &mom_cpu);
        momentum = mom_cpu.data<float>()[0];
      }

      // cacl saved_mean and var_ref
      phi::DenseTensor var_ref;
      var_ref.Resize({C});
      var_ref.mutable_data<float>(place);
      {
        phi::DenseTensor x_sum;
        {
          framework::NPUAttributeMap attr_input = {{"keep_dims", false},
                                                   {"axes", axes}};
          x_sum.Resize({C});
          x_sum.mutable_data<float>(place);
          const auto &runner =
              NpuOpRunner("ReduceSumD", {*x}, {x_sum}, attr_input);
          runner.Run(stream);
        }

        phi::DenseTensor x_square;
        {
          x_square.Resize(x->dims());
          x_square.mutable_data<float>(place);
          const auto &runner = NpuOpRunner("Square", {*x}, {x_square}, {});
          runner.Run(stream);
        }

        phi::DenseTensor x_square_sum;
        {
          framework::NPUAttributeMap attr_input = {{"keep_dims", false},
                                                   {"axes", axes}};
          x_square_sum.Resize({C});
          x_square_sum.mutable_data<float>(place);
          const auto &runner =
              NpuOpRunner("ReduceSumD", {x_square}, {x_square_sum}, attr_input);
          runner.Run(stream);
        }

        auto comm = paddle::platform::HCCLCommContext::Instance().Get(0, place);

        float device_counts = 0.0;
        if (comm) {
          HcclDataType dtype = platform::ToHCCLDataType(
              framework::TransToProtoVarType(mean_out->dtype()));

          phi::DenseTensor device_count_tensor;
          {
            device_count_tensor.Resize({1});
            device_count_tensor.mutable_data<float>(place);
            FillNpuTensorWithConstant<float>(&device_count_tensor, 1);
          }

          // HcclAllReduce device_count_tensor
          {
            void *sendbuff = reinterpret_cast<void *>(
                const_cast<float *>(device_count_tensor.data<float>()));
            void *recvbuff = sendbuff;
            PADDLE_ENFORCE_NPU_SUCCESS(platform::dynload::HcclAllReduce(
                sendbuff,
                recvbuff,
                1,
                dtype,
                HCCL_REDUCE_SUM,
                comm->comm(),
                reinterpret_cast<void *>(stream)));
          }

          std::vector<float> device_count_vec(1);
          paddle::framework::TensorToVector(
              device_count_tensor, ctx.device_context(), &device_count_vec);
          device_counts = device_count_vec[0];

          // HcclAllReduce x_sum
          {
            void *sendbuff = reinterpret_cast<void *>(
                const_cast<float *>(x_sum.data<float>()));
            void *recvbuff = sendbuff;
            PADDLE_ENFORCE_NPU_SUCCESS(platform::dynload::HcclAllReduce(
                sendbuff,
                recvbuff,
                C,
                dtype,
                HCCL_REDUCE_SUM,
                comm->comm(),
                reinterpret_cast<void *>(stream)));
          }

          // HcclAllReduce x_square_sum
          {
            void *sendbuff = reinterpret_cast<void *>(
                const_cast<float *>(x_square_sum.data<float>()));
            void *recvbuff = sendbuff;
            PADDLE_ENFORCE_NPU_SUCCESS(platform::dynload::HcclAllReduce(
                sendbuff,
                recvbuff,
                C,
                dtype,
                HCCL_REDUCE_SUM,
                comm->comm(),
                reinterpret_cast<void *>(stream)));
          }
        }

        // cacl saved_mean
        {
          framework::NPUAttributeMap attr_input = {
              {"value", 1.0f * C / x_numel / device_counts}};
          saved_mean->mutable_data<float>(place);
          const auto &runner =
              NpuOpRunner("Muls", {x_sum}, {*saved_mean}, attr_input);
          runner.Run(stream);
        }

        // cacl var_ref
        {
          phi::DenseTensor saved_mean_square;
          {
            saved_mean_square.Resize({C});
            saved_mean_square.mutable_data<float>(place);
            const auto &runner =
                NpuOpRunner("Square", {*saved_mean}, {saved_mean_square}, {});
            runner.Run(stream);
          }

          phi::DenseTensor var_ref_tmp;
          var_ref_tmp.Resize({C});
          var_ref_tmp.mutable_data<float>(place);
          {
            framework::NPUAttributeMap attr_input = {
                {"value", 1.0f * C / x_numel / device_counts}};
            const auto &runner =
                NpuOpRunner("Muls", {x_square_sum}, {var_ref_tmp}, attr_input);
            runner.Run(stream);
          }

          // cacl var_ref
          {
            const auto &runner = NpuOpRunner(
                "Sub", {var_ref_tmp, saved_mean_square}, {var_ref}, {});
            runner.Run(stream);
          }
        }
      }

      training_or_inference<T>(ctx,
                               stream,
                               place,
                               layout,
                               test_mode,
                               N,
                               C,
                               H,
                               W,
                               epsilon,
                               momentum,
                               saved_mean,
                               &var_ref,
                               x,
                               scale,
                               bias,
                               mean,
                               variance,
                               mean_out,
                               variance_out,
                               saved_mean,
                               saved_variance,
                               y);
    }
  }
};

template <typename DeviceContext, typename T>
class SyncBatchNormNPUGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    float epsilon = ctx.Attr<float>("epsilon");
    const std::string layout_str = ctx.Attr<std::string>("data_layout");
    const DataLayout layout = phi::StringToDataLayout(layout_str);

    const auto *d_y = ctx.Input<phi::DenseTensor>(framework::GradVarName("Y"));
    const auto *scale = ctx.Input<phi::DenseTensor>("Scale");
    auto *d_x = ctx.Output<phi::DenseTensor>(framework::GradVarName("X"));
    auto *d_scale =
        ctx.Output<phi::DenseTensor>(framework::GradVarName("Scale"));
    auto *d_bias = ctx.Output<phi::DenseTensor>(framework::GradVarName("Bias"));
    const auto *saved_mean = ctx.Input<phi::DenseTensor>("SavedMean");

    const phi::DenseTensor *x;
    if (ctx.HasInput("Y")) {
      PADDLE_ENFORCE_EQ(true,
                        false,
                        platform::errors::InvalidArgument(
                            "sync_batch_norm_grad doesn't support input Y"));
    } else {
      x = ctx.Input<phi::DenseTensor>("X");
    }

    int N, C, H, W, D;
    phi::funcs::ExtractNCWHD(x->dims(), layout, &N, &C, &H, &W, &D);

    int x_numel = x->numel();
    auto place = ctx.GetPlace();
    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();

    std::vector<int> axes;
    if (layout == phi::DataLayout::kNCHW) {
      axes = {0, 2, 3};
    } else if (layout == phi::DataLayout::kNHWC) {
      axes = {0, 1, 2};
    }

    std::vector<int> multiples;
    if (layout == phi::DataLayout::kNCHW)
      multiples = {N, 1, H, W};
    else if (layout == phi::DataLayout::kNHWC)
      multiples = {N, H, W, 1};

    auto comm = paddle::platform::HCCLCommContext::Instance().Get(0, place);
    HcclDataType dtype = platform::ToHCCLDataType(
        framework::TransToProtoVarType(scale->dtype()));

    float device_counts = 0.0;
    if (comm) {
      phi::DenseTensor device_count_tensor;
      {
        device_count_tensor.Resize({1});
        device_count_tensor.mutable_data<float>(place);
        FillNpuTensorWithConstant<float>(&device_count_tensor, 1);
      }

      // HcclAllReduce device_count_tensor
      {
        void *sendbuff = reinterpret_cast<void *>(
            const_cast<float *>(device_count_tensor.data<float>()));
        void *recvbuff = sendbuff;
        PADDLE_ENFORCE_NPU_SUCCESS(
            platform::dynload::HcclAllReduce(sendbuff,
                                             recvbuff,
                                             1,
                                             dtype,
                                             HCCL_REDUCE_SUM,
                                             comm->comm(),
                                             reinterpret_cast<void *>(stream)));
      }

      std::vector<float> device_count_vec(1);
      paddle::framework::TensorToVector(
          device_count_tensor, ctx.device_context(), &device_count_vec);
      device_counts = device_count_vec[0];
      PADDLE_ENFORCE_GE(
          device_counts,
          2,
          platform::errors::PreconditionNotMet("device_counts should >= 2."));
    }

    // cacl var_ref
    phi::DenseTensor var_ref;
    var_ref.Resize({C});
    var_ref.mutable_data<float>(place);
    {
      // cacl var_ref
      {
        phi::DenseTensor x_square;
        {
          x_square.Resize(x->dims());
          x_square.mutable_data<float>(place);
          const auto &runner = NpuOpRunner("Square", {*x}, {x_square}, {});
          runner.Run(stream);
        }

        phi::DenseTensor x_square_sum;
        {
          framework::NPUAttributeMap attr_input = {{"keep_dims", false},
                                                   {"axes", axes}};
          x_square_sum.Resize({C});
          x_square_sum.mutable_data<float>(place);
          const auto &runner =
              NpuOpRunner("ReduceSumD", {x_square}, {x_square_sum}, attr_input);
          runner.Run(stream);
        }

        phi::DenseTensor x_square_sum_mean;
        {
          framework::NPUAttributeMap attr_input = {
              {"value", 1.0f * C / x_numel}};
          x_square_sum_mean.Resize({C});
          x_square_sum_mean.mutable_data<float>(place);
          const auto &runner = NpuOpRunner(
              "Muls", {x_square_sum}, {x_square_sum_mean}, attr_input);
          runner.Run(stream);
        }

        phi::DenseTensor mean_square;
        {
          mean_square.Resize({C});
          mean_square.mutable_data<float>(place);
          const auto &runner =
              NpuOpRunner("Square", {*saved_mean}, {mean_square}, {});
          runner.Run(stream);
        }

        // cacl var_ref
        {
          const auto &runner = NpuOpRunner(
              "Sub", {x_square_sum_mean, mean_square}, {var_ref}, {});
          runner.Run(stream);
        }
      }
    }

    phi::DenseTensor saved_mean_tile_1;
    {
      saved_mean_tile_1.Resize({C});
      saved_mean_tile_1.mutable_data<float>(place);
      paddle::framework::TensorCopySync(*saved_mean, place, &saved_mean_tile_1);
      if (layout == phi::DataLayout::kNCHW)
        saved_mean_tile_1.Resize({1, C, 1, 1});
      else if (layout == phi::DataLayout::kNHWC)
        saved_mean_tile_1.Resize({1, 1, 1, C});
    }

    phi::DenseTensor saved_mean_tile;
    {
      framework::NPUAttributeMap attr_input = {{"multiples", multiples}};
      saved_mean_tile.Resize(x->dims());
      saved_mean_tile.mutable_data<float>(place);
      const auto &runner = NpuOpRunner(
          "TileD", {saved_mean_tile_1}, {saved_mean_tile}, attr_input);
      runner.Run(stream);
    }

    phi::DenseTensor x_sub_saved_mean;
    {
      x_sub_saved_mean.Resize(x->dims());
      x_sub_saved_mean.mutable_data<float>(place);
      const auto &runner =
          NpuOpRunner("Sub", {*x, saved_mean_tile}, {x_sub_saved_mean}, {});
      runner.Run(stream);
    }

    phi::DenseTensor var_ref_tile_1;
    {
      var_ref_tile_1.Resize({C});
      var_ref_tile_1.mutable_data<float>(place);
      paddle::framework::TensorCopySync(var_ref, place, &var_ref_tile_1);
      if (layout == phi::DataLayout::kNCHW)
        var_ref_tile_1.Resize({1, C, 1, 1});
      else if (layout == phi::DataLayout::kNHWC)
        var_ref_tile_1.Resize({1, 1, 1, C});
    }

    phi::DenseTensor var_ref_tile;
    {
      framework::NPUAttributeMap attr_input = {{"multiples", multiples}};
      var_ref_tile.Resize(x->dims());
      var_ref_tile.mutable_data<float>(place);
      const auto &runner =
          NpuOpRunner("TileD", {var_ref_tile_1}, {var_ref_tile}, attr_input);
      runner.Run(stream);
    }

    phi::DenseTensor var_ref_tile_add_epsilon;
    {
      framework::NPUAttributeMap attr_input = {{"value", epsilon}};
      var_ref_tile_add_epsilon.Resize(x->dims());
      var_ref_tile_add_epsilon.mutable_data<float>(place);
      const auto &runner = NpuOpRunner(
          "Adds", {var_ref_tile}, {var_ref_tile_add_epsilon}, attr_input);
      runner.Run(stream);
    }

    phi::DenseTensor var_ref_tile_add_epsilon_sqrt;
    {
      var_ref_tile_add_epsilon_sqrt.Resize(x->dims());
      var_ref_tile_add_epsilon_sqrt.mutable_data<float>(place);
      const auto &runner = NpuOpRunner("Sqrt",
                                       {var_ref_tile_add_epsilon},
                                       {var_ref_tile_add_epsilon_sqrt},
                                       {});
      runner.Run(stream);
    }

    phi::DenseTensor dy_mul_x_sub_mean_for_scale;
    {
      if (framework::TransToProtoVarType(d_y->dtype()) ==
          framework::proto::VarType::FP16) {
        dy_mul_x_sub_mean_for_scale.Resize(x->dims());
        dy_mul_x_sub_mean_for_scale.mutable_data<float>(place);
        const auto &runner = NpuOpRunner(
            "Mul", {*d_y, x_sub_saved_mean}, {dy_mul_x_sub_mean_for_scale}, {});
        runner.Run(stream);
      } else {
        dy_mul_x_sub_mean_for_scale.Resize(x->dims());
        dy_mul_x_sub_mean_for_scale.mutable_data<float>(place);
        const auto &runner = NpuOpRunner(
            "Mul", {*d_y, x_sub_saved_mean}, {dy_mul_x_sub_mean_for_scale}, {});
        runner.Run(stream);
      }
    }

    phi::DenseTensor dy_mul_x_sub_mean;
    {
      if (framework::TransToProtoVarType(d_y->dtype()) ==
          framework::proto::VarType::FP16) {
        dy_mul_x_sub_mean.Resize(x->dims());
        dy_mul_x_sub_mean.mutable_data<float>(place);
        const auto &runner = NpuOpRunner(
            "Mul", {*d_y, x_sub_saved_mean}, {dy_mul_x_sub_mean}, {});
        runner.Run(stream);
      } else {
        dy_mul_x_sub_mean.Resize(x->dims());
        dy_mul_x_sub_mean.mutable_data<float>(place);
        const auto &runner = NpuOpRunner(
            "Mul", {*d_y, x_sub_saved_mean}, {dy_mul_x_sub_mean}, {});
        runner.Run(stream);
      }
    }

    // HcclAllReduce dy_mul_x_sub_mean
    if (comm) {
      {
        void *sendbuff = reinterpret_cast<void *>(
            const_cast<float *>(dy_mul_x_sub_mean.data<float>()));
        void *recvbuff = sendbuff;
        PADDLE_ENFORCE_NPU_SUCCESS(
            platform::dynload::HcclAllReduce(sendbuff,
                                             recvbuff,
                                             C,
                                             dtype,
                                             HCCL_REDUCE_SUM,
                                             comm->comm(),
                                             reinterpret_cast<void *>(stream)));
      }

      {
        framework::NPUAttributeMap attr_input = {
            {"value", 1.0f / device_counts}};
        const auto &runner = NpuOpRunner(
            "Muls", {dy_mul_x_sub_mean}, {dy_mul_x_sub_mean}, attr_input);
        runner.Run(stream);
      }
    }

    // cacl d_x
    if (d_x) {
      phi::DenseTensor dy_mean;
      {
        if (framework::TransToProtoVarType(d_y->dtype()) ==
            framework::proto::VarType::FP16) {
          framework::NPUAttributeMap attr_input = {{"keep_dims", false},
                                                   {"axes", axes}};
          dy_mean.Resize({C});
          dy_mean.mutable_data<float>(place);
          const auto &runner =
              NpuOpRunner("ReduceMeanD", {*d_y}, {dy_mean}, attr_input);
          runner.Run(stream);
        } else {
          framework::NPUAttributeMap attr_input = {{"keep_dims", false},
                                                   {"axes", axes}};
          dy_mean.Resize({C});
          dy_mean.mutable_data<float>(place);
          const auto &runner =
              NpuOpRunner("ReduceMeanD", {*d_y}, {dy_mean}, attr_input);
          runner.Run(stream);
        }
      }

      // HcclAllReduce dy_mean
      if (comm) {
        {
          void *sendbuff = reinterpret_cast<void *>(
              const_cast<float *>(dy_mean.data<float>()));
          void *recvbuff = sendbuff;
          PADDLE_ENFORCE_NPU_SUCCESS(platform::dynload::HcclAllReduce(
              sendbuff,
              recvbuff,
              C,
              dtype,
              HCCL_REDUCE_SUM,
              comm->comm(),
              reinterpret_cast<void *>(stream)));
        }

        {
          framework::NPUAttributeMap attr_input = {
              {"value", 1.0f / device_counts}};
          const auto &runner =
              NpuOpRunner("Muls", {dy_mean}, {dy_mean}, attr_input);
          runner.Run(stream);
        }
      }

      phi::DenseTensor dy_mean_tile_1;
      {
        dy_mean_tile_1.Resize({C});
        dy_mean_tile_1.mutable_data<float>(place);
        paddle::framework::TensorCopySync(dy_mean, place, &dy_mean_tile_1);
        if (layout == phi::DataLayout::kNCHW)
          dy_mean_tile_1.Resize({1, C, 1, 1});
        else if (layout == phi::DataLayout::kNHWC)
          dy_mean_tile_1.Resize({1, 1, 1, C});
      }

      phi::DenseTensor dy_mean_tile;
      {
        framework::NPUAttributeMap attr_input = {{"multiples", multiples}};
        dy_mean_tile.Resize(x->dims());
        dy_mean_tile.mutable_data<float>(place);
        const auto &runner =
            NpuOpRunner("TileD", {dy_mean_tile_1}, {dy_mean_tile}, attr_input);
        runner.Run(stream);
      }

      phi::DenseTensor dy_sub_dy_mean;
      {
        if (framework::TransToProtoVarType(d_y->dtype()) ==
            framework::proto::VarType::FP16) {
          dy_sub_dy_mean.Resize(x->dims());
          dy_sub_dy_mean.mutable_data<float>(place);
          const auto &runner =
              NpuOpRunner("Sub", {*d_y, dy_mean_tile}, {dy_sub_dy_mean}, {});
          runner.Run(stream);
        } else {
          dy_sub_dy_mean.Resize(x->dims());
          dy_sub_dy_mean.mutable_data<float>(place);
          const auto &runner =
              NpuOpRunner("Sub", {*d_y, dy_mean_tile}, {dy_sub_dy_mean}, {});
          runner.Run(stream);
        }
      }

      phi::DenseTensor dy_mul_x_sub_mean_mean;
      {
        framework::NPUAttributeMap attr_input = {{"keep_dims", false},
                                                 {"axes", axes}};
        dy_mul_x_sub_mean_mean.Resize({C});
        dy_mul_x_sub_mean_mean.mutable_data<float>(place);
        const auto &runner = NpuOpRunner("ReduceMeanD",
                                         {dy_mul_x_sub_mean},
                                         {dy_mul_x_sub_mean_mean},
                                         attr_input);
        runner.Run(stream);
      }

      phi::DenseTensor dy_mul_x_sub_mean_mean_tile_1;
      {
        dy_mul_x_sub_mean_mean_tile_1.Resize({C});
        dy_mul_x_sub_mean_mean_tile_1.mutable_data<float>(place);
        paddle::framework::TensorCopySync(
            dy_mul_x_sub_mean_mean, place, &dy_mul_x_sub_mean_mean_tile_1);
        if (layout == phi::DataLayout::kNCHW)
          dy_mul_x_sub_mean_mean_tile_1.Resize({1, C, 1, 1});
        else if (layout == phi::DataLayout::kNHWC)
          dy_mul_x_sub_mean_mean_tile_1.Resize({1, 1, 1, C});
      }

      phi::DenseTensor dy_mul_x_sub_mean_mean_tile;
      {
        framework::NPUAttributeMap attr_input = {{"multiples", multiples}};
        dy_mul_x_sub_mean_mean_tile.Resize(x->dims());
        dy_mul_x_sub_mean_mean_tile.mutable_data<float>(place);
        const auto &runner = NpuOpRunner("TileD",
                                         {dy_mul_x_sub_mean_mean_tile_1},
                                         {dy_mul_x_sub_mean_mean_tile},
                                         attr_input);
        runner.Run(stream);
      }

      // (x - mean) * np.mean(dy * (x - mean), axis=axis)
      // x_sub_saved_mean * dy_mul_x_sub_mean_mean_tile
      phi::DenseTensor tmp1;
      {
        tmp1.Resize(x->dims());
        tmp1.mutable_data<float>(place);
        const auto &runner = NpuOpRunner(
            "Mul", {x_sub_saved_mean, dy_mul_x_sub_mean_mean_tile}, {tmp1}, {});
        runner.Run(stream);
      }

      // (x - mean) * np.mean(dy * (x - mean), axis=axis) / (var + epsilon)
      // tmp1 / (var + epsilon)
      // tmp1 / var_ref_tile_add_epsilon
      phi::DenseTensor tmp2;
      {
        tmp2.Resize(x->dims());
        tmp2.mutable_data<float>(place);
        const auto &runner =
            NpuOpRunner("Div", {tmp1, var_ref_tile_add_epsilon}, {tmp2}, {});
        runner.Run(stream);
      }

      // dy - np.mean(dy, axis) - (x - mean) * np.mean(dy * (x - mean), axis) /
      // (var + epsilon)
      // dy_sub_dy_mean - tmp2
      phi::DenseTensor tmp3;
      {
        tmp3.Resize(x->dims());
        tmp3.mutable_data<float>(place);
        const auto &runner =
            NpuOpRunner("Sub", {dy_sub_dy_mean, tmp2}, {tmp3}, {});
        runner.Run(stream);
      }

      phi::DenseTensor scale_tile_1;
      {
        scale_tile_1.Resize({C});
        scale_tile_1.mutable_data<float>(place);
        paddle::framework::TensorCopySync(*scale, place, &scale_tile_1);
        if (layout == phi::DataLayout::kNCHW)
          scale_tile_1.Resize({1, C, 1, 1});
        else if (layout == phi::DataLayout::kNHWC)
          scale_tile_1.Resize({1, 1, 1, C});
      }

      phi::DenseTensor scale_tile;
      {
        framework::NPUAttributeMap attr_input = {{"multiples", multiples}};
        scale_tile.Resize(x->dims());
        scale_tile.mutable_data<float>(place);
        const auto &runner =
            NpuOpRunner("TileD", {scale_tile_1}, {scale_tile}, attr_input);
        runner.Run(stream);
      }

      // scale * (dy - np.mean(dy, axis) - (x - mean) * np.mean(dy * (x - mean),
      // axis) / (var + epsilon))
      // scale * tmp3
      phi::DenseTensor dx_1;
      {
        dx_1.Resize(x->dims());
        dx_1.mutable_data<float>(place);

        const auto &runner = NpuOpRunner("Mul", {scale_tile, tmp3}, {dx_1}, {});
        runner.Run(stream);
      }

      // dx_1 / var_ref_tile_add_epsilon_sqrt
      {
        d_x->Resize(x->dims());
        d_x->mutable_data<T>(place);
        const auto &runner = NpuOpRunner(
            "Div", {dx_1, var_ref_tile_add_epsilon_sqrt}, {*d_x}, {});
        runner.Run(stream);
      }
    }

    // cacl d_scale
    if (d_scale) {
      phi::DenseTensor d_scale_2;
      {
        d_scale_2.Resize(x->dims());
        d_scale_2.mutable_data<float>(place);
        const auto &runner = NpuOpRunner(
            "Div",
            {dy_mul_x_sub_mean_for_scale, var_ref_tile_add_epsilon_sqrt},
            {d_scale_2},
            {});
        runner.Run(stream);
      }

      {
        framework::NPUAttributeMap attr_input = {{"keep_dims", false},
                                                 {"axes", axes}};
        d_scale->mutable_data<float>(place);
        const auto &runner =
            NpuOpRunner("ReduceSumD", {d_scale_2}, {*d_scale}, attr_input);
        runner.Run(stream);
      }
    }

    // cacl d_bias
    if (d_bias) {
      if (framework::TransToProtoVarType(d_y->dtype()) ==
          framework::proto::VarType::FP16) {
        framework::NPUAttributeMap attr_input = {{"keep_dims", false},
                                                 {"axes", axes}};
        d_bias->mutable_data<float>(place);
        const auto &runner =
            NpuOpRunner("ReduceSumD", {*d_y}, {*d_bias}, attr_input);
        runner.Run(stream);
      } else {
        framework::NPUAttributeMap attr_input = {{"keep_dims", false},
                                                 {"axes", axes}};
        d_bias->mutable_data<float>(place);
        const auto &runner =
            NpuOpRunner("ReduceSumD", {*d_y}, {*d_bias}, attr_input);
        runner.Run(stream);
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_NPU_KERNEL(
    sync_batch_norm,
    ops::SyncBatchNormNPUKernel<plat::NPUDeviceContext, float>);
REGISTER_OP_NPU_KERNEL(
    sync_batch_norm_grad,
    ops::SyncBatchNormNPUGradKernel<plat::NPUDeviceContext, float>);
