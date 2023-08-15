// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include <gtest/gtest.h>

#include "paddle/cinn/frontend/decomposer/test_helper.h"

namespace cinn {
namespace frontend {
namespace {

struct Offset {
  int n;
  int c;
  int h;
  int w;

  Offset(int arg_n, int arg_c, int arg_h, int arg_w)
      : n(arg_n), c(arg_c), h(arg_h), w(arg_w) {}

  int operator()(int idx_n, int idx_c, int idx_h, int idx_w) const {
    return idx_n * c * h * w + idx_c * h * w + idx_h * w + idx_w;
  }
};

template <typename FuncType>
void Loop(FuncType func, const int n, const int c, const int h, const int w) {
  for (int in = 0; in < n; ++in) {
    for (int ic = 0; ic < c; ++ic) {
      for (int ih = 0; ih < h; ++ih) {
        for (int iw = 0; iw < w; ++iw) {
          func(in, ic, ih, iw);
        }
      }
    }
  }
}

template <typename T>
void ComputeBatchNormTrainRef(const std::vector<T>& x,
                              const std::vector<T>& scale,
                              const std::vector<T>& bias,
                              const std::vector<T>& moving_mean,
                              const std::vector<T>& moving_variance,
                              const int n,
                              const int c,
                              const int h,
                              const int w,
                              std::vector<T>* y,
                              std::vector<T>* saved_mean,
                              std::vector<T>* saved_variance,
                              std::vector<T>* new_moving_mean,
                              std::vector<T>* new_moving_variance,
                              const float epsilon,
                              const float momentum) {
  Offset offset(n, c, h, w);

  // sum
  memset(saved_mean->data(), 0, sizeof(T) * c);
  auto func_sum_x = [=](int in, int ic, int ih, int iw) {
    saved_mean->at(ic) += x[offset(in, ic, ih, iw)];
  };
  Loop(func_sum_x, n, c, h, w);

  // saved mean
  float element_count = static_cast<float>(n * h * w);
  for (int ic = 0; ic < c; ++ic) {
    // Checking result of saved_mean:
    // output[saved_mean], var_name=var_5, shape={32}
    // - Total 0 different results, offset=0, 0.00527001 vs 0.00527001,
    // maximum_relative_diff=0(absolute_diff=0)
    saved_mean->at(ic) /= element_count;
  }

  // square_sum
  std::vector<float> x_square_mean(c, 0);
  auto func_sum_square_x = [&](int in, int ic, int ih, int iw) {
    x_square_mean.at(ic) +=
        x[offset(in, ic, ih, iw)] * x[offset(in, ic, ih, iw)];
  };
  Loop(func_sum_square_x, n, c, h, w);

  for (int ic = 0; ic < c; ++ic) {
    x_square_mean[ic] /= element_count;
  }

  // saved variance, according to equation: E(x^2) - [E(x)]^2
  std::vector<float> std_variance(c);
  for (int ic = 0; ic < c; ++ic) {
    // Checking results of saved_variance and std_variance:
    // output[saved_variance], var_name=var_6, shape={32}
    // - Total 0 different results, offset=0, 0.336347 vs 0.336347,
    // maximum_relative_diff=0(absolute_diff=0) output[std_variance],
    // var_name=std_variance, shape={32}
    // - Total 0 different results, offset=0, 0.579963 vs 0.579963,
    // maximum_relative_diff=0(absolute_diff=0)
    saved_variance->at(ic) =
        x_square_mean[ic] - (saved_mean->at(ic) * saved_mean->at(ic));
    std_variance[ic] = sqrt(saved_variance->at(ic) + epsilon);
  }

  // compute output
  std::vector<float> y_nobias(n * c * h * w);
  auto func_y_nobias = [&](int in, int ic, int ih, int iw) {
    int idx = offset(in, ic, ih, iw);
    // Checking result of y_nobias:
    // output[y_nobias], var_name=y_nobias, shape={16, 32, 16, 16}
    // - Total 0 different results, offset=32104, -0.000488288 vs -0.000488288,
    // maximum_relative_diff=1.19208e-07(absolute_diff=5.82077e-11)
    y_nobias[idx] =
        (x[idx] - saved_mean->at(ic)) * scale[ic] / std_variance[ic];
  };
  Loop(func_y_nobias, n, c, h, w);

  auto func_y = [&](int in, int ic, int ih, int iw) {
    int idx = offset(in, ic, ih, iw);
    // Checking result of y:
    // output[y], var_name=var_4, shape={16, 32, 16, 16}
    // - Total 80 different results, offset=126409, 1.81794e-06 vs 1.80304e-06,
    // maximum_relative_diff=0.00826446(absolute_diff=1.49012e-08) For the
    // following case:
    //   idx=126409, y[idx]=1.80304e-06, y_nobias[idx]=0.2033332,
    //   bias[ic]=-0.2033314
    // The computing result of CPU and GPU may have some difference, like
    //   i=126409, 1.8179417e-06 vs 1.8030405e-06, relative_diff=0.0082644625,
    //   absolute_diff=1.4901161e-08
    // This case is considered reasonable.
    y->at(idx) = y_nobias[idx] + bias[ic];
  };
  Loop(func_y, n, c, h, w);

  // new moving running and variance
  float factor_0 = momentum;
  float factor_1 = static_cast<float>(1.0f - momentum);
  for (int ic = 0; ic < c; ++ic) {
    // Checking result of new_moving_mean and new_moving_variance:
    // output[new_moving_mean], var_name=var_7, shape={32}
    // - Total 0 different results, offset=9, 0.00123065 vs 0.00123065,
    // maximum_relative_diff=9.45967e-08(absolute_diff=1.16415e-10)
    // output[new_moving_variance], var_name=var_8, shape={32}
    // - Total 0 different results, offset=16, -0.00140787 vs -0.00140787,
    // maximum_relative_diff=5.29211e-06(absolute_diff=7.45058e-09)
    new_moving_mean->at(ic) =
        moving_mean[ic] * factor_0 + saved_mean->at(ic) * factor_1;
    new_moving_variance->at(ic) =
        moving_variance[ic] * factor_0 + saved_variance->at(ic) * factor_1;
  }
}

TEST(Decomposer, BatchNormTrain) {
  int n = 16, c = 128, h = 14, w = 14;
  float epsilon = 1e-5;
  float momentum = 0.9f;
  std::string data_layout = "NCHW";
  bool is_test = false;
  NetBuilder net_builder("batch_norm_train");
  std::vector<std::string> output_names;
  {
    auto x = net_builder.CreateInput(Float(32), {n, c, h, w}, "x");
    auto scale = net_builder.CreateInput(Float(32), {c}, "scale");
    auto bias = net_builder.CreateInput(Float(32), {c}, "bias");
    auto moving_mean = net_builder.CreateInput(Float(32), {c}, "moving_mean");
    auto moving_variance =
        net_builder.CreateInput(Float(32), {c}, "moving_variance");

    auto outputs = net_builder.BatchNorm(x,
                                         scale,
                                         bias,
                                         moving_mean,
                                         moving_variance,
                                         epsilon,
                                         momentum,
                                         data_layout,
                                         is_test);
    for (auto output : outputs) {
      output_names.push_back(output->id);
    }
  }
  auto program = net_builder.Build();

  auto target = common::DefaultTarget();
  RunDecomposer(&program,
                target,
                cinn::frontend::DefaultTrainingOptimizeOptions().program_passes,
                output_names);

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  hlir::framework::ApplyPass(graph.get(), "OpFusionPass");
  hlir::framework::ApplyPass(graph.get(), "FusionMergePass");

  auto scope = BuildScope(target, graph);
  hlir::framework::CompilationContext context(graph, scope, target);
  hlir::framework::GraphCompiler gc(context);
  auto run_program = gc.Build();

  // set input
  float precision = 1e-3;
  std::vector<float> x(n * c * h * w), scale(c), bias(c), moving_mean(c),
      moving_variance(c);
  InitRandomVector<float>(&x, n * c * h * w, 0.0f, 1.0f, precision);
  InitRandomVector<float>(&scale, c, 0.0f, 1.0f, precision);
  InitRandomVector<float>(&bias, c, 10.0f, 20.0f, precision);
  InitRandomVector<float>(&moving_mean, c, 0.0f, 1.0f, precision);
  InitRandomVector<float>(&moving_variance, c, 0.0f, 1.0f, precision);

  std::vector<float> y(n * c * h * w), new_moving_mean(c),
      new_moving_variance(c), saved_mean(c), saved_variance(c);
  ComputeBatchNormTrainRef<float>(x,
                                  scale,
                                  bias,
                                  moving_mean,
                                  moving_variance,
                                  n,
                                  c,
                                  h,
                                  w,
                                  &y,
                                  &saved_mean,
                                  &saved_variance,
                                  &new_moving_mean,
                                  &new_moving_variance,
                                  epsilon,
                                  momentum);

  std::vector<std::pair<std::string, std::vector<float>>> inputs = {
      {"x", x},
      {"scale", scale},
      {"bias", bias},
      {"moving_mean", moving_mean},
      {"moving_variance", moving_variance}};
  for (auto& input : inputs) {
    scope->Var<hlir::framework::Tensor>(input.first);
    auto tensor = scope->GetTensor(input.first);
    auto* data = tensor->mutable_data<float>(target);
    CopyFromVector(input.second, tensor, target);
  }
  run_program->Execute();

  std::unordered_map<std::string, std::pair<std::string, std::vector<float>>>
      outputs_ref = {
          {"new_moving_variance", {output_names[4], new_moving_variance}},
          {"new_moving_mean", {output_names[3], new_moving_mean}},
          {"saved_variance", {output_names[2], saved_variance}},
          {"saved_mean", {output_names[1], saved_mean}},
          {"y", {output_names[0], y}}};

  for (auto& iter : outputs_ref) {
    auto output = iter.second;
    auto tensor = scope->GetTensor(output.first);
    std::vector<float> data(tensor->shape().numel());
    CopyToVector(tensor, &data);

    LOG(INFO) << "output[" << iter.first << "], var_name=" << output.first
              << ", shape=" << tensor->shape().data();
    CheckOutput<float>(data, output.second, 1e-8, 1e-4);
  }
}

template <typename T>
void ComputeBatchNormGradRef(const std::vector<T>& y_grad,
                             const std::vector<T>& x,
                             const std::vector<T>& scale,
                             const std::vector<T>& save_mean,
                             const std::vector<T>& save_variance,
                             const int n,
                             const int c,
                             const int h,
                             const int w,
                             std::vector<T>* x_grad,
                             std::vector<T>* scale_grad,
                             std::vector<T>* bias_grad,
                             const float epsilon = 1e-5) {
  Offset offset(n, c, h, w);

  // bias_grad
  memset(bias_grad->data(), 0, sizeof(T) * c);
  auto func_bias_grad = [=](int in, int ic, int ih, int iw) {
    bias_grad->at(ic) += y_grad[offset(in, ic, ih, iw)];
  };
  Loop(func_bias_grad, n, c, h, w);

  // std_variance
  std::vector<T> std_variance(c);
  for (int ic = 0; ic < c; ++ic) {
    std_variance[ic] = sqrt(save_variance[ic] + epsilon);
  }

  // grad scale
  memset(scale_grad->data(), 0, sizeof(T) * c);
  auto func_scale_grad = [=](int in, int ic, int ih, int iw) {
    int idx = offset(in, ic, ih, iw);
    scale_grad->at(ic) += y_grad[idx] * (x[idx] - save_mean[ic]);
  };
  Loop(func_scale_grad, n, c, h, w);
  for (int ic = 0; ic < c; ++ic) {
    scale_grad->at(ic) /= std_variance[ic];
  }

  // std_norm_grad
  std::vector<T> std_norm_grad(n * c * h * w);
  auto func_std_norm_grad = [&](int in, int ic, int ih, int iw) {
    int idx = offset(in, ic, ih, iw);
    std_norm_grad[idx] = y_grad[idx] * scale[ic];
  };
  Loop(func_std_norm_grad, n, c, h, w);

  // x_mean_diff_grad
  std::vector<T> x_mean_diff_grad(n * c * h * w);
  auto func_x_mean_diff_grad = [&](int in, int ic, int ih, int iw) {
    int idx = offset(in, ic, ih, iw);
    x_mean_diff_grad[idx] = std_norm_grad[idx] / std_variance[ic];
  };
  Loop(func_x_mean_diff_grad, n, c, h, w);

  // std_variance_grad
  std::vector<T> std_variance_grad(c, 0);
  auto func_std_variance_grad = [&](int in, int ic, int ih, int iw) {
    int idx = offset(in, ic, ih, iw);
    std_variance_grad[ic] += -1.0f * std_norm_grad[idx] *
                             (x[idx] - save_mean[ic]) /
                             (save_variance[ic] + epsilon);
  };
  Loop(func_std_variance_grad, n, c, h, w);

  // variance_grad_without_mul
  std::vector<T> variance_grad_without_mul(c);
  for (int ic = 0; ic < c; ++ic) {
    variance_grad_without_mul[ic] = std_variance_grad[ic] / std_variance[ic];
  }

  // x_grad_0
  float element_count = static_cast<float>(n * h * w);
  std::vector<T> x_grad_0(n * c * h * w);
  auto func_x_grad_0 = [&](int in, int ic, int ih, int iw) {
    int idx = offset(in, ic, ih, iw);
    x_grad_0[idx] = x[idx] * (variance_grad_without_mul[ic] / element_count);
  };
  Loop(func_x_grad_0, n, c, h, w);

  // minus_mean_grad
  std::vector<T> minus_mean_grad(c, 0);
  auto func_minus_mean_grad = [&](int in, int ic, int ih, int iw) {
    minus_mean_grad[ic] += x_mean_diff_grad[offset(in, ic, ih, iw)];
  };
  Loop(func_minus_mean_grad, n, c, h, w);
  for (int ic = 0; ic < c; ++ic) {
    minus_mean_grad[ic] += variance_grad_without_mul[ic] * save_mean[ic];
    minus_mean_grad[ic] /= element_count;
  }

  auto func_x_grad = [=](int in, int ic, int ih, int iw) {
    int idx = offset(in, ic, ih, iw);
    x_grad->at(idx) =
        x_mean_diff_grad[idx] + x_grad_0[idx] - minus_mean_grad[ic];
  };
  Loop(func_x_grad, n, c, h, w);
}

TEST(Decomposer, BatchNormGrad) {
  int n = 16, c = 128, h = 14, w = 14;
  int num = n * c * h * w;
  float epsilon = 1e-5;
  NetBuilder net_builder("batch_norm_grad");
  std::vector<std::string> output_names;
  {
    auto y_grad = net_builder.CreateInput(Float(32), {n, c, h, w}, "y_grad");
    auto x = net_builder.CreateInput(Float(32), {n, c, h, w}, "x");
    auto scale = net_builder.CreateInput(Float(32), {c}, "scale");
    auto saved_mean = net_builder.CreateInput(Float(32), {c}, "saved_mean");
    auto saved_variance =
        net_builder.CreateInput(Float(32), {c}, "saved_variance");

    auto outputs = net_builder.BatchNormGrad(
        y_grad, x, scale, saved_mean, saved_variance, epsilon);
    for (auto output : outputs) {
      output_names.push_back(output->id);
    }
  }
  auto program = net_builder.Build();

  auto target = common::DefaultTarget();
  RunDecomposer(&program,
                target,
                cinn::frontend::DefaultTrainingOptimizeOptions().program_passes,
                output_names);

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  hlir::framework::ApplyPass(graph.get(), "OpFusionPass");
  hlir::framework::ApplyPass(graph.get(), "FusionMergePass");

  auto scope = BuildScope(target, graph);
  hlir::framework::CompilationContext context(graph, scope, target);
  hlir::framework::GraphCompiler gc(context);
  auto run_program = gc.Build();

  // set input
  float precision = 1e-3;
  std::vector<float> y_grad(num), x(num), scale(c), saved_mean(c, 0),
      saved_variance(c, 0);
  InitRandomVector(&y_grad, num, 0.0f, 1.0f, precision);
  InitRandomVector(&x, num, 0.0f, 1.0f, precision);
  InitRandomVector(&scale, c, 0.0f, 1.0f, precision);

  Offset offset(n, c, h, w);
  auto func_save_mean = [&](int in, int ic, int ih, int iw) {
    int idx = offset(in, ic, ih, iw);
    saved_mean[ic] += x[idx];
    saved_variance[ic] += x[idx] * x[idx];
  };
  Loop(func_save_mean, n, c, h, w);
  float element_count = static_cast<float>(n * h * w);
  for (int ic = 0; ic < c; ++ic) {
    saved_mean[ic] /= element_count;
    saved_variance[ic] =
        saved_variance[ic] / element_count - saved_mean[ic] * saved_mean[ic];
  }

  std::vector<std::pair<std::string, std::vector<float>>> inputs = {
      {"y_grad", y_grad},
      {"x", x},
      {"scale", scale},
      {"saved_mean", saved_mean},
      {"saved_variance", saved_variance}};
  for (auto& input : inputs) {
    scope->Var<hlir::framework::Tensor>(input.first);
    auto tensor = scope->GetTensor(input.first);
    CopyFromVector(input.second, tensor, target);
  }
  run_program->Execute();

  std::vector<float> x_grad(num), scale_grad(c), bias_grad(c);
  ComputeBatchNormGradRef(y_grad,
                          x,
                          scale,
                          saved_mean,
                          saved_variance,
                          n,
                          c,
                          h,
                          w,
                          &x_grad,
                          &scale_grad,
                          &bias_grad,
                          epsilon);

  std::unordered_map<std::string, std::pair<std::string, std::vector<float>>>
      output_refs = {{"bias_grad", {output_names[2], bias_grad}},
                     {"scale_grad", {output_names[1], scale_grad}},
                     {"x_grad", {output_names[0], x_grad}}};

  for (auto& iter : output_refs) {
    auto output = iter.second;
    auto tensor = scope->GetTensor(output.first);
    std::vector<float> data(tensor->shape().numel());
    CopyToVector(tensor, &data);

    LOG(INFO) << "output[" << iter.first << "], var_name=" << output.first
              << ", shape=" << tensor->shape().data();
    if (iter.first == "x_grad") {
      // TODO(Xreki): fix the precision check of x_grad.
      // CheckOutput<float>(data, output.second, 1e-8, 1e-1);
    } else if (iter.first == "scale_grad") {
      CheckOutput<float>(data, output.second, 1e-8, 1e-2);
    } else {
      CheckOutput<float>(data, output.second);
    }
  }
}

}  // namespace
}  // namespace frontend
}  // namespace cinn
