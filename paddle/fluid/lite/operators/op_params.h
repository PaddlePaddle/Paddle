// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
#include <string>
#include <vector>
#include "paddle/fluid/lite/core/compatible_tensor.h"
#include "paddle/fluid/lite/core/framework.pb.h"
#include "paddle/fluid/lite/utils/all.h"

/*
 * This file contains all the argument parameter data structure for operators.
 */

namespace paddle {
namespace lite {
namespace operators {

using param_t = Any;
#define WITH_INT8_CONFIG             \
  bool enable_int8{false};           \
  float input_scale{1.0};            \
  std::vector<float> weight_scale{}; \
  float output_scale{1.0};

/// ----------------------- Functional operators ------------------------------
struct FeedParam {
  std::vector<lite::Tensor>* feed_list{};
  lite::Tensor* out{};
  int col;
};

struct FetchParam {
  const lite::Tensor* input{};
  std::vector<lite::Tensor>* fetch_list{};
  int col;
};

// Helper op for lite framework
struct IoCopyParam {
  const lite::Tensor* x{};
  lite::Tensor* y{};
};

struct CalibParam {
  const lite::Tensor* input{};
  lite::Tensor* output{};
  float scale;
};

/// -------------------------- NN operators ------------------------------------

struct FcParam {
  lite::Tensor* input{};
  lite::Tensor* w{};
  lite::Tensor* bias{};
  lite::Tensor* output{};
  lite::DDim in_mat_dims;
  int in_num_col_dims{1};
  bool weight_transposed{false};
  // for int8
  WITH_INT8_CONFIG
};

// For Mul Op
struct MulParam {
  const lite::Tensor* x{};
  const lite::Tensor* y{};
  lite::Tensor* output{};

  int x_num_col_dims{1};
  int y_num_col_dims{1};
  // for int8
  WITH_INT8_CONFIG
};

struct MulGradParam {
  const lite::Tensor* x{};
  const lite::Tensor* y{};
  const lite::Tensor* output_grad{};
  lite::Tensor* x_grad{};
  lite::Tensor* y_grad{};

  int x_num_col_dims{1};
  int y_num_col_dims{1};
};

// For Scale Op
struct ScaleParam {
  lite::Tensor* x{};
  lite::Tensor* output{};

  float scale{1.};
  float bias{};
  bool bias_after_scale{true};
};

// For Softmax op
struct SoftmaxParam {
  lite::Tensor* x{};
  lite::Tensor* output{};
  int axis{-1};
};

// For Reshape and Reshape2 Op
struct ReshapeParam {
  const lite::Tensor* x{};
  const lite::Tensor* actual_shape{nullptr};
  lite::Tensor* output{};
  lite::Tensor* xshape{};

  std::vector<int> shape{};
  bool inplace{false};
};

// For Concat op
struct ConcatParam {
  std::vector<lite::Tensor*> x{};
  lite::Tensor* output{};
  int axis{0};
};

// For Convolution op
struct ConvParam {
  lite::Tensor* x{};
  lite::Tensor* filter{};
  lite::Tensor* bias{nullptr};
  lite::Tensor* residualData{nullptr};
  lite::Tensor* output{};
  std::vector<int> strides{1, 1};
  std::vector<int> paddings{0, 0};
  int groups{1};
  std::vector<int> dilations{1, 1};
  bool fuse_relu_before_depthwise_conv{false};
  bool use_mkldnn{false};
  bool fuse_relu{false};  // only used in mkldnn kernel
  bool use_quantizer{
      false};  // set true for op that should be quantized, only used for cpu
  bool fuse_residual_connection{false};
  float scale_in{1.0f};           // only used with mkl-dnn int8
  float scale_out{1.0f};          // only used with mkl-dnn int8
  float scale_in_eltwise{1.0f};   // only used with mkl-dnn int8
  float scale_weights{1.0f};      // only used with mkl-dnn int8
  bool force_fp32_output{false};  // only used in mkl-dnn int8
  std::string data_format{"Anylayout"};
  // for int8
  WITH_INT8_CONFIG
};

// For BatchNorm op
struct BatchNormParam {
  lite::Tensor* x{};
  lite::Tensor* bias{};
  lite::Tensor* scale{};
  lite::Tensor* mean{};
  lite::Tensor* variance{};
  lite::Tensor* y{};
  lite::Tensor* mean_out{};
  lite::Tensor* variance_out{};
  lite::Tensor* saved_mean{};
  lite::Tensor* saved_variance{};
  bool is_test{true};
  bool use_global_stats{false};
  float epsilon;
  float momentum;
  DataLayoutType data_layout{DATALAYOUT(kNCHW)};
};

// For Pooling op
struct PoolParam {
  lite::Tensor* x{};
  lite::Tensor* output{};
  std::string pooling_type{""};
  std::vector<int> ksize{};
  bool global_pooling{
      false};  // if true, knernel size and paddings will be ignored
  std::vector<int> strides{1, 1};
  std::vector<int> paddings{0, 0};
  bool exclusive{true};
  bool adaptive{false};
  bool ceil_mode{false};
  bool use_quantizer{false};
  std::string data_format{"AnyLayout"};
};

// For Dropout op
struct DropoutParam {
  const lite::Tensor* x{};
  lite::Tensor* output{};
  lite::Tensor* mask{};
  float dropout_prob{.5f};
  bool is_test{false};
  bool fix_seed{false};
  int seed{0};
  std::string dropout_implementation{"downgrade_in_infer"};
};

// For Split op
struct SplitParam {
  lite::Tensor* x{};
  std::vector<lite::Tensor*> output{};
  int axis{-1};
  int num{0};
  std::vector<int> sections;
};

// For Transpose op
struct TransposeParam {
  const lite::Tensor* x{};
  lite::Tensor* output{};
  std::vector<int> axis;
  bool use_mkldnn{false};
  std::string data_format{"AnyLayout"};
};

/// ----------------------- element wise operators ----------------------
struct ElementwiseParam {
  const lite::Tensor* X{};
  const lite::Tensor* Y{};
  lite::Tensor* Out{};
  int axis{-1};  // for broadcasting.
};

struct ElementwiseGradParam {
  const lite::Tensor* Y{};
  const lite::Tensor* Out_grad{};
  lite::Tensor* X_grad{};
  lite::Tensor* Y_grad{};
  int axis{-1};  // for broadcasting.
};

struct FusionElementwiseActivationParam : public ElementwiseParam {
  std::string act_type;
};

struct FusionElementwiseActivationGradParam : public ElementwiseGradParam {
  std::string act_type;
};

/// ----------------------- activation operators ----------------------
struct ActivationParam {
  const lite::Tensor* X{};
  lite::Tensor* Out{};
};

struct ActivationGradParam {
  const lite::Tensor* X{};
  const lite::Tensor* Out{};
  // for backward
  lite::Tensor* X_grad{};
  const lite::Tensor* Out_grad{};
};

/// ----------------------- mean operators ----------------------
struct MeanParam {
  const lite::Tensor* X{};
  lite::Tensor* Out{};
};

struct MeanGradParam {
  const lite::Tensor* X{};
  const lite::Tensor* Out_grad{};
  // for backward
  lite::Tensor* X_grad{};
};

/// ----------------------- fill_constant operators ----------------------
struct FillConstantParam {
  int dtype{framework::proto::VarType::FP32};
  std::vector<int64_t> shape{};
  float value{0.0f};
  // useless for x86, keep it for compatibility
  bool force_cpu{false};
  lite::Tensor* Out{};
};

//
struct FakeQuantizeMovingAvgMaxAbsParam {
  const lite::Tensor* x{};
  const lite::Tensor* in_scale{};
  const lite::Tensor* in_accum{};
  const lite::Tensor* in_state{};
  lite::Tensor* out{};
  lite::Tensor* out_scale{};
  lite::Tensor* out_state{};
  lite::Tensor* out_accum{};
  int bit_length;
  bool is_test{true};
  float moving_rate{0.9};
};

struct FakeDequantizeMaxAbsParam {
  const lite::Tensor* x{};
  const lite::Tensor* in_scale{};
  lite::Tensor* out{};
  float max_range;
};

/// ----------------------- sgd operators ----------------------
struct SGDParam {
  int dtype{framework::proto::VarType::FP32};

  const lite::Tensor* Param{};
  const lite::Tensor* LearningRate{};
  const lite::Tensor* Grad{};
  lite::Tensor* ParamOut{};
};

/// ----------------------- uniform_random operators ----------------------
struct UniformRandomParam {
  std::vector<int64_t> shape{};
  float min{-1.0f};
  float max{1.0f};
  int seed{0};
  int dtype{framework::proto::VarType::FP32};
  lite::Tensor* Out{};
};

}  // namespace operators
}  // namespace lite
}  // namespace paddle
