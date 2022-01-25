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
#include <random>
#include <vector>

#include "gtest/gtest.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/operators/fused/attn_feed_forward.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/platform/float16.h"

namespace framework = paddle::framework;
namespace platform = paddle::platform;

USE_OP(matmul);
USE_OP(elementwise_add);

// get paddle matmul op results as baseline
template <typename T>
void GetLinearOp(const std::vector<T> &x, const std::vector<T> &y,
                 const framework::DDim &x_dim, const framework::DDim &y_dim,
                 const platform::CUDADeviceContext &ctx, bool transpose_a,
                 bool transpose_b, float alpha, std::vector<T> *out) {
  framework::Scope scope;
  auto var_x = scope.Var("X");
  auto tensor_x = var_x->GetMutable<framework::LoDTensor>();
  auto var_y = scope.Var("Y");
  auto tensor_y = var_y->GetMutable<framework::LoDTensor>();
  auto var_out = scope.Var("Out");
  auto tensor_out = var_out->GetMutable<framework::LoDTensor>();

  tensor_x->Resize(x_dim);
  tensor_y->Resize(y_dim);
  tensor_out->Resize({x_dim[0], x_dim[1], y_dim[0]});

  auto x_ptr = tensor_x->mutable_data<T>(ctx.GetPlace());
  auto y_ptr = tensor_y->mutable_data<T>(ctx.GetPlace());
  auto z_ptr = tensor_out->mutable_data<T>(ctx.GetPlace());
  auto size_x = static_cast<size_t>(framework::product(x_dim));
  auto size_y = static_cast<size_t>(framework::product(y_dim));
  auto size_z = x_dim[0] * x_dim[1] * y_dim[0];
  cudaMemcpy(x_ptr, x.data(), size_x * sizeof(T), cudaMemcpyHostToDevice);
  cudaMemcpy(y_ptr, y.data(), size_y * sizeof(T), cudaMemcpyHostToDevice);

  framework::AttributeMap attrs;
  attrs.insert({"transpose_X", transpose_a});
  attrs.insert({"transpose_Y", transpose_b});
  attrs.insert({"alpha", alpha});

  auto op = framework::OpRegistry::CreateOp(
      "matmul", {{"X", {"X"}}, {"Y", {"Y"}}}, {{"Out", {"Out"}}}, attrs);
  op->Run(scope, ctx.GetPlace());

  cudaMemcpy(out->data(), z_ptr, size_z * sizeof(T), cudaMemcpyDeviceToHost);
  ctx.Wait();
}

// get paddle elementwise_add op results as baseline
template <typename T>
void GetElementwiseAddOp(const std::vector<T> &x, const std::vector<T> &y,
                         const int bsz_seq, const int output_size,
                         const platform::CUDADeviceContext &ctx,
                         std::vector<T> *out) {
  framework::Scope scope;
  auto var_x = scope.Var("X");
  auto tensor_x = var_x->GetMutable<framework::LoDTensor>();
  auto var_y = scope.Var("Y");
  auto tensor_y = var_y->GetMutable<framework::LoDTensor>();
  auto var_out = scope.Var("Out");
  auto tensor_out = var_out->GetMutable<framework::LoDTensor>();

  tensor_x->Resize({bsz_seq, output_size});
  tensor_y->Resize({output_size});
  tensor_out->Resize({bsz_seq, output_size});

  auto x_ptr = tensor_x->mutable_data<T>(ctx.GetPlace());
  auto y_ptr = tensor_y->mutable_data<T>(ctx.GetPlace());
  auto z_ptr = tensor_out->mutable_data<T>(ctx.GetPlace());
  auto size_x = bsz_seq * output_size;
  auto size_y = output_size;
  auto size_z = bsz_seq * output_size;
  cudaMemcpy(x_ptr, x.data(), size_x * sizeof(T), cudaMemcpyHostToDevice);
  cudaMemcpy(y_ptr, y.data(), size_y * sizeof(T), cudaMemcpyHostToDevice);

  framework::AttributeMap attrs;
  auto op = framework::OpRegistry::CreateOp("elementwise_add",
                                            {{"X", {"X"}}, {"Y", {"Y"}}},
                                            {{"Out", {"Out"}}}, attrs);
  op->Run(scope, ctx.GetPlace());
  cudaMemcpy(out->data(), z_ptr, size_z * sizeof(T), cudaMemcpyDeviceToHost);
  ctx.Wait();
}

// get paddle matmul_grad op results as baseline
template <typename T>
void GetLinearOpGrad(const std::vector<T> &x_vec, const std::vector<T> &y_vec,
                     const std::vector<T> &dout_vec,
                     const framework::DDim &x_dim, const framework::DDim &y_dim,
                     const framework::DDim &out_dim,
                     const platform::CUDADeviceContext &ctx, bool transpose_a,
                     bool transpose_b, float alpha, std::vector<T> *dinput_vec,
                     std::vector<T> *dweight_vec) {
  framework::Scope scope;
  auto var_x = scope.Var("X");
  auto tensor_x = var_x->GetMutable<framework::LoDTensor>();
  auto var_y = scope.Var("Y");
  auto tensor_y = var_y->GetMutable<framework::LoDTensor>();
  auto var_dout = scope.Var("DOut");
  auto tensor_dout = var_dout->GetMutable<framework::LoDTensor>();
  tensor_x->Resize(x_dim);
  tensor_y->Resize(y_dim);
  tensor_dout->Resize(out_dim);

  auto var_dx = scope.Var("DX");
  auto tensor_dx = var_dx->GetMutable<framework::LoDTensor>();
  auto var_dy = scope.Var("DY");
  auto tensor_dy = var_dy->GetMutable<framework::LoDTensor>();
  tensor_dx->Resize(x_dim);
  tensor_dy->Resize(y_dim);

  auto x_ptr = tensor_x->mutable_data<T>(ctx.GetPlace());
  auto y_ptr = tensor_y->mutable_data<T>(ctx.GetPlace());
  auto dout_ptr = tensor_dout->mutable_data<T>(ctx.GetPlace());
  auto dinput_ptr = tensor_dx->mutable_data<T>(ctx.GetPlace());
  auto dweight_ptr = tensor_dy->mutable_data<T>(ctx.GetPlace());

  auto size_x = static_cast<size_t>(framework::product(x_dim));
  auto size_y = static_cast<size_t>(framework::product(y_dim));
  auto size_z = x_dim[0] * x_dim[1] * y_dim[0];
  cudaMemcpy(x_ptr, x_vec.data(), size_x * sizeof(T), cudaMemcpyHostToDevice);
  cudaMemcpy(y_ptr, y_vec.data(), size_y * sizeof(T), cudaMemcpyHostToDevice);
  cudaMemcpy(dout_ptr, dout_vec.data(), size_z * sizeof(T),
             cudaMemcpyHostToDevice);

  bool use_mkldnn = false;
  std::vector<int> fused_reshape_X = {};
  std::vector<int> fused_reshape_Y = {};
  std::vector<int> fused_reshape_Out = {};
  std::vector<int> fused_transpose_X = {};
  std::vector<int> fused_transpose_Y = {};
  std::vector<int> fused_transpose_Out = {};
  bool use_quantizer = false, force_fp32_output = false;
  std::string mkldnn_data_type = "float32";
  float Scale_x = 1.0, Scale_y = 1.0, Scale_out = 1.0;

  framework::AttributeMap attrs;
  attrs.insert({"transpose_X", transpose_a});
  attrs.insert({"transpose_Y", transpose_b});
  attrs.insert({"alpha", alpha});
  attrs.insert({"use_mkldnn", use_mkldnn});
  attrs.insert({"fused_reshape_X", fused_reshape_X});
  attrs.insert({"fused_reshape_Y", fused_reshape_Y});
  attrs.insert({"fused_reshape_Out", fused_reshape_Out});
  attrs.insert({"fused_transpose_X", fused_transpose_X});
  attrs.insert({"fused_transpose_Y", fused_transpose_Y});
  attrs.insert({"fused_transpose_Out", fused_transpose_Out});
  attrs.insert({"use_quantizer", use_quantizer});
  attrs.insert({"mkldnn_data_type", mkldnn_data_type});
  attrs.insert({"Scale_x", Scale_x});
  attrs.insert({"Scale_y", Scale_y});
  attrs.insert({"Scale_out", Scale_out});
  attrs.insert({"force_fp32_output", force_fp32_output});

  auto op = framework::OpRegistry::CreateOp(
      "matmul_grad", {{"Out@GRAD", {"DOut"}}, {"X", {"X"}}, {"Y", {"Y"}}},
      {{"X@GRAD", {"DX"}}, {"Y@GRAD", {"DY"}}}, attrs);
  op->Run(scope, ctx.GetPlace());

  cudaMemcpy(dinput_vec->data(), dinput_ptr, size_x * sizeof(T),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(dweight_vec->data(), dweight_ptr, size_y * sizeof(T),
             cudaMemcpyDeviceToHost);
  ctx.Wait();
}

// get paddle elementwise_add_grad op results as baseline
template <typename T>
void GetElementwiseAddOpGrad(const std::vector<T> &dout_vec, const int bsz_seq,
                             const int output_size,
                             const platform::CUDADeviceContext &ctx,
                             std::vector<T> *dy_vec) {
  framework::Scope scope;
  auto var_x = scope.Var("X");
  auto tensor_x = var_x->GetMutable<framework::LoDTensor>();
  auto var_y = scope.Var("Y");
  auto tensor_y = var_y->GetMutable<framework::LoDTensor>();
  auto var_dout = scope.Var("DOut");
  auto tensor_dout = var_dout->GetMutable<framework::LoDTensor>();
  tensor_x->Resize({bsz_seq, output_size});
  tensor_y->Resize({output_size});
  tensor_dout->Resize({bsz_seq, output_size});

  auto var_dx = scope.Var("DX");
  auto tensor_dx = var_dx->GetMutable<framework::LoDTensor>();
  auto var_dy = scope.Var("DY");
  auto tensor_dy = var_dy->GetMutable<framework::LoDTensor>();
  tensor_dx->Resize({bsz_seq, output_size});
  tensor_dy->Resize({output_size});

  auto dout_ptr = tensor_dout->mutable_data<T>(ctx.GetPlace());
  auto tensor_dy_ptr = tensor_dy->mutable_data<T>(ctx.GetPlace());
  auto size_z = static_cast<size_t>(bsz_seq * output_size);
  cudaMemcpy(dout_ptr, dout_vec.data(), size_z * sizeof(T),
             cudaMemcpyHostToDevice);

  int axis = -1;
  bool use_mkldnn = false, use_quantizer = false;
  std::string mkldnn_data_type = "float32";
  std::string x_data_format = "", y_data_format = "";
  float Scale_x = 1.0, Scale_y = 1.0, Scale_out = 1.0;

  framework::AttributeMap attrs;
  attrs.insert({"axis", axis});
  attrs.insert({"use_mkldnn", use_mkldnn});
  attrs.insert({"x_data_format", x_data_format});
  attrs.insert({"y_data_format", y_data_format});
  attrs.insert({"use_quantizer", use_quantizer});
  attrs.insert({"mkldnn_data_type", mkldnn_data_type});
  attrs.insert({"Scale_x", Scale_x});
  attrs.insert({"Scale_y", Scale_y});
  attrs.insert({"Scale_out", Scale_out});

  auto op = framework::OpRegistry::CreateOp(
      "elementwise_add_grad",
      {{"Out@GRAD", {"DOut"}}, {"X", {"X"}}, {"Y", {"Y"}}},
      {{"X@GRAD", {"DX"}}, {"Y@GRAD", {"DY"}}}, attrs);
  op->Run(scope, ctx.GetPlace());

  auto size_y = static_cast<size_t>(output_size);
  cudaMemcpy(dy_vec->data(), tensor_dy_ptr, size_y * sizeof(T),
             cudaMemcpyDeviceToHost);
  ctx.Wait();
}

template <typename T>
class TestFeedForward {
 public:
  TestFeedForward() {
    batch_size_ = 16;
    seq_len_ = 128;
    num_head_ = 16;
    dim_head_ = 64;
    dim_embed_ = 1024;
    has_bias_ = false;
  }

  TestFeedForward(int batch_size, int seq_len, int num_head, int dim_head,
                  int dim_embed, bool has_bias) {
    batch_size_ = batch_size;
    seq_len_ = seq_len;
    num_head_ = num_head;
    dim_head_ = dim_head;
    dim_embed_ = dim_embed;
    has_bias_ = has_bias;
  }

  ~TestFeedForward() { delete ctx_; }

  void SetUp() {
    bsz_seq_ = batch_size_ * seq_len_;
    output_size_ = 3 * num_head_ * dim_head_;
    input_size_ = dim_embed_;
    ctx_ = new platform::CUDADeviceContext(place_);

    size_src_ = bsz_seq_ * dim_embed_;         // src: [bs, seq_len, em_dim]
    size_weight_ = dim_embed_ * output_size_;  // weight: [output_size, em_dim]
    size_output_ =
        bsz_seq_ * output_size_;  // output: [bs, seq_len, output_size]
    size_bias_ = output_size_;

    base_out_vec_.resize(size_output_);
    base_bias_out_vec_.resize(size_output_);
    base_dinput_vec_.resize(size_src_);
    base_dweight_vec_.resize(size_weight_);
    base_dbias_vec_.resize(size_bias_);

    src_vec_.resize(size_src_);
    weight_vec_.resize(size_weight_);
    bias_vec_.resize(size_bias_);
    doutput_vec_.resize(size_output_);

    std::default_random_engine random(time(NULL));
    std::uniform_real_distribution<float> dis(0.0, 1.0);
    for (int i = 0; i < size_src_; i++) {
      src_vec_[i] = static_cast<T>(dis(random));
    }
    for (int i = 0; i < size_weight_; i++) {
      weight_vec_[i] = static_cast<T>(dis(random));
    }
    for (int i = 0; i < size_bias_; i++) {
      bias_vec_[i] = static_cast<T>(dis(random));
    }
    for (int i = 0; i < size_output_; i++) {
      doutput_vec_[i] = static_cast<T>(dis(random));
    }

    framework::TensorFromVector<T>(src_vec_, *ctx_, &src_);
    src_.Resize({batch_size_, seq_len_, dim_embed_});
    framework::TensorFromVector<T>(weight_vec_, *ctx_, &weight_);
    weight_.Resize({output_size_, dim_embed_});
    out_.Resize({batch_size_, seq_len_, output_size_});
    out_.mutable_data<T>(place_);
    if (has_bias_) {
      framework::TensorFromVector<T>(bias_vec_, *ctx_, &bias_);
      bias_.Resize({output_size_});
      bias_out_.Resize({batch_size_, seq_len_, output_size_});
      bias_out_.mutable_data<T>(place_);
    }
    framework::TensorFromVector<T>(doutput_vec_, *ctx_, &doutput_);
    doutput_.Resize({batch_size_, seq_len_, output_size_});

    dinput_.Resize({batch_size_, seq_len_, dim_embed_});
    dinput_.mutable_data<T>(place_);
    dweight_.Resize({output_size_, dim_embed_});
    dweight_.mutable_data<T>(place_);
    if (has_bias_) {
      dbias_.Resize({output_size_});
      dbias_.mutable_data<T>(place_);
    }
  }

  void BaselineForward() {
    bool transpose_a = false, transpose_b = true;
    float alpha = 1;
    GetLinearOp(src_vec_, weight_vec_, src_.dims(), weight_.dims(), *ctx_,
                transpose_a, transpose_b, alpha, &base_out_vec_);
    if (has_bias_) {
      GetElementwiseAddOp(base_out_vec_, bias_vec_, bsz_seq_, output_size_,
                          *ctx_, &base_bias_out_vec_);
    }
    ctx_->Wait();
  }

  // get forward results of feedforward.
  void FusedForward() {
    T *p_weight = weight_.data<T>();
    T *p_src = src_.data<T>();
    T *p_output = out_.data<T>();

    T *p_bias = nullptr;
    T *p_bias_output = nullptr;
    if (has_bias_) {
      p_bias = bias_.data<T>();
      p_bias_output = bias_out_.data<T>();
    }
    auto qkv_compute = paddle::operators::FeedForward<T>(
        *ctx_, bsz_seq_, output_size_, input_size_, has_bias_);
    qkv_compute.ComputeForward(p_weight, p_src, p_bias, p_output,
                               p_bias_output);
    ctx_->Wait();
  }

  void BaselineBackward() {
    bool transpose_a = false, transpose_b = true;
    float alpha = 1;

    GetLinearOpGrad(src_vec_, weight_vec_, doutput_vec_, src_.dims(),
                    weight_.dims(), out_.dims(), *ctx_, transpose_a,
                    transpose_b, alpha, &base_dinput_vec_, &base_dweight_vec_);
    if (has_bias_) {
      GetElementwiseAddOpGrad(doutput_vec_, bsz_seq_, output_size_, *ctx_,
                              &base_dbias_vec_);
    }
    ctx_->Wait();
  }

  // get backward results of feedforward.
  void FusedBackward() {
    T *p_weight = weight_.data<T>();
    T *p_src = src_.data<T>();
    T *p_doutput = doutput_.data<T>();
    T *p_dinput = dinput_.data<T>();
    T *p_dweight = dweight_.data<T>();

    T *bias_ptr = nullptr;
    if (has_bias_) {
      bias_ptr = dbias_.data<T>();
    }
    auto qkv_compute = paddle::operators::FeedForward<T>(
        *ctx_, bsz_seq_, output_size_, input_size_, has_bias_);
    qkv_compute.ComputeBackward(p_src, p_weight, p_doutput, p_dinput, p_dweight,
                                bias_ptr);
    ctx_->Wait();
  }

  void Run() {
    SetUp();
    BaselineForward();
    FusedForward();
    BaselineBackward();
    FusedBackward();
  }

  // check forward correctness between baseline and results of feedforward.
  void CheckOut(const T diff, bool is_relative_atol = false) {
    std::vector<T> out(size_output_);
    std::vector<T> bias_out(size_output_);
    paddle::framework::TensorToVector(out_, *ctx_, &out);
    if (has_bias_) {
      paddle::framework::TensorToVector(bias_out_, *ctx_, &bias_out);
    }
    ctx_->Wait();

    for (int i = 0; i < size_output_; i++) {
      if (is_relative_atol) {
        EXPECT_LT(std::abs((out[i] - base_out_vec_[i]) / base_out_vec_[i]),
                  diff);
      } else {
        EXPECT_LT(std::abs(out[i] - base_out_vec_[i]), diff);
      }
      if (has_bias_) {
        if (is_relative_atol) {
          EXPECT_LT(std::abs((bias_out[i] - base_bias_out_vec_[i]) /
                             base_bias_out_vec_[i]),
                    diff);
        } else {
          EXPECT_LT(std::abs(bias_out[i] - base_bias_out_vec_[i]), diff);
        }
      }
    }
  }

  // check backward correctness between baseline and results of feedforward.
  void CheckGrad(const T diff, bool is_relative_atol = false) {
    std::vector<T> h_dinput(size_src_);
    paddle::framework::TensorToVector(dinput_, *ctx_, &h_dinput);
    for (int i = 0; i < size_src_; i++) {
      if (is_relative_atol) {
        EXPECT_LT(
            std::abs((h_dinput[i] - base_dinput_vec_[i]) / base_dinput_vec_[i]),
            diff);
      } else {
        EXPECT_LT(std::abs(h_dinput[i] - base_dinput_vec_[i]), diff);
      }
    }
    std::vector<T> h_dweight(size_weight_);
    paddle::framework::TensorToVector(dweight_, *ctx_, &h_dweight);
    for (int i = 0; i < size_weight_; i++) {
      if (is_relative_atol) {
        EXPECT_LT(std::abs((h_dweight[i] - base_dweight_vec_[i]) /
                           base_dweight_vec_[i]),
                  diff);
      } else {
        EXPECT_LT(std::abs(h_dweight[i] - base_dweight_vec_[i]), diff);
      }
    }
    if (has_bias_) {
      std::vector<T> h_dbias(size_bias_);
      paddle::framework::TensorToVector(dbias_, *ctx_, &h_dbias);
      for (int i = 0; i < size_bias_; i++) {
        if (is_relative_atol) {
          EXPECT_LT(
              std::abs((h_dbias[i] - base_dbias_vec_[i]) / base_dbias_vec_[i]),
              diff);
        } else {
          EXPECT_LT(std::abs(h_dbias[i] - base_dbias_vec_[i]), diff);
        }
      }
    }
  }

 private:
  int batch_size_, seq_len_, num_head_, dim_head_, dim_embed_;
  int bsz_seq_, output_size_, input_size_;
  bool has_bias_;
  int size_src_, size_weight_, size_bias_, size_output_;

  framework::Tensor src_, weight_, bias_, out_, bias_out_;
  framework::Tensor dinput_, dweight_, dbias_, doutput_;
  std::vector<T> src_vec_, weight_vec_, bias_vec_, out_vec_, bias_out_vec_;
  std::vector<T> dinput_vec_, dweight_vec_, dbias_vec_, doutput_vec_;

  // results of baseline.
  std::vector<T> base_out_vec_, base_bias_out_vec_;
  std::vector<T> base_dinput_vec_, base_dweight_vec_, base_dbias_vec_;

  platform::CUDAPlace place_;
  platform::CUDADeviceContext *ctx_;
};

// test for fp32, fp16, fp32+bias and fp16+bias
TEST(FeedForward, GPUFeedforwardBertLargeSizeFp32) {
  int batch_size = 16;
  int seq_len = 128;
  int num_head = 16;
  int dim_head = 64;
  int dim_embed = 1024;
  bool has_bias = false;
  TestFeedForward<float> test(batch_size, seq_len, num_head, dim_head,
                              dim_embed, has_bias);
  test.Run();
  test.CheckOut(static_cast<float>(1e-5));
  test.CheckGrad(static_cast<float>(1e-5));
}

TEST(FeedForward, GPUFeedforwardBertLargeSizeFp16) {
  int batch_size = 16;
  int seq_len = 128;
  int num_head = 16;
  int dim_head = 64;
  int dim_embed = 1024;
  bool has_bias = false;
  TestFeedForward<paddle::platform::float16> test(
      batch_size, seq_len, num_head, dim_head, dim_embed, has_bias);
  test.Run();
  test.CheckOut(static_cast<paddle::platform::float16>(1e-5));
  test.CheckGrad(static_cast<paddle::platform::float16>(1e-5));
}

TEST(FeedForward, GPUFeedforwardBertLargeSizeFp32Bias) {
  int batch_size = 16;
  int seq_len = 128;
  int num_head = 16;
  int dim_head = 64;
  int dim_embed = 1024;
  bool has_bias = true;
  TestFeedForward<float> test(batch_size, seq_len, num_head, dim_head,
                              dim_embed, has_bias);
  test.Run();
  test.CheckOut(static_cast<float>(1e-5));
  test.CheckGrad(static_cast<float>(1e-3));
}

TEST(FeedForward, GPUFeedforwardBertLargeSizeFp16Bias) {
  int batch_size = 16;
  int seq_len = 128;
  int num_head = 16;
  int dim_head = 64;
  int dim_embed = 1024;
  bool has_bias = true;
  TestFeedForward<paddle::platform::float16> test(
      batch_size, seq_len, num_head, dim_head, dim_embed, has_bias);
  test.Run();
  test.CheckOut(static_cast<paddle::platform::float16>(1e-2));
  test.CheckGrad(static_cast<paddle::platform::float16>(1e-2), true);
}
