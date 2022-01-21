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

#include <time.h>

#include <random>
#include <vector>

#include "paddle/fluid/operators/fused/fused_dropout_test.h"
#include "paddle/fluid/operators/fused/fused_layernorm_residual_dropout_bias.h"

/**
 * @brief The unit test of fused_layernorm_residual_dropout_bias
 */

template <typename T>
struct TestFusedLayernormResidualDropoutBias {
  uint32_t rows;
  uint32_t cols;
  uint64_t seed;
  float dropout_prob, epsilon;
  bool is_upscale_in_train;
  bool is_test;  // default false,  Set to true for inference only
  bool has_bias = true;
  bool has_scale = true;
  bool has_layernorm_bias = true;
  framework::Tensor src, residual, bias, out, mask, scale, layernorm_bias,
      layernorm_out, means, vars;
  framework::Tensor dsrc, dbias;

  std::vector<T> src_vec, residual_vec, bias_vec;
  std::vector<LayerNormParamType<T>> means_vec, vars_vec, scale_vec,
      layernorm_bias_vec;
  std::vector<T> correct_out, correct_dsrc, correct_dbias,
      correct_layernorm_out;
  std::vector<LayerNormParamType<T>> correct_means, correct_vars;
  std::vector<uint8_t> correct_mask;

  platform::CUDAPlace place;
  platform::CUDADeviceContext *ctx;

  TestFusedLayernormResidualDropoutBias() {
    rows = 32;
    cols = 32;
    seed = 0;
    dropout_prob = 0.0;
    is_upscale_in_train = false;
    is_test = false;
    has_bias = true;
    has_scale = true;
    has_layernorm_bias = true;
    epsilon = 0.00001f;
    platform::DeviceContextPool &pool = platform::DeviceContextPool::Instance();
    auto devicectx = pool.Get(place);
    ctx = reinterpret_cast<platform::CUDADeviceContext *>(devicectx);
  }

  TestFusedLayernormResidualDropoutBias(int _rows, int _cols,
                                        uint64_t _seed = 0,
                                        float _dropout_prob = 0.0,
                                        float _epsilon = 0.00001f,
                                        bool _is_upscale_in_train = false,
                                        bool _is_test = false) {
    rows = _rows;
    cols = _cols;
    seed = _seed;
    dropout_prob = _dropout_prob;
    epsilon = _epsilon;
    is_upscale_in_train = _is_upscale_in_train;
    is_test = _is_test;
    has_bias = true;
    has_scale = true;
    has_layernorm_bias = true;
    platform::DeviceContextPool &pool = platform::DeviceContextPool::Instance();
    auto devicectx = pool.Get(place);
    ctx = reinterpret_cast<platform::CUDADeviceContext *>(devicectx);
  }

  ~TestFusedLayernormResidualDropoutBias() {}

  void SetUp() {
    using U = LayerNormParamType<T>;
    const int n = rows * cols;
    correct_out.resize(n);
    correct_mask.resize(n);
    correct_dsrc.resize(n);
    correct_dbias.resize(cols);
    correct_means.resize(rows);
    correct_vars.resize(rows);
    correct_layernorm_out.resize(n);

    src_vec.resize(n);
    residual_vec.resize(n);
    if (has_bias) {
      bias_vec.resize(cols);
    }
    if (has_scale) {
      scale_vec.resize(cols);
    }
    if (has_layernorm_bias) {
      layernorm_bias_vec.resize(cols);
    }
    std::default_random_engine random(time(NULL));
    std::uniform_real_distribution<float> dis(0.0, 1.0);

    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        src_vec[i * cols + j] = static_cast<T>(dis(random));
        residual_vec[i * cols + j] = static_cast<T>(dis(random));
        if (i == 0) {
          if (has_bias) {
            bias_vec[j] = static_cast<T>(dis(random));
          }
          if (has_scale) {
            scale_vec[j] = static_cast<U>(dis(random));
          }
          if (has_layernorm_bias) {
            layernorm_bias_vec[j] = static_cast<U>(dis(random));
          }
        }
      }
    }

    framework::TensorFromVector<T>(src_vec, *ctx, &src);
    src.Resize({rows, cols});
    framework::TensorFromVector<T>(residual_vec, *ctx, &residual);
    residual.Resize({rows, cols});
    if (has_bias) {
      framework::TensorFromVector<T>(bias_vec, *ctx, &bias);
      bias.Resize({cols});
    }
    if (has_scale) {
      framework::TensorFromVector<U>(scale_vec, *ctx, &scale);
      scale.Resize({cols});
    }
    if (has_layernorm_bias) {
      framework::TensorFromVector<U>(layernorm_bias_vec, *ctx, &layernorm_bias);
      layernorm_bias.Resize({cols});
    }

    {
      out.Resize({rows, cols});
      out.mutable_data<T>(place);
      mask.Resize({rows, cols});
      mask.mutable_data<uint8_t>(place);
      means.Resize({rows});
      means.mutable_data<U>(place);
      vars.Resize({rows});
      vars.mutable_data<U>(place);
      layernorm_out.Resize({rows, cols});
      layernorm_out.mutable_data<T>(place);
      dsrc.Resize({rows, cols});
      dsrc.mutable_data<T>(place);

      if (has_bias) {
        dbias.Resize({cols});
        dbias.mutable_data<T>(place);
      }
    }
  }

  void BaseForward() {
    using U = LayerNormParamType<T>;
    std::vector<T> out1(rows * cols), out2(rows * cols);
    if (has_bias) {
      // add bias
      for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
          out1[i * cols + j] = src_vec[i * cols + j] + bias_vec[j];
        }
      }
      // call dropout
      Dropout<T>(out1, src.dims(), &out2, &correct_mask, *ctx, seed,
                 dropout_prob, is_upscale_in_train, is_test);
    } else {
      Dropout<T>(src_vec, src.dims(), &out2, &correct_mask, *ctx, seed,
                 dropout_prob, is_upscale_in_train, is_test);
    }
    // add residual
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        correct_out[i * cols + j] =
            residual_vec[i * cols + j] + out2[i * cols + j];
      }
    }

    LayerNorm<T>(scale_vec, layernorm_bias_vec, correct_out, &correct_means,
                 &correct_vars, &correct_layernorm_out, epsilon, rows, cols,
                 *ctx);
    ctx->Wait();
  }

  void FusedForward() {
    using U = LayerNormParamType<T>;
    int VecSize = MAX_CACHE_BYTES / sizeof(T);
    if (cols % 4 != 0) {
      VecSize = 1;
    }
    int threads = paddle::operators::GetDesiredBlockDim(cols / VecSize);
    const int increment = ((cols - 1) / (threads * VecSize) + 1) * VecSize;

    T *bias_ptr = nullptr;
    U *scale_ptr = nullptr;
    U *layernorm_bias_ptr = nullptr;
    if (has_bias) {
      bias_ptr = bias.data<T>();
    }
    if (has_scale) {
      scale_ptr = scale.data<U>();
    }
    if (has_layernorm_bias) {
      layernorm_bias_ptr = layernorm_bias.data<U>();
    }

    paddle::operators::LaunchLayernormResidualDropoutBias<T, uint8_t, U, false>(
        rows, cols, increment, seed, dropout_prob, epsilon, is_upscale_in_train,
        is_test, src.data<T>(), residual.data<T>(), bias_ptr, scale_ptr,
        layernorm_bias_ptr, mask.data<uint8_t>(), out.data<T>(),
        layernorm_out.data<T>(), means.data<U>(), vars.data<U>(), *ctx);
    ctx->Wait();
  }

  void Run() {
    SetUp();
    BaseForward();
    FusedForward();
  }

  void CheckOut(const T diff) {
    using U = LayerNormParamType<T>;
    const int n = rows * cols;
    std::vector<T> _out(n), _layernorm_out(n);
    std::vector<U> _means(rows), _vars(cols);
    std::vector<uint8_t> _mask(n);
    framework::TensorToVector(out, *ctx, &_out);
    framework::TensorToVector(layernorm_out, *ctx, &_layernorm_out);
    framework::TensorToVector(means, *ctx, &_means);
    framework::TensorToVector(vars, *ctx, &_vars);
    if (!is_test) {
      framework::TensorToVector(mask, *ctx, &_mask);
    }
    ctx->Wait();

    for (int i = 0; i < n; i++) {
      EXPECT_LT(std::abs(_out[i] - correct_out[i]), diff);
      EXPECT_LT(std::abs(_layernorm_out[i] - correct_layernorm_out[i]), diff);
      if (!is_test) EXPECT_EQ(_mask[i], correct_mask[i]);
    }
    for (int i = 0; i < rows; i++) {
      EXPECT_LT(std::abs(_means[i] - correct_means[i]), static_cast<U>(diff));
      EXPECT_LT(std::abs(_vars[i] - correct_vars[i]), static_cast<U>(diff));
    }
  }
};

template <typename T>
static void BaseTest(const bool is_fp16 = false) {
  const int rows = 16;
  T default_diff = !is_fp16 ? static_cast<T>(1e-4) : static_cast<T>(1e-2);
  for (auto cols : {16, 17}) {
    for (auto has_bias : {true, false}) {
      for (auto has_scale : {true, false}) {
        for (auto has_layernorm_bias : {true, false}) {
          TestFusedLayernormResidualDropoutBias<T> test(rows, cols);
          test.has_bias = has_bias;
          test.has_scale = has_scale;
          test.has_layernorm_bias = has_layernorm_bias;
          test.Run();
          test.CheckOut(default_diff);
        }
      }
    }
  }
}

TEST(FusedDropout, GPUFusedLayernormResidualDropoutBias) { BaseTest<float>(); }

TEST(FusedDropout, GPUFusedLayernormResidualDropoutBiasDouble) {
  BaseTest<double>();
}

TEST(FusedDropout, GPUFusedLayernormResidualDropoutBiasFp16) {
  BaseTest<platform::float16>(true);
}

TEST(FusedDropout, GPUFusedLayernormResidualDropoutBiasIsUpscaleInTrain) {
  const int rows = 16;
  const int cols = 16;
  for (auto is_upscale_in_train : {true, false}) {
    TestFusedLayernormResidualDropoutBias<float> test(
        rows, cols, 0, 1.0, 0.00001f, is_upscale_in_train, false);
    test.Run();
    test.CheckOut(static_cast<float>(1e-4));
  }
}

TEST(FusedDropout, GPUFusedLayernormResidualDropoutBiasIsTest) {
  const int rows = 16;
  const int cols = 16;
  TestFusedLayernormResidualDropoutBias<float> test(rows, cols, 0, 0.35,
                                                    0.00001f, true, true);
  test.Run();
  test.CheckOut(static_cast<float>(1e-4));
}

TEST(FusedDropout, GPUFusedLayernormResidualDropoutBiasSeed) {
  const int rows = 16;
  const int cols = 16;
  TestFusedLayernormResidualDropoutBias<float> test(rows, cols, 125, 0.0,
                                                    0.00001f, false, false);
  test.Run();
  test.CheckOut(static_cast<float>(1e-4));
}

TEST(FusedDropout, GPUFusedLayernormResidualDropoutLargeShape) {
  const int rows = 512;
  const int cols = 512;
  TestFusedLayernormResidualDropoutBias<float> test(rows, cols);
  test.Run();
  test.CheckOut(static_cast<float>(1e-4));
}
