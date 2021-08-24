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
#include "paddle/fluid/operators/fused/fused_residual_dropout_bias.h"

namespace framework = paddle::framework;
namespace platform = paddle::platform;

/**
 * @brief the unittest of fused_residual_dropout_bias
 * 1. random input data
 * 2. add bias, call paddle dropout op, add residual, and get the base result
 * 3. call FusedResidualDropoutBias function get fused result
 * 4. compare ther base result and fused result
 */

template <typename T>
struct TestFusedResidualDropoutBias {
  uint32_t _rows;
  uint32_t _cols;
  uint64_t _seed;
  float _dropout_prob;
  bool _is_upscale_in_train;
  bool _is_test;  // default false,  Set to true for inference only
  bool _has_bias = true;
  framework::Tensor _src, _residual, _bias, _out, _mask;
  framework::Tensor _dsrc, _dbias;

  std::vector<T> _src_vec, _residual_vec, _bias_vec, _out_vec, _mask_vec;
  std::vector<T> _correct_out, _correct_dsrc, _correct_dbias;
  std::vector<uint8_t> _correct_mask;

  platform::CUDAPlace _place;
  platform::CUDADeviceContext *_ctx;

  TestFusedResidualDropoutBias() {
    _rows = 32;
    _cols = 32;
    _seed = 0;
    _dropout_prob = 0.0;
    _is_upscale_in_train = false;
    _is_test = false;
    _has_bias = true;
    _ctx = new platform::CUDADeviceContext(_place);
  }

  TestFusedResidualDropoutBias(int rows, int cols, uint64_t seed = 0,
                               float dropout_prob = 0.0,
                               bool is_upscale_in_train = false,
                               bool is_test = false) {
    _rows = rows;
    _cols = cols;
    _seed = seed;
    _dropout_prob = dropout_prob;
    _is_upscale_in_train = is_upscale_in_train;
    _is_test = is_test;
    _has_bias = true;
    _ctx = new platform::CUDADeviceContext(_place);
  }

  ~TestFusedResidualDropoutBias() { delete _ctx; }

  void SetUp() {
    const int n = _rows * _cols;
    _correct_out.resize(n);
    _correct_mask.resize(n);
    _correct_dsrc.resize(n);
    _correct_dbias.resize(_cols);

    _src_vec.resize(n);
    _residual_vec.resize(n);
    _bias_vec.resize(_cols);
    std::default_random_engine random(time(NULL));
    std::uniform_real_distribution<float> dis(0.0, 1.0);

    for (int i = 0; i < _rows; i++) {
      for (int j = 0; j < _cols; j++) {
        _src_vec[i * _cols + j] = static_cast<T>(dis(random));
        _residual_vec[i * _cols + j] = static_cast<T>(dis(random));
        if (i == 0) _bias_vec[j] = dis(random);
      }
    }

    framework::TensorFromVector<T>(_src_vec, *_ctx, &_src);
    _src.Resize({_rows, _cols});
    framework::TensorFromVector<T>(_residual_vec, *_ctx, &_residual);
    _residual.Resize({_rows, _cols});
    if (_has_bias) {
      framework::TensorFromVector<T>(_bias_vec, *_ctx, &_bias);
      _bias.Resize({_cols});
    }

    {
      _out.Resize({_rows, _cols});
      _out.mutable_data<T>(_place);
      _mask.Resize({_rows, _cols});
      _mask.mutable_data<uint8_t>(_place);
      _dsrc.Resize({_rows, _cols});
      _dsrc.mutable_data<T>(_place);

      if (_has_bias) {
        _dbias.Resize({_cols});
        _dbias.mutable_data<T>(_place);
      }
    }
  }

  void BaseForward() {
    std::vector<T> out1(_rows * _cols), out2(_rows * _cols);
    if (_has_bias) {
      // add bias
      for (int i = 0; i < _rows; i++) {
        for (int j = 0; j < _cols; j++) {
          out1[i * _cols + j] = _src_vec[i * _cols + j] + _bias_vec[j];
        }
      }
      // call dropout
      Dropout<T>(out1.data(), _src.dims(), out2.data(), &_correct_mask, *_ctx,
                 _seed, _dropout_prob, _is_upscale_in_train, _is_test);
    } else {
      Dropout<T>(_src_vec.data(), _src.dims(), out2.data(), &_correct_mask,
                 *_ctx, _seed, _dropout_prob, _is_upscale_in_train, _is_test);
    }
    // add residual
    for (int i = 0; i < _rows; i++) {
      for (int j = 0; j < _cols; j++) {
        _correct_out[i * _cols + j] =
            _residual_vec[i * _cols + j] + out2[i * _cols + j];
      }
    }
    _ctx->Wait();
  }

  void BaseBackward() {
    DropoutGrad<T>(_correct_dsrc.data(), _src.dims(), _correct_out.data(),
                   _correct_mask.data(), *_ctx, _dropout_prob,
                   _is_upscale_in_train);
    // calc dbias
    memset(&_correct_dbias[0], 0, _cols * sizeof(T));
    for (int i = 0; i < _rows; i++) {
      for (int j = 0; j < _cols; j++) {
        _correct_dbias[j] += _correct_out[i * _cols + j];
      }
    }
  }

  void FusedForward() {
    auto threads = paddle::operators::Get1DBlocksAnd2DGrids(
        *_ctx, (uint64_t)_rows, (uint64_t)_cols);
    const int VecSize = 4;
    const int increment =
        ((_cols - 1) / (threads.first.x * threads.second.x * VecSize) + 1) *
        VecSize;

    T *bias_ptr = nullptr;
    if (_has_bias) {
      bias_ptr = _bias.data<T>();
    }
    if (_is_test) {
      paddle::operators::LaunchResidualDropoutBiasIsTest<T>(
          _rows, _cols, _dropout_prob, _is_upscale_in_train, _src.data<T>(),
          _residual.data<T>(), bias_ptr, _out.data<T>(), *_ctx);
    } else {
      paddle::operators::LaunchResidualDropoutBias<T, uint8_t>(
          _rows, _cols, increment, _seed, _dropout_prob, _is_upscale_in_train,
          _src.data<T>(), _residual.data<T>(), bias_ptr, _mask.data<uint8_t>(),
          _out.data<T>(), *_ctx);
    }
    _ctx->Wait();
  }

  void FusedBackward() {
    if (_is_test) return;

    T *bias_ptr = nullptr;
    if (_has_bias) {
      bias_ptr = _dbias.data<T>();
    }
    paddle::operators::LaunchResidualDropoutBiasGrad<T, uint8_t>(
        _out.data<T>(), _mask.data<uint8_t>(), _dropout_prob,
        _is_upscale_in_train, _rows, _cols, _dsrc.data<T>(), bias_ptr, *_ctx);
  }

  void Run() {
    SetUp();
    BaseForward();
    FusedForward();
    BaseBackward();
    FusedBackward();
  }

  void CheckOut(const T diff) {
    const int n = _rows * _cols;
    std::vector<T> out(n);
    std::vector<uint8_t> mask(n);
    cudaMemcpy(out.data(), _out.data<T>(), _rows * _cols * sizeof(T),
               cudaMemcpyDeviceToHost);
    if (!_is_test) {
      cudaMemcpy(mask.data(), _mask.data<uint8_t>(),
                 _rows * _cols * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    }
    _ctx->Wait();

    for (int i = 0; i < n; i++) {
      EXPECT_LT(std::abs(out[i] - _correct_out[i]), diff);
      if (!_is_test) EXPECT_EQ(mask[i], _correct_mask[i]);
    }
  }

  void CheckGrad(const T diff) {
    if (_is_test) return;

    const int n = _rows * _cols;

    std::vector<T> dsrc(n);
    cudaMemcpy(dsrc.data(), _dsrc.data<T>(), _rows * _cols * sizeof(T),
               cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; i++) {
      EXPECT_LT(std::abs(dsrc[i] - _correct_dsrc[i]), diff);
    }

    if (_has_bias) {
      std::vector<T> dbias(_cols);
      cudaMemcpy(dbias.data(), _dbias.data<T>(), _cols * sizeof(T),
                 cudaMemcpyDeviceToHost);
      _ctx->Wait();
      for (int i = 0; i < _cols; i++) {
        EXPECT_LT(std::abs(dbias[i] - _correct_dbias[i]), diff);
      }
    }
  }
};

TEST(FusedDropout, GPUFusedRedisualDorpoutBias) {
  const int rows = 16;
  const int cols = 16;
  TestFusedResidualDropoutBias<float> test(rows, cols);
  test.Run();
  test.CheckOut(static_cast<float>(1e-5));
  test.CheckGrad(static_cast<float>(1e-5));
}

TEST(FusedDropout, GPUFusedRedisualDorpoutBiasDouble) {
  const int rows = 16;
  const int cols = 16;
  TestFusedResidualDropoutBias<double> test(rows, cols);
  test.Run();
  test.CheckOut(static_cast<double>(1e-5));
  test.CheckGrad(static_cast<double>(1e-5));
}

// test fp16, For inference, check_grad is not required. ref: test_dropout_op.py
TEST(FusedDropout, GPUFusedRedisualDorpoutBiasFp16) {
  const int rows = 16;
  const int cols = 16;
  TestFusedResidualDropoutBias<platform::float16> test(rows, cols);
  test.Run();
  test.CheckOut(static_cast<platform::float16>(1e-2));
}

// test no bias and cols % 4 == 0
TEST(FusedDropout, GPUFusedRedisualDorpoutBiasNoBias) {
  const int rows = 16;
  const int cols = 16;
  TestFusedResidualDropoutBias<float> test(rows, cols);
  test._has_bias = false;
  test.Run();
  test.CheckOut(static_cast<float>(1e-5));
  test.CheckGrad(static_cast<float>(1e-5));
}

// test no bias and cols % 4 != 0
TEST(FusedDropout, GPUFusedRedisualDorpoutBiasNoBias2) {
  const int rows = 16;
  const int cols = 17;
  TestFusedResidualDropoutBias<float> test(rows, cols);
  test._has_bias = false;
  test.Run();
  test.CheckOut(static_cast<float>(1e-5));
  test.CheckGrad(static_cast<float>(1e-5));
}

// test add bias and cols % 4 != 0
TEST(FusedDropout, GPUFusedRedisualDorpoutBias2) {
  const int rows = 16;
  const int cols = 17;
  TestFusedResidualDropoutBias<float> test(rows, cols);
  test.Run();
  test.CheckOut(static_cast<float>(1e-5));
  test.CheckGrad(static_cast<float>(1e-5));
}

TEST(FusedDropout, GPUFusedRedisualDorpoutBias3) {
  const int rows = 16;
  const int cols = 16;
  TestFusedResidualDropoutBias<float> test(rows, cols, 0, 1.0, false, false);
  test.Run();
  test.CheckOut(static_cast<float>(1e-5));
  test.CheckGrad(static_cast<float>(1e-5));
}

TEST(FusedDropout, GPUFusedRedisualDorpoutBias4) {
  const int rows = 16;
  const int cols = 16;
  TestFusedResidualDropoutBias<float> test(rows, cols, 0, 1.0, false, false);
  test.Run();
  test.CheckOut(static_cast<float>(1e-5));
  test.CheckGrad(static_cast<float>(1e-5));
}

TEST(FusedDropout, GPUFusedRedisualDorpoutBias5) {
  const int rows = 16;
  const int cols = 16;
  TestFusedResidualDropoutBias<float> test(rows, cols, 0, 1.0, true, false);
  test.Run();
  test.CheckOut(static_cast<float>(1e-5));
  test.CheckGrad(static_cast<float>(1e-5));
}

TEST(FusedDropout, GPUFusedRedisualDorpoutBias6) {
  const int rows = 16;
  const int cols = 16;
  TestFusedResidualDropoutBias<float> test(rows, cols, 0, 0.35, true, true);
  test.Run();
  test.CheckOut(static_cast<float>(1e-5));
  test.CheckGrad(static_cast<float>(1e-5));
}

TEST(FusedDropout, GPUFusedRedisualDorpoutBias7) {
  const int rows = 16;
  const int cols = 16;
  TestFusedResidualDropoutBias<float> test(rows, cols, 125, 0.0, false, false);
  test.Run();
  test.CheckOut(static_cast<float>(1e-5));
  test.CheckGrad(static_cast<float>(1e-5));
}
