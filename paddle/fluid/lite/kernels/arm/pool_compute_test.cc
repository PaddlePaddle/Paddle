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

#include "paddle/fluid/lite/kernels/arm/pool_compute.h"
#include <gtest/gtest.h>
#include <limits>
#include <string>
#include <vector>
#include "paddle/fluid/lite/arm/math/funcs.h"
#include "paddle/fluid/lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

int PoolOutputSize(int input_size, int filter_size, int padding, int stride,
                   bool ceil_mode) {
  int output_size;
  if (!ceil_mode) {
    output_size = (input_size - filter_size + 2 * padding) / stride + 1;
  } else {
    output_size =
        (input_size - filter_size + 2 * padding + stride - 1) / stride + 1;
  }
  return output_size;
}

std::vector<int64_t> compute_output_shape(operators::PoolParam* param_) {
  const auto x_dims = param_->x->dims();
  std::vector<int>& ksize = param_->ksize;
  if (param_->global_pooling) {
    ksize.resize(static_cast<size_t>(x_dims.size()) - 2);
    for (size_t i = 0; i < ksize.size(); ++i) {
      param_->paddings[i] = 0;
      ksize[i] = static_cast<int>(x_dims[i + 2]);
    }
  }

  std::vector<int64_t> output_shape({x_dims[0], x_dims[1]});
  if (param_->adaptive) {
    output_shape.insert(output_shape.end(), param_->ksize.begin(),
                        param_->ksize.end());
  } else {
    for (size_t i = 0; i < param_->ksize.size(); ++i) {
      output_shape.push_back(
          PoolOutputSize(x_dims[i + 2], param_->ksize[i], param_->paddings[i],
                         param_->strides[i], param_->ceil_mode));
    }
  }
  return output_shape;
}

void pool_compute_ref(const operators::PoolParam& param) {
  auto& in_dims = param.x->dims();
  auto& out_dims = param.output->dims();

  const float* src_ptr = param.x->data<const float>();
  float* dst_ptr = param.output->mutable_data<float>();

  std::vector<int> ksize = param.ksize;
  std::vector<int> strides = param.strides;
  std::vector<int> paddings = param.paddings;

  std::string pooling_type = param.pooling_type;
  bool global_pooling = param.global_pooling;
  bool exclusive = param.exclusive;
  bool adaptive = param.adaptive;
  bool ceil_mode = param.ceil_mode;
  bool use_quantizer = param.use_quantizer;
  std::string data_format = param.data_format;

  int in_n = in_dims[0];
  int in_c = in_dims[1];
  int in_h = in_dims[2];
  int in_w = in_dims[3];
  int size_in_n = in_c * in_h * in_w;
  int size_in_c = in_h * in_w;

  int out_h = out_dims[2];
  int out_w = out_dims[3];
  int size_out_n = in_c * out_h * out_w;
  int size_out_c = out_h * out_w;

  int window_h = ksize[0];
  int window_w = ksize[1];
  int stride_h = strides[0];
  int stride_w = strides[1];
  int pad_h = paddings[0];
  int pad_w = paddings[1];

  if (global_pooling == true) {
    ksize[0] = in_h;
    ksize[1] = in_w;

    for (int n = 0; n < in_n; ++n) {
      for (int c = 0; c < in_c; ++c) {
        const float* src = src_ptr + n * in_c * in_h * in_w + c * in_h * in_w;
        float res = src[0];
        if (pooling_type == "max") {
          for (int i = 1; i < in_h * in_w; ++i) {
            float cur_val = src[i];
            res = cur_val > res ? cur_val : res;
          }
        } else if (pooling_type == "avg") {
          for (int i = 1; i < in_h * in_w; ++i) {
            float cur_val = src[i];
            res += cur_val;
          }
          res /= (in_h * in_w);
        }
        dst_ptr[n * in_c * out_h * out_w + c] = res;
      }
    }
    return;
  }

  for (int ind_n = 0; ind_n < in_n; ++ind_n) {
    for (int ind_c = 0; ind_c < in_c; ++ind_c) {
      for (int ind_h = 0; ind_h < out_h; ++ind_h) {
        int sh = ind_h * stride_h;
        int eh = sh + window_h;
        sh = (sh - pad_h) < 0 ? 0 : sh - pad_h;
        eh = (eh - pad_h) > in_h ? in_h : eh - pad_h;

        for (int ind_w = 0; ind_w < out_w; ++ind_w) {
          int sw = ind_w * stride_w;
          int ew = sw + window_w;
          sw = (sw - pad_w) < 0 ? 0 : sw - pad_w;
          ew = (ew - pad_w) > in_w ? in_w : ew - pad_w;

          float result = static_cast<float>(0);

          int dst_ind =
              ind_n * size_out_n + ind_c * size_out_c + ind_h * out_w + ind_w;

          for (int kh = sh; kh < eh; ++kh) {
            for (int kw = sw; kw < ew; ++kw) {
              int src_ind =
                  ind_n * size_in_n + ind_c * size_in_c + kh * in_w + kw;

              if (kh == sh && kw == sw) {
                result = src_ptr[src_ind];
              } else {
                if (pooling_type == "max") {
                  result =
                      result >= src_ptr[src_ind] ? result : src_ptr[src_ind];
                }
                if (pooling_type == "avg" && exclusive == false) {
                  // Pooling_average_include_padding
                  result += src_ptr[src_ind];
                }
                if (pooling_type == "avg" && exclusive == true) {
                  // Pooling_average_include_padding
                  result += src_ptr[src_ind];
                }
              }
            }
          }
          if (pooling_type == "avg" && exclusive == false) {
            // Pooling_average_include_padding
            // result /= param.window_h * param.window_w;
            // LOG(ERROR)<<"cpu"<<param.window_h * param.window_w;
            int bh = window_h;
            int bw = window_w;
            if (ew == in_w) {
              bw = sw + window_w >= in_w + pad_w ? in_w + pad_w : sw + window_w;
              bw -= sw;
            }
            if (eh == in_h) {
              bh = sh + window_h >= in_h + pad_h ? in_h + pad_h : sh + window_h;
              bh -= sh;
            }
            result /= bh * bw;
          }
          if (pooling_type == "avg" && exclusive == true) {
            // Pooling_average_exclude_padding
            result /= (ew - sw) * (eh - sh);
          }
          dst_ptr[dst_ind] = result;
        }
      }
    }
  }
}

TEST(pool_arm, init) {
  PoolCompute pool;
  ASSERT_EQ(pool.precision(), PRECISION(kFloat));
  ASSERT_EQ(pool.target(), TARGET(kARM));
}

TEST(pool_arm, compute) {
  PoolCompute pool;
  operators::PoolParam param;

  lite::Tensor x;
  lite::Tensor output;
  lite::Tensor output_ref;

  for (auto pooling_type : {"avg", "max"}) {
    for (auto global_pooling : {true}) {
      // for (auto ksize: {3}) { // TODO(yuanshuai): ksize enable 2, 3
      for (auto stride : {1, 2}) {
        for (auto pad : {0, 1}) {
          for (auto n : {1, 3, 4, 11}) {
            for (auto c : {1, 3, 11 /* ,1024 */}) {  // speedup for ci
              for (auto h : {2, 3, 4, 11}) {
                for (auto w : {2, 3, 4, 11}) {
                  LOG(INFO) << "n:" << n << " c:" << c << " h:" << h
                            << " w:" << w  // << " ksize:" << ksize
                            << " stride:" << stride << " pad:" << pad
                            << " pooling_type:" << pooling_type
                            << " global_pooling:" << global_pooling;

                  // init x, output
                  x.Resize(DDim(std::vector<int64_t>({n, c, h, w})));
                  auto* x_data = x.mutable_data<float>();
                  for (int i = 0; i < x.dims().production(); ++i) {
                    x_data[i] = i;
                  }

                  // fill param
                  param.x = &x;
                  param.output = &output;
                  param.pooling_type = pooling_type;
                  // param.ksize = {ksize, ksize}; //TODO(yuanshuai): ksize
                  // enable
                  param.ksize = {h, w};
                  param.global_pooling = global_pooling;
                  param.strides = {stride, stride};
                  param.paddings = {pad, pad};
                  param.exclusive = true;
                  param.adaptive = false;
                  param.ceil_mode = false;
                  param.use_quantizer = false;

                  const std::vector<int64_t>& output_shape =
                      compute_output_shape(&param);
                  output.Resize(DDim(output_shape));
                  output_ref.Resize(DDim(output_shape));

                  // compute
                  pool.SetParam(param);
                  pool.Run();

                  // compute ref
                  param.output = &output_ref;
                  pool_compute_ref(param);

                  // compare
                  auto* output_data = output.mutable_data<float>();
                  auto* output_ref_data = output_ref.mutable_data<float>();
                  for (int i = 0; i < output.dims().production(); i++) {
                    EXPECT_NEAR(output_data[i], output_ref_data[i], 1e-5);
                    float tmp = output_data[i] - output_ref_data[i];
                    tmp = tmp < 0 ? -tmp : tmp;
                    if (tmp > 1e-5) {
                      std::cout << "output_data[0]:" << output_data[0]
                                << std::endl;
                      std::cout << "output_ref_data[0]:" << output_ref_data[0]
                                << std::endl;
                      std::cout
                          << "x.dims().production():" << x.dims().production()
                          << std::endl;
                      for (int ii = 0; ii < x.dims().production(); ++ii) {
                        std::cout << x_data[ii] << " ";
                      }
                      std::cout;
                      exit(0);
                    }
                  }

                  VLOG(3) << "compare pass";
                }
              }
            }
          }
        }  // pad
      }    // stride
           //} // ksize TODO(yuanshuai): ksize enable
    }      // global_pooling
  }        // pooling_type
}

TEST(pool, retrive_op) {
  auto pool = KernelRegistry::Global().Create<TARGET(kARM), PRECISION(kFloat)>(
      "pool2d");
  ASSERT_FALSE(pool.empty());
  ASSERT_TRUE(pool.front());
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

USE_LITE_KERNEL(pool2d, kARM, kFloat, kNCHW, def);
