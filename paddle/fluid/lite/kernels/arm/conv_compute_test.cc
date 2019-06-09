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

#include "paddle/fluid/lite/kernels/arm/conv_compute.h"
#include <gtest/gtest.h>
#include <memory>
#include <utility>
#include <vector>
#include "paddle/fluid/lite/arm/math/funcs.h"
#include "paddle/fluid/lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

template <typename dtype>
void conv_compute_ref(const operators::ConvParam& param) {
  auto input = param.x;
  auto filter = param.filter;
  auto output = param.output;
  DDim input_dims = param.x->dims();
  DDim filter_dims = param.filter->dims();
  DDim output_dims = param.output->dims();
  std::vector<int> paddings = param.paddings;
  std::vector<int> strides = param.strides;
  std::vector<int> dilations = param.dilations;
  int groups = param.groups;

  auto input_data = param.x->data<float>();
  auto output_data = param.output->mutable_data<float>();
  auto filter_data = param.filter->mutable_data<float>();
  const float* bias_data = nullptr;
  if (param.bias != nullptr) {
    bias_data = param.bias->mutable_data<float>();
  }
  bool flag_bias = bias_data != nullptr;
  bool flag_relu = false;  // TODO(hong19860320) param.relu

  int num = input_dims[0];
  int chout = output_dims[1];
  int hout = output_dims[2];
  int wout = output_dims[3];

  int chin = input_dims[1];
  int hin = input_dims[2];
  int win = input_dims[3];
  int out_c_group = chout / groups;
  int in_c_group = chin / groups;

  int stride_h = strides[0];
  int stride_w = strides[1];
  int dilation_h = dilations[0];
  int dilation_w = dilations[1];
  int padding_h = paddings[0];
  int padding_w = paddings[1];
  int kernel_h = filter_dims[2];
  int kernel_w = filter_dims[3];

  for (int n = 0; n < num; ++n) {
    for (int g = 0; g < groups; ++g) {
      for (int oc = 0; oc < out_c_group; ++oc) {
        for (int oh = 0; oh < hout; ++oh) {
          for (int ow = 0; ow < wout; ++ow) {
            int out_idx = n * groups * out_c_group * hout * wout +
                          g * out_c_group * hout * wout + oc * hout * wout +
                          oh * wout + ow;
            output_data[out_idx] = 0.0f;
            for (int ic = 0; ic < in_c_group; ++ic) {
              for (int kh = 0; kh < kernel_h; ++kh) {
                for (int kw = 0; kw < kernel_w; ++kw) {
                  int iw = ow * stride_w - padding_w + kw * (dilation_w);
                  int ih = oh * stride_h - padding_h + kh * (dilation_h);
                  if (iw < 0 || iw >= win) continue;
                  if (ih < 0 || ih >= hin) continue;

                  int iidx = n * chin * hin * win + g * in_c_group * hin * win +
                             ic * hin * win + ih * win + iw;
                  int widx =
                      g * out_c_group * in_c_group * kernel_h * kernel_w +
                      oc * in_c_group * kernel_h * kernel_w +
                      ic * kernel_h * kernel_w + kh * kernel_w + kw;

                  output_data[out_idx] +=
                      (dtype)input_data[iidx] * (dtype)filter_data[widx];
                }
              }
            }
            output_data[out_idx] +=
                flag_bias ? static_cast<float>(bias_data[g * out_c_group + oc])
                          : 0.f;
            if (flag_relu) {
              output_data[out_idx] =
                  output_data[out_idx] > 0.f ? output_data[out_idx] : 0.f;
            }
          }
        }
      }
    }
  }
}

TEST(conv_arm, retrive_op) {
  auto conv =
      KernelRegistry::Global().Create<TARGET(kARM), PRECISION(kFloat)>("conv2d");
  ASSERT_FALSE(conv.empty());
  ASSERT_TRUE(conv.front());
}

TEST(conv_arm, init) {
  ConvCompute conv;
  ASSERT_EQ(conv.precision(), PRECISION(kFloat));
  ASSERT_EQ(conv.target(), TARGET(kARM));
}

TEST(conv_arm, compute) {
  ConvCompute conv;
  operators::ConvParam param;

  lite::Tensor input;
  lite::Tensor filter;
  lite::Tensor bias;
  lite::Tensor output;
  lite::Tensor output_ref;

  DeviceInfo::Init();
  std::unique_ptr<KernelContext> ctx(new KernelContext);
  ctx->As<ARMContext>();
  conv.SetContext(std::move(ctx));
  for (auto n : {1, 2}) {
    for (auto chin : {3, 8, /*32, 128*/}) {
      for (auto chout : {3, 8, /*32, 128*/}) {
        for (auto hin : {7, 14, 28, /*56 , 112, 224, 512*/}) {
          for (auto win : {7, 14, 28, /*56, 112, 224, 512*/}) {
            for (auto flag_bias : {false , true}) {
              for (auto flag_relu : {false , true}) {
                for (auto depthwise : {false, true}) {
                  for (auto dilation : {1 /*, 2*/}) {
                    for (auto stride : {1, 2}) {
                      for (auto padding : {0, 1}) {
                        for (auto ks : {/*1, */3/*, 5*/}) {
                          int group = 1;
                          if (depthwise) {  // depthwise conv ?
                            group = chin;
                            chout = chin;
                            // remove the follow code if
                            // all kernels are implemented.
                            if (ks == 5) {
                              stride = 2;
                              padding = 2;
                            }
                          }
                          // get input, filter and output shape
                          std::vector<int64_t> input_shape = {n, chin, hin,
                                                              win};
                          std::vector<int64_t> filter_shape = {
                              chout, chin / group, ks, ks};
                          std::vector<int64_t> output_shape({n, chout});
                          const int dkernel = dilation * (ks - 1) + 1;
                          output_shape.push_back(
                              (hin + 2 * padding - dkernel) / stride + 1);
                          output_shape.push_back(
                              (win + 2 * padding - dkernel) / stride + 1);
                          // resize input, filter and output
                          input.Resize(DDim(input_shape));
                          filter.Resize(DDim(filter_shape));
                          output.Resize(DDim(output_shape));
                          output_ref.Resize(DDim(output_shape));
                          auto* input_data = input.mutable_data<float>();
                          auto* filter_data = filter.mutable_data<float>();
                          auto* output_data = output.mutable_data<float>();
                          auto* output_ref_data =
                              output_ref.mutable_data<float>();
                          for (int i = 0; i < input.dims().production(); i++) {
                            input_data[i] = static_cast<float>(i % 128);
                          }
                          for (int i = 0; i < filter.dims().production(); i++) {
                            filter_data[i] = i / 1000.0f;
                          }
                          param.x = &input;
                          param.filter = &filter;
                          param.output = &output;
                          param.bias = nullptr;
                          // TODO(hong19860320) param.relu = flag_relu;
                          param.paddings = std::vector<int>({padding, padding});
                          param.strides = std::vector<int>({stride, stride});
                          param.dilations =
                              std::vector<int>({dilation, dilation});
                          param.groups = group;
                          conv.SetParam(param);
                          conv.Run();
                          param.output = &output_ref;
                          conv_compute_ref<float>(param);
                          for (int i = 0; i < output.dims().production(); i++) {
                            EXPECT_NEAR(output_data[i], output_ref_data[i],
                                        1e-3);
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
#if 0
// for testing gemm like conv
  int n = 1;
  int chin = 8;
  int chout = 8;
  int hin = 14;
  int win = 14;
  int flag_bias = false;
  int flag_relu = false;
  int dilation = 1;
  int stride = 1;
  int padding = 1;
  int ks = 5;
  int group = 1;
  // get input, filter and output shape
  std::vector<int64_t> input_shape = {n, chin, hin, win};
  std::vector<int64_t> filter_shape = {chout, chin / group, ks, ks};
  std::vector<int64_t> output_shape({n, chout});
  const int dkernel = dilation * (ks - 1) + 1;
  output_shape.push_back((hin + 2 * padding - dkernel) / stride + 1);
  output_shape.push_back((win + 2 * padding - dkernel) / stride + 1);
  // resize input, filter and output
  input.Resize(DDim(input_shape));
  filter.Resize(DDim(filter_shape));
  output.Resize(DDim(output_shape));
  output_ref.Resize(DDim(output_shape));
  auto* input_data = input.mutable_data<float>();
  auto* filter_data = filter.mutable_data<float>();
  auto* output_data = output.mutable_data<float>();
  auto* output_ref_data = output_ref.mutable_data<float>();
  for (int i = 0; i < input.dims().production(); i++) {
    input_data[i] = static_cast<float>(i % 128);
  }
  for (int i = 0; i < filter.dims().production(); i++) {
    filter_data[i] = i / 1000.0f;
  }
  param.x = &input;
  param.filter = &filter;
  param.output = &output;
  param.bias = nullptr;
  // TODO(hong19860320) param.relu = flag_relu;
  param.paddings = std::vector<int>({padding, padding});
  param.strides = std::vector<int>({stride, stride});
  param.dilations = std::vector<int>({dilation, dilation});
  param.groups = group;
  conv.SetParam(param);
  conv.Run();
  param.output = &output_ref;
  conv_compute_ref<float>(param);
  for (int i = 0; i < output.dims().production(); i++) {
    EXPECT_NEAR(output_data[i], output_ref_data[i], 1e-3);
  }
#endif
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

USE_LITE_KERNEL(conv2d, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(depthwise_conv2d, kARM, kFloat, kNCHW, def);
