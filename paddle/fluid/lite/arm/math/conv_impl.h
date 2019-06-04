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

#include "paddle/fluid/lite/core/context.h"
#include "paddle/fluid/lite/core/target_wrapper.h"
#include "paddle/fluid/lite/operators/op_params.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {

// TODO(TJ): move to somewhere else common
template <TargetType TType, PrecisionType PType, typename Param>
class ImplBase {
 public:
  ImplBase() {}
  virtual ~ImplBase() {}

  virtual bool create(const Param& param, Context<TType>* ctx) { return false; }

  virtual bool init(const Param& param, Context<TType>* ctx) { return false; }

  virtual bool run(Param& param) { return false; }
  // void set_op_name(const char* name){_op_name = name;}
  // const char* get_op_name() { return _op_name.c_str();}

 protected:
  Param* param_;
  Context<TType>* ctx_;
};

void conv_3x3s1_direct_fp32(const float* din, float* dout, int num, int chout,
                            int hout, int wout, int chin, int hin, int win,
                            const float* weights, const float* bias,
                            const operators::ConvParam& param,
                            Context<TARGET(kARM)>* ctx);

void conv_3x3s1_direct_int8(const int8_t* din, int32_t* dout, int num,
                            int chout, int hout, int wout, int chin, int hin,
                            int win, const int8_t* weights, const int32_t* bias,
                            const operators::ConvParam& param,
                            Context<TARGET(kARM)>* ctx, PrecisionType out_type,
                            const float* scale);

void conv_3x3s1_direct_int7(const int8_t* din, int32_t* dout, int num,
                            int chout, int hout, int wout, int chin, int hin,
                            int win, const int8_t* weights, const int32_t* bias,
                            const operators::ConvParam& param,
                            Context<TARGET(kARM)>* ctx, PrecisionType out_type,
                            const float* scale);

void conv_3x3s2_direct_fp32(const float* din, float* dout, int num, int chout,
                            int hout, int wout, int chin, int hin, int win,
                            const float* weights, const float* bias,
                            const operators::ConvParam& param,
                            Context<TARGET(kARM)>* ctx);

int conv_3x3s2_direct_int8_c_num();

void conv_3x3s2_direct_int8(const int8_t* din, int32_t* dout, int num,
                            int chout, int hout, int wout, int chin, int hin,
                            int win, const int8_t* weights, const int32_t* bias,
                            const operators::ConvParam& param,
                            Context<TARGET(kARM)>* ctx, PrecisionType out_type,
                            const float* scale);

void conv_1x5s1_direct(const void* din, void* dout, int num, int chout,
                       int hout, int wout, int chin, int hin, int win,
                       const void* weights, const void* bias, int group,
                       int kernel_w, int kernel_h, int stride_w, int stride_h,
                       int dila_w, int dila_h, int pad_w, int pad_h,
                       bool flag_bias, bool flag_relu,
                       Context<TARGET(kARM)>& ctx, void* work_space,
                       const void* idx_ptr);

void conv_5x1s1_direct(const void* din, void* dout, int num, int chout,
                       int hout, int wout, int chin, int hin, int win,
                       const void* weights, const void* bias, int group,
                       int kernel_w, int kernel_h, int stride_w, int stride_h,
                       int dila_w, int dila_h, int pad_w, int pad_h,
                       bool flag_bias, bool flag_relu,
                       Context<TARGET(kARM)>& ctx, void* work_space,
                       const void* idx_ptr);

void conv1x1s1_gemm(const float* din, float* dout, int num, int chout, int hout,
                    int wout, int chin, int hin, int win, const float* weights,
                    const float* bias, const operators::ConvParam& param,
                    Context<TARGET(kARM)>* ctx, const int* idx_ptr);

void conv1x1s1_gemm_int8(const int8_t* din, int32_t* dout, int num, int chout,
                         int hout, int wout, int chin, int hin, int win,
                         const int8_t* weights, const int32_t* bias,
                         const operators::ConvParam& param,
                         Context<TARGET(kARM)>* ctx, PrecisionType out_type,
                         const float* scale, const int32_t* idx_ptr);

void conv_im2col_gemm(const float* din, float* dout, int num, int chout,
                      int hout, int wout, int chin, int hin, int win,
                      const float* weights, const float* bias,
                      const operators::ConvParam& param,
                      Context<TARGET(kARM)>* ctx, const int* idx_ptr);

void conv_im2col_gemm_int8(const int8_t* din, int32_t* dout, int num, int chout,
                           int hout, int wout, int chin, int hin, int win,
                           const int8_t* weights, const int32_t* bias,
                           const operators::ConvParam& param,
                           Context<TARGET(kARM)>* ctx, PrecisionType out_type,
                           const float* scale, const int32_t* idx_ptr);

/**
 * \brief depthwise convolution, kernel size 3x3, stride 1, pad 1, with bias
 */
void conv_depthwise_3x3(const float* din, float* dout, int num, int chout,
                        int hout, int wout, int chin, int hin, int win,
                        const float* weights, const float* bias,
                        const operators::ConvParam& param,
                        Context<TARGET(kARM)>* ctx);

void conv_depthwise_3x3_int8(const int8_t* din, int32_t* dout, int num,
                             int chout, int hout, int wout, int chin, int hin,
                             int win, const int8_t* weights,
                             const int32_t* bias,
                             const operators::ConvParam& param,
                             Context<TARGET(kARM)>* ctx, PrecisionType out_type,
                             const float* scale);

void conv_depthwise_3x3_int7(const int8_t* din, int32_t* dout, int num,
                             int chout, int hout, int wout, int chin, int hin,
                             int win, int8_t* weights, const int32_t* bias,
                             const operators::ConvParam& param,
                             Context<TARGET(kARM)>* ctx, PrecisionType out_type,
                             const float* scale);

void conv_depthwise_5x5(const float* din, float* dout, int num, int chout,
                        int hout, int wout, int chin, int hin, int win,
                        const float* weights, const float* bias,
                        const operators::ConvParam& param,
                        Context<TARGET(kARM)>* ctx);

void conv_depthwise_5x5_int8(const int8_t* din, int32_t* dout, int num,
                             int chout, int hout, int wout, int chin, int hin,
                             int win, const int8_t* weights,
                             const int32_t* bias,
                             const operators::ConvParam& param,
                             Context<TARGET(kARM)>* ctx, PrecisionType out_type,
                             const float* scale);

void conv_arm_winograd3x3(const float* din, float* dout, int num, int chout,
                          int hout, int wout, int chin, int hin, int win,
                          const float* weights, const float* bias,
                          const operators::ConvParam& param,
                          Context<TARGET(kARM)>* ctx);

void winograd_transform_weights(void* dout, const void* din, int ch_out,
                                int ch_in, void* work_space);

void compute_offset(int* idx_out, int h, int w, int kernel_h, int kernel_w,
                    int height, int width, int pad_h, int pad_w, int dilation_h,
                    int dilation_w);

void fill_bias(float* tensor, const float* bias, int channel, int channel_size);

void fill_bias_int8(int* tensor, const int* bias, int channel,
                    int channel_size);

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
