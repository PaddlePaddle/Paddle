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

#ifndef ANAKIN_SABER_FUNCS_IMPL_ARM_NEON_IMPL_CONV_ARM_IMPL_H
#include "saber/core/context.h"
#include "saber/core/tensor.h"
#include "saber/saber_funcs_param.h"

#ifdef USE_ARM_PLACE

namespace anakin {

namespace saber {

void conv_3x3s1_direct_fp32(const float* din, float* dout, int num, int chout,
                            int hout, int wout, int chin, int hin, int win,
                            const float* weights, const float* bias,
                            ConvParam<ARM>& param, Context<ARM>* ctx);

void conv_3x3s1_direct_int8(const int8_t* din, int32_t* dout, int num,
                            int chout, int hout, int wout, int chin, int hin,
                            int win, const int8_t* weights, const int32_t* bias,
                            ConvParam<ARM>& param, Context<ARM>* ctx,
                            DataType out_type, const float* scale);

void conv_3x3s1_direct_int7(const int8_t* din, int32_t* dout, int num,
                            int chout, int hout, int wout, int chin, int hin,
                            int win, const int8_t* weights, const int32_t* bias,
                            ConvParam<ARM>& param, Context<ARM>* ctx,
                            DataType out_type, const float* scale);

void conv_3x3s2_direct_fp32(const float* din, float* dout, int num, int chout,
                            int hout, int wout, int chin, int hin, int win,
                            const float* weights, const float* bias,
                            ConvParam<ARM>& param, Context<ARM>* ctx);

int conv_3x3s2_direct_int8_c_num();
void conv_3x3s2_direct_int8(const int8_t* din, int32_t* dout, int num,
                            int chout, int hout, int wout, int chin, int hin,
                            int win, const int8_t* weights, const int32_t* bias,
                            ConvParam<ARM>& param, Context<ARM>* ctx,
                            DataType out_type, const float* scale);

void conv_1x5s1_direct(const void* din, void* dout, int num, int chout,
                       int hout, int wout, int chin, int hin, int win,
                       const void* weights, const void* bias, int group,
                       int kernel_w, int kernel_h, int stride_w, int stride_h,
                       int dila_w, int dila_h, int pad_w, int pad_h,
                       bool flag_bias, bool flag_relu, Context<ARM>& ctx,
                       void* work_space, const void* idx_ptr);

void conv_5x1s1_direct(const void* din, void* dout, int num, int chout,
                       int hout, int wout, int chin, int hin, int win,
                       const void* weights, const void* bias, int group,
                       int kernel_w, int kernel_h, int stride_w, int stride_h,
                       int dila_w, int dila_h, int pad_w, int pad_h,
                       bool flag_bias, bool flag_relu, Context<ARM>& ctx,
                       void* work_space, const void* idx_ptr);

void conv1x1s1_gemm(const float* din, float* dout, int num, int chout, int hout,
                    int wout, int chin, int hin, int win, const float* weights,
                    const float* bias, ConvParam<ARM>& param, Context<ARM>* ctx,
                    const int* idx_ptr);

void conv1x1s1_gemm_int8(const int8_t* din, int32_t* dout, int num, int chout,
                         int hout, int wout, int chin, int hin, int win,
                         const int8_t* weights, const int32_t* bias,
                         ConvParam<ARM>& param, Context<ARM>* ctx,
                         DataType out_type, const float* scale,
                         const int32_t* idx_ptr);

void conv_im2col_gemm(const float* din, float* dout, int num, int chout,
                      int hout, int wout, int chin, int hin, int win,
                      const float* weights, const float* bias,
                      ConvParam<ARM>& param, Context<ARM>* ctx,
                      const int* idx_ptr);

void conv_im2col_gemm_int8(const int8_t* din, int32_t* dout, int num, int chout,
                           int hout, int wout, int chin, int hin, int win,
                           const int8_t* weights, const int32_t* bias,
                           ConvParam<ARM>& param, Context<ARM>* ctx,
                           DataType out_type, const float* scale,
                           const int32_t* idx_ptr);

/**
 * \brief depthwise convolution, kernel size 3x3, stride 1, pad 1, with bias
 */
void conv_depthwise_3x3(const float* din, float* dout, int num, int chout,
                        int hout, int wout, int chin, int hin, int win,
                        const float* weights, const float* bias,
                        ConvParam<ARM>& param, Context<ARM>* ctx);

void conv_depthwise_3x3_int8(const int8_t* din, int32_t* dout, int num,
                             int chout, int hout, int wout, int chin, int hin,
                             int win, const int8_t* weights,
                             const int32_t* bias, ConvParam<ARM>& param,
                             Context<ARM>* ctx, DataType out_type,
                             const float* scale);

void conv_depthwise_3x3_int7(const int8_t* din, int32_t* dout, int num,
                             int chout, int hout, int wout, int chin, int hin,
                             int win, int8_t* weights, const int32_t* bias,
                             ConvParam<ARM>& param, Context<ARM>* ctx,
                             DataType out_type, const float* scale);

void conv_depthwise_5x5(const float* din, float* dout, int num, int chout,
                        int hout, int wout, int chin, int hin, int win,
                        const float* weights, const float* bias,
                        ConvParam<ARM>& param, Context<ARM>* ctx);

void conv_depthwise_5x5_int8(const int8_t* din, int32_t* dout, int num,
                             int chout, int hout, int wout, int chin, int hin,
                             int win, const int8_t* weights,
                             const int32_t* bias, ConvParam<ARM>& param,
                             Context<ARM>* ctx, DataType out_type,
                             const float* scale);

void conv_arm_winograd3x3(const float* din, float* dout, int num, int chout,
                          int hout, int wout, int chin, int hin, int win,
                          const float* weights, const float* bias,
                          ConvParam<ARM>& param, Context<ARM>* ctx);

void winograd_transform_weights(void* dout, const void* din, int ch_out,
                                int ch_in, void* work_space);

void compute_offset(int* idx_out, int h, int w, int kernel_h, int kernel_w,
                    int height, int width, int pad_h, int pad_w, int dilation_h,
                    int dilation_w);

void fill_bias(float* tensor, const float* bias, int channel, int channel_size);

void fill_bias_int8(int* tensor, const int* bias, int channel,
                    int channel_size);

}  // namespace saber

}  // namespace anakin

#endif  // USE_ARM_PLACE

#endif  // ANAKIN_SABER_FUNCS_IMPL_ARM_NEON_IMPL_CONV_ARM_IMPL_H
