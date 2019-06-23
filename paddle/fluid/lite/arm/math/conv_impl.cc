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

// #include "saber/funcs/impl/arm/neon/impl/conv_arm_depthwise.h"
// #include "saber/funcs/impl/arm/neon/impl/conv_arm_impl.h"
// #include "saber/funcs/impl/arm/neon/impl/gemm_prepacked_int8.h"
// #include "saber/funcs/impl/arm/neon/impl/gemv_arm_int8.h"
// #include "saber/funcs/impl/arm/neon/impl/sgemv_arm.h"

#include "paddle/fluid/lite/arm/math/conv_impl.h"
#include <arm_neon.h>
#include "paddle/fluid/lite/arm/math/gemm_prepacked_int8.h"
#include "paddle/fluid/lite/arm/math/gemv_arm_int8.h"
#include "paddle/fluid/lite/arm/math/packed_sgemm.h"
#include "paddle/fluid/lite/arm/math/sgemv.h"
#include "paddle/fluid/lite/core/context.h"
#include "paddle/fluid/lite/core/target_wrapper.h"
#include "paddle/fluid/lite/operators/op_params.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {

/**
 * \brief neon implementation to add bias
 * @param tensor
 * @param bias
 * @param channel
 * @param channel_size
 */
void fill_bias(float* tensor, const float* bias, int channel,
               int channel_size) {
  if (tensor == nullptr) {
    return;
  }
  float* data = tensor;

  for (int j = 0; j < channel; ++j) {
    float32x4_t vdata = vdupq_n_f32(bias[j]);
    int i = 0;
    for (; i < channel_size - 3; i += 4) {
      vst1q_f32(data + i, vdata);
    }
    for (; i < channel_size; i++) {
      data[i] = bias[j];
    }
    data += channel_size;
  }
}

void fill_bias_int8(int* tensor, const int* bias, int channel,
                    int channel_size) {
  if (tensor == nullptr) {
    return;
  }
  int* data = tensor;
  for (int j = 0; j < channel; ++j) {
    int32x4_t vdata = vdupq_n_s32(bias[j]);
    int i = 0;
    for (; i < channel_size - 3; i += 4) {
      vst1q_s32(data + i, vdata);
    }
    for (; i < channel_size; i++) {
      data[i] = bias[j];
    }
    data += channel_size;
  }
}

/**
 * \brief inline funcs used in im2col
 * @param a
 * @param b
 * @return
 */
inline bool is_a_ge_zero_and_a_lt_b(int a, int b) {
  return static_cast<unsigned>(a) < static_cast<unsigned>(b);
}

/**
 * \brief normal im2col function for gemm conv
 * @tparam dtype
 * @param data_im
 * @param channels
 * @param height
 * @param width
 * @param kernel_size
 * @param pad
 * @param stride
 * @param data_col
 */
template <typename Dtype>
void im2col(const Dtype* data_im, const int channels, const int height,
            const int width, const int kernel_h, const int kernel_w,
            const int pad_h, const int pad_w, const int stride_h,
            const int stride_w, const int dilation_h, const int dilation_w,
            Dtype* data_col) {
  const int output_h =
      (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  const int output_w =
      (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  const int channel_size = height * width;
  for (int channel = channels; channel--; data_im += channel_size) {
    for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
      for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
        int input_row = -pad_h + kernel_row * dilation_h;
        for (int output_rows = output_h; output_rows; output_rows--) {
          if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
            for (int output_cols = output_w; output_cols; output_cols--) {
              *(data_col++) = 0;
            }
          } else {
            int input_col = -pad_w + kernel_col * dilation_w;
            for (int output_col = output_w; output_col; output_col--) {
              if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                *(data_col++) = data_im[input_row * width + input_col];
              } else {
                *(data_col++) = 0;
              }
              input_col += stride_w;
            }
          }
          input_row += stride_h;
        }
      }
    }
  }
}
void compute_offset(int* idx_out, int h, int w, int kernel_h, int kernel_w,
                    int height, int width, int pad_h, int pad_w, int dilation_h,
                    int dilation_w) {
  int idx_h[kernel_h];  // NOLINT
  int idx_w[kernel_w];  // NOLINT
  for (int i = 0; i < kernel_h; ++i) {
    idx_h[i] = h - pad_h + i * dilation_h;
  }
  for (int i = 0; i < kernel_w; ++i) {
    idx_w[i] = w - pad_w + i * dilation_w;
  }
  for (int k_h = 0; k_h < kernel_h; ++k_h) {
    for (int k_w = 0; k_w < kernel_w; ++k_w) {
      idx_out[k_h * kernel_w + k_w] =
          (idx_h[k_h] >= 0 && idx_w[k_w] >= 0 && idx_h[k_h] < height &&
           idx_w[k_w] < width)
              ? idx_h[k_h] * width + idx_w[k_w]
              : -1;
    }
  }
}
template <typename Dtype>
void im2col3x3(const Dtype* data_im, const int channels, const int height,
               const int width, const int kernel_h, const int kernel_w,
               const int pad_h, const int pad_w, const int stride_h,
               const int stride_w, const int dilation_h, const int dilation_w,
               Dtype* data_col, const int* idx) {
  const int output_h =
      (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  const int output_w =
      (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  int kernel_stride = kernel_h * kernel_w;
  int in_channel_stride = height * width;
  const int* idx_out = idx;
  Dtype* data_col_ptr = data_col;

  bool flag_continue = false;
  if (dilation_h == 1 && dilation_w == 1) {
    flag_continue = true;
  }

  for (int o = 0; o < output_h * output_w; o += 1) {
    const Dtype* data_im_ptr = data_im;

    // int* idx_out_d = idx_out;

    int idx_out_d0 = idx_out[0];
    int idx_out_d1 = idx_out[1];
    int idx_out_d2 = idx_out[2];
    int idx_out_d3 = idx_out[3];
    int idx_out_d4 = idx_out[4];
    int idx_out_d5 = idx_out[5];
    int idx_out_d6 = idx_out[6];
    int idx_out_d7 = idx_out[7];
    int idx_out_d8 = idx_out[8];

    for (int i = 0; i < channels; i += 1) {
      if (idx_out_d0 >= 0 && idx_out_d2 >= 0 && idx_out_d6 >= 0 &&
          idx_out_d8 >= 0) {
        if (flag_continue) {
          memcpy(data_col_ptr, data_im_ptr + idx_out_d0,
                 kernel_w * sizeof(Dtype));
          memcpy(data_col_ptr + kernel_w, data_im_ptr + idx_out_d3,
                 kernel_w * sizeof(Dtype));
          memcpy(data_col_ptr + kernel_w + kernel_w, data_im_ptr + idx_out_d6,
                 kernel_w * sizeof(Dtype));
        } else {
          data_col_ptr[0] = data_im_ptr[idx_out_d0];
          data_col_ptr[1] = data_im_ptr[idx_out_d1];
          data_col_ptr[2] = data_im_ptr[idx_out_d2];
          data_col_ptr[3] = data_im_ptr[idx_out_d3];
          data_col_ptr[4] = data_im_ptr[idx_out_d4];
          data_col_ptr[5] = data_im_ptr[idx_out_d5];
          data_col_ptr[6] = data_im_ptr[idx_out_d6];
          data_col_ptr[7] = data_im_ptr[idx_out_d7];
          data_col_ptr[8] = data_im_ptr[idx_out_d8];
        }
      } else {
        data_col_ptr[0] = (idx_out_d0 < 0) ? 0 : data_im_ptr[idx_out_d0];
        data_col_ptr[1] = (idx_out_d1 < 0) ? 0 : data_im_ptr[idx_out_d1];
        data_col_ptr[2] = (idx_out_d2 < 0) ? 0 : data_im_ptr[idx_out_d2];
        data_col_ptr[3] = (idx_out_d3 < 0) ? 0 : data_im_ptr[idx_out_d3];
        data_col_ptr[4] = (idx_out_d4 < 0) ? 0 : data_im_ptr[idx_out_d4];
        data_col_ptr[5] = (idx_out_d5 < 0) ? 0 : data_im_ptr[idx_out_d5];
        data_col_ptr[6] = (idx_out_d6 < 0) ? 0 : data_im_ptr[idx_out_d6];
        data_col_ptr[7] = (idx_out_d7 < 0) ? 0 : data_im_ptr[idx_out_d7];
        data_col_ptr[8] = (idx_out_d8 < 0) ? 0 : data_im_ptr[idx_out_d8];
      }
      data_im_ptr += height * width;
      data_col_ptr += kernel_stride;
    }
    // data_col_ptr += channels * kernel_stride;
    // idx_out += kernel_stride * 2;
    idx_out += kernel_stride;
  }
}

/**
 * \brief convolution function for kernel size 1x1, stride size 1, gemm
 * implementation
 */
void conv1x1s1_gemm(const float* i_data, float* o_data, int num, int oc, int oh,
                    int ow, int ic, int ih, int win, const float* weights,
                    const float* bias, const operators::ConvParam& param,
                    ARMContext* ctx, const int* idx_ptr) {
  int channel_size_out = ow * oh;
  int channel_size_in = win * ih;

  const int group = param.groups;
  const int m = oc / group;
  const int n = oh * ow;
  const int k = ic / group;

  bool flag_relu = param.fuse_relu;
  bool flag_bias = param.bias != nullptr;
  // if (param.activation_param.has_active) {
  //   if (param.activation_param.active == Active_relu &&
  //       fabs(param.activation_param.negative_slope) < 1e-6f) {
  //     flag_relu = true;
  //   }
  // }
  int hblock = get_hblock(ctx->arch());
  int m_roundup = hblock * ((m + hblock - 1) / hblock);
  int weights_size_per_group = m * k;
  if (n > 1) {
    weights_size_per_group = ((m_roundup * k + 15) / 16) * 16;
  }

  // int weights_size_per_group = m_roundup * k;//oc * ic / (group *
  // group);
  //! use gemv when the output channel size = 1
  for (int b = 0; b < num; ++b) {
    // dC
    for (int g = 0; g < group; ++g) {
      float* dout_group =
          static_cast<float*>(o_data) + (b * oc + g * m) * channel_size_out;
      const float* din_group = static_cast<const float*>(i_data) +
                               (b * ic + g * k) * channel_size_in;
      const float* weights_group =
          static_cast<const float*>(weights) + g * weights_size_per_group;
      const float* bias_group = static_cast<const float*>(bias) + g * m;

      if (n == 1) {
        sgemv(weights_group, din_group, dout_group, false, m, k, flag_bias,
              bias_group, flag_relu);
      } else {
        sgemm_prepack(weights_group, din_group, bias_group, dout_group, m, n, k,
                      flag_bias, flag_relu, false, ctx);
      }
    }
  }
}

void conv1x1s1_gemm_int8(const int8_t* i_data, int32_t* o_data, int num, int oc,
                         int oh, int ow, int ic, int ih, int win,
                         const int8_t* weights, const int32_t* bias,
                         const operators::ConvParam& param, ARMContext* ctx,
                         PrecisionType out_type, const float* scale,
                         const int32_t* idx_ptr) {
  int group = param.groups;
  int channel_size_out = ow * oh;
  int channel_size_in = win * ih;
  const int m = oc / group;
  const int n = oh * ow;
  const int k = ic / group;
  int hblock = get_hblock_int8(ctx->arch());
  int k_roundup = ROUNDUP(k, KBLOCK_INT8);
  int m_roundup = ROUNDUP(m, hblock);
  int weights_size_per_group = m * k;
  if (n > 1) {
    weights_size_per_group = ((m_roundup * k_roundup + 15) / 16) * 16;
  }
  bool flag_relu = param.fuse_relu;
  bool flag_bias = param.bias != nullptr;
  //! use gemv when the output channel size = 1
  for (int b = 0; b < num; ++b) {
    // dC
    for (int g = 0; g < group; ++g) {
      signed char* dout_group =
          reinterpret_cast<signed char*>(o_data) +
          (b * oc + g * m) * channel_size_out * PrecisionTypeLength(out_type);
      const int8_t* din_group = i_data + (b * ic + g * k) * channel_size_in;
      const int8_t* weights_group = weights + g * weights_size_per_group;
      const int* bias_group = bias + g * m;
      const float* scale_group = scale + g * m;
      if (n == 1) {
        if (out_type == PRECISION(kFloat)) {
          gemv_int8(weights_group, din_group,
                    reinterpret_cast<float*>(dout_group), false, m, k,
                    scale_group, flag_bias, bias_group, flag_relu);
        } else if (out_type == PRECISION(kInt8)) {  // int8
          gemv_int8(weights_group, din_group, dout_group, false, m, k,
                    scale_group, flag_bias, bias_group, flag_relu);
        } else {
          gemv_int8(weights_group, din_group,
                    reinterpret_cast<int*>(dout_group), false, m, k,
                    scale_group, flag_bias, bias_group, flag_relu);
        }
      } else {
        if (out_type == PRECISION(kFloat)) {
          gemm_prepack_int8(weights_group, din_group, bias_group,
                            reinterpret_cast<float*>(dout_group), m, n, k,
                            flag_bias, flag_relu, false, scale_group, ctx);
        } else if (out_type == PRECISION(kInt8)) {  // int8
          gemm_prepack_int8(weights_group, din_group, bias_group, dout_group, m,
                            n, k, flag_bias, flag_relu, false, scale_group,
                            ctx);
        } else {
          gemm_prepack_int8(weights_group, din_group, bias_group,
                            reinterpret_cast<int*>(dout_group), m, n, k,
                            flag_bias, flag_relu, false, scale_group, ctx);
        }
      }
    }
  }
}

/**
 * \brief convolution function for kernel size 3x3, stride size 2, gemm
 * implementation
 */
void conv_im2col_gemm(const float* i_data, float* o_data, int num, int oc,
                      int oh, int ow, int ic, int ih, int win,
                      const float* weights, const float* bias,
                      const operators::ConvParam& param, ARMContext* ctx,
                      const int* idx_ptr) {
  const int group = param.groups;
  auto filter_dims = param.filter->dims();
  const int kernel_h = filter_dims[2];
  const int kernel_w = filter_dims[3];  // nchw
  const int m = oc / group;
  const int n = oh * ow;
  const int k = ic * kernel_h * kernel_w / group;
  const int chin_per_group = ic / group;
  int channel_size_out = ow * oh;
  int channel_size_in = win * ih;
  bool flag_relu = param.fuse_relu;
  bool flag_bias = param.bias != nullptr;
  // if (param.activation_param.has_active) {
  //   if (param.activation_param.active == Active_relu &&
  //       fabs(param.activation_param.negative_slope) < 1e-6f) {
  //     flag_relu = true;
  //   }
  // }
  int hblock = get_hblock(ctx->arch());
  int m_roundup = hblock * ((m + hblock - 1) / hblock);
  int weights_size_per_group = m * k;
  if (n > 1) {
    weights_size_per_group = ((m_roundup * k + 15) / 16) * 16;
  }

  bool flag_im2col2 = (kernel_h == 3 && kernel_w == 3 &&
                       param.strides[0] == 1 && param.strides[1] == 1 && n > 1);

  float* tmp_work_space =
      ctx->workspace_data<float>() + ctx->l2_cache_size() / sizeof(float);

  //! use gemv when the output channel size = 1
  for (int b = 0; b < num; ++b) {
    // dC
    for (int g = 0; g < group; ++g) {
      float* dout_group = o_data + (b * oc + g * m) * channel_size_out;
      const float* din_group =
          i_data + (b * ic + g * chin_per_group) * channel_size_in;
      const float* weights_group = weights + g * weights_size_per_group;
      const float* bias_group = bias + g * m;
      float* dB = tmp_work_space;

      if (flag_im2col2) {
        im2col3x3(din_group, chin_per_group, ih, win, kernel_h, kernel_w,
                  param.paddings[0], param.paddings[1], param.strides[0],
                  param.strides[1], param.dilations[0], param.dilations[1], dB,
                  idx_ptr);
      } else {
        im2col(din_group, chin_per_group, ih, win, kernel_h, kernel_w,
               param.paddings[0], param.paddings[1], param.strides[0],
               param.strides[1], param.dilations[0], param.dilations[1], dB);
      }
      if (n == 1) {
        sgemv(weights_group, dB, dout_group, false, m, k, flag_bias, bias_group,
              flag_relu);
      } else {
        sgemm_prepack(weights_group, dB, bias_group, dout_group, m, n, k,
                      flag_bias, flag_relu, flag_im2col2, ctx);
      }
    }
  }
}

void conv_im2col_gemm_int8(const int8_t* i_data, int32_t* o_data, int num,
                           int oc, int oh, int ow, int ic, int ih, int win,
                           const int8_t* weights, const int32_t* bias,
                           const operators::ConvParam& param, ARMContext* ctx,
                           PrecisionType out_type, const float* scale,
                           const int32_t* idx_ptr) {
  int group = param.groups;
  auto filter_dims = param.filter->dims();
  int kernel_h = filter_dims[2];
  int kernel_w = filter_dims[3];
  int stride_h = param.strides[0];
  int stride_w = param.strides[1];
  int dila_h = param.dilations[0];
  int dila_w = param.dilations[1];
  int pad_h = param.paddings[0];
  int pad_w = param.paddings[1];
  const int m = oc / group;
  const int n = oh * ow;
  const int k = ic * kernel_h * kernel_w / group;
  const int chin_per_group = ic / group;
  int channel_size_out = ow * oh;
  int channel_size_in = win * ih;
  bool flag_relu = param.fuse_relu;
  bool flag_bias = param.bias != nullptr;

  int hblock = get_hblock_int8(ctx->arch());
  int k_roundup = ROUNDUP(k, KBLOCK_INT8);
  int m_roundup = ROUNDUP(m, hblock);
  int weights_size_per_group = m * k;
  if (n > 1) {
    weights_size_per_group = ((m_roundup * k_roundup + 15) / 16) * 16;
  }

  bool flag_im2col2 = (kernel_h == 3 && kernel_w == 3 && stride_h == 1 &&
                       stride_w == 1 && n > 1);

  int8_t* tmp_work_space = ctx->workspace_data<int8_t>() + ctx->l2_cache_size();

  //! use gemv when the output channel size = 1
  for (int b = 0; b < num; ++b) {
    // dC
    for (int g = 0; g < group; ++g) {
      signed char* dout_group =
          reinterpret_cast<signed char*>(o_data) +
          (b * oc + g * m) * channel_size_out * PrecisionTypeLength(out_type);
      const int8_t* din_group = static_cast<const int8_t*>(i_data) +
                                (b * ic + g * chin_per_group) * channel_size_in;
      const int8_t* weights_group =
          static_cast<const int8_t*>(weights) + g * weights_size_per_group;
      const int* bias_group = static_cast<const int*>(bias) + g * m;
      int8_t* dB = tmp_work_space;
      const float* scale_group = scale + g * m;

      if (flag_im2col2) {
        im2col3x3(din_group, chin_per_group, ih, win, kernel_h, kernel_w, pad_h,
                  pad_w, stride_h, stride_w, dila_h, dila_w, dB, idx_ptr);

      } else {
        im2col(din_group, chin_per_group, ih, win, kernel_h, kernel_w, pad_h,
               pad_w, stride_h, stride_w, dila_h, dila_w, dB);
      }
      if (n == 1) {
        if (out_type == PRECISION(kFloat)) {
          gemv_int8(weights_group, dB, reinterpret_cast<float*>(dout_group),
                    false, m, k, scale_group, flag_bias, bias_group, flag_relu);
        } else if (out_type == PRECISION(kInt8)) {  // int8
          gemv_int8(weights_group, dB, dout_group, false, m, k, scale_group,
                    flag_bias, bias_group, flag_relu);
        } else {
          gemv_int8(weights_group, dB, reinterpret_cast<int*>(dout_group),
                    false, m, k, scale_group, flag_bias, bias_group, flag_relu);
        }
      } else {
        if (out_type == PRECISION(kFloat)) {
          gemm_prepack_int8(weights_group, dB, bias_group,
                            reinterpret_cast<float*>(dout_group), m, n, k,
                            flag_bias, flag_relu, flag_im2col2, scale_group,
                            ctx);
        } else if (out_type == PRECISION(kInt8)) {  // int8
          gemm_prepack_int8(weights_group, dB, bias_group, dout_group, m, n, k,
                            flag_bias, flag_relu, flag_im2col2, scale_group,
                            ctx);
        } else {
          gemm_prepack_int8(
              weights_group, dB, bias_group, reinterpret_cast<int*>(dout_group),
              m, n, k, flag_bias, flag_relu, flag_im2col2, scale_group, ctx);
        }
      }
    }
  }
}

void conv_depthwise_3x3(const float* i_data, float* o_data, int num, int oc,
                        int oh, int ow, int ic, int ih, int win,
                        const float* weights, const float* bias,
                        const operators::ConvParam& param, ARMContext* ctx) {
  int pad = param.paddings[1];
  int stride = param.strides[1];
  bool flag_relu = param.fuse_relu;
  bool flag_bias = param.bias != nullptr;
  // if (param.activation_param.has_active) {
  //   if (param.activation_param.active == Active_relu &&
  //       fabs(param.activation_param.negative_slope) < 1e-6f) {
  //     flag_relu = true;
  //   }
  // }
  if (pad == 1) {
    conv_depthwise_3x3p1(i_data, o_data, num, oc, oh, ow, ic, ih, win, weights,
                         bias, stride, flag_bias, flag_relu, ctx);
  } else if (pad == 0 && ih > 2) {
    conv_depthwise_3x3p0(i_data, o_data, num, oc, oh, ow, ic, ih, win, weights,
                         bias, stride, flag_bias, flag_relu, ctx);
  } else {
    LOG(FATAL) << "unsupport this type 3x3 dw conv";
  }
}

void conv_depthwise_5x5(const float* i_data, float* o_data, int num, int oc,
                        int oh, int ow, int ic, int ih, int win,
                        const float* weights, const float* bias,
                        const operators::ConvParam& param, ARMContext* ctx) {
  int pad = param.paddings[1];
  int stride = param.strides[1];
  bool flag_relu = param.fuse_relu;
  bool flag_bias = param.bias != nullptr;
  // if (param.activation_param.has_active &&
  //     fabs(param.activation_param.negative_slope) < 1e-6f) {
  //   if (param.activation_param.active == Active_relu) {
  //     flag_relu = true;
  //   }
  // }
  if (pad == 2 && stride == 2) {
    conv_depthwise_5x5s2(i_data, o_data, num, oc, oh, ow, ic, ih, win, weights,
                         bias, pad, flag_bias, flag_relu, ctx);
  } else if (stride == 1) {
    conv_depthwise_5x5s1(i_data, o_data, num, oc, oh, ow, ic, ih, win, weights,
                         bias, pad, flag_bias, flag_relu, ctx);
  } else {
    LOG(FATAL) << "unsupport this type 5x5 dw conv";
  }
}

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
