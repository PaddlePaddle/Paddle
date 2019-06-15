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

#include <arm_neon.h>
#include "paddle/fluid/lite/arm/math/conv_block_utils.h"
#include "paddle/fluid/lite/arm/math/conv_impl.h"
#include "paddle/fluid/lite/core/context.h"
#include "paddle/fluid/lite/operators/op_params.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {

void conv_depthwise_5x5s1_int8(int32_t* dout, const int8_t* din,
                               const int8_t* weights, const int* bias,
                               bool flag_bias, bool flag_relu, const int num,
                               const int chin, const int hin, const int win,
                               const int hout, const int wout, ARMContext* ctx,
                               PrecisionType out_type, const float* scale);

void conv_depthwise_5x5_int8(const int8_t* din, int32_t* dout, int num,
                             int chout, int hout, int wout, int chin, int hin,
                             int win, const int8_t* weights,
                             const int32_t* bias,
                             const operators::ConvParam& param, ARMContext* ctx,
                             PrecisionType out_type, const float* scale) {
  int stride_h = param.strides[0];
  bool flag_relu = param.fuse_relu;
  bool flag_bias = param.bias != nullptr;
  // if (param.activation_param.has_active){
  //     if (param.activation_param.active == Active_relu ||
  //     fabs(param.activation_param.negative_slope) > 1e-6f){
  //         flag_relu = true;
  //     }
  // }
  if (stride_h == 1) {
#ifdef __aarch64__
    conv_depthwise_5x5s1_int8(dout, din, weights, bias, flag_bias, flag_relu,
                              num, chin, hin, win, hout, wout, ctx, out_type,
                              scale);
#else

    LOG(FATAL) << "5x5 dw conv armv7 has not impl";
#endif
  }
}

/**
 * \brief depthwise convolution, kernel size 5x5, stride 1, pad 1, with bias,
 * width > 4
 */
// 2 line
#ifdef __aarch64__

template <typename Dtype>
inline void prefetch(const Dtype* din) {
#ifdef __aarch64__
  asm volatile("PRFM PLDL1KEEP, [%[din]] \n" : : [din] "r"(din) : "memory");
#else
  asm volatile("pld [%[din]] \n" : : [din] "r"(din) : "memory");
#endif
}

void conv_depthwise_5x5s1_int8(
    int32_t* dout, const int8_t* din, const int8_t* weights,
    const int32_t* bias, bool flag_bias, bool flag_relu, const int num,
    const int chin, const int hin, const int win, const int hout,
    const int wout, ARMContext* ctx, PrecisionType od_type,
    float const* scales) {  /// scale_size = channel-out

  // printf("5*5 multiply\n");
  int size_in_channel = win * hin;
  int size_out_channel = wout * hout;
  int w_stride = 5 * 5;

  static int const stride_w = 1;
  int const stride_h = stride_w;
  int const chout = chin;
  int const pad_w = 2;
  int const pad_h = pad_w;

  int const wout_round = ((wout + 7) / 8) * 8;
  int const win_round = wout_round * stride_w + 5 - 1;
  int const hout_round = ((hout + 2) / 3) * 3;
  int const hin_round = hout_round * stride_h + 5 - 1;
  int const tile_h = hout_round / 3;
  int const tile_w = wout_round / 8;

  int const pre_in_size = hin_round * win_round;
  int const pre_out_size = hout_round * wout_round;
  int const pre_io_size = pre_in_size + pre_out_size * sizeof(int);

  int const hs = -pad_h;
  int const he = hs + hin_round;
  int const ws = -pad_w;
  int const we = ws + win_round;

  // signed char* tmp_work_space = new signed char [1024*5];
  signed char* tmp_work_space = ctx->workspace_data<signed char>();
  signed char* ptr_zero = tmp_work_space;
  int* ptr_write = reinterpret_cast<int*>(ptr_zero + win_round);
  signed char* pre_data =
      reinterpret_cast<signed char*>(ptr_write + wout_round);

  memset(ptr_zero, 0, win_round * sizeof(signed char));

  for (int n = 0; n < num; ++n) {
    signed char const* din_batch = din + n * chin * size_in_channel;
    int* dout_batch = dout + n * chout * size_out_channel;

    // #pragma omp parallel for
    for (int c = 0; c < chout; c++) {
#ifdef USE_OPENMP
      int const thno = omp_get_thread_num();
#else
      int const thno = 0;
#endif
      signed char const* din_channel = din_batch + c * size_in_channel;
      signed char* pre_din = pre_data + thno * pre_io_size;
      int* pre_out = reinterpret_cast<int*>(pre_din + pre_in_size);
      int* dout_ptr = pre_out;

      prepack_input_nxw(din_channel, pre_din, c, c + 1, hs, he, ws, we, 1, win,
                        hin, ptr_zero);

      signed char const* wei_ptr = weights + c * w_stride;
      int bias_val = flag_bias ? bias[c] : 0.f;

      int8x8_t wr00 = vdup_n_s8(wei_ptr[0 * 5 + 0]);
      int8x8_t wr01 = vdup_n_s8(wei_ptr[0 * 5 + 1]);
      int8x8_t wr02 = vdup_n_s8(wei_ptr[0 * 5 + 2]);
      int8x8_t wr03 = vdup_n_s8(wei_ptr[0 * 5 + 3]);
      int8x8_t wr04 = vdup_n_s8(wei_ptr[0 * 5 + 4]);

      int8x8_t wr10 = vdup_n_s8(wei_ptr[1 * 5 + 0]);
      int8x8_t wr11 = vdup_n_s8(wei_ptr[1 * 5 + 1]);
      int8x8_t wr12 = vdup_n_s8(wei_ptr[1 * 5 + 2]);
      int8x8_t wr13 = vdup_n_s8(wei_ptr[1 * 5 + 3]);
      int8x8_t wr14 = vdup_n_s8(wei_ptr[1 * 5 + 4]);

      int8x8_t wr20 = vdup_n_s8(wei_ptr[2 * 5 + 0]);
      int8x8_t wr21 = vdup_n_s8(wei_ptr[2 * 5 + 1]);
      int8x8_t wr22 = vdup_n_s8(wei_ptr[2 * 5 + 2]);
      int8x8_t wr23 = vdup_n_s8(wei_ptr[2 * 5 + 3]);
      int8x8_t wr24 = vdup_n_s8(wei_ptr[2 * 5 + 4]);

      int8x8_t wr30 = vdup_n_s8(wei_ptr[3 * 5 + 0]);
      int8x8_t wr31 = vdup_n_s8(wei_ptr[3 * 5 + 1]);
      int8x8_t wr32 = vdup_n_s8(wei_ptr[3 * 5 + 2]);
      int8x8_t wr33 = vdup_n_s8(wei_ptr[3 * 5 + 3]);
      int8x8_t wr34 = vdup_n_s8(wei_ptr[3 * 5 + 4]);

      int8x8_t wr40 = vdup_n_s8(wei_ptr[4 * 5 + 0]);
      int8x8_t wr41 = vdup_n_s8(wei_ptr[4 * 5 + 1]);
      int8x8_t wr42 = vdup_n_s8(wei_ptr[4 * 5 + 2]);
      int8x8_t wr43 = vdup_n_s8(wei_ptr[4 * 5 + 3]);
      int8x8_t wr44 = vdup_n_s8(wei_ptr[4 * 5 + 4]);

      int* doutr0 = nullptr;
      int* doutr1 = nullptr;
      int* doutr2 = nullptr;

      signed char const* dr0 = pre_din;
      signed char const* dr1 = dr0 + win_round;
      signed char const* dr2 = dr1 + win_round;
      signed char const* dr3 = dr2 + win_round;
      signed char const* dr4 = dr3 + win_round;
      signed char const* dr5 = dr4 + win_round;
      signed char const* dr6 = dr5 + win_round;

      signed char const* din_ptr0 = nullptr;
      signed char const* din_ptr1 = nullptr;
      signed char const* din_ptr2 = nullptr;
      signed char const* din_ptr3 = nullptr;
      signed char const* din_ptr4 = nullptr;
      signed char const* din_ptr5 = nullptr;
      signed char const* din_ptr6 = nullptr;

      for (int h = 0; h < tile_h; h++) {
        // printf("c:%d h:%d\n", c, h);
        doutr0 = dout_ptr;
        doutr1 = doutr0 + wout_round;
        doutr2 = doutr1 + wout_round;

        din_ptr0 = dr0;
        din_ptr1 = dr1;
        din_ptr2 = dr2;
        din_ptr3 = dr3;
        din_ptr4 = dr4;
        din_ptr5 = dr5;
        din_ptr6 = dr6;

        prefetch(doutr0);
        prefetch(doutr1);
        prefetch(doutr2);
        prefetch(din_ptr0);
        prefetch(din_ptr1);
        prefetch(din_ptr2);
        prefetch(din_ptr3);
        prefetch(din_ptr4);
        prefetch(din_ptr5);
        prefetch(din_ptr6);

        for (int j = 0; j < tile_w; ++j) {
          // printf("j:%d\n", j);
          int32x4_t voutr00 = vdupq_n_s32(bias_val);
          int32x4_t voutr01 = vdupq_n_s32(bias_val);
          int32x4_t voutr10 = vdupq_n_s32(bias_val);
          int32x4_t voutr11 = vdupq_n_s32(bias_val);
          int32x4_t voutr20 = vdupq_n_s32(bias_val);
          int32x4_t voutr21 = vdupq_n_s32(bias_val);

          // din data
          int8x8_t vinr00 = vld1_s8(din_ptr0 + 0);
          int8x8_t vinr01 = vld1_s8(din_ptr0 + 8);
          int8x8_t vinr10 = vld1_s8(din_ptr1 + 0);
          int8x8_t vinr11 = vld1_s8(din_ptr1 + 8);
          int8x8_t vinr20 = vld1_s8(din_ptr2 + 0);
          int8x8_t vinr21 = vld1_s8(din_ptr2 + 8);
          int8x8_t vinr30 = vld1_s8(din_ptr3 + 0);
          int8x8_t vinr31 = vld1_s8(din_ptr3 + 8);
          int8x8_t vinr40 = vld1_s8(din_ptr4 + 0);
          int8x8_t vinr41 = vld1_s8(din_ptr4 + 8);
          int8x8_t vinr50 = vld1_s8(din_ptr5 + 0);
          int8x8_t vinr51 = vld1_s8(din_ptr5 + 8);
          int8x8_t vinr60 = vld1_s8(din_ptr6 + 0);
          int8x8_t vinr61 = vld1_s8(din_ptr6 + 8);

          /// the first row
          // r0
          int8x8_t vtmp1 = vext_s8(vinr00, vinr01, 1);  // 12345678
          int8x8_t vtmp2 = vext_s8(vinr00, vinr01, 2);  // 2345678
          int8x8_t vtmp3 = vext_s8(vinr00, vinr01, 3);  // 345678
          int8x8_t vtmp4 = vext_s8(vinr00, vinr01, 4);  // 45678

          int16x8_t tvoutr0 = vmull_s8(vinr00, wr00);
          tvoutr0 = vmlal_s8(tvoutr0, vtmp1, wr01);
          voutr00 = vaddw_s16(voutr00, vget_low_s16(tvoutr0));
          voutr01 = vaddw_s16(voutr01, vget_high_s16(tvoutr0));
          tvoutr0 = vmull_s8(vtmp2, wr02);
          tvoutr0 = vmlal_s8(tvoutr0, vtmp3, wr03);
          voutr00 = vaddw_s16(voutr00, vget_low_s16(tvoutr0));
          voutr01 = vaddw_s16(voutr01, vget_high_s16(tvoutr0));
          tvoutr0 = vmull_s8(vtmp4, wr04);
          voutr00 = vaddw_s16(voutr00, vget_low_s16(tvoutr0));
          voutr01 = vaddw_s16(voutr01, vget_high_s16(tvoutr0));

          // r1
          vtmp1 = vext_s8(vinr10, vinr11, 1);  // 12345678
          vtmp2 = vext_s8(vinr10, vinr11, 2);  // 2345678
          vtmp3 = vext_s8(vinr10, vinr11, 3);  // 345678
          vtmp4 = vext_s8(vinr10, vinr11, 4);  // 45678

          tvoutr0 = vmull_s8(vinr10, wr10);
          tvoutr0 = vmlal_s8(tvoutr0, vtmp1, wr11);
          voutr00 = vaddw_s16(voutr00, vget_low_s16(tvoutr0));
          voutr01 = vaddw_s16(voutr01, vget_high_s16(tvoutr0));
          tvoutr0 = vmull_s8(vtmp2, wr12);
          tvoutr0 = vmlal_s8(tvoutr0, vtmp3, wr13);
          voutr00 = vaddw_s16(voutr00, vget_low_s16(tvoutr0));
          voutr01 = vaddw_s16(voutr01, vget_high_s16(tvoutr0));
          tvoutr0 = vmull_s8(vtmp4, wr14);
          voutr00 = vaddw_s16(voutr00, vget_low_s16(tvoutr0));
          voutr01 = vaddw_s16(voutr01, vget_high_s16(tvoutr0));

          int16x8_t tvoutr1 = vmull_s8(vinr10, wr00);
          tvoutr1 = vmlal_s8(tvoutr1, vtmp1, wr01);
          voutr10 = vaddw_s16(voutr10, vget_low_s16(tvoutr1));
          voutr11 = vaddw_s16(voutr11, vget_high_s16(tvoutr1));
          tvoutr1 = vmull_s8(vtmp2, wr02);
          tvoutr1 = vmlal_s8(tvoutr1, vtmp3, wr03);
          voutr10 = vaddw_s16(voutr10, vget_low_s16(tvoutr1));
          voutr11 = vaddw_s16(voutr11, vget_high_s16(tvoutr1));
          tvoutr1 = vmull_s8(vtmp4, wr04);
          voutr10 = vaddw_s16(voutr10, vget_low_s16(tvoutr1));
          voutr11 = vaddw_s16(voutr11, vget_high_s16(tvoutr1));

          // r2
          vtmp1 = vext_s8(vinr20, vinr21, 1);  // 12345678
          vtmp2 = vext_s8(vinr20, vinr21, 2);  // 2345678
          vtmp3 = vext_s8(vinr20, vinr21, 3);  // 345678
          vtmp4 = vext_s8(vinr20, vinr21, 4);  // 45678

          tvoutr0 = vmull_s8(vinr20, wr20);
          tvoutr0 = vmlal_s8(tvoutr0, vtmp1, wr21);
          voutr00 = vaddw_s16(voutr00, vget_low_s16(tvoutr0));
          voutr01 = vaddw_s16(voutr01, vget_high_s16(tvoutr0));
          tvoutr0 = vmull_s8(vtmp2, wr22);
          tvoutr0 = vmlal_s8(tvoutr0, vtmp3, wr23);
          voutr00 = vaddw_s16(voutr00, vget_low_s16(tvoutr0));
          voutr01 = vaddw_s16(voutr01, vget_high_s16(tvoutr0));
          tvoutr0 = vmull_s8(vtmp4, wr24);
          voutr00 = vaddw_s16(voutr00, vget_low_s16(tvoutr0));
          voutr01 = vaddw_s16(voutr01, vget_high_s16(tvoutr0));

          tvoutr1 = vmull_s8(vinr20, wr10);
          tvoutr1 = vmlal_s8(tvoutr1, vtmp1, wr11);
          voutr10 = vaddw_s16(voutr10, vget_low_s16(tvoutr1));
          voutr11 = vaddw_s16(voutr11, vget_high_s16(tvoutr1));
          tvoutr1 = vmull_s8(vtmp2, wr12);
          tvoutr1 = vmlal_s8(tvoutr1, vtmp3, wr13);
          voutr10 = vaddw_s16(voutr10, vget_low_s16(tvoutr1));
          voutr11 = vaddw_s16(voutr11, vget_high_s16(tvoutr1));
          tvoutr1 = vmull_s8(vtmp4, wr14);
          voutr10 = vaddw_s16(voutr10, vget_low_s16(tvoutr1));
          voutr11 = vaddw_s16(voutr11, vget_high_s16(tvoutr1));

          int16x8_t tvoutr2 = vmull_s8(vinr20, wr00);
          tvoutr2 = vmlal_s8(tvoutr2, vtmp1, wr01);
          voutr20 = vaddw_s16(voutr20, vget_low_s16(tvoutr2));
          voutr21 = vaddw_s16(voutr21, vget_high_s16(tvoutr2));
          tvoutr2 = vmull_s8(vtmp2, wr02);
          tvoutr2 = vmlal_s8(tvoutr2, vtmp3, wr03);
          voutr20 = vaddw_s16(voutr20, vget_low_s16(tvoutr2));
          voutr21 = vaddw_s16(voutr21, vget_high_s16(tvoutr2));
          tvoutr2 = vmull_s8(vtmp4, wr04);
          voutr20 = vaddw_s16(voutr20, vget_low_s16(tvoutr2));
          voutr21 = vaddw_s16(voutr21, vget_high_s16(tvoutr2));

          // r3
          vtmp1 = vext_s8(vinr30, vinr31, 1);  // 12345678
          vtmp2 = vext_s8(vinr30, vinr31, 2);  // 2345678
          vtmp3 = vext_s8(vinr30, vinr31, 3);  // 345678
          vtmp4 = vext_s8(vinr30, vinr31, 4);  // 45678

          tvoutr0 = vmull_s8(vinr30, wr30);
          tvoutr0 = vmlal_s8(tvoutr0, vtmp1, wr31);
          voutr00 = vaddw_s16(voutr00, vget_low_s16(tvoutr0));
          voutr01 = vaddw_s16(voutr01, vget_high_s16(tvoutr0));
          tvoutr0 = vmull_s8(vtmp2, wr32);
          tvoutr0 = vmlal_s8(tvoutr0, vtmp3, wr33);
          voutr00 = vaddw_s16(voutr00, vget_low_s16(tvoutr0));
          voutr01 = vaddw_s16(voutr01, vget_high_s16(tvoutr0));
          tvoutr0 = vmull_s8(vtmp4, wr34);
          voutr00 = vaddw_s16(voutr00, vget_low_s16(tvoutr0));
          voutr01 = vaddw_s16(voutr01, vget_high_s16(tvoutr0));

          tvoutr1 = vmull_s8(vinr30, wr20);
          tvoutr1 = vmlal_s8(tvoutr1, vtmp1, wr21);
          voutr10 = vaddw_s16(voutr10, vget_low_s16(tvoutr1));
          voutr11 = vaddw_s16(voutr11, vget_high_s16(tvoutr1));
          tvoutr1 = vmull_s8(vtmp2, wr22);
          tvoutr1 = vmlal_s8(tvoutr1, vtmp3, wr23);
          voutr10 = vaddw_s16(voutr10, vget_low_s16(tvoutr1));
          voutr11 = vaddw_s16(voutr11, vget_high_s16(tvoutr1));
          tvoutr1 = vmull_s8(vtmp4, wr24);
          voutr10 = vaddw_s16(voutr10, vget_low_s16(tvoutr1));
          voutr11 = vaddw_s16(voutr11, vget_high_s16(tvoutr1));

          tvoutr2 = vmull_s8(vinr30, wr10);
          tvoutr2 = vmlal_s8(tvoutr2, vtmp1, wr11);
          voutr20 = vaddw_s16(voutr20, vget_low_s16(tvoutr2));
          voutr21 = vaddw_s16(voutr21, vget_high_s16(tvoutr2));
          tvoutr2 = vmull_s8(vtmp2, wr12);
          tvoutr2 = vmlal_s8(tvoutr2, vtmp3, wr13);
          voutr20 = vaddw_s16(voutr20, vget_low_s16(tvoutr2));
          voutr21 = vaddw_s16(voutr21, vget_high_s16(tvoutr2));
          tvoutr2 = vmull_s8(vtmp4, wr14);
          voutr20 = vaddw_s16(voutr20, vget_low_s16(tvoutr2));
          voutr21 = vaddw_s16(voutr21, vget_high_s16(tvoutr2));

          // r4
          vtmp1 = vext_s8(vinr40, vinr41, 1);  // 12345678
          vtmp2 = vext_s8(vinr40, vinr41, 2);  // 2345678
          vtmp3 = vext_s8(vinr40, vinr41, 3);  // 345678
          vtmp4 = vext_s8(vinr40, vinr41, 4);  // 45678

          tvoutr0 = vmull_s8(vinr40, wr40);
          tvoutr0 = vmlal_s8(tvoutr0, vtmp1, wr41);
          voutr00 = vaddw_s16(voutr00, vget_low_s16(tvoutr0));
          voutr01 = vaddw_s16(voutr01, vget_high_s16(tvoutr0));
          tvoutr0 = vmull_s8(vtmp2, wr42);
          tvoutr0 = vmlal_s8(tvoutr0, vtmp3, wr43);
          voutr00 = vaddw_s16(voutr00, vget_low_s16(tvoutr0));
          voutr01 = vaddw_s16(voutr01, vget_high_s16(tvoutr0));
          tvoutr0 = vmull_s8(vtmp4, wr44);
          voutr00 = vaddw_s16(voutr00, vget_low_s16(tvoutr0));
          voutr01 = vaddw_s16(voutr01, vget_high_s16(tvoutr0));

          tvoutr1 = vmull_s8(vinr40, wr30);
          tvoutr1 = vmlal_s8(tvoutr1, vtmp1, wr31);
          voutr10 = vaddw_s16(voutr10, vget_low_s16(tvoutr1));
          voutr11 = vaddw_s16(voutr11, vget_high_s16(tvoutr1));
          tvoutr1 = vmull_s8(vtmp2, wr32);
          tvoutr1 = vmlal_s8(tvoutr1, vtmp3, wr33);
          voutr10 = vaddw_s16(voutr10, vget_low_s16(tvoutr1));
          voutr11 = vaddw_s16(voutr11, vget_high_s16(tvoutr1));
          tvoutr1 = vmull_s8(vtmp4, wr34);
          voutr10 = vaddw_s16(voutr10, vget_low_s16(tvoutr1));
          voutr11 = vaddw_s16(voutr11, vget_high_s16(tvoutr1));

          tvoutr2 = vmull_s8(vinr40, wr20);
          tvoutr2 = vmlal_s8(tvoutr2, vtmp1, wr21);
          voutr20 = vaddw_s16(voutr20, vget_low_s16(tvoutr2));
          voutr21 = vaddw_s16(voutr21, vget_high_s16(tvoutr2));
          tvoutr2 = vmull_s8(vtmp2, wr22);
          tvoutr2 = vmlal_s8(tvoutr2, vtmp3, wr23);
          voutr20 = vaddw_s16(voutr20, vget_low_s16(tvoutr2));
          voutr21 = vaddw_s16(voutr21, vget_high_s16(tvoutr2));
          tvoutr2 = vmull_s8(vtmp4, wr24);
          voutr20 = vaddw_s16(voutr20, vget_low_s16(tvoutr2));
          voutr21 = vaddw_s16(voutr21, vget_high_s16(tvoutr2));

          // r5
          vtmp1 = vext_s8(vinr50, vinr51, 1);  // 12345678
          vtmp2 = vext_s8(vinr50, vinr51, 2);  // 2345678
          vtmp3 = vext_s8(vinr50, vinr51, 3);  // 345678
          vtmp4 = vext_s8(vinr50, vinr51, 4);  // 45678

          tvoutr1 = vmull_s8(vinr50, wr40);
          tvoutr1 = vmlal_s8(tvoutr1, vtmp1, wr41);
          voutr10 = vaddw_s16(voutr10, vget_low_s16(tvoutr1));
          voutr11 = vaddw_s16(voutr11, vget_high_s16(tvoutr1));
          tvoutr1 = vmull_s8(vtmp2, wr42);
          tvoutr1 = vmlal_s8(tvoutr1, vtmp3, wr43);
          voutr10 = vaddw_s16(voutr10, vget_low_s16(tvoutr1));
          voutr11 = vaddw_s16(voutr11, vget_high_s16(tvoutr1));
          tvoutr1 = vmull_s8(vtmp4, wr44);
          voutr10 = vaddw_s16(voutr10, vget_low_s16(tvoutr1));
          voutr11 = vaddw_s16(voutr11, vget_high_s16(tvoutr1));

          tvoutr2 = vmull_s8(vinr50, wr30);
          tvoutr2 = vmlal_s8(tvoutr2, vtmp1, wr31);
          voutr20 = vaddw_s16(voutr20, vget_low_s16(tvoutr2));
          voutr21 = vaddw_s16(voutr21, vget_high_s16(tvoutr2));
          tvoutr2 = vmull_s8(vtmp2, wr32);
          tvoutr2 = vmlal_s8(tvoutr2, vtmp3, wr33);
          voutr20 = vaddw_s16(voutr20, vget_low_s16(tvoutr2));
          voutr21 = vaddw_s16(voutr21, vget_high_s16(tvoutr2));
          tvoutr2 = vmull_s8(vtmp4, wr34);
          voutr20 = vaddw_s16(voutr20, vget_low_s16(tvoutr2));
          voutr21 = vaddw_s16(voutr21, vget_high_s16(tvoutr2));

          // r6
          vtmp1 = vext_s8(vinr60, vinr61, 1);  // 12345678
          vtmp2 = vext_s8(vinr60, vinr61, 2);  // 2345678
          vtmp3 = vext_s8(vinr60, vinr61, 3);  // 345678
          vtmp4 = vext_s8(vinr60, vinr61, 4);  // 45678

          tvoutr2 = vmull_s8(vinr60, wr40);
          tvoutr2 = vmlal_s8(tvoutr2, vtmp1, wr41);
          voutr20 = vaddw_s16(voutr20, vget_low_s16(tvoutr2));
          voutr21 = vaddw_s16(voutr21, vget_high_s16(tvoutr2));
          tvoutr2 = vmull_s8(vtmp2, wr42);
          tvoutr2 = vmlal_s8(tvoutr2, vtmp3, wr43);
          voutr20 = vaddw_s16(voutr20, vget_low_s16(tvoutr2));
          voutr21 = vaddw_s16(voutr21, vget_high_s16(tvoutr2));
          tvoutr2 = vmull_s8(vtmp4, wr44);
          voutr20 = vaddw_s16(voutr20, vget_low_s16(tvoutr2));
          voutr21 = vaddw_s16(voutr21, vget_high_s16(tvoutr2));

          /// data shift 8 bytes
          din_ptr0 += 8;
          din_ptr1 += 8;
          din_ptr2 += 8;
          din_ptr3 += 8;
          din_ptr4 += 8;
          din_ptr5 += 8;
          din_ptr6 += 8;

          /// store
          vst1q_s32(doutr0, voutr00);
          vst1q_s32(doutr1, voutr10);
          vst1q_s32(doutr2, voutr20);
          doutr0 += 4;
          doutr1 += 4;
          doutr2 += 4;
          vst1q_s32(doutr0, voutr01);
          vst1q_s32(doutr1, voutr11);
          vst1q_s32(doutr2, voutr21);
          doutr0 += 4;
          doutr1 += 4;
          doutr2 += 4;
        }  /// end of tile_w

        dr0 = dr3;
        dr1 = dr4;
        dr2 = dr5;
        dr3 = dr6;
        dr4 = dr3 + win_round;
        dr5 = dr4 + win_round;
        dr6 = dr5 + win_round;

        dout_ptr = dout_ptr + 3 * wout_round;
      }  /// end of tile_h

      if (scales == 0) {
        write_to_output_numc(pre_out, dout_batch, 1, hout_round, c, c + 1, 0,
                             hout, 0, wout_round, chout, hout, wout, flag_relu,
                             ptr_write);
      } else if (od_type == PRECISION(kFloat)) {
        write2_to_output_numc(pre_out, reinterpret_cast<float*>(dout_batch), 1,
                              hout_round, c, c + 1, 0, hout, 0, wout_round,
                              chout, hout, wout, flag_relu,
                              reinterpret_cast<float*>(ptr_write), scales);
      } else if (od_type == PRECISION(kInt8)) {
        write2_to_output_numc(
            pre_out, reinterpret_cast<signed char*>(dout_batch), 1, hout_round,
            c, c + 1, 0, hout, 0, wout_round, chout, hout, wout, flag_relu,
            reinterpret_cast<signed char*>(ptr_write), scales);
      }
      // else if (od_type == AK_INT32) {
      //     write2_to_output_numc(pre_out, (int*)dout_batch, 1, hout_round, c,
      //     c+1,
      //         0, hout, 0, wout_round, chout, hout, wout, flag_relu,
      //         (int*)ptr_write, scales);
      // }
    }  /// end of chout
  }    /// end of batch num
}

#endif  // __aarch64__

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
