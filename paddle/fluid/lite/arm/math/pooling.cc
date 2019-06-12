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

#include "paddle/fluid/lite/arm/math/pooling.h"
#include <algorithm>
#include <limits>
#include "paddle/fluid/lite/arm/math/funcs.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {

void pooling_basic(const void* din, void* dout, int num, int chout, int hout,
                   int wout, int chin, int hin, int win,
                   const std::vector<int>& ksize,
                   const std::vector<int>& strides,
                   const std::vector<int>& paddings, bool global_pooling,
                   bool exclusive, bool adaptive, bool ceil_mode,
                   bool use_quantizer, const std::string& pooling_type) {
  // no need to pad input tensor, border is zero pad inside this function
  int kernel_h = ksize[0];
  int kernel_w = ksize[1];
  int stride_h = strides[0];
  int stride_w = strides[1];
  int pad_h = paddings[0];
  int pad_w = paddings[1];
  int size_channel_in = win * hin;
  int size_channel_out = wout * hout;

  float* data_out = static_cast<float*>(dout);
  const float* data_in = static_cast<const float*>(din);

  if (global_pooling) {
    if (pooling_type == "max") {  // Pooling_max
      for (int n = 0; n < num; ++n) {
        float* data_out_batch = data_out + n * chout * size_channel_out;
        const float* data_in_batch = data_in + n * chin * size_channel_in;
#pragma omp parallel for
        for (int c = 0; c < chout; ++c) {
          const float* data_in_channel =
              data_in_batch + c * size_channel_in;  // in address
          data_out_batch[c] = data_in_channel[0];
          for (int i = 0; i < size_channel_in; ++i) {
            data_out_batch[c] = data_out_batch[c] > data_in_channel[i]
                                    ? data_out_batch[c]
                                    : data_in_channel[i];
          }
        }
      }

    } else if (pooling_type == "avg") {
      // Pooling_average_include_padding
      // Pooling_average_exclude_padding
      for (int n = 0; n < num; ++n) {
        float* data_out_batch = data_out + n * chout * size_channel_out;
        const float* data_in_batch = data_in + n * chin * size_channel_in;
#pragma omp parallel for
        for (int c = 0; c < chout; ++c) {
          const float* data_in_channel =
              data_in_batch + c * size_channel_in;  // in address
          float sum = 0.f;
          for (int i = 0; i < size_channel_in; ++i) {
            sum += data_in_channel[i];
          }
          data_out_batch[c] = sum / size_channel_in;
        }
      }
    } else {
      LOG(FATAL) << "not support";
    }
    return;
  }

  if (pooling_type == "max") {
    // Pooling_max
    for (int n = 0; n < num; ++n) {
      float* data_out_channel = data_out + n * chout * size_channel_out;
      const float* data_in_batch = data_in + n * chin * size_channel_in;
#pragma omp parallel for
      for (int q = 0; q < chout; q++) {
        float* data_out_row = data_out_channel + q * size_channel_out;
        const float* data_in_channel = data_in_batch + q * size_channel_in;

        for (int i = 0; i < hout; i++) {
          for (int j = 0; j < wout; j++) {
            int hstart = i * stride_h - pad_h;
            int wstart = j * stride_w - pad_w;
            int hend = std::min(hstart + kernel_h, hin + pad_h);
            int wend = std::min(wstart + kernel_w, win + pad_w);
            hstart = std::max(hstart, 0);
            wstart = std::max(wstart, 0);
            hend = std::min(hend, hin);
            wend = std::min(wend, win);

            data_out_row[j] = data_in_channel[hstart * win + wstart];
            for (int h = hstart; h < hend; ++h) {
              for (int w = wstart; w < wend; ++w) {
                data_out_row[j] = data_out_row[j] > data_in_channel[h * win + w]
                                      ? data_out_row[j]
                                      : data_in_channel[h * win + w];
              }
            }
          }
          data_out_row += wout;
        }
      }
    }
  } else if (pooling_type == "avg") {
    if (exclusive == false) {
      // Pooling_average_include_padding
      for (int n = 0; n < num; ++n) {
        int pool_size =
            kernel_w *
            kernel_h;  // (hend - hstart) * (wend - wstart); // problem
        float* data_out_channel = data_out + n * chout * size_channel_out;
        const float* data_in_batch = data_in + n * chin * size_channel_in;
#pragma omp parallel for
        for (int q = 0; q < chout; q++) {
          float* data_out_row = data_out_channel + q * size_channel_out;
          const float* data_in_channel = data_in_batch + q * size_channel_in;
          for (int i = 0; i < hout; i++) {
            for (int j = 0; j < wout; j++) {
              int hstart = i * stride_h - pad_h;
              int wstart = j * stride_w - pad_w;
              int hend = std::min(hstart + kernel_h, hin + pad_h);
              int wend = std::min(wstart + kernel_w, win + pad_w);
              hstart = std::max(hstart, 0);
              wstart = std::max(wstart, 0);
              hend = std::min(hend, hin);
              wend = std::min(wend, win);

              int bh = kernel_h;
              int bw = kernel_w;
              if (wend == win) {
                bw = wstart + kernel_w >= win + pad_w ? win + pad_w
                                                      : wstart + kernel_w;
                bw -= wstart;
              }
              if (hend == hin) {
                bh = hstart + kernel_h >= hin + pad_h ? hin + pad_h
                                                      : hstart + kernel_h;
                bh -= hstart;
              }
              pool_size = bh * bw;

              data_out_row[j] = data_in_channel[hstart * win + wstart];
              float sum = 0.f;
              for (int h = hstart; h < hend; ++h) {
                for (int w = wstart; w < wend; ++w) {
                  sum += data_in_channel[h * win + w];
                }
              }
              data_out_row[j] = sum / pool_size;
            }
            data_out_row += wout;
          }
        }
      }
    } else {  // exclusive == true, Pooling_average_exclude_padding
      for (int n = 0; n < num; ++n) {
        float* data_out_channel = data_out + n * chout * size_channel_out;
        const float* data_in_batch = data_in + n * chin * size_channel_in;
#pragma omp parallel for
        for (int q = 0; q < chout; q++) {
          float* data_out_row = data_out_channel + q * size_channel_out;
          const float* data_in_channel = data_in_batch + q * size_channel_in;
          for (int i = 0; i < hout; i++) {
            for (int j = 0; j < wout; j++) {
              int hstart = i * stride_h - pad_h;
              int wstart = j * stride_w - pad_w;
              int hend = std::min(hstart + kernel_h, hin + pad_h);
              int wend = std::min(wstart + kernel_w, win + pad_w);
              hstart = std::max(hstart, 0);
              wstart = std::max(wstart, 0);
              hend = std::min(hend, hin);
              wend = std::min(wend, win);

              data_out_row[j] = data_in_channel[hstart * win + wstart];
              float sum = 0.f;
              for (int h = hstart; h < hend; ++h) {
                for (int w = wstart; w < wend; ++w) {
                  sum += data_in_channel[h * win + w];
                }
              }
              int pool_size = (hend - hstart) * (wend - wstart);
              data_out_row[j] = sum / pool_size;
            }
            data_out_row += wout;
          }
        }
      }
    }

  } else {
    LOG(FATAL) << "not support";
  }
}

void pooling_global(const void* din, void* dout, int num, int chout, int hout,
                    int wout, int chin, int hin, int win,
                    const std::vector<int>& ksize,
                    const std::vector<int>& strides,
                    const std::vector<int>& paddings, bool global_pooling,
                    bool exclusive, bool adaptive, bool ceil_mode,
                    bool use_quantizer, const std::string& pooling_type) {
  int size_channel_in = win * hin;
  float* data_out = static_cast<float*>(dout);
  const float* data_in = static_cast<const float*>(din);

  int cnt = size_channel_in / 8;

#if 0
    LOG(INFO) << "size_channel_in:" << size_channel_in;
    LOG(INFO) << "cnt:" << cnt;
    LOG(INFO) << "num:" << num;
    LOG(INFO) << "chout:" << chout;
    LOG(INFO) << "hout:" << hout;
    LOG(INFO) << "wout:" << wout;

    LOG(INFO) << "chin:" << chin;
    LOG(INFO) << "hin:" << hin;
    LOG(INFO) << "win:" << win;
    LOG(INFO) << "pooling_type " << pooling_type;
#endif

  for (int n = 0; n < num; ++n) {
    float* data_out_batch = data_out + n * chout;
    const float* data_in_batch = data_in + n * chin * size_channel_in;
    if (pooling_type == "max") {
#pragma omp parallel for
      for (int c = 0; c < chout; ++c) {
        const float* data_in_channel = data_in_batch + c * size_channel_in;
        int i = 0;
        float minval = std::numeric_limits<float>::lowest();
        float32x4_t vmax = vdupq_n_f32(minval);
#ifdef __aarch64__
        for (; i < cnt; i++) {
          float32x4_t vdin1 = vld1q_f32(data_in_channel);
          vmax = vmaxq_f32(vdin1, vmax);
          float32x4_t vdin2 = vld1q_f32(data_in_channel + 4);
          vmax = vmaxq_f32(vmax, vdin2);
          data_in_channel += 8;
        }
#else
        int num = cnt;
        if (num > 0) {
          asm volatile(
              "max_loop:                                        @main loop\n"
              "vld1.f32   {d0-d1}, [%[data_in_channel]]!        @load q1, "
              "data_in_channel\n"
              "vmax.f32   %q[vmax], %q[vmax], q0                @max vmax, "
              "vmax, data_in_channel\n"
              "vld1.f32   {d2-d3}, [%[data_in_channel]]!        @ load 2nd 4 "
              "data"
              "vmax.f32   %q[vmax], %q[vmax], q1                @ compare 2nd "
              "4 datas\n"
              "subs       %[num], #1                            @subs num, 1\n"
              "bne        max_loop                              @bne num\n"
              : [data_in_channel] "+r"(data_in_channel), [num] "+r"(num),
                [vmax] "+w"(vmax)
              :
              : "cc", "memory", "q0", "q1");
        }
#endif  // __aarch64__
        float32x2_t vmax_tmp =
            vmax_f32(vget_low_f32(vmax), vget_high_f32(vmax));
        float tmp1 = vget_lane_f32(vmax_tmp, 0);
        float tmp2 = vget_lane_f32(vmax_tmp, 1);
        float max_tmp = tmp1 > tmp2 ? tmp1 : tmp2;
        for (i = cnt * 8; i < size_channel_in; ++i) {
          /* code */
          max_tmp = max_tmp > data_in_channel[0] ? max_tmp : data_in_channel[0];
          data_in_channel++;
        }
        data_out_batch[c] = max_tmp;
      }
    } else {
#pragma omp parallel for
      for (int c = 0; c < chout; c++) {
        const float* data_in_channel =
            data_in_batch + c * size_channel_in;  // in address
        int i = 0;
        float32x4_t vsum = vdupq_n_f32(0.0f);
#ifdef __aarch64__
        for (; i < cnt; i++) {  //
          vsum = vaddq_f32(vld1q_f32(data_in_channel), vsum);
          data_in_channel += 4;
        }
#else
        int num = cnt;
        if (num > 0) {
          asm volatile(
              "add_loop:                                        @main loop\n"
              "vld1.f32   {d0-d1}, [%[data_in_channel]]!        @load q1, "
              "data_in_channel\n"
              "vadd.f32   %q[vsum], %q[vsum], q0                @add vmax, "
              "vmax, data_in_channel\n"
              "subs        %[num], #1                           @subs num, 1\n"
              "bne        add_loop                              @bne num\n"
              : [data_in_channel] "+r"(data_in_channel), [num] "+r"(num),
                [vsum] "+w"(vsum)
              :
              : "cc", "memory", "q0");
        }
#endif  // __aarch64__
        float32x2_t vsum_tmp =
            vadd_f32(vget_low_f32(vsum), vget_high_f32(vsum));
        float sum = vget_lane_f32(vsum_tmp, 0) + vget_lane_f32(vsum_tmp, 1);
        for (i = cnt * 4; i < size_channel_in; i++) {
          sum += data_in_channel[0];
          data_in_channel++;
        }
        data_out_batch[c] = sum / size_channel_in;
      }
    }
  }
}

void pooling2x2s2_max(const void* din, void* dout, int num, int chout, int hout,
                      int wout, int chin, int hin, int win,
                      const std::vector<int>& ksize,
                      const std::vector<int>& strides,
                      const std::vector<int>& paddings, bool global_pooling,
                      bool exclusive, bool adaptive, bool ceil_mode,
                      bool use_quantizer, const std::string& pooling_type) {
  int size_channel_out = wout * hout;
  int size_channel_in = win * hin;
  float* data_out = static_cast<float*>(dout);
  const float* data_in = static_cast<const float*>(din);

  int w_even = (win >> 1) << 1;
  // int w_remains = w_in - w_even; // should be 0 or 1
  int h_even = (hin >> 1) << 1;
  // int h_remains = h_in - h_even; // should be 0 or 1
  int w_unroll_size = (w_even >> 3) << 3;
  // int w_unroll_remian = w_even - w_unroll_size;
  int w_in_2 = win << 1;
  float32x4_t vzero = vdupq_n_f32(0.f);

  for (int n = 0; n < num; ++n) {
    float* data_out_batch = data_out + n * chout * size_channel_out;
    const float* data_in_batch = data_in + n * chin * size_channel_in;
#pragma omp parallel for
    for (int c = 0; c < chout; c++) {
      float* data_out_channel = data_out_batch + c * size_channel_out;
      const float* data_in_channel = data_in_batch + c * size_channel_in;
      const float* r0 = data_in_channel;
      const float* r1 = r0 + win;
      int h = 0;
      for (; h < h_even; h += 2) {
        int w = 0;
#ifdef __aarch64__
        for (; w < w_unroll_size; w += 8) {
          float32x4_t dr00 = vld1q_f32(&r0[w]);
          float32x4_t dr01 = vld1q_f32(&r0[w + 4]);
          float32x4_t dr10 = vld1q_f32(&r1[w]);
          float32x4_t dr11 = vld1q_f32(&r1[w + 4]);
          float32x4_t dmax1 = vmaxq_f32(dr00, dr10);
          float32x4_t dmax2 = vmaxq_f32(dr01, dr11);
#ifdef __aarch64__
          float32x4_t dmax = vpmaxq_f32(dmax1, dmax2);
#else
          float32x2_t dmaxl =
              vpmax_f32(vget_low_f32(dmax1), vget_high_f32(dmax1));
          float32x2_t dmaxh =
              vpmax_f32(vget_low_f32(dmax2), vget_high_f32(dmax2));
          float32x4_t dmax = vcombine_f32(dmaxl, dmaxh);
#endif
          vst1q_f32(&data_out_channel[w >> 1], dmax);
        }
#else
        w = w_unroll_size;
        int num = w_unroll_size >> 3;
        const float* dr0 = r0;
        const float* dr1 = r1;
        float* dr_out = data_out_channel;
        if (num > 0) {
          asm volatile(
              "s2_max_loop:                                     @main loop\n"
              "vld1.f32   {d0-d3}, [%[dr0]]!                    @load q0, dr0\n"
              "vld1.f32   {d4-d7}, [%[dr1]]!                    @load q1, dr1\n"
              "vmax.f32   q0, q0, q2                            @max q0, q0, "
              "q2\n"
              "vmax.f32   q1, q1, q3                            @max q1, q1, "
              "q2\n"
              "vpmax.f32  d4, d0, d1                            @max d4, d0, "
              "d1\n"
              "vpmax.f32  d5, d2, d3                            @max d5, d2, "
              "d3\n"
              "vst1.f32   {d4-d5}, [%[dr_out]]!                 @vst1 q2, "
              "dr_out\n"
              "subs       %[num], #1                            @subs num, 1\n"
              "bne        s2_max_loop                           @bne num\n"
              : [dr0] "+r"(dr0), [dr1] "+r"(dr1), [dr_out] "+r"(dr_out),
                [num] "+r"(num)
              :
              : "cc", "memory", "q0", "q1", "q2", "q3");
        }
#endif  // __aarch64__
        for (; w < w_even; w += 2) {
          data_out_channel[w >> 1] =
              std::max(std::max(r0[w], r0[w + 1]), std::max(r1[w], r1[w + 1]));
        }
        for (; w < win; ++w) {  // run 0 or 1 time
          data_out_channel[w >> 1] = std::max(r0[w], r1[w]);
        }
        r0 += w_in_2;  // << 1;
        r1 += w_in_2;  // << 1;
        data_out_channel += wout;
      }
      // process remain row (odd, last row)
      for (; h < hin; h++) {  // run 0 or 1 time
        int w = 0;
#ifdef __aarch64__
        for (; w < w_unroll_size; w += 8) {
          float32x4_t dr00 = vld1q_f32(&r0[w]);
          float32x4_t dr01 = vld1q_f32(&r0[w + 4]);
#ifdef __aarch64__
          float32x4_t dmax = vpmaxq_f32(dr00, dr01);
#else
          float32x2_t dmaxl =
              vpmax_f32(vget_low_f32(dr00), vget_high_f32(dr00));
          float32x2_t dmaxh =
              vpmax_f32(vget_low_f32(dr01), vget_high_f32(dr01));
          float32x4_t dmax = vcombine_f32(dmaxl, dmaxh);
#endif
          float32x4_t dmax_cmp_zero = vmaxq_f32(dmax, vzero);
          vst1q_f32(&data_out_channel[w >> 1], dmax_cmp_zero);
        }
#else
        w = w_unroll_size;
        int num = w_unroll_size >> 3;
        const float* dr0 = r0;
        float* dr_out = data_out_channel;
        if (num > 0) {
          asm volatile(
              "s2_max_loop1:                                        @main "
              "loop\n"
              "vld1.f32   {d0-d3}, [%[dr0]]!                    @load q0, dr0\n"
              "vpmax.f32  d4, d0, d1                            @max d4, d0, "
              "d1\n"
              "vpmax.f32  d5, d2, d3                            @max d5, d2, "
              "d3\n"
              "vst1.f32   {d4-d5}, [%[dr_out]]!                 @vst1 q2, "
              "dr_out\n"
              "subs       %[num], #1                            @subs num, 1\n"
              "bne        s2_max_loop1                          @bne num\n"
              : [dr0] "+r"(dr0), [dr_out] "+r"(dr_out), [num] "+r"(num)
              :
              : "cc", "memory", "q0", "q1", "q2");
        }
#endif  // __aarch64__
        for (; w < w_even; w += 2) {
          data_out_channel[w >> 1] = std::max(std::max(r0[w], r0[w + 1]), 0.f);
        }
        for (; w < win; ++w) {  // run 0 or 1 time
          data_out_channel[w >> 1] = std::max(r0[w], 0.f);
        }
      }
    }
  }
}

void pooling2x2s2_ave(const void* din, void* dout, int num, int chout, int hout,
                      int wout, int chin, int hin, int win,
                      const std::vector<int>& ksize,
                      const std::vector<int>& strides,
                      const std::vector<int>& paddings, bool global_pooling,
                      bool exclusive, bool adaptive, bool ceil_mode,
                      bool use_quantizer, const std::string& pooling_type) {
  int size_channel_out = wout * hout;
  int size_channel_in = win * hin;
  float* data_out = static_cast<float*>(dout);
  const float* data_in = static_cast<const float*>(din);

  int w_even = (win >> 1) << 1;
  // int w_remains = w_in - w_even; // should be 0 or 1
  int h_even = (hin >> 1) << 1;
  // int h_remains = h_in - h_even; // should be 0 or 1
  int w_unroll_size = (w_even >> 3) << 3;
  // int w_unroll_remian = w_even - w_unroll_size;
  int w_in_2 = win << 1;
  float32x4_t vcoef = vdupq_n_f32(0.25f);  // divided by 4

  for (int n = 0; n < num; ++n) {
    float* data_out_batch = data_out + n * chout * size_channel_out;
    const float* data_in_batch = data_in + n * chin * size_channel_in;
#pragma omp parallel for
    for (int c = 0; c < chout; c++) {
      float* data_out_channel = data_out_batch + c * size_channel_out;
      const float* data_in_channel = data_in_batch + c * size_channel_in;
      const float* r0 = data_in_channel;
      const float* r1 = r0 + win;
      int h = 0;
      for (; h < h_even; h += 2) {
        int w = 0;
#ifdef __aarch64__
        for (; w < w_unroll_size; w += 8) {
          float32x4_t dr00 = vld1q_f32(&r0[w]);
          float32x4_t dr01 = vld1q_f32(&r0[w + 4]);
          float32x4_t dr10 = vld1q_f32(&r1[w]);
          float32x4_t dr11 = vld1q_f32(&r1[w + 4]);
          float32x4_t dsum1 = vaddq_f32(dr00, dr10);
          float32x4_t dsum2 = vaddq_f32(dr01, dr11);
#ifdef __aarch64__
          float32x4_t dsum = vpaddq_f32(dsum1, dsum2);
#else
          float32x2_t dsuml =
              vpadd_f32(vget_low_f32(dsum1), vget_high_f32(dsum1));
          float32x2_t dsumh =
              vpadd_f32(vget_low_f32(dsum2), vget_high_f32(dsum2));
          float32x4_t dsum = vcombine_f32(dsuml, dsumh);
#endif
          float32x4_t res = vmulq_f32(dsum, vcoef);
          vst1q_f32(&data_out_channel[w >> 1], res);
        }
#else
        w = w_unroll_size;
        int num = w_unroll_size >> 3;
        const float* dr0 = r0;
        const float* dr1 = r1;
        float* dr_out = data_out_channel;

        if (num > 0) {
          asm volatile(
              "1:                                               @ main loop\n"
              "vld1.f32   {d0-d3}, [%[dr0]]!                    @ load q0, "
              "dr0\n"
              "vld1.f32   {d4-d7}, [%[dr1]]!                    @ load q1, "
              "dr1\n"
              "vadd.f32   q0, q0, q2                            @ add q0, q0, "
              "q2\n"
              "vadd.f32   q1, q1, q3                            @ add q1, q1, "
              "q2\n"
              "vpadd.f32  d4, d0, d1                            @ add d4, d0, "
              "d1\n"
              "vpadd.f32  d5, d2, d3                            @ add d5, d2, "
              "d3\n"
              "vmul.f32   q2, q2, %q[vcoef]                     @ mul q2, q2, "
              "vcoef\n"
              "vst1.f32   {d4-d5}, [%[dr_out]]!                 @ vst1 q2, "
              "dr_out\n"
              "subs       %[num], #1                            @ subs num, 1\n"
              "bne        1b                                    @ bne num\n"
              : [dr0] "+r"(dr0), [dr1] "+r"(dr1), [dr_out] "+r"(dr_out),
                [vcoef] "+w"(vcoef), [num] "+r"(num)
              : "r"(dr0), "r"(dr1), "r"(dr_out), "r"(num), "w"(vcoef)
              : "cc", "memory", "q0", "q1", "q2", "q3");
        }
#endif  // __aarch64__
        for (; w < w_even; w += 2) {
          data_out_channel[w >> 1] =
              (r0[w] + r0[w + 1] + r1[w] + r1[w + 1]) / 4.f;
        }
        for (; w < win; ++w) {  // run 0 or 1 time
          data_out_channel[w >> 1] = (r0[w] + r1[w]) / 4.f;
        }
        r0 += w_in_2;  // << 1;
        r1 += w_in_2;  // << 1;
        data_out_channel += wout;
      }
      // process remain row (odd, last row)
      for (; h < hin; h++) {  // run 0 or 1 time
        int w = 0;
#ifdef __aarch64__
        for (; w < w_unroll_size; w += 8) {
          float32x4_t dr00 = vld1q_f32(&r0[w]);
          float32x4_t dr01 = vld1q_f32(&r0[w + 4]);
#ifdef __aarch64__
          float32x4_t dsum = vpaddq_f32(dr00, dr01);
#else
          float32x2_t dsuml =
              vpadd_f32(vget_low_f32(dr00), vget_high_f32(dr00));
          float32x2_t dsumh =
              vpadd_f32(vget_low_f32(dr01), vget_high_f32(dr01));
          float32x4_t dsum = vcombine_f32(dsuml, dsumh);
#endif
          float32x4_t res = vmulq_f32(dsum, vcoef);
          vst1q_f32(&data_out_channel[w >> 1], res);
        }
#else
        w = w_unroll_size;
        int num = w_unroll_size >> 3;
        const float* dr0 = r0;
        float* dr_out = data_out_channel;

        if (num > 0) {
          asm volatile(
              "1:                                               @ main loop\n"
              "vld1.f32   {d0-d3}, [%[dr0]]!                    @ load q0, "
              "dr0\n"
              "vpadd.f32  d4, d0, d1                            @ add d4, d0, "
              "d1\n"
              "vpadd.f32  d5, d2, d3                            @ add d5, d2, "
              "d3\n"
              "vmul.f32   q2, q2, %q[vcoef]                     @ mul q2, q2, "
              "vcoef\n"
              "vst1.f32   {d4-d5}, [%[dr_out]]!                 @ vst1 q2, "
              "dr_out\n"
              "subs       %[num], #1                            @ subs num, 1\n"
              "bne        1b                                    @ bne num\n"
              : [dr0] "+r"(dr0), [dr_out] "+r"(dr_out), [vcoef] "+w"(vcoef),
                [num] "+r"(num)
              : "r"(dr0), "r"(dr_out), "r"(num), "w"(vcoef)
              : "cc", "memory", "q0", "q1", "q2");
        }
#endif  // __aarch64__
        for (; w < w_even; w += 2) {
          data_out_channel[w >> 1] = (r0[w] + r0[w + 1]) / 4.f;
        }
        for (; w < win; ++w) {  // run 0 or 1 time
          data_out_channel[w >> 1] = r0[w] / 4.f;
        }
      }
    }
  }
}

void pooling3x3s1p1_max(const void* din, void* dout, int num, int chout,
                        int hout, int wout, int chin, int hin, int win,
                        const std::vector<int>& ksize,
                        const std::vector<int>& strides,
                        const std::vector<int>& paddings, bool global_pooling,
                        bool exclusive, bool adaptive, bool ceil_mode,
                        bool use_quantizer, const std::string& pooling_type) {
  // no need to pad input tensor, pad_size is not used, default border is zero
  // padded
  int ch_in = chin;
  int h_in = hin;
  int w_in = win;

  int ch_out = chout;
  int h_out = hout;
  int w_out = wout;

  int size_channel_out = w_out * h_out;
  int size_channel_in = win * hin;
  float* data_out = static_cast<float*>(dout);
  const float* data_in = static_cast<const float*>(din);

  int w_even = (w_in >> 1) << 1;
  // int w_remains = w_in - w_even; // should be 0 or 1
  int h_even = (h_in >> 1) << 1;
  // int h_remains = h_in - h_even; // should be 0 or 1
  // int w_unroll_size = (w_even >> 3) << 3;
  // int w_unroll_remian = w_even - w_unroll_size;
  int w_in_2 = w_in << 1;
  int w_unroll_size = (w_in - 2) >> 2;
  int w_unroll_remian = w_in - 2 - w_unroll_size * 4;
  float minval = std::numeric_limits<float>::lowest();
  float32x4_t vzero = vdupq_n_f32(minval);  // zero pad

  for (int n = 0; n < num; ++n) {
    float* data_out_batch = data_out + n * ch_out * size_channel_out;
    const float* data_in_batch = data_in + n * ch_in * size_channel_in;
#pragma omp parallel for
    for (int c = 0; c < ch_out; c++) {
      float* data_out_channel = data_out_batch + c * size_channel_out;
      const float* data_in_channel = data_in_batch + c * size_channel_in;
      const float* r0 = data_in_channel;
      const float* r1 = r0 + w_in;
      const float* r2 = r1 + w_in;
      int cnt_num = w_unroll_size;  // w_in / 4
      float* dr_out = data_out_channel;
      const float* dr0 = r0;
      const float* dr1 = r1;
      const float* dr2 = r2;
      int w = 0;
      int cnt = 1;
      // left
      data_out_channel[0] =
          std::max(std::max(r0[0], r0[1]), std::max(r1[0], r1[1]));
// first row with zero pad
#ifdef __aarch64__
      for (; w <= w_in - 6; w += 4) {
        float32x4_t vr0_1234 = vld1q_f32(&r0[w]);
        float32x4_t vr1_1234 = vld1q_f32(&r1[w]);
        float32x4_t vr0_5678 = vld1q_f32(&r0[w + 4]);
        float32x4_t vr1_5678 = vld1q_f32(&r1[w + 4]);
        float32x4_t vmax_1234 = vmaxq_f32(vr0_1234, vr1_1234);
        float32x4_t vmax_5678 = vmaxq_f32(vr0_5678, vr1_5678);

        float32x4_t vmax_2345 = vextq_f32(vmax_1234, vmax_5678, 1);
        float32x4_t vmax_3456 = vextq_f32(vmax_1234, vmax_5678, 2);
        float32x2_t vmax_12_34 =
            vpmax_f32(vget_low_f32(vmax_1234), vget_high_f32(vmax_1234));
        float32x2_t vmax_23_45 =
            vpmax_f32(vget_low_f32(vmax_2345), vget_high_f32(vmax_2345));
        float32x2_t vmax_34_56 =
            vpmax_f32(vget_low_f32(vmax_3456), vget_high_f32(vmax_3456));
        float32x2_t vmax_123_345 = vmax_f32(vmax_12_34, vmax_23_45);
        float32x2_t vmax_234_456 = vmax_f32(vmax_23_45, vmax_34_56);
        float32x4_t vmax = vdupq_n_f32(vget_lane_f32(vmax_123_345, 0));
        vmax = vsetq_lane_f32(vget_lane_f32(vmax_234_456, 0), vmax, 1);
        vmax = vsetq_lane_f32(vget_lane_f32(vmax_123_345, 1), vmax, 2);
        vmax = vsetq_lane_f32(vget_lane_f32(vmax_234_456, 1), vmax, 3);
        vst1q_f32(&data_out_channel[cnt], vmax);
        cnt += 4;
      }

#else
      dr_out = dr_out + 1;

      if (cnt_num > 0) {
        asm volatile(
            "1:                                             @main loop\n"
            "vld1.f32  {d0-d1}, [%[dr0]]!                     @load d0-d5, "
            "dr0\n"
            "vld1.f32  {d4-d5}, [%[dr1]]!                    @load d4-d7, dr1\n"
            "vld1.f32  {d2}, [%[dr0]]!                     @load d0-d5, dr0\n"
            "vld1.f32  {d6}, [%[dr1]]!                    @load d4-d7, dr1\n"
            "vmax.f32  q5, q0, q2                            @max "
            "r0_1234,r1_1234\n"
            "vmax.f32  d12, d2, d6                            @max "
            "r0_5678,r1_5678\n"
            //"vmov.f32  s7,s6                                 @mov s7, s6\n"
            "vext.f32  q0, q5, q6, #1                        @vext max_2345\n"
            "vext.f32  q2, q5, q6, #2                        @vext max_3456\n"
            "vpmax.f32 d2, d10, d11                          @pmax d4, "
            "max_1234, max_1234\n"
            "vpmax.f32 d3, d0, d1                            @pmax d4, "
            "max_2345, max_2345\n"
            "vpmax.f32 d6, d4, d5                            @pmax d6, "
            "max_3456, max_3456\n"
            "vmax.f32  d8, d2, d3                            @max d2, "
            "vmax_12_34, vmax_23_45\n"
            "vmax.f32  d9, d3, d6                            @max d2, "
            "vmax_23_45, vmax_34_56\n"
            "sub       %[dr0], #8                            @sub w, 8\n"
            "sub       %[dr1], #8                            @sub w, 8\n"
            // swap
            "vmov.f32  s0, s17                               @mov \n"
            "vmov.f32  s17, s18                              @mov \n"
            "vmov.f32  s18, s0                               @mov \n"
            "subs      %[cnt_num], #1                        @subs cnt_num, "
            "#1\n"
            "vst1.f32  d8, [%[dr_out]]!                      @vst1 d0, dr_out\n"
            "vst1.f32  d9, [%[dr_out]]!                      @vst1 d0, dr_out\n"
            "bne       1b                                    @bne s1_max_loop\n"
            : [dr0] "+r"(dr0), [dr1] "+r"(dr1), [dr_out] "+r"(dr_out),
              [cnt_num] "+r"(cnt_num)
            : "r"(dr0), "r"(dr1), "r"(dr_out), "r"(cnt_num)
            : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6");
      }

#endif
      // remian
      w = w_unroll_size * 4;
      for (int j = 0; j < w_unroll_remian; j++) {
        float tmp_max = std::max(r0[j + w], r1[j + w]);
        tmp_max = std::max(tmp_max, std::max(r0[j + w + 1], r1[j + w + 1]));
        tmp_max = std::max(tmp_max, std::max(r0[j + w + 2], r1[j + w + 2]));
        data_out_channel[j + w + 1] = tmp_max;
      }
      // right
      float tmp = std::max(r0[w_in - 2], r1[w_in - 2]);
      tmp = std::max(tmp, std::max(r0[w_in - 1], r1[w_in - 1]));
      data_out_channel[w_out - 1] = tmp;

      // r0 = r1;
      // r1 = r0 + w_in;
      // r2 = r1 + w_in;
      data_out_channel += w_out;
      int h = 0;
      for (; h < h_in - 2; h += 1) {
        // deal with left pad
        float maxr0 = std::max(r0[0], r0[1]);
        float maxr1 = std::max(r1[0], r1[1]);
        float maxr2 = std::max(r2[0], r2[1]);
        data_out_channel[0] = std::max(std::max(maxr0, maxr1), maxr2);
#ifdef __aarch64__
        w = 0;
        cnt = 1;
        for (; w <= w_in - 6; w += 4) {
          float32x4_t vr0_1234 = vld1q_f32(&r0[w]);
          float32x4_t vr1_1234 = vld1q_f32(&r1[w]);
          float32x4_t vr2_1234 = vld1q_f32(&r2[w]);
          float32x4_t vr0_5678 = vld1q_f32(&r0[w + 4]);
          float32x4_t vr1_5678 = vld1q_f32(&r1[w + 4]);
          float32x4_t vr2_5678 = vld1q_f32(&r2[w + 4]);
          float32x4_t vmax_1234 = vmaxq_f32(vr0_1234, vr1_1234);
          vmax_1234 = vmaxq_f32(vmax_1234, vr2_1234);
          float32x4_t vmax_5678 = vmaxq_f32(vr0_5678, vr1_5678);
          vmax_5678 = vmaxq_f32(vmax_5678, vr2_5678);

          float32x4_t vmax_2345 = vextq_f32(vmax_1234, vmax_5678, 1);
          float32x4_t vmax_3456 = vextq_f32(vmax_1234, vmax_5678, 2);
          float32x2_t vmax_12_34 =
              vpmax_f32(vget_low_f32(vmax_1234), vget_high_f32(vmax_1234));
          float32x2_t vmax_23_45 =
              vpmax_f32(vget_low_f32(vmax_2345), vget_high_f32(vmax_2345));
          float32x2_t vmax_34_56 =
              vpmax_f32(vget_low_f32(vmax_3456), vget_high_f32(vmax_3456));
          float32x2_t vmax_123_345 = vmax_f32(vmax_12_34, vmax_23_45);
          float32x2_t vmax_234_456 = vmax_f32(vmax_23_45, vmax_34_56);
          float32x4_t vmax = vdupq_n_f32(vget_lane_f32(vmax_123_345, 0));
          vmax = vsetq_lane_f32(vget_lane_f32(vmax_234_456, 0), vmax, 1);
          vmax = vsetq_lane_f32(vget_lane_f32(vmax_123_345, 1), vmax, 2);
          vmax = vsetq_lane_f32(vget_lane_f32(vmax_234_456, 1), vmax, 3);
          vst1q_f32(&data_out_channel[cnt], vmax);
          cnt += 4;
        }
#else
        dr_out = data_out_channel + 1;
        dr0 = r0;
        dr1 = r1;
        dr2 = r2;
        cnt_num = w_unroll_size;
        if (cnt_num > 0) {
          asm volatile(
              "1:                                                     @main "
              "loop\n"
              "vld1.f32  {d0-d1}, [%[dr0]]!                     @load d0-d5, "
              "dr0\n"
              "vld1.f32  {d4-d5}, [%[dr1]]!                    @load d4-d7, "
              "dr1\n"
              "vld1.f32  {d8-d9}, [%[dr2]]!                    @load d4-d7, "
              "dr1\n"
              "vld1.f32  {d2}, [%[dr0]]!                     @load d0-d5, dr0\n"
              "vld1.f32  {d6}, [%[dr1]]!                    @load d4-d7, dr1\n"
              "vld1.f32  {d10}, [%[dr2]]!                  @load d4-d7, dr1\n"
              "vmax.f32  q7, q0, q2                            @max "
              "r0_1234,r1_1234\n"
              "vmax.f32  d16, d2, d6                            @max "
              "r0_5678,r1_5678\n"
              "vmax.f32  q3, q7, q4                            @max "
              "r0_1234,r1_1234\n"
              "vmax.f32  d12, d16, d10                            @max "
              "r0_5678,r1_5678\n"
              //"vmov.f32  s7,s6                                 @mov s7, s6\n"
              "vext.f32  q0, q3, q6, #1                        @vext max_2345\n"
              "vext.f32  q2, q3, q6, #2                        @vext max_3456\n"
              "vpmax.f32 d2, d6, d7                            @pmax d4, "
              "max_1234, max_1234\n"
              "vpmax.f32 d3, d0, d1                            @pmax d4, "
              "max_2345, max_2345\n"
              "vpmax.f32 d6, d4, d5                            @pmax d6, "
              "max_3456, max_3456\n"
              "vmax.f32  d8, d2, d3                            @max d2, "
              "vmax_12_34, vmax_23_45\n"
              "vmax.f32  d9, d3, d6                            @max d2, "
              "vmax_23_45, vmax_34_56\n"
              "sub       %[dr0], #8                            @sub w, 8\n"
              "sub       %[dr1], #8                            @sub w, 8\n"
              "sub       %[dr2], #8                            @sub w, 8\n"
              // swap
              "vmov.f32  s0, s17                               @mov \n"
              "vmov.f32  s17, s18                              @mov \n"
              "vmov.f32  s18, s0                               @mov \n"
              "subs      %[cnt_num], #1                        @subs cnt_num, "
              "#1\n"
              "vst1.f32  d8, [%[dr_out]]!                      @vst1 d0, "
              "dr_out\n"
              "vst1.f32  d9, [%[dr_out]]!                      @vst1 d0, "
              "dr_out\n"
              "bne       1b                                    @ bne "
              "s1_max_loop\n"
              : [dr0] "+r"(dr0), [dr1] "+r"(dr1), [dr2] "+r"(dr2),
                [dr_out] "+r"(dr_out), [cnt_num] "+r"(cnt_num)
              : "r"(dr0), "r"(dr1), "r"(dr_out), "r"(cnt_num)
              : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7",
                "q8");
        }
#endif
        // remian
        w = w_unroll_size * 4;
        for (int j = 0; j < w_unroll_remian; j++) {
          float tmp_max = std::max(r0[j + w], r1[j + w]);
          tmp_max = std::max(tmp_max, std::max(r0[j + w + 1], r1[j + w + 1]));
          tmp_max = std::max(tmp_max, std::max(r0[j + w + 2], r1[j + w + 2]));
          tmp_max = std::max(tmp_max, std::max(r2[j + w], r2[j + w + 1]));
          tmp_max = std::max(tmp_max, r2[j + w + 2]);
          data_out_channel[j + w + 1] = tmp_max;
        }
        // right
        tmp = std::max(r0[w_in - 2], r1[w_in - 2]);
        tmp = std::max(tmp, std::max(r0[w_in - 1], r1[w_in - 1]));
        tmp = std::max(tmp, std::max(r2[w_in - 2], r2[w_in - 1]));
        data_out_channel[w_out - 1] = tmp;

        r0 = r1;
        r1 = r2;
        r2 = r1 + w_in;
        data_out_channel += w_out;
      }

      // the last two line
      float maxr0 = std::max(r0[0], r0[1]);
      float maxr1 = std::max(r1[0], r1[1]);
      data_out_channel[0] = std::max(maxr0, maxr1);
#ifdef __aarch64__
      w = 0;
      cnt = 1;
      for (; w <= w_in - 6; w += 4) {
        float32x4_t vr0_1234 = vld1q_f32(&r0[w]);
        float32x4_t vr1_1234 = vld1q_f32(&r1[w]);
        float32x4_t vr0_5678 = vld1q_f32(&r0[w + 4]);
        float32x4_t vr1_5678 = vld1q_f32(&r1[w + 4]);
        float32x4_t vmax_1234 = vmaxq_f32(vr0_1234, vr1_1234);
        float32x4_t vmax_5678 = vmaxq_f32(vr0_5678, vr1_5678);

        float32x4_t vmax_2345 = vextq_f32(vmax_1234, vmax_5678, 1);
        float32x4_t vmax_3456 = vextq_f32(vmax_1234, vmax_5678, 2);
        float32x2_t vmax_12_34 =
            vpmax_f32(vget_low_f32(vmax_1234), vget_high_f32(vmax_1234));
        float32x2_t vmax_23_45 =
            vpmax_f32(vget_low_f32(vmax_2345), vget_high_f32(vmax_2345));
        float32x2_t vmax_34_56 =
            vpmax_f32(vget_low_f32(vmax_3456), vget_high_f32(vmax_3456));
        float32x2_t vmax_123_345 = vmax_f32(vmax_12_34, vmax_23_45);
        float32x2_t vmax_234_456 = vmax_f32(vmax_23_45, vmax_34_56);
        float32x4_t vmax = vdupq_n_f32(vget_lane_f32(vmax_123_345, 0));
        vmax = vsetq_lane_f32(vget_lane_f32(vmax_234_456, 0), vmax, 1);
        vmax = vsetq_lane_f32(vget_lane_f32(vmax_123_345, 1), vmax, 2);
        vmax = vsetq_lane_f32(vget_lane_f32(vmax_234_456, 1), vmax, 3);
        vst1q_f32(&data_out_channel[cnt], vmax);
        cnt += 4;
      }
#else
      dr_out = data_out_channel + 1;
      dr0 = r0;
      dr1 = r1;
      cnt_num = w_unroll_size;
      if (cnt_num > 0) {
        asm volatile(
            "1:                                 @main loop\n"
            "vld1.f32  {d0-d1}, [%[dr0]]!                     @load d0-d5, "
            "dr0\n"
            "vld1.f32  {d4-d5}, [%[dr1]]!                    @load d4-d7, dr1\n"
            "vld1.f32  {d2}, [%[dr0]]!                     @load d0-d5, dr0\n"
            "vld1.f32  {d6}, [%[dr1]]!                    @load d4-d7, dr1\n"
            "vmax.f32  q5, q0, q2                            @max "
            "r0_1234,r1_1234\n"
            "vmax.f32  d12, d2, d6                            @max "
            "r0_5678,r1_5678\n"
            //"vmov.f32  s7,s6                                 @mov s7, s6\n"
            "vext.f32  q0, q5, q6, #1                        @vext max_2345\n"
            "vext.f32  q2, q5, q6, #2                        @vext max_3456\n"
            "vpmax.f32 d2, d10, d11                          @pmax d4, "
            "max_1234, max_1234\n"
            "vpmax.f32 d3, d0, d1                            @pmax d4, "
            "max_2345, max_2345\n"
            "vpmax.f32 d6, d4, d5                            @pmax d6, "
            "max_3456, max_3456\n"
            "vmax.f32  d8, d2, d3                            @max d2, "
            "vmax_12_34, vmax_23_45\n"
            "vmax.f32  d9, d3, d6                            @max d2, "
            "vmax_23_45, vmax_34_56\n"
            "sub       %[dr0], #8                            @sub w, 8\n"
            "sub       %[dr1], #8                            @sub w, 8\n"
            // swap
            "vmov.f32  s0, s17                               @mov \n"
            "vmov.f32  s17, s18                              @mov \n"
            "vmov.f32  s18, s0                               @mov \n"
            "subs      %[cnt_num], #1                        @subs cnt_num, "
            "#1\n"
            "vst1.f32  d8, [%[dr_out]]!                      @vst1 d0, dr_out\n"
            "vst1.f32  d9, [%[dr_out]]!                      @vst1 d0, dr_out\n"
            "bne       1b                                    @bne s1_max_loop\n"
            : [dr0] "+r"(dr0), [dr1] "+r"(dr1), [dr_out] "+r"(dr_out),
              [cnt_num] "+r"(cnt_num)
            : "r"(dr0), "r"(dr1), "r"(dr_out), "r"(cnt_num)
            : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6");
      }
#endif
      // remian
      w = w_unroll_size * 4;
      for (int j = 0; j < w_unroll_remian; j++) {
        float tmp_max = std::max(r0[j + w], r1[j + w]);
        tmp_max = std::max(tmp_max, std::max(r0[j + w + 1], r1[j + w + 1]));
        tmp_max = std::max(tmp_max, std::max(r0[j + w + 2], r1[j + w + 2]));
        data_out_channel[j + w + 1] = tmp_max;
      }
      tmp = std::max(r0[w_in - 2], r1[w_in - 2]);
      tmp = std::max(tmp, std::max(r0[w_in - 1], r1[w_in - 1]));
      data_out_channel[w_out - 1] = tmp;
    }
  }
}

void pooling3x3s1p1_ave(const void* din, void* dout, int num, int chout,
                        int hout, int wout, int chin, int hin, int win,
                        const std::vector<int>& ksize,
                        const std::vector<int>& strides,
                        const std::vector<int>& paddings, bool global_pooling,
                        bool exclusive, bool adaptive, bool ceil_mode,
                        bool use_quantizer, const std::string& pooling_type) {
  int w_in = win;
  int h_in = hin;
  int ch_in = chin;

  int w_out = wout;
  int h_out = hout;
  int ch_out = chout;

  int size_channel_out = w_out * h_out;
  int size_channel_in = w_in * h_in;
  float* data_out = static_cast<float*>(dout);
  const float* data_in = static_cast<const float*>(din);

  int w_even = (w_in >> 1) << 1;
  int h_even = (h_in >> 1) << 1;
  int w_in_2 = w_in << 1;
  int w_unroll_size = (w_in - 2) >> 2;
  int w_unroll_remian = w_in - 2 - w_unroll_size * 4;
  float32x4_t vzero = vdupq_n_f32(0.f);        // zero pad
  float32x4_t vcoef = vdupq_n_f32(1.f / 9.f);  // zero pad

  for (int n = 0; n < num; ++n) {
    float* data_out_batch = data_out + n * ch_out * size_channel_out;
    const float* data_in_batch = data_in + n * ch_in * size_channel_in;
#pragma omp parallel for
    for (int c = 0; c < ch_out; c++) {
      float* data_out_channel = data_out_batch + c * size_channel_out;
      const float* data_in_channel = data_in_batch + c * size_channel_in;
      const float* r0 = data_in_channel;
      const float* r1 = r0 + w_in;
      const float* r2 = r1 + w_in;
      int cnt_num = w_unroll_size;  // w_in / 4
      float* dr_out = data_out_channel;
      const float* dr0 = r0;
      const float* dr1 = r1;
      const float* dr2 = r2;
      int w = 0;
      int cnt = 1;
      // left
      data_out_channel[0] = (r0[0] + r0[1] + r1[0] + r1[1]) / 9.f;
// first row with zero pad
#ifdef __aarch64__
      for (; w <= w_in - 6; w += 4) {
        float32x4_t vr0_1234 = vld1q_f32(&r0[w]);
        float32x4_t vr1_1234 = vld1q_f32(&r1[w]);
        float32x4_t vr0_5678 = vld1q_f32(&r0[w + 4]);
        float32x4_t vr1_5678 = vld1q_f32(&r1[w + 4]);
        float32x4_t vsum_1234 = vaddq_f32(vr0_1234, vr1_1234);
        float32x4_t vsum_5678 = vaddq_f32(vr0_5678, vr1_5678);

        float32x4_t vsum_2345 = vextq_f32(vsum_1234, vsum_5678, 1);
        float32x4_t vsum_3456 = vextq_f32(vsum_1234, vsum_5678, 2);
        float32x4_t vsum = vaddq_f32(vsum_1234, vsum_2345);
        vsum = vaddq_f32(vsum, vsum_3456);
        vsum = vmulq_f32(vsum, vcoef);
        vst1q_f32(&data_out_channel[cnt], vsum);
        cnt += 4;
      }

#else
      dr_out = dr_out + 1;

      if (cnt_num > 0) {
        asm volatile(
            "1:                                    @main loop\n"
            "vld1.f32  {d0-d1}, [%[dr0]]!                     @load d0-d5, "
            "dr0\n"
            "vld1.f32  {d4-d5}, [%[dr1]]!                    @load d4-d7, dr1\n"
            "vld1.f32  {d2}, [%[dr0]]!                     @load d0-d5, dr0\n"
            "vld1.f32  {d6}, [%[dr1]]!                    @load d4-d7, dr1\n"
            "vadd.f32  q5, q0, q2                            @max "
            "r0_1234,r1_1234\n"
            "vadd.f32  d12, d2, d6                            @max "
            "r0_5678,r1_5678\n"
            //"vmov.f32  s7,s6                                 @mov s7, s6\n"
            "vext.f32  q0, q5, q6, #1                        @vext max_2345\n"
            "vext.f32  q2, q5, q6, #2                        @vext max_3456\n"
            "vadd.f32  q1, q5, q0                            @add 1234 + 2345\n"
            "vadd.f32  q1, q1, q2                            @add + 3456\n"
            "vmul.f32  q4, q1, %q[vcoef]                     @mul * 1/9.f \n"
            "sub       %[dr0], #8                            @sub w, 8\n"
            "sub       %[dr1], #8                            @sub w, 8\n"
            "subs      %[cnt_num], #1                        @subs cnt_num, "
            "#1\n"
            "vst1.f32  d8, [%[dr_out]]!                      @vst1 d0, dr_out\n"
            "vst1.f32  d9, [%[dr_out]]!                      @vst1 d0, dr_out\n"
            "bne       1b                                    @bne s1_max_loop\n"
            : [dr0] "+r"(dr0), [dr1] "+r"(dr1), [dr_out] "+r"(dr_out),
              [cnt_num] "+r"(cnt_num), [vcoef] "+w"(vcoef)
            : "r"(dr0), "r"(dr1), "r"(dr_out), "r"(cnt_num)
            : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6");
      }

#endif
      // remian
      w = w_unroll_size * 4;
      for (int j = 0; j < w_unroll_remian; j++) {
        float tmp_sum = r0[j + w] + r1[j + w];
        tmp_sum += (r0[j + w + 1] + r1[j + w + 1]);
        tmp_sum += (r0[j + w + 2] + r1[j + w + 2]);
        data_out_channel[j + w + 1] = tmp_sum / 9.f;
      }
      // right
      float tmp = r0[w_in - 2] + r1[w_in - 2];
      tmp += (r0[w_in - 1] + r1[w_in - 1]);
      data_out_channel[w_out - 1] = tmp / 9.f;

      // r0 = r1;
      // r1 = r0 + w_in;
      // r2 = r1 + w_in;
      data_out_channel += w_out;
      int h = 0;
      for (; h < h_in - 2; h += 1) {
        // deal with left pad
        float maxr0 = r0[0] + r0[1];
        float maxr1 = r1[0] + r1[1];
        float maxr2 = r2[0] + r2[1];
        data_out_channel[0] = (maxr0 + maxr1 + maxr2) / 9.f;
#ifdef __aarch64__
        w = 0;
        cnt = 1;
        for (; w <= w_in - 6; w += 4) {
          float32x4_t vr0_1234 = vld1q_f32(&r0[w]);
          float32x4_t vr1_1234 = vld1q_f32(&r1[w]);
          float32x4_t vr2_1234 = vld1q_f32(&r2[w]);
          float32x4_t vr0_5678 = vld1q_f32(&r0[w + 4]);
          float32x4_t vr1_5678 = vld1q_f32(&r1[w + 4]);
          float32x4_t vr2_5678 = vld1q_f32(&r2[w + 4]);
          float32x4_t vsum_1234 = vaddq_f32(vr0_1234, vr1_1234);
          vsum_1234 = vaddq_f32(vsum_1234, vr2_1234);
          float32x4_t vsum_5678 = vaddq_f32(vr0_5678, vr1_5678);
          vsum_5678 = vaddq_f32(vsum_5678, vr2_5678);

          float32x4_t vsum_2345 = vextq_f32(vsum_1234, vsum_5678, 1);
          float32x4_t vsum_3456 = vextq_f32(vsum_1234, vsum_5678, 2);
          float32x4_t vsum = vaddq_f32(vsum_1234, vsum_2345);
          vsum = vaddq_f32(vsum, vsum_3456);
          vsum = vmulq_f32(vsum, vcoef);
          vst1q_f32(&data_out_channel[cnt], vsum);
          cnt += 4;
        }
#else
        dr_out = data_out_channel + 1;
        dr0 = r0;
        dr1 = r1;
        dr2 = r2;
        cnt_num = w_unroll_size;
        if (cnt_num > 0) {
          asm volatile(
              "1:                                    @main loop\n"
              "vld1.f32  {d0-d1}, [%[dr0]]!                     @load d0-d5, "
              "dr0\n"
              "vld1.f32  {d4-d5}, [%[dr1]]!                    @load d4-d7, "
              "dr1\n"
              "vld1.f32  {d8-d9}, [%[dr2]]!                    @load d4-d7, "
              "dr1\n"
              "vld1.f32  {d2}, [%[dr0]]!                     @load d0-d5, dr0\n"
              "vld1.f32  {d6}, [%[dr1]]!                    @load d4-d7, dr1\n"
              "vld1.f32  {d10}, [%[dr2]]!                  @load d4-d7, dr1\n"
              "vadd.f32  q7, q0, q2                            @max "
              "r0_1234,r1_1234\n"
              "vadd.f32  d16, d2, d6                            @max "
              "r0_5678,r1_5678\n"
              "vadd.f32  q3, q7, q4                            @max "
              "r0_1234,r1_1234\n"
              "vadd.f32  d12, d16, d10                            @max "
              "r0_5678,r1_5678\n"
              //"vmov.f32  s7,s6                                 @mov s7, s6\n"
              "vext.f32  q0, q3, q6, #1                        @vext max_2345\n"
              "vext.f32  q2, q3, q6, #2                        @vext max_3456\n"
              "vadd.f32  q1, q3, q0                            @add 1234 + "
              "2345\n"
              "vadd.f32  q1, q1, q2                            @add + 3456\n"
              "vmul.f32  q4, q1, %q[vcoef]                     @mul * 1/9.f \n"
              "sub       %[dr0], #8                            @sub w, 8\n"
              "sub       %[dr1], #8                            @sub w, 8\n"
              "sub       %[dr2], #8                            @sub w, 8\n"
              "subs      %[cnt_num], #1                        @subs cnt_num, "
              "#1\n"
              "vst1.f32  d8, [%[dr_out]]!                      @vst1 d0, "
              "dr_out\n"
              "vst1.f32  d9, [%[dr_out]]!                      @vst1 d0, "
              "dr_out\n"
              "bne       1b                                   @bne "
              "s1_max_loop\n"
              : [dr0] "+r"(dr0), [dr1] "+r"(dr1), [dr2] "+r"(dr2),
                [dr_out] "+r"(dr_out), [cnt_num] "+r"(cnt_num),
                [vcoef] "+w"(vcoef)
              : "r"(dr0), "r"(dr1), "r"(dr_out), "r"(cnt_num)
              : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7",
                "q8");
        }
#endif
        // remian
        w = w_unroll_size * 4;
        for (int j = 0; j < w_unroll_remian; j++) {
          float tmp_sum = r0[j + w] + r1[j + w];
          tmp_sum += (r0[j + w + 1] + r1[j + w + 1]);
          tmp_sum += (r0[j + w + 2] + r1[j + w + 2]);
          tmp_sum += (r2[j + w + 1] + r2[j + w + 2]);
          tmp_sum += r2[j + w];
          data_out_channel[j + w + 1] = tmp_sum / 9.f;
        }
        // right
        tmp = r0[w_in - 2] + r1[w_in - 2];
        tmp += (r0[w_in - 1] + r1[w_in - 1]);
        tmp += (r2[w_in - 2] + r2[w_in - 1]);
        data_out_channel[w_out - 1] = tmp / 9.f;

        r0 = r1;
        r1 = r2;
        r2 = r1 + w_in;
        data_out_channel += w_out;
      }

      // the last two line
      float maxr0 = (r0[0] + r0[1]);
      float maxr1 = (r1[0] + r1[1]);
      data_out_channel[0] = (maxr0 + maxr1) / 9.f;
#ifdef __aarch64__
      w = 0;
      cnt = 1;
      for (; w <= w_in - 6; w += 4) {
        float32x4_t vr0_1234 = vld1q_f32(&r0[w]);
        float32x4_t vr1_1234 = vld1q_f32(&r1[w]);
        float32x4_t vr0_5678 = vld1q_f32(&r0[w + 4]);
        float32x4_t vr1_5678 = vld1q_f32(&r1[w + 4]);
        float32x4_t vsum_1234 = vaddq_f32(vr0_1234, vr1_1234);
        float32x4_t vsum_5678 = vaddq_f32(vr0_5678, vr1_5678);

        float32x4_t vsum_2345 = vextq_f32(vsum_1234, vsum_5678, 1);
        float32x4_t vsum_3456 = vextq_f32(vsum_1234, vsum_5678, 2);
        float32x4_t vsum = vaddq_f32(vsum_1234, vsum_2345);
        vsum = vaddq_f32(vsum, vsum_3456);
        vsum = vmulq_f32(vsum, vcoef);
        vst1q_f32(&data_out_channel[cnt], vsum);
        cnt += 4;
      }
#else
      dr_out = data_out_channel + 1;
      dr0 = r0;
      dr1 = r1;
      cnt_num = w_unroll_size;
      if (cnt_num > 0) {
        asm volatile(
            "1:                                 @main loop\n"
            "vld1.f32  {d0-d1}, [%[dr0]]!                     @load d0-d5, "
            "dr0\n"
            "vld1.f32  {d4-d5}, [%[dr1]]!                    @load d4-d7, dr1\n"
            "vld1.f32  {d2}, [%[dr0]]!                     @load d0-d5, dr0\n"
            "vld1.f32  {d6}, [%[dr1]]!                    @load d4-d7, dr1\n"
            "vadd.f32  q5, q0, q2                            @max "
            "r0_1234,r1_1234\n"
            "vadd.f32  d12, d2, d6                            @max "
            "r0_5678,r1_5678\n"
            //"vmov.f32  s7,s6                                 @mov s7, s6\n"
            "vext.f32  q0, q5, q6, #1                        @vext max_2345\n"
            "vext.f32  q2, q5, q6, #2                        @vext max_3456\n"
            "vadd.f32  q1, q5, q0                            @add 1234 + 2345\n"
            "vadd.f32  q1, q1, q2                            @add + 3456\n"
            "vmul.f32  q4, q1, %q[vcoef]                     @mul * 1/9.f \n"
            "sub       %[dr0], #8                            @sub w, 8\n"
            "sub       %[dr1], #8                            @sub w, 8\n"
            "subs      %[cnt_num], #1                        @subs cnt_num, "
            "#1\n"
            "vst1.f32  d8, [%[dr_out]]!                      @vst1 d0, dr_out\n"
            "vst1.f32  d9, [%[dr_out]]!                      @vst1 d0, dr_out\n"
            "bne       1b                                   @bne s1_max_loop\n"
            : [dr0] "+r"(dr0), [dr1] "+r"(dr1), [dr_out] "+r"(dr_out),
              [cnt_num] "+r"(cnt_num), [vcoef] "+w"(vcoef)
            : "r"(dr0), "r"(dr1), "r"(dr_out), "r"(cnt_num)
            : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6");
      }
#endif
      // remian
      w = w_unroll_size * 4;
      for (int j = 0; j < w_unroll_remian; j++) {
        float tmp_sum = r0[j + w] + r1[j + w];
        tmp_sum += (r0[j + w + 1] + r1[j + w + 1]);
        tmp_sum += (r0[j + w + 2] + r1[j + w + 2]);
        data_out_channel[j + w + 1] = tmp_sum / 9.f;
      }
      // right
      tmp = r0[w_in - 2] + r1[w_in - 2];
      tmp += (r0[w_in - 1] + r1[w_in - 1]);
      data_out_channel[w_out - 1] = tmp / 9.f;
    }
  }
}

void pooling3x3s2p1_max(const void* din, void* dout, int num, int chout,
                        int hout, int wout, int chin, int hin, int win,
                        const std::vector<int>& ksize,
                        const std::vector<int>& strides,
                        const std::vector<int>& paddings, bool global_pooling,
                        bool exclusive, bool adaptive, bool ceil_mode,
                        bool use_quantizer, const std::string& pooling_type) {
  int size_channel_out = wout * hout;
  int size_channel_in = win * hin;
  float* data_out = static_cast<float*>(dout);
  const float* data_in = static_cast<const float*>(din);

  int kernel_h = ksize[0];
  int kernel_w = ksize[1];
  int stride_h = strides[0];
  int stride_w = strides[1];
  int pad_h = paddings[0];
  int pad_w = paddings[1];

  int pad_top = pad_h;
  int pad_left = pad_w;
  int w_needed = wout * 2 + 1;
  int h_needed = hout * 2 + 1;
  int pad_right = w_needed - win - pad_left;
  int pad_bottom = h_needed - hin - pad_top;
  int w_even = (win >> 1) << 1;
  int h_even = (hin >> 1) << 1;
  int w_in_2 = win << 1;
  float minval = std::numeric_limits<float>::lowest();
  float32x4_t vzero = vdupq_n_f32(minval);  // zero pad
  int cnt_col = (win - 1) / 8;
  // remain
  int remain = ((win - 1) % 8) / 2;

  for (int n = 0; n < num; ++n) {
    float* data_out_batch = data_out + n * chout * size_channel_out;
    const float* data_in_batch = data_in + n * chin * size_channel_in;
#pragma omp parallel for
    for (int c = 0; c < chout; c++) {
      float* data_out_channel = data_out_batch + c * size_channel_out;
      const float* data_in_channel = data_in_batch + c * size_channel_in;
      const float* r0 = data_in_channel;
      const float* r1 = r0 + win;
      const float* r2 = r1 + win;
      float* dr_out = data_out_channel;
      const float* dr0 = r0;
      const float* dr1 = r1;
      const float* dr2 = r2;
      int w = 1;
      int cnt = 1;
      int cnt_num = cnt_col;
      int cnt_num1 = remain;
      data_out_channel[0] =
          std::max(std::max(r0[0], r0[1]), std::max(r1[0], r1[1]));
// first row with zero pad
#ifdef __aarch64__
      for (; w < win - 8; w += 8) {
        float32x4_t vr0_1234 = vld1q_f32(&r0[w]);
        float32x4_t vr0_5678 = vld1q_f32(&r0[w + 4]);
        float32x4_t vr0_9101112 = vld1q_f32(&r0[w + 8]);
        float32x4_t vr1_1234 = vld1q_f32(&r1[w]);
        float32x4_t vr1_5678 = vld1q_f32(&r1[w + 4]);
        float32x4_t vr1_9101112 = vld1q_f32(&r1[w + 8]);
        float32x4_t vmax_1234 = vmaxq_f32(vr0_1234, vr1_1234);
        float32x4_t vmax_5678 = vmaxq_f32(vr0_5678, vr1_5678);
        float32x4_t vmax_9101112 = vmaxq_f32(vr0_9101112, vr1_9101112);
        float32x4_t vmax_2345 = vextq_f32(vmax_1234, vmax_5678, 1);
        float32x4_t vmax_6789 = vextq_f32(vmax_5678, vmax_9101112, 1);
        float32x2_t vmax_12_34 =
            vpmax_f32(vget_low_f32(vmax_1234), vget_high_f32(vmax_1234));
        float32x2_t vmax_23_45 =
            vpmax_f32(vget_low_f32(vmax_2345), vget_high_f32(vmax_2345));
        float32x2_t vmax_56_78 =
            vpmax_f32(vget_low_f32(vmax_5678), vget_high_f32(vmax_5678));
        float32x2_t vmax_67_89 =
            vpmax_f32(vget_low_f32(vmax_6789), vget_high_f32(vmax_6789));
        float32x2_t vmax_123_345 = vmax_f32(vmax_12_34, vmax_23_45);
        float32x2_t vmax_567_789 = vmax_f32(vmax_56_78, vmax_67_89);
        vst1_f32(&data_out_channel[cnt], vmax_123_345);
        vst1_f32(&data_out_channel[cnt + 2], vmax_567_789);
        cnt += 4;
      }
      for (; w < w_even - 1; w += 2) {
        float32x4_t vr0 = vld1q_f32(&r0[w]);
        float32x4_t vr1 = vld1q_f32(&r1[w]);
        vr0 = vsetq_lane_f32(minval, vr0, 3);
        vr1 = vsetq_lane_f32(minval, vr1, 3);
        float32x4_t vmax1 = vmaxq_f32(vr0, vr1);
        float32x2_t vmax2 =
            vpmax_f32(vget_low_f32(vmax1), vget_high_f32(vmax1));
        vmax2 = vpmax_f32(vmax2, vmax2);
        data_out_channel[cnt] = vget_lane_f32(vmax2, 0);
        cnt++;
      }
#else
      dr0 = dr0 + 1;
      dr1 = dr1 + 1;
      dr_out = dr_out + 1;
      if (cnt_num > 0 || cnt_num1 > 0) {
        asm volatile(
            "cmp       %[cnt_num], #0                        @cmp cnt_num, 0\n"
            "ble       3f                                  @ble exit\n"
            "1:                                    @main loop\n"
            "vld1.f32  {d0-d3}, [%[dr0]]!                     @load d0-d5, "
            "dr0\n"
            "vld1.f32  {d6-d9}, [%[dr1]]!                    @load d4-d7, dr1\n"
            "vld1.f32  {d4-d5}, [%[dr0]]!                     @load d0-d5, "
            "dr0\n"
            "vld1.f32  {d10-d11}, [%[dr1]]!                    @load d4-d7, "
            "dr1\n"
            "vmax.f32  q6, q0, q3                            @max "
            "r0_1234,r1_1234\n"
            "vmax.f32  q7, q1, q4                            @max "
            "r0_5678,r1_5678\n"
            "vmax.f32  q8, q2, q5                            @max "
            "r0_9101112,r1_9101112\n"
            //"vmov.f32  s7,s6                                 @mov s7, s6\n"
            "vext.f32  q0, q6, q7, #1                        @vext max_2345\n"
            "vext.f32  q1, q7, q8, #1                        @vext max_6789\n"
            "vpmax.f32 d4, d12, d13                          @pmax d4, "
            "vmax_1234, vmax_1234\n"
            "vpmax.f32 d6, d14, d15                          @pmax d6, "
            "vmax_5678, vmax_5678\n"
            "vpmax.f32 d5, d0, d1                            @pmax d5, "
            "vmax_2345, vmax_2345\n"
            "vpmax.f32 d7, d2, d3                            @pmax d7, "
            "vmax_6789, vmax_6789\n"
            "vmax.f32 d8, d4, d5                             @max d2, "
            "vmax_12_34, vmax_23_45\n"
            "vmax.f32 d9, d6, d7                             @max d2, "
            "vmax_56_78, vmax_67_89\n"
            "sub       %[dr0], #16                           @add w, 8\n"
            "sub       %[dr1], #16                           @add w, 8\n"
            "vst1.f32  d8, [%[dr_out]]!                      @vst1 d0, dr_out\n"
            "vst1.f32  d9, [%[dr_out]]!                      @vst1 d0, dr_out\n"
            "subs      %[cnt_num], #1                            @subs "
            "cnt_num, #1\n"
            "bne       1b                                   @bne s3_max_loop\n"
            "3:                                           @loop \n"
            "cmp       %[cnt_num1], #0                           @cmp cnt_num, "
            "0\n"
            "ble       4f                                  @ble exit\n"
            "2:                                             @main loop\n"
            "vld1.f32  {d0-d1}, [%[dr0]]!                     @load d0-d1, "
            "dr0\n"
            "vld1.f32  {d2-d3}, [%[dr1]]!                     @load d2-d3, "
            "dr1\n"
            "vmov.f32  s3,s2                                 @movs3, s2\n"
            "vmov.f32  s7,s6                                 @movs7, s6\n"
            "vmax.f32  q0, q0, q1                            @max q0, q0, q1\n"
            "vpmax.f32 d0, d0, d1                            @pmax d0, d0,d1\n"
            "vpmax.f32 d0, d0, d0                            @pmax d0, d0, d0\n"
            "vst1.f32  d0[0], [%[dr_out]]!                   @vst  d0[0], "
            "dr_out\n"
            "sub       %[dr0], #8                            @add w, 6\n"
            "sub       %[dr1], #8                            @add w, 6\n"
            "subs      %[cnt_num1], #1                           @subs "
            "cnt_num, #1\n"
            "bne       2b                                   @bne "
            "s3_max_loop_1\n"
            "4:                                           @exit\n"
            : [dr0] "+r"(dr0), [dr1] "+r"(dr1), [dr_out] "+r"(dr_out),
              [cnt_num] "+r"(cnt_num), [cnt_num1] "+r"(cnt_num1)
            : "r"(dr0), "r"(dr1), "r"(dr_out), "r"(cnt_num), "r"(cnt_num1)
            : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7",
              "q8", "q9");
      }
// printf("cnt_num: %d, cnt_num1: %d \n",cnt_num, cnt_num1);
#endif
      // int w = w_even - 1;
      if (pad_right) {
        // deal with right pad
        int wstart = (w_even >> 1) * stride_w - pad_w;
        int wend = std::min(std::min(wstart + kernel_w, win + pad_w), win);
        float tmp = r0[wstart];  // std::numeric_limits<float>::min();
        for (int i = wstart; i < wend; i++) {  // only run 1 or 2 times
          tmp = std::max(tmp, std::max(r0[i], r1[i]));
        }
        data_out_channel[w_even >> 1] = tmp;
        // cnt ++;
      }

      r0 = r1;
      r1 = r0 + win;
      r2 = r1 + win;
      data_out_channel += wout;
      int h = 2;
      for (; h < h_even; h += 2) {
        // deal with left pad
        float maxr0 = std::max(r0[0], r0[1]);
        float maxr1 = std::max(r1[0], r1[1]);
        float maxr2 = std::max(r2[0], r2[1]);
        data_out_channel[0] = std::max(std::max(maxr0, maxr1), maxr2);
#ifdef __aarch64__
        w = 1;
        cnt = 1;
        for (; w < win - 8; w += 8) {
          float32x4_t vr0_1234 = vld1q_f32(&r0[w]);
          float32x4_t vr0_5678 = vld1q_f32(&r0[w + 4]);
          float32x4_t vr0_9101112 = vld1q_f32(&r0[w + 8]);
          float32x4_t vr1_1234 = vld1q_f32(&r1[w]);
          float32x4_t vr1_5678 = vld1q_f32(&r1[w + 4]);
          float32x4_t vr1_9101112 = vld1q_f32(&r1[w + 8]);
          float32x4_t vr2_1234 = vld1q_f32(&r2[w]);
          float32x4_t vr2_5678 = vld1q_f32(&r2[w + 4]);
          float32x4_t vr2_9101112 = vld1q_f32(&r2[w + 8]);
          float32x4_t vmax_1234 = vmaxq_f32(vr0_1234, vr1_1234);
          vmax_1234 = vmaxq_f32(vmax_1234, vr2_1234);
          float32x4_t vmax_5678 = vmaxq_f32(vr0_5678, vr1_5678);
          vmax_5678 = vmaxq_f32(vmax_5678, vr2_5678);
          float32x4_t vmax_9101112 = vmaxq_f32(vr0_9101112, vr1_9101112);
          vmax_9101112 = vmaxq_f32(vmax_9101112, vr2_9101112);
          float32x4_t vmax_2345 = vextq_f32(vmax_1234, vmax_5678, 1);
          float32x4_t vmax_6789 = vextq_f32(vmax_5678, vmax_9101112, 1);
          float32x2_t vmax_12_34 =
              vpmax_f32(vget_low_f32(vmax_1234), vget_high_f32(vmax_1234));
          float32x2_t vmax_23_45 =
              vpmax_f32(vget_low_f32(vmax_2345), vget_high_f32(vmax_2345));
          float32x2_t vmax_56_78 =
              vpmax_f32(vget_low_f32(vmax_5678), vget_high_f32(vmax_5678));
          float32x2_t vmax_67_89 =
              vpmax_f32(vget_low_f32(vmax_6789), vget_high_f32(vmax_6789));
          float32x2_t vmax_123_345 = vmax_f32(vmax_12_34, vmax_23_45);
          float32x2_t vmax_567_789 = vmax_f32(vmax_56_78, vmax_67_89);
          vst1_f32(&data_out_channel[cnt], vmax_123_345);
          vst1_f32(&data_out_channel[cnt + 2], vmax_567_789);
          cnt += 4;
        }
        for (; w < w_even - 1; w += 2) {
          float32x4_t vr0 = vld1q_f32(&r0[w]);
          float32x4_t vr1 = vld1q_f32(&r1[w]);
          float32x4_t vr2 = vld1q_f32(&r2[w]);
          vr0 = vsetq_lane_f32(minval, vr0, 3);
          vr1 = vsetq_lane_f32(minval, vr1, 3);
          vr2 = vsetq_lane_f32(minval, vr2, 3);
          float32x4_t vmax1 = vmaxq_f32(vr0, vr1);
          vmax1 = vmaxq_f32(vmax1, vr2);
          float32x2_t vmax2 =
              vpmax_f32(vget_low_f32(vmax1), vget_high_f32(vmax1));
          float32x2_t vmax = vpmax_f32(vmax2, vmax2);
          data_out_channel[cnt] = vget_lane_f32(vmax, 0);
          cnt++;
        }
#else
        dr_out = data_out_channel + 1;
        dr0 = (r0 + 1);
        dr1 = (r1 + 1);
        dr2 = (r2 + 1);
        cnt_num = cnt_col;
        cnt_num1 = remain;
        if (cnt_num > 0 || cnt_num1 > 0) {
          asm volatile(
              "cmp       %[cnt_num], #0                        @cmp cnt_num, "
              "0\n"
              "ble       3f                                  @ble exit\n"
              "1:                                @main loop\n"
              "vld1.f32  {d0-d3}, [%[dr0]]!                     @load d0-d5, "
              "dr0\n"
              "vld1.f32  {d6-d9}, [%[dr1]]!                    @load d4-d7, "
              "dr1\n"
              "vld1.f32  {d12-d15}, [%[dr2]]!                   @load d4-d7, "
              "dr1\n"
              "vld1.f32  {d4-d5}, [%[dr0]]!                     @load d0-d5, "
              "dr0\n"
              "vld1.f32  {d10-d11}, [%[dr1]]!                    @load d4-d7, "
              "dr1\n"
              "vld1.f32  {d16-d17}, [%[dr2]]!                   @load d4-d7, "
              "dr1\n"
              "vmax.f32  q9, q0, q3                            @max q0,q0,q2\n"
              "vmax.f32  q10, q1, q4                           @max q1,q1,q3\n"
              "vmax.f32  q11, q2, q5                           @max q1,q1,q3\n"
              "vmax.f32  q0, q9, q6                            @max q0,q0,q2 "
              "1234\n"
              "vmax.f32  q3, q10, q7                           @max q1,q1,q3 "
              "5678\n"
              "vmax.f32  q1, q11, q8                           @max q1,q1,q3 "
              "9101112\n"
              //"vmov.f32  s7,s6                               @mov s7, s6\n"
              "vext.f32  q4, q0, q3, #1                        @vext 2345\n"
              "vext.f32  q2, q3, q1, #1                        @vext 6789\n"
              "vpmax.f32 d10, d0, d1                           @pmax d10, "
              "vmax_1234, vmax_1234\n"
              "vpmax.f32 d12, d6, d7                           @pmax d12, "
              "vmax_5678, vmax_5678\n"
              "vpmax.f32 d11, d8, d9                           @pmax d11, "
              "vmax_2345, vmax_2345\n"
              "vpmax.f32 d13, d4, d5                           @pmax d13, "
              "vmax_6789, vmax_6789\n"
              "vmax.f32 d0, d10, d11                          @pmax d0, "
              "vmax_12_34, vmax_23_45\n"
              "vmax.f32 d1, d12, d13                          @pmax d1, "
              "vmax_56_78, vmax_67_89\n"
              "sub       %[dr0], #16                           @add w, 8\n"
              "sub       %[dr1], #16                           @add w, 8\n"
              "sub       %[dr2], #16                           @add w, 8\n"
              "vst1.f32  d0, [%[dr_out]]!                      @vst1 d0, "
              "dr_out\n"
              "vst1.f32  d1, [%[dr_out]]!                      @vst1 d0, "
              "dr_out\n"
              "subs      %[cnt_num], #1                            @subs "
              "cnt_num, #1\n"
              "bne       1b                                   @bne "
              "s3_max_loop_mid\n"
              "3:                                       @loop \n"
              "cmp       %[cnt_num1], #0                           @cmp "
              "cnt_num, 0\n"
              "ble       4f                                 @ble exit1\n"
              "2:                              @mid loop\n"
              "vld1.f32  {d0-d1}, [%[dr0]]!                     @load d0-d1, "
              "dr0\n"
              "vld1.f32  {d2-d3}, [%[dr1]]!                    @load d2-d3, "
              "dr1\n"
              "vld1.f32  {d4-d5}, [%[dr2]]!                     @load d2-d3, "
              "dr1\n"
              "vmov.f32  s3,s2                                 @movs3, s2\n"
              "vmov.f32  s7,s6                                 @movs7, s6\n"
              "vmov.f32  s11,s10                               @movs11, s10\n"
              "vmax.f32  q0, q0, q1                            @max q0, q0, "
              "q1\n"
              "vmax.f32  q0, q0, q2                            @max q0, q0, "
              "q2\n"
              "vpmax.f32 d0, d0, d1                            @pmax d0, "
              "d0,d1\n"
              "vpmax.f32 d0, d0, d0                            @pmax d0, d0, "
              "d0\n"
              "vst1.f32  d0[0], [%[dr_out]]!                   @vst  d0[0], "
              "dr_out\n"
              "sub       %[dr0], #8                            @add w, 6\n"
              "sub       %[dr1], #8                            @add w, 6\n"
              "sub       %[dr2], #8                            @add w, 6\n"
              "subs      %[cnt_num1], #1                       @subs cnt_num, "
              "#1\n"
              "bne       2b                                    @bne "
              "s3_max_loop_mid_1\n"
              "4:                                           @exit\n"
              : [dr0] "+r"(dr0), [dr1] "+r"(dr1), [dr2] "+r"(dr2),
                [dr_out] "+r"(dr_out), [cnt_num] "+r"(cnt_num),
                [cnt_num1] "+r"(cnt_num1)
              : "r"(dr0), "r"(dr1), "r"(dr2), "r"(dr_out), "r"(cnt_num),
                "r"(cnt_num1)
              : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7",
                "q8", "q9", "q10", "q11", "q12");
        }
#endif
        if (pad_right) {
          // deal with right pad
          int wstart = (w_even >> 1) * stride_w - pad_w;
          int wend = std::min(std::min(wstart + kernel_w, win + pad_w), win);
          float tmp = r0[wstart];  // std::numeric_limits<float>::min();
          for (int i = wstart; i < wend; i++) {
            tmp = std::max(tmp, std::max(r0[i], r1[i]));
            tmp = std::max(tmp, r2[i]);
          }
          data_out_channel[w_even >> 1] = tmp;
          // cnt ++;
        }
        r0 = r2;
        r1 = r0 + win;
        r2 = r1 + win;
        data_out_channel += wout;
      }

      if (pad_bottom) {
        // deal with bottom pad
        // first row with zero pad
        int hstart = (h >> 1) * stride_h - pad_h;
        int hend = std::min(std::min(hstart + kernel_h, hin + pad_h), hin);

        if (hstart == hend - 1) {  // only one lline
          data_out_channel[0] = std::max(r0[0], r0[1]);
#ifdef __aarch64__
          w = 1;
          cnt = 1;
          for (; w < win - 8; w += 8) {
            float32x4_t vmax_1234 = vld1q_f32(&r0[w]);
            float32x4_t vmax_5678 = vld1q_f32(&r0[w + 4]);
            float32x4_t vmax_9101112 = vld1q_f32(&r0[w + 8]);
            float32x4_t vmax_2345 = vextq_f32(vmax_1234, vmax_5678, 1);
            float32x4_t vmax_6789 = vextq_f32(vmax_5678, vmax_9101112, 1);
            float32x2_t vmax_12_34 =
                vpmax_f32(vget_low_f32(vmax_1234), vget_high_f32(vmax_1234));
            float32x2_t vmax_23_45 =
                vpmax_f32(vget_low_f32(vmax_2345), vget_high_f32(vmax_2345));
            float32x2_t vmax_56_78 =
                vpmax_f32(vget_low_f32(vmax_5678), vget_high_f32(vmax_5678));
            float32x2_t vmax_67_89 =
                vpmax_f32(vget_low_f32(vmax_6789), vget_high_f32(vmax_6789));
            float32x2_t vmax_123_345 = vmax_f32(vmax_12_34, vmax_23_45);
            float32x2_t vmax_567_789 = vmax_f32(vmax_56_78, vmax_67_89);
            vst1_f32(&data_out_channel[cnt], vmax_123_345);
            vst1_f32(&data_out_channel[cnt + 2], vmax_567_789);
            cnt += 4;
          }
          for (; w < w_even - 1; w += 2) {
            float32x4_t vr0 = vld1q_f32(&r0[w]);
            vr0 = vsetq_lane_f32(minval, vr0, 3);
            float32x2_t vmax = vpmax_f32(vget_low_f32(vr0), vget_high_f32(vr0));
            vmax = vpmax_f32(vmax, vmax);
            data_out_channel[cnt] = vget_lane_f32(vmax, 0);
            cnt++;
          }
#else
          dr_out = data_out_channel + 1;
          dr0 = (r0 + 1);
          cnt_num = cnt_col;
          cnt_num1 = remain;
          if (cnt_num > 0 || cnt_num1 > 0) {
            asm volatile(
                "cmp       %[cnt_num], #0                        @cmp cnt_num, "
                "0\n"
                "ble       3f                                  @ble exit\n"
                "1:                                @main loop\n"
                "vld1.f32  {d0-d3}, [%[dr0]]!                     @load d0-d3, "
                "dr0\n"
                "vld1.f32  {d4-d5}, [%[dr0]]!                     @load d0-d3, "
                "dr0\n"
                "vext.f32  q4, q0, q1, #1                        @vext q4, q0, "
                "q1, 1 2345\n"
                "vext.f32  q5, q1, q2, #1                        @vext q5, q0, "
                "q1, 1 6789\n"
                "vpmax.f32 d12, d0, d1                           @pmax d12, "
                "vmax_1234, vmax_1234\n"
                "vpmax.f32 d14, d2, d3                           @pmax d14, "
                "vmax_5678, vmax_5678\n"
                "vpmax.f32 d13, d8, d9                           @pmax d13, "
                "vmax_2345, vmax_2345\n"
                "vpmax.f32 d15, d10, d11                         @pmax d15, "
                "vmax_6789, vmax_6789\n"
                "vmax.f32  d0, d12, d13                          @max d0, "
                "vmax_12_34,vmax_23_45\n"
                "vmax.f32  d1, d14, d15                          @pmax d2, "
                "vmax_56_78, vmax_67_89\n"
                "sub       %[dr0], #16                           @add w, 6\n"
                "vst1.f32  d0, [%[dr_out]]!                      @vst1 d0, "
                "dr_out\n"
                "vst1.f32  d1, [%[dr_out]]!                      @vst1 d0, "
                "dr_out\n"
                "subs      %[cnt_num], #1                            @subs "
                "cnt_num, #1\n"
                "bne       1b                                   @bne "
                "s3_max_loop_bot\n"
                "3:                                             @loop \n"
                "cmp       %[cnt_num1], #0                           @cmp "
                "cnt_num, 0\n"
                "ble       4f                                 @ble exit\n"
                "2:                                             @bot loop\n"
                "vld1.f32  {d0-d1}, [%[dr0]]!                     @load d0-d1, "
                "dr0\n"
                "vmov.f32  s3,s2                                 @movs3, s2\n"
                "vpmax.f32 d0, d0, d1                            @pmax d0, "
                "d0,d1\n"
                "vpmax.f32 d0, d0, d0                            @pmax d0, d0, "
                "d0\n"
                "vst1.f32  d0[0], [%[dr_out]]!                   @vst  d0[0], "
                "dr_out\n"
                "sub       %[dr0], #8                            @add w, 2\n"
                "subs      %[cnt_num1], #1                           @subs "
                "cnt_num, #1\n"
                "bne       2b                                   @bne "
                "s3_max_loop_bot_1\n"
                "4:                                          @exit\n"
                : [dr0] "+r"(dr0), [dr1] "+r"(dr1), [dr_out] "+r"(dr_out),
                  [cnt_num] "+r"(cnt_num), [cnt_num1] "+r"(cnt_num1)
                : "r"(dr0), "r"(dr1), "r"(dr_out), "r"(cnt_num), "r"(cnt_num1)
                : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6",
                  "q7", "q8");
          }
#endif
          if (pad_right) {
            // deal with right pad
            int wstart = (w_even >> 1) * stride_w - pad_w;
            int wend = std::min(std::min(wstart + kernel_w, win + pad_w), win);
            float tmp = r0[wstart];  // std::numeric_limits<float>::min();
            for (int i = wstart; i < wend; i++) {
              tmp = std::max(tmp, r0[i]);
            }
            data_out_channel[w_even >> 1] = tmp;
          }
        } else {  // two lines
          data_out_channel[0] =
              std::max(std::max(r0[0], r0[1]), std::max(r1[0], r1[1]));
#ifdef __aarch64__
          w = 1;
          cnt = 1;
          for (; w < win - 8; w += 8) {
            float32x4_t vr0_1234 = vld1q_f32(&r0[w]);
            float32x4_t vr0_5678 = vld1q_f32(&r0[w + 4]);
            float32x4_t vr0_9101112 = vld1q_f32(&r0[w + 8]);
            float32x4_t vr1_1234 = vld1q_f32(&r1[w]);
            float32x4_t vr1_5678 = vld1q_f32(&r1[w + 4]);
            float32x4_t vr1_9101112 = vld1q_f32(&r1[w + 8]);
            float32x4_t vmax_1234 = vmaxq_f32(vr0_1234, vr1_1234);
            float32x4_t vmax_5678 = vmaxq_f32(vr0_5678, vr1_5678);
            float32x4_t vmax_9101112 = vmaxq_f32(vr0_9101112, vr1_9101112);
            float32x4_t vmax_2345 = vextq_f32(vmax_1234, vmax_5678, 1);
            float32x4_t vmax_6789 = vextq_f32(vmax_5678, vmax_9101112, 1);
            float32x2_t vmax_12_34 =
                vpmax_f32(vget_low_f32(vmax_1234), vget_high_f32(vmax_1234));
            float32x2_t vmax_23_45 =
                vpmax_f32(vget_low_f32(vmax_2345), vget_high_f32(vmax_2345));
            float32x2_t vmax_56_78 =
                vpmax_f32(vget_low_f32(vmax_5678), vget_high_f32(vmax_5678));
            float32x2_t vmax_67_89 =
                vpmax_f32(vget_low_f32(vmax_6789), vget_high_f32(vmax_6789));
            float32x2_t vmax_123_345 = vmax_f32(vmax_12_34, vmax_23_45);
            float32x2_t vmax_567_789 = vmax_f32(vmax_56_78, vmax_67_89);
            vst1_f32(&data_out_channel[cnt], vmax_123_345);
            vst1_f32(&data_out_channel[cnt + 2], vmax_567_789);
            cnt += 4;
          }
          for (; w < w_even - 1; w += 2) {
            float32x4_t vr0 = vld1q_f32(&r0[w]);
            float32x4_t vr1 = vld1q_f32(&r1[w]);
            vr0 = vsetq_lane_f32(minval, vr0, 3);
            vr1 = vsetq_lane_f32(minval, vr1, 3);
            float32x4_t vmax1 = vmaxq_f32(vr0, vr1);
            float32x2_t vmax2 =
                vpmax_f32(vget_low_f32(vmax1), vget_high_f32(vmax1));
            vmax2 = vpmax_f32(vmax2, vmax2);
            data_out_channel[cnt] = vget_lane_f32(vmax2, 0);
            cnt++;
          }
#else
          dr_out = data_out_channel + 1;
          dr0 = (r0 + 1);
          dr1 = (r1 + 1);
          cnt_num = cnt_col;
          cnt_num1 = remain;
          if (cnt_num > 0 || cnt_num1 > 0) {
            asm volatile(
                "cmp       %[cnt_num], #0                        @cmp cnt_num, "
                "0\n"
                "ble       3f                                  @ble exit\n"
                "1:                               @main loop\n"
                "vld1.f32  {d0-d3}, [%[dr0]]!                     @load d0-d5, "
                "dr0\n"
                "vld1.f32  {d6-d9}, [%[dr1]]!                    @load d4-d7, "
                "dr1\n"
                "vld1.f32  {d4-d5}, [%[dr0]]!                     @load d0-d3, "
                "dr0\n"
                "vld1.f32  {d10-d11}, [%[dr1]]!                  @load d4-d7, "
                "dr1\n"
                "vmax.f32  q6, q0, q3                            @max q0,q0,q2 "
                "1234\n"
                "vmax.f32  q7, q1, q4                            @max q1,q1,q3 "
                "5678\n"
                "vmax.f32  q8, q2, q5                            @max q1,q1,q3 "
                "9101112\n"
                //"vmov.f32  s7,s6                                 @mov s7,
                // s6\n"
                "vext.f32  q0, q6, q7, #1                        @vext q0, "
                "2345\n"
                "vext.f32  q1, q7, q8, #1                        @vext q1, "
                "6789\n"
                "vpmax.f32 d4, d12, d13                          @pmax d4, "
                "vmax_1234, vmax_1234\n"
                "vpmax.f32 d6, d14, d15                          @pmax d6, "
                "vmax_5678, vmax_5678\n"
                "vpmax.f32 d5, d0, d1                            @pmax d5, "
                "vmax_2345, vmax_2345\n"
                "vpmax.f32 d7, d2, d3                            @pmax d7, "
                "vmax_6789, vmax_6789\n"
                "vmax.f32 d8, d4, d5                             @max d2, "
                "vmax_12_34, vmax_23_45\n"
                "vmax.f32 d9, d6, d7                             @max d2, "
                "vmax_56_78, vmax_67_89\n"
                "sub       %[dr0], #16                           @add w, 8\n"
                "sub       %[dr1], #16                           @add w, 8\n"
                "vst1.f32  d8, [%[dr_out]]!                      @vst1 d0, "
                "dr_out\n"
                "vst1.f32  d9, [%[dr_out]]!                      @vst1 d0, "
                "dr_out\n"
                "subs      %[cnt_num], #1                            @subs "
                "cnt_num, #1\n"
                "bne       1b                                   @bne "
                "s3_max_loop_bot\n"
                "3:                                      @loop \n"
                "cmp       %[cnt_num1], #0                           @cmp "
                "cnt_num, 0\n"
                "ble       4f                                 @ble exit\n"
                "2:                                             @bot loop\n"
                "vld1.f32  {d0-d1}, [%[dr0]]!                     @load d0-d1, "
                "dr0\n"
                "vld1.f32  {d2-d3}, [%[dr1]]!                     @load d2-d3, "
                "dr1\n"
                "vmov.f32  s3,s2                                 @movs3, s2\n"
                "vmov.f32  s7,s6                                 @movs7, s6\n"
                "vmax.f32  q0, q0, q1                            @max q0, q0, "
                "q1\n"
                "vpmax.f32 d0, d0, d1                            @pmax d0, "
                "d0,d1\n"
                "vpmax.f32 d0, d0, d0                            @pmax d0, d0, "
                "d0\n"
                "vst1.f32  d0[0], [%[dr_out]]!                   @vst  d0[0], "
                "dr_out\n"
                "sub       %[dr0], #8                            @add w, 6\n"
                "sub       %[dr1], #8                            @add w, 6\n"
                "subs      %[cnt_num1], #1                           @subs "
                "cnt_num, #1\n"
                "bne       2b                                   @bne "
                "s3_max_loop_bot_1\n"
                "4:                                          @exit\n"
                : [dr0] "+r"(dr0), [dr1] "+r"(dr1), [dr_out] "+r"(dr_out),
                  [cnt_num] "+r"(cnt_num), [cnt_num1] "+r"(cnt_num1)
                : "r"(dr0), "r"(dr1), "r"(dr_out), "r"(cnt_num), "r"(cnt_num1)
                : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6",
                  "q7", "q8", "q9");
          }
#endif
          if (pad_right) {
            // deal with right pad
            int wstart = (w_even >> 1) * stride_w - pad_w;
            int wend = std::min(std::min(wstart + kernel_w, win + pad_w), win);
            float tmp = r0[wstart];  // std::numeric_limits<float>::min();
            for (int i = wstart; i < wend; i++) {  // only run 1 or 2 times
              tmp = std::max(tmp, std::max(r0[i], r1[i]));
            }
            data_out_channel[w_even >> 1] = tmp;
          }
        }
      }
    }
  }
}

void pooling3x3s2p1_ave(const void* din, void* dout, int num, int chout,
                        int hout, int wout, int chin, int hin, int win,
                        const std::vector<int>& ksize,
                        const std::vector<int>& strides,
                        const std::vector<int>& paddings, bool global_pooling,
                        bool exclusive, bool adaptive, bool ceil_mode,
                        bool use_quantizer, const std::string& pooling_type) {
  int size_channel_out = wout * hout;
  int size_channel_in = win * hin;
  float* data_out = static_cast<float*>(dout);
  const float* data_in = static_cast<const float*>(din);

  int kernel_h = ksize[0];
  int kernel_w = ksize[1];
  int stride_h = strides[0];
  int stride_w = strides[1];
  int pad_h = paddings[0];
  int pad_w = paddings[1];

  int pad_top = pad_h;
  int pad_left = pad_w;
  int w_needed = wout * 2 + 1;
  int h_needed = hout * 2 + 1;
  int pad_right = w_needed - win - pad_left;
  int pad_bottom = h_needed - hin - pad_top;
  int w_even = (win >> 1) << 1;
  int h_even = (hin >> 1) << 1;
  int w_in_2 = win << 1;
  int w_unroll_size = (win - 1) / 8;
  // remain
  int w_unroll_remian = ((win - 1) % 8) / 2;

  for (int n = 0; n < num; ++n) {
    float* data_out_batch = data_out + n * chout * size_channel_out;
    const float* data_in_batch = data_in + n * chin * size_channel_in;
#pragma omp parallel for
    for (int c = 0; c < chout; c++) {
      float* data_out_channel = data_out_batch + c * size_channel_out;
      const float* data_in_channel = data_in_batch + c * size_channel_in;
      const float* r0 = data_in_channel;
      const float* r1 = r0 + win;
      const float* r2 = r1 + win;
      int cnt_num = w_unroll_size;
      int cnt_num1 = w_unroll_remian;
      float* dr_out = data_out_channel;
      const float* dr0 = r0;
      const float* dr1 = r1;
      const float* dr2 = r2;
      int w = 1;
      int cnt = 1;
      float32x4_t vcoef = vdupq_n_f32(1.f / 9.f);
      float32x4_t vzero = vdupq_n_f32(0.f);
      data_out_channel[0] = (r0[0] + r0[1] + r1[0] + r1[1]) / 9.f;
// first row with zero pad
#ifdef __aarch64__
      for (; w < win - 8; w += 8) {
        float32x4_t vr0_1234 = vld1q_f32(&r0[w]);
        float32x4_t vr0_5678 = vld1q_f32(&r0[w + 4]);
        float32x4_t vr0_9101112 = vld1q_f32(&r0[w + 8]);
        float32x4_t vr1_1234 = vld1q_f32(&r1[w]);
        float32x4_t vr1_5678 = vld1q_f32(&r1[w + 4]);
        float32x4_t vr1_9101112 = vld1q_f32(&r1[w + 8]);
        float32x4_t vsum_1234 = vaddq_f32(vr0_1234, vr1_1234);
        float32x4_t vsum_5678 = vaddq_f32(vr0_5678, vr1_5678);
        float32x4_t vsum_9101112 = vaddq_f32(vr0_9101112, vr1_9101112);

        float32x4_t vsum_2345 = vextq_f32(vsum_1234, vsum_5678, 1);
        float32x4_t vsum_3456 = vextq_f32(vsum_1234, vsum_5678, 2);
        float32x4_t vsum_4567 = vextq_f32(vsum_1234, vsum_5678, 3);
        float32x4_t vsum_6789 = vextq_f32(vsum_5678, vsum_9101112, 1);
        float32x4_t vsum_123_345 = vaddq_f32(vsum_1234, vsum_2345);
        vsum_123_345 = vaddq_f32(vsum_123_345, vsum_3456);
        float32x4_t vsum_567_789 = vaddq_f32(vsum_4567, vsum_5678);
        vsum_567_789 = vaddq_f32(vsum_567_789, vsum_6789);
        vsum_123_345 =
            vsetq_lane_f32(vgetq_lane_f32(vsum_123_345, 2), vsum_123_345, 1);
        vsum_123_345 =
            vsetq_lane_f32(vgetq_lane_f32(vsum_567_789, 1), vsum_123_345, 2);
        vsum_123_345 =
            vsetq_lane_f32(vgetq_lane_f32(vsum_567_789, 3), vsum_123_345, 3);
        float32x4_t vrst = vmulq_f32(vsum_123_345, vcoef);
        vst1q_f32(&data_out_channel[cnt], vrst);
        cnt += 4;
      }
      for (; w < w_even - 1; w += 2) {
        float32x4_t vr0 = vld1q_f32(&r0[w]);
        float32x4_t vr1 = vld1q_f32(&r1[w]);
        vr0 = vsetq_lane_f32(0.f, vr0, 3);
        vr1 = vsetq_lane_f32(0.f, vr1, 3);
        float32x4_t vsum1 = vaddq_f32(vr0, vr1);
        float32x2_t vsum2 =
            vpadd_f32(vget_low_f32(vsum1), vget_high_f32(vsum1));
        vsum2 = vpadd_f32(vsum2, vsum2);
        float32x2_t vrst = vmul_f32(vsum2, vget_low_f32(vcoef));
        data_out_channel[cnt] = vget_lane_f32(vrst, 0);
        cnt++;
      }
#else
      dr0 = dr0 + 1;
      dr1 = dr1 + 1;
      dr_out = dr_out + 1;
      // printf("cnt_num: %d, cnt_num1: %d \n",cnt_num, cnt_num1);
      if (cnt_num > 0 || cnt_num1 > 0) {
        asm volatile(
            "cmp       %[cnt_num], #0                        @cmp cnt_num, 0\n"
            "ble       3f                                  @ble exit\n"
            "1:                                    @main loop\n"
            "vld1.f32  {d0-d3}, [%[dr0]]!                     @load d0-d5, "
            "dr0\n"
            "vld1.f32  {d6-d9}, [%[dr1]]!                    @load d4-d7, dr1\n"
            "vld1.f32  {d4-d5}, [%[dr0]]!                     @load d0-d5, "
            "dr0\n"
            "vld1.f32  {d10-d11}, [%[dr1]]!                    @load d4-d7, "
            "dr1\n"
            "vadd.f32  q6, q0, q3                            @max "
            "r0_1234,r1_1234\n"
            "vadd.f32  q7, q1, q4                            @max "
            "r0_5678,r1_5678\n"
            "vadd.f32  q8, q2, q5                            @max "
            "r0_9101112,r1_9101112\n"
            //"vmov.f32  s7,s6                                 @mov s7, s6\n"
            "vext.f32  q0, q6, q7, #1                        @vext max_2345\n"
            "vext.f32  q1, q6, q7, #3                        @vext max_4567\n"
            "vext.f32  q2, q6, q7, #2                        @vext max_3456\n"
            "vext.f32  q3, q7, q8, #1                        @vext max_6789\n"
            "vadd.f32  q4, q6, q0                            @add 1234, 2345 \n"
            "vadd.f32  q5, q7, q1                            @add 5678, 4567 \n"
            "vadd.f32  q4, q4, q2                            @add 3456, sum1 \n"
            "vadd.f32  q5, q5, q3                            @add 6789, sum2 \n"
            "vmov.f32  s17, s18                              @mov \n"
            "vmov.f32  s18, s21                              @mov \n"
            "vmov.f32  s19, s23                              @mov \n"
            "vmul.f32  q4, q4, %q[vcoef]                     @mul \n"
            "sub       %[dr0], #16                           @add w, 8\n"
            "sub       %[dr1], #16                           @add w, 8\n"
            "subs      %[cnt_num], #1                        @subs cnt_num, "
            "#1\n"
            "vst1.f32  d8, [%[dr_out]]!                      @vst1 d0, dr_out\n"
            "vst1.f32  d9, [%[dr_out]]!                      @vst1 d0, dr_out\n"
            "bne       1b                           @bne s3_max_loop\n"
            "3:                                           @loop \n"
            "cmp       %[cnt_num1], #0                           @cmp cnt_num, "
            "0\n"
            "ble       4f                                  @ble exit\n"
            "2:                                  @main loop\n"
            "vld1.f32  {d0-d1}, [%[dr0]]!                     @load d0-d1, "
            "dr0\n"
            "vld1.f32  {d2-d3}, [%[dr1]]!                     @load d2-d3, "
            "dr1\n"
            "vext.f32  q0, %q[vzero], q0, #3                 @ ext v0_0123\n"
            "vext.f32  q1, %q[vzero], q1, #3                 @ ext v1_0123\n"
            "vadd.f32  q0, q0, q1                            @add q0, q0, q1\n"
            "vpadd.f32 d0, d0, d1                            @padd d0, d0,d1\n"
            "vpadd.f32 d0, d0, d0                            @padd d0, d0, d0\n"
            "vmul.f32  d0, d0, %e[vcoef]                     @mul \n"
            "sub       %[dr0], #8                            @add w, 6\n"
            "sub       %[dr1], #8                            @add w, 6\n"
            "subs      %[cnt_num1], #1                       @subs cnt_num, "
            "#1\n"
            "vst1.f32  d0[0], [%[dr_out]]!                   @vst  d0[0], "
            "dr_out\n"
            "bne       2b                         @bne s3_max_loop_1\n"
            "4:                                           @exit\n"
            : [dr0] "+r"(dr0), [dr1] "+r"(dr1), [dr_out] "+r"(dr_out),
              [cnt_num] "+r"(cnt_num), [cnt_num1] "+r"(cnt_num1),
              [vcoef] "+w"(vcoef), [vzero] "+w"(vzero)
            : "r"(dr0), "r"(dr1), "r"(dr_out), "r"(cnt_num), "r"(cnt_num1)
            : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7",
              "q8", "q9");
      }
// printf("cnt_num: %d, cnt_num1: %d \n",cnt_num, cnt_num1);
#endif
      // int w = w_even - 1;
      if (pad_right) {
        // deal with right pad
        int wstart = (w_even >> 1) * stride_w - pad_w;
        int wend = std::min(std::min(wstart + kernel_w, win + pad_w), win);
        float tmp = 0.f;  // std::numeric_limits<float>::min();
        for (int i = wstart; i < wend; i++) {  // only run 1 or 2 times
          tmp += (r0[i] + r1[i]);
        }
        data_out_channel[w_even >> 1] = tmp / 9.f;
        // cnt ++;
      }

      r0 = r1;
      r1 = r0 + win;
      r2 = r1 + win;
      data_out_channel += wout;
      int h = 2;
      for (; h < h_even; h += 2) {
        // deal with left pad
        float sum0 = r0[0] + r0[1];
        float sum1 = r1[0] + r1[1];
        float sum2 = r2[0] + r2[1];
        data_out_channel[0] = (sum0 + sum1 + sum2) / 9.f;
#ifdef __aarch64__
        w = 1;
        cnt = 1;
        for (; w < win - 8; w += 8) {
          float32x4_t vr0_1234 = vld1q_f32(&r0[w]);
          float32x4_t vr0_5678 = vld1q_f32(&r0[w + 4]);
          float32x4_t vr0_9101112 = vld1q_f32(&r0[w + 8]);
          float32x4_t vr1_1234 = vld1q_f32(&r1[w]);
          float32x4_t vr1_5678 = vld1q_f32(&r1[w + 4]);
          float32x4_t vr1_9101112 = vld1q_f32(&r1[w + 8]);
          float32x4_t vr2_1234 = vld1q_f32(&r2[w]);
          float32x4_t vr2_5678 = vld1q_f32(&r2[w + 4]);
          float32x4_t vr2_9101112 = vld1q_f32(&r2[w + 8]);
          float32x4_t vsum_1234 = vaddq_f32(vr0_1234, vr1_1234);
          float32x4_t vsum_5678 = vaddq_f32(vr0_5678, vr1_5678);
          float32x4_t vsum_9101112 = vaddq_f32(vr0_9101112, vr1_9101112);
          vsum_1234 = vaddq_f32(vsum_1234, vr2_1234);
          vsum_5678 = vaddq_f32(vsum_5678, vr2_5678);
          vsum_9101112 = vaddq_f32(vsum_9101112, vr2_9101112);

          float32x4_t vsum_2345 = vextq_f32(vsum_1234, vsum_5678, 1);
          float32x4_t vsum_3456 = vextq_f32(vsum_1234, vsum_5678, 2);
          float32x4_t vsum_4567 = vextq_f32(vsum_1234, vsum_5678, 3);
          float32x4_t vsum_6789 = vextq_f32(vsum_5678, vsum_9101112, 1);
          float32x4_t vsum_123_345 = vaddq_f32(vsum_1234, vsum_2345);
          vsum_123_345 = vaddq_f32(vsum_123_345, vsum_3456);
          float32x4_t vsum_567_789 = vaddq_f32(vsum_4567, vsum_5678);
          vsum_567_789 = vaddq_f32(vsum_567_789, vsum_6789);
          vsum_123_345 =
              vsetq_lane_f32(vgetq_lane_f32(vsum_123_345, 2), vsum_123_345, 1);
          vsum_123_345 =
              vsetq_lane_f32(vgetq_lane_f32(vsum_567_789, 1), vsum_123_345, 2);
          vsum_123_345 =
              vsetq_lane_f32(vgetq_lane_f32(vsum_567_789, 3), vsum_123_345, 3);
          float32x4_t vrst = vmulq_f32(vsum_123_345, vcoef);
          vst1q_f32(&data_out_channel[cnt], vrst);
          cnt += 4;
        }
        for (; w < w_even - 1; w += 2) {
          float32x4_t vr0 = vld1q_f32(&r0[w]);
          float32x4_t vr1 = vld1q_f32(&r1[w]);
          float32x4_t vr2 = vld1q_f32(&r2[w]);
          vr0 = vsetq_lane_f32(0.f, vr0, 3);
          vr1 = vsetq_lane_f32(0.f, vr1, 3);
          vr2 = vsetq_lane_f32(0.f, vr2, 3);
          float32x4_t vsum1 = vaddq_f32(vr0, vr1);
          vsum1 = vaddq_f32(vsum1, vr2);
          float32x2_t vsum2 =
              vpadd_f32(vget_low_f32(vsum1), vget_high_f32(vsum1));
          float32x2_t vsum = vpadd_f32(vsum2, vsum2);
          data_out_channel[cnt] = vget_lane_f32(vsum, 0) / 9.f;
          cnt++;
        }
#else
        dr_out = data_out_channel + 1;
        dr0 = (r0 + 1);
        dr1 = (r1 + 1);
        dr2 = (r2 + 1);
        cnt_num = w_unroll_size;
        cnt_num1 = w_unroll_remian;
        if (cnt_num > 0 || cnt_num1 > 0) {
          asm volatile(
              "cmp       %[cnt_num], #0                        @cmp cnt_num, "
              "0\n"
              "ble       3f                                  @ble exit\n"
              "1:                                @main loop\n"
              "vld1.f32  {d0-d3}, [%[dr0]]!                     @load d0-d5, "
              "dr0\n"
              "vld1.f32  {d6-d9}, [%[dr1]]!                    @load d4-d7, "
              "dr1\n"
              "vld1.f32  {d12-d15}, [%[dr2]]!                   @load d4-d7, "
              "dr1\n"
              "vld1.f32  {d4-d5}, [%[dr0]]!                     @load d0-d5, "
              "dr0\n"
              "vld1.f32  {d10-d11}, [%[dr1]]!                    @load d4-d7, "
              "dr1\n"
              "vld1.f32  {d16-d17}, [%[dr2]]!                   @load d4-d7, "
              "dr1\n"
              "vadd.f32  q9, q0, q3                            @max q0,q0,q2\n"
              "vadd.f32  q10, q1, q4                           @max q1,q1,q3\n"
              "vadd.f32  q11, q2, q5                           @max q1,q1,q3\n"
              "vadd.f32  q6, q9, q6                            @max q0,q0,q2 "
              "1234\n"
              "vadd.f32  q7, q10, q7                           @max q1,q1,q3 "
              "5678\n"
              "vadd.f32  q8, q11, q8                           @max q1,q1,q3 "
              "9101112\n"
              //"vmov.f32  s7,s6                               @mov s7, s6\n"
              "vext.f32  q0, q6, q7, #1                        @vext max_2345\n"
              "vext.f32  q1, q6, q7, #3                        @vext max_4567\n"
              "vext.f32  q2, q6, q7, #2                        @vext max_3456\n"
              "vext.f32  q3, q7, q8, #1                        @vext max_6789\n"
              "vadd.f32  q4, q6, q0                            @add 1234, 2345 "
              "\n"
              "vadd.f32  q5, q7, q1                            @add 5678, 4567 "
              "\n"
              "vadd.f32  q4, q4, q2                            @add 3456, sum1 "
              "\n"
              "vadd.f32  q5, q5, q3                            @add 6789, sum2 "
              "\n"
              "vmov.f32  s17, s18                              @mov \n"
              "vmov.f32  s18, s21                              @mov \n"
              "vmov.f32  s19, s23                              @mov \n"
              "vmul.f32  q4, q4, %q[vcoef]                     @mul \n"
              "sub       %[dr0], #16                           @add w, 8\n"
              "sub       %[dr1], #16                           @add w, 8\n"
              "sub       %[dr2], #16                           @add w, 8\n"
              "subs      %[cnt_num], #1                            @subs "
              "cnt_num, #1\n"
              "vst1.f32  d8, [%[dr_out]]!                      @vst1 d0, "
              "dr_out\n"
              "vst1.f32  d9, [%[dr_out]]!                      @vst1 d0, "
              "dr_out\n"
              "bne       1b                       @bne s3_max_loop_mid\n"
              "3:                                       @loop \n"
              "cmp       %[cnt_num1], #0                           @cmp "
              "cnt_num, 0\n"
              "ble       4f                                 @ble exit1\n"
              "2:                              @mid loop\n"
              "vld1.f32  {d0-d1}, [%[dr0]]!                     @load d0-d1, "
              "dr0\n"
              "vld1.f32  {d2-d3}, [%[dr1]]!                    @load d2-d3, "
              "dr1\n"
              "vld1.f32  {d4-d5}, [%[dr2]]!                     @load d2-d3, "
              "dr1\n"
              "vext.f32  q0, %q[vzero], q0, #3                 @ ext v0_0123\n"
              "vext.f32  q1, %q[vzero], q1, #3                 @ ext v1_0123\n"
              "vext.f32  q2, %q[vzero], q2, #3                 @ ext v1_0123\n"
              "vadd.f32  q0, q0, q1                            @add q0, q0, "
              "q1\n"
              "vadd.f32  q0, q0, q2                            @add q0, q0, "
              "q1\n"
              "vpadd.f32 d0, d0, d1                            @padd d0, "
              "d0,d1\n"
              "vpadd.f32 d0, d0, d0                            @padd d0, d0, "
              "d0\n"
              "vmul.f32  d0, d0, %e[vcoef]                     @mul \n"
              "sub       %[dr0], #8                            @add w, 6\n"
              "sub       %[dr1], #8                            @add w, 6\n"
              "sub       %[dr2], #8                            @add w, 6\n"
              "subs      %[cnt_num1], #1                       @subs cnt_num, "
              "#1\n"
              "vst1.f32  d0[0], [%[dr_out]]!                   @vst  d0[0], "
              "dr_out\n"
              "bne       2b                     @bne s3_max_loop_mid_1\n"
              "4:                                           @exit\n"
              : [dr0] "+r"(dr0), [dr1] "+r"(dr1), [dr2] "+r"(dr2),
                [dr_out] "+r"(dr_out), [cnt_num] "+r"(cnt_num),
                [cnt_num1] "+r"(cnt_num1), [vcoef] "+w"(vcoef),
                [vzero] "+w"(vzero)
              : "r"(dr0), "r"(dr1), "r"(dr2), "r"(dr_out), "r"(cnt_num),
                "r"(cnt_num1)
              : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7",
                "q8", "q9", "q10", "q11", "q12");
        }
#endif
        if (pad_right) {
          // deal with right pad
          int wstart = (w_even >> 1) * stride_w - pad_w;
          int wend = std::min(std::min(wstart + kernel_w, win + pad_w), win);
          float tmp = 0.f;
          for (int i = wstart; i < wend; i++) {
            tmp += (r0[i] + r1[i] + r2[i]);
          }
          data_out_channel[w_even >> 1] = tmp / 9.f;
          // cnt ++;
        }
        r0 = r2;
        r1 = r0 + win;
        r2 = r1 + win;
        data_out_channel += wout;
      }

      if (pad_bottom) {
        // deal with bottom pad
        // first row with zero pad
        int hstart = (h >> 1) * stride_h - pad_h;
        int hend = std::min(std::min(hstart + kernel_h, hin + pad_h), hin);

        if (hstart == hend - 1) {  // only one lline
          data_out_channel[0] = (r0[0] + r0[1]) / 9.f;
#ifdef __aarch64__
          w = 1;
          cnt = 1;
          for (; w < win - 8; w += 8) {
            float32x4_t vsum_1234 = vld1q_f32(&r0[w]);
            float32x4_t vsum_5678 = vld1q_f32(&r0[w + 4]);
            float32x4_t vsum_9101112 = vld1q_f32(&r0[w + 8]);

            float32x4_t vsum_2345 = vextq_f32(vsum_1234, vsum_5678, 1);
            float32x4_t vsum_3456 = vextq_f32(vsum_1234, vsum_5678, 2);
            float32x4_t vsum_4567 = vextq_f32(vsum_1234, vsum_5678, 3);
            float32x4_t vsum_6789 = vextq_f32(vsum_5678, vsum_9101112, 1);
            float32x4_t vsum_123_345 = vaddq_f32(vsum_1234, vsum_2345);
            vsum_123_345 = vaddq_f32(vsum_123_345, vsum_3456);
            float32x4_t vsum_567_789 = vaddq_f32(vsum_4567, vsum_5678);
            vsum_567_789 = vaddq_f32(vsum_567_789, vsum_6789);
            vsum_123_345 = vsetq_lane_f32(vgetq_lane_f32(vsum_123_345, 2),
                                          vsum_123_345, 1);
            vsum_123_345 = vsetq_lane_f32(vgetq_lane_f32(vsum_567_789, 1),
                                          vsum_123_345, 2);
            vsum_123_345 = vsetq_lane_f32(vgetq_lane_f32(vsum_567_789, 3),
                                          vsum_123_345, 3);
            float32x4_t vrst = vmulq_f32(vsum_123_345, vcoef);
            vst1q_f32(&data_out_channel[cnt], vrst);
            cnt += 4;
          }
          for (; w < w_even - 1; w += 2) {
            float32x4_t vr0 = vld1q_f32(&r0[w]);
            vr0 = vsetq_lane_f32(0.f, vr0, 3);
            float32x2_t vsum = vpadd_f32(vget_low_f32(vr0), vget_high_f32(vr0));
            vsum = vpadd_f32(vsum, vsum);
            data_out_channel[cnt] = vget_lane_f32(vsum, 0) / 9.f;
            cnt++;
          }
#else
          dr_out = data_out_channel + 1;
          dr0 = (r0 + 1);
          cnt_num = w_unroll_size;
          cnt_num1 = w_unroll_remian;
          if (cnt_num > 0 || cnt_num1 > 0) {
            asm volatile(
                "cmp       %[cnt_num], #0                        @cmp cnt_num, "
                "0\n"
                "ble       3f                                  @ble exit\n"
                "1:                                @main loop\n"
                "vld1.f32  {d12-d15}, [%[dr0]]!                     @load "
                "d0-d3, dr0\n"
                "vld1.f32  {d16-d17}, [%[dr0]]!                     @load "
                "d0-d3, dr0\n"
                "vext.f32  q0, q6, q7, #1                        @vext "
                "max_2345\n"
                "vext.f32  q1, q6, q7, #3                        @vext "
                "max_4567\n"
                "vext.f32  q2, q6, q7, #2                        @vext "
                "max_3456\n"
                "vext.f32  q3, q7, q8, #1                        @vext "
                "max_6789\n"
                "vadd.f32  q4, q6, q0                            @add 1234, "
                "2345 \n"
                "vadd.f32  q5, q7, q1                            @add 5678, "
                "4567 \n"
                "vadd.f32  q4, q4, q2                            @add 3456, "
                "sum1 \n"
                "vadd.f32  q5, q5, q3                            @add 6789, "
                "sum2 \n"
                "vmov.f32  s17, s18                              @mov \n"
                "vmov.f32  s18, s21                              @mov \n"
                "vmov.f32  s19, s23                              @mov \n"
                "vmul.f32  q4, q4, %q[vcoef]                     @mul \n"
                "sub       %[dr0], #16                           @add w, 6\n"
                "subs      %[cnt_num], #1                            @subs "
                "cnt_num, #1\n"
                "vst1.f32  d8, [%[dr_out]]!                      @vst1 d0, "
                "dr_out\n"
                "vst1.f32  d9, [%[dr_out]]!                      @vst1 d0, "
                "dr_out\n"
                "bne       1b                       @bne s3_max_loop_bot\n"
                "3:                                       @loop \n"
                "cmp       %[cnt_num1], #0                           @cmp "
                "cnt_num, 0\n"
                "ble       4f                                 @ble exit\n"
                "2:                              @bot loop\n"
                "vld1.f32  {d0-d1}, [%[dr0]]!                     @load d0-d1, "
                "dr0\n"
                "vext.f32  q0, %q[vzero], q0, #3                 @ ext "
                "v0_0123\n"
                "vpadd.f32 d0, d0, d1                            @padd d0, "
                "d0,d1\n"
                "vpadd.f32 d0, d0, d0                            @padd d0, d0, "
                "d0\n"
                "vmul.f32  d0, d0, %e[vcoef]                     @mul \n"
                "sub       %[dr0], #8                            @add w, 2\n"
                "subs      %[cnt_num1], #1                       @subs "
                "cnt_num, #1\n"
                "vst1.f32  d0[0], [%[dr_out]]!                   @vst  d0[0], "
                "dr_out\n"
                "bne       2b                     @bne s3_max_loop_bot_1\n"
                "4:                                          @exit\n"
                : [dr0] "+r"(dr0), [dr1] "+r"(dr1), [dr_out] "+r"(dr_out),
                  [cnt_num] "+r"(cnt_num), [cnt_num1] "+r"(cnt_num1),
                  [vcoef] "+w"(vcoef), [vzero] "+w"(vzero)
                : "r"(dr0), "r"(dr1), "r"(dr_out), "r"(cnt_num), "r"(cnt_num1)
                : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6",
                  "q7", "q8");
          }
#endif
          if (pad_right) {
            // deal with right pad
            int wstart = (w_even >> 1) * stride_w - pad_w;
            int wend = std::min(std::min(wstart + kernel_w, win + pad_w), win);
            float tmp = 0.f;
            for (int i = wstart; i < wend; i++) {
              tmp += r0[i];
            }
            data_out_channel[w_even >> 1] = tmp / 9.f;
          }
        } else {  // two lines
          data_out_channel[0] = (r0[0] + r0[1] + r1[0] + r1[1]) / 9.f;
#ifdef __aarch64__
          w = 1;
          cnt = 1;
          for (; w < win - 8; w += 8) {
            float32x4_t vr0_1234 = vld1q_f32(&r0[w]);
            float32x4_t vr0_5678 = vld1q_f32(&r0[w + 4]);
            float32x4_t vr0_9101112 = vld1q_f32(&r0[w + 8]);
            float32x4_t vr1_1234 = vld1q_f32(&r1[w]);
            float32x4_t vr1_5678 = vld1q_f32(&r1[w + 4]);
            float32x4_t vr1_9101112 = vld1q_f32(&r1[w + 8]);

            float32x4_t vsum_1234 = vaddq_f32(vr0_1234, vr1_1234);
            float32x4_t vsum_5678 = vaddq_f32(vr0_5678, vr1_5678);
            float32x4_t vsum_9101112 = vaddq_f32(vr0_9101112, vr1_9101112);
            float32x4_t vsum_2345 = vextq_f32(vsum_1234, vsum_5678, 1);
            float32x4_t vsum_3456 = vextq_f32(vsum_1234, vsum_5678, 2);
            float32x4_t vsum_4567 = vextq_f32(vsum_1234, vsum_5678, 3);
            float32x4_t vsum_6789 = vextq_f32(vsum_5678, vsum_9101112, 1);
            float32x4_t vsum_123_345 = vaddq_f32(vsum_1234, vsum_2345);
            vsum_123_345 = vaddq_f32(vsum_123_345, vsum_3456);
            float32x4_t vsum_567_789 = vaddq_f32(vsum_4567, vsum_5678);
            vsum_567_789 = vaddq_f32(vsum_567_789, vsum_6789);
            vsum_123_345 = vsetq_lane_f32(vgetq_lane_f32(vsum_123_345, 2),
                                          vsum_123_345, 1);
            vsum_123_345 = vsetq_lane_f32(vgetq_lane_f32(vsum_567_789, 1),
                                          vsum_123_345, 2);
            vsum_123_345 = vsetq_lane_f32(vgetq_lane_f32(vsum_567_789, 3),
                                          vsum_123_345, 3);
            float32x4_t vrst = vmulq_f32(vsum_123_345, vcoef);
            vst1q_f32(&data_out_channel[cnt], vrst);
            cnt += 4;
          }
          for (; w < w_even - 1; w += 2) {
            float32x4_t vr0 = vld1q_f32(&r0[w]);
            float32x4_t vr1 = vld1q_f32(&r1[w]);
            vr0 = vsetq_lane_f32(0.f, vr0, 3);
            vr1 = vsetq_lane_f32(0.f, vr1, 3);
            float32x4_t vsum1 = vaddq_f32(vr0, vr1);
            float32x2_t vsum2 =
                vpadd_f32(vget_low_f32(vsum1), vget_high_f32(vsum1));
            vsum2 = vpadd_f32(vsum2, vsum2);
            float32x2_t vrst = vmul_f32(vsum2, vget_low_f32(vcoef));
            data_out_channel[cnt] = vget_lane_f32(vrst, 0);
            cnt++;
          }
#else
          dr_out = data_out_channel + 1;
          dr0 = (r0 + 1);
          dr1 = (r1 + 1);
          cnt_num = w_unroll_size;
          cnt_num1 = w_unroll_remian;
          if (cnt_num > 0 || cnt_num1 > 0) {
            asm volatile(
                "cmp       %[cnt_num], #0                        @cmp cnt_num, "
                "0\n"
                "ble       3f                                  @ble exit\n"
                "1:                               @main loop\n"
                "vld1.f32  {d0-d3}, [%[dr0]]!                     @load d0-d5, "
                "dr0\n"
                "vld1.f32  {d6-d9}, [%[dr1]]!                    @load d4-d7, "
                "dr1\n"
                "vld1.f32  {d4-d5}, [%[dr0]]!                     @load d0-d3, "
                "dr0\n"
                "vld1.f32  {d10-d11}, [%[dr1]]!                  @load d4-d7, "
                "dr1\n"
                "vmax.f32  q6, q0, q3                            @max q0,q0,q2 "
                "1234\n"
                "vmax.f32  q7, q1, q4                            @max q1,q1,q3 "
                "5678\n"
                "vmax.f32  q8, q2, q5                            @max q1,q1,q3 "
                "9101112\n"
                //"vmov.f32  s7,s6                                 @mov s7,
                // s6\n"
                "vext.f32  q0, q6, q7, #1                        @vext "
                "max_2345\n"
                "vext.f32  q1, q6, q7, #3                        @vext "
                "max_4567\n"
                "vext.f32  q2, q6, q7, #2                        @vext "
                "max_3456\n"
                "vext.f32  q3, q7, q8, #1                        @vext "
                "max_6789\n"
                "vadd.f32  q4, q6, q0                            @add 1234, "
                "2345 \n"
                "vadd.f32  q5, q7, q1                            @add 5678, "
                "4567 \n"
                "vadd.f32  q4, q4, q2                            @add 3456, "
                "sum1 \n"
                "vadd.f32  q5, q5, q3                            @add 6789, "
                "sum2 \n"
                "vmov.f32  s17, s18                              @mov \n"
                "vmov.f32  s18, s21                              @mov \n"
                "vmov.f32  s19, s23                              @mov \n"
                "vmul.f32  q4, q4, %q[vcoef]                     @mul \n"
                "sub       %[dr0], #16                           @add w, 8\n"
                "sub       %[dr1], #16                           @add w, 8\n"
                "subs      %[cnt_num], #1                            @subs "
                "cnt_num, #1\n"
                "vst1.f32  d8, [%[dr_out]]!                      @vst1 d0, "
                "dr_out\n"
                "vst1.f32  d9, [%[dr_out]]!                      @vst1 d0, "
                "dr_out\n"
                "bne       1b                      @bne s3_max_loop_bot\n"
                "3:                                      @loop \n"
                "cmp       %[cnt_num1], #0                           @cmp "
                "cnt_num, 0\n"
                "ble       4f                                 @ble exit\n"
                "2:                             @bot loop\n"
                "vld1.f32  {d0-d1}, [%[dr0]]!                     @load d0-d1, "
                "dr0\n"
                "vld1.f32  {d2-d3}, [%[dr1]]!                     @load d2-d3, "
                "dr1\n"
                "vext.f32  q0, %q[vzero], q0, #3                 @ ext "
                "v0_0123\n"
                "vext.f32  q1, %q[vzero], q1, #3                 @ ext "
                "v1_0123\n"
                "vadd.f32  q0, q0, q1                            @add q0, q0, "
                "q1\n"
                "vpadd.f32 d0, d0, d1                            @padd d0, "
                "d0,d1\n"
                "vpadd.f32 d0, d0, d0                            @padd d0, d0, "
                "d0\n"
                "vmul.f32  d0, d0, %e[vcoef]                     @mul \n"
                "sub       %[dr0], #8                            @add w, 6\n"
                "sub       %[dr1], #8                            @add w, 6\n"
                "subs      %[cnt_num1], #1                           @subs "
                "cnt_num, #1\n"
                "vst1.f32  d0[0], [%[dr_out]]!                   @vst  d0[0], "
                "dr_out\n"
                "bne       2b                    @bne s3_max_loop_bot_1\n"
                "4:                                          @exit\n"
                : [dr0] "+r"(dr0), [dr1] "+r"(dr1), [dr_out] "+r"(dr_out),
                  [cnt_num] "+r"(cnt_num), [cnt_num1] "+r"(cnt_num1),
                  [vcoef] "+w"(vcoef), [vzero] "+w"(vzero)
                : "r"(dr0), "r"(dr1), "r"(dr_out), "r"(cnt_num), "r"(cnt_num1)
                : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6",
                  "q7", "q8", "q9");
          }
#endif
          if (pad_right) {
            // deal with right pad
            int wstart = (w_even >> 1) * stride_w - pad_w;
            int wend = std::min(std::min(wstart + kernel_w, win + pad_w), win);
            float tmp = 0.f;
            for (int i = wstart; i < wend; i++) {  // only run 1 or 2 times
              tmp += (r0[i] + r1[i]);
            }
            data_out_channel[w_even >> 1] = tmp / 9.f;
          }
        }
      }
    }
  }
}

void pooling3x3s2p0_max(const void* din, void* dout, int num, int chout,
                        int hout, int wout, int chin, int hin, int win,
                        const std::vector<int>& ksize,
                        const std::vector<int>& strides,
                        const std::vector<int>& paddings, bool global_pooling,
                        bool exclusive, bool adaptive, bool ceil_mode,
                        bool use_quantizer, const std::string& pooling_type) {
  int w_in = win;
  int h_in = hin;
  int ch_in = chin;

  int w_out = wout;
  int h_out = hout;
  int ch_out = chout;

  int kernel_h = ksize[0];
  int kernel_w = ksize[1];
  int stride_h = strides[0];
  int stride_w = strides[1];
  int pad_h = paddings[0];
  int pad_w = paddings[1];

  int size_channel_out = w_out * h_out;
  int size_channel_in = w_in * h_in;
  float* data_out = static_cast<float*>(dout);
  const float* data_in = static_cast<const float*>(din);

  int pad_top = pad_h;
  int pad_left = pad_w;
  int w_needed = w_out * 2 + 1;
  int h_needed = h_out * 2 + 1;
  int pad_right = w_needed - w_in - pad_left;
  int pad_bottom = h_needed - h_in - pad_top;
  int w_even = ((w_in - 1) >> 1) << 1;
  // int w_remains = w_in - w_even; // should be 0 or 1
  int h_even = ((h_in - 1) >> 1) << 1;
  // int h_remains = h_in - h_even; // should be 0 or 1
  int w_unroll_size = w_in >> 3;
  int w_unroll_remian = (w_in - w_unroll_size * 8 - 1) / 2;
  int w_in_2 = w_in << 1;
  float minval = std::numeric_limits<float>::lowest();
  float32x4_t vzero = vdupq_n_f32(minval);  // zero pad
  // printf("minval: %.2f\n", minval);

  for (int n = 0; n < num; ++n) {
    float* data_out_batch = data_out + n * ch_out * size_channel_out;
    const float* data_in_batch = data_in + n * ch_in * size_channel_in;
#pragma omp parallel for
    for (int c = 0; c < ch_out; c++) {
      float* data_out_channel = data_out_batch + c * size_channel_out;
      const float* data_in_channel = data_in_batch + c * size_channel_in;
      const float* r0 = data_in_channel;
      const float* r1 = r0 + w_in;
      const float* r2 = r1 + w_in;
      int cnt_num = w_unroll_size;
      // w = w_in - 8;
      int cnt_num1 = w_unroll_remian;
      float* dr_out = data_out_channel;
      const float* dr0 = r0;
      const float* dr1 = r1;
      const float* dr2 = r2;
      int w = 0;
      int cnt = 0;
      // data_out_channel[0] = std::max(std::max(r0[0], r0[1]), std::max(r1[0],
      // r1[1]));
      // first row with zero pad
      // r0 = r1;
      // r1 = r0 + w_in;
      // r2 = r1 + w_in;
      // data_out_channel += w_out;
      int h = 0;
      for (; h < h_even; h += 2) {
        // deal with left pad
        float maxr0 = std::max(r0[0], r0[1]);
        float maxr1 = std::max(r1[0], r1[1]);
        float maxr2 = std::max(r2[0], r2[1]);
// data_out_channel[0] = std::max(std::max(maxr0, maxr1), maxr2);
#ifdef __aarch64__
        w = 0;
        cnt = 0;
        for (; w < w_in - 8; w += 8) {
          float32x4_t vr0_1234 = vld1q_f32(&r0[w]);
          float32x4_t vr0_5678 = vld1q_f32(&r0[w + 4]);
          float32x4_t vr0_9101112 = vld1q_f32(&r0[w + 8]);
          float32x4_t vr1_1234 = vld1q_f32(&r1[w]);
          float32x4_t vr1_5678 = vld1q_f32(&r1[w + 4]);
          float32x4_t vr1_9101112 = vld1q_f32(&r1[w + 8]);
          float32x4_t vr2_1234 = vld1q_f32(&r2[w]);
          float32x4_t vr2_5678 = vld1q_f32(&r2[w + 4]);
          float32x4_t vr2_9101112 = vld1q_f32(&r2[w + 8]);
          float32x4_t vmax_1234 = vmaxq_f32(vr0_1234, vr1_1234);
          vmax_1234 = vmaxq_f32(vmax_1234, vr2_1234);
          float32x4_t vmax_5678 = vmaxq_f32(vr0_5678, vr1_5678);
          vmax_5678 = vmaxq_f32(vmax_5678, vr2_5678);
          float32x4_t vmax_9101112 = vmaxq_f32(vr0_9101112, vr1_9101112);
          vmax_9101112 = vmaxq_f32(vmax_9101112, vr2_9101112);
          float32x4_t vmax_2345 = vextq_f32(vmax_1234, vmax_5678, 1);
          float32x4_t vmax_6789 = vextq_f32(vmax_5678, vmax_9101112, 1);
          float32x2_t vmax_12_34 =
              vpmax_f32(vget_low_f32(vmax_1234), vget_high_f32(vmax_1234));
          float32x2_t vmax_23_45 =
              vpmax_f32(vget_low_f32(vmax_2345), vget_high_f32(vmax_2345));
          float32x2_t vmax_56_78 =
              vpmax_f32(vget_low_f32(vmax_5678), vget_high_f32(vmax_5678));
          float32x2_t vmax_67_89 =
              vpmax_f32(vget_low_f32(vmax_6789), vget_high_f32(vmax_6789));
          float32x2_t vmax_123_345 = vmax_f32(vmax_12_34, vmax_23_45);
          float32x2_t vmax_567_789 = vmax_f32(vmax_56_78, vmax_67_89);
          vst1_f32(&data_out_channel[cnt], vmax_123_345);
          vst1_f32(&data_out_channel[cnt + 2], vmax_567_789);
          cnt += 4;
        }
        for (; w < w_even - 1; w += 2) {
          float32x4_t vr0 = vld1q_f32(&r0[w]);
          float32x4_t vr1 = vld1q_f32(&r1[w]);
          float32x4_t vr2 = vld1q_f32(&r2[w]);
          vr0 = vsetq_lane_f32(minval, vr0, 3);
          vr1 = vsetq_lane_f32(minval, vr1, 3);
          vr2 = vsetq_lane_f32(minval, vr2, 3);
          float32x4_t vmax1 = vmaxq_f32(vr0, vr1);
          vmax1 = vmaxq_f32(vmax1, vr2);
          float32x2_t vmax2 =
              vpmax_f32(vget_low_f32(vmax1), vget_high_f32(vmax1));
          float32x2_t vmax = vpmax_f32(vmax2, vmax2);
          data_out_channel[cnt] = vget_lane_f32(vmax, 0);
          cnt++;
        }
#else
        dr_out = data_out_channel;  // + 1;
        dr0 = r0;                   // (r0 + 1);
        dr1 = r1;                   // (r1 + 1);
        dr2 = r2;                   // (r2 + 1);
        cnt_num = w_unroll_size;
        cnt_num1 = w_unroll_remian;
        if (cnt_num > 0 || cnt_num1 > 0) {
          asm volatile(
              "cmp       %[cnt_num], #0                        @cmp cnt_num, "
              "0\n"
              "ble       3f                                  @ble exit\n"
              "1:                                @main loop\n"
              "vld1.f32  {d0-d3}, [%[dr0]]!                     @load d0-d5, "
              "dr0\n"
              "vld1.f32  {d6-d9}, [%[dr1]]!                    @load d4-d7, "
              "dr1\n"
              "vld1.f32  {d12-d15}, [%[dr2]]!                   @load d4-d7, "
              "dr1\n"
              "vld1.f32  {d4}, [%[dr0]]!                     @load d0-d5, dr0\n"
              "vld1.f32  {d10}, [%[dr1]]!                    @load d4-d7, dr1\n"
              "vld1.f32  {d16}, [%[dr2]]!                   @load d4-d7, dr1\n"
              "vmax.f32  q9, q0, q3                            @max q0,q0,q2\n"
              "vmax.f32  q10, q1, q4                           @max q1,q1,q3\n"
              "vmax.f32  d22, d4, d10                           @max q1,q1,q3\n"
              "vmax.f32  q0, q9, q6                            @max q0,q0,q2 "
              "1234\n"
              "vmax.f32  q3, q10, q7                           @max q1,q1,q3 "
              "5678\n"
              "vmax.f32  d2, d22, d16                           @max q1,q1,q3 "
              "9101112\n"
              //"vmov.f32  s7,s6                               @mov s7, s6\n"
              "vext.f32  q4, q0, q3, #1                        @vext 2345\n"
              "vext.f32  q2, q3, q1, #1                        @vext 6789\n"
              "vpmax.f32 d10, d0, d1                           @pmax d10, "
              "vmax_1234, vmax_1234\n"
              "vpmax.f32 d12, d6, d7                           @pmax d12, "
              "vmax_5678, vmax_5678\n"
              "vpmax.f32 d11, d8, d9                           @pmax d11, "
              "vmax_2345, vmax_2345\n"
              "vpmax.f32 d13, d4, d5                           @pmax d13, "
              "vmax_6789, vmax_6789\n"
              "vmax.f32 d0, d10, d11                          @pmax d0, "
              "vmax_12_34, vmax_23_45\n"
              "vmax.f32 d1, d12, d13                          @pmax d1, "
              "vmax_56_78, vmax_67_89\n"
              "sub       %[dr0], #8                           @add w, 8\n"
              "sub       %[dr1], #8                           @add w, 8\n"
              "sub       %[dr2], #8                           @add w, 8\n"
              "vst1.f32  d0, [%[dr_out]]!                      @vst1 d0, "
              "dr_out\n"
              "vst1.f32  d1, [%[dr_out]]!                      @vst1 d0, "
              "dr_out\n"
              "subs      %[cnt_num], #1                            @subs "
              "cnt_num, #1\n"
              "bne       1b                       @bne s3_max_loop_mid\n"
              "3:                                       @loop \n"
              "cmp       %[cnt_num1], #0                           @cmp "
              "cnt_num, 0\n"
              "ble       4f                                 @ble exit1\n"
              "2:                              @mid loop\n"
              "vld1.f32  {d0-d1}, [%[dr0]]!                     @load d0-d1, "
              "dr0\n"
              "vld1.f32  {d2-d3}, [%[dr1]]!                    @load d2-d3, "
              "dr1\n"
              "vld1.f32  {d4-d5}, [%[dr2]]!                     @load d2-d3, "
              "dr1\n"
              "vmov.f32  s3,s2                                 @movs3, s2\n"
              "vmov.f32  s7,s6                                 @movs7, s6\n"
              "vmov.f32  s11,s10                               @movs11, s10\n"
              "vmax.f32  q0, q0, q1                            @max q0, q0, "
              "q1\n"
              "vmax.f32  q0, q0, q2                            @max q0, q0, "
              "q2\n"
              "vpmax.f32 d0, d0, d1                            @pmax d0, "
              "d0,d1\n"
              "vpmax.f32 d0, d0, d0                            @pmax d0, d0, "
              "d0\n"
              "vst1.f32  d0[0], [%[dr_out]]!                   @vst  d0[0], "
              "dr_out\n"
              "sub       %[dr0], #8                            @add w, 6\n"
              "sub       %[dr1], #8                            @add w, 6\n"
              "sub       %[dr2], #8                            @add w, 6\n"
              "subs      %[cnt_num1], #1                       @subs cnt_num, "
              "#1\n"
              "bne       2b                     @bne s3_max_loop_mid_1\n"
              "4:                                           @exit\n"
              : [dr0] "+r"(dr0), [dr1] "+r"(dr1), [dr2] "+r"(dr2),
                [dr_out] "+r"(dr_out), [cnt_num] "+r"(cnt_num),
                [cnt_num1] "+r"(cnt_num1)
              : "r"(dr0), "r"(dr1), "r"(dr2), "r"(dr_out), "r"(cnt_num),
                "r"(cnt_num1)
              : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7",
                "q8", "q9", "q10", "q11", "q12");
        }
#endif
        if (pad_right) {
          // deal with right pad
          int wstart = (w_even >> 1) * stride_w - pad_w;
          int wend = std::min(std::min(wstart + kernel_w, w_in + pad_w), w_in);
          float tmp = r0[wstart];  // std::numeric_limits<float>::min();
          for (int i = wstart; i < wend; i++) {
            tmp = std::max(tmp, std::max(r0[i], r1[i]));
            tmp = std::max(tmp, r2[i]);
          }
          data_out_channel[w_even >> 1] = tmp;
          // cnt ++;
        }
        r0 = r2;
        r1 = r0 + w_in;
        r2 = r1 + w_in;
        data_out_channel += w_out;
      }

      if (pad_bottom) {
// deal with bottom pad
// first row with zero pad
// int hstart = (h >> 1) * stride_h - pad_h;
// int hend = std::min(std::min(hstart + kernel_h, h_in + pad_h),h_in);
// data_out_channel[0] = std::max(std::max(r0[0], r0[1]), std::max(r1[0],
// r1[1]));
#ifdef __aarch64__
        w = 0;
        cnt = 0;
        for (; w < w_in - 8; w += 8) {
          float32x4_t vr0_1234 = vld1q_f32(&r0[w]);
          float32x4_t vr0_5678 = vld1q_f32(&r0[w + 4]);
          float32x4_t vr0_9101112 = vld1q_f32(&r0[w + 8]);
          float32x4_t vr1_1234 = vld1q_f32(&r1[w]);
          float32x4_t vr1_5678 = vld1q_f32(&r1[w + 4]);
          float32x4_t vr1_9101112 = vld1q_f32(&r1[w + 8]);
          float32x4_t vmax_1234 = vmaxq_f32(vr0_1234, vr1_1234);
          float32x4_t vmax_5678 = vmaxq_f32(vr0_5678, vr1_5678);
          float32x4_t vmax_9101112 = vmaxq_f32(vr0_9101112, vr1_9101112);
          float32x4_t vmax_2345 = vextq_f32(vmax_1234, vmax_5678, 1);
          float32x4_t vmax_6789 = vextq_f32(vmax_5678, vmax_9101112, 1);
          float32x2_t vmax_12_34 =
              vpmax_f32(vget_low_f32(vmax_1234), vget_high_f32(vmax_1234));
          float32x2_t vmax_23_45 =
              vpmax_f32(vget_low_f32(vmax_2345), vget_high_f32(vmax_2345));
          float32x2_t vmax_56_78 =
              vpmax_f32(vget_low_f32(vmax_5678), vget_high_f32(vmax_5678));
          float32x2_t vmax_67_89 =
              vpmax_f32(vget_low_f32(vmax_6789), vget_high_f32(vmax_6789));
          float32x2_t vmax_123_345 = vmax_f32(vmax_12_34, vmax_23_45);
          float32x2_t vmax_567_789 = vmax_f32(vmax_56_78, vmax_67_89);
          vst1_f32(&data_out_channel[cnt], vmax_123_345);
          vst1_f32(&data_out_channel[cnt + 2], vmax_567_789);
          cnt += 4;
        }
        for (; w < w_even - 1; w += 2) {
          float32x4_t vr0 = vld1q_f32(&r0[w]);
          float32x4_t vr1 = vld1q_f32(&r1[w]);
          vr0 = vsetq_lane_f32(minval, vr0, 3);
          vr1 = vsetq_lane_f32(minval, vr1, 3);
          float32x4_t vmax1 = vmaxq_f32(vr0, vr1);
          float32x2_t vmax2 =
              vpmax_f32(vget_low_f32(vmax1), vget_high_f32(vmax1));
          vmax2 = vpmax_f32(vmax2, vmax2);
          data_out_channel[cnt] = vget_lane_f32(vmax2, 0);
          cnt++;
        }
#else
        dr_out = data_out_channel;  // + 1;
        dr0 = r0;                   // (r0 + 1);
        dr1 = r1;                   // (r1 + 1);
        cnt_num = w_unroll_size;
        cnt_num1 = w_unroll_remian;
        if (cnt_num > 0 || cnt_num1 > 0) {
          asm volatile(
              "cmp       %[cnt_num], #0                        @cmp cnt_num, "
              "0\n"
              "ble       3f                                  @ble exit\n"
              "1:                               @main loop\n"
              "vld1.f32  {d0-d3}, [%[dr0]]!                     @load d0-d5, "
              "dr0\n"
              "vld1.f32  {d6-d9}, [%[dr1]]!                    @load d4-d7, "
              "dr1\n"
              "vld1.f32  {d4}, [%[dr0]]!                     @load d0-d3, dr0\n"
              "vld1.f32  {d10}, [%[dr1]]!                  @load d4-d7, dr1\n"
              "vmax.f32  q6, q0, q3                            @max q0,q0,q2 "
              "1234\n"
              "vmax.f32  q7, q1, q4                            @max q1,q1,q3 "
              "5678\n"
              "vmax.f32  d16, d4, d10                            @max q1,q1,q3 "
              "9101112\n"
              //"vmov.f32  s7,s6                                 @mov s7, s6\n"
              "vext.f32  q0, q6, q7, #1                        @vext q0, 2345\n"
              "vext.f32  q1, q7, q8, #1                        @vext q1, 6789\n"
              "vpmax.f32 d4, d12, d13                          @pmax d4, "
              "vmax_1234, vmax_1234\n"
              "vpmax.f32 d6, d14, d15                          @pmax d6, "
              "vmax_5678, vmax_5678\n"
              "vpmax.f32 d5, d0, d1                            @pmax d5, "
              "vmax_2345, vmax_2345\n"
              "vpmax.f32 d7, d2, d3                            @pmax d7, "
              "vmax_6789, vmax_6789\n"
              "vmax.f32 d8, d4, d5                             @max d2, "
              "vmax_12_34, vmax_23_45\n"
              "vmax.f32 d9, d6, d7                             @max d2, "
              "vmax_56_78, vmax_67_89\n"
              "sub       %[dr0], #8                           @add w, 8\n"
              "sub       %[dr1], #8                           @add w, 8\n"
              "vst1.f32  d8, [%[dr_out]]!                      @vst1 d0, "
              "dr_out\n"
              "vst1.f32  d9, [%[dr_out]]!                      @vst1 d0, "
              "dr_out\n"
              "subs      %[cnt_num], #1                            @subs "
              "cnt_num, #1\n"
              "bne       1b                      @bne s3_max_loop_bot\n"
              "3:                                      @loop \n"
              "cmp       %[cnt_num1], #0                           @cmp "
              "cnt_num, 0\n"
              "ble       4f                                 @ble exit\n"
              "2:                             @bot loop\n"
              "vld1.f32  {d0-d1}, [%[dr0]]!                     @load d0-d1, "
              "dr0\n"
              "vld1.f32  {d2-d3}, [%[dr1]]!                     @load d2-d3, "
              "dr1\n"
              "vmov.f32  s3,s2                                 @movs3, s2\n"
              "vmov.f32  s7,s6                                 @movs7, s6\n"
              "vmax.f32  q0, q0, q1                            @max q0, q0, "
              "q1\n"
              "vpmax.f32 d0, d0, d1                            @pmax d0, "
              "d0,d1\n"
              "vpmax.f32 d0, d0, d0                            @pmax d0, d0, "
              "d0\n"
              "vst1.f32  d0[0], [%[dr_out]]!                   @vst  d0[0], "
              "dr_out\n"
              "sub       %[dr0], #8                            @add w, 6\n"
              "sub       %[dr1], #8                            @add w, 6\n"
              "subs      %[cnt_num1], #1                           @subs "
              "cnt_num, #1\n"
              "bne       2b                    @bne s3_max_loop_bot_1\n"
              "4:                                          @exit\n"
              : [dr0] "+r"(dr0), [dr1] "+r"(dr1), [dr_out] "+r"(dr_out),
                [cnt_num] "+r"(cnt_num), [cnt_num1] "+r"(cnt_num1)
              : "r"(dr0), "r"(dr1), "r"(dr_out), "r"(cnt_num), "r"(cnt_num1)
              : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7",
                "q8", "q9");
        }
#endif
        if (pad_right) {
          // deal with right pad
          int wstart = (w_even >> 1) * stride_w - pad_w;
          int wend = std::min(std::min(wstart + kernel_w, w_in + pad_w), w_in);
          float tmp = r0[wstart];  // std::numeric_limits<float>::min();
          for (int i = wstart; i < wend; i++) {  // only run 1 or 2 times
            tmp = std::max(tmp, std::max(r0[i], r1[i]));
          }
          data_out_channel[w_even >> 1] = tmp;
        }
      }
    }
  }
}

void pooling3x3s2p0_ave(const void* din, void* dout, int num, int chout,
                        int hout, int wout, int chin, int hin, int win,
                        const std::vector<int>& ksize,
                        const std::vector<int>& strides,
                        const std::vector<int>& paddings, bool global_pooling,
                        bool exclusive, bool adaptive, bool ceil_mode,
                        bool use_quantizer, const std::string& pooling_type) {
  int w_in = win;
  int h_in = hin;
  int ch_in = chin;

  int w_out = wout;
  int h_out = hout;
  int ch_out = chout;

  int kernel_h = ksize[0];
  int kernel_w = ksize[1];
  int stride_h = strides[0];
  int stride_w = strides[1];
  int pad_h = paddings[0];
  int pad_w = paddings[1];

  int size_channel_out = w_out * h_out;
  int size_channel_in = w_in * h_in;
  float* data_out = static_cast<float*>(dout);
  const float* data_in = static_cast<const float*>(din);

  int pad_top = pad_h;
  int pad_left = pad_w;
  int w_needed = w_out * 2 + 1;
  int h_needed = h_out * 2 + 1;
  int pad_right = w_needed - w_in - pad_left;
  int pad_bottom = h_needed - h_in - pad_top;
  int w_even = ((w_in - 1) >> 1) << 1;
  int h_even = ((h_in - 1) >> 1) << 1;
  int w_in_2 = w_in << 1;
  int w_unroll_size = w_in >> 3;
  int w_unroll_remian = (w_even - w_unroll_size * 8 - 1) / 2;
  for (int n = 0; n < num; ++n) {
    float* data_out_batch = data_out + n * ch_out * size_channel_out;
    const float* data_in_batch = data_in + n * ch_in * size_channel_in;
#pragma omp parallel for
    for (int c = 0; c < ch_out; c++) {
      float* data_out_channel = data_out_batch + c * size_channel_out;
      const float* data_in_channel = data_in_batch + c * size_channel_in;
      const float* r0 = data_in_channel;
      const float* r1 = r0 + w_in;
      const float* r2 = r1 + w_in;
      int cnt_num = w_unroll_size;
      // w = w_in - 8;
      int cnt_num1 = w_unroll_remian;
      float* dr_out = data_out_channel;
      const float* dr0 = r0;
      const float* dr1 = r1;
      const float* dr2 = r2;

      float32x4_t vcoef = vdupq_n_f32(1.f / 9.f);
      float32x4_t vzero = vdupq_n_f32(0.f);

      int h = 0;
      for (; h < h_even; h += 2) {
// LOG(INFO) << "h: " << h<<", dr0:" << r0 <<", dr1: "<<r1 << ",dr2: "<<r2;
// deal with left pad
//   float sum0 = r0[0] + r0[1];
//   float sum1 = r1[0] + r1[1];
//  float sum2 = r2[0] + r2[1];
//  data_out_channel[0] = (sum0 + sum1 + sum2) / 9.f;
#if 1  // def __aarch64__
        int w = 0;
        int cnt = 0;
        for (; w < w_in - 8; w += 8) {
          float32x4_t vr0_1234 = vld1q_f32(&r0[w]);
          float32x4_t vr0_5678 = vld1q_f32(&r0[w + 4]);
          float32x4_t vr0_9101112 = vld1q_f32(&r0[w + 8]);
          float32x4_t vr1_1234 = vld1q_f32(&r1[w]);
          float32x4_t vr1_5678 = vld1q_f32(&r1[w + 4]);
          float32x4_t vr1_9101112 = vld1q_f32(&r1[w + 8]);
          float32x4_t vr2_1234 = vld1q_f32(&r2[w]);
          float32x4_t vr2_5678 = vld1q_f32(&r2[w + 4]);
          float32x4_t vr2_9101112 = vld1q_f32(&r2[w + 8]);
          float32x4_t vsum_1234 = vaddq_f32(vr0_1234, vr1_1234);
          float32x4_t vsum_5678 = vaddq_f32(vr0_5678, vr1_5678);
          float32x4_t vsum_9101112 = vaddq_f32(vr0_9101112, vr1_9101112);
          vsum_1234 = vaddq_f32(vsum_1234, vr2_1234);
          vsum_5678 = vaddq_f32(vsum_5678, vr2_5678);
          vsum_9101112 = vaddq_f32(vsum_9101112, vr2_9101112);

          float32x4_t vsum_2345 = vextq_f32(vsum_1234, vsum_5678, 1);
          float32x4_t vsum_3456 = vextq_f32(vsum_1234, vsum_5678, 2);
          float32x4_t vsum_4567 = vextq_f32(vsum_1234, vsum_5678, 3);
          float32x4_t vsum_6789 = vextq_f32(vsum_5678, vsum_9101112, 1);
          float32x4_t vsum_123_345 = vaddq_f32(vsum_1234, vsum_2345);
          vsum_123_345 = vaddq_f32(vsum_123_345, vsum_3456);
          float32x4_t vsum_567_789 = vaddq_f32(vsum_4567, vsum_5678);
          vsum_567_789 = vaddq_f32(vsum_567_789, vsum_6789);
          vsum_123_345 =
              vsetq_lane_f32(vgetq_lane_f32(vsum_123_345, 2), vsum_123_345, 1);
          vsum_123_345 =
              vsetq_lane_f32(vgetq_lane_f32(vsum_567_789, 1), vsum_123_345, 2);
          vsum_123_345 =
              vsetq_lane_f32(vgetq_lane_f32(vsum_567_789, 3), vsum_123_345, 3);
          float32x4_t vrst = vmulq_f32(vsum_123_345, vcoef);
          vst1q_f32(&data_out_channel[cnt], vrst);
          cnt += 4;
        }
        for (; w < w_even - 1; w += 2) {
          float32x4_t vr0 = vld1q_f32(&r0[w]);
          float32x4_t vr1 = vld1q_f32(&r1[w]);
          float32x4_t vr2 = vld1q_f32(&r2[w]);
          vr0 = vsetq_lane_f32(0.f, vr0, 3);
          vr1 = vsetq_lane_f32(0.f, vr1, 3);
          vr2 = vsetq_lane_f32(0.f, vr2, 3);
          float32x4_t vsum1 = vaddq_f32(vr0, vr1);
          vsum1 = vaddq_f32(vsum1, vr2);
          float32x2_t vsum2 =
              vpadd_f32(vget_low_f32(vsum1), vget_high_f32(vsum1));
          float32x2_t vsum = vpadd_f32(vsum2, vsum2);
          data_out_channel[cnt] = vget_lane_f32(vsum, 0) / 9.f;
          cnt++;
        }
#else
        dr_out = data_out_channel;  // + 1;
        dr0 = r0;                   // (r0 + 1);
        dr1 = r1;                   // (r1 + 1);
        dr2 = r2;                   // (r2 + 1);
        cnt_num = w_unroll_size;
        cnt_num1 = w_unroll_remian;
        //  LOG(INFO) << "cnt_num: " << cnt_num <<"cnt_num1: "<< cnt_num1;
        if (cnt_num > 0 || cnt_num1 > 0) {
          asm volatile(
              "cmp       %[cnt_num], #0                        @cmp cnt_num, "
              "0\n"
              "ble       loop3_ave_p0                                  @ble "
              "exit\n"
              "s3_ave_loop_mid_p0:                                @main loop\n"
              "vld1.f32  {d0-d3}, [%[dr0]]!                     @load d0-d5, "
              "dr0\n"
              "vld1.f32  {d6-d9}, [%[dr1]]!                    @load d4-d7, "
              "dr1\n"
              "vld1.f32  {d12-d15}, [%[dr2]]!                   @load d4-d7, "
              "dr1\n"
              "vld1.f32  {d4}, [%[dr0]]!                     @load d0-d5, dr0\n"
              "vld1.f32  {d10}, [%[dr1]]!                    @load d4-d7, dr1\n"
              "vld1.f32  {d16}, [%[dr2]]!                   @load d4-d7, dr1\n"
              "vadd.f32  q9, q0, q3                            @max q0,q0,q2\n"
              "vadd.f32  q10, q1, q4                           @max q1,q1,q3\n"
              "vadd.f32  d22, d4, d10                           @max q1,q1,q3\n"
              "vadd.f32  q6, q9, q6                            @max q0,q0,q2 "
              "1234\n"
              "vadd.f32  q7, q10, q7                           @max q1,q1,q3 "
              "5678\n"
              "vadd.f32  d16, d22, d16                           @max q1,q1,q3 "
              "9101112\n"
              //"vmov.f32  s7,s6                               @mov s7, s6\n"
              "vext.f32  q0, q6, q7, #1                        @vext max_2345\n"
              "vext.f32  q1, q6, q7, #3                        @vext max_4567\n"
              "vext.f32  q2, q6, q7, #2                        @vext max_3456\n"
              "vext.f32  q3, q7, q8, #1                        @vext max_6789\n"
              "vadd.f32  q4, q6, q0                            @add 1234, 2345 "
              "\n"
              "vadd.f32  q5, q7, q1                            @add 5678, 4567 "
              "\n"
              "vadd.f32  q4, q4, q2                            @add 3456, sum1 "
              "\n"
              "vadd.f32  q5, q5, q3                            @add 6789, sum2 "
              "\n"
              "vmov.f32  s17, s18                              @mov \n"
              "vmov.f32  s18, s21                              @mov \n"
              "vmov.f32  s19, s23                              @mov \n"
              "vmul.f32  q4, q4, %q[vcoef]                     @mul \n"
              "sub       %[dr0], #8                           @add w, 8\n"
              "sub       %[dr1], #8                           @add w, 8\n"
              "sub       %[dr2], #8                           @add w, 8\n"
              "subs      %[cnt_num], #1                            @subs "
              "cnt_num, #1\n"
              "vst1.f32  d8, [%[dr_out]]!                      @vst1 d0, "
              "dr_out\n"
              "vst1.f32  d9, [%[dr_out]]!                      @vst1 d0, "
              "dr_out\n"
              "bne       s3_ave_loop_mid_p0                       @bne "
              "s3_max_loop_mid\n"
              "loop3_ave_p0:                                       @loop \n"
              "cmp       %[cnt_num1], #0                           @cmp "
              "cnt_num, 0\n"
              "ble       exit1_ave_p0                                 @ble "
              "exit1\n"
              "s3_ave_loop_mid_1_p0:                              @mid loop\n"
              "vld1.f32  {d0-d1}, [%[dr0]]!                     @load d0-d1, "
              "dr0\n"
              "vld1.f32  {d2-d3}, [%[dr1]]!                    @load d2-d3, "
              "dr1\n"
              "vld1.f32  {d4-d5}, [%[dr2]]!                     @load d2-d3, "
              "dr1\n"
              "vext.f32  q0, %q[vzero], q0, #3                 @ ext v0_0123\n"
              "vext.f32  q1, %q[vzero], q1, #3                 @ ext v1_0123\n"
              "vext.f32  q2, %q[vzero], q2, #3                 @ ext v1_0123\n"
              "vadd.f32  q0, q0, q1                            @add q0, q0, "
              "q1\n"
              "vadd.f32  q0, q0, q2                            @add q0, q0, "
              "q1\n"
              "vpadd.f32 d0, d0, d1                            @padd d0, "
              "d0,d1\n"
              "vpadd.f32 d0, d0, d0                            @padd d0, d0, "
              "d0\n"
              "vmul.f32  d0, d0, %e[vcoef]                     @mul \n"
              "sub       %[dr0], #8                            @add w, 6\n"
              "sub       %[dr1], #8                            @add w, 6\n"
              "sub       %[dr2], #8                            @add w, 6\n"
              "subs      %[cnt_num1], #1                       @subs cnt_num, "
              "#1\n"
              "vst1.f32  d0[0], [%[dr_out]]!                   @vst  d0[0], "
              "dr_out\n"
              "bne       s3_ave_loop_mid_1_p0                     @bne "
              "s3_max_loop_mid_1\n"
              "exit1_ave_p0:                                           @exit\n"
              : [dr0] "+r"(dr0), [dr1] "+r"(dr1), [dr2] "+r"(dr2),
                [dr_out] "+r"(dr_out), [cnt_num] "+r"(cnt_num),
                [cnt_num1] "+r"(cnt_num1), [vcoef] "+w"(vcoef),
                [vzero] "+w"(vzero)
              : "r"(dr0), "r"(dr1), "r"(dr2), "r"(dr_out), "r"(cnt_num),
                "r"(cnt_num1)
              : "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9",
                "q10", "q11", "q12");
        }
#endif
        if (pad_right) {
          // deal with right pad
          int wstart = (w_even >> 1) * stride_w - pad_w;
          int wend = std::min(std::min(wstart + kernel_w, w_in + pad_w), w_in);
          float tmp = 0.f;
          int pool_size = 3 * (wend - wstart);
          for (int i = wstart; i < wend; i++) {
            tmp += (r0[i] + r1[i] + r2[i]);
          }
          data_out_channel[w_even >> 1] = tmp / pool_size;
          // cnt ++;
        }
        r0 = r2;
        r1 = r0 + w_in;
        r2 = r1 + w_in;
        data_out_channel += w_out;
      }

      if (pad_bottom) {
// deal with bottom pad
// first row with zero pad
// int hstart = (h >> 1) * stride_h - pad_h;
// int hend = std::min(std::min(hstart + kernel_h, h_in + pad_h),h_in);
// data_out_channel[0] =(r0[0] + r0[1] + r1[0] + r1[1]) / 9.f;
#if 1  // def __aarch64__
        int w = 0;
        int cnt = 0;
        vcoef = vdupq_n_f32(1.f / 6.f);
        for (; w < w_in - 8; w += 8) {
          float32x4_t vr0_1234 = vld1q_f32(&r0[w]);
          float32x4_t vr0_5678 = vld1q_f32(&r0[w + 4]);
          float32x4_t vr0_9101112 = vld1q_f32(&r0[w + 8]);
          float32x4_t vr1_1234 = vld1q_f32(&r1[w]);
          float32x4_t vr1_5678 = vld1q_f32(&r1[w + 4]);
          float32x4_t vr1_9101112 = vld1q_f32(&r1[w + 8]);

          float32x4_t vsum_1234 = vaddq_f32(vr0_1234, vr1_1234);
          float32x4_t vsum_5678 = vaddq_f32(vr0_5678, vr1_5678);
          float32x4_t vsum_9101112 = vaddq_f32(vr0_9101112, vr1_9101112);
          float32x4_t vsum_2345 = vextq_f32(vsum_1234, vsum_5678, 1);
          float32x4_t vsum_3456 = vextq_f32(vsum_1234, vsum_5678, 2);
          float32x4_t vsum_4567 = vextq_f32(vsum_1234, vsum_5678, 3);
          float32x4_t vsum_6789 = vextq_f32(vsum_5678, vsum_9101112, 1);
          float32x4_t vsum_123_345 = vaddq_f32(vsum_1234, vsum_2345);
          vsum_123_345 = vaddq_f32(vsum_123_345, vsum_3456);
          float32x4_t vsum_567_789 = vaddq_f32(vsum_4567, vsum_5678);
          vsum_567_789 = vaddq_f32(vsum_567_789, vsum_6789);
          vsum_123_345 =
              vsetq_lane_f32(vgetq_lane_f32(vsum_123_345, 2), vsum_123_345, 1);
          vsum_123_345 =
              vsetq_lane_f32(vgetq_lane_f32(vsum_567_789, 1), vsum_123_345, 2);
          vsum_123_345 =
              vsetq_lane_f32(vgetq_lane_f32(vsum_567_789, 3), vsum_123_345, 3);
          float32x4_t vrst = vmulq_f32(vsum_123_345, vcoef);
          vst1q_f32(&data_out_channel[cnt], vrst);
          cnt += 4;
        }
        for (; w < w_even - 1; w += 2) {
          float32x4_t vr0 = vld1q_f32(&r0[w]);
          float32x4_t vr1 = vld1q_f32(&r1[w]);
          vr0 = vsetq_lane_f32(0.f, vr0, 3);
          vr1 = vsetq_lane_f32(0.f, vr1, 3);
          float32x4_t vsum1 = vaddq_f32(vr0, vr1);
          float32x2_t vsum2 =
              vpadd_f32(vget_low_f32(vsum1), vget_high_f32(vsum1));
          vsum2 = vpadd_f32(vsum2, vsum2);
          float32x2_t vrst = vmul_f32(vsum2, vget_low_f32(vcoef));
          data_out_channel[cnt] = vget_lane_f32(vrst, 0);
          cnt++;
        }
#else
        dr_out = data_out_channel;  // + 1;
        dr0 = r0;                   // (r0 + 1);
        dr1 = r1;                   // (r1 + 1);
        cnt_num = w_unroll_size;
        cnt_num1 = w_unroll_remian;
        // LOG(INFO) << "dr0:" << dr0 <<", dr1: "<<dr1 << ",dr2: "<<dr2;
        if (cnt_num > 0 || cnt_num1 > 0) {
          asm volatile(
              "cmp       %[cnt_num], #0                        @cmp cnt_num, "
              "0\n"
              "ble       2f                                  @ble exit\n"
              "1:                               @main loop\n"
              "vld1.f32  {d0-d3}, [%[dr0]]!                     @load d0-d5, "
              "dr0\n"
              "vld1.f32  {d6-d9}, [%[dr1]]!                    @load d4-d7, "
              "dr1\n"
              "vld1.f32  {d4}, [%[dr0]]!                     @load d0-d3, dr0\n"
              "vld1.f32  {d10}, [%[dr1]]!                  @load d4-d7, dr1\n"
              "vadd.f32  q6, q0, q3                            @max q0,q0,q2 "
              "1234\n"
              "vadd.f32  q7, q1, q4                            @max q1,q1,q3 "
              "5678\n"
              "vadd.f32  d16, d4, d10                           @max q1,q1,q3 "
              "9101112\n"
              //"vmov.f32  s7,s6                                 @mov s7, s6\n"
              "vext.f32  q0, q6, q7, #1                        @vext max_2345\n"
              "vext.f32  q1, q6, q7, #3                        @vext max_4567\n"
              "vext.f32  q2, q6, q7, #2                        @vext max_3456\n"
              "vext.f32  q3, q7, q8, #1                        @vext max_6789\n"
              "vadd.f32  q4, q6, q0                            @add 1234, 2345 "
              "\n"
              "vadd.f32  q5, q7, q1                            @add 5678, 4567 "
              "\n"
              "vadd.f32  q4, q4, q2                            @add 3456, sum1 "
              "\n"
              "vadd.f32  q5, q5, q3                            @add 6789, sum2 "
              "\n"
              "vmov.f32  s17, s18                              @mov \n"
              "vmov.f32  s18, s21                              @mov \n"
              "vmov.f32  s19, s23                              @mov \n"
              "vmul.f32  q4, q4, %q[vcoef]                     @mul \n"
              "sub       %[dr0], #8                           @add w, 8\n"
              "sub       %[dr1], #8                           @add w, 8\n"
              "subs      %[cnt_num], #1                            @subs "
              "cnt_num, #1\n"
              "vst1.f32  d8, [%[dr_out]]!                      @vst1 d0, "
              "dr_out\n"
              "vst1.f32  d9, [%[dr_out]]!                      @vst1 d0, "
              "dr_out\n"
              "bne       1b                      @bne s3_max_loop_bot\n"
              "2:                                      @loop \n"
              "cmp       %[cnt_num1], #0                           @cmp "
              "cnt_num, 0\n"
              "ble       3f                                 @ble exit\n"
              "4:                             @bot loop\n"
              "vld1.f32  {d0-d1}, [%[dr0]]!                     @load d0-d1, "
              "dr0\n"
              "vld1.f32  {d2-d3}, [%[dr1]]!                     @load d2-d3, "
              "dr1\n"
              "vext.f32  q0, %q[vzero], q0, #3                 @ ext v0_0123\n"
              "vext.f32  q1, %q[vzero], q1, #3                 @ ext v1_0123\n"
              "vadd.f32  q0, q0, q1                            @add q0, q0, "
              "q1\n"
              "vpadd.f32 d0, d0, d1                            @padd d0, "
              "d0,d1\n"
              "vpadd.f32 d0, d0, d0                            @padd d0, d0, "
              "d0\n"
              "vmul.f32  d0, d0, %e[vcoef]                     @mul \n"
              "sub       %[dr0], #8                            @add w, 6\n"
              "sub       %[dr1], #8                            @add w, 6\n"
              "subs      %[cnt_num1], #1                           @subs "
              "cnt_num, #1\n"
              "vst1.f32  d0[0], [%[dr_out]]!                   @vst  d0[0], "
              "dr_out\n"
              "bne       4b                    @bne s3_max_loop_bot_1\n"
              "3:                                          @exit\n"
              : [dr0] "+r"(dr0), [dr1] "+r"(dr1), [dr_out] "+r"(dr_out),
                [cnt_num] "+r"(cnt_num), [cnt_num1] "+r"(cnt_num1),
                [vcoef] "+w"(vcoef), [vzero] "+w"(vzero)
              : "r"(dr0), "r"(dr1), "r"(dr_out), "r"(cnt_num), "r"(cnt_num1)
              : "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9");
        }

#endif
        if (pad_right) {
          // deal with right pad
          int wstart = (w_even >> 1) * stride_w - pad_w;
          int wend = std::min(std::min(wstart + kernel_w, w_in + pad_w), w_in);
          float tmp = 0.f;
          int pool_size = 2 * (wend - wstart);
          for (int i = wstart; i < wend; i++) {  // only run 1 or 2 times
            tmp += (r0[i] + r1[i]);
          }
          data_out_channel[w_even >> 1] = tmp / pool_size;
        }
      }
    }
  }
}

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
