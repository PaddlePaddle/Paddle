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

void pooling_basic(const float* din, float* dout, int num, int chout, int hout,
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
  if (global_pooling) {
    if (pooling_type == "max") {  // Pooling_max
      for (int n = 0; n < num; ++n) {
        float* dout_batch = dout + n * chout * size_channel_out;
        const float* din_batch = din + n * chin * size_channel_in;
#pragma omp parallel for
        for (int c = 0; c < chout; ++c) {
          const float* din_ch = din_batch + c * size_channel_in;  // in address
          float tmp1 = din_ch[0];
          for (int i = 0; i < size_channel_in; ++i) {
            float tmp2 = din_ch[i];
            tmp1 = tmp1 > tmp2 ? tmp1 : tmp2;
          }
          dout_batch[c] = tmp1;
        }
      }
    } else if (pooling_type == "avg") {
      // Pooling_average_include_padding
      // Pooling_average_exclude_padding
      for (int n = 0; n < num; ++n) {
        float* dout_batch = dout + n * chout * size_channel_out;
        const float* din_batch = din + n * chin * size_channel_in;
#pragma omp parallel for
        for (int c = 0; c < chout; ++c) {
          const float* din_ch = din_batch + c * size_channel_in;  // in address
          float sum = 0.f;
          for (int i = 0; i < size_channel_in; ++i) {
            sum += din_ch[i];
          }
          dout_batch[c] = sum / size_channel_in;
        }
      }
    } else {
      LOG(FATAL) << "unsupported pooling type: " << pooling_type;
    }
  } else {
    if (pooling_type == "max") {
      // Pooling_max
      for (int n = 0; n < num; ++n) {
        float* dout_ch = dout + n * chout * size_channel_out;
        const float* din_batch = din + n * chin * size_channel_in;
#pragma omp parallel for
        for (int c = 0; c < chout; c++) {
          float* dout_row = dout_ch + c * size_channel_out;
          const float* din_ch = din_batch + c * size_channel_in;
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
              int pool_size = (hend - hstart) * (wend - wstart);
              if (pool_size == 0) continue;
              float tmp1 = din_ch[hstart * win + wstart];
              for (int h = hstart; h < hend; ++h) {
                for (int w = wstart; w < wend; ++w) {
                  float tmp2 = din_ch[h * win + w];
                  tmp1 = tmp1 > tmp2 ? tmp1 : tmp2;
                }
              }
              dout_row[j] = tmp1;
            }
            dout_row += wout;
          }
        }
      }
    } else if (pooling_type == "avg") {
      if (exclusive) {
        // Pooling_average_exclude_padding
        for (int n = 0; n < num; ++n) {
          float* dout_ch = dout + n * chout * size_channel_out;
          const float* din_batch = din + n * chin * size_channel_in;
#pragma omp parallel for
          for (int c = 0; c < chout; c++) {
            float* dout_row = dout_ch + c * size_channel_out;
            const float* din_ch = din_batch + c * size_channel_in;
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
                int pool_size = (hend - hstart) * (wend - wstart);
                if (pool_size == 0) continue;
                float sum = 0.f;
                for (int h = hstart; h < hend; ++h) {
                  for (int w = wstart; w < wend; ++w) {
                    sum += din_ch[h * win + w];
                  }
                }
                dout_row[j] = sum / pool_size;
              }
              dout_row += wout;
            }
          }
        }
      } else {  // Pooling_average_include_padding
        for (int n = 0; n < num; ++n) {
          float* dout_ch = dout + n * chout * size_channel_out;
          const float* din_batch = din + n * chin * size_channel_in;
#pragma omp parallel for
          for (int c = 0; c < chout; c++) {
            float* dout_row = dout_ch + c * size_channel_out;
            const float* din_ch = din_batch + c * size_channel_in;
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
                int pool_size = (hend - hstart) * (wend - wstart);
                if (pool_size == 0) continue;
                float sum = 0.f;
                for (int h = hstart; h < hend; ++h) {
                  for (int w = wstart; w < wend; ++w) {
                    sum += din_ch[h * win + w];
                  }
                }
                dout_row[j] = sum / (kernel_w * kernel_h);
              }
              dout_row += wout;
            }
          }
        }
      }
    } else {
      LOG(FATAL) << "unsupported pooling type: " << pooling_type;
    }
  }
}

void pooling_global_max(const float* din, float* dout, int num, int chout,
                        int hout, int wout, int chin, int hin, int win) {
  int size_channel_in = win * hin;
  int cnt = size_channel_in / 8;
  for (int n = 0; n < num; ++n) {
    float* dout_batch = dout + n * chout;
    const float* din_batch = din + n * chin * size_channel_in;
#pragma omp parallel for
    for (int c = 0; c < chout; ++c) {
      const float* din_ch = din_batch + c * size_channel_in;
      int i = 0;
      float minval = std::numeric_limits<float>::lowest();
      float32x4_t vmax = vdupq_n_f32(minval);
#ifdef __aarch64__
      for (; i < cnt; i++) {
        float32x4_t vdin1 = vld1q_f32(din_ch);
        vmax = vmaxq_f32(vdin1, vmax);
        float32x4_t vdin2 = vld1q_f32(din_ch + 4);
        vmax = vmaxq_f32(vmax, vdin2);
        din_ch += 8;
      }
#else
      int cnt_num = cnt;
      if (cnt_num > 0) {
        asm volatile(
            "max_loop:                          @main loop\n"
            "vld1.f32   {d0-d1}, [%[din_ch]]!   @load q1,din_ch\n"
            "vmax.f32   %q[vmax], %q[vmax], q0  @max vmax,vmax,din_ch\n"
            "vld1.f32   {d2-d3}, [%[din_ch]]!   @load 2nd 4 data\n"
            "vmax.f32   %q[vmax], %q[vmax], q1  @compare 2nd 4 datas\n"
            "subs       %[cnt_num], #1          @cnt_num--\n"
            "bne        max_loop                @bne cnt_num\n"
            : [din_ch] "+r"(din_ch), [cnt_num] "+r"(cnt_num), [vmax] "+w"(vmax)
            :
            : "cc", "memory", "q0", "q1");
      }
#endif  // __aarch64__
      float32x2_t vmax_tmp = vmax_f32(vget_low_f32(vmax), vget_high_f32(vmax));
      float tmp1 = vget_lane_f32(vmax_tmp, 0);
      float tmp2 = vget_lane_f32(vmax_tmp, 1);
      float max_tmp = tmp1 > tmp2 ? tmp1 : tmp2;
      for (i = cnt * 8; i < size_channel_in; ++i) {
        /* code */
        max_tmp = max_tmp > din_ch[0] ? max_tmp : din_ch[0];
        din_ch++;
      }
      dout_batch[c] = max_tmp;
    }
  }
}

void pooling_global_avg(const float* din, float* dout, int num, int chout,
                        int hout, int wout, int chin, int hin, int win) {
  int size_channel_in = win * hin;
  int cnt = size_channel_in / 4;
  for (int n = 0; n < num; ++n) {
    float* dout_batch = dout + n * chout;
    const float* din_batch = din + n * chin * size_channel_in;
#pragma omp parallel for
    for (int c = 0; c < chout; c++) {
      const float* din_ch = din_batch + c * size_channel_in;  // in address
      int i = 0;
      float32x4_t vsum = vdupq_n_f32(0.0f);
#ifdef __aarch64__
      for (; i < cnt; i++) {
        vsum = vaddq_f32(vld1q_f32(din_ch), vsum);
        din_ch += 4;
      }
#else
      int cnt_num = cnt;
      if (cnt_num > 0) {
        asm volatile(
            "add_loop:                          @main loop\n"
            "vld1.f32   {d0-d1}, [%[din_ch]]!   @load q1,din_ch\n"
            "vadd.f32   %q[vsum], %q[vsum], q0  @add vmax,vmax, din_ch\n"
            "subs       %[cnt_num], #1          @cnt_num--\n"
            "bne        add_loop                @bne num\n"
            : [din_ch] "+r"(din_ch), [cnt_num] "+r"(cnt_num), [vsum] "+w"(vsum)
            :
            : "cc", "memory", "q0");
      }
#endif  // __aarch64__
      float32x2_t vsum_tmp = vadd_f32(vget_low_f32(vsum), vget_high_f32(vsum));
      float sum = vget_lane_f32(vsum_tmp, 0) + vget_lane_f32(vsum_tmp, 1);
      for (i = cnt * 4; i < size_channel_in; i++) {
        sum += din_ch[0];
        din_ch++;
      }
      dout_batch[c] = sum / size_channel_in;
    }
  }
}

void pooling2x2s2_max(const float* din, float* dout, int num, int chout,
                      int hout, int wout, int chin, int hin, int win) {
  int kernel = 2;
  int stride = 2;
  int padding = 0;
  int size_channel_out = wout * hout;
  int size_channel_in = win * hin;

  int w_needed = (wout << 1);
  int h_needed = (hout << 1);
  int w_limit = w_needed > win ? win : w_needed;
  int h_limit = h_needed > hin ? hin : h_needed;
  int w_even = (w_limit >> 1) << 1;
  int h_even = (h_limit >> 1) << 1;
  int w_unroll_size = (w_even >> 3) << 3;
  // int w_unroll_remain = w_even - w_unroll_size;
  int w_in_2 = win << 1;
  for (int n = 0; n < num; ++n) {
    float* dout_batch = dout + n * chout * size_channel_out;
    const float* din_batch = din + n * chin * size_channel_in;
#pragma omp parallel for
    for (int c = 0; c < chout; c++) {
      float* dout_ch = dout_batch + c * size_channel_out;
      const float* din_ch = din_batch + c * size_channel_in;
      const float* r0 = din_ch;
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
          vst1q_f32(&dout_ch[w >> 1], dmax);
        }
#else
        float* dr_out = dout_ch;
        const float* dr0 = r0;
        const float* dr1 = r1;
        int cnt_num = w_unroll_size >> 3;
        if (cnt_num > 0) {
          asm volatile(
              "s2_max_loop:                      @main loop\n"
              "vld1.f32   {d0-d3}, [%[dr0]]!     @load q0,dr0\n"
              "vld1.f32   {d4-d7}, [%[dr1]]!     @load q1,dr1\n"
              "vmax.f32   q0, q0, q2             @max q0,q0,q2\n"
              "vmax.f32   q1, q1, q3             @max q1,q1,q2\n"
              "vpmax.f32  d4, d0, d1             @max d4,d0,d1\n"
              "vpmax.f32  d5, d2, d3             @max d5,d2,d3\n"
              "vst1.f32   {d4-d5}, [%[dr_out]]!  @vst1 q2,dr_out\n"
              "subs       %[cnt_num], #1         @cnt_num--\n"
              "bne        s2_max_loop            @bne cnt_num\n"
              : [dr0] "+r"(dr0), [dr1] "+r"(dr1), [dr_out] "+r"(dr_out),
                [cnt_num] "+r"(cnt_num)
              :
              : "cc", "memory", "q0", "q1", "q2", "q3");
        }
        w = w_unroll_size;
#endif  // __aarch64__
        for (; w < w_even; w += 2) {
          dout_ch[w >> 1] =
              std::max(std::max(r0[w], r0[w + 1]), std::max(r1[w], r1[w + 1]));
        }
        for (; w < w_limit; ++w) {  // run 0 or 1 time
          dout_ch[w >> 1] = std::max(r0[w], r1[w]);
        }
        r0 += w_in_2;  // << 1;
        r1 += w_in_2;  // << 1;
        dout_ch += wout;
      }
      // process remain row (odd, last row)
      for (; h < h_limit; h++) {  // run 0 or 1 time
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
          vst1q_f32(&dout_ch[w >> 1], dmax);
        }
#else
        float* dr_out = dout_ch;
        const float* dr0 = r0;
        int cnt_num = w_unroll_size >> 3;
        if (cnt_num > 0) {
          asm volatile(
              "s2_max_loop1:                      @main loop\n"
              "vld1.f32   {d0-d3}, [%[dr0]]!      @load q0,dr0\n"
              "vpmax.f32  d4, d0, d1              @max d4,d0,d1\n"
              "vpmax.f32  d5, d2, d3              @max d5,d2,d3\n"
              "vst1.f32   {d4-d5}, [%[dr_out]]!   @vst1 q2,dr_out\n"
              "subs       %[cnt_num], #1          @cnt_num--\n"
              "bne        s2_max_loop1            @bne cnt_num\n"
              : [dr0] "+r"(dr0), [dr_out] "+r"(dr_out), [cnt_num] "+r"(cnt_num)
              :
              : "cc", "memory", "q0", "q1", "q2");
        }
        w = w_unroll_size;
#endif  // __aarch64__
        for (; w < w_even; w += 2) {
          dout_ch[w >> 1] = std::max(r0[w], r0[w + 1]);
        }
        for (; w < w_limit; ++w) {  // run 0 or 1 time
          dout_ch[w >> 1] = r0[w];
        }
      }
    }
  }
}

void pooling2x2s2_avg(const float* din, float* dout, int num, int chout,
                      int hout, int wout, int chin, int hin, int win,
                      bool exclusive) {
  int kernel = 2;
  int stride = 2;
  int padding = 0;
  int size_channel_out = wout * hout;
  int size_channel_in = win * hin;

  int w_needed = (wout << 1);
  int h_needed = (hout << 1);
  int w_limit = w_needed > win ? win : w_needed;
  int h_limit = h_needed > hin ? hin : h_needed;
  int w_even = (w_limit >> 1) << 1;
  int h_even = (h_limit >> 1) << 1;
  int w_unroll_size = (w_even >> 3) << 3;
  // int w_unroll_remain = w_even - w_unroll_size;
  int w_in_2 = win << 1;
  const float coef = 1.f / 4.f;
  const float coef_1 = exclusive ? 1.f : coef;
  const float coef_2 = exclusive ? 1.f / 2.f : coef;
  float32x4_t vcoef = vdupq_n_f32(coef);
  float32x4_t vcoef_1 = vdupq_n_f32(coef_1);
  float32x4_t vcoef_2 = vdupq_n_f32(coef_2);
  for (int n = 0; n < num; ++n) {
    float* dout_batch = dout + n * chout * size_channel_out;
    const float* din_batch = din + n * chin * size_channel_in;
#pragma omp parallel for
    for (int c = 0; c < chout; c++) {
      float* dout_ch = dout_batch + c * size_channel_out;
      const float* din_ch = din_batch + c * size_channel_in;
      const float* r0 = din_ch;
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
          vst1q_f32(&dout_ch[w >> 1], res);
        }
#else
        float* dr_out = dout_ch;
        const float* dr0 = r0;
        const float* dr1 = r1;
        int cnt_num = w_unroll_size >> 3;
        if (cnt_num > 0) {
          asm volatile(
              "1:                                @main loop\n"
              "vld1.f32   {d0-d3}, [%[dr0]]!     @load q0,dr0\n"
              "vld1.f32   {d4-d7}, [%[dr1]]!     @load q1,dr1\n"
              "vadd.f32   q0, q0, q2             @add q0,q0,q2\n"
              "vadd.f32   q1, q1, q3             @add q1,q1,q2\n"
              "vpadd.f32  d4, d0, d1             @add d4,d0,d1\n"
              "vpadd.f32  d5, d2, d3             @add d5,d2,d3\n"
              "vmul.f32   q2, q2, %q[vcoef]    @mul q2,q2,vcoef\n"
              "vst1.f32   {d4-d5}, [%[dr_out]]!  @vst1 q2,dr_out\n"
              "subs       %[cnt_num], #1         @cnt_num--\n"
              "bne        1b                     @bne cnt_num\n"
              : [dr0] "+r"(dr0), [dr1] "+r"(dr1), [dr_out] "+r"(dr_out),
                [vcoef] "+w"(vcoef), [cnt_num] "+r"(cnt_num)
              : "r"(dr0), "r"(dr1), "r"(dr_out), "r"(cnt_num), "w"(vcoef)
              : "cc", "memory", "q0", "q1", "q2", "q3");
        }
        w = w_unroll_size;
#endif  // __aarch64__
        for (; w < w_even; w += 2) {
          dout_ch[w >> 1] = (r0[w] + r0[w + 1] + r1[w] + r1[w + 1]) * coef;
        }
        for (; w < w_limit; ++w) {  // run 0 or 1 time
          dout_ch[w >> 1] = (r0[w] + r1[w]) * coef_2;
        }
        r0 += w_in_2;  // << 1;
        r1 += w_in_2;  // << 1;
        dout_ch += wout;
      }
      // process remain row (odd, last row)
      for (; h < h_limit; h++) {  // run 0 or 1 time
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
          float32x4_t res = vmulq_f32(dsum, vcoef_2);
          vst1q_f32(&dout_ch[w >> 1], res);
        }
#else
        float* dr_out = dout_ch;
        const float* dr0 = r0;
        int cnt_num = w_unroll_size >> 3;
        if (cnt_num > 0) {
          asm volatile(
              "1:                                @main loop\n"
              "vld1.f32   {d0-d3}, [%[dr0]]!     @load q0,dr0\n"
              "vpadd.f32  d4, d0, d1             @add d4,d0,d1\n"
              "vpadd.f32  d5, d2, d3             @add d5,d2,d3\n"
              "vmul.f32   q2, q2, %q[vcoef_2]    @mul q2,q2,vcoef_2\n"
              "vst1.f32   {d4-d5}, [%[dr_out]]!  @vst1 q2,dr_out\n"
              "subs       %[cnt_num], #1         @cnt_num--\n"
              "bne        1b                     @bne cnt_num\n"
              : [dr0] "+r"(dr0), [dr_out] "+r"(dr_out), [vcoef_2] "+w"(vcoef_2),
                [cnt_num] "+r"(cnt_num)
              : "r"(dr0), "r"(dr_out), "r"(cnt_num), "w"(vcoef_2)
              : "cc", "memory", "q0", "q1", "q2");
        }
        w = w_unroll_size;
#endif  // __aarch64__
        for (; w < w_even; w += 2) {
          dout_ch[w >> 1] = (r0[w] + r0[w + 1]) * coef_2;
        }
        for (; w < w_limit; ++w) {  // run 0 or 1 time
          dout_ch[w >> 1] = r0[w] * coef_1;
        }
      }
    }
  }
}

void pooling3x3s1p1_max(const float* din, float* dout, int num, int chout,
                        int hout, int wout, int chin, int hin, int win) {
  int kernel = 3;
  int stride = 1;
  int padding = 1;
  int size_channel_out = wout * hout;
  int size_channel_in = win * hin;

  int w_unroll_size = ((win - 2) >> 2) << 2;
  int w_unroll_remain = win - 2 - w_unroll_size;
  const float minval = std::numeric_limits<float>::lowest();
  for (int n = 0; n < num; ++n) {
    float* dout_batch = dout + n * chout * size_channel_out;
    const float* din_batch = din + n * chin * size_channel_in;
#pragma omp parallel for
    for (int c = 0; c < chout; c++) {
      float* dout_ch = dout_batch + c * size_channel_out;
      const float* din_ch = din_batch + c * size_channel_in;
      const float* r0 = din_ch;
      const float* r1 = r0 + win;
      const float* r2 = r1 + win;
      int cnt_num = w_unroll_size >> 2;  // w_unroll_size / 4
      float* dr_out = dout_ch;
      const float* dr0 = r0;
      const float* dr1 = r1;
      const float* dr2 = r2;
      int w = 0;
      int cnt = 1;
      // left
      dout_ch[0] = std::max(std::max(r0[0], r0[1]), std::max(r1[0], r1[1]));
// first row with zero pad
#ifdef __aarch64__
      for (; w < w_unroll_size; w += 4) {
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
        vst1q_f32(&dout_ch[cnt], vmax);
        cnt += 4;
      }

#else
      dr_out = dr_out + 1;
      if (cnt_num > 0) {
        asm volatile(
            "1:                              @main loop\n"
            "vld1.f32  {d0-d1}, [%[dr0]]!    @load d0-d5,dr0\n"
            "vld1.f32  {d4-d5}, [%[dr1]]!    @load d4-d7,dr1\n"
            "vld1.f32  {d2}, [%[dr0]]!       @load d0-d5,dr0\n"
            "vld1.f32  {d6}, [%[dr1]]!       @load d4-d7,dr1\n"
            "vmax.f32  q5, q0, q2            @max r0_1234,r1_1234\n"
            "vmax.f32  d12, d2, d6           @max r0_5678,r1_5678\n"
            //"vmov.f32  s7,s6               @mov s7,s6\n"
            "vext.f32  q0, q5, q6, #1        @vext max_2345\n"
            "vext.f32  q2, q5, q6, #2        @vext max_3456\n"
            "vpmax.f32 d2, d10, d11          @pmax d4,max_1234,max_1234\n"
            "vpmax.f32 d3, d0, d1            @pmax d4,max_2345,max_2345\n"
            "vpmax.f32 d6, d4, d5            @pmax d6,max_3456,max_3456\n"
            "vmax.f32  d8, d2, d3            @max d2,vmax_12_34,vmax_23_45\n"
            "vmax.f32  d9, d3, d6            @max d2,vmax_23_45,vmax_34_56\n"
            "sub       %[dr0], #8            @sub w,8\n"
            "sub       %[dr1], #8            @sub w,8\n"
            // swap
            "vmov.f32  s0, s17               @mov\n"
            "vmov.f32  s17, s18              @mov\n"
            "vmov.f32  s18, s0               @mov\n"
            "subs      %[cnt_num], #1        @subs cnt_num,#1\n"
            "vst1.f32  d8, [%[dr_out]]!      @vst1 d0,dr_out\n"
            "vst1.f32  d9, [%[dr_out]]!      @vst1 d0,dr_out\n"
            "bne       1b                    @bne s1_max_loop\n"
            : [dr0] "+r"(dr0), [dr1] "+r"(dr1), [dr_out] "+r"(dr_out),
              [cnt_num] "+r"(cnt_num)
            : "r"(dr0), "r"(dr1), "r"(dr_out), "r"(cnt_num)
            : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6");
      }

#endif
      // remain
      w = w_unroll_size;
      for (int j = 0; j < w_unroll_remain; j++) {
        float tmp_max = std::max(r0[j + w], r1[j + w]);
        tmp_max = std::max(tmp_max, std::max(r0[j + w + 1], r1[j + w + 1]));
        tmp_max = std::max(tmp_max, std::max(r0[j + w + 2], r1[j + w + 2]));
        dout_ch[j + w + 1] = tmp_max;
      }
      // right
      float tmp = std::max(r0[win - 2], r1[win - 2]);
      tmp = std::max(tmp, std::max(r0[win - 1], r1[win - 1]));
      dout_ch[wout - 1] = tmp;

      // r0 = r1;
      // r1 = r0 + w_in;
      // r2 = r1 + w_in;
      dout_ch += wout;
      int h = 0;
      for (; h < hin - 2; h += 1) {
        // deal with left pad
        float maxr0 = std::max(r0[0], r0[1]);
        float maxr1 = std::max(r1[0], r1[1]);
        float maxr2 = std::max(r2[0], r2[1]);
        dout_ch[0] = std::max(std::max(maxr0, maxr1), maxr2);
#ifdef __aarch64__
        w = 0;
        cnt = 1;
        for (; w < w_unroll_size; w += 4) {
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
          vst1q_f32(&dout_ch[cnt], vmax);
          cnt += 4;
        }
#else
        dr_out = dout_ch + 1;
        dr0 = r0;
        dr1 = r1;
        dr2 = r2;
        cnt_num = w_unroll_size >> 2;
        if (cnt_num > 0) {
          asm volatile(
              "1:                             @main loop\n"
              "vld1.f32  {d0-d1}, [%[dr0]]!   @load d0-d5,dr0\n"
              "vld1.f32  {d4-d5}, [%[dr1]]!   @load d4-d7,dr1\n"
              "vld1.f32  {d8-d9}, [%[dr2]]!   @load d4-d7,dr1\n"
              "vld1.f32  {d2}, [%[dr0]]!      @load d0-d5,dr0\n"
              "vld1.f32  {d6}, [%[dr1]]!      @load d4-d7,dr1\n"
              "vld1.f32  {d10}, [%[dr2]]!     @load d4-d7, dr1\n"
              "vmax.f32  q7, q0, q2           @max r0_1234,r1_1234\n"
              "vmax.f32  d16, d2, d6          @max r0_5678,r1_5678\n"
              "vmax.f32  q3, q7, q4           @max r0_1234,r1_1234\n"
              "vmax.f32  d12, d16, d10        @max r0_5678,r1_5678\n"
              //"vmov.f32  s7,s6              @mov s7,s6\n"
              "vext.f32  q0, q3, q6, #1       @vext max_2345\n"
              "vext.f32  q2, q3, q6, #2       @vext max_3456\n"
              "vpmax.f32 d2, d6, d7           @pmax d4,max_1234,max_1234\n"
              "vpmax.f32 d3, d0, d1           @pmax d4,max_2345,max_2345\n"
              "vpmax.f32 d6, d4, d5           @pmax d6,max_3456,max_3456\n"
              "vmax.f32  d8, d2, d3           @max d2,vmax_12_34,vmax_23_45\n"
              "vmax.f32  d9, d3, d6           @max d2,vmax_23_45,vmax_34_56\n"
              "sub       %[dr0], #8           @sub w,8\n"
              "sub       %[dr1], #8           @sub w,8\n"
              "sub       %[dr2], #8           @sub w,8\n"
              // swap
              "vmov.f32  s0, s17              @mov\n"
              "vmov.f32  s17, s18             @mov\n"
              "vmov.f32  s18, s0              @mov\n"
              "subs      %[cnt_num], #1       @subs cnt_num,#1\n"
              "vst1.f32  d8, [%[dr_out]]!     @vst1 d0,dr_out\n"
              "vst1.f32  d9, [%[dr_out]]!     @vst1 d0,dr_out\n"
              "bne       1b                   @bne s1_max_loop\n"
              : [dr0] "+r"(dr0), [dr1] "+r"(dr1), [dr2] "+r"(dr2),
                [dr_out] "+r"(dr_out), [cnt_num] "+r"(cnt_num)
              : "r"(dr0), "r"(dr1), "r"(dr_out), "r"(cnt_num)
              : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7",
                "q8");
        }
#endif
        // remain
        w = w_unroll_size;
        for (int j = 0; j < w_unroll_remain; j++) {
          float tmp_max = std::max(r0[j + w], r1[j + w]);
          tmp_max = std::max(tmp_max, std::max(r0[j + w + 1], r1[j + w + 1]));
          tmp_max = std::max(tmp_max, std::max(r0[j + w + 2], r1[j + w + 2]));
          tmp_max = std::max(tmp_max, std::max(r2[j + w], r2[j + w + 1]));
          tmp_max = std::max(tmp_max, r2[j + w + 2]);
          dout_ch[j + w + 1] = tmp_max;
        }
        // right
        tmp = std::max(r0[win - 2], r1[win - 2]);
        tmp = std::max(tmp, std::max(r0[win - 1], r1[win - 1]));
        tmp = std::max(tmp, std::max(r2[win - 2], r2[win - 1]));
        dout_ch[wout - 1] = tmp;

        r0 = r1;
        r1 = r2;
        r2 = r1 + win;
        dout_ch += wout;
      }

      // the last two line
      float maxr0 = std::max(r0[0], r0[1]);
      float maxr1 = std::max(r1[0], r1[1]);
      dout_ch[0] = std::max(maxr0, maxr1);
#ifdef __aarch64__
      w = 0;
      cnt = 1;
      for (; w < w_unroll_size; w += 4) {
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
        vst1q_f32(&dout_ch[cnt], vmax);
        cnt += 4;
      }
#else
      dr_out = dout_ch + 1;
      dr0 = r0;
      dr1 = r1;
      cnt_num = w_unroll_size >> 2;
      if (cnt_num > 0) {
        asm volatile(
            "1:                              @main loop\n"
            "vld1.f32  {d0-d1}, [%[dr0]]!    @load d0-d5,dr0\n"
            "vld1.f32  {d4-d5}, [%[dr1]]!    @load d4-d7,dr1\n"
            "vld1.f32  {d2}, [%[dr0]]!       @load d0-d5,dr0\n"
            "vld1.f32  {d6}, [%[dr1]]!       @load d4-d7,dr1\n"
            "vmax.f32  q5, q0, q2            @max r0_1234,r1_1234\n"
            "vmax.f32  d12, d2, d6           @max r0_5678,r1_5678\n"
            //"vmov.f32  s7,s6               @mov s7,s6\n"
            "vext.f32  q0, q5, q6, #1        @vext max_2345\n"
            "vext.f32  q2, q5, q6, #2        @vext max_3456\n"
            "vpmax.f32 d2, d10, d11          @pmax d4,max_1234,max_1234\n"
            "vpmax.f32 d3, d0, d1            @pmax d4,max_2345,max_2345\n"
            "vpmax.f32 d6, d4, d5            @pmax d6,max_3456,max_3456\n"
            "vmax.f32  d8, d2, d3            @max d2,vmax_12_34,vmax_23_45\n"
            "vmax.f32  d9, d3, d6            @max d2,vmax_23_45,vmax_34_56\n"
            "sub       %[dr0], #8            @sub w,8\n"
            "sub       %[dr1], #8            @sub w,8\n"
            // swap
            "vmov.f32  s0, s17               @mov\n"
            "vmov.f32  s17, s18              @mov\n"
            "vmov.f32  s18, s0               @mov\n"
            "subs      %[cnt_num], #1        @subs cnt_num,#1\n"
            "vst1.f32  d8, [%[dr_out]]!      @vst1 d0,dr_out\n"
            "vst1.f32  d9, [%[dr_out]]!      @vst1 d0,dr_out\n"
            "bne       1b                    @bne s1_max_loop\n"
            : [dr0] "+r"(dr0), [dr1] "+r"(dr1), [dr_out] "+r"(dr_out),
              [cnt_num] "+r"(cnt_num)
            : "r"(dr0), "r"(dr1), "r"(dr_out), "r"(cnt_num)
            : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6");
      }
#endif
      // remian
      w = w_unroll_size;
      for (int j = 0; j < w_unroll_remain; j++) {
        float tmp_max = std::max(r0[j + w], r1[j + w]);
        tmp_max = std::max(tmp_max, std::max(r0[j + w + 1], r1[j + w + 1]));
        tmp_max = std::max(tmp_max, std::max(r0[j + w + 2], r1[j + w + 2]));
        dout_ch[j + w + 1] = tmp_max;
      }
      tmp = std::max(r0[win - 2], r1[win - 2]);
      tmp = std::max(tmp, std::max(r0[win - 1], r1[win - 1]));
      dout_ch[wout - 1] = tmp;
    }
  }
}

void pooling3x3s1p1_avg(const float* din, float* dout, int num, int chout,
                        int hout, int wout, int chin, int hin, int win,
                        bool exclusive) {
  int kernel = 3;
  int stride = 1;
  int padding = 1;
  int size_channel_out = wout * hout;
  int size_channel_in = win * hin;

  int w_unroll_size = ((win - 2) >> 2) << 2;
  int w_unroll_remain = win - 2 - w_unroll_size;
  const float coef = 1.f / 9.f;
  const float coef_2 = exclusive ? 1.f / 2.f : coef;
  const float coef_4 = exclusive ? 1.f / 4.f : coef;
  const float coef_6 = exclusive ? 1.f / 6.f : coef;
  float32x4_t vcoef = vdupq_n_f32(coef);
  float32x4_t vcoef_2 = vdupq_n_f32(coef_2);
  float32x4_t vcoef_4 = vdupq_n_f32(coef_4);
  float32x4_t vcoef_6 = vdupq_n_f32(coef_6);
  for (int n = 0; n < num; ++n) {
    float* dout_batch = dout + n * chout * size_channel_out;
    const float* din_batch = din + n * chin * size_channel_in;
#pragma omp parallel for
    for (int c = 0; c < chout; c++) {
      float* dout_ch = dout_batch + c * size_channel_out;
      const float* din_ch = din_batch + c * size_channel_in;
      const float* r0 = din_ch;
      const float* r1 = r0 + win;
      const float* r2 = r1 + win;
      int cnt_num = w_unroll_size >> 2;  // w_unroll_size / 4
      float* dr_out = dout_ch;
      const float* dr0 = r0;
      const float* dr1 = r1;
      const float* dr2 = r2;
      int w = 0;
      int cnt = 1;
      // left
      dout_ch[0] = (r0[0] + r0[1] + r1[0] + r1[1]) * coef_4;
// first row with zero pad
#ifdef __aarch64__
      for (; w < w_unroll_size; w += 4) {
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
        vsum = vmulq_f32(vsum, vcoef_6);
        vst1q_f32(&dout_ch[cnt], vsum);
        cnt += 4;
      }
#else
      dr_out = dr_out + 1;
      if (cnt_num > 0) {
        asm volatile(
            "1:                               @main loop\n"
            "vld1.f32  {d0-d1}, [%[dr0]]!     @load d0-d5,dr0\n"
            "vld1.f32  {d4-d5}, [%[dr1]]!     @load d4-d7,dr1\n"
            "vld1.f32  {d2}, [%[dr0]]!        @load d0-d5,dr0\n"
            "vld1.f32  {d6}, [%[dr1]]!        @load d4-d7,dr1\n"
            "vadd.f32  q5, q0, q2             @max r0_1234,r1_1234\n"
            "vadd.f32  d12, d2, d6            @max r0_5678,r1_5678\n"
            //"vmov.f32  s7,s6                @mov s7,s6\n"
            "vext.f32  q0, q5, q6, #1         @vext max_2345\n"
            "vext.f32  q2, q5, q6, #2         @vext max_3456\n"
            "vadd.f32  q1, q5, q0             @add 1234+2345\n"
            "vadd.f32  q1, q1, q2             @add + 3456\n"
            "vmul.f32  q4, q1, %q[vcoef_6]    @mul * 1/9.f\n"
            "sub       %[dr0], #8             @sub w,8\n"
            "sub       %[dr1], #8             @sub w,8\n"
            "subs      %[cnt_num], #1         @subs cnt_num,#1\n"
            "vst1.f32  d8, [%[dr_out]]!       @vst1 d0,dr_out\n"
            "vst1.f32  d9, [%[dr_out]]!       @vst1 d0,dr_out\n"
            "bne       1b                     @bne s1_max_loop\n"
            : [dr0] "+r"(dr0), [dr1] "+r"(dr1), [dr_out] "+r"(dr_out),
              [cnt_num] "+r"(cnt_num), [vcoef_6] "+w"(vcoef_6)
            : "r"(dr0), "r"(dr1), "r"(dr_out), "r"(cnt_num)
            : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6");
      }

#endif
      // remain
      w = w_unroll_size;
      for (int j = 0; j < w_unroll_remain; j++) {
        float tmp_sum = r0[j + w] + r1[j + w];
        tmp_sum += (r0[j + w + 1] + r1[j + w + 1]);
        tmp_sum += (r0[j + w + 2] + r1[j + w + 2]);
        dout_ch[j + w + 1] = tmp_sum * coef_6;
      }
      // right
      float tmp = r0[win - 2] + r1[win - 2];
      tmp += (r0[win - 1] + r1[win - 1]);
      dout_ch[wout - 1] = tmp * coef_4;

      // r0 = r1;
      // r1 = r0 + w_in;
      // r2 = r1 + w_in;
      dout_ch += wout;
      int h = 0;
      for (; h < hin - 2; h += 1) {
        // deal with left pad
        float maxr0 = r0[0] + r0[1];
        float maxr1 = r1[0] + r1[1];
        float maxr2 = r2[0] + r2[1];
        dout_ch[0] = (maxr0 + maxr1 + maxr2) * coef_6;
#ifdef __aarch64__
        w = 0;
        cnt = 1;
        for (; w < w_unroll_size; w += 4) {
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
          vst1q_f32(&dout_ch[cnt], vsum);
          cnt += 4;
        }
#else
        dr_out = dout_ch + 1;
        dr0 = r0;
        dr1 = r1;
        dr2 = r2;
        cnt_num = w_unroll_size >> 2;
        if (cnt_num > 0) {
          asm volatile(
              "1:                            @main loop\n"
              "vld1.f32  {d0-d1}, [%[dr0]]!  @load d0-d5,dr0\n"
              "vld1.f32  {d4-d5}, [%[dr1]]!  @load d4-d7,dr1\n"
              "vld1.f32  {d8-d9}, [%[dr2]]!  @load d4-d7,dr1\n"
              "vld1.f32  {d2}, [%[dr0]]!     @load d0-d5,dr0\n"
              "vld1.f32  {d6}, [%[dr1]]!     @load d4-d7,dr1\n"
              "vld1.f32  {d10}, [%[dr2]]!    @load d4-d7,dr1\n"
              "vadd.f32  q7, q0, q2          @max r0_1234,r1_1234\n"
              "vadd.f32  d16, d2, d6         @max r0_5678,r1_5678\n"
              "vadd.f32  q3, q7, q4          @max r0_1234,r1_1234\n"
              "vadd.f32  d12, d16, d10       @max r0_5678,r1_5678\n"
              //"vmov.f32  s7,s6             @mov s7,s6\n"
              "vext.f32  q0, q3, q6, #1      @vext max_2345\n"
              "vext.f32  q2, q3, q6, #2      @vext max_3456\n"
              "vadd.f32  q1, q3, q0          @add 1234+2345\n"
              "vadd.f32  q1, q1, q2          @add+3456\n"
              "vmul.f32  q4, q1, %q[vcoef]   @mul*1/9.f\n"
              "sub       %[dr0], #8          @sub w,8\n"
              "sub       %[dr1], #8          @sub w,8\n"
              "sub       %[dr2], #8          @sub w,8\n"
              "subs      %[cnt_num], #1      @subs cnt_num,#1\n"
              "vst1.f32  d8, [%[dr_out]]!    @vst1 d0,dr_out\n"
              "vst1.f32  d9, [%[dr_out]]!    @vst1 d0,dr_out\n"
              "bne       1b                  @bne s1_max_loop\n"
              : [dr0] "+r"(dr0), [dr1] "+r"(dr1), [dr2] "+r"(dr2),
                [dr_out] "+r"(dr_out), [cnt_num] "+r"(cnt_num),
                [vcoef] "+w"(vcoef)
              : "r"(dr0), "r"(dr1), "r"(dr_out), "r"(cnt_num)
              : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7",
                "q8");
        }
#endif
        // remain
        w = w_unroll_size;
        for (int j = 0; j < w_unroll_remain; j++) {
          float tmp_sum = r0[j + w] + r1[j + w];
          tmp_sum += (r0[j + w + 1] + r1[j + w + 1]);
          tmp_sum += (r0[j + w + 2] + r1[j + w + 2]);
          tmp_sum += (r2[j + w + 1] + r2[j + w + 2]);
          tmp_sum += r2[j + w];
          dout_ch[j + w + 1] = tmp_sum * coef;
        }
        // right
        tmp = r0[win - 2] + r1[win - 2];
        tmp += (r0[win - 1] + r1[win - 1]);
        tmp += (r2[win - 2] + r2[win - 1]);
        dout_ch[wout - 1] = tmp * coef_6;

        r0 = r1;
        r1 = r2;
        r2 = r1 + win;
        dout_ch += wout;
      }

      // last line
      float maxr0 = (r0[0] + r0[1]);
      float maxr1 = (r1[0] + r1[1]);
      dout_ch[0] = (maxr0 + maxr1) * coef_4;
#ifdef __aarch64__
      w = 0;
      cnt = 1;
      for (; w < w_unroll_size; w += 4) {
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
        vsum = vmulq_f32(vsum, vcoef_6);
        vst1q_f32(&dout_ch[cnt], vsum);
        cnt += 4;
      }
#else
      dr_out = dout_ch + 1;
      dr0 = r0;
      dr1 = r1;
      cnt_num = w_unroll_size >> 2;
      if (cnt_num > 0) {
        asm volatile(
            "1:                              @main loop\n"
            "vld1.f32  {d0-d1}, [%[dr0]]!    @load d0-d5,dr0\n"
            "vld1.f32  {d4-d5}, [%[dr1]]!    @load d4-d7,dr1\n"
            "vld1.f32  {d2}, [%[dr0]]!       @load d0-d5,dr0\n"
            "vld1.f32  {d6}, [%[dr1]]!       @load d4-d7,dr1\n"
            "vadd.f32  q5, q0, q2            @max r0_1234,r1_1234\n"
            "vadd.f32  d12, d2, d6           @max r0_5678,r1_5678\n"
            //"vmov.f32  s7,s6               @mov s7,s6\n"
            "vext.f32  q0, q5, q6, #1        @vext max_2345\n"
            "vext.f32  q2, q5, q6, #2        @vext max_3456\n"
            "vadd.f32  q1, q5, q0            @add 1234+2345\n"
            "vadd.f32  q1, q1, q2            @add + 3456\n"
            "vmul.f32  q4, q1, %q[vcoef_6]   @mul * 1/9.f\n"
            "sub       %[dr0], #8            @sub w,8\n"
            "sub       %[dr1], #8            @sub w,8\n"
            "subs      %[cnt_num], #1        @subs cnt_num,#1\n"
            "vst1.f32  d8, [%[dr_out]]!      @vst1 d0,dr_out\n"
            "vst1.f32  d9, [%[dr_out]]!      @vst1 d0,dr_out\n"
            "bne       1b                    @bne s1_max_loop\n"
            : [dr0] "+r"(dr0), [dr1] "+r"(dr1), [dr_out] "+r"(dr_out),
              [cnt_num] "+r"(cnt_num), [vcoef_6] "+w"(vcoef_6)
            : "r"(dr0), "r"(dr1), "r"(dr_out), "r"(cnt_num)
            : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6");
      }
#endif
      // remain
      w = w_unroll_size;
      for (int j = 0; j < w_unroll_remain; j++) {
        float tmp_sum = r0[j + w] + r1[j + w];
        tmp_sum += (r0[j + w + 1] + r1[j + w + 1]);
        tmp_sum += (r0[j + w + 2] + r1[j + w + 2]);
        dout_ch[j + w + 1] = tmp_sum * coef_6;
      }
      // right
      tmp = r0[win - 2] + r1[win - 2];
      tmp += (r0[win - 1] + r1[win - 1]);
      dout_ch[wout - 1] = tmp * coef_4;
    }
  }
}

void pooling3x3s2p1_max(const float* din, float* dout, int num, int chout,
                        int hout, int wout, int chin, int hin, int win) {
  int kernel = 3;
  int stride = 2;
  int padding = 1;
  int size_channel_out = wout * hout;
  int size_channel_in = win * hin;

  int w_needed = (wout << 1) + 1;
  int h_needed = (hout << 1) + 1;
  int w_limit = w_needed > win ? win : w_needed;
  int h_limit = h_needed > hin ? hin : h_needed;
  int w_even = (w_limit >> 1) << 1;
  int h_even = (h_limit >> 1) << 1;
  int w_unroll_size = ((w_even - 1) >> 3) << 3;
  int w_unroll_remain = w_even - 1 - w_unroll_size;
  int w_remain = w_needed - w_limit - padding;
  int h_remain = h_needed - h_limit - padding;
  int w_in_2 = win << 1;
  float minval = std::numeric_limits<float>::lowest();
  for (int n = 0; n < num; ++n) {
    float* dout_batch = dout + n * chout * size_channel_out;
    const float* din_batch = din + n * chin * size_channel_in;
#pragma omp parallel for
    for (int c = 0; c < chout; c++) {
      float* dout_ch = dout_batch + c * size_channel_out;
      const float* din_ch = din_batch + c * size_channel_in;
      const float* r0 = din_ch;
      const float* r1 = r0 + win;
      const float* r2 = r1 + win;
      int cnt_num = w_unroll_size >> 3;
      int cnt_num_remain = w_unroll_remain >> 1;
      float* dr_out = dout_ch;
      const float* dr0 = r0;
      const float* dr1 = r1;
      const float* dr2 = r2;
      int w = 1;
      int cnt = 1;
      dout_ch[0] = std::max(std::max(r0[0], r0[1]), std::max(r1[0], r1[1]));
// first row with zero pad
#if __aarch64__
      for (; w < w_unroll_size; w += 8) {
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
        vst1_f32(&dout_ch[cnt], vmax_123_345);
        vst1_f32(&dout_ch[cnt + 2], vmax_567_789);
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
        dout_ch[cnt] = vget_lane_f32(vmax2, 0);
        cnt++;
      }
#else
      dr0 = dr0 + 1;
      dr1 = dr1 + 1;
      dr_out = dr_out + 1;
      // LOG(INFO) << "cnt_num: " << cnt_num << " cnt_num_remain: " <<
      // cnt_num_remain;
      if (cnt_num > 0 || cnt_num_remain > 0) {
        asm volatile(
            "cmp       %[cnt_num], #0        @cmp cnt_num,0\n"
            "ble       3f                    @ble exit\n"
            "1:                              @main loop\n"
            "vld1.f32  {d0-d3}, [%[dr0]]!    @load d0-d5,dr0\n"
            "vld1.f32  {d6-d9}, [%[dr1]]!    @load d4-d7,dr1\n"
            "vld1.f32  {d4-d5}, [%[dr0]]!    @load d0-d5,dr0\n"
            "vld1.f32  {d10-d11}, [%[dr1]]!  @load d4-d7,dr1\n"
            "vmax.f32  q6, q0, q3            @max r0_1234,r1_1234\n"
            "vmax.f32  q7, q1, q4            @max r0_5678,r1_5678\n"
            "vmax.f32  q8, q2, q5            @max r0_9101112,r1_9101112\n"
            //"vmov.f32  s7,s6               @mov s7,s6\n"
            "vext.f32  q0, q6, q7, #1        @vext max_2345\n"
            "vext.f32  q1, q7, q8, #1        @vext max_6789\n"
            "vpmax.f32 d4, d12, d13          @pmax d4,vmax_1234,vmax_1234\n"
            "vpmax.f32 d6, d14, d15          @pmax d6,vmax_5678,vmax_5678\n"
            "vpmax.f32 d5, d0, d1            @pmax d5,vmax_2345,vmax_2345\n"
            "vpmax.f32 d7, d2, d3            @pmax d7,vmax_6789,vmax_6789\n"
            "vmax.f32 d8, d4, d5             @max d2,vmax_12_34,vmax_23_45\n"
            "vmax.f32 d9, d6, d7             @max d2,vmax_56_78,vmax_67_89\n"
            "sub       %[dr0], #16           @add w,8\n"
            "sub       %[dr1], #16           @add w, 8\n"
            "vst1.f32  d8, [%[dr_out]]!      @vst1 d0,dr_out\n"
            "vst1.f32  d9, [%[dr_out]]!      @vst1 d0,dr_out\n"
            "subs      %[cnt_num], #1        @subs cnt_num, #1\n"
            "bne       1b                    @bne s3_max_loop\n"
            "3:                              @loop \n"
            "cmp       %[cnt_num_remain], #0 @cmp cnt_num,0\n"
            "ble       4f                    @ble exit\n"
            "2:                              @main loop\n"
            "vld1.f32  {d0-d1}, [%[dr0]]!    @load d0-d1,dr0\n"
            "vld1.f32  {d2-d3}, [%[dr1]]!    @load d2-d3,dr1\n"
            "vmov.f32  s3,s2                 @movs3,s2\n"
            "vmov.f32  s7,s6                 @movs7,s6\n"
            "vmax.f32  q0, q0, q1            @max q0,q0,q1\n"
            "vpmax.f32 d0, d0, d1            @pmax d0,d0,d1\n"
            "vpmax.f32 d0, d0, d0            @pmax d0,d0,d0\n"
            "vst1.f32  d0[0], [%[dr_out]]!   @vst d0[0],dr_out\n"
            "sub       %[dr0], #8            @add w,6\n"
            "sub       %[dr1], #8            @add w,6\n"
            "subs      %[cnt_num_remain], #1 @subs cnt_num,#1\n"
            "bne       2b                    @bne s3_max_loop_1\n"
            "4:                              @exit\n"
            : [dr0] "+r"(dr0), [dr1] "+r"(dr1), [dr_out] "+r"(dr_out),
              [cnt_num] "+r"(cnt_num), [cnt_num_remain] "+r"(cnt_num_remain)
            : "r"(dr0), "r"(dr1), "r"(dr_out), "r"(cnt_num), "r"(cnt_num_remain)
            : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7",
              "q8", "q9");
      }
#endif
      // int w = w_even - 1;
      if (w_remain > 0) {
        // deal with right pad
        int wstart = (w_even >> 1) * stride - padding;
        int wend = std::min(std::min(wstart + kernel, win + padding), win);
        float tmp = r0[wstart];  // std::numeric_limits<float>::min();
        for (int i = wstart; i < wend; i++) {  // only run 1 or 2 times
          tmp = std::max(tmp, std::max(r0[i], r1[i]));
        }
        dout_ch[w_even >> 1] = tmp;
        // cnt ++;
      }

      r0 = r1;
      r1 = r0 + win;
      r2 = r1 + win;
      dout_ch += wout;
      int h = 2;
      for (; h < h_even; h += 2) {
        // deal with left pad
        float maxr0 = std::max(r0[0], r0[1]);
        float maxr1 = std::max(r1[0], r1[1]);
        float maxr2 = std::max(r2[0], r2[1]);
        dout_ch[0] = std::max(std::max(maxr0, maxr1), maxr2);
#if __aarch64__
        w = 1;
        cnt = 1;
        for (; w < w_unroll_size; w += 8) {
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
          vst1_f32(&dout_ch[cnt], vmax_123_345);
          vst1_f32(&dout_ch[cnt + 2], vmax_567_789);
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
          dout_ch[cnt] = vget_lane_f32(vmax, 0);
          cnt++;
        }
#else
        dr_out = dout_ch + 1;
        dr0 = (r0 + 1);
        dr1 = (r1 + 1);
        dr2 = (r2 + 1);
        cnt_num = w_unroll_size >> 3;
        cnt_num_remain = w_unroll_remain >> 1;
        if (cnt_num > 0 || cnt_num_remain > 0) {
          asm volatile(
              "cmp       %[cnt_num], #0        @cmp cnt_num,0\n"
              "ble       3f                    @ble exit\n"
              "1:                              @main loop\n"
              "vld1.f32  {d0-d3}, [%[dr0]]!    @load d0-d5,dr0\n"
              "vld1.f32  {d6-d9}, [%[dr1]]!    @load d4-d7,dr1\n"
              "vld1.f32  {d12-d15}, [%[dr2]]!  @load d4-d7,dr1\n"
              "vld1.f32  {d4-d5}, [%[dr0]]!    @load d0-d5,dr0\n"
              "vld1.f32  {d10-d11}, [%[dr1]]!  @load d4-d7,dr1\n"
              "vld1.f32  {d16-d17}, [%[dr2]]!  @load d4-d7,dr1\n"
              "vmax.f32  q9, q0, q3            @max q0,q0,q2\n"
              "vmax.f32  q10, q1, q4           @max q1,q1,q3\n"
              "vmax.f32  q11, q2, q5           @max q1,q1,q3\n"
              "vmax.f32  q0, q9, q6            @max q0,q0,q2 1234\n"
              "vmax.f32  q3, q10, q7           @max q1,q1,q3 5678\n"
              "vmax.f32  q1, q11, q8           @max q1,q1,q3 9101112\n"
              //"vmov.f32  s7,s6               @mov s7, s6\n"
              "vext.f32  q4, q0, q3, #1        @vext 2345\n"
              "vext.f32  q2, q3, q1, #1        @vext 6789\n"
              "vpmax.f32 d10, d0, d1           @pmax d10,vmax_1234,vmax_1234\n"
              "vpmax.f32 d12, d6, d7           @pmax d12,vmax_5678,vmax_5678\n"
              "vpmax.f32 d11, d8, d9           @pmax d11,vmax_2345,vmax_2345\n"
              "vpmax.f32 d13, d4, d5           @pmax d13,vmax_6789,vmax_6789\n"
              "vmax.f32 d0, d10, d11           @pmax d0,vmax_12_34,vmax_23_45\n"
              "vmax.f32 d1, d12, d13           @pmax d1,vmax_56_78,vmax_67_89\n"
              "sub       %[dr0], #16           @add w,8\n"
              "sub       %[dr1], #16           @add w,8\n"
              "sub       %[dr2], #16           @add w,8\n"
              "vst1.f32  d0, [%[dr_out]]!      @vst1 d0,dr_out\n"
              "vst1.f32  d1, [%[dr_out]]!      @vst1 d0,dr_out\n"
              "subs      %[cnt_num], #1        @subs cnt_num,#1\n"
              "bne       1b                    @bne s3_max_loop_mid\n"
              "3:                              @loop \n"
              "cmp       %[cnt_num_remain], #0 @cmp cnt_num,0\n"
              "ble       4f                    @ble exit1\n"
              "2:                              @mid loop\n"
              "vld1.f32  {d0-d1}, [%[dr0]]!    @load d0-d1,dr0\n"
              "vld1.f32  {d2-d3}, [%[dr1]]!    @load d2-d3,dr1\n"
              "vld1.f32  {d4-d5}, [%[dr2]]!    @load d2-d3,dr1\n"
              "vmov.f32  s3,s2                 @movs3,s2\n"
              "vmov.f32  s7,s6                 @movs7,s6\n"
              "vmov.f32  s11,s10               @movs11,s10\n"
              "vmax.f32  q0, q0, q1            @max q0,q0,q1\n"
              "vmax.f32  q0, q0, q2            @max q0,q0,q2\n"
              "vpmax.f32 d0, d0, d1            @pmax d0,d0,d1\n"
              "vpmax.f32 d0, d0, d0            @pmax d0, d0,d0\n"
              "vst1.f32  d0[0], [%[dr_out]]!   @vst d0[0],dr_out\n"
              "sub       %[dr0], #8            @add w,6\n"
              "sub       %[dr1], #8            @add w,6\n"
              "sub       %[dr2], #8            @add w,6\n"
              "subs      %[cnt_num_remain], #1 @subs cnt_num,#1\n"
              "bne       2b                    @bne s3_max_loop_mid_1\n"
              "4:                              @exit\n"
              : [dr0] "+r"(dr0), [dr1] "+r"(dr1), [dr2] "+r"(dr2),
                [dr_out] "+r"(dr_out), [cnt_num] "+r"(cnt_num),
                [cnt_num_remain] "+r"(cnt_num_remain)
              : "r"(dr0), "r"(dr1), "r"(dr2), "r"(dr_out), "r"(cnt_num),
                "r"(cnt_num_remain)
              : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7",
                "q8", "q9", "q10", "q11", "q12");
        }
#endif
        if (w_remain > 0) {
          // deal with right pad
          int wstart = (w_even >> 1) * stride - padding;
          int wend = std::min(std::min(wstart + kernel, win + padding), win);
          float tmp = r0[wstart];  // std::numeric_limits<float>::min();
          for (int i = wstart; i < wend; i++) {
            tmp = std::max(tmp, std::max(r0[i], r1[i]));
            tmp = std::max(tmp, r2[i]);
          }
          dout_ch[w_even >> 1] = tmp;
          // cnt ++;
        }
        r0 = r2;
        r1 = r0 + win;
        r2 = r1 + win;
        dout_ch += wout;
      }

      if (h_remain > 0) {
        // deal with bottom pad
        // first row with zero pad
        int hstart = (h >> 1) * stride - padding;
        int hend = std::min(std::min(hstart + kernel, hin + padding), hin);
        if (hstart == hend - 1) {  // only one lline
          dout_ch[0] = std::max(r0[0], r0[1]);
#if __aarch64__
          w = 1;
          cnt = 1;
          for (; w < w_unroll_size; w += 8) {
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
            vst1_f32(&dout_ch[cnt], vmax_123_345);
            vst1_f32(&dout_ch[cnt + 2], vmax_567_789);
            cnt += 4;
          }
          for (; w < w_even - 1; w += 2) {
            float32x4_t vr0 = vld1q_f32(&r0[w]);
            vr0 = vsetq_lane_f32(minval, vr0, 3);
            float32x2_t vmax = vpmax_f32(vget_low_f32(vr0), vget_high_f32(vr0));
            vmax = vpmax_f32(vmax, vmax);
            dout_ch[cnt] = vget_lane_f32(vmax, 0);
            cnt++;
          }
#else
          dr_out = dout_ch + 1;
          dr0 = (r0 + 1);
          cnt_num = w_unroll_size >> 3;
          cnt_num_remain = w_unroll_remain >> 1;
          // LOG(INFO) << "cnt_num: " << cnt_num << " cnt_num_remain: " <<
          // cnt_num_remain;
          if (cnt_num > 0 || cnt_num_remain > 0) {
            asm volatile(
                "cmp       %[cnt_num], #0            @cmp cnt_num,0\n"
                "ble       3f                        @ble exit\n"
                "1:                                  @main loop\n"
                "vld1.f32  {d0-d3}, [%[dr0]]!        @load d0-d3,dr0\n"
                "vld1.f32  {d4-d5}, [%[dr0]]!        @load d0-d3,dr0\n"
                "vext.f32  q4, q0, q1, #1            @vmax_2345\n"
                "vext.f32  q5, q1, q2, #1            @vmax_6789\n"
                "vpmax.f32 d12, d0, d1               @vmax_12_34\n"
                "vpmax.f32 d14, d2, d3               @vmax_56_78\n"
                "vpmax.f32 d13, d8, d9               @vmax_23_45\n"
                "vpmax.f32 d15, d10, d11             @vmax_67_89\n"
                "vmax.f32  d0, d12, d13              @12_34,23_45\n"
                "vmax.f32  d1, d14, d15              @56_78,67_89\n"
                "sub       %[dr0], #16               @add w,6\n"
                "vst1.f32  d0, [%[dr_out]]!          @vst1 d0,dr_out\n"
                "vst1.f32  d1, [%[dr_out]]!          @vst1 d0,dr_out\n"
                "subs      %[cnt_num], #1            @subs cnt_num,#1\n"
                "bne       1b                        @bne s3_max_loop_bot\n"
                "3:                                  @loop \n"
                "cmp       %[cnt_num_remain], #0     @cmp cnt_num,0\n"
                "ble       4f                        @ble exit\n"
                "2:                                  @bot loop\n"
                "vld1.f32  {d0-d1}, [%[dr0]]!        @load d0-d1,dr0\n"
                "vmov.f32  s3,s2                     @movs3, s2\n"
                "vpmax.f32 d0, d0, d1                @pmax d0,d0,d1\n"
                "vpmax.f32 d0, d0, d0                @pmax d0,d0,d0\n"
                "vst1.f32  d0[0], [%[dr_out]]!       @vst d0[0],dr_out\n"
                "sub       %[dr0], #8                @add w,2\n"
                "subs      %[cnt_num_remain], #1     @subs cnt_num,#1\n"
                "bne       2b                        @bne s3_max_loop_bot_1\n"
                "4:                                  @exit\n"
                : [dr0] "+r"(dr0), [dr1] "+r"(dr1), [dr_out] "+r"(dr_out),
                  [cnt_num] "+r"(cnt_num), [cnt_num_remain] "+r"(cnt_num_remain)
                : "r"(dr0), "r"(dr1), "r"(dr_out), "r"(cnt_num),
                  "r"(cnt_num_remain)
                : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6",
                  "q7", "q8");
          }
#endif
          if (w_remain > 0) {
            // deal with right pad
            int wstart = (w_even >> 1) * stride - padding;
            int wend = std::min(std::min(wstart + kernel, win + padding), win);
            float tmp = r0[wstart];  // std::numeric_limits<float>::min();
            for (int i = wstart; i < wend; i++) {
              tmp = std::max(tmp, r0[i]);
            }
            dout_ch[w_even >> 1] = tmp;
          }
        } else {  // two lines
          dout_ch[0] = std::max(std::max(r0[0], r0[1]), std::max(r1[0], r1[1]));
#ifdef __aarch64__
          w = 1;
          cnt = 1;
          for (; w < w_unroll_size; w += 8) {
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
            vst1_f32(&dout_ch[cnt], vmax_123_345);
            vst1_f32(&dout_ch[cnt + 2], vmax_567_789);
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
            dout_ch[cnt] = vget_lane_f32(vmax2, 0);
            cnt++;
          }
#else
          dr_out = dout_ch + 1;
          dr0 = (r0 + 1);
          dr1 = (r1 + 1);
          cnt_num = w_unroll_size >> 3;
          cnt_num_remain = w_unroll_remain >> 1;
          // LOG(INFO) << "cnt_num: " << cnt_num << " cnt_num_remain: " <<
          // cnt_num_remain;
          if (cnt_num > 0 || cnt_num_remain > 0) {
            asm volatile(
                "cmp       %[cnt_num], #0         @cmp cnt_num,0\n"
                "ble       3f                     @ble exit\n"
                "1:                               @main loop\n"
                "vld1.f32  {d0-d3}, [%[dr0]]!     @load d0-d5,dr0\n"
                "vld1.f32  {d6-d9}, [%[dr1]]!     @load d4-d7,dr1\n"
                "vld1.f32  {d4-d5}, [%[dr0]]!     @load d0-d3,dr0\n"
                "vld1.f32  {d10-d11}, [%[dr1]]!   @load d4-d7,dr1\n"
                "vmax.f32  q6, q0, q3             @max q0,q0,q2 1234\n"
                "vmax.f32  q7, q1, q4             @max q1,q1,q3 5678\n"
                "vmax.f32  q8, q2, q5             @max q1,q1,q3 9101112\n"
                //"vmov.f32  s7,s6                @mov s7, s6\n"
                "vext.f32  q0, q6, q7, #1         @vext q0,2345\n"
                "vext.f32  q1, q7, q8, #1         @vext q1,6789\n"
                "vpmax.f32 d4, d12, d13           @pmax "
                "d4,vmax_1234,vmax_1234\n"
                "vpmax.f32 d6, d14, d15           @pmax "
                "d6,vmax_5678,vmax_5678\n"
                "vpmax.f32 d5, d0, d1             @pmax "
                "d5,vmax_2345,vmax_2345\n"
                "vpmax.f32 d7, d2, d3             @pmax "
                "d7,vmax_6789,vmax_6789\n"
                "vmax.f32 d8, d4, d5              @max "
                "d2,vmax_12_34,vmax_23_45\n"
                "vmax.f32 d9, d6, d7              @max "
                "d2,vmax_56_78,vmax_67_89\n"
                "sub       %[dr0], #16            @add w,8\n"
                "sub       %[dr1], #16            @add w,8\n"
                "vst1.f32  d8, [%[dr_out]]!       @vst1 d0,dr_out\n"
                "vst1.f32  d9, [%[dr_out]]!       @vst1 d0,dr_out\n"
                "subs      %[cnt_num], #1         @subs cnt_num,#1\n"
                "bne       1b                     @bne s3_max_loop_bot\n"
                "3:                               @loop \n"
                "cmp       %[cnt_num_remain], #0  @cmp cnt_num,0\n"
                "ble       4f                     @ble exit\n"
                "2:                               @bot loop\n"
                "vld1.f32  {d0-d1}, [%[dr0]]!     @load d0-d1,dr0\n"
                "vld1.f32  {d2-d3}, [%[dr1]]!     @load d2-d3,dr1\n"
                "vmov.f32  s3,s2                  @movs3, s2\n"
                "vmov.f32  s7,s6                  @movs7, s6\n"
                "vmax.f32  q0, q0, q1             @max q0,q0,q1\n"
                "vpmax.f32 d0, d0, d1             @pmax d0,d0,d1\n"
                "vpmax.f32 d0, d0, d0             @pmax d0,d0,d0\n"
                "vst1.f32  d0[0], [%[dr_out]]!    @vst d0[0],dr_out\n"
                "sub       %[dr0], #8             @add w,6\n"
                "sub       %[dr1], #8             @add w,6\n"
                "subs      %[cnt_num_remain], #1  @subs cnt_num,#1\n"
                "bne       2b                     @bne s3_max_loop_bot_1\n"
                "4:                               @exit\n"
                : [dr0] "+r"(dr0), [dr1] "+r"(dr1), [dr_out] "+r"(dr_out),
                  [cnt_num] "+r"(cnt_num), [cnt_num_remain] "+r"(cnt_num_remain)
                : "r"(dr0), "r"(dr1), "r"(dr_out), "r"(cnt_num),
                  "r"(cnt_num_remain)
                : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6",
                  "q7", "q8", "q9");
          }
#endif
          if (w_remain > 0) {
            // deal with right pad
            int wstart = (w_even >> 1) * stride - padding;
            int wend = std::min(std::min(wstart + kernel, win + padding), win);
            float tmp = r0[wstart];  // std::numeric_limits<float>::min();
            for (int i = wstart; i < wend; i++) {  // only run 1 or 2 times
              tmp = std::max(tmp, std::max(r0[i], r1[i]));
            }
            dout_ch[w_even >> 1] = tmp;
          }
        }
      }
    }
  }
}

void pooling3x3s2p1_avg(const float* din, float* dout, int num, int chout,
                        int hout, int wout, int chin, int hin, int win,
                        bool exclusive) {
  int kernel = 3;
  int stride = 2;
  int padding = 1;
  int size_channel_out = wout * hout;
  int size_channel_in = win * hin;

  int w_needed = (wout << 1) + 1;
  int h_needed = (hout << 1) + 1;
  int w_limit = w_needed > win ? win : w_needed;
  int h_limit = h_needed > hin ? hin : h_needed;
  int w_even = (w_limit >> 1) << 1;
  int h_even = (h_limit >> 1) << 1;
  int w_unroll_size = ((w_even - 1) >> 3) << 3;
  int w_unroll_remain = w_even - 1 - w_unroll_size;
  int w_remain = w_needed - w_limit - padding;
  int h_remain = h_needed - h_limit - padding;
  int w_in_2 = win << 1;
  const float coef = 1.f / 9.f;
  const float coef_1 = exclusive ? 1.f : coef;
  const float coef_2 = exclusive ? 1.f / 2.f : coef;
  const float coef_3 = exclusive ? 1.f / 3.f : coef;
  const float coef_4 = exclusive ? 1.f / 4.f : coef;
  const float coef_6 = exclusive ? 1.f / 6.f : coef;
  float32x4_t vcoef = vdupq_n_f32(coef);
  float32x4_t vcoef_1 = vdupq_n_f32(coef_1);
  float32x4_t vcoef_2 = vdupq_n_f32(coef_2);
  float32x4_t vcoef_3 = vdupq_n_f32(coef_3);
  float32x4_t vcoef_4 = vdupq_n_f32(coef_4);
  float32x4_t vcoef_6 = vdupq_n_f32(coef_6);
  for (int n = 0; n < num; ++n) {
    float* dout_batch = dout + n * chout * size_channel_out;
    const float* din_batch = din + n * chin * size_channel_in;
#pragma omp parallel for
    for (int c = 0; c < chout; c++) {
      float* dout_ch = dout_batch + c * size_channel_out;
      const float* din_ch = din_batch + c * size_channel_in;
      const float* r0 = din_ch;
      const float* r1 = r0 + win;
      const float* r2 = r1 + win;
      int cnt_num = w_unroll_size >> 3;
      int cnt_num_remain = w_unroll_remain >> 1;
      float* dr_out = dout_ch;
      const float* dr0 = r0;
      const float* dr1 = r1;
      const float* dr2 = r2;
      int w = 1;
      int cnt = 1;
      float32x4_t vzero = vdupq_n_f32(0.f);
      dout_ch[0] = (r0[0] + r0[1] + r1[0] + r1[1]) * coef_4;
// first row with zero pad
#ifdef __aarch64__
      for (; w < w_unroll_size; w += 8) {
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
        float32x4_t vrst = vmulq_f32(vsum_123_345, vcoef_6);
        vst1q_f32(&dout_ch[cnt], vrst);
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
        float32x2_t vrst = vmul_f32(vsum2, vget_low_f32(vcoef_6));
        dout_ch[cnt] = vget_lane_f32(vrst, 0);
        cnt++;
      }
#else
      dr0 = dr0 + 1;
      dr1 = dr1 + 1;
      dr_out = dr_out + 1;
      // LOG(INFO) << "cnt_num: " << cnt_num << " cnt_num_remain: " <<
      // cnt_num_remain;
      if (cnt_num > 0 || cnt_num_remain > 0) {
        asm volatile(
            "cmp       %[cnt_num], #0        @cmp cnt_num,0\n"
            "ble       3f                    @ble exit\n"
            "1:                              @main loop\n"
            "vld1.f32  {d0-d3}, [%[dr0]]!    @load d0-d5,dr0\n"
            "vld1.f32  {d6-d9}, [%[dr1]]!    @load d4-d7,dr1\n"
            "vld1.f32  {d4-d5}, [%[dr0]]!    @load d0-d5,dr0\n"
            "vld1.f32  {d10-d11}, [%[dr1]]!  @load d4-d7,dr1\n"
            "vadd.f32  q6, q0, q3            @max r0_1234,r1_1234\n"
            "vadd.f32  q7, q1, q4            @max r0_5678,r1_5678\n"
            "vadd.f32  q8, q2, q5            @max r0_9101112,r1_9101112\n"
            //"vmov.f32  s7,s6               @mov s7, s6\n"
            "vext.f32  q0, q6, q7, #1        @vext max_2345\n"
            "vext.f32  q1, q6, q7, #3        @vext max_4567\n"
            "vext.f32  q2, q6, q7, #2        @vext max_3456\n"
            "vext.f32  q3, q7, q8, #1        @vext max_6789\n"
            "vadd.f32  q4, q6, q0            @add 1234, 2345\n"
            "vadd.f32  q5, q7, q1            @add 5678, 4567\n"
            "vadd.f32  q4, q4, q2            @add 3456, sum1\n"
            "vadd.f32  q5, q5, q3            @add 6789, sum2\n"
            "vmov.f32  s17, s18              @mov\n"
            "vmov.f32  s18, s21              @mov\n"
            "vmov.f32  s19, s23              @mov\n"
            "vmul.f32  q4, q4, %q[vcoef_6]   @mul\n"
            "sub       %[dr0], #16           @add w,8\n"
            "sub       %[dr1], #16           @add w,8\n"
            "subs      %[cnt_num], #1        @subs cnt_num,#1\n"
            "vst1.f32  d8, [%[dr_out]]!      @vst1 d0,dr_out\n"
            "vst1.f32  d9, [%[dr_out]]!      @vst1 d0,dr_out\n"
            "bne       1b                    @bne s3_max_loop\n"
            "3:                              @loop\n"
            "cmp       %[cnt_num_remain], #0 @cnt_num_remain<=0\n"
            "ble       4f                    @ble exit\n"
            "2:                              @main loop\n"
            "vld1.f32  {d0-d1}, [%[dr0]]!    @load d0-d1,dr0\n"
            "vld1.f32  {d2-d3}, [%[dr1]]!    @load d2-d3,dr1\n"
            "vext.f32  q0, %q[vzero], q0, #3 @ext v0_0123\n"
            "vext.f32  q1, %q[vzero], q1, #3 @ext v1_0123\n"
            "vadd.f32  q0, q0, q1            @add q0,q0,q1\n"
            "vpadd.f32 d0, d0, d1            @padd d0,d0,d1\n"
            "vpadd.f32 d0, d0, d0            @padd d0, d0,d0\n"
            "vmul.f32  d0, d0, %e[vcoef_6]   @mul\n"
            "sub       %[dr0], #8            @add w,6\n"
            "sub       %[dr1], #8            @add w,6\n"
            "subs      %[cnt_num_remain], #1 @subs cnt_num,#1\n"
            "vst1.f32  d0[0], [%[dr_out]]!   @vst d0[0],dr_out\n"
            "bne       2b                    @bne s3_max_loop_1\n"
            "4:                              @exit\n"
            : [dr0] "+r"(dr0), [dr1] "+r"(dr1), [dr_out] "+r"(dr_out),
              [cnt_num] "+r"(cnt_num), [cnt_num_remain] "+r"(cnt_num_remain),
              [vcoef_6] "+w"(vcoef_6), [vzero] "+w"(vzero)
            : "r"(dr0), "r"(dr1), "r"(dr_out), "r"(cnt_num), "r"(cnt_num_remain)
            : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7",
              "q8", "q9");
      }
#endif
      // int w = w_even - 1;
      if (w_remain > 0) {
        // deal with right pad
        int wstart = (w_even >> 1) * stride - padding;
        int wend = std::min(std::min(wstart + kernel, win + padding), win);
        float tmp1 = 0.f;  // std::numeric_limits<float>::min();
        float tmp2 = exclusive ? 1.0f / (2.f * (wend - wstart)) : coef;
        for (int i = wstart; i < wend; i++) {  // only run 1 or 2 times
          tmp1 += (r0[i] + r1[i]);
        }
        dout_ch[w_even >> 1] = tmp1 * tmp2;
        // cnt ++;
      }

      r0 = r1;
      r1 = r0 + win;
      r2 = r1 + win;
      dout_ch += wout;
      int h = 2;
      for (; h < h_even; h += 2) {
        // deal with left pad
        float sum0 = r0[0] + r0[1];
        float sum1 = r1[0] + r1[1];
        float sum2 = r2[0] + r2[1];
        dout_ch[0] = (sum0 + sum1 + sum2) * coef_6;
#ifdef __aarch64__
        w = 1;
        cnt = 1;
        for (; w < w_unroll_size; w += 8) {
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
          vst1q_f32(&dout_ch[cnt], vrst);
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
          dout_ch[cnt] = vget_lane_f32(vsum, 0) * coef;
          cnt++;
        }
#else
        dr_out = dout_ch + 1;
        dr0 = (r0 + 1);
        dr1 = (r1 + 1);
        dr2 = (r2 + 1);
        cnt_num = w_unroll_size >> 3;
        cnt_num_remain = w_unroll_remain >> 1;
        if (cnt_num > 0 || cnt_num_remain > 0) {
          asm volatile(
              "cmp       %[cnt_num], #0        @cmp cnt_num,0\n"
              "ble       3f                    @ble exit\n"
              "1:                              @main loop\n"
              "vld1.f32  {d0-d3}, [%[dr0]]!    @load d0-d5, "
              "dr0\n"
              "vld1.f32  {d6-d9}, [%[dr1]]!    @load d4-d7,dr1\n"
              "vld1.f32  {d12-d15}, [%[dr2]]!  @load d4-d7,dr1\n"
              "vld1.f32  {d4-d5}, [%[dr0]]!    @load d0-d5,dr0\n"
              "vld1.f32  {d10-d11}, [%[dr1]]!  @load d4-d7,dr1\n"
              "vld1.f32  {d16-d17}, [%[dr2]]!  @load d4-d7,dr1\n"
              "vadd.f32  q9, q0, q3            @max q0,q0,q2\n"
              "vadd.f32  q10, q1, q4           @max q1,q1,q3\n"
              "vadd.f32  q11, q2, q5           @max q1,q1,q3\n"
              "vadd.f32  q6, q9, q6            @max q0,q0,q2 1234\n"
              "vadd.f32  q7, q10, q7           @max q1,q1,q3 5678\n"
              "vadd.f32  q8, q11, q8           @max q1,q1,q3 9101112\n"
              //"vmov.f32  s7,s6               @mov s7, s6\n"
              "vext.f32  q0, q6, q7, #1        @vext max_2345\n"
              "vext.f32  q1, q6, q7, #3        @vext max_4567\n"
              "vext.f32  q2, q6, q7, #2        @vext max_3456\n"
              "vext.f32  q3, q7, q8, #1        @vext max_6789\n"
              "vadd.f32  q4, q6, q0            @add 1234,2345\n"
              "vadd.f32  q5, q7, q1            @add 5678,4567\n"
              "vadd.f32  q4, q4, q2            @add 3456,sum1\n"
              "vadd.f32  q5, q5, q3            @add 6789,sum2\n"
              "vmov.f32  s17, s18              @mov\n"
              "vmov.f32  s18, s21              @mov\n"
              "vmov.f32  s19, s23              @mov\n"
              "vmul.f32  q4, q4, %q[vcoef]     @mul\n"
              "sub       %[dr0], #16           @add w,8\n"
              "sub       %[dr1], #16           @add w,8\n"
              "sub       %[dr2], #16           @add w, 8\n"
              "subs      %[cnt_num], #1        @subs cnt_num,#1\n"
              "vst1.f32  d8, [%[dr_out]]!      @vst1 d0,dr_out\n"
              "vst1.f32  d9, [%[dr_out]]!      @vst1 d0,dr_out\n"
              "bne       1b                    @bne s3_max_loop_mid\n"
              "3:                              @loop\n"
              "cmp       %[cnt_num_remain], #0 @cnt_num_remain<=0\n"
              "ble       4f                    @ble exit1\n"
              "2:                              @mid loop\n"
              "vld1.f32  {d0-d1}, [%[dr0]]!    @load d0-d1,dr0\n"
              "vld1.f32  {d2-d3}, [%[dr1]]!    @load d2-d3,dr1\n"
              "vld1.f32  {d4-d5}, [%[dr2]]!    @load d2-d3,dr1\n"
              "vext.f32  q0, %q[vzero], q0, #3 @ext v0_0123\n"
              "vext.f32  q1, %q[vzero], q1, #3 @ext v1_0123\n"
              "vext.f32  q2, %q[vzero], q2, #3 @ext v1_0123\n"
              "vadd.f32  q0, q0, q1            @add q0,q0,q1\n"
              "vadd.f32  q0, q0, q2            @add q0,q0,q1\n"
              "vpadd.f32 d0, d0, d1            @padd d0,d0,d1\n"
              "vpadd.f32 d0, d0, d0            @padd d0,d0,d0\n"
              "vmul.f32  d0, d0, %e[vcoef]     @mul\n"
              "sub       %[dr0], #8            @add w,6\n"
              "sub       %[dr1], #8            @add w,6\n"
              "sub       %[dr2], #8            @add w,6\n"
              "subs      %[cnt_num_remain], #1 @cnt_num_remain--\n"
              "vst1.f32  d0[0], [%[dr_out]]!   @vst d0[0],dr_out\n"
              "bne       2b                    @bne s3_max_loop_mid_1\n"
              "4:                              @exit\n"
              : [dr0] "+r"(dr0), [dr1] "+r"(dr1), [dr2] "+r"(dr2),
                [dr_out] "+r"(dr_out), [cnt_num] "+r"(cnt_num),
                [cnt_num_remain] "+r"(cnt_num_remain), [vcoef] "+w"(vcoef),
                [vzero] "+w"(vzero)
              : "r"(dr0), "r"(dr1), "r"(dr2), "r"(dr_out), "r"(cnt_num),
                "r"(cnt_num_remain)
              : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7",
                "q8", "q9", "q10", "q11", "q12");
        }
#endif
        if (w_remain > 0) {
          // deal with right pad
          int wstart = (w_even >> 1) * stride - padding;
          int wend = std::min(std::min(wstart + kernel, win + padding), win);
          float tmp1 = 0.f;
          float tmp2 = exclusive ? 1.0f / (3.f * (wend - wstart)) : coef;
          for (int i = wstart; i < wend; i++) {
            tmp1 += (r0[i] + r1[i] + r2[i]);
          }
          dout_ch[w_even >> 1] = tmp1 * tmp2;
          // cnt ++;
        }
        r0 = r2;
        r1 = r0 + win;
        r2 = r1 + win;
        dout_ch += wout;
      }

      if (h_remain > 0) {
        // deal with bottom pad
        // first row with zero pad
        int hstart = (h >> 1) * stride - padding;
        int hend = std::min(std::min(hstart + kernel, hin + padding), hin);
        if (hstart == hend - 1) {  // only one line
          dout_ch[0] = (r0[0] + r0[1]) * coef_2;
#ifdef __aarch64__
          w = 1;
          cnt = 1;
          for (; w < w_unroll_size; w += 8) {
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
            float32x4_t vrst = vmulq_f32(vsum_123_345, vcoef_3);
            vst1q_f32(&dout_ch[cnt], vrst);
            cnt += 4;
          }
          for (; w < w_even - 1; w += 2) {
            float32x4_t vr0 = vld1q_f32(&r0[w]);
            vr0 = vsetq_lane_f32(0.f, vr0, 3);
            float32x2_t vsum = vpadd_f32(vget_low_f32(vr0), vget_high_f32(vr0));
            vsum = vpadd_f32(vsum, vsum);
            dout_ch[cnt] = vget_lane_f32(vsum, 0) * coef_3;
            cnt++;
          }
#else
          dr_out = dout_ch + 1;
          dr0 = (r0 + 1);
          cnt_num = w_unroll_size >> 3;
          cnt_num_remain = w_unroll_remain >> 1;
          if (cnt_num > 0 || cnt_num_remain > 0) {
            asm volatile(
                "cmp       %[cnt_num], #0         @cmp cnt_num,0\n"
                "ble       3f                     @ble exit\n"
                "1:                               @main loop\n"
                "vld1.f32  {d12-d15}, [%[dr0]]!   @load d0-d3,dr0\n"
                "vld1.f32  {d16-d17}, [%[dr0]]!   @load d0-d3,dr0\n"
                "vext.f32  q0, q6, q7, #1         @vext max_2345\n"
                "vext.f32  q1, q6, q7, #3         @vext max_4567\n"
                "vext.f32  q2, q6, q7, #2         @vext max_3456\n"
                "vext.f32  q3, q7, q8, #1         @vext max_6789\n"
                "vadd.f32  q4, q6, q0             @add 1234,2345\n"
                "vadd.f32  q5, q7, q1             @add 5678,4567\n"
                "vadd.f32  q4, q4, q2             @add 3456,sum1\n"
                "vadd.f32  q5, q5, q3             @add 6789,sum2\n"
                "vmov.f32  s17, s18               @mov\n"
                "vmov.f32  s18, s21               @mov\n"
                "vmov.f32  s19, s23               @mov\n"
                "vmul.f32  q4, q4, %q[vcoef_3]    @mul\n"
                "sub       %[dr0], #16            @add w,6\n"
                "subs      %[cnt_num], #1         @subs cnt_num,#1\n"
                "vst1.f32  d8, [%[dr_out]]!       @vst1 d0,dr_out\n"
                "vst1.f32  d9, [%[dr_out]]!       @vst1 d0,dr_out\n"
                "bne       1b                     @bne s3_max_loop_bot\n"
                "3:                               @loop\n"
                "cmp       %[cnt_num_remain], #0  @cnt_num_remain<=0\n"
                "ble       4f                     @ble exit\n"
                "2:                               @bot loop\n"
                "vld1.f32  {d0-d1}, [%[dr0]]!     @load d0-d1,dr0\n"
                "vext.f32  q0, %q[vzero], q0, #3  @ext v0_0123\n"
                "vpadd.f32 d0, d0, d1             @padd d0,d0,d1\n"
                "vpadd.f32 d0, d0, d0             @padd d0,d0,d0\n"
                "vmul.f32  d0, d0, %e[vcoef_3]    @mul\n"
                "sub       %[dr0], #8             @add w,2\n"
                "subs      %[cnt_num_remain], #1  @cnt_num_remain--\n"
                "vst1.f32  d0[0], [%[dr_out]]!    @vst d0[0],dr_out\n"
                "bne       2b                     @bne s3_max_loop_bot_1\n"
                "4:                               @exit\n"
                : [dr0] "+r"(dr0), [dr1] "+r"(dr1), [dr_out] "+r"(dr_out),
                  [cnt_num] "+r"(cnt_num),
                  [cnt_num_remain] "+r"(cnt_num_remain),
                  [vcoef_3] "+w"(vcoef_3), [vzero] "+w"(vzero)
                : "r"(dr0), "r"(dr1), "r"(dr_out), "r"(cnt_num),
                  "r"(cnt_num_remain)
                : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6",
                  "q7", "q8");
          }
#endif
          if (w_remain > 0) {
            // deal with right pad
            int wstart = (w_even >> 1) * stride - padding;
            int wend = std::min(std::min(wstart + kernel, win + padding), win);
            float tmp1 = 0.f;
            float tmp2 = exclusive ? 1.0f / (1.f * (wend - wstart)) : coef;
            for (int i = wstart; i < wend; i++) {
              tmp1 += r0[i];
            }
            dout_ch[w_even >> 1] = tmp1 * tmp2;
          }
        } else {  // two lines
          dout_ch[0] = (r0[0] + r0[1] + r1[0] + r1[1]) * coef_4;
#ifdef __aarch64__
          w = 1;
          cnt = 1;
          for (; w < w_unroll_size; w += 8) {
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
            float32x4_t vrst = vmulq_f32(vsum_123_345, vcoef_6);
            vst1q_f32(&dout_ch[cnt], vrst);
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
            float32x2_t vrst = vmul_f32(vsum2, vget_low_f32(vcoef_6));
            dout_ch[cnt] = vget_lane_f32(vrst, 0);
            cnt++;
          }
#else
          dr_out = dout_ch + 1;
          dr0 = (r0 + 1);
          dr1 = (r1 + 1);
          cnt_num = w_unroll_size >> 3;
          cnt_num_remain = w_unroll_remain >> 1;
          if (cnt_num > 0 || cnt_num_remain > 0) {
            asm volatile(
                "cmp       %[cnt_num], #0        @cmp cnt_num,0\n"
                "ble       3f                    @ble exit\n"
                "1:                              @main loop\n"
                "vld1.f32  {d0-d3}, [%[dr0]]!    @load d0-d5,dr0\n"
                "vld1.f32  {d6-d9}, [%[dr1]]!    @load d4-d7,dr1\n"
                "vld1.f32  {d4-d5}, [%[dr0]]!    @load d0-d3,dr0\n"
                "vld1.f32  {d10-d11}, [%[dr1]]!  @load d4-d7,dr1\n"
                "vadd.f32  q6, q0, q3            @add q0,q0,q2 1234\n"
                "vadd.f32  q7, q1, q4            @add q1,q1,q3 5678\n"
                "vadd.f32  q8, q2, q5            @add q1,q1,q3 9101112\n"
                //"vmov.f32  s7,s6               @mov s7,s6\n"
                "vext.f32  q0, q6, q7, #1        @vext max_2345\n"
                "vext.f32  q1, q6, q7, #3        @vext max_4567\n"
                "vext.f32  q2, q6, q7, #2        @vext max_3456\n"
                "vext.f32  q3, q7, q8, #1        @vext max_6789\n"
                "vadd.f32  q4, q6, q0            @add 1234,2345\n"
                "vadd.f32  q5, q7, q1            @add 5678,4567\n"
                "vadd.f32  q4, q4, q2            @add 3456,sum1\n"
                "vadd.f32  q5, q5, q3            @add 6789,sum2\n"
                "vmov.f32  s17, s18              @mov\n"
                "vmov.f32  s18, s21              @mov\n"
                "vmov.f32  s19, s23              @mov\n"
                "vmul.f32  q4, q4, %q[vcoef_6]   @mul\n"
                "sub       %[dr0], #16           @add w,8\n"
                "sub       %[dr1], #16           @add w,8\n"
                "subs      %[cnt_num], #1        @subs cnt_num,#1\n"
                "vst1.f32  d8, [%[dr_out]]!      @vst1 d0,dr_out\n"
                "vst1.f32  d9, [%[dr_out]]!      @vst1 d0, dr_out\n"
                "bne       1b                    @bne s3_max_loop_bot\n"
                "3:                              @loop\n"
                "cmp       %[cnt_num_remain], #0 @cnt_num_remain<=0\n"
                "ble       4f                    @ble exit\n"
                "2:                              @bot loop\n"
                "vld1.f32  {d0-d1}, [%[dr0]]!    @load d0-d1,dr0\n"
                "vld1.f32  {d2-d3}, [%[dr1]]!    @load d2-d3,dr1\n"
                "vext.f32  q0, %q[vzero], q0, #3 @ext v0_0123\n"
                "vext.f32  q1, %q[vzero], q1, #3 @ext v1_0123\n"
                "vadd.f32  q0, q0, q1            @add q0,q0,q1\n"
                "vpadd.f32 d0, d0, d1            @padd d0,d0,d1\n"
                "vpadd.f32 d0, d0, d0            @padd d0,d0,d0\n"
                "vmul.f32  d0, d0, %e[vcoef_6]   @mul\n"
                "sub       %[dr0], #8            @add w,6\n"
                "sub       %[dr1], #8            @add w,6\n"
                "subs      %[cnt_num_remain], #1 @cnt_num_remain--\n"
                "vst1.f32  d0[0], [%[dr_out]]!   @vst d0[0],dr_out\n"
                "bne       2b                    @bne s3_max_loop_bot_1\n"
                "4:                              @exit\n"
                : [dr0] "+r"(dr0), [dr1] "+r"(dr1), [dr_out] "+r"(dr_out),
                  [cnt_num] "+r"(cnt_num),
                  [cnt_num_remain] "+r"(cnt_num_remain),
                  [vcoef_6] "+w"(vcoef_6), [vzero] "+w"(vzero)
                : "r"(dr0), "r"(dr1), "r"(dr_out), "r"(cnt_num),
                  "r"(cnt_num_remain)
                : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6",
                  "q7", "q8", "q9");
          }
#endif
          if (w_remain > 0) {
            // deal with right pad
            int wstart = (w_even >> 1) * stride - padding;
            int wend = std::min(std::min(wstart + kernel, win + padding), win);
            float tmp1 = 0.f;
            float tmp2 = exclusive ? 1.0f / (2.f * (wend - wstart)) : coef;
            for (int i = wstart; i < wend; i++) {  // only run 1 or 2 times
              tmp1 += (r0[i] + r1[i]);
            }
            dout_ch[w_even >> 1] = tmp1 * tmp2;
          }
        }
      }
    }
  }
}

void pooling3x3s2p0_max(const float* din, float* dout, int num, int chout,
                        int hout, int wout, int chin, int hin, int win) {
  int kernel = 3;
  int stride = 2;
  int padding = 0;
  int size_channel_out = wout * hout;
  int size_channel_in = win * hin;

  int w_needed = (wout << 1) + 1;
  int h_needed = (hout << 1) + 1;
  int w_limit = w_needed > win ? win : w_needed;
  int h_limit = h_needed > hin ? hin : h_needed;
  int w_even = ((w_limit - 1) >> 1) << 1;
  int h_even = ((h_limit - 1) >> 1) << 1;
  int w_unroll_size = (w_even >> 3) << 3;
  int w_unroll_remain = w_even - w_unroll_size;
  int w_remain = w_needed - w_limit;
  int h_remain = h_needed - h_limit;
  int w_in_2 = win << 1;
  float minval = std::numeric_limits<float>::lowest();
  for (int n = 0; n < num; ++n) {
    float* dout_batch = dout + n * chout * size_channel_out;
    const float* din_batch = din + n * chin * size_channel_in;
#pragma omp parallel for
    for (int c = 0; c < chout; c++) {
      float* dout_ch = dout_batch + c * size_channel_out;
      const float* din_ch = din_batch + c * size_channel_in;
      const float* r0 = din_ch;
      const float* r1 = r0 + win;
      const float* r2 = r1 + win;
      // w = w_in - 8;
      float* dr_out = dout_ch;
      const float* dr0 = r0;
      const float* dr1 = r1;
      const float* dr2 = r2;
      int w = 0;
      int cnt = 0;
      // dout_ch[0] = std::max(std::max(r0[0], r0[1]), std::max(r1[0],
      // r1[1]));
      // first row with zero pad
      // r0 = r1;
      // r1 = r0 + w_in;
      // r2 = r1 + w_in;
      // dout_channel += w_out;
      int h = 0;
      for (; h < h_even; h += 2) {
        // deal with left pad
        float maxr0 = std::max(r0[0], r0[1]);
        float maxr1 = std::max(r1[0], r1[1]);
        float maxr2 = std::max(r2[0], r2[1]);
// dout_ch[0] = std::max(std::max(maxr0, maxr1), maxr2);
#ifdef __aarch64__
        w = 0;
        cnt = 0;
        for (; w < w_unroll_size; w += 8) {
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
          vst1_f32(&dout_ch[cnt], vmax_123_345);
          vst1_f32(&dout_ch[cnt + 2], vmax_567_789);
          cnt += 4;
        }
        for (; w < w_even; w += 2) {
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
          dout_ch[cnt] = vget_lane_f32(vmax, 0);
          cnt++;
        }
#else
        dr_out = dout_ch;  // + 1;
        dr0 = r0;          // (r0 + 1);
        dr1 = r1;          // (r1 + 1);
        dr2 = r2;          // (r2 + 1);
        int cnt_num = w_unroll_size >> 3;
        int cnt_num_remain = w_unroll_remain >> 1;
        if (cnt_num > 0 || cnt_num_remain > 0) {
          asm volatile(
              "cmp       %[cnt_num], #0           @cmp cnt_num,0\n"
              "ble       3f                       @ble exit\n"
              "1:                                 @main loop\n"
              "vld1.f32  {d0-d3}, [%[dr0]]!       @load d0-d5,dr0\n"
              "vld1.f32  {d6-d9}, [%[dr1]]!       @load d4-d7,dr1\n"
              "vld1.f32  {d12-d15}, [%[dr2]]!     @load d4-d7,dr1\n"
              "vld1.f32  {d4}, [%[dr0]]!          @load d0-d5,dr0\n"
              "vld1.f32  {d10}, [%[dr1]]!         @load d4-d7,dr1\n"
              "vld1.f32  {d16}, [%[dr2]]!         @load d4-d7,dr1\n"
              "vmax.f32  q9, q0, q3               @max q0,q0,q2\n"
              "vmax.f32  q10, q1, q4              @max q1,q1,q3\n"
              "vmax.f32  d22, d4, d10             @max q1,q1,q3\n"
              "vmax.f32  q0, q9, q6               @max q0,q0,q2 1234\n"
              "vmax.f32  q3, q10, q7              @max q1,q1,q3 5678\n"
              "vmax.f32  d2, d22, d16             @max q1,q1,q3 9101112\n"
              //"vmov.f32  s7,s6                  @mov s7, s6\n"
              "vext.f32  q4, q0, q3, #1           @vext 2345\n"
              "vext.f32  q2, q3, q1, #1           @vext 6789\n"
              "vpmax.f32 d10, d0, d1              @pmax "
              "d10,vmax_1234,vmax_1234\n"
              "vpmax.f32 d12, d6, d7              @pmax "
              "d12,vmax_5678,vmax_5678\n"
              "vpmax.f32 d11, d8, d9              @pmax "
              "d11,vmax_2345,vmax_2345\n"
              "vpmax.f32 d13, d4, d5              @pmax "
              "d13,vmax_6789,vmax_6789\n"
              "vmax.f32 d0, d10, d11              @pmax "
              "d0,vmax_12_34,vmax_23_45\n"
              "vmax.f32 d1, d12, d13              @pmax "
              "d1,vmax_56_78,vmax_67_89\n"
              "sub       %[dr0], #8               @add w,8\n"
              "sub       %[dr1], #8               @add w,8\n"
              "sub       %[dr2], #8               @add w,8\n"
              "vst1.f32  d0, [%[dr_out]]!         @vst1 d0,dr_out\n"
              "vst1.f32  d1, [%[dr_out]]!         @vst1 d0,dr_out\n"
              "subs      %[cnt_num], #1           @cnt_num--\n"
              "bne       1b                       @bne s3_max_loop_mid\n"
              "3:                                 @loop\n"
              "cmp       %[cnt_num_remain], #0    @cmp cnt_num_remain,0\n"
              "ble       4f                       @ble exit1\n"
              "2:                                 @mid loop\n"
              "vld1.f32  {d0-d1}, [%[dr0]]!       @load d0-d1,dr0\n"
              "vld1.f32  {d2-d3}, [%[dr1]]!       @load d2-d3,dr1\n"
              "vld1.f32  {d4-d5}, [%[dr2]]!       @load d2-d3,dr1\n"
              "vmov.f32  s3,s2                    @movs3,s2\n"
              "vmov.f32  s7,s6                    @movs7,s6\n"
              "vmov.f32  s11,s10                  @movs11,s10\n"
              "vmax.f32  q0, q0, q1               @max q0,q0,q1\n"
              "vmax.f32  q0, q0, q2               @max q0,q0,q2\n"
              "vpmax.f32 d0, d0, d1               @pmax d0,d0,d1\n"
              "vpmax.f32 d0, d0, d0               @pmax d0,d0,d0\n"
              "vst1.f32  d0[0], [%[dr_out]]!      @vst d0[0],dr_out\n"
              "sub       %[dr0], #8               @add w,6\n"
              "sub       %[dr1], #8               @add w,6\n"
              "sub       %[dr2], #8               @add w,6\n"
              "subs      %[cnt_num_remain], #1    @cnt_num_remain--\n"
              "bne       2b                       @bne s3_max_loop_mid_1\n"
              "4:                                 @exit\n"
              : [dr0] "+r"(dr0), [dr1] "+r"(dr1), [dr2] "+r"(dr2),
                [dr_out] "+r"(dr_out), [cnt_num] "+r"(cnt_num),
                [cnt_num_remain] "+r"(cnt_num_remain)
              : "r"(dr0), "r"(dr1), "r"(dr2), "r"(dr_out), "r"(cnt_num),
                "r"(cnt_num_remain)
              : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7",
                "q8", "q9", "q10", "q11", "q12");
        }
#endif
        if (w_remain > 0) {
          // deal with right pad
          int wstart = (w_even >> 1) * stride - padding;
          int wend = std::min(std::min(wstart + kernel, win + padding), win);
          float tmp = r0[wstart];  // std::numeric_limits<float>::min();
          for (int i = wstart; i < wend; i++) {
            tmp = std::max(tmp, std::max(r0[i], r1[i]));
            tmp = std::max(tmp, r2[i]);
          }
          dout_ch[w_even >> 1] = tmp;
          // cnt ++;
        }
        r0 = r2;
        r1 = r0 + win;
        r2 = r1 + win;
        dout_ch += wout;
      }

      if (h_remain > 0) {
// deal with bottom pad
// first row with zero pad
// int hstart = (h >> 1) * stride_h - pad_h;
// int hend = std::min(std::min(hstart + kernel_h, hin + pad_h), hin);
// dout_ch[0] = std::max(std::max(r0[0], r0[1]), std::max(r1[0],
// r1[1]));
#ifdef __aarch64__
        w = 0;
        cnt = 0;
        for (; w < w_unroll_size; w += 8) {
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
          vst1_f32(&dout_ch[cnt], vmax_123_345);
          vst1_f32(&dout_ch[cnt + 2], vmax_567_789);
          cnt += 4;
        }
        for (; w < w_even; w += 2) {
          float32x4_t vr0 = vld1q_f32(&r0[w]);
          float32x4_t vr1 = vld1q_f32(&r1[w]);
          vr0 = vsetq_lane_f32(minval, vr0, 3);
          vr1 = vsetq_lane_f32(minval, vr1, 3);
          float32x4_t vmax1 = vmaxq_f32(vr0, vr1);
          float32x2_t vmax2 =
              vpmax_f32(vget_low_f32(vmax1), vget_high_f32(vmax1));
          vmax2 = vpmax_f32(vmax2, vmax2);
          dout_ch[cnt] = vget_lane_f32(vmax2, 0);
          cnt++;
        }
#else
        dr_out = dout_ch;  // + 1;
        dr0 = r0;          // (r0 + 1);
        dr1 = r1;          // (r1 + 1);
        int cnt_num = w_unroll_size >> 3;
        int cnt_num_remain = w_unroll_remain >> 1;
        if (cnt_num > 0 || cnt_num_remain > 0) {
          asm volatile(
              "cmp       %[cnt_num], #0           @cmp cnt_num,0\n"
              "ble       3f                       @ble exit\n"
              "1:                                 @main loop\n"
              "vld1.f32  {d0-d3}, [%[dr0]]!       @load d0-d5,dr0\n"
              "vld1.f32  {d6-d9}, [%[dr1]]!       @load d4-d7,dr1\n"
              "vld1.f32  {d4}, [%[dr0]]!          @load d0-d3,dr0\n"
              "vld1.f32  {d10}, [%[dr1]]!         @load d4-d7,dr1\n"
              "vmax.f32  q6, q0, q3               @max q0,q0,q2 1234\n"
              "vmax.f32  q7, q1, q4               @max q1,q1,q3 5678\n"
              "vmax.f32  d16, d4, d10             @max q1,q1,q3 9101112\n"
              //"vmov.f32  s7,s6                  @mov s7,s6\n"
              "vext.f32  q0, q6, q7, #1           @vext q0,2345\n"
              "vext.f32  q1, q7, q8, #1           @vext q1,6789\n"
              "vpmax.f32 d4, d12, d13             @pmax "
              "d4,vmax_1234,vmax_1234\n"
              "vpmax.f32 d6, d14, d15             @pmax "
              "d6,vmax_5678,vmax_5678\n"
              "vpmax.f32 d5, d0, d1               @pmax "
              "d5,vmax_2345,vmax_2345\n"
              "vpmax.f32 d7, d2, d3               @pmax "
              "d7,vmax_6789,vmax_6789\n"
              "vmax.f32 d8, d4, d5                @max "
              "d2,vmax_12_34,vmax_23_45\n"
              "vmax.f32 d9, d6, d7                @max "
              "d2,vmax_56_78,vmax_67_89\n"
              "sub       %[dr0], #8               @add w,8\n"
              "sub       %[dr1], #8               @add w,8\n"
              "vst1.f32  d8, [%[dr_out]]!         @vst1 d0,dr_out\n"
              "vst1.f32  d9, [%[dr_out]]!         @vst1 d0,dr_out\n"
              "subs      %[cnt_num], #1           @subs cnt_num,#1\n"
              "bne       1b                       @bne s3_max_loop_bot\n"
              "3:                                 @loop \n"
              "cmp       %[cnt_num_remain], #0    @cmp cnt_num_remain,0\n"
              "ble       4f                       @ble exit\n"
              "2:                                 @bot loop\n"
              "vld1.f32  {d0-d1}, [%[dr0]]!       @load d0-d1,dr0\n"
              "vld1.f32  {d2-d3}, [%[dr1]]!       @load d2-d3,dr1\n"
              "vmov.f32  s3,s2                    @movs3,s2\n"
              "vmov.f32  s7,s6                    @movs7,s6\n"
              "vmax.f32  q0, q0, q1               @max q0,q0,q1\n"
              "vpmax.f32 d0, d0, d1               @pmax d0,d0,d1\n"
              "vpmax.f32 d0, d0, d0               @pmax d0,d0,d0\n"
              "vst1.f32  d0[0], [%[dr_out]]!      @vst d0[0],dr_out\n"
              "sub       %[dr0], #8               @add w,6\n"
              "sub       %[dr1], #8               @add w,6\n"
              "subs      %[cnt_num_remain], #1    @cnt_num_remain--\n"
              "bne       2b                       @bne s3_max_loop_bot_1\n"
              "4:                                 @exit\n"
              : [dr0] "+r"(dr0), [dr1] "+r"(dr1), [dr_out] "+r"(dr_out),
                [cnt_num] "+r"(cnt_num), [cnt_num_remain] "+r"(cnt_num_remain)
              : "r"(dr0), "r"(dr1), "r"(dr_out), "r"(cnt_num),
                "r"(cnt_num_remain)
              : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7",
                "q8", "q9");
        }
#endif
        if (w_remain > 0) {
          // deal with right pad
          int wstart = (w_even >> 1) * stride - padding;
          int wend = std::min(std::min(wstart + kernel, win + padding), win);
          float tmp = r0[wstart];  // std::numeric_limits<float>::min();
          for (int i = wstart; i < wend; i++) {  // only run 1 or 2 times
            tmp = std::max(tmp, std::max(r0[i], r1[i]));
          }
          dout_ch[w_even >> 1] = tmp;
        }
      }
    }
  }
}

void pooling3x3s2p0_avg(const float* din, float* dout, int num, int chout,
                        int hout, int wout, int chin, int hin, int win,
                        bool exclusive) {
  int kernel = 3;
  int stride = 2;
  int padding = 0;
  int size_channel_out = wout * hout;
  int size_channel_in = win * hin;

  int w_needed = (wout << 1) + 1;
  int h_needed = (hout << 1) + 1;
  int w_limit = w_needed > win ? win : w_needed;
  int h_limit = h_needed > hin ? hin : h_needed;
  int w_even = ((w_limit - 1) >> 1) << 1;
  int h_even = ((h_limit - 1) >> 1) << 1;
  int w_unroll_size = (w_even >> 3) << 3;
  int w_unroll_remain = w_even - w_unroll_size;
  int w_remain = w_needed - w_limit;
  int h_remain = h_needed - h_limit;
  int w_in_2 = win << 1;
  const float coef = 1.f / 9.f;
  const float coef_6 = exclusive ? 1.f / 6.f : coef;
  float32x4_t vcoef = vdupq_n_f32(coef);
  float32x4_t vcoef_6 = vdupq_n_f32(coef_6);
  for (int n = 0; n < num; ++n) {
    float* dout_batch = dout + n * chout * size_channel_out;
    const float* din_batch = din + n * chin * size_channel_in;
#pragma omp parallel for
    for (int c = 0; c < chout; c++) {
      float* dout_ch = dout_batch + c * size_channel_out;
      const float* din_ch = din_batch + c * size_channel_in;
      const float* r0 = din_ch;
      const float* r1 = r0 + win;
      const float* r2 = r1 + win;
      // w = w_in - 8;
      float* dr_out = dout_ch;
      const float* dr0 = r0;
      const float* dr1 = r1;
      const float* dr2 = r2;

      float32x4_t vzero = vdupq_n_f32(0.f);

      int h = 0;
      for (; h < h_even; h += 2) {
// LOG(INFO) << "h: " << h <<", dr0:" << r0 << ", dr1: " << r1 <<
// ",dr2: " <<r2; deal with left pad float sum0 = r0[0] + r0[1]; float
// sum1 = r1[0] + r1[1]; float sum2 = r2[0] + r2[1]; dout_channel[0] =
// (sum0 + sum1 + sum2) / 9.f;
#ifdef __aarch64__
        int w = 0;
        int cnt = 0;
        for (; w < w_unroll_size; w += 8) {
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
          vst1q_f32(&dout_ch[cnt], vrst);
          cnt += 4;
        }
        for (; w < w_even; w += 2) {
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
          dout_ch[cnt] = vget_lane_f32(vsum, 0) * coef;
          cnt++;
        }
#else
        dr_out = dout_ch;  // + 1;
        dr0 = r0;          // (r0 + 1);
        dr1 = r1;          // (r1 + 1);
        dr2 = r2;          // (r2 + 1);
        int cnt_num = w_unroll_size >> 3;
        int cnt_num_remain = w_unroll_remain >> 1;
        // LOG(INFO) << "cnt_num: " << cnt_num << " cnt_num_remain: " <<
        // cnt_num_remain;
        if (cnt_num > 0 || cnt_num_remain > 0) {
          asm volatile(
              "cmp       %[cnt_num], #0           @cmp cnt_num, 0\n"
              "ble       loop3_ave_p0             @ble exit\n"
              "s3_ave_loop_mid_p0:                @main loop\n"
              "vld1.f32  {d0-d3}, [%[dr0]]!       @load d0-d5, dr0\n"
              "vld1.f32  {d6-d9}, [%[dr1]]!       @load d4-d7, dr1\n"
              "vld1.f32  {d12-d15}, [%[dr2]]!     @load d4-d7, dr2\n"
              "vld1.f32  {d4}, [%[dr0]]!          @load d0-d5, dr0\n"
              "vld1.f32  {d10}, [%[dr1]]!         @load d4-d7, dr1\n"
              "vld1.f32  {d16}, [%[dr2]]!         @load d4-d7, dr2\n"
              "vadd.f32  q9, q0, q3               @max q0,q0,q2\n"
              "vadd.f32  q10, q1, q4              @max q1,q1,q3\n"
              "vadd.f32  d22, d4, d10             @max q1,q1,q3\n"
              "vadd.f32  q6, q9, q6               @max q0,q0,q2 1234\n"
              "vadd.f32  q7, q10, q7              @max q1,q1,q3 5678\n"
              "vadd.f32  d16, d22, d16            @max q1,q1,q3 9101112\n"
              //"vmov.f32  s7,s6                  @mov s7, s6\n"
              "vext.f32  q0, q6, q7, #1           @vext max_2345\n"
              "vext.f32  q1, q6, q7, #3           @vext max_4567\n"
              "vext.f32  q2, q6, q7, #2           @vext max_3456\n"
              "vext.f32  q3, q7, q8, #1           @vext max_6789\n"
              "vadd.f32  q4, q6, q0               @add 1234, 2345\n"
              "vadd.f32  q5, q7, q1               @add 5678, 4567\n"
              "vadd.f32  q4, q4, q2               @add 3456, sum1\n"
              "vadd.f32  q5, q5, q3               @add 6789, sum2\n"
              "vmov.f32  s17, s18                 @mov\n"
              "vmov.f32  s18, s21                 @mov\n"
              "vmov.f32  s19, s23                 @mov\n"
              "vmul.f32  q4, q4, %q[vcoef]        @mul\n"
              "sub       %[dr0], #8               @add w,8\n"
              "sub       %[dr1], #8               @add w,8\n"
              "sub       %[dr2], #8               @add w,8\n"
              "subs      %[cnt_num], #1           @cnt_num--\n"
              "vst1.f32  d8, [%[dr_out]]!         @vst1 d0,dr_out\n"
              "vst1.f32  d9, [%[dr_out]]!         @vst1 d0,dr_out\n"
              "bne       s3_ave_loop_mid_p0       @bne s3_max_loop_mid\n"
              "loop3_ave_p0:                      @loop\n"
              "cmp       %[cnt_num_remain], #0    @cmp cnt_num_remain,0\n"
              "ble       exit1_ave_p0             @ble exit1\n"
              "s3_ave_loop_mid_1_p0:              @mid loop\n"
              "vld1.f32  {d0-d1}, [%[dr0]]!       @load d0-d1,dr0\n"
              "vld1.f32  {d2-d3}, [%[dr1]]!       @load d2-d3,dr1\n"
              "vld1.f32  {d4-d5}, [%[dr2]]!       @load d2-d3,dr1\n"
              "vext.f32  q0, %q[vzero], q0, #3    @ext v0_0123\n"
              "vext.f32  q1, %q[vzero], q1, #3    @ext v1_0123\n"
              "vext.f32  q2, %q[vzero], q2, #3    @ext v1_0123\n"
              "vadd.f32  q0, q0, q1               @add q0,q0,q1\n"
              "vadd.f32  q0, q0, q2               @add q0,q0,q1\n"
              "vpadd.f32 d0, d0, d1               @padd d0,d0,d1\n"
              "vpadd.f32 d0, d0, d0               @padd d0,d0,d0\n"
              "vmul.f32  d0, d0, %e[vcoef]        @mul\n"
              "sub       %[dr0], #8               @add w,6\n"
              "sub       %[dr1], #8               @add w,6\n"
              "sub       %[dr2], #8               @add w,6\n"
              "subs      %[cnt_num_remain], #1    @cnt_num_remain--\n"
              "vst1.f32  d0[0], [%[dr_out]]!      @vst d0[0],dr_out\n"
              "bne       s3_ave_loop_mid_1_p0     @bne s3_max_loop_mid_1\n"
              "exit1_ave_p0:                      @exit\n"
              : [dr0] "+r"(dr0), [dr1] "+r"(dr1), [dr2] "+r"(dr2),
                [dr_out] "+r"(dr_out), [cnt_num] "+r"(cnt_num),
                [cnt_num_remain] "+r"(cnt_num_remain), [vcoef] "+w"(vcoef),
                [vzero] "+w"(vzero)
              : "r"(dr0), "r"(dr1), "r"(dr2), "r"(dr_out), "r"(cnt_num),
                "r"(cnt_num_remain)
              : "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9",
                "q10", "q11", "q12");
        }
#endif
        if (w_remain > 0) {
          // deal with right pad
          int wstart = (w_even >> 1) * stride - padding;
          int wend = std::min(std::min(wstart + kernel, win + padding), win);
          float tmp1 = 0.f;
          float tmp2 = exclusive ? 1.0f / (3.f * (wend - wstart)) : coef;
          for (int i = wstart; i < wend; i++) {
            tmp1 += (r0[i] + r1[i] + r2[i]);
          }
          dout_ch[w_even >> 1] = tmp1 * tmp2;
          // cnt ++;
        }
        r0 = r2;
        r1 = r0 + win;
        r2 = r1 + win;
        dout_ch += wout;
      }

      if (h_remain > 0) {
// deal with bottom pad
// first row with zero pad
// int hstart = (h >> 1) * stride_h - pad_h;
// int hend = std::min(std::min(hstart + kernel_h, hin + padding_h),
// hin); data_out_channel[0] =(r0[0] + r0[1] + r0[2] + r1[0] + r1[1] +
// r1[2]) / 9.f;
#ifdef __aarch64__
        int w = 0;
        int cnt = 0;
        for (; w < w_unroll_size; w += 8) {
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
          float32x4_t vrst = vmulq_f32(vsum_123_345, vcoef_6);
          vst1q_f32(&dout_ch[cnt], vrst);
          cnt += 4;
        }
        for (; w < w_even; w += 2) {
          float32x4_t vr0 = vld1q_f32(&r0[w]);
          float32x4_t vr1 = vld1q_f32(&r1[w]);
          vr0 = vsetq_lane_f32(0.f, vr0, 3);
          vr1 = vsetq_lane_f32(0.f, vr1, 3);
          float32x4_t vsum1 = vaddq_f32(vr0, vr1);
          float32x2_t vsum2 =
              vpadd_f32(vget_low_f32(vsum1), vget_high_f32(vsum1));
          vsum2 = vpadd_f32(vsum2, vsum2);
          float32x2_t vrst = vmul_f32(vsum2, vget_low_f32(vcoef_6));
          dout_ch[cnt] = vget_lane_f32(vrst, 0);
          cnt++;
        }
#else
        dr_out = dout_ch;  // + 1;
        dr0 = r0;          // (r0 + 1);
        dr1 = r1;          // (r1 + 1);
        int cnt_num = w_unroll_size >> 3;
        int cnt_num_remain = w_unroll_remain >> 1;
        // LOG(INFO) << "cnt_num: " << cnt_num << " cnt_num_remain: " <<
        // cnt_num_remain;
        if (cnt_num > 0 || cnt_num_remain > 0) {
          asm volatile(
              "cmp       %[cnt_num], #0           @cmp cnt_num,0\n"
              "ble       2f                       @ble exit\n"
              "1:                                 @main loop\n"
              "vld1.f32  {d0-d3}, [%[dr0]]!       @load d0-d5,dr0\n"
              "vld1.f32  {d6-d9}, [%[dr1]]!       @load d4-d7,dr1\n"
              "vld1.f32  {d4}, [%[dr0]]!          @load d0-d3,dr0\n"
              "vld1.f32  {d10}, [%[dr1]]!         @load d4-d7,dr1\n"
              "vadd.f32  q6, q0, q3               @max q0,q0,q2 1234\n"
              "vadd.f32  q7, q1, q4               @max q1,q1,q3 5678\n"
              "vadd.f32  d16, d4, d10             @max q1,q1,q3 9101112\n"
              //"vmov.f32  s7,s6                  @mov s7, s6\n"
              "vext.f32  q0, q6, q7, #1           @vext max_2345\n"
              "vext.f32  q1, q6, q7, #3           @vext max_4567\n"
              "vext.f32  q2, q6, q7, #2           @vext max_3456\n"
              "vext.f32  q3, q7, q8, #1           @vext max_6789\n"
              "vadd.f32  q4, q6, q0               @add 1234,2345\n"
              "vadd.f32  q5, q7, q1               @add 5678,4567\n"
              "vadd.f32  q4, q4, q2               @add 3456,sum1\n"
              "vadd.f32  q5, q5, q3               @add 6789,sum2\n"
              "vmov.f32  s17, s18                 @mov\n"
              "vmov.f32  s18, s21                 @mov\n"
              "vmov.f32  s19, s23                 @mov\n"
              "vmul.f32  q4, q4, %q[vcoef_6]      @mul\n"
              "sub       %[dr0], #8               @add w,8\n"
              "sub       %[dr1], #8               @add w,8\n"
              "subs      %[cnt_num], #1           @cnt_num--\n"
              "vst1.f32  d8, [%[dr_out]]!         @vst1 d0,dr_out\n"
              "vst1.f32  d9, [%[dr_out]]!         @vst1 d0,dr_out\n"
              "bne       1b                       @bne s3_max_loop_bot\n"
              "2:                                 @loop\n"
              "cmp       %[cnt_num_remain], #0    @cmp cnt_num_remain, 0\n"
              "ble       3f                       @ble exit\n"
              "4:                                 @bot loop\n"
              "vld1.f32  {d0-d1}, [%[dr0]]!       @load d0-d1,dr0\n"
              "vld1.f32  {d2-d3}, [%[dr1]]!       @load d2-d3,dr1\n"
              "vext.f32  q0, %q[vzero], q0, #3    @ext v0_0123\n"
              "vext.f32  q1, %q[vzero], q1, #3    @ext v1_0123\n"
              "vadd.f32  q0, q0, q1               @add q0,q0,q1\n"
              "vpadd.f32 d0, d0, d1               @padd d0,d0,d1\n"
              "vpadd.f32 d0, d0, d0               @padd d0,d0,d0\n"
              "vmul.f32  d0, d0, %e[vcoef_6]      @mul\n"
              "sub       %[dr0], #8               @add w,6\n"
              "sub       %[dr1], #8               @add w,6\n"
              "subs      %[cnt_num_remain], #1    @cnt_num_remain--\n"
              "vst1.f32  d0[0], [%[dr_out]]!      @vst d0[0],dr_out\n"
              "bne       4b                       @bne s3_max_loop_bot_1\n"
              "3:                                 @exit\n"
              : [dr0] "+r"(dr0), [dr1] "+r"(dr1), [dr_out] "+r"(dr_out),
                [cnt_num] "+r"(cnt_num), [cnt_num_remain] "+r"(cnt_num_remain),
                [vcoef_6] "+w"(vcoef_6), [vzero] "+w"(vzero)
              : "r"(dr0), "r"(dr1), "r"(dr_out), "r"(cnt_num),
                "r"(cnt_num_remain)
              : "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9");
        }

#endif
        if (w_remain > 0) {
          // deal with right pad
          int wstart = (w_even >> 1) * stride - padding;
          int wend = std::min(std::min(wstart + kernel, win + padding), win);
          float tmp1 = 0.f;
          float tmp2 = exclusive ? 1.0f / (2.f * (wend - wstart)) : coef;
          for (int i = wstart; i < wend; i++) {  // only run 1 or 2 times
            tmp1 += (r0[i] + r1[i]);
          }
          dout_ch[w_even >> 1] = tmp1 * tmp2;
        }
      }
    }
  }
}

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
