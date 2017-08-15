/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#if defined(__ARM_NEON__) || defined(__ARM_NEON)
#define HAVE_NEON
#include <arm_neon.h>
#endif

namespace paddle {

namespace neon {

template <int filterSize, int stride>
struct DepthwiseConvNeonKernel {};

#ifdef HAVE_NEON

template <>
struct DepthwiseConvNeonKernel<3, 1> {
  static void run(const float* inputPaddedData,
                  const float* filterData,
                  int batchSize,
                  int outputChannels,
                  int outputHeight,
                  int outputWidth,
                  int inputChannels,
                  int inputPaddedHeight,
                  int inputPaddedWidth,
                  int filterMultiplier,
                  int filterSize,
                  int stride,
                  float* outputData) {
    // check filterSize and stride
    int w = inputPaddedWidth;
    int h = inputPaddedHeight;
    int inch = inputChannels;

    int outw = outputWidth;
    int outh = outputHeight;
    int outch = outputChannels;

    const float* kernel = filterData;
    for (int num = 0; num < batchSize; num++) {
      for (int p = 0; p < outch; p++) {
        float* out = outputData + (num * outch + p) * (outw * outh);
        const float* kernel0 = kernel + p * 9;

        {
          int q = p;
          float* outptr = out;
          const float* img0 = inputPaddedData + (num * inch + q) * (w * h);

          const float* r0 = img0;
          const float* r1 = img0 + w;
          const float* r2 = img0 + w * 2;

          float32x4_t _k0123 = vld1q_f32(kernel0);
          float32x4_t _k3456 = vld1q_f32(kernel0 + 3);
          float32x4_t _k6789 = vld1q_f32(kernel0 + 6);

          int i = 0;

          for (; i < outh; i++) {
            int nn = outw >> 2;
            int remain = outw & 3;
#if __aarch64__
            for (; nn > 0; nn--) {
              // float32x4_t _sum1 = vld1q_f32(outptr);
              float32x4_t _sum1 = vdupq_n_f32(0.f);
              float32x4_t _sum2 = vdupq_n_f32(0.f);

              float32x4_t _r00 = vld1q_f32(r0);
              float32x4_t _r00n = vld1q_f32(r0 + 4);
              float32x4_t _r01 = vextq_f32(_r00, _r00n, 1);
              float32x4_t _r02 = vextq_f32(_r00, _r00n, 2);

              float32x4_t _r10 = vld1q_f32(r1);
              float32x4_t _r10n = vld1q_f32(r1 + 4);
              float32x4_t _r11 = vextq_f32(_r10, _r10n, 1);
              float32x4_t _r12 = vextq_f32(_r10, _r10n, 2);

              float32x4_t _r20 = vld1q_f32(r2);
              float32x4_t _r20n = vld1q_f32(r2 + 4);
              float32x4_t _r21 = vextq_f32(_r20, _r20n, 1);
              float32x4_t _r22 = vextq_f32(_r20, _r20n, 2);

              _sum1 = vfmaq_laneq_f32(_sum1, _r00, _k0123, 0);
              _sum2 = vfmaq_laneq_f32(_sum2, _r01, _k0123, 1);
              _sum1 = vfmaq_laneq_f32(_sum1, _r02, _k0123, 2);
              _sum2 = vfmaq_laneq_f32(_sum2, _r10, _k3456, 0);
              _sum1 = vfmaq_laneq_f32(_sum1, _r11, _k3456, 1);
              _sum2 = vfmaq_laneq_f32(_sum2, _r12, _k3456, 2);
              _sum1 = vfmaq_laneq_f32(_sum1, _r20, _k6789, 0);
              _sum2 = vfmaq_laneq_f32(_sum2, _r21, _k6789, 1);
              _sum1 = vfmaq_laneq_f32(_sum1, _r22, _k6789, 2);

              _sum1 = vaddq_f32(_sum1, _sum2);

              vst1q_f32(outptr, _sum1);

              r0 += 4;
              r1 += 4;
              r2 += 4;
              outptr += 4;
            }
#else
            if (nn > 0) {
              asm volatile(
                  "pld        [%2, #192]          \n"
                  "vld1.f32   {d16-d18}, [%2]     \n"  // r0
                  "add        %2, #16             \n"

                  "veor       q13, q13            \n"
                  "veor       q14, q14            \n"

                  "vext.32    q10, q8, q9, #1     \n"
                  "vext.32    q11, q8, q9, #2     \n"

                  "0:                             \n"

                  "veor       q7, q7             \n"

                  "vmla.f32   q7, q8, %e10[0]     \n"
                  "vmla.f32   q13, q10, %e10[1]   \n"
                  "vmla.f32   q14, q11, %f10[0]   \n"

                  "pld        [%3, #192]          \n"
                  "vld1.f32   {d16-d18}, [%3]     \n"  // r1
                  "add        %3, #16             \n"

                  "vmla.f32   q7, q8, %e11[0]     \n"

                  "vext.32    q10, q8, q9, #1     \n"
                  "vext.32    q11, q8, q9, #2     \n"

                  "vmla.f32   q13, q10, %e11[1]   \n"
                  "vmla.f32   q14, q11, %f11[0]   \n"

                  "pld        [%4, #192]          \n"
                  "vld1.f32   {d16-d18}, [%4]     \n"  // r2
                  "add        %4, #16             \n"

                  "vmla.f32   q7, q8, %e12[0]     \n"

                  "vext.32    q10, q8, q9, #1     \n"
                  "vext.32    q11, q8, q9, #2     \n"

                  "vmla.f32   q13, q10, %e12[1]   \n"
                  "vmla.f32   q14, q11, %f12[0]   \n"

                  "pld        [%2, #192]          \n"
                  "vld1.f32   {d16-d18}, [%2]     \n"  // r0
                  "add        %2, #16             \n"

                  "vadd.f32   q7, q7, q13         \n"
                  "veor       q13, q13            \n"
                  "vadd.f32   q7, q7, q14         \n"
                  "veor       q14, q14            \n"

                  "vext.32    q10, q8, q9, #1     \n"
                  "vext.32    q11, q8, q9, #2     \n"

                  "vst1.f32   {d14-d15}, [%1]!    \n"

                  "subs       %0, #1              \n"
                  "bne        0b                  \n"

                  "sub        %2, #16             \n"
                  : "=r"(nn),      // %0
                    "=r"(outptr),  // %1
                    "=r"(r0),      // %2
                    "=r"(r1),      // %3
                    "=r"(r2)       // %4
                  : "0"(nn),
                    "1"(outptr),
                    "2"(r0),
                    "3"(r1),
                    "4"(r2),
                    "w"(_k0123),  // %10
                    "w"(_k3456),  // %11
                    "w"(_k6789)   // %12
                  : "cc",
                    "memory",
                    "q7",
                    "q8",
                    "q9",
                    "q10",
                    "q11",
                    "q12",
                    "q13",
                    "q14",
                    "q15");
            }
#endif  // __aarch64__
            for (; remain > 0; remain--) {
              float32x4_t _r00 = vld1q_f32(r0);
              float32x4_t _r10 = vld1q_f32(r1);
              float32x4_t _r20 = vld1q_f32(r2);

              float32x4_t _sum = vmulq_f32(_r00, _k0123);
              _sum = vmlaq_f32(_sum, _r10, _k3456);
              _sum = vmlaq_f32(_sum, _r20, _k6789);

              _sum = vsetq_lane_f32(0.f, _sum, 3);

#if __aarch64__
              *outptr = vaddvq_f32(_sum);
#else
              float32x2_t _ss =
                  vadd_f32(vget_low_f32(_sum), vget_high_f32(_sum));
              _ss = vpadd_f32(_ss, _ss);

              *outptr = vget_lane_f32(_ss, 0);
#endif  // __aarch64__

              r0++;
              r1++;
              r2++;
              outptr++;
            }

            r0 += 2;
            r1 += 2;
            r2 += 2;
          }
          kernel0 += 9;
        }
      }
    }
  }
};

template <>
struct DepthwiseConvNeonKernel<3, 2> {
  static void run(const float* inputPaddedData,
                  const float* filterData,
                  int batchSize,
                  int outputChannels,
                  int outputHeight,
                  int outputWidth,
                  int inputChannels,
                  int inputPaddedHeight,
                  int inputPaddedWidth,
                  int filterMultiplier,
                  int filterSize,
                  int stride,
                  float* outputData) {
    // do check filterSize == 3 and stride == 2
    int w = inputPaddedWidth;
    int h = inputPaddedHeight;
    int inch = inputChannels;

    int outw = outputWidth;
    int outh = outputHeight;
    int outch = outputChannels;

    const int tailstep = w - 2 * outw + w;

    const float* kernel = filterData;

    for (int num = 0; num < batchSize; num++) {
      for (int p = 0; p < outch; p++) {
        float* out = outputData + (num * outch + p) * (outw * outh);
        // float* out = top_blob.data + top_blob.alignDimSize * p;
        const float* kernel0 = kernel + p * 9;

        {
          int q = p;
          float* outptr = out;
          const float* img0 = inputPaddedData + (num * inch + q) * (w * h);

          const float* r0 = img0;
          const float* r1 = img0 + w;
          const float* r2 = img0 + w * 2;

          float32x4_t _k0123 = vld1q_f32(kernel0);
          float32x4_t _k3456 = vld1q_f32(kernel0 + 3);
          float32x4_t _k6789 = vld1q_f32(kernel0 + 6);

          int i = 0;

          for (; i < outh; i++) {
            int nn = outw >> 2;
            int remain = outw & 3;

#if __aarch64__
            for (; nn > 0; nn--) {
              float32x4_t _outp = vdupq_n_f32(0.f);

              float32x4x2_t _r0 = vld2q_f32(r0);
              float32x4x2_t _r0n = vld2q_f32(r0 + 8);

              float32x4_t _r00 = _r0.val[0];                       // 0 2 4 6
              float32x4_t _r01 = _r0.val[1];                       // 1 3 5 7
              float32x4_t _r02 = vextq_f32(_r00, _r0n.val[0], 1);  // 2 4 6 8

              _outp = vfmaq_laneq_f32(_outp, _r00, _k0123, 0);
              _outp = vfmaq_laneq_f32(_outp, _r01, _k0123, 1);
              _outp = vfmaq_laneq_f32(_outp, _r02, _k0123, 2);

              float32x4x2_t _r1 = vld2q_f32(r1);
              float32x4x2_t _r1n = vld2q_f32(r1 + 8);

              float32x4_t _r10 = _r1.val[0];
              float32x4_t _r11 = _r1.val[1];
              float32x4_t _r12 = vextq_f32(_r10, _r1n.val[0], 1);

              _outp = vfmaq_laneq_f32(_outp, _r10, _k3456, 0);
              _outp = vfmaq_laneq_f32(_outp, _r11, _k3456, 1);
              _outp = vfmaq_laneq_f32(_outp, _r12, _k3456, 2);

              float32x4x2_t _r2 = vld2q_f32(r2);
              float32x4x2_t _r2n = vld2q_f32(r2 + 8);

              float32x4_t _r20 = _r2.val[0];
              float32x4_t _r21 = _r2.val[1];
              float32x4_t _r22 = vextq_f32(_r20, _r2n.val[0], 1);

              _outp = vfmaq_laneq_f32(_outp, _r20, _k6789, 0);
              _outp = vfmaq_laneq_f32(_outp, _r21, _k6789, 1);
              _outp = vfmaq_laneq_f32(_outp, _r22, _k6789, 2);

              vst1q_f32(outptr, _outp);

              r0 += 8;
              r1 += 8;
              r2 += 8;
              outptr += 4;
            }
#else
            if (nn > 0) {
              asm volatile(
                  "pld        [%2, #256]          \n"
                  "vld2.f32   {d4-d7}, [%2]!      \n"

                  "veor       q10, q10            \n"
                  "veor       q11, q11            \n"

                  "0:                             \n"
                  //  "pld        [%1, #128]          \n"
                  //  "vld1.f32   {d0-d1}, [%1]       \n"
                  "veor       q0, q0            \n"

                  "vmla.f32   q0, q2, %e10[0]     \n"
                  "vmla.f32   q10, q3, %e10[1]    \n"

                  "pld        [%2, #256]          \n"
                  "vld2.f32   {d16-d19}, [%2]     \n"
                  "vext.32    q1, q2, q8, #1      \n"

                  "vmla.f32   q11, q1, %f10[0]    \n"

                  "pld        [%3, #256]          \n"
                  "vld2.f32   {d4-d7}, [%3]!      \n"

                  "vmla.f32   q0, q2, %e11[0]     \n"
                  "vmla.f32   q10, q3, %e11[1]    \n"

                  "pld        [%3, #256]          \n"
                  "vld2.f32   {d16-d19}, [%3]     \n"
                  "vext.32    q1, q2, q8, #1      \n"

                  "vmla.f32   q11, q1, %f11[0]    \n"

                  "pld        [%4, #256]          \n"
                  "vld2.f32   {d4-d7}, [%4]!      \n"

                  "vmla.f32   q0, q2, %e12[0]     \n"
                  "vmla.f32   q10, q3, %e12[1]    \n"

                  "pld        [%4, #256]          \n"
                  "vld2.f32   {d16-d19}, [%4]     \n"
                  "vext.32    q1, q2, q8, #1      \n"

                  "vmla.f32   q11, q1, %f12[0]    \n"

                  "pld        [%2, #256]          \n"
                  "vld2.f32   {d4-d7}, [%2]!      \n"

                  "vadd.f32   q0, q0, q10         \n"
                  "veor       q10, q10            \n"
                  "vadd.f32   q0, q0, q11         \n"
                  "veor       q11, q11            \n"

                  "subs       %0, #1              \n"
                  "vst1.f32   {d0-d1}, [%1]!      \n"
                  "bne        0b                  \n"
                  "sub        %2, #32             \n"
                  : "=r"(nn),      // %0
                    "=r"(outptr),  // %1
                    "=r"(r0),      // %2
                    "=r"(r1),
                    "=r"(r2)
                  : "0"(nn),
                    "1"(outptr),
                    "2"(r0),
                    "3"(r1),
                    "4"(r2),
                    "w"(_k0123),  // %10
                    "w"(_k3456),  // %11
                    "w"(_k6789)   // %12
                  : "cc",
                    "memory",
                    "q0",
                    "q1",
                    "q2",
                    "q3",
                    "q8",
                    "q9",
                    "q10",
                    "q11",
                    "q12",
                    "q13",
                    "q14",
                    "q15");
            }
#endif  // __aarch64__
            for (; remain > 0; remain--) {
              float32x4_t _r00 = vld1q_f32(r0);
              float32x4_t _r10 = vld1q_f32(r1);
              float32x4_t _r20 = vld1q_f32(r2);

              float32x4_t _sum = vmulq_f32(_r00, _k0123);
              _sum = vmlaq_f32(_sum, _r10, _k3456);
              _sum = vmlaq_f32(_sum, _r20, _k6789);

              _sum = vsetq_lane_f32(0.f, _sum, 3);

#if __aarch64__
              *outptr = vaddvq_f32(_sum);
#else
              float32x2_t _ss =
                  vadd_f32(vget_low_f32(_sum), vget_high_f32(_sum));
              _ss = vpadd_f32(_ss, _ss);

              *outptr = vget_lane_f32(_ss, 0);
#endif  // __aarch64__

              r0 += 2;
              r1 += 2;
              r2 += 2;
              outptr++;
            }

            r0 += tailstep;
            r1 += tailstep;
            r2 += tailstep;
          }
          kernel0 += 9;
        }
      }
    }
  }
};

#endif  // HAVE_NEON

}  // namespace neon
}  // namespace paddle
