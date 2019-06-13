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

#include "paddle/fluid/lite/arm/math/conv_impl.h"
#include "paddle/fluid/lite/arm/math/packed_sgemm.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {

#if 1
void transpose(float* data_out, const float* data_in, int w_in, int h_in);
void transform_input_f6x6(float* dout, const float* din);
void transform_output_f6x6(float* output, const float* din, float bias);
#if 0
ConvWinogradF63::ConvWinogradF63() {

}

ConvWinogradF63::~ConvWinogradF63() {

}

bool ConvWinogradF63::init(const size_t l1_cache, const size_t l2_cache, \
    const int chout, const int chin, const int hin, \
    const int win, const int threads) {

    return true;
}

bool ConvWinogradF63::operator()(const float *trans_weights, const float *din, \
    float *dout, void *workspace) {

    return true;
}
#endif
void conv_winograd3x3(const float* din, float* dout, int num, int chout,
                      int hout, int wout, int chin, int hin, int win,
                      const float* weights, const float* bias,
                      const operators::ConvParam& param, ARMContext* ctx) {
  int threads = ctx->threads();

  const int pad_h = param.paddings[0];
  const int pad_w = param.paddings[1];
  int size_in_channel = win * hin;
  int size_out_channel = wout * hout;
  bool flag_relu = false;
  bool flag_bias = param.bias != nullptr;
//   if (param.activation_param.has_active) {
//     if (param.activation_param.active == Active_relu &&
//         fabs(param.activation_param.negative_slope) < 1e-6f) {
//       flag_relu = true;
//     }
//   }

  //! transform input
  int tile_w = (wout + 5) / 6;
  int tile_h = (hout + 5) / 6;
  int size_tile = tile_h * tile_w;
  int size_trans_channel = 8 * 8 * size_tile;
  int max_ch = chin > chout ? chin : chout;

  int m = chout;
  int n = size_tile;
  int k = chin;

  float* tmp_work_space = ctx->workspace_data<float>() +
                          ctx->l2_cache_size() / sizeof(float);

  //! tmp data buffer for input transform
  float* tmp_data1 = tmp_work_space;
  //! tmp data buffer for dot mul
  float* tmp_data2 = tmp_data1 + size_trans_channel * max_ch;

  // SaberTimer<ARM> t1;
  // Context<ARM> ctx1;

  for (int i = 0; i < num; ++i) {
    const float* din_batch = din + i * chin * size_in_channel;
    float* dout_batch = dout + i * chout * size_out_channel;

    // t1.start(ctx1);
    //! transform input Bt * data * B
#pragma omp parallel for num_threads(threads)
    for (int j = 0; j < chin; ++j) {
      const float* din_channel = din_batch + j * size_in_channel;
      float* data_trans_channel = tmp_data1 + j * size_trans_channel;

      for (int h = 0; h < tile_h; h++) {
        for (int w = 0; w < tile_w; w++) {
          //! prepare data 8x8
          //! row 8
          float data_in_tmp[8][8] = {0.f};
          // memset(data_in_tmp[0], 0, sizeof(float) * 64);
          for (int j = 0; j < 8; ++j) {
            int start_row = h * 6 + j - pad_h;
            if (start_row >= 0 && start_row < hin) {
              for (int k = 0; k < 8; ++k) {
                int start_col = w * 6 + k - pad_w;
                if (start_col >= 0 && start_col < win) {
                  data_in_tmp[j][k] = din_channel[start_row * win + start_col];
                }
              }
            }
          }
          transform_input_f6x6(data_trans_channel, data_in_tmp[0]);
          data_trans_channel += 64;
        }
      }
    }
    //! end of transform input

#if 1
    ////////////////////////////////////////////////////////////////////////////////
    //! dot mul
    //! transpose input, convert from ch_in * tile_h * tile_w * 64 to
    //! 64 * ch_in * tile_h * tile_w
    int hblock = get_hblock(ctx->arch());
    int m_round = hblock * ((chout + hblock - 1) / hblock);
    int stride_a = m_round * chin;
    int stride_b = chin * size_tile;
    int stride_c = chout * size_tile;
    transpose(tmp_data2, tmp_data1, 64, stride_b);

    // t1.end(ctx1);
    // LOG(INFO) << "winograd conv transform input time: " <<
    // t1.get_average_ms();

    // t1.clear();
    // t1.start(ctx1);

    //! gemm
    //#pragma omp parallel for
    for (int l = 0; l < 64; ++l) {
      const float* ptr_a = weights + l * stride_a;
      const float* ptr_b = tmp_data2 + l * stride_b;
      float* ptr_c = tmp_data1 + l * stride_c;
      sgemm_prepack(ptr_a, ptr_b, nullptr, ptr_c, chout, size_tile, chin, false,
                    false, false, ctx);
      // gemmer(ptr_a, chin, ptr_b, size_tile, ptr_c, size_tile, 1.f, 0.f,
      // false);
    }

    //! transpose output, convert from 64 * ch_out * tile_h * tile_w to
    //! ch_out * tile_h * tile_w * 64
    transpose(tmp_data2, tmp_data1, stride_c, 64);
    //! end of dot mul
#endif
    // t1.end(ctx1);
    // LOG(INFO) << "winograd conv dot mul time: " << t1.get_average_ms();

    // t1.clear();
    // t1.start(ctx1);
#if 1
    ///////////////////////////////////////////////////////////////////////////////
    //! transform output
#pragma omp parallel for
    for (int i = 0; i < chout; ++i) {
      float bias_value = flag_bias ? bias[i] : 0.f;
      float* dout_tmp = tmp_data2 + i * size_trans_channel;
      float* dout_channel = dout_batch + i * size_out_channel;

      for (int h = 0; h < tile_h; ++h) {
        for (int w = 0; w < tile_w; ++w) {
          float out_tmp[6][6];

          transform_output_f6x6(out_tmp[0], dout_tmp, bias_value);
          dout_tmp += 64;

          for (int j = 0; j < 6; ++j) {
            int end_row = h * 6 + j;
            if (end_row < hout) {
              for (int k = 0; k < 6; ++k) {
                int end_col = w * 6 + k;
                if (end_col < wout) {
                  if (flag_relu) {
                    dout_channel[end_row * wout + end_col] =
                        out_tmp[j][k] > 0.f ? out_tmp[j][k] : 0.f;
                  } else {
                    dout_channel[end_row * wout + end_col] = out_tmp[j][k];
                  }
                }
              }
            }
          }
        }
      }
    }
    //! end of transform output
#endif
    // t1.end(ctx1);
    // LOG(INFO) << "winograd conv transform output time: " <<
    // t1.get_average_ms();
  }
}

/**
 * \brief transpose with arm neon optimization
 * @param data_out
 * @param data_in
 * @param w_in
 * @param h_in
 */
void transpose(float* data_out, const float* data_in, int w_in, int h_in) {
  int nw = w_in >> 2;
  int nh = h_in >> 2;
  int size_in = w_in * h_in;

  float* ptr_out = data_out;
  const float* ptr_in = data_in;
#pragma omp parallel for
  for (int h = 0; h < nh; h++) {
    const float* ptr_din_row = ptr_in + h * 4 * w_in;
    for (int w = 0; w < nw; w++) {
      float* data_out_ptr = ptr_out + w * 4 * h_in + h * 4;
      const float* din0 = ptr_din_row;
      const float* din1 = din0 + w_in;
      const float* din2 = din1 + w_in;
      const float* din3 = din2 + w_in;

      float* dout0 = data_out_ptr;
      float* dout1 = dout0 + h_in;
      float* dout2 = dout1 + h_in;
      float* dout3 = dout2 + h_in;
#ifdef __aarch64__
      asm("ldr    q0, [%[in0]]                                            \n" /*load input 0*/
          "ldr    q1, [%[in1]]                                \n"
          "ldr    q2, [%[in2]]                                \n"
          "ldr    q3, [%[in3]]                                \n"
          "trn1   v4.4s, v0.4s, v1.4s                         \n"
          "trn2   v5.4s, v0.4s, v1.4s                         \n"
          "trn1   v6.4s, v2.4s, v3.4s                         \n"
          "trn2   v7.4s, v2.4s, v3.4s                         \n"
          "trn1   v8.2d, v4.2d, v6.2d                         \n"
          "trn1   v9.2d, v5.2d, v7.2d                         \n"
          "trn2   v10.2d, v4.2d, v6.2d                        \n"
          "trn2   v11.2d, v5.2d, v7.2d                        \n"
          "str    q8, [%[out0]]                               \n"
          "str    q9, [%[out1]]                               \n"
          "str   q10, [%[out2]]                               \n"
          "str   q11, [%[out3]]                               \n"
          :
          : [out0] "r"(dout0), [out1] "r"(dout1), [out2] "r"(dout2),
            [out3] "r"(dout3), [in0] "r"(din0), [in1] "r"(din1),
            [in2] "r"(din2), [in3] "r"(din3)
          : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
            "v11");
#else
      asm("vld1.32 {d0, d1}, [%[in0]]    \n"
          "vld1.32 {d2, d3}, [%[in1]]    \n"
          "vld1.32 {d4, d5}, [%[in2]]    \n"
          "vld1.32 {d6, d7}, [%[in3]]    \n"
          "vtrn.32 q0, q1                \n"
          "vtrn.32 q2, q3                \n"
          "vswp d1, d4                   \n"
          "vswp d3, d6                   \n"
          "vst1.32 {d0, d1}, [%[out0]]   \n"
          "vst1.32 {d2, d3}, [%[out1]]   \n"
          "vst1.32 {d4, d5}, [%[out2]]   \n"
          "vst1.32 {d6, d7}, [%[out3]]   \n"
          :
          : [out0] "r"(dout0), [out1] "r"(dout1), [out2] "r"(dout2),
            [out3] "r"(dout3), [in0] "r"(din0), [in1] "r"(din1),
            [in2] "r"(din2), [in3] "r"(din3)
          : "q0", "q1", "q2", "q3");
#endif
      ptr_din_row += 4;
    }
  }
  // remian
  for (int h = 0; h < h_in; h++) {
    for (int w = nw * 4; w < w_in; w++) {
      const float* data_in_ptr = ptr_in + h * w_in + w;
      float* data_out_ptr = ptr_out + w * h_in + h;
      *data_out_ptr = *data_in_ptr;
    }
  }
  for (int w = 0; w < w_in; w++) {
    for (int h = nh * 4; h < h_in; h++) {
      const float* data_in_ptr = ptr_in + h * w_in + w;
      float* data_out_ptr = ptr_out + w * h_in + h;
      *data_out_ptr = *data_in_ptr;
    }
  }
}

/**
 * \brief winograd transform conv3x3 weights, f63
 * this is done in op initialization or creation, only do once
 * dout = G * g * GT, where G is the transform coeff, g is the input weights
 * @param dout
 * @param din
 * @param ch_out
 * @param ch_in
 * @param work_space
 */
void winograd_transform_weights(void* dout, const void* din, int ch_out,
                                int ch_in, void* work_space) {
  const float coeff[8][3] = {{1.0f, 0.0f, 0.0f},
                             {-2.0f / 9, -2.0f / 9, -2.0f / 9},
                             {-2.0f / 9, 2.0f / 9, -2.0f / 9},
                             {1.0f / 90, 1.0f / 45, 2.0f / 45},
                             {1.0f / 90, -1.0f / 45, 2.0f / 45},
                             {32.0f / 45, 16.0f / 45, 8.0f / 45},
                             {32.0f / 45, -16.0f / 45, 8.0f / 45},
                             {0.0f, 0.0f, 1.0f}};

  float* ptr_out = (float*)work_space;

  for (int i = 0; i < ch_out; i++) {
    for (int j = 0; j < ch_in; j++) {
      const float* kernel0 =
          static_cast<const float*>(din) + (i * ch_in + j) * 9;
      float* ptr_channel = ptr_out + (i * ch_in + j) * 64;

      //! transform kernel, transposed
      const float* k0 = kernel0;
      const float* k1 = kernel0 + 3;
      const float* k2 = kernel0 + 6;

      //! h
      float tmp[8][3];
      for (int i = 0; i < 8; i++) {
        tmp[i][0] =
            k0[0] * coeff[i][0] + k0[1] * coeff[i][1] + k0[2] * coeff[i][2];
        tmp[i][1] =
            k1[0] * coeff[i][0] + k1[1] * coeff[i][1] + k1[2] * coeff[i][2];
        tmp[i][2] =
            k2[0] * coeff[i][0] + k2[1] * coeff[i][1] + k2[2] * coeff[i][2];
      }

      //! v
      for (int j = 0; j < 8; j++) {
        float* tmpp = &tmp[j][0];
        for (int i = 0; i < 8; i++) {
          ptr_channel[j * 8 + i] = tmpp[0] * coeff[i][0] +
                                   tmpp[1] * coeff[i][1] +
                                   tmpp[2] * coeff[i][2];
        }
      }
    }
  }
  transpose(static_cast<float*>(dout), ptr_out, 64, ch_out * ch_in);
}

/**
 * \brief winograd conv, transform input, f6x3
 * dout = BT * d * B, whrer B is the transform
 * BT = 1      0   -21/4       0     21/4        0   -1   0
 *      0      1       1   -17/4    -17/4        1    1   0
 *      0     -1       1    17/4    -17/4       -1    1   0
 *      0    1/2     1/4    -5/2     -5/4        2    1   0
 *      0   -1/2     1/4     5/2     -5/4       -2    1   0
 *      0      2       4    -5/2       -5      1/2    1   0
 *      0     -2       4     5/2       -5     -1/2    1   0
 *      0     -1       0    21/4        0    -21/4    0   1
 * @param dout
 * @param din
 */
void transform_input_f6x6(float* dout, const float* din) {
  float tmp[8][8];
  //! BT * d
  for (int m = 0; m < 8; m++) {
    tmp[0][m] = din[0] - din[6] + (din[4] - din[2]) * 5.25f;
    tmp[7][m] = din[7] - din[1] + (din[3] - din[5]) * 5.25f;

    float tmp12a = din[2] + din[6] - din[4] * 4.25f;
    float tmp12b = din[1] + din[5] - din[3] * 4.25f;

    tmp[1][m] = tmp12a + tmp12b;
    tmp[2][m] = tmp12a - tmp12b;

    float tmp34a = din[6] + din[2] * 0.25f - din[4] * 1.25f;
    float tmp34b = din[1] * 0.5f - din[3] * 2.5f + din[5] * 2.f;

    tmp[3][m] = tmp34a + tmp34b;
    tmp[4][m] = tmp34a - tmp34b;

    float tmp56a = din[6] + (din[2] - din[4] * 1.25f) * 4.f;
    float tmp56b = din[1] * 2.f - din[3] * 2.5f + din[5] * 0.5f;

    tmp[5][m] = tmp56a + tmp56b;
    tmp[6][m] = tmp56a - tmp56b;

    din += 8;
  }

  for (int m = 0; m < 8; m++) {
    const float* tmp0 = tmp[m];

    dout[0] = tmp0[0] - tmp0[6] + (tmp0[4] - tmp0[2]) * 5.25f;
    dout[7] = tmp0[7] - tmp0[1] + (tmp0[3] - tmp0[5]) * 5.25f;

    float tmp12a = tmp0[2] + tmp0[6] - tmp0[4] * 4.25f;
    float tmp12b = tmp0[1] + tmp0[5] - tmp0[3] * 4.25f;

    dout[1] = tmp12a + tmp12b;
    dout[2] = tmp12a - tmp12b;

    float tmp34a = tmp0[6] + tmp0[2] * 0.25f - tmp0[4] * 1.25f;
    float tmp34b = tmp0[1] * 0.5f - tmp0[3] * 2.5f + tmp0[5] * 2.f;

    dout[3] = tmp34a + tmp34b;
    dout[4] = tmp34a - tmp34b;

    float tmp56a = tmp0[6] + (tmp0[2] - tmp0[4] * 1.25f) * 4.f;
    float tmp56b = tmp0[1] * 2.f - tmp0[3] * 2.5f + tmp0[5] * 0.5f;

    dout[5] = tmp56a + tmp56b;
    dout[6] = tmp56a - tmp56b;

    dout += 8;
  }
}

/**
 * \brief winograd conv, transform output, f63
 * out = AT * din * A
 * AT = 1      1       1       1        1        1        1   0
 *      0      1      -1       2       -2      1/2     -1/2   0
 *      0      1       1       4        4      1/4      1/4   0
 *      0      1      -1       8       -8      1/8     -1/8   0
 *      0      1       1      16       16     1/16     1/16   0
 *      0      1      -1      32      -32     1/32    -1/32   1
 * @param output
 * @param din
 * @param bias
 */
void transform_output_f6x6(float* output, const float* din, float bias) {
  float tmp[6][8];
  for (int m = 0; m < 8; m++) {
    float tmp024a = din[1] + din[2];
    float tmp135a = din[1] - din[2];

    float tmp024b = din[3] + din[4];
    float tmp135b = din[3] - din[4];

    float tmp024c = din[5] + din[6];
    float tmp135c = din[5] - din[6];

    tmp[0][m] = din[0] + tmp024a + tmp024b + tmp024c;
    tmp[2][m] = tmp024a + tmp024b * 4 + tmp024c * 0.25f;
    tmp[4][m] = tmp024a + tmp024b * 16 + tmp024c * 0.0625f;

    tmp[1][m] = tmp135a + tmp135b * 2 + tmp135c * 0.5f;
    tmp[3][m] = tmp135a + tmp135b * 8 + tmp135c * 0.125f;
    tmp[5][m] = din[7] + tmp135a + tmp135b * 32 + tmp135c * 0.03125f;

    din += 8;
  }

  for (int m = 0; m < 6; m++) {
    const float* tmp0 = tmp[m];

    float tmp024a = tmp0[1] + tmp0[2];
    float tmp135a = tmp0[1] - tmp0[2];

    float tmp024b = tmp0[3] + tmp0[4];
    float tmp135b = tmp0[3] - tmp0[4];

    float tmp024c = tmp0[5] + tmp0[6];
    float tmp135c = tmp0[5] - tmp0[6];

    output[0] = bias + tmp0[0] + tmp024a + tmp024b + tmp024c;
    output[2] = bias + tmp024a + tmp024b * 4 + tmp024c * 0.25f;
    output[4] = bias + tmp024a + tmp024b * 16 + tmp024c * 0.0625f;

    output[1] = bias + tmp135a + tmp135b * 2 + tmp135c * 0.5f;
    output[3] = bias + tmp135a + tmp135b * 8 + tmp135c * 0.125f;
    output[5] = bias + tmp0[7] + tmp135a + tmp135b * 32 + tmp135c * 0.03125f;

    output += 6;
  }
}
#endif

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
