#include "paddle/fluid/lite/arm/math/conv_depthwise.h"
#include <arm_neon.h>

namespace paddle {
namespace lite {
namespace arm {
namespace math {

void conv_depthwise_3x3s1p0_bias(float* dout, const float* din, \
    const float* weights, const float* bias, bool flag_bias, \
    const int num, const int ch_in, const int h_in, const int w_in, \
    const int h_out, const int w_out, ARMContext* ctx);

//! for input width <= 4
void conv_depthwise_3x3s1p0_bias_s(float* dout, const float* din, \
    const float* weights, const float* bias, bool flag_bias, \
    const int num, const int ch_in, const int h_in, const int w_in, \
    const int h_out, const int w_out, ARMContext* ctx);

void conv_depthwise_3x3s2p0_bias(float* dout, const float* din, \
    const float* weights, const float* bias, bool flag_bias, \
    const int num, const int ch_in, const int h_in, const int w_in, \
    const int h_out, const int w_out, ARMContext* ctx);

//! for input width <= 4
void conv_depthwise_3x3s2p0_bias_s(float* dout, const float* din, \
    const float* weights, const float* bias, bool flag_bias, \
    const int num, const int ch_in, const int h_in, const int w_in, \
    const int h_out, const int w_out, ARMContext* ctx);

void conv_depthwise_3x3s1p0_bias_relu(float* dout, const float* din, \
    const float* weights, const float* bias, bool flag_bias, \
    const int num, const int ch_in, const int h_in, const int w_in, \
    const int h_out, const int w_out, ARMContext* ctx);

//! for input width <= 4
void conv_depthwise_3x3s1p0_bias_s_relu(float* dout, const float* din, \
    const float* weights, const float* bias, bool flag_bias, \
    const int num, const int ch_in, const int h_in, const int w_in, \
    const int h_out, const int w_out, ARMContext* ctx);

void conv_depthwise_3x3s2p0_bias_relu(float* dout, const float* din, \
    const float* weights, const float* bias, bool flag_bias, \
    const int num, const int ch_in, const int h_in, const int w_in, \
    const int h_out, const int w_out, ARMContext* ctx);

//! for input width <= 4
void conv_depthwise_3x3s2p0_bias_s_relu(float* dout, const float* din, \
    const float* weights, const float* bias, bool flag_bias, \
    const int num, const int ch_in, const int h_in, const int w_in, \
    const int h_out, const int w_out, ARMContext* ctx);

void conv_depthwise_3x3p0(const float* din, float* dout, \
                          int num, int ch_out, int h_out, int w_out, \
                          int ch_in, int h_in, int w_in, \
                          const float* weights, const float* bias, \
                          int stride, bool flag_bias, bool flag_relu, ARMContext* ctx){
    if (stride == 1) {
        if (flag_relu) {
            if (w_in > 5) {
                conv_depthwise_3x3s1p0_bias_relu(dout, din, weights, bias, flag_bias, \
                    num, ch_in, h_in, w_in, h_out, w_out, ctx);
            } else {
                conv_depthwise_3x3s1p0_bias_s_relu(dout, din, weights, bias, flag_bias, \
                    num, ch_in, h_in, w_in, h_out, w_out, ctx);
            }
        } else {
            if (w_in > 5) {
                conv_depthwise_3x3s1p0_bias(dout, din, weights, bias, flag_bias, \
                    num, ch_in, h_in, w_in, h_out, w_out, ctx);
            } else {
                conv_depthwise_3x3s1p0_bias_s(dout, din, weights, bias, flag_bias, \
                    num, ch_in, h_in, w_in, h_out, w_out, ctx);
            }

        }
    } else { //! stride = 2
        if (flag_relu) {
            if (w_in > 8){
                conv_depthwise_3x3s2p0_bias_relu(dout, din, weights, bias, flag_bias, \
                    num, ch_in, h_in, w_in, h_out, w_out, ctx);
            }else{
                conv_depthwise_3x3s2p0_bias_s_relu(dout, din, weights, bias, flag_bias, \
                    num, ch_in, h_in, w_in, h_out, w_out, ctx);
            }
        } else {
            if (w_in > 8){
                conv_depthwise_3x3s2p0_bias(dout, din, weights, bias, flag_bias, \
                    num, ch_in, h_in, w_in, h_out, w_out, ctx);
            }else{
                conv_depthwise_3x3s2p0_bias_s(dout, din, weights, bias, flag_bias, \
                    num, ch_in, h_in, w_in, h_out, w_out, ctx);
            }

        }
    }
}
/**
 * \brief depthwise convolution, kernel size 3x3, stride 1, pad 1, with bias, width > 4
 */
//4line
void conv_depthwise_3x3s1p0_bias(float* dout, const float* din, \
    const float* weights, const float* bias, bool flag_bias, \
    const int num, const int ch_in, const int h_in, const int w_in, \
    const int h_out, const int w_out, ARMContext* ctx) {
    //! pad is done implicit
    const float zero[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    //! for 4x6 convolution window
    const unsigned int right_pad_idx[8] = {5, 4, 3, 2, 1, 0, 0, 0};

    float* zero_ptr = ctx->workspace_data<float>();
    memset(zero_ptr, 0, w_in * sizeof(float));
    float* write_ptr = zero_ptr + w_in;

    int size_in_channel = w_in * h_in;
    int size_out_channel = w_out * h_out;
    int w_stride = 9;

    int tile_w = w_out >> 2;
    int remain = w_out % 4;

    unsigned int size_pad_right = (unsigned int)(6 + (tile_w << 2) - w_in);
    const int remian_idx[4] = {0, 1, 2, 3};

    uint32x4_t vmask_rp1 = vcgeq_u32(vld1q_u32(right_pad_idx), vdupq_n_u32(size_pad_right));
    uint32x4_t vmask_rp2 = vcgeq_u32(vld1q_u32(right_pad_idx + 4), vdupq_n_u32(size_pad_right));
    uint32x4_t vmask_result = vcgtq_s32(vdupq_n_s32(remain), vld1q_s32(remian_idx));

    unsigned int vmask[8];
    vst1q_u32(vmask, vmask_rp1);
    vst1q_u32(vmask + 4, vmask_rp2);

    unsigned int rmask[4];
    vst1q_u32(rmask, vmask_result);

    float32x4_t vzero = vdupq_n_f32(0.f);

    for (int n = 0; n < num; ++n) {
        const float *din_batch = din + n * ch_in * size_in_channel;
        float *dout_batch = dout + n * ch_in * size_out_channel;
#pragma omp parallel for
#ifdef __aarch64__
        for (int c = 0; c < ch_in; c++) {
            float* dout_ptr = dout_batch + c * size_out_channel;

            const float* din_ch_ptr = din_batch + c * size_in_channel;

            float bias_val = flag_bias ? bias[c] : 0.f;
            float vbias[4] = {bias_val, bias_val, bias_val, bias_val};

            const float* wei_ptr = weights + c * w_stride;

            float32x4_t wr0 = vld1q_f32(wei_ptr);
            float32x4_t wr1 = vld1q_f32(wei_ptr + 3);
            float32x4_t wr2 = vld1q_f32(wei_ptr + 6);
            // wr0 = vsetq_lane_f32(0.f, wr0, 3);
            // wr1 = vsetq_lane_f32(0.f, wr1, 3);
            // wr2 = vsetq_lane_f32(0.f, wr2, 3);

            float *doutr0 = dout_ptr;
            float *doutr1 = doutr0 + w_out;
            float *doutr2 = doutr1 + w_out;
            float *doutr3 = doutr2 + w_out;

            const float *dr0 = din_ch_ptr;
            const float *dr1 = dr0 + w_in;
            const float *dr2 = dr1 + w_in;
            const float *dr3 = dr2 + w_in;
            const float *dr4 = dr3 + w_in;
            const float *dr5 = dr4 + w_in;

            const float *din_ptr0 = dr0;
            const float *din_ptr1 = dr1;
            const float *din_ptr2 = dr2;
            const float *din_ptr3 = dr3;
            const float *din_ptr4 = dr4;
            const float *din_ptr5 = dr5;

            for (int i = 0; i < h_out; i += 4){
                //! process top pad pad_h = 1
                din_ptr0 = dr0;
                din_ptr1 = dr1;
                din_ptr2 = dr2;
                din_ptr3 = dr3;
                din_ptr4 = dr4;
                din_ptr5 = dr5;

                doutr0 = dout_ptr;
                doutr1 = doutr0 + w_out;
                doutr2 = doutr1 + w_out;
                doutr3 = doutr2 + w_out;

                dr0 = dr4;
                dr1 = dr5;
                dr2 = dr1 + w_in;
                dr3 = dr2 + w_in;
                dr4 = dr3 + w_in;
                dr5 = dr4 + w_in;

                //! process bottom pad
                if (i + 5 >= h_in) {
                    switch (i + 5 - h_in) {
                        case 5:
                            din_ptr1 = zero_ptr;
                        case 4:
                            din_ptr2 = zero_ptr;
                        case 3:
                            din_ptr3 = zero_ptr;
                        case 2:
                            din_ptr4 = zero_ptr;
                        case 1:
                            din_ptr5 = zero_ptr;
                        case 0:
                            din_ptr5 = zero_ptr;
                        default:
                            break;
                    }
                }
                //! process bottom remain
                if (i + 4 > h_out) {
                    switch (i + 4 - h_out) {
                        case 3:
                            doutr1 = write_ptr;
                        case 2:
                            doutr2 = write_ptr;
                        case 1:
                            doutr3 = write_ptr;
                        default:
                            break;
                    }
                }

                int cnt = tile_w;
                asm volatile(
                    "PRFM PLDL1KEEP, [%[din_ptr0]] \n"
                    "PRFM PLDL1KEEP, [%[din_ptr1]] \n"
                    "PRFM PLDL1KEEP, [%[din_ptr2]] \n"
                    "PRFM PLDL1KEEP, [%[din_ptr3]] \n"
                    "PRFM PLDL1KEEP, [%[din_ptr4]] \n"
                    "PRFM PLDL1KEEP, [%[din_ptr5]] \n"
                    "movi   v21.4s, #0x0\n"                         /* out0 = 0 */

                    "ld1 {v0.4s}, [%[din_ptr0]], #16   \n" /*vld1q_f32(din_ptr0)*/
                    "ld1 {v2.4s}, [%[din_ptr1]], #16   \n" /*vld1q_f32(din_ptr0)*/
                    "ld1 {v4.4s}, [%[din_ptr2]], #16   \n" /*vld1q_f32(din_ptr0)*/
                    "ld1 {v6.4s}, [%[din_ptr3]], #16   \n" /*vld1q_f32(din_ptr0)*/

                    "ld1 {v1.4s}, [%[din_ptr0]]   \n" /*vld1q_f32(din_ptr0)*/
                    "ld1 {v3.4s}, [%[din_ptr1]]   \n" /*vld1q_f32(din_ptr0)*/
                    "ld1 {v5.4s}, [%[din_ptr2]]   \n" /*vld1q_f32(din_ptr0)*/
                    "ld1 {v7.4s}, [%[din_ptr3]]   \n" /*vld1q_f32(din_ptr0)*/

                    "ld1 {v12.4s}, [%[bias_val]]     \n" /*vdupq_n_f32(bias_val)*/
                    "ld1 {v13.4s}, [%[bias_val]]      \n" /*vdupq_n_f32(bias_val)*/
                    "ld1 {v14.4s}, [%[bias_val]]      \n" /*vdupq_n_f32(bias_val)*/
                    "ld1 {v15.4s}, [%[bias_val]]      \n" /*vdupq_n_f32(bias_val)*/

                    "ext  v16.16b, v0.16b, v1.16b, #4 \n" /* v16 = 1234 */
                    "ext  v17.16b, v0.16b, v1.16b, #8 \n" /* v17 = 2345 */

                //mid
                    // "cmp  %[cnt], #1                \n"
                    // "blt 5f                         \n"
                    "4:                             \n"
                    //r0
                    "fmla v12.4s ,  v0.4s,  %[w0].s[0]\n" /* outr00 += din0_0123 * w0[0]*/

                    "ld1 {v8.4s}, [%[din_ptr4]], #16   \n" /*vld1q_f32(din_ptr0)*/
                    "ld1 {v10.4s}, [%[din_ptr5]], #16   \n" /*vld1q_f32(din_ptr0)*/
                    "ld1 {v0.4s}, [%[din_ptr0]], #16   \n" /*vld1q_f32(din_ptr0)*/

                    "fmla v12.4s ,  v16.4s,  %[w0].s[1]\n" /* outr00 += din0_1234 * w0[1]*/

                    "ld1 {v9.4s}, [%[din_ptr4]]   \n" /*vld1q_f32(din_ptr0)*/
                    "ld1 {v11.4s}, [%[din_ptr5]]   \n" /*vld1q_f32(din_ptr0)*/
                    "ld1 {v1.4s}, [%[din_ptr0]]        \n" /*vld1q_f32(din_ptr0)*/

                    "fmla v12.4s ,  v17.4s,  %[w0].s[2]\n" /* outr00 += din0_2345 * w0[2]*/

                    "ext  v16.16b, v2.16b, v3.16b, #4 \n" /* v16 = 1234*/
                    "ext  v17.16b, v2.16b, v3.16b, #8 \n" /* v16 = 2345 */

                    //r1
                    "fmla v13.4s ,  v2.4s,  %[w0].s[0]\n" /* outr00 += din0_0123 * w0[0]*/
                    "fmla v12.4s ,  v2.4s,  %[w1].s[0]\n" /* outr00 += din0_0123 * w0[0]*/

                    "ld1 {v2.4s}, [%[din_ptr1]], #16   \n" /*vld1q_f32(din_ptr0)*/

                    "fmla v13.4s ,  v16.4s,  %[w0].s[1]\n" /* outr00 += din0_1234 * w0[1]*/
                    "fmla v12.4s ,  v16.4s,  %[w1].s[1]\n" /* outr00 += din0_1234 * w0[1]*/

                    "ld1 {v3.4s}, [%[din_ptr1]]   \n" /*vld1q_f32(din_ptr0)*/

                    "fmla v13.4s ,  v17.4s,  %[w0].s[2]\n" /* outr00 += din0_2345 * w0[2]*/
                    "fmla v12.4s ,  v17.4s,  %[w1].s[2]\n" /* outr00 += din0_2345 * w0[2]*/

                    "ext  v16.16b, v4.16b, v5.16b, #4 \n" /* v16 = 1234*/
                    "ext  v17.16b, v4.16b, v5.16b, #8 \n" /* v16 = 2345 */

                    //r2
                    "fmla v14.4s ,  v4.4s,  %[w0].s[0]\n" /* outr00 += din0_0123 * w0[0]*/
                    "fmla v13.4s ,  v4.4s,  %[w1].s[0]\n" /* outr00 += din0_0123 * w0[0]*/
                    "fmla v12.4s ,  v4.4s,  %[w2].s[0]\n" /* outr00 += din0_0123 * w0[0]*/

                    "ld1 {v4.4s}, [%[din_ptr2]], #16   \n" /*vld1q_f32(din_ptr0)*/

                    "fmla v14.4s ,  v16.4s,  %[w0].s[1]\n" /* outr00 += din0_1234 * w0[1]*/
                    "fmla v13.4s ,  v16.4s,  %[w1].s[1]\n" /* outr00 += din0_1234 * w0[1]*/
                    "fmla v12.4s ,  v16.4s,  %[w2].s[1]\n" /* outr00 += din0_1234 * w0[1]*/

                    "ld1 {v5.4s}, [%[din_ptr2]]   \n" /*vld1q_f32(din_ptr0)*/

                    "fmla v14.4s ,  v17.4s,  %[w0].s[2]\n" /* outr00 += din0_2345 * w0[2]*/
                    "fmla v13.4s ,  v17.4s,  %[w1].s[2]\n" /* outr00 += din0_2345 * w0[2]*/
                    "fmla v12.4s ,  v17.4s,  %[w2].s[2]\n" /* outr00 += din0_2345 * w0[2]*/

                    "ext  v16.16b, v6.16b, v7.16b, #4 \n" /* v16 = 1234*/
                    "ext  v17.16b, v6.16b, v7.16b, #8 \n" /* v16 = 2345 */

                    //r3
                    "fmla v15.4s ,  v6.4s,  %[w0].s[0]\n" /* outr00 += din0_0123 * w0[0]*/
                    "fmla v14.4s ,  v6.4s,  %[w1].s[0]\n" /* outr00 += din0_0123 * w0[0]*/
                    "fmla v13.4s ,  v6.4s,  %[w2].s[0]\n" /* outr00 += din0_0123 * w0[0]*/

                    "ld1 {v6.4s}, [%[din_ptr3]], #16   \n" /*vld1q_f32(din_ptr0)*/
                    "st1 {v12.4s}, [%[doutr0]], #16     \n"

                    "fmla v15.4s ,  v16.4s,  %[w0].s[1]\n" /* outr00 += din0_1234 * w0[1]*/
                    "fmla v14.4s ,  v16.4s,  %[w1].s[1]\n" /* outr00 += din0_1234 * w0[1]*/
                    "fmla v13.4s ,  v16.4s,  %[w2].s[1]\n" /* outr00 += din0_1234 * w0[1]*/

                    "ld1 {v7.4s}, [%[din_ptr3]]   \n" /*vld1q_f32(din_ptr0)*/
                    "ld1 {v12.4s}, [%[bias_val]]      \n" /*vdupq_n_f32(bias_val)*/

                    "fmla v15.4s ,  v17.4s,  %[w0].s[2]\n" /* outr00 += din0_2345 * w0[2]*/
                    "fmla v14.4s ,  v17.4s,  %[w1].s[2]\n" /* outr00 += din0_2345 * w0[2]*/
                    "fmla v13.4s ,  v17.4s,  %[w2].s[2]\n" /* outr00 += din0_2345 * w0[2]*/

                    "ext  v16.16b, v8.16b, v9.16b, #4 \n" /* v16 = 1234*/
                    "ext  v17.16b, v8.16b, v9.16b, #8 \n" /* v16 = 2345 */

                    //r4
                    "fmla v15.4s ,  v8.4s,  %[w1].s[0]\n" /* outr00 += din0_0123 * w0[0]*/
                    "fmla v14.4s ,  v8.4s,  %[w2].s[0]\n" /* outr00 += din0_0123 * w0[0]*/

                    "st1 {v13.4s}, [%[doutr1]], #16     \n"

                    "fmla v15.4s ,  v16.4s,  %[w1].s[1]\n" /* outr00 += din0_1234 * w0[1]*/
                    "fmla v14.4s ,  v16.4s,  %[w2].s[1]\n" /* outr00 += din0_1234 * w0[1]*/

                    "ld1 {v13.4s}, [%[bias_val]]      \n" /*vdupq_n_f32(bias_val)*/

                    "fmla v15.4s ,  v17.4s,  %[w1].s[2]\n" /* outr00 += din0_2345 * w0[2]*/
                    "fmla v14.4s ,  v17.4s,  %[w2].s[2]\n" /* outr00 += din0_2345 * w0[2]*/

                    "ext  v16.16b, v10.16b, v11.16b, #4 \n" /* v16 = 1234*/
                    "ext  v17.16b, v10.16b, v11.16b, #8 \n" /* v16 = 2345 */

                    //r5
                    "fmla v15.4s ,  v10.4s,  %[w2].s[0]\n" /* outr00 += din0_0123 * w0[0]*/

                    "st1 {v14.4s}, [%[doutr2]], #16     \n"

                    "fmla v15.4s ,  v16.4s,  %[w2].s[1]\n" /* outr00 += din0_1234 * w0[1]*/

                    "ld1 {v14.4s}, [%[bias_val]]      \n" /*vdupq_n_f32(bias_val)*/

                    "fmla v15.4s ,  v17.4s,  %[w2].s[2]\n" /* outr00 += din0_2345 * w0[2]*/

                    "ext  v16.16b, v0.16b, v1.16b, #4 \n" /* v16 = 1234*/
                    "ext  v17.16b, v0.16b, v1.16b, #8 \n" /* v16 = 2345 */

                    "subs %[cnt], %[cnt], #1 \n"

                    "st1 {v15.4s}, [%[doutr3]], #16     \n"
                    "ld1 {v15.4s}, [%[bias_val]]      \n" /*vdupq_n_f32(bias_val)*/

                    "bne 4b \n"

                //right
                    "5:                             \n"
                    "cmp  %[remain], #1             \n"
                    "blt 0f                         \n"
                    "ld1 {v18.4s, v19.4s}, [%[vmask]]         \n"
                    "ld1 {v22.4s}, [%[doutr0]]         \n"
                    "ld1 {v23.4s}, [%[doutr1]]         \n"
                    "ld1 {v24.4s}, [%[doutr2]]         \n"
                    "ld1 {v25.4s}, [%[doutr3]]         \n"

                    "bif v0.16b, %[vzero].16b, v18.16b \n"
                    "bif v1.16b, %[vzero].16b, v19.16b \n"
                    "bif v2.16b, %[vzero].16b, v18.16b \n"
                    "bif v3.16b, %[vzero].16b, v19.16b \n"

                    "ext  v16.16b, v0.16b, v1.16b, #4 \n" /* v16 = 1234*/
                    "ext  v17.16b, v0.16b, v1.16b, #8 \n" /* v16 = 2345 */
                    "ld1 {v8.4s}, [%[din_ptr4]], #16   \n" /*vld1q_f32(din_ptr0)*/
                    "ld1 {v10.4s}, [%[din_ptr5]], #16   \n" /*vld1q_f32(din_ptr0)*/
                    "ld1 {v9.4s}, [%[din_ptr4]]   \n" /*vld1q_f32(din_ptr0)*/
                    "ld1 {v11.4s}, [%[din_ptr5]]   \n" /*vld1q_f32(din_ptr0)*/

                    //r0
                    "fmla v12.4s,  v0.4s,  %[w0].s[0]\n" /* outr00 += din0_0123 * w0[0]*/

                    "bif v4.16b, %[vzero].16b, v18.16b \n"
                    "bif v5.16b, %[vzero].16b, v19.16b \n"
                    "bif v6.16b, %[vzero].16b, v18.16b \n"
                    "bif v7.16b, %[vzero].16b, v19.16b \n"

                    "fmla v12.4s,  v16.4s,  %[w0].s[1]\n" /* outr00 += din0_1234 * w0[1]*/

                    "bif v8.16b, %[vzero].16b, v18.16b \n"
                    "bif v9.16b, %[vzero].16b, v19.16b \n"
                    "bif v10.16b, %[vzero].16b, v18.16b \n"
                    "bif v11.16b, %[vzero].16b, v19.16b \n"

                    "fmla v12.4s ,  v17.4s,  %[w0].s[2]\n" /* outr00 += din0_2345 * w0[2]*/

                    "ext  v16.16b, v2.16b, v3.16b, #4 \n" /* v16 = 1234*/
                    "ext  v17.16b, v2.16b, v3.16b, #8 \n" /* v16 = 2345 */
                    "ld1 {v18.4s}, [%[rmask]]         \n"

                    //r1
                    "fmla v13.4s ,  v2.4s,  %[w0].s[0]\n" /* outr00 += din0_0123 * w0[0]*/
                    "fmla v12.4s ,  v2.4s,  %[w1].s[0]\n" /* outr00 += din0_0123 * w0[0]*/

                    "fmla v13.4s ,  v16.4s,  %[w0].s[1]\n" /* outr00 += din0_1234 * w0[1]*/
                    "fmla v12.4s ,  v16.4s,  %[w1].s[1]\n" /* outr00 += din0_1234 * w0[1]*/

                    "fmla v13.4s ,  v17.4s,  %[w0].s[2]\n" /* outr00 += din0_2345 * w0[2]*/
                    "fmla v12.4s ,  v17.4s,  %[w1].s[2]\n" /* outr00 += din0_2345 * w0[2]*/

                    "ext  v16.16b, v4.16b, v5.16b, #4 \n" /* v16 = 1234*/
                    "ext  v17.16b, v4.16b, v5.16b, #8 \n" /* v16 = 2345 */

                    //r2
                    "fmla v14.4s ,  v4.4s,  %[w0].s[0]\n" /* outr00 += din0_0123 * w0[0]*/
                    "fmla v13.4s ,  v4.4s,  %[w1].s[0]\n" /* outr00 += din0_0123 * w0[0]*/
                    "fmla v12.4s ,  v4.4s,  %[w2].s[0]\n" /* outr00 += din0_0123 * w0[0]*/

                    "fmla v14.4s ,  v16.4s,  %[w0].s[1]\n" /* outr00 += din0_1234 * w0[1]*/
                    "fmla v13.4s ,  v16.4s,  %[w1].s[1]\n" /* outr00 += din0_1234 * w0[1]*/
                    "fmla v12.4s ,  v16.4s,  %[w2].s[1]\n" /* outr00 += din0_1234 * w0[1]*/

                    "fmla v14.4s ,  v17.4s,  %[w0].s[2]\n" /* outr00 += din0_2345 * w0[2]*/
                    "fmla v13.4s ,  v17.4s,  %[w1].s[2]\n" /* outr00 += din0_2345 * w0[2]*/
                    "fmla v12.4s ,  v17.4s,  %[w2].s[2]\n" /* outr00 += din0_2345 * w0[2]*/

                    "ext  v16.16b, v6.16b, v7.16b, #4 \n" /* v16 = 1234*/
                    "ext  v17.16b, v6.16b, v7.16b, #8 \n" /* v16 = 2345 */

                    //r3
                    "fmla v15.4s ,  v6.4s,  %[w0].s[0]\n" /* outr00 += din0_0123 * w0[0]*/
                    "fmla v14.4s ,  v6.4s,  %[w1].s[0]\n" /* outr00 += din0_0123 * w0[0]*/
                    "fmla v13.4s ,  v6.4s,  %[w2].s[0]\n" /* outr00 += din0_0123 * w0[0]*/

                    "bif v12.16b, v22.16b, v18.16b \n"

                    "fmla v15.4s ,  v16.4s,  %[w0].s[1]\n" /* outr00 += din0_1234 * w0[1]*/
                    "fmla v14.4s ,  v16.4s,  %[w1].s[1]\n" /* outr00 += din0_1234 * w0[1]*/
                    "fmla v13.4s ,  v16.4s,  %[w2].s[1]\n" /* outr00 += din0_1234 * w0[1]*/

                    "st1 {v12.4s}, [%[doutr0]], #16     \n"

                    "fmla v15.4s ,  v17.4s,  %[w0].s[2]\n" /* outr00 += din0_2345 * w0[2]*/
                    "fmla v14.4s ,  v17.4s,  %[w1].s[2]\n" /* outr00 += din0_2345 * w0[2]*/
                    "fmla v13.4s ,  v17.4s,  %[w2].s[2]\n" /* outr00 += din0_2345 * w0[2]*/

                    "ext  v16.16b, v8.16b, v9.16b, #4 \n" /* v16 = 1234*/
                    "ext  v17.16b, v8.16b, v9.16b, #8 \n" /* v16 = 2345 */

                    //r3
                    "fmla v15.4s ,  v8.4s,  %[w1].s[0]\n" /* outr00 += din0_0123 * w0[0]*/
                    "fmla v14.4s ,  v8.4s,  %[w2].s[0]\n" /* outr00 += din0_0123 * w0[0]*/

                    "bif v13.16b, v23.16b, v18.16b \n"

                    "fmla v15.4s ,  v16.4s,  %[w1].s[1]\n" /* outr00 += din0_1234 * w0[1]*/
                    "fmla v14.4s ,  v16.4s,  %[w2].s[1]\n" /* outr00 += din0_1234 * w0[1]*/

                    "st1 {v13.4s}, [%[doutr1]], #16     \n"

                    "fmla v15.4s ,  v17.4s,  %[w1].s[2]\n" /* outr00 += din0_2345 * w0[2]*/
                    "fmla v14.4s ,  v17.4s,  %[w2].s[2]\n" /* outr00 += din0_2345 * w0[2]*/

                    "ext  v16.16b, v10.16b, v11.16b, #4 \n" /* v16 = 1234*/
                    "ext  v17.16b, v10.16b, v11.16b, #8 \n" /* v16 = 2345 */

                    //r3
                    "fmla v15.4s ,  v10.4s,  %[w2].s[0]\n" /* outr00 += din0_0123 * w0[0]*/

                    "bif v14.16b, v24.16b, v18.16b \n"

                    "fmla v15.4s ,  v16.4s,  %[w2].s[1]\n" /* outr00 += din0_1234 * w0[1]*/

                    "st1 {v14.4s}, [%[doutr2]], #16     \n"

                    "fmla v15.4s ,  v17.4s,  %[w2].s[2]\n" /* outr00 += din0_2345 * w0[2]*/

                    "bif v15.16b, v25.16b, v18.16b \n"

                    "st1 {v15.4s}, [%[doutr3]], #16     \n"
                    //end
                    "0:                             \n"
                :[cnt]"+r"(cnt), [din_ptr0] "+r"(din_ptr0), [din_ptr1] "+r"(din_ptr1), [din_ptr2] "+r"(din_ptr2), \
                        [din_ptr3] "+r"(din_ptr3), [din_ptr4] "+r"(din_ptr4), [din_ptr5] "+r"(din_ptr5), \
                        [doutr0] "+r"(doutr0), [doutr1] "+r"(doutr1), [doutr2] "+r"(doutr2), [doutr3] "+r"(doutr3)
                : [w0]"w"(wr0), [w1]"w"(wr1), [w2]"w"(wr2), [bias_val] "r" (vbias), \
                      [vmask] "r" (vmask), [rmask] "r" (rmask), [vzero] "w" (vzero), [remain] "r" (remain)
                :"cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",\
                      "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18",\
                      "v19", "v20", "v21", "v22", "v23", "v24", "v25"
                );
                dout_ptr = dout_ptr + 4 * w_out;
            }
        }
#else
        for (int i = 0; i < ch_in; ++i) {
            const float *din_channel = din_batch + i * size_in_channel;

            const float *weight_ptr = weights + i * 9;
            float32x4_t wr0 = vld1q_f32(weight_ptr);
            float32x4_t wr1 = vld1q_f32(weight_ptr + 3);
            float32x4_t wr2 = vld1q_f32(weight_ptr + 6);
            float bias_val = flag_bias ? bias[i] : 0.f;

            float *dout_channel = dout_batch + i * size_out_channel;

            const float *dr0 = din_channel;
            const float *dr1 = dr0 + w_in;
            const float *dr2 = dr1 + w_in;
            const float *dr3 = dr2 + w_in;

            const float *din0_ptr = nullptr;
            const float *din1_ptr = nullptr;
            const float *din2_ptr = nullptr;
            const float *din3_ptr = nullptr;

            float *doutr0 = nullptr;
            float *doutr1 = nullptr;

            float* ptr_zero = const_cast<float*>(zero);

        for (int i = 0; i < h_out; i += 2){
                din0_ptr = dr0;
                din1_ptr = dr1;
                din2_ptr = dr2;
                din3_ptr = dr3;

                doutr0 = dout_channel;
                doutr1 = dout_channel + w_out;

                dr0 = dr2;
                dr1 = dr3;
                dr2 = dr1 + w_in;
                dr3 = dr2 + w_in;
                //! process bottom pad
                if (i + 3 >= h_in) {
                    switch (i + 3 - h_in) {
                        case 3:
                            din1_ptr = zero_ptr;
                        case 2:
                            din2_ptr = zero_ptr;
                        case 1:
                            din3_ptr = zero_ptr;
                        case 0:
                            din3_ptr = zero_ptr;
                        default:
                            break;
                    }
                }
                //! process bottom remain
                if (i + 2 > h_out) {
                    doutr1 = write_ptr;
                }
                int cnt = tile_w;
                unsigned int* rmask_ptr = rmask;
                unsigned int* vmask_ptr = vmask;
                asm volatile(
                    "pld [%[din0_ptr]]                             @ preload data\n"
                    "pld [%[din1_ptr]]                      @ preload data\n"
                    "pld [%[din2_ptr]]                      @ preload data\n"
                    "pld [%[din3_ptr]]                      @ preload data\n"

                    "vld1.32  {d16-d17}, [%[din0_ptr]]!    @ load din r0\n"
                    "vld1.32  {d20-d21}, [%[din1_ptr]]!    @ load din r1\n"
                    "vld1.32  {d24-d25}, [%[din2_ptr]]!    @ load din r2\n"
                    "vld1.32  {d28-d29}, [%[din3_ptr]]!    @ load din r3\n"
                    "vld1.32  {d18}, [%[din0_ptr]]    @ load din r0\n"
                    "vld1.32  {d22}, [%[din1_ptr]]    @ load din r0\n"
                    "vld1.32  {d26}, [%[din2_ptr]]    @ load din r0\n"
                    "vld1.32  {d30}, [%[din3_ptr]]    @ load din r0\n"

                    "vdup.32 q4, %[bias_val]                            @ and \n" //q4 = vbias
                    "vdup.32 q5, %[bias_val]                            @ and \n" //q5 = vbias

                    "vext.32  q6, q8, q9, #1     @ 1234\n"
                    "vext.32  q7, q8, q9, #2     @ 2345\n"
                //mid
                     "1:                                    @ right pad entry\n"
                    //r0
                    "vmla.f32 q4, q8, %e[wr0][0]  @ q4 += 0123 * wr0[0]\n"

                    "pld [%[din0_ptr]]                             @ preload data\n"
                    "pld [%[din1_ptr]]                             @ preload data\n"
                    "pld [%[din2_ptr]]                             @ preload data\n"
                    "pld [%[din3_ptr]]                             @ preload data\n"

                    "vmla.f32 q4, q6, %e[wr0][1]  @ q4 += 1234 * wr0[1]\n"

                    "vld1.32  {d16-d17}, [%[din0_ptr]]!    @ load din r0\n"

                    "vmla.f32 q4, q7, %f[wr0][0]  @ q4 += 2345 * wr0[2]\n"

                    "vld1.32  {d18}, [%[din0_ptr]]    @ load din r0\n"

                    "vext.32  q6, q10, q11, #1     @ 1234\n"
                    "vext.32  q7, q10, q11, #2     @ 2345\n"

                    //r1
                    "vmla.f32 q5, q10, %e[wr0][0]  @ q4 += 1234 * wr0[0]\n"
                    "vmla.f32 q4, q10, %e[wr1][0]  @ q4 += 1234 * wr0[1]\n"

                    "vld1.32  {d20-d21}, [%[din1_ptr]]!    @ load din r0\n"

                    "vmla.f32 q5, q6, %e[wr0][1]  @ q4 += 1234 * wr0[1]\n"
                    "vmla.f32 q4, q6, %e[wr1][1]  @ q4 += 1234 * wr0[1]\n"

                    "vld1.32  {d22}, [%[din1_ptr]]    @ load din r0\n"

                    "vmla.f32 q5, q7, %f[wr0][0]  @ q4 += 1234 * wr0[1]\n"
                    "vmla.f32 q4, q7, %f[wr1][0]  @ q4 += 1234 * wr0[1]\n"

                    "vext.32  q6, q12, q13, #1     @ 1234\n"
                    "vext.32  q7, q12, q13, #2     @ 2345\n"

                    //r2
                    "vmla.f32 q5, q12, %e[wr1][0]  @ q4 += 1234 * wr0[0]\n"
                    "vmla.f32 q4, q12, %e[wr2][0]  @ q4 += 1234 * wr0[1]\n"

                    "vld1.32  {d24-d25}, [%[din2_ptr]]!    @ load din r0\n"

                    "vmla.f32 q5, q6, %e[wr1][1]  @ q4 += 1234 * wr0[1]\n"
                    "vmla.f32 q4, q6, %e[wr2][1]  @ q4 += 1234 * wr0[1]\n"

                    "vld1.32  {d26}, [%[din2_ptr]]    @ load din r0\n"

                    "vmla.f32 q5, q7, %f[wr1][0]  @ q4 += 1234 * wr0[1]\n"
                    "vmla.f32 q4, q7, %f[wr2][0]  @ q4 += 1234 * wr0[1]\n"

                    "vext.32  q6, q14, q15, #1     @ 1234\n"
                    "vext.32  q7, q14, q15, #2     @ 2345\n"

                    //r3
                    "vmla.f32 q5, q14, %e[wr2][0]  @ q4 += 0123 * wr0[0]\n"

                    "vld1.32  {d28-d29}, [%[din3_ptr]]!    @ load din r0\n"
                    "vst1.32  {d8-d9},   [%[dout_ptr1]]!  @ store result, add pointer\n"

                    "vmla.f32 q5, q6, %e[wr2][1]  @ q4 += 1234 * wr0[1]\n"

                    "vld1.32  {d30}, [%[din3_ptr]]    @ load din r0\n"
                    "vdup.32 q4, %[bias_val]                            @ and \n" //q4 = vbias

                    "vmla.f32 q5, q7, %f[wr2][0]  @ q4 += 2345 * wr0[2]\n"

                    "vext.32  q6, q8, q9, #1     @ 1234\n"
                    "vext.32  q7, q8, q9, #2     @ 2345\n"

                    "vst1.32  {d10-d11},   [%[dout_ptr2]]!  @ store result, add pointer\n"

                    "subs %[cnt], #1 @ loop count minus 1\n"

                    "vdup.32 q5, %[bias_val]                            @ and \n" //q4 = vbias

                    "bne    1b                             @ jump to main loop start point\n"

                //right
                    "3:                                    @ right pad entry\n"
                    "cmp %[remain], #1                             @ check whether has mid cols\n"
                    "blt  0f                                @ jump to main loop start point\n"
                    "vld1.32  {d19}, [%[vmask]]!    @ load din r0\n"
                    "vld1.32  {d23}, [%[vmask]]!    @ load din r0\n"

                    "vld1.32  {d27}, [%[vmask]]!    @ load din r0\n"
                    "vld1.32  {d31}, [%[vmask]]!    @ load din r0\n"

                    "vbif d16, %e[vzero], d19              @ bit select, deal with right pad\n"
                    "vbif d17, %e[vzero], d23              @ bit select, deal with right pad\n"
                    "vbif d18, %e[vzero], d27             @ bit select, deal with right pad\n"

                    "vbif d20, %e[vzero], d19              @ bit select, deal with right pad\n"
                    "vbif d21, %e[vzero], d23              @ bit select, deal with right pad\n"
                    "vbif d22, %e[vzero], d27             @ bit select, deal with right pad\n"

                    "vext.32  q6, q8, q9, #1     @ 1234\n"
                    "vext.32  q7, q8, q9, #2     @ 2345\n"

                    //r0
                    "vmla.f32 q4, q8, %e[wr0][0]  @ q4 += 0123 * wr0[0]\n"

                    "vbif d24, %e[vzero], d19              @ bit select, deal with right pad\n"
                    "vbif d25, %e[vzero], d23              @ bit select, deal with right pad\n"
                    "vbif d26, %e[vzero], d27             @ bit select, deal with right pad\n"

                    "vmla.f32 q4, q6, %e[wr0][1]  @ q4 += 1234 * wr0[1]\n"

                    "vbif d28, %e[vzero], d19              @ bit select, deal with right pad\n"
                    "vbif d29, %e[vzero], d23              @ bit select, deal with right pad\n"
                    "vbif d30, %e[vzero], d27             @ bit select, deal with right pad\n"

                    "vmla.f32 q4, q7, %f[wr0][0]  @ q4 += 2345 * wr0[2]\n"

                    "vext.32  q6, q10, q11, #1     @ 1234\n"
                    "vext.32  q7, q10, q11, #2     @ 2345\n"

                    //r1
                    "vmla.f32 q5, q10, %e[wr0][0]  @ q4 += 1234 * wr0[0]\n"
                    "vmla.f32 q4, q10, %e[wr1][0]  @ q4 += 1234 * wr0[1]\n"

                    "vld1.32  {d19}, [%[rmask]]!    @ load din r0\n"
                    "vld1.32  {d23}, [%[rmask]]!    @ load din r0\n"

                    "vmla.f32 q5, q6, %e[wr0][1]  @ q4 += 1234 * wr0[1]\n"
                    "vmla.f32 q4, q6, %e[wr1][1]  @ q4 += 1234 * wr0[1]\n"

                    "vld1.32  {d16-d17}, [%[dout_ptr1]]    @ load din r0\n"
                    "vld1.32  {d20-d21}, [%[dout_ptr2]]    @ load din r0\n"

                    "vmla.f32 q5, q7, %f[wr0][0]  @ q4 += 1234 * wr0[1]\n"
                    "vmla.f32 q4, q7, %f[wr1][0]  @ q4 += 1234 * wr0[1]\n"

                    "vext.32  q6, q12, q13, #1     @ 1234\n"
                    "vext.32  q7, q12, q13, #2     @ 2345\n"

                    //r2
                    "vmla.f32 q5, q12, %e[wr1][0]  @ q4 += 1234 * wr0[0]\n"
                    "vmla.f32 q4, q12, %e[wr2][0]  @ q4 += 1234 * wr0[1]\n"

                    "vmla.f32 q5, q6, %e[wr1][1]  @ q4 += 1234 * wr0[1]\n"
                    "vmla.f32 q4, q6, %e[wr2][1]  @ q4 += 1234 * wr0[1]\n"

                    "vmla.f32 q5, q7, %f[wr1][0]  @ q4 += 1234 * wr0[1]\n"
                    "vmla.f32 q4, q7, %f[wr2][0]  @ q4 += 1234 * wr0[1]\n"

                    "vext.32  q6, q14, q15, #1     @ 1234\n"
                    "vext.32  q7, q14, q15, #2     @ 2345\n"

                    //r3
                    "vmla.f32 q5, q14, %e[wr2][0]  @ q4 += 0123 * wr0[0]\n"

                    "vbif d8, d16, d19              @ bit select, deal with right pad\n"
                    "vbif d9, d17, d23              @ bit select, deal with right pad\n"

                    "vmla.f32 q5, q6, %e[wr2][1]  @ q4 += 1234 * wr0[1]\n"

                    "vst1.32  {d8-d9},   [%[dout_ptr1]]!  @ store result, add pointer\n"

                    "vmla.f32 q5, q7, %f[wr2][0]  @ q4 += 2345 * wr0[2]\n"

                    "vbif d10, d20, d19              @ bit select, deal with right pad\n"
                    "vbif d11, d21, d23              @ bit select, deal with right pad\n"

                    "vst1.32  {d10-d11},   [%[dout_ptr2]]!  @ store result, add pointer\n"
                    "0:                         \n"

                :[dout_ptr1] "+r"(doutr0), [dout_ptr2] "+r"(doutr1), \
                    [din0_ptr] "+r"(din0_ptr), [din1_ptr] "+r"(din1_ptr), \
                    [din2_ptr] "+r"(din2_ptr), [din3_ptr] "+r"(din3_ptr), \
                    [cnt] "+r"(cnt), [rmask] "+r" (rmask_ptr), [vmask] "+r" (vmask_ptr)
                :[wr0] "w"(wr0), [wr1] "w"(wr1), [wr2] "w"(wr2), [bias_val] "r" (bias_val), \
                [vzero] "w" (vzero), [remain] "r" (remain)
                :"cc", "memory", "q4", "q5", "q6", "q7", "q8", "q9", \
                        "q10", "q11", "q12", "q13", "q14", "q15"
                );
                dout_channel += 2 * w_out;
            } //! end of processing mid rows
        }
#endif

    }

}

/**
 * \brief depthwise convolution kernel 3x3, stride 2
 */
//w_in > 7
void conv_depthwise_3x3s2p0_bias(float* dout, const float* din, \
    const float* weights, const float* bias, bool flag_bias, \
    const int num, const int ch_in, const int h_in, const int w_in, \
    const int h_out, const int w_out, ARMContext* ctx) {
    int right_pad_idx[8] = {0, 2, 4, 6, 1, 3, 5, 7};
    int out_pad_idx[4] = {0, 1, 2, 3};

    int tile_w = w_out >> 2;
    int cnt_remain = w_out % 4;

    unsigned int size_right_remain = (unsigned int)(w_in - (tile_w << 3));

    uint32x4_t vmask_rp1 = vcgtq_s32(vdupq_n_s32(size_right_remain), vld1q_s32(right_pad_idx));//0 2 4 6
    uint32x4_t vmask_rp2 = vcgtq_s32(vdupq_n_s32(size_right_remain), vld1q_s32(right_pad_idx + 4));//1 3 5 7
    uint32x4_t wmask = vcgtq_s32(vdupq_n_s32(cnt_remain), vld1q_s32(out_pad_idx));//0 1 2 3
    int size_in_channel = w_in * h_in;
    int size_out_channel = w_out * h_out;

    float* zero_ptr = ctx->workspace_data<float>();
    memset(zero_ptr, 0, w_in * sizeof(float));
    float* write_ptr = zero_ptr + w_in;

    unsigned int dmask[12];

    vst1q_u32(dmask, vmask_rp1);
    vst1q_u32(dmask + 4, vmask_rp2);
    vst1q_u32(dmask + 8, wmask);

    for (int n = 0; n < num; ++n) {
        const float *din_batch = din + n * ch_in * size_in_channel;
        float *dout_batch = dout + n * ch_in * size_out_channel;
#pragma omp parallel for
        for (int i = 0; i < ch_in; ++i) {
            const float* din_channel = din_batch + i * size_in_channel;
            float* dout_channel = dout_batch + i * size_out_channel;

            const float *weight_ptr = weights + i * 9;
            float32x4_t wr0 = vld1q_f32(weight_ptr);
            float32x4_t wr1 = vld1q_f32(weight_ptr + 3);
            float32x4_t wr2 = vld1q_f32(weight_ptr + 6);

            float32x4_t vzero = vdupq_n_f32(0.f);

            float32x4_t wbias;
            float bias_c = 0.f;
            if (flag_bias) {
                wbias  = vdupq_n_f32(bias[i]);
                bias_c = bias[i];
            } else {
                wbias = vdupq_n_f32(0.f);
            }

            const float *dr0 = din_channel;
            const float *dr1 = dr0 + w_in;
            const float *dr2 = dr1 + w_in;
            const float *dr3 = dr2 + w_in;
            const float *dr4 = dr3 + w_in;

            const float *din0_ptr = dr0;
            const float *din1_ptr = dr1;
            const float *din2_ptr = dr2;
            const float *din3_ptr = dr3;
            const float *din4_ptr = dr4;

            float *doutr0 = dout_channel;
            float *doutr0_ptr = nullptr;
            float *doutr1_ptr = nullptr;

#ifdef __aarch64__
            for (int i = 0; i < h_out; i += 2){
                din0_ptr = dr0;
                din1_ptr = dr1;
                din2_ptr = dr2;
                din3_ptr = dr3;
                din4_ptr = dr4;

                doutr0_ptr = doutr0;
                doutr1_ptr = doutr0 + w_out;

                dr0 = dr4;
                dr1 = dr0 + w_in;
                dr2 = dr1 + w_in;
                dr3 = dr2 + w_in;
                dr4 = dr3 + w_in;

                //! process bottom pad
                if (i + 4 >= h_in) {
                    switch (i + 4 - h_in) {
                        case 4:
                            din1_ptr = zero_ptr;
                        case 3:
                            din2_ptr = zero_ptr;
                        case 2:
                            din3_ptr = zero_ptr;
                        case 1:
                            din4_ptr = zero_ptr;
                        case 0:
                            din4_ptr = zero_ptr;
                        default:
                            break;
                    }
                }
                //! process output pad
                if (i + 2 > h_out) {
                    doutr1_ptr = write_ptr;
                }
                int cnt = tile_w;
                asm volatile (
                //top
                // Load up 12 elements (3 vectors) from each of 8 sources.
                "0:                                      \n"
                    "prfm pldl1keep, [%[inptr0]]             \n"
                    "prfm pldl1keep, [%[inptr1]]             \n"
                    "prfm pldl1keep, [%[inptr2]]             \n"
                    "prfm pldl1keep, [%[inptr3]]             \n"
                    "prfm pldl1keep, [%[inptr4]]             \n"
                    "ld2  {v0.4s, v1.4s}, [%[inptr0]], #32    \n" //v0={0,2,4,6} v1={1,3,5,7}
                    "ld2  {v2.4s, v3.4s}, [%[inptr1]], #32    \n"
                    "ld2  {v4.4s, v5.4s}, [%[inptr2]], #32    \n"
                    "ld2  {v6.4s, v7.4s}, [%[inptr3]], #32    \n"
                    "ld2  {v8.4s, v9.4s}, [%[inptr4]], #32    \n"

                    "and  v16.16b, %[vbias].16b, %[vbias].16b  \n" //v10 = vbias
                    "and  v17.16b, %[vbias].16b, %[vbias].16b  \n" //v16 = vbias

                    "ld1 {v15.4s}, [%[inptr0]]                 \n"
                    "ld1 {v18.4s}, [%[inptr1]]                 \n"
                    "ld1 {v19.4s}, [%[inptr2]]                 \n"
                    "ld1 {v20.4s}, [%[inptr3]]                 \n"
                    "ld1 {v21.4s}, [%[inptr4]]                 \n"

                    "ext  v10.16b, v0.16b, v15.16b, #4     \n" // v10 = {2,4,6,8}
                //mid
                    "2:                                          \n"
                    //r0
                    "fmul v11.4s, v0.4s, %[w0].s[0]            \n" // {0,2,4,6} * w00
                    "fmul v12.4s, v1.4s, %[w0].s[1]            \n" // {1,3,5,7} * w01
                    "fmla v16.4s, v10.4s, %[w0].s[2]            \n" // {2,4,6,8} * w02

                    "ext  v10.16b, v2.16b, v18.16b, #4     \n" // v10 = {2,4,6,8}
                    "ld2  {v0.4s, v1.4s}, [%[inptr0]], #32    \n" //v0={0,2,4,6} v1={1,3,5,7}

                    //r1
                    "fmla v11.4s, v2.4s, %[w1].s[0]            \n" // {0,2,4,6} * w00
                    "fmla v12.4s, v3.4s, %[w1].s[1]            \n" // {1,3,5,7} * w01
                    "fmla v16.4s, v10.4s, %[w1].s[2]            \n" // {2,4,6,8} * w02

                    "ext  v10.16b, v4.16b, v19.16b, #4     \n" // v10 = {2,4,6,8}

                    "ld2  {v2.4s, v3.4s}, [%[inptr1]], #32    \n"

                    //r2
                    "fmul v13.4s, v4.4s, %[w0].s[0]            \n" // {0,2,4,6} * w00
                    "fmla v11.4s, v4.4s, %[w2].s[0]            \n" // {0,2,4,6} * w00

                    "fmul v14.4s, v5.4s, %[w0].s[1]            \n" // {1,3,5,7} * w01
                    "fmla v12.4s, v5.4s, %[w2].s[1]            \n" // {1,3,5,7} * w01

                    "fmla v17.4s, v10.4s, %[w0].s[2]            \n" // {2,4,6,8} * w02
                    "fmla v16.4s, v10.4s, %[w2].s[2]            \n" // {2,4,6,8} * w02

                    "ext  v10.16b, v6.16b, v20.16b, #4     \n" // v10 = {2,4,6,8}

                    "ld2  {v4.4s, v5.4s}, [%[inptr2]], #32    \n"

                    //r3
                    "fmla v13.4s, v6.4s, %[w1].s[0]            \n" // {0,2,4,6} * w00
                    "fmla v14.4s, v7.4s, %[w1].s[1]            \n" // {1,3,5,7} * w01
                    "fmla v17.4s, v10.4s, %[w1].s[2]            \n" // {2,4,6,8} * w02

                    "ext  v10.16b, v8.16b, v21.16b, #4     \n" // v10 = {2,4,6,8}

                    "ld2  {v6.4s, v7.4s}, [%[inptr3]], #32    \n"

                    "fadd v16.4s, v16.4s, v11.4s                  \n"
                    "fadd v16.4s, v16.4s, v12.4s                  \n"

                    //r4
                    "fmla v13.4s, v8.4s, %[w2].s[0]            \n" // {0,2,4,6} * w00
                    "fmla v14.4s, v9.4s, %[w2].s[1]            \n" // {1,3,5,7} * w01
                    "fmla v17.4s, v10.4s, %[w2].s[2]            \n" // {2,4,6,8} * w02

                    "ld2  {v8.4s, v9.4s}, [%[inptr4]], #32    \n"
                    "ld1 {v15.4s}, [%[inptr0]]                 \n"
                    "ld1 {v18.4s}, [%[inptr1]]                 \n"
                    "st1 {v16.4s}, [%[outptr0]], #16              \n"

                    "fadd v17.4s, v17.4s, v13.4s                  \n"

                    "ld1 {v19.4s}, [%[inptr2]]                 \n"
                    "ld1 {v20.4s}, [%[inptr3]]                 \n"
                    "ld1 {v21.4s}, [%[inptr4]]                 \n"

                    "fadd v17.4s, v17.4s, v14.4s                  \n"

                    "ext  v10.16b, v0.16b, v15.16b, #4     \n" // v10 = {2,4,6,8}
                    "and  v16.16b, %[vbias].16b, %[vbias].16b  \n" //v10 = vbias
                    "subs %[cnt], %[cnt], #1                    \n"

                    "st1 {v17.4s}, [%[outptr1]], #16              \n"

                    "and  v17.16b, %[vbias].16b, %[vbias].16b  \n" //v16 = vbias

                    "bne  2b                                    \n"

                //right
                    "1:                                          \n"
                    "cmp %[remain], #1                           \n"
                    "blt 4f                                     \n"
                    "3:                                         \n"
                    "bif  v0.16b, %[vzero].16b, %[mask1].16b    \n" //pipei
                    "bif  v1.16b, %[vzero].16b, %[mask2].16b    \n" //pipei

                    "bif  v2.16b, %[vzero].16b, %[mask1].16b    \n" //pipei
                    "bif  v3.16b, %[vzero].16b, %[mask2].16b    \n" //pipei

                    "bif  v4.16b, %[vzero].16b, %[mask1].16b    \n" //pipei
                    "bif  v5.16b, %[vzero].16b, %[mask2].16b    \n" //pipei

                    "ext  v10.16b, v0.16b, %[vzero].16b, #4     \n" // v10 = {2,4,6,8}

                    "bif  v6.16b, %[vzero].16b, %[mask1].16b    \n" //pipei
                    "bif  v7.16b, %[vzero].16b, %[mask2].16b    \n" //pipei

                    //r0
                    "fmul v11.4s, v0.4s, %[w0].s[0]            \n" // {0,2,4,6} * w00
                    "fmul v12.4s, v1.4s, %[w0].s[1]            \n" // {1,3,5,7} * w01
                    "fmla v16.4s, v10.4s, %[w0].s[2]            \n" // {2,4,6,8} * w02

                    "ext  v10.16b, v2.16b, %[vzero].16b, #4     \n" // v10 = {2,4,6,8}
                    "bif  v8.16b, %[vzero].16b, %[mask1].16b    \n" //pipei
                    "bif  v9.16b, %[vzero].16b, %[mask2].16b    \n" //pipei

                    //r1
                    "fmla v11.4s, v2.4s, %[w1].s[0]            \n" // {0,2,4,6} * w00
                    "fmla v12.4s, v3.4s, %[w1].s[1]            \n" // {1,3,5,7} * w01
                    "fmla v16.4s, v10.4s, %[w1].s[2]            \n" // {2,4,6,8} * w02

                    "ext  v10.16b, v4.16b, %[vzero].16b, #4     \n" // v10 = {2,4,6,8}

                    //r2
                    "fmul v13.4s, v4.4s, %[w0].s[0]            \n" // {0,2,4,6} * w00
                    "fmla v11.4s, v4.4s, %[w2].s[0]            \n" // {0,2,4,6} * w00

                    "fmul v14.4s, v5.4s, %[w0].s[1]            \n" // {1,3,5,7} * w01
                    "fmla v12.4s, v5.4s, %[w2].s[1]            \n" // {1,3,5,7} * w01

                    "fmla v17.4s, v10.4s, %[w0].s[2]            \n" // {2,4,6,8} * w02
                    "fmla v16.4s, v10.4s, %[w2].s[2]            \n" // {2,4,6,8} * w02

                    "ext  v10.16b, v6.16b, %[vzero].16b, #4     \n" // v10 = {2,4,6,8}

                    //r3
                    "fmla v13.4s, v6.4s, %[w1].s[0]            \n" // {0,2,4,6} * w00
                    "fmla v14.4s, v7.4s, %[w1].s[1]            \n" // {1,3,5,7} * w01
                    "fmla v17.4s, v10.4s, %[w1].s[2]            \n" // {2,4,6,8} * w02

                    "ext  v10.16b, v8.16b, %[vzero].16b, #4     \n" // v10 = {2,4,6,8}
                    "ld1 {v0.4s}, [%[outptr0]]                  \n"

                    "fadd v16.4s, v16.4s, v11.4s                  \n"
                    "fadd v16.4s, v16.4s, v12.4s                  \n"
                    "ld1 {v1.4s}, [%[outptr1]]                  \n"

                    //r4
                    "fmla v13.4s, v8.4s, %[w2].s[0]            \n" // {0,2,4,6} * w00
                    "fmla v14.4s, v9.4s, %[w2].s[1]            \n" // {1,3,5,7} * w01
                    "fmla v17.4s, v10.4s, %[w2].s[2]            \n" // {2,4,6,8} * w02

                    "bif  v16.16b, v0.16b, %[wmask].16b    \n" //pipei

                    "fadd v17.4s, v17.4s, v13.4s                  \n"

                    "st1 {v16.4s}, [%[outptr0]], #16              \n"

                    "fadd v17.4s, v17.4s, v14.4s                  \n"

                    "bif  v17.16b, v1.16b, %[wmask].16b    \n" //pipei

                    "st1 {v17.4s}, [%[outptr1]], #16              \n"
                "4:                                          \n"
                : [inptr0] "+r"(din0_ptr), [inptr1] "+r"(din1_ptr), [inptr2] "+r"(din2_ptr), [inptr3] "+r"(din3_ptr), \
                  [inptr4] "+r"(din4_ptr), [outptr0] "+r"(doutr0_ptr), [outptr1] "+r"(doutr1_ptr), \
                  [cnt] "+r" (cnt)
                : [vzero] "w" (vzero), [w0] "w" (wr0), [w1] "w" (wr1), [w2] "w" (wr2), [remain] "r" (cnt_remain), \
                  [mask1] "w" (vmask_rp1), [mask2] "w" (vmask_rp2), [wmask] "w" (wmask), [vbias] "w" (wbias)
                : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", \
                  "v15", "v16", "v17", "v18", "v19", "v20", "v21"
                );
                doutr0 = doutr0 + 2 * w_out;
            }
#else
            for (int i = 0; i < h_out; i++){
                din0_ptr = dr0;
                din1_ptr = dr1;
                din2_ptr = dr2;

                doutr0_ptr = doutr0;

                dr0 = dr2;
                dr1 = dr0 + w_in;
                dr2 = dr1 + w_in;

                //! process bottom pad
                if (i + 2 > h_in) {
                    switch (i + 2 - h_in) {
                        case 2:
                            din1_ptr = zero_ptr;
                        case 1:
                            din2_ptr = zero_ptr;
                        default:
                            break;
                    }
                }
                int cnt = tile_w;
                unsigned int* mask_ptr = dmask;
                asm volatile (
                // Load up 12 elements (3 vectors) from each of 8 sources.
                    "0:                                                     \n"
                        "vmov.u32 q9, #0                                \n"
                        "vld2.32  {d20-d23}, [%[din0_ptr]]!             @ load din r1\n"
                        "vld2.32  {d24-d27}, [%[din1_ptr]]!             @ load din r1\n"
                        "vld2.32  {d28-d31}, [%[din2_ptr]]!             @ load din r1\n"
                        "pld [%[din0_ptr]]                              @ preload data\n"
                        "pld [%[din1_ptr]]                              @ preload data\n"
                        "pld [%[din2_ptr]]                              @ preload data\n"

                        "vld1.32  {d16}, [%[din0_ptr]]                  @ load din r0\n"//q2={8,10,12,14}

                        "vdup.32 q3, %[bias]                            @ and \n" //q10 = vbias
                    //mid
                        "2:                                             \n"
                        "vext.32  q6, q10, q8, #1                       @ shift left 1 \n" //q6 = {2,4,6,8}
                        "vld1.32 {d16}, [%[din1_ptr]]                   @ load din r1\n"//q2={8,10,12,14}

                        "vmul.f32 q4, q10, %e[wr0][0]                   @ mul weight 0, out0\n"// q0 * w00
                        "vmul.f32 q5, q11, %e[wr0][1]                   @ mul weight 0, out0\n"// q1 * w01
                        "vmla.f32 q3,  q6, %f[wr0][0]                   @ mul weight 0, out0\n"// q6 * w02

                        "vext.32  q7, q12, q8, #1                       @ shift left 1 \n" //q6 = {2,4,6,8}
                        "vld1.32 {d16}, [%[din2_ptr]]                   @ load din r1\n"//q2={8,10,12,14}

                        "vld2.32  {d20-d23}, [%[din0_ptr]]!             @ load din r0\n"//v0={0,2,4,6} v1={1,3,5,7}

                        "vmla.f32 q4, q12, %e[wr1][0]                   @ mul weight 1, out0\n"// q0 * w00
                        "vmla.f32 q5, q13, %e[wr1][1]                   @ mul weight 1, out0\n"// q1 * w01
                        "vmla.f32 q3,  q7, %f[wr1][0]                   @ mul weight 1, out0\n"// q6 * w02

                        "vext.32  q6, q14, q8, #1                       @ shift left 1 \n" //q6 = {2,4,6,8}

                        "vld2.32  {d24-d27}, [%[din1_ptr]]!             @ load din r1\n"//v0={0,2,4,6} v1={1,3,5,7}

                        "vmla.f32 q4, q14, %e[wr2][0]                   @ mul weight 2, out0\n"// q0 * w00
                        "vmla.f32 q5, q15, %e[wr2][1]                   @ mul weight 2, out0\n"// q1 * w01
                        "vmla.f32 q3,  q6, %f[wr2][0]                   @ mul weight 2, out0\n"// q6 * w02

                        "vld2.32  {d28-d31}, [%[din2_ptr]]!             @ load din r2\n"//v4={0,2,4,6} v5={1,3,5,7}

                        "vadd.f32 q3, q3, q4                            @ add \n"
                        "vadd.f32 q3, q3, q5                            @ add \n"

                        "subs %[cnt], #1                                \n"

                        "vld1.32  {d16}, [%[din0_ptr]]                  @ load din r0\n"//q2={8,10,12,14}

                        "vst1.32 {d6-d7}, [%[outptr]]!                  \n"

                        "vdup.32  q3, %[bias]                           @ and \n" //q10 = vbias
                        "bne  2b                                        \n"

                        //right
                        "1:                                             \n"
                        "cmp %[remain], #1                              \n"
                        "blt 3f                                         \n"

                        "vld1.f32   {d12-d15}, [%[mask_ptr]]!           @ load mask\n"

                        "vbif q10, q9, q6                               @ bit select, deal with right pad\n"
                        "vbif q11, q9, q7                               @ bit select, deal with right pad\n"
                        "vbif q12, q9, q6                               @ bit select, deal with right pad\n"
                        "vbif q13, q9, q7                               @ bit select, deal with right pad\n"
                        "vbif q14, q9, q6                               @ bit select, deal with right pad\n"
                        "vbif q15, q9, q7                               @ bit select, deal with right pad\n"

                        "vext.32 q6, q10, q9, #1                        @ shift left 1 \n" //q6 = {2,4,6,8}
                        "vext.32 q7, q12, q9, #1                        @ shift left 1 \n" //q6 = {2,4,6,8}

                        "vmul.f32 q4, q10, %e[wr0][0]                   @ mul weight 0, out0\n"// q0 * w00
                        "vmul.f32 q5, q11, %e[wr0][1]                   @ mul weight 0, out0\n"// q1 * w01
                        "vmla.f32 q3,  q6, %f[wr0][0]                   @ mul weight 0, out0\n"// q6 * w02

                        "vext.32 q6, q14, q9, #1                        @ shift left 1 \n" //q6 = {2,4,6,8}
                        "vld1.f32   {d20-d21}, [%[outptr]]              @ load output\n"

                        "vmla.f32 q4, q12, %e[wr1][0]                   @ mul weight 1, out0\n"// q0 * w00
                        "vmla.f32 q5, q13, %e[wr1][1]                   @ mul weight 1, out0\n"// q1 * w01
                        "vmla.f32 q3,  q7, %f[wr1][0]                   @ mul weight 1, out0\n"// q6 * w02

                        "vld1.f32   {d22-d23}, [%[mask_ptr]]            @ load mask\n"

                        "vmla.f32 q4, q14, %e[wr2][0]                   @ mul weight 2, out0\n"// q0 * w00
                        "vmla.f32 q5, q15, %e[wr2][1]                   @ mul weight 2, out0\n"// q1 * w01
                        "vmla.f32 q3,  q6, %f[wr2][0]                   @ mul weight 2, out0\n"// q6 * w02

                        "vadd.f32 q3, q3, q4                            @ add \n"
                        "vadd.f32 q3, q3, q5                            @ add \n"

                        "vbif.f32 q3, q10, q11                          @ write mask\n"

                        "vst1.32 {d6-d7}, [%[outptr]]!                  \n"
                        "3:                                             \n"
                : [din0_ptr] "+r"(din0_ptr), [din1_ptr] "+r"(din1_ptr), [din2_ptr] "+r"(din2_ptr), \
                    [outptr] "+r"(doutr0_ptr), [cnt] "+r" (cnt), [mask_ptr] "+r"(mask_ptr)
                : [remain] "r" (cnt_remain), [wr0] "w" (wr0), [wr1] "w" (wr1), [wr2] "w" (wr2), \
                    [bias] "r"(bias_c)
                : "cc", "memory", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
                );

                doutr0 = doutr0 + w_out;

            }
#endif
        }
    }
}

//4line
void conv_depthwise_3x3s1p0_bias_relu(float* dout, const float* din, \
    const float* weights, const float* bias, bool flag_bias, \
    const int num, const int ch_in, const int h_in, const int w_in, \
    const int h_out, const int w_out, ARMContext* ctx) {

    //! pad is done implicit
    const float zero[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    //! for 4x6 convolution window
    const unsigned int right_pad_idx[8] = {5, 4, 3, 2, 1, 0, 0, 0};

    float* zero_ptr = ctx->workspace_data<float>();
    memset(zero_ptr, 0, w_in * sizeof(float));
    float* write_ptr = zero_ptr + w_in;

    int size_in_channel = w_in * h_in;
    int size_out_channel = w_out * h_out;
    int w_stride = 9;

    int tile_w = w_out >> 2;
    int remain = w_out % 4;

    unsigned int size_pad_right = (unsigned int)(6 + (tile_w << 2) - w_in);
    const int remian_idx[4] = {0, 1, 2, 3};

    uint32x4_t vmask_rp1 = vcgeq_u32(vld1q_u32(right_pad_idx), vdupq_n_u32(size_pad_right));
    uint32x4_t vmask_rp2 = vcgeq_u32(vld1q_u32(right_pad_idx + 4), vdupq_n_u32(size_pad_right));
    uint32x4_t vmask_result = vcgtq_s32(vdupq_n_s32(remain), vld1q_s32(remian_idx));

    unsigned int vmask[8];
    vst1q_u32(vmask, vmask_rp1);
    vst1q_u32(vmask + 4, vmask_rp2);

    unsigned int rmask[4];
    vst1q_u32(rmask, vmask_result);

    float32x4_t vzero = vdupq_n_f32(0.f);

    for (int n = 0; n < num; ++n) {
        const float *din_batch = din + n * ch_in * size_in_channel;
        float *dout_batch = dout + n * ch_in * size_out_channel;
#pragma omp parallel for
#ifdef __aarch64__
        for (int c = 0; c < ch_in; c++) {
            float* dout_ptr = dout_batch + c * size_out_channel;

            const float* din_ch_ptr = din_batch + c * size_in_channel;

            float bias_val = flag_bias ? bias[c] : 0.f;
            float vbias[4] = {bias_val, bias_val, bias_val, bias_val};

            const float* wei_ptr = weights + c * w_stride;

            float32x4_t wr0 = vld1q_f32(wei_ptr);
            float32x4_t wr1 = vld1q_f32(wei_ptr + 3);
            float32x4_t wr2 = vld1q_f32(wei_ptr + 6);
            // wr0 = vsetq_lane_f32(0.f, wr0, 3);
            // wr1 = vsetq_lane_f32(0.f, wr1, 3);
            // wr2 = vsetq_lane_f32(0.f, wr2, 3);

            float *doutr0 = dout_ptr;
            float *doutr1 = doutr0 + w_out;
            float *doutr2 = doutr1 + w_out;
            float *doutr3 = doutr2 + w_out;

            const float *dr0 = din_ch_ptr;
            const float *dr1 = dr0 + w_in;
            const float *dr2 = dr1 + w_in;
            const float *dr3 = dr2 + w_in;
            const float *dr4 = dr3 + w_in;
            const float *dr5 = dr4 + w_in;

            const float *din_ptr0 = dr0;
            const float *din_ptr1 = dr1;
            const float *din_ptr2 = dr2;
            const float *din_ptr3 = dr3;
            const float *din_ptr4 = dr4;
            const float *din_ptr5 = dr5;

            for (int i = 0; i < h_out; i += 4){
                //! process top pad pad_h = 1
                din_ptr0 = dr0;
                din_ptr1 = dr1;
                din_ptr2 = dr2;
                din_ptr3 = dr3;
                din_ptr4 = dr4;
                din_ptr5 = dr5;

                doutr0 = dout_ptr;
                doutr1 = doutr0 + w_out;
                doutr2 = doutr1 + w_out;
                doutr3 = doutr2 + w_out;

                dr0 = dr4;
                dr1 = dr5;
                dr2 = dr1 + w_in;
                dr3 = dr2 + w_in;
                dr4 = dr3 + w_in;
                dr5 = dr4 + w_in;

                //! process bottom pad
                if (i + 5 >= h_in) {
                    switch (i + 5 - h_in) {
                        case 5:
                            din_ptr1 = zero_ptr;
                        case 4:
                            din_ptr2 = zero_ptr;
                        case 3:
                            din_ptr3 = zero_ptr;
                        case 2:
                            din_ptr4 = zero_ptr;
                        case 1:
                            din_ptr5 = zero_ptr;
                        case 0:
                            din_ptr5 = zero_ptr;
                        default:
                            break;
                    }
                }
                //! process bottom remain
                if (i + 4 > h_out) {
                    switch (i + 4 - h_out) {
                        case 3:
                            doutr1 = write_ptr;
                        case 2:
                            doutr2 = write_ptr;
                        case 1:
                            doutr3 = write_ptr;
                        default:
                            break;
                    }
                }

                int cnt = tile_w;
                asm volatile(
                    "PRFM PLDL1KEEP, [%[din_ptr0]] \n"
                    "PRFM PLDL1KEEP, [%[din_ptr1]] \n"
                    "PRFM PLDL1KEEP, [%[din_ptr2]] \n"
                    "PRFM PLDL1KEEP, [%[din_ptr3]] \n"
                    "PRFM PLDL1KEEP, [%[din_ptr4]] \n"
                    "PRFM PLDL1KEEP, [%[din_ptr5]] \n"
                    "movi   v21.4s, #0x0\n"                         /* out0 = 0 */

                    "ld1 {v0.4s}, [%[din_ptr0]], #16   \n" /*vld1q_f32(din_ptr0)*/
                    "ld1 {v2.4s}, [%[din_ptr1]], #16   \n" /*vld1q_f32(din_ptr0)*/
                    "ld1 {v4.4s}, [%[din_ptr2]], #16   \n" /*vld1q_f32(din_ptr0)*/
                    "ld1 {v6.4s}, [%[din_ptr3]], #16   \n" /*vld1q_f32(din_ptr0)*/

                    "ld1 {v1.4s}, [%[din_ptr0]]   \n" /*vld1q_f32(din_ptr0)*/
                    "ld1 {v3.4s}, [%[din_ptr1]]   \n" /*vld1q_f32(din_ptr0)*/
                    "ld1 {v5.4s}, [%[din_ptr2]]   \n" /*vld1q_f32(din_ptr0)*/
                    "ld1 {v7.4s}, [%[din_ptr3]]   \n" /*vld1q_f32(din_ptr0)*/

                    "ld1 {v12.4s}, [%[bias_val]]     \n" /*vdupq_n_f32(bias_val)*/
                    "ld1 {v13.4s}, [%[bias_val]]      \n" /*vdupq_n_f32(bias_val)*/
                    "ld1 {v14.4s}, [%[bias_val]]      \n" /*vdupq_n_f32(bias_val)*/
                    "ld1 {v15.4s}, [%[bias_val]]      \n" /*vdupq_n_f32(bias_val)*/

                    "ext  v16.16b, v0.16b, v1.16b, #4 \n" /* v16 = 1234 */
                    "ext  v17.16b, v0.16b, v1.16b, #8 \n" /* v17 = 2345 */

                //mid
                    "4:                             \n"
                    //r0
                    "fmla v12.4s ,  v0.4s,  %[w0].s[0]\n" /* outr00 += din0_0123 * w0[0]*/

                    "ld1 {v8.4s}, [%[din_ptr4]], #16   \n" /*vld1q_f32(din_ptr0)*/
                    "ld1 {v10.4s}, [%[din_ptr5]], #16   \n" /*vld1q_f32(din_ptr0)*/
                    "ld1 {v0.4s}, [%[din_ptr0]], #16   \n" /*vld1q_f32(din_ptr0)*/

                    "fmla v12.4s ,  v16.4s,  %[w0].s[1]\n" /* outr00 += din0_1234 * w0[1]*/

                    "ld1 {v9.4s}, [%[din_ptr4]]   \n" /*vld1q_f32(din_ptr0)*/
                    "ld1 {v11.4s}, [%[din_ptr5]]   \n" /*vld1q_f32(din_ptr0)*/
                    "ld1 {v1.4s}, [%[din_ptr0]]        \n" /*vld1q_f32(din_ptr0)*/

                    "fmla v12.4s ,  v17.4s,  %[w0].s[2]\n" /* outr00 += din0_2345 * w0[2]*/

                    "ext  v16.16b, v2.16b, v3.16b, #4 \n" /* v16 = 1234*/
                    "ext  v17.16b, v2.16b, v3.16b, #8 \n" /* v16 = 2345 */

                    //r1
                    "fmla v13.4s ,  v2.4s,  %[w0].s[0]\n" /* outr00 += din0_0123 * w0[0]*/
                    "fmla v12.4s ,  v2.4s,  %[w1].s[0]\n" /* outr00 += din0_0123 * w0[0]*/

                    "ld1 {v2.4s}, [%[din_ptr1]], #16   \n" /*vld1q_f32(din_ptr0)*/

                    "fmla v13.4s ,  v16.4s,  %[w0].s[1]\n" /* outr00 += din0_1234 * w0[1]*/
                    "fmla v12.4s ,  v16.4s,  %[w1].s[1]\n" /* outr00 += din0_1234 * w0[1]*/

                    "ld1 {v3.4s}, [%[din_ptr1]]   \n" /*vld1q_f32(din_ptr0)*/

                    "fmla v13.4s ,  v17.4s,  %[w0].s[2]\n" /* outr00 += din0_2345 * w0[2]*/
                    "fmla v12.4s ,  v17.4s,  %[w1].s[2]\n" /* outr00 += din0_2345 * w0[2]*/

                    "ext  v16.16b, v4.16b, v5.16b, #4 \n" /* v16 = 1234*/
                    "ext  v17.16b, v4.16b, v5.16b, #8 \n" /* v16 = 2345 */

                    //r2
                    "fmla v14.4s ,  v4.4s,  %[w0].s[0]\n" /* outr00 += din0_0123 * w0[0]*/
                    "fmla v13.4s ,  v4.4s,  %[w1].s[0]\n" /* outr00 += din0_0123 * w0[0]*/
                    "fmla v12.4s ,  v4.4s,  %[w2].s[0]\n" /* outr00 += din0_0123 * w0[0]*/

                    "ld1 {v4.4s}, [%[din_ptr2]], #16   \n" /*vld1q_f32(din_ptr0)*/

                    "fmla v14.4s ,  v16.4s,  %[w0].s[1]\n" /* outr00 += din0_1234 * w0[1]*/
                    "fmla v13.4s ,  v16.4s,  %[w1].s[1]\n" /* outr00 += din0_1234 * w0[1]*/
                    "fmla v12.4s ,  v16.4s,  %[w2].s[1]\n" /* outr00 += din0_1234 * w0[1]*/

                    "ld1 {v5.4s}, [%[din_ptr2]]   \n" /*vld1q_f32(din_ptr0)*/

                    "fmla v14.4s ,  v17.4s,  %[w0].s[2]\n" /* outr00 += din0_2345 * w0[2]*/
                    "fmla v13.4s ,  v17.4s,  %[w1].s[2]\n" /* outr00 += din0_2345 * w0[2]*/
                    "fmla v12.4s ,  v17.4s,  %[w2].s[2]\n" /* outr00 += din0_2345 * w0[2]*/

                    "ext  v16.16b, v6.16b, v7.16b, #4 \n" /* v16 = 1234*/
                    "ext  v17.16b, v6.16b, v7.16b, #8 \n" /* v16 = 2345 */

                    //r3
                    "fmla v15.4s ,  v6.4s,  %[w0].s[0]\n" /* outr00 += din0_0123 * w0[0]*/
                    "fmla v14.4s ,  v6.4s,  %[w1].s[0]\n" /* outr00 += din0_0123 * w0[0]*/
                    "fmla v13.4s ,  v6.4s,  %[w2].s[0]\n" /* outr00 += din0_0123 * w0[0]*/

                    "fmax v12.4s, v12.4s, %[vzero].4s \n" /* relu */

                    "ld1 {v6.4s}, [%[din_ptr3]], #16   \n" /*vld1q_f32(din_ptr0)*/

                    "fmla v15.4s ,  v16.4s,  %[w0].s[1]\n" /* outr00 += din0_1234 * w0[1]*/
                    "fmla v14.4s ,  v16.4s,  %[w1].s[1]\n" /* outr00 += din0_1234 * w0[1]*/
                    "fmla v13.4s ,  v16.4s,  %[w2].s[1]\n" /* outr00 += din0_1234 * w0[1]*/

                    "st1 {v12.4s}, [%[doutr0]], #16     \n"
                    "ld1 {v7.4s}, [%[din_ptr3]]   \n" /*vld1q_f32(din_ptr0)*/

                    "fmla v15.4s ,  v17.4s,  %[w0].s[2]\n" /* outr00 += din0_2345 * w0[2]*/
                    "fmla v14.4s ,  v17.4s,  %[w1].s[2]\n" /* outr00 += din0_2345 * w0[2]*/
                    "fmla v13.4s ,  v17.4s,  %[w2].s[2]\n" /* outr00 += din0_2345 * w0[2]*/

                    "ext  v16.16b, v8.16b, v9.16b, #4 \n" /* v16 = 1234*/
                    "ext  v17.16b, v8.16b, v9.16b, #8 \n" /* v16 = 2345 */
                    "ld1 {v12.4s}, [%[bias_val]]      \n" /*vdupq_n_f32(bias_val)*/

                    //r4
                    "fmla v15.4s ,  v8.4s,  %[w1].s[0]\n" /* outr00 += din0_0123 * w0[0]*/
                    "fmla v14.4s ,  v8.4s,  %[w2].s[0]\n" /* outr00 += din0_0123 * w0[0]*/

                    "fmax v13.4s, v13.4s, %[vzero].4s \n" /* relu */

                    "fmla v15.4s ,  v16.4s,  %[w1].s[1]\n" /* outr00 += din0_1234 * w0[1]*/
                    "fmla v14.4s ,  v16.4s,  %[w2].s[1]\n" /* outr00 += din0_1234 * w0[1]*/

                    "st1 {v13.4s}, [%[doutr1]], #16     \n"

                    "fmla v15.4s ,  v17.4s,  %[w1].s[2]\n" /* outr00 += din0_2345 * w0[2]*/
                    "fmla v14.4s ,  v17.4s,  %[w2].s[2]\n" /* outr00 += din0_2345 * w0[2]*/

                    "ext  v16.16b, v10.16b, v11.16b, #4 \n" /* v16 = 1234*/
                    "ext  v17.16b, v10.16b, v11.16b, #8 \n" /* v16 = 2345 */
                    "ld1 {v13.4s}, [%[bias_val]]      \n" /*vdupq_n_f32(bias_val)*/
                    //r5
                    "fmla v15.4s ,  v10.4s,  %[w2].s[0]\n" /* outr00 += din0_0123 * w0[0]*/

                    "fmax v14.4s, v14.4s, %[vzero].4s \n" /* relu */

                    "fmla v15.4s ,  v16.4s,  %[w2].s[1]\n" /* outr00 += din0_1234 * w0[1]*/

                    "st1 {v14.4s}, [%[doutr2]], #16     \n"

                    "fmla v15.4s ,  v17.4s,  %[w2].s[2]\n" /* outr00 += din0_2345 * w0[2]*/

                    "ext  v16.16b, v0.16b, v1.16b, #4 \n" /* v16 = 1234*/
                    "ext  v17.16b, v0.16b, v1.16b, #8 \n" /* v16 = 2345 */
                    "ld1 {v14.4s}, [%[bias_val]]      \n" /*vdupq_n_f32(bias_val)*/
                    "fmax v15.4s, v15.4s, %[vzero].4s \n" /* relu */

                    "subs %[cnt], %[cnt], #1 \n"

                    "st1 {v15.4s}, [%[doutr3]], #16     \n"
                    "ld1 {v15.4s}, [%[bias_val]]      \n" /*vdupq_n_f32(bias_val)*/

                    "bne 4b \n"

                //right
                    "5:                             \n"
                    "cmp  %[remain], #1             \n"
                    "blt 0f                         \n"
                    "ld1 {v18.4s, v19.4s}, [%[vmask]]         \n"
                    "ld1 {v22.4s}, [%[doutr0]]         \n"
                    "ld1 {v23.4s}, [%[doutr1]]         \n"
                    "ld1 {v24.4s}, [%[doutr2]]         \n"
                    "ld1 {v25.4s}, [%[doutr3]]         \n"

                    "bif v0.16b, %[vzero].16b, v18.16b \n"
                    "bif v1.16b, %[vzero].16b, v19.16b \n"
                    "bif v2.16b, %[vzero].16b, v18.16b \n"
                    "bif v3.16b, %[vzero].16b, v19.16b \n"

                    "ext  v16.16b, v0.16b, v1.16b, #4 \n" /* v16 = 1234*/
                    "ext  v17.16b, v0.16b, v1.16b, #8 \n" /* v16 = 2345 */
                    "ld1 {v8.4s}, [%[din_ptr4]], #16   \n" /*vld1q_f32(din_ptr0)*/
                    "ld1 {v10.4s}, [%[din_ptr5]], #16   \n" /*vld1q_f32(din_ptr0)*/
                    "ld1 {v9.4s}, [%[din_ptr4]]   \n" /*vld1q_f32(din_ptr0)*/
                    "ld1 {v11.4s}, [%[din_ptr5]]   \n" /*vld1q_f32(din_ptr0)*/

                    //r0
                    "fmla v12.4s,  v0.4s,  %[w0].s[0]\n" /* outr00 += din0_0123 * w0[0]*/

                    "bif v4.16b, %[vzero].16b, v18.16b \n"
                    "bif v5.16b, %[vzero].16b, v19.16b \n"
                    "bif v6.16b, %[vzero].16b, v18.16b \n"
                    "bif v7.16b, %[vzero].16b, v19.16b \n"

                    "fmla v12.4s,  v16.4s,  %[w0].s[1]\n" /* outr00 += din0_1234 * w0[1]*/

                    "bif v8.16b, %[vzero].16b, v18.16b \n"
                    "bif v9.16b, %[vzero].16b, v19.16b \n"
                    "bif v10.16b, %[vzero].16b, v18.16b \n"
                    "bif v11.16b, %[vzero].16b, v19.16b \n"

                    "fmla v12.4s ,  v17.4s,  %[w0].s[2]\n" /* outr00 += din0_2345 * w0[2]*/

                    "ext  v16.16b, v2.16b, v3.16b, #4 \n" /* v16 = 1234*/
                    "ext  v17.16b, v2.16b, v3.16b, #8 \n" /* v16 = 2345 */
                    "ld1 {v18.4s}, [%[rmask]]         \n"

                    //r1
                    "fmla v13.4s ,  v2.4s,  %[w0].s[0]\n" /* outr00 += din0_0123 * w0[0]*/
                    "fmla v12.4s ,  v2.4s,  %[w1].s[0]\n" /* outr00 += din0_0123 * w0[0]*/

                    "fmla v13.4s ,  v16.4s,  %[w0].s[1]\n" /* outr00 += din0_1234 * w0[1]*/
                    "fmla v12.4s ,  v16.4s,  %[w1].s[1]\n" /* outr00 += din0_1234 * w0[1]*/

                    "fmla v13.4s ,  v17.4s,  %[w0].s[2]\n" /* outr00 += din0_2345 * w0[2]*/
                    "fmla v12.4s ,  v17.4s,  %[w1].s[2]\n" /* outr00 += din0_2345 * w0[2]*/

                    "ext  v16.16b, v4.16b, v5.16b, #4 \n" /* v16 = 1234*/
                    "ext  v17.16b, v4.16b, v5.16b, #8 \n" /* v16 = 2345 */

                    //r2
                    "fmla v14.4s ,  v4.4s,  %[w0].s[0]\n" /* outr00 += din0_0123 * w0[0]*/
                    "fmla v13.4s ,  v4.4s,  %[w1].s[0]\n" /* outr00 += din0_0123 * w0[0]*/
                    "fmla v12.4s ,  v4.4s,  %[w2].s[0]\n" /* outr00 += din0_0123 * w0[0]*/

                    "fmla v14.4s ,  v16.4s,  %[w0].s[1]\n" /* outr00 += din0_1234 * w0[1]*/
                    "fmla v13.4s ,  v16.4s,  %[w1].s[1]\n" /* outr00 += din0_1234 * w0[1]*/
                    "fmla v12.4s ,  v16.4s,  %[w2].s[1]\n" /* outr00 += din0_1234 * w0[1]*/

                    "fmla v14.4s ,  v17.4s,  %[w0].s[2]\n" /* outr00 += din0_2345 * w0[2]*/
                    "fmla v13.4s ,  v17.4s,  %[w1].s[2]\n" /* outr00 += din0_2345 * w0[2]*/
                    "fmla v12.4s ,  v17.4s,  %[w2].s[2]\n" /* outr00 += din0_2345 * w0[2]*/

                    "ext  v16.16b, v6.16b, v7.16b, #4 \n" /* v16 = 1234*/
                    "ext  v17.16b, v6.16b, v7.16b, #8 \n" /* v16 = 2345 */

                    //r3
                    "fmla v15.4s ,  v6.4s,  %[w0].s[0]\n" /* outr00 += din0_0123 * w0[0]*/
                    "fmla v14.4s ,  v6.4s,  %[w1].s[0]\n" /* outr00 += din0_0123 * w0[0]*/
                    "fmla v13.4s ,  v6.4s,  %[w2].s[0]\n" /* outr00 += din0_0123 * w0[0]*/

                    "fmax v12.4s, v12.4s, %[vzero].4s \n" /* relu */

                    "fmla v15.4s ,  v16.4s,  %[w0].s[1]\n" /* outr00 += din0_1234 * w0[1]*/
                    "fmla v14.4s ,  v16.4s,  %[w1].s[1]\n" /* outr00 += din0_1234 * w0[1]*/
                    "fmla v13.4s ,  v16.4s,  %[w2].s[1]\n" /* outr00 += din0_1234 * w0[1]*/

                    "bif v12.16b, v22.16b, v18.16b \n"

                    "fmla v15.4s ,  v17.4s,  %[w0].s[2]\n" /* outr00 += din0_2345 * w0[2]*/
                    "fmla v14.4s ,  v17.4s,  %[w1].s[2]\n" /* outr00 += din0_2345 * w0[2]*/
                    "fmla v13.4s ,  v17.4s,  %[w2].s[2]\n" /* outr00 += din0_2345 * w0[2]*/

                    "ext  v16.16b, v8.16b, v9.16b, #4 \n" /* v16 = 1234*/
                    "ext  v17.16b, v8.16b, v9.16b, #8 \n" /* v16 = 2345 */
                    "st1 {v12.4s}, [%[doutr0]], #16     \n"

                    //r3
                    "fmla v15.4s ,  v8.4s,  %[w1].s[0]\n" /* outr00 += din0_0123 * w0[0]*/
                    "fmla v14.4s ,  v8.4s,  %[w2].s[0]\n" /* outr00 += din0_0123 * w0[0]*/

                    "fmax v13.4s, v13.4s, %[vzero].4s \n" /* relu */

                    "fmla v15.4s ,  v16.4s,  %[w1].s[1]\n" /* outr00 += din0_1234 * w0[1]*/
                    "fmla v14.4s ,  v16.4s,  %[w2].s[1]\n" /* outr00 += din0_1234 * w0[1]*/

                    "bif v13.16b, v23.16b, v18.16b \n"

                    "fmla v15.4s ,  v17.4s,  %[w1].s[2]\n" /* outr00 += din0_2345 * w0[2]*/
                    "fmla v14.4s ,  v17.4s,  %[w2].s[2]\n" /* outr00 += din0_2345 * w0[2]*/
                    "st1 {v13.4s}, [%[doutr1]], #16     \n"

                    "ext  v16.16b, v10.16b, v11.16b, #4 \n" /* v16 = 1234*/
                    "ext  v17.16b, v10.16b, v11.16b, #8 \n" /* v16 = 2345 */

                    //r3
                    "fmla v15.4s ,  v10.4s,  %[w2].s[0]\n" /* outr00 += din0_0123 * w0[0]*/

                    "fmax v14.4s, v14.4s, %[vzero].4s \n" /* relu */

                    "fmla v15.4s ,  v16.4s,  %[w2].s[1]\n" /* outr00 += din0_1234 * w0[1]*/

                    "bif v14.16b, v24.16b, v18.16b \n"

                    "fmla v15.4s ,  v17.4s,  %[w2].s[2]\n" /* outr00 += din0_2345 * w0[2]*/

                    "st1 {v14.4s}, [%[doutr2]], #16     \n"

                    "fmax v15.4s, v15.4s, %[vzero].4s \n" /* relu */

                    "bif v15.16b, v25.16b, v18.16b \n"

                    "st1 {v15.4s}, [%[doutr3]], #16     \n"
                    //end
                    "0:                             \n"
                :[cnt]"+r"(cnt), [din_ptr0] "+r"(din_ptr0), [din_ptr1] "+r"(din_ptr1), [din_ptr2] "+r"(din_ptr2), \
                        [din_ptr3] "+r"(din_ptr3), [din_ptr4] "+r"(din_ptr4), [din_ptr5] "+r"(din_ptr5), \
                        [doutr0] "+r"(doutr0), [doutr1] "+r"(doutr1), [doutr2] "+r"(doutr2), [doutr3] "+r"(doutr3)
                : [w0]"w"(wr0), [w1]"w"(wr1), [w2]"w"(wr2), [bias_val] "r" (vbias), \
                      [vmask] "r" (vmask), [rmask] "r" (rmask), [vzero] "w" (vzero), [remain] "r" (remain)
                :"cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",\
                      "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18",\
                      "v19", "v20", "v21", "v22", "v23", "v24", "v25"
                );
                dout_ptr = dout_ptr + 4 * w_out;
            }
        }
#else
        for (int i = 0; i < ch_in; ++i) {
            const float *din_channel = din_batch + i * size_in_channel;

            const float *weight_ptr = weights + i * 9;
            float32x4_t wr0 = vld1q_f32(weight_ptr);
            float32x4_t wr1 = vld1q_f32(weight_ptr + 3);
            float32x4_t wr2 = vld1q_f32(weight_ptr + 6);
            float bias_val = flag_bias ? bias[i] : 0.f;

            float *dout_channel = dout_batch + i * size_out_channel;

            const float *dr0 = din_channel;
            const float *dr1 = dr0 + w_in;
            const float *dr2 = dr1 + w_in;
            const float *dr3 = dr2 + w_in;

            const float *din0_ptr = nullptr;
            const float *din1_ptr = nullptr;
            const float *din2_ptr = nullptr;
            const float *din3_ptr = nullptr;

            float *doutr0 = nullptr;
            float *doutr1 = nullptr;

            float* ptr_zero = const_cast<float*>(zero);

            for (int i = 0; i < h_out; i += 2){
                //! process top pad pad_h = 1
                din0_ptr = dr0;
                din1_ptr = dr1;
                din2_ptr = dr2;
                din3_ptr = dr3;

                doutr0 = dout_channel;
                doutr1 = dout_channel + w_out;

                dr0 = dr2;
                dr1 = dr3;
                dr2 = dr1 + w_in;
                dr3 = dr2 + w_in;
                //! process bottom pad
                if (i + 3 >= h_in) {
                    switch (i + 3 - h_in) {
                        case 3:
                            din1_ptr = zero_ptr;
                        case 2:
                            din2_ptr = zero_ptr;
                        case 1:
                            din3_ptr = zero_ptr;
                        case 0:
                            din3_ptr = zero_ptr;
                        default:
                            break;
                    }
                }
                //! process bottom remain
                if (i + 2 > h_out) {
                    doutr1 = write_ptr;
                }
                int cnt = tile_w;
                unsigned int* rmask_ptr = rmask;
                unsigned int* vmask_ptr = vmask;
                asm volatile(
                    "pld [%[din0_ptr]]                             @ preload data\n"
                    "pld [%[din1_ptr]]                      @ preload data\n"
                    "pld [%[din2_ptr]]                      @ preload data\n"
                    "pld [%[din3_ptr]]                      @ preload data\n"

                    "vld1.32  {d16-d17}, [%[din0_ptr]]!    @ load din r0\n"
                    "vld1.32  {d20-d21}, [%[din1_ptr]]!    @ load din r1\n"
                    "vld1.32  {d24-d25}, [%[din2_ptr]]!    @ load din r2\n"
                    "vld1.32  {d28-d29}, [%[din3_ptr]]!    @ load din r3\n"
                    "vld1.32  {d18}, [%[din0_ptr]]    @ load din r0\n"
                    "vld1.32  {d22}, [%[din1_ptr]]    @ load din r0\n"
                    "vld1.32  {d26}, [%[din2_ptr]]    @ load din r0\n"
                    "vld1.32  {d30}, [%[din3_ptr]]    @ load din r0\n"

                    "vdup.32 q4, %[bias_val]                            @ and \n" //q4 = vbias
                    "vdup.32 q5, %[bias_val]                            @ and \n" //q5 = vbias

                    "vext.32  q6, q8, q9, #1     @ 1234\n"
                    "vext.32  q7, q8, q9, #2     @ 2345\n"

                //mid
                     "1:                                    @ right pad entry\n"
                    //r0
                    "vmla.f32 q4, q8, %e[wr0][0]  @ q4 += 0123 * wr0[0]\n"

                    "pld [%[din0_ptr]]                             @ preload data\n"
                    "pld [%[din1_ptr]]                             @ preload data\n"
                    "pld [%[din2_ptr]]                             @ preload data\n"
                    "pld [%[din3_ptr]]                             @ preload data\n"

                    "vmla.f32 q4, q6, %e[wr0][1]  @ q4 += 1234 * wr0[1]\n"

                    "vld1.32  {d16-d17}, [%[din0_ptr]]!    @ load din r0\n"

                    "vmla.f32 q4, q7, %f[wr0][0]  @ q4 += 2345 * wr0[2]\n"

                    "vld1.32  {d18}, [%[din0_ptr]]    @ load din r0\n"

                    "vext.32  q6, q10, q11, #1     @ 1234\n"
                    "vext.32  q7, q10, q11, #2     @ 2345\n"

                    //r1
                    "vmla.f32 q5, q10, %e[wr0][0]  @ q4 += 1234 * wr0[0]\n"
                    "vmla.f32 q4, q10, %e[wr1][0]  @ q4 += 1234 * wr0[1]\n"

                    "vld1.32  {d20-d21}, [%[din1_ptr]]!    @ load din r0\n"

                    "vmla.f32 q5, q6, %e[wr0][1]  @ q4 += 1234 * wr0[1]\n"
                    "vmla.f32 q4, q6, %e[wr1][1]  @ q4 += 1234 * wr0[1]\n"

                    "vld1.32  {d22}, [%[din1_ptr]]    @ load din r0\n"

                    "vmla.f32 q5, q7, %f[wr0][0]  @ q4 += 1234 * wr0[1]\n"
                    "vmla.f32 q4, q7, %f[wr1][0]  @ q4 += 1234 * wr0[1]\n"

                    "vext.32  q6, q12, q13, #1     @ 1234\n"
                    "vext.32  q7, q12, q13, #2     @ 2345\n"

                    //r2
                    "vmla.f32 q5, q12, %e[wr1][0]  @ q4 += 1234 * wr0[0]\n"
                    "vmla.f32 q4, q12, %e[wr2][0]  @ q4 += 1234 * wr0[1]\n"

                    "vld1.32  {d24-d25}, [%[din2_ptr]]!    @ load din r0\n"

                    "vmla.f32 q5, q6, %e[wr1][1]  @ q4 += 1234 * wr0[1]\n"
                    "vmla.f32 q4, q6, %e[wr2][1]  @ q4 += 1234 * wr0[1]\n"

                    "vld1.32  {d26}, [%[din2_ptr]]    @ load din r0\n"

                    "vmla.f32 q5, q7, %f[wr1][0]  @ q4 += 1234 * wr0[1]\n"
                    "vmla.f32 q4, q7, %f[wr2][0]  @ q4 += 1234 * wr0[1]\n"

                    "vext.32  q6, q14, q15, #1     @ 1234\n"
                    "vext.32  q7, q14, q15, #2     @ 2345\n"

                    //r3
                    "vmla.f32 q5, q14, %e[wr2][0]  @ q4 += 0123 * wr0[0]\n"

                    "vld1.32  {d28-d29}, [%[din3_ptr]]!    @ load din r0\n"
                    "vmax.f32 q4, q4, %q[vzero]          @ relu \n"

                    "vmla.f32 q5, q6, %e[wr2][1]  @ q4 += 1234 * wr0[1]\n"

                    "vld1.32  {d30}, [%[din3_ptr]]    @ load din r0\n"
                    "vst1.32  {d8-d9},   [%[dout_ptr1]]!  @ store result, add pointer\n"

                    "vmla.f32 q5, q7, %f[wr2][0]  @ q4 += 2345 * wr0[2]\n"

                    "vext.32  q6, q8, q9, #1     @ 1234\n"
                    "vext.32  q7, q8, q9, #2     @ 2345\n"
                    "vmax.f32 q5, q5, %q[vzero]          @ relu \n"

                    "vdup.32 q4, %[bias_val]                            @ and \n" //q4 = vbias

                    "vst1.32  {d10-d11},   [%[dout_ptr2]]!  @ store result, add pointer\n"

                    "subs %[cnt], #1 @ loop count minus 1\n"

                    "vdup.32 q5, %[bias_val]                            @ and \n" //q4 = vbias

                    "bne    1b                             @ jump to main loop start point\n"

                //right
                    "3:                                    @ right pad entry\n"
                    "cmp %[remain], #1                             @ check whether has mid cols\n"
                    "blt  0f                                @ jump to main loop start point\n"
                    "vld1.32  {d19}, [%[vmask]]!    @ load din r0\n"
                    "vld1.32  {d23}, [%[vmask]]!    @ load din r0\n"

                    "vld1.32  {d27}, [%[vmask]]!    @ load din r0\n"
                    "vld1.32  {d31}, [%[vmask]]!    @ load din r0\n"

                    "vbif d16, %e[vzero], d19              @ bit select, deal with right pad\n"
                    "vbif d17, %e[vzero], d23              @ bit select, deal with right pad\n"
                    "vbif d18, %e[vzero], d27             @ bit select, deal with right pad\n"

                    "vbif d20, %e[vzero], d19              @ bit select, deal with right pad\n"
                    "vbif d21, %e[vzero], d23              @ bit select, deal with right pad\n"
                    "vbif d22, %e[vzero], d27             @ bit select, deal with right pad\n"

                    "vext.32  q6, q8, q9, #1     @ 1234\n"
                    "vext.32  q7, q8, q9, #2     @ 2345\n"

                    //r0
                    "vmla.f32 q4, q8, %e[wr0][0]  @ q4 += 0123 * wr0[0]\n"

                    "vbif d24, %e[vzero], d19              @ bit select, deal with right pad\n"
                    "vbif d25, %e[vzero], d23              @ bit select, deal with right pad\n"
                    "vbif d26, %e[vzero], d27             @ bit select, deal with right pad\n"

                    "vmla.f32 q4, q6, %e[wr0][1]  @ q4 += 1234 * wr0[1]\n"

                    "vbif d28, %e[vzero], d19              @ bit select, deal with right pad\n"
                    "vbif d29, %e[vzero], d23              @ bit select, deal with right pad\n"
                    "vbif d30, %e[vzero], d27             @ bit select, deal with right pad\n"

                    "vmla.f32 q4, q7, %f[wr0][0]  @ q4 += 2345 * wr0[2]\n"

                    "vext.32  q6, q10, q11, #1     @ 1234\n"
                    "vext.32  q7, q10, q11, #2     @ 2345\n"

                    //r1
                    "vmla.f32 q5, q10, %e[wr0][0]  @ q4 += 1234 * wr0[0]\n"
                    "vmla.f32 q4, q10, %e[wr1][0]  @ q4 += 1234 * wr0[1]\n"

                    "vld1.32  {d19}, [%[rmask]]!    @ load din r0\n"
                    "vld1.32  {d23}, [%[rmask]]!    @ load din r0\n"

                    "vmla.f32 q5, q6, %e[wr0][1]  @ q4 += 1234 * wr0[1]\n"
                    "vmla.f32 q4, q6, %e[wr1][1]  @ q4 += 1234 * wr0[1]\n"

                    "vld1.32  {d16-d17}, [%[dout_ptr1]]    @ load din r0\n"
                    "vld1.32  {d20-d21}, [%[dout_ptr2]]    @ load din r0\n"

                    "vmla.f32 q5, q7, %f[wr0][0]  @ q4 += 1234 * wr0[1]\n"
                    "vmla.f32 q4, q7, %f[wr1][0]  @ q4 += 1234 * wr0[1]\n"

                    "vext.32  q6, q12, q13, #1     @ 1234\n"
                    "vext.32  q7, q12, q13, #2     @ 2345\n"

                    //r2
                    "vmla.f32 q5, q12, %e[wr1][0]  @ q4 += 1234 * wr0[0]\n"
                    "vmla.f32 q4, q12, %e[wr2][0]  @ q4 += 1234 * wr0[1]\n"

                    "vmla.f32 q5, q6, %e[wr1][1]  @ q4 += 1234 * wr0[1]\n"
                    "vmla.f32 q4, q6, %e[wr2][1]  @ q4 += 1234 * wr0[1]\n"

                    "vmla.f32 q5, q7, %f[wr1][0]  @ q4 += 1234 * wr0[1]\n"
                    "vmla.f32 q4, q7, %f[wr2][0]  @ q4 += 1234 * wr0[1]\n"

                    "vext.32  q6, q14, q15, #1     @ 1234\n"
                    "vext.32  q7, q14, q15, #2     @ 2345\n"

                    //r3
                    "vmla.f32 q5, q14, %e[wr2][0]  @ q4 += 0123 * wr0[0]\n"

                    "vmax.f32 q4, q4, %q[vzero]          @ relu \n"

                    "vmla.f32 q5, q6, %e[wr2][1]  @ q4 += 1234 * wr0[1]\n"

                    "vbif d8, d16, d19              @ bit select, deal with right pad\n"
                    "vbif d9, d17, d23              @ bit select, deal with right pad\n"

                    "vmla.f32 q5, q7, %f[wr2][0]  @ q4 += 2345 * wr0[2]\n"

                    "vst1.32  {d8-d9},   [%[dout_ptr1]]!  @ store result, add pointer\n"

                    "vmax.f32 q5, q5, %q[vzero]          @ relu \n"

                    "vbif d10, d20, d19              @ bit select, deal with right pad\n"
                    "vbif d11, d21, d23              @ bit select, deal with right pad\n"

                    "vst1.32  {d10-d11},   [%[dout_ptr2]]!  @ store result, add pointer\n"
                    "0:                         \n"

                :[dout_ptr1] "+r"(doutr0), [dout_ptr2] "+r"(doutr1), \
                    [din0_ptr] "+r"(din0_ptr), [din1_ptr] "+r"(din1_ptr), \
                    [din2_ptr] "+r"(din2_ptr), [din3_ptr] "+r"(din3_ptr), \
                    [cnt] "+r"(cnt), [rmask] "+r" (rmask_ptr), [vmask] "+r" (vmask_ptr)
                :[wr0] "w"(wr0), [wr1] "w"(wr1), [wr2] "w"(wr2), [bias_val] "r" (bias_val), \
                [vzero] "w" (vzero), [remain] "r" (remain)
                :"cc", "memory", "q4", "q5", "q6", "q7", "q8", "q9", \
                        "q10", "q11", "q12", "q13", "q14", "q15"
                );
                dout_channel += 2 * w_out;
            } //! end of processing mid rows
        }
#endif

    }

}
/**
 * \brief depthwise convolution kernel 3x3, stride 2, with reulu
 */
//w_in > 7
void conv_depthwise_3x3s2p0_bias_relu(float* dout, const float* din, \
    const float* weights, const float* bias, bool flag_bias, \
    const int num, const int ch_in, const int h_in, const int w_in, \
    const int h_out, const int w_out, ARMContext* ctx) {

    int right_pad_idx[8] = {0, 2, 4, 6, 1, 3, 5, 7};
    int out_pad_idx[4] = {0, 1, 2, 3};

    int tile_w = w_out >> 2;
    int cnt_remain = w_out % 4;

    unsigned int size_right_remain = (unsigned int)(w_in - (tile_w << 3));

    uint32x4_t vmask_rp1 = vcgtq_s32(vdupq_n_s32(size_right_remain), vld1q_s32(right_pad_idx));//0 2 4 6
    uint32x4_t vmask_rp2 = vcgtq_s32(vdupq_n_s32(size_right_remain), vld1q_s32(right_pad_idx + 4));//1 3 5 7
    uint32x4_t wmask = vcgtq_s32(vdupq_n_s32(cnt_remain), vld1q_s32(out_pad_idx));//0 1 2 3
    int size_in_channel = w_in * h_in;
    int size_out_channel = w_out * h_out;

    float* zero_ptr = ctx->workspace_data<float>();
    memset(zero_ptr, 0, w_in * sizeof(float));
    float* write_ptr = zero_ptr + w_in;

    unsigned int dmask[12];

    vst1q_u32(dmask, vmask_rp1);
    vst1q_u32(dmask + 4, vmask_rp2);
    vst1q_u32(dmask + 8, wmask);

    for (int n = 0; n < num; ++n) {
        const float *din_batch = din + n * ch_in * size_in_channel;
        float *dout_batch = dout + n * ch_in * size_out_channel;
#pragma omp parallel for
        for (int i = 0; i < ch_in; ++i) {
            const float* din_channel = din_batch + i * size_in_channel;
            float* dout_channel = dout_batch + i * size_out_channel;

            const float *weight_ptr = weights + i * 9;
            float32x4_t wr0 = vld1q_f32(weight_ptr);
            float32x4_t wr1 = vld1q_f32(weight_ptr + 3);
            float32x4_t wr2 = vld1q_f32(weight_ptr + 6);

            float32x4_t vzero = vdupq_n_f32(0.f);

            float32x4_t wbias;
            float bias_c = 0.f;
            if (flag_bias) {
                wbias  = vdupq_n_f32(bias[i]);
                bias_c = bias[i];
            } else {
                wbias = vdupq_n_f32(0.f);
            }

            const float *dr0 = din_channel;
            const float *dr1 = dr0 + w_in;
            const float *dr2 = dr1 + w_in;
            const float *dr3 = dr2 + w_in;
            const float *dr4 = dr3 + w_in;

            const float *din0_ptr = dr0;
            const float *din1_ptr = dr1;
            const float *din2_ptr = dr2;
            const float *din3_ptr = dr3;
            const float *din4_ptr = dr4;

            float *doutr0 = dout_channel;
            float *doutr0_ptr = nullptr;
            float *doutr1_ptr = nullptr;

#ifdef __aarch64__
            for (int i = 0; i < h_out; i += 2){
                din0_ptr = dr0;
                din1_ptr = dr1;
                din2_ptr = dr2;
                din3_ptr = dr3;
                din4_ptr = dr4;

                doutr0_ptr = doutr0;
                doutr1_ptr = doutr0 + w_out;

                dr0 = dr4;
                dr1 = dr0 + w_in;
                dr2 = dr1 + w_in;
                dr3 = dr2 + w_in;
                dr4 = dr3 + w_in;

                //! process bottom pad
                if (i + 4 >= h_in) {
                    switch (i + 4 - h_in) {
                        case 4:
                            din1_ptr = zero_ptr;
                        case 3:
                            din2_ptr = zero_ptr;
                        case 2:
                            din3_ptr = zero_ptr;
                        case 1:
                            din4_ptr = zero_ptr;
                        case 0:
                            din4_ptr = zero_ptr;
                        default:
                            break;
                    }
                }
                //! process output pad
                if (i + 2 > h_out) {
                    doutr1_ptr = write_ptr;
                }
                int cnt = tile_w;
                asm volatile (
                //top
                // Load up 12 elements (3 vectors) from each of 8 sources.
                "0:                                      \n"
                    "prfm pldl1keep, [%[inptr0]]             \n"
                    "prfm pldl1keep, [%[inptr1]]             \n"
                    "prfm pldl1keep, [%[inptr2]]             \n"
                    "prfm pldl1keep, [%[inptr3]]             \n"
                    "prfm pldl1keep, [%[inptr4]]             \n"
                    "ld2  {v0.4s, v1.4s}, [%[inptr0]], #32    \n" //v0={0,2,4,6} v1={1,3,5,7}
                    "ld2  {v2.4s, v3.4s}, [%[inptr1]], #32    \n"
                    "ld2  {v4.4s, v5.4s}, [%[inptr2]], #32    \n"
                    "ld2  {v6.4s, v7.4s}, [%[inptr3]], #32    \n"
                    "ld2  {v8.4s, v9.4s}, [%[inptr4]], #32    \n"

                    "and  v16.16b, %[vbias].16b, %[vbias].16b  \n" //v10 = vbias
                    "and  v17.16b, %[vbias].16b, %[vbias].16b  \n" //v16 = vbias

                    "ld1 {v15.4s}, [%[inptr0]]                 \n"
                    "ld1 {v18.4s}, [%[inptr1]]                 \n"
                    "ld1 {v19.4s}, [%[inptr2]]                 \n"
                    "ld1 {v20.4s}, [%[inptr3]]                 \n"
                    "ld1 {v21.4s}, [%[inptr4]]                 \n"

                    "ext  v10.16b, v0.16b, v15.16b, #4     \n" // v10 = {2,4,6,8}
                //mid
                    "2:                                          \n"
                    //r0
                    "fmul v11.4s, v0.4s, %[w0].s[0]            \n" // {0,2,4,6} * w00
                    "fmul v12.4s, v1.4s, %[w0].s[1]            \n" // {1,3,5,7} * w01
                    "fmla v16.4s, v10.4s, %[w0].s[2]            \n" // {2,4,6,8} * w02

                    "ext  v10.16b, v2.16b, v18.16b, #4     \n" // v10 = {2,4,6,8}
                    "ld2  {v0.4s, v1.4s}, [%[inptr0]], #32    \n" //v0={0,2,4,6} v1={1,3,5,7}

                    //r1
                    "fmla v11.4s, v2.4s, %[w1].s[0]            \n" // {0,2,4,6} * w00
                    "fmla v12.4s, v3.4s, %[w1].s[1]            \n" // {1,3,5,7} * w01
                    "fmla v16.4s, v10.4s, %[w1].s[2]            \n" // {2,4,6,8} * w02

                    "ext  v10.16b, v4.16b, v19.16b, #4     \n" // v10 = {2,4,6,8}

                    "ld2  {v2.4s, v3.4s}, [%[inptr1]], #32    \n"

                    //r2
                    "fmul v13.4s, v4.4s, %[w0].s[0]            \n" // {0,2,4,6} * w00
                    "fmla v11.4s, v4.4s, %[w2].s[0]            \n" // {0,2,4,6} * w00

                    "fmul v14.4s, v5.4s, %[w0].s[1]            \n" // {1,3,5,7} * w01
                    "fmla v12.4s, v5.4s, %[w2].s[1]            \n" // {1,3,5,7} * w01

                    "fmla v17.4s, v10.4s, %[w0].s[2]            \n" // {2,4,6,8} * w02
                    "fmla v16.4s, v10.4s, %[w2].s[2]            \n" // {2,4,6,8} * w02

                    "ext  v10.16b, v6.16b, v20.16b, #4     \n" // v10 = {2,4,6,8}

                    "ld2  {v4.4s, v5.4s}, [%[inptr2]], #32    \n"

                    //r3
                    "fmla v13.4s, v6.4s, %[w1].s[0]            \n" // {0,2,4,6} * w00
                    "fmla v14.4s, v7.4s, %[w1].s[1]            \n" // {1,3,5,7} * w01
                    "fmla v17.4s, v10.4s, %[w1].s[2]            \n" // {2,4,6,8} * w02

                    "ext  v10.16b, v8.16b, v21.16b, #4     \n" // v10 = {2,4,6,8}

                    "ld2  {v6.4s, v7.4s}, [%[inptr3]], #32    \n"

                    "fadd v16.4s, v16.4s, v11.4s                  \n"
                    "fadd v16.4s, v16.4s, v12.4s                  \n"

                    //r4
                    "fmla v13.4s, v8.4s, %[w2].s[0]            \n" // {0,2,4,6} * w00
                    "fmla v14.4s, v9.4s, %[w2].s[1]            \n" // {1,3,5,7} * w01
                    "fmla v17.4s, v10.4s, %[w2].s[2]            \n" // {2,4,6,8} * w02

                    "ld2  {v8.4s, v9.4s}, [%[inptr4]], #32    \n"
                    "ld1 {v15.4s}, [%[inptr0]]                 \n"
                    "ld1 {v18.4s}, [%[inptr1]]                 \n"
                    "fmax v16.4s, v16.4s, %[vzero].4s            \n" /* relu */

                    "fadd v17.4s, v17.4s, v13.4s                  \n"

                    "ld1 {v19.4s}, [%[inptr2]]                 \n"
                    "ld1 {v20.4s}, [%[inptr3]]                 \n"
                    "ld1 {v21.4s}, [%[inptr4]]                 \n"
                    "st1 {v16.4s}, [%[outptr0]], #16              \n"

                    "fadd v17.4s, v17.4s, v14.4s                  \n"

                    "ext  v10.16b, v0.16b, v15.16b, #4     \n" // v10 = {2,4,6,8}
                    "and  v16.16b, %[vbias].16b, %[vbias].16b  \n" //v10 = vbias
                    "fmax v17.4s, v17.4s, %[vzero].4s            \n" /* relu */

                    "subs %[cnt], %[cnt], #1                    \n"

                    "st1 {v17.4s}, [%[outptr1]], #16              \n"

                    "and  v17.16b, %[vbias].16b, %[vbias].16b  \n" //v16 = vbias

                    "bne  2b                                    \n"

                //right
                    "1:                                          \n"
                    "cmp %[remain], #1                           \n"
                    "blt 4f                                     \n"
                    "3:                                         \n"
                    "bif  v0.16b, %[vzero].16b, %[mask1].16b    \n" //pipei
                    "bif  v1.16b, %[vzero].16b, %[mask2].16b    \n" //pipei

                    "bif  v2.16b, %[vzero].16b, %[mask1].16b    \n" //pipei
                    "bif  v3.16b, %[vzero].16b, %[mask2].16b    \n" //pipei

                    "bif  v4.16b, %[vzero].16b, %[mask1].16b    \n" //pipei
                    "bif  v5.16b, %[vzero].16b, %[mask2].16b    \n" //pipei

                    "ext  v10.16b, v0.16b, %[vzero].16b, #4     \n" // v10 = {2,4,6,8}

                    "bif  v6.16b, %[vzero].16b, %[mask1].16b    \n" //pipei
                    "bif  v7.16b, %[vzero].16b, %[mask2].16b    \n" //pipei

                    //r0
                    "fmul v11.4s, v0.4s, %[w0].s[0]            \n" // {0,2,4,6} * w00
                    "fmul v12.4s, v1.4s, %[w0].s[1]            \n" // {1,3,5,7} * w01
                    "fmla v16.4s, v10.4s, %[w0].s[2]            \n" // {2,4,6,8} * w02

                    "ext  v10.16b, v2.16b, %[vzero].16b, #4     \n" // v10 = {2,4,6,8}
                    "bif  v8.16b, %[vzero].16b, %[mask1].16b    \n" //pipei
                    "bif  v9.16b, %[vzero].16b, %[mask2].16b    \n" //pipei

                    //r1
                    "fmla v11.4s, v2.4s, %[w1].s[0]            \n" // {0,2,4,6} * w00
                    "fmla v12.4s, v3.4s, %[w1].s[1]            \n" // {1,3,5,7} * w01
                    "fmla v16.4s, v10.4s, %[w1].s[2]            \n" // {2,4,6,8} * w02

                    "ext  v10.16b, v4.16b, %[vzero].16b, #4     \n" // v10 = {2,4,6,8}

                    //r2
                    "fmul v13.4s, v4.4s, %[w0].s[0]            \n" // {0,2,4,6} * w00
                    "fmla v11.4s, v4.4s, %[w2].s[0]            \n" // {0,2,4,6} * w00

                    "fmul v14.4s, v5.4s, %[w0].s[1]            \n" // {1,3,5,7} * w01
                    "fmla v12.4s, v5.4s, %[w2].s[1]            \n" // {1,3,5,7} * w01

                    "fmla v17.4s, v10.4s, %[w0].s[2]            \n" // {2,4,6,8} * w02
                    "fmla v16.4s, v10.4s, %[w2].s[2]            \n" // {2,4,6,8} * w02

                    "ext  v10.16b, v6.16b, %[vzero].16b, #4     \n" // v10 = {2,4,6,8}

                    //r3
                    "fmla v13.4s, v6.4s, %[w1].s[0]            \n" // {0,2,4,6} * w00
                    "fmla v14.4s, v7.4s, %[w1].s[1]            \n" // {1,3,5,7} * w01
                    "fmla v17.4s, v10.4s, %[w1].s[2]            \n" // {2,4,6,8} * w02

                    "ext  v10.16b, v8.16b, %[vzero].16b, #4     \n" // v10 = {2,4,6,8}
                    "ld1 {v0.4s}, [%[outptr0]]                  \n"

                    "fadd v16.4s, v16.4s, v11.4s                  \n"
                    "fadd v16.4s, v16.4s, v12.4s                  \n"
                    "ld1 {v1.4s}, [%[outptr1]]                  \n"

                    //r4
                    "fmla v13.4s, v8.4s, %[w2].s[0]            \n" // {0,2,4,6} * w00
                    "fmla v14.4s, v9.4s, %[w2].s[1]            \n" // {1,3,5,7} * w01
                    "fmla v17.4s, v10.4s, %[w2].s[2]            \n" // {2,4,6,8} * w02

                    "fmax v16.4s, v16.4s, %[vzero].4s            \n" /* relu */

                    "fadd v17.4s, v17.4s, v13.4s                  \n"

                    "bif  v16.16b, v0.16b, %[wmask].16b    \n" //pipei

                    "fadd v17.4s, v17.4s, v14.4s                  \n"

                    "st1 {v16.4s}, [%[outptr0]], #16              \n"

                    "fmax v17.4s, v17.4s, %[vzero].4s            \n" /* relu */

                    "bif  v17.16b, v1.16b, %[wmask].16b    \n" //pipei

                    "st1 {v17.4s}, [%[outptr1]], #16              \n"
                "4:                                          \n"
                : [inptr0] "+r"(din0_ptr), [inptr1] "+r"(din1_ptr), [inptr2] "+r"(din2_ptr), [inptr3] "+r"(din3_ptr), \
                  [inptr4] "+r"(din4_ptr), [outptr0] "+r"(doutr0_ptr), [outptr1] "+r"(doutr1_ptr), \
                  [cnt] "+r" (cnt)
                : [vzero] "w" (vzero), [w0] "w" (wr0), [w1] "w" (wr1), [w2] "w" (wr2), [remain] "r" (cnt_remain), \
                  [mask1] "w" (vmask_rp1), [mask2] "w" (vmask_rp2), [wmask] "w" (wmask), [vbias] "w" (wbias)
                : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", \
                  "v15", "v16", "v17", "v18", "v19", "v20", "v21"
                );
                doutr0 = doutr0 + 2 * w_out;
            }
#else
            for (int i = 0; i < h_out; i++){
                din0_ptr = dr0;
                din1_ptr = dr1;
                din2_ptr = dr2;

                doutr0_ptr = doutr0;

                dr0 = dr2;
                dr1 = dr0 + w_in;
                dr2 = dr1 + w_in;

                //! process bottom pad
                if (i + 2 > h_in) {
                    switch (i + 2 - h_in) {
                        case 2:
                            din1_ptr = zero_ptr;
                        case 1:
                            din2_ptr = zero_ptr;
                        default:
                            break;
                    }
                }
                int cnt = tile_w;
                unsigned int* mask_ptr = dmask;
                asm volatile (
                // Load up 12 elements (3 vectors) from each of 8 sources.
                    "0:                                                     \n"
                        "vmov.u32 q9, #0                                \n"
                        "vld2.32  {d20-d23}, [%[din0_ptr]]!             @ load din r1\n"
                        "vld2.32  {d24-d27}, [%[din1_ptr]]!             @ load din r1\n"
                        "vld2.32  {d28-d31}, [%[din2_ptr]]!             @ load din r1\n"
                        "pld [%[din0_ptr]]                              @ preload data\n"
                        "pld [%[din1_ptr]]                              @ preload data\n"
                        "pld [%[din2_ptr]]                              @ preload data\n"

                        "vld1.32  {d16}, [%[din0_ptr]]                  @ load din r0\n"//q2={8,10,12,14}

                        "vdup.32 q3, %[bias]                            @ and \n" //q10 = vbias
                    //mid
                        "2:                                             \n"
                        "vext.32  q6, q10, q8, #1                       @ shift left 1 \n" //q6 = {2,4,6,8}
                        "vld1.32 {d16}, [%[din1_ptr]]                   @ load din r1\n"//q2={8,10,12,14}

                        "vmul.f32 q4, q10, %e[wr0][0]                   @ mul weight 0, out0\n"// q0 * w00
                        "vmul.f32 q5, q11, %e[wr0][1]                   @ mul weight 0, out0\n"// q1 * w01
                        "vmla.f32 q3,  q6, %f[wr0][0]                   @ mul weight 0, out0\n"// q6 * w02

                        "vext.32  q7, q12, q8, #1                       @ shift left 1 \n" //q6 = {2,4,6,8}
                        "vld1.32 {d16}, [%[din2_ptr]]                   @ load din r1\n"//q2={8,10,12,14}

                        "vld2.32  {d20-d23}, [%[din0_ptr]]!             @ load din r0\n"//v0={0,2,4,6} v1={1,3,5,7}

                        "vmla.f32 q4, q12, %e[wr1][0]                   @ mul weight 1, out0\n"// q0 * w00
                        "vmla.f32 q5, q13, %e[wr1][1]                   @ mul weight 1, out0\n"// q1 * w01
                        "vmla.f32 q3,  q7, %f[wr1][0]                   @ mul weight 1, out0\n"// q6 * w02

                        "vext.32  q6, q14, q8, #1                       @ shift left 1 \n" //q6 = {2,4,6,8}

                        "vld2.32  {d24-d27}, [%[din1_ptr]]!             @ load din r1\n"//v0={0,2,4,6} v1={1,3,5,7}

                        "vmla.f32 q4, q14, %e[wr2][0]                   @ mul weight 2, out0\n"// q0 * w00
                        "vmla.f32 q5, q15, %e[wr2][1]                   @ mul weight 2, out0\n"// q1 * w01
                        "vmla.f32 q3,  q6, %f[wr2][0]                   @ mul weight 2, out0\n"// q6 * w02

                        "vld2.32  {d28-d31}, [%[din2_ptr]]!             @ load din r2\n"//v4={0,2,4,6} v5={1,3,5,7}

                        "vadd.f32 q3, q3, q4                            @ add \n"
                        "vadd.f32 q3, q3, q5                            @ add \n"

                        "subs %[cnt], #1                                \n"
                        "vmax.f32 q3, q3, q9                     @ relu \n"

                        "vld1.32  {d16}, [%[din0_ptr]]                  @ load din r0\n"//q2={8,10,12,14}

                        "vst1.32 {d6-d7}, [%[outptr]]!                  \n"

                        "vdup.32  q3, %[bias]                           @ and \n" //q10 = vbias
                        "bne  2b                                        \n"

                        //right
                        "1:                                             \n"
                        "cmp %[remain], #1                              \n"
                        "blt 3f                                         \n"

                        "vld1.f32   {d12-d15}, [%[mask_ptr]]!           @ load mask\n"

                        "vbif q10, q9, q6                               @ bit select, deal with right pad\n"
                        "vbif q11, q9, q7                               @ bit select, deal with right pad\n"
                        "vbif q12, q9, q6                               @ bit select, deal with right pad\n"
                        "vbif q13, q9, q7                               @ bit select, deal with right pad\n"
                        "vbif q14, q9, q6                               @ bit select, deal with right pad\n"
                        "vbif q15, q9, q7                               @ bit select, deal with right pad\n"

                        "vext.32 q6, q10, q9, #1                        @ shift left 1 \n" //q6 = {2,4,6,8}
                        "vext.32 q7, q12, q9, #1                        @ shift left 1 \n" //q6 = {2,4,6,8}

                        "vmul.f32 q4, q10, %e[wr0][0]                   @ mul weight 0, out0\n"// q0 * w00
                        "vmul.f32 q5, q11, %e[wr0][1]                   @ mul weight 0, out0\n"// q1 * w01
                        "vmla.f32 q3,  q6, %f[wr0][0]                   @ mul weight 0, out0\n"// q6 * w02

                        "vext.32 q6, q14, q9, #1                        @ shift left 1 \n" //q6 = {2,4,6,8}
                        "vld1.f32   {d20-d21}, [%[outptr]]              @ load output\n"

                        "vmla.f32 q4, q12, %e[wr1][0]                   @ mul weight 1, out0\n"// q0 * w00
                        "vmla.f32 q5, q13, %e[wr1][1]                   @ mul weight 1, out0\n"// q1 * w01
                        "vmla.f32 q3,  q7, %f[wr1][0]                   @ mul weight 1, out0\n"// q6 * w02

                        "vld1.f32   {d22-d23}, [%[mask_ptr]]            @ load mask\n"

                        "vmla.f32 q4, q14, %e[wr2][0]                   @ mul weight 2, out0\n"// q0 * w00
                        "vmla.f32 q5, q15, %e[wr2][1]                   @ mul weight 2, out0\n"// q1 * w01
                        "vmla.f32 q3,  q6, %f[wr2][0]                   @ mul weight 2, out0\n"// q6 * w02

                        "vadd.f32 q3, q3, q4                            @ add \n"
                        "vadd.f32 q3, q3, q5                            @ add \n"

                         "vmax.f32 q3, q3, q9                     @ relu \n"

                        "vbif.f32 q3, q10, q11                          @ write mask\n"

                        "vst1.32 {d6-d7}, [%[outptr]]!                  \n"
                        "3:                                             \n"
                : [din0_ptr] "+r"(din0_ptr), [din1_ptr] "+r"(din1_ptr), [din2_ptr] "+r"(din2_ptr), \
                    [outptr] "+r"(doutr0_ptr), [cnt] "+r" (cnt), [mask_ptr] "+r"(mask_ptr)
                : [remain] "r" (cnt_remain), [wr0] "w" (wr0), [wr1] "w" (wr1), [wr2] "w" (wr2), \
                    [bias] "r"(bias_c)
                : "cc", "memory", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
                );

                doutr0 = doutr0 + w_out;

            }
#endif
        }
    }
}
/**
 * \brief depthwise convolution, kernel size 3x3, stride 1, pad 1, with bias, width <= 4
 */
void conv_depthwise_3x3s1p0_bias_s(float* dout, const float* din, \
    const float* weights, const float* bias, bool flag_bias, \
    const int num, const int ch_in, const int h_in, const int w_in, \
    const int h_out, const int w_out, ARMContext* ctx) {
    //! 3x3s1 convolution, implemented by direct algorithm
    //! pad is done implicit
    //! for 4x6 convolution window
    const int right_pad_idx[8] = {5, 4, 3, 2, 1, 0, 0, 0};
    const float zero_ptr[4] = {0.f, 0.f, 0.f, 0.f};

    float32x4_t vzero = vdupq_n_f32(0.f);
    uint32x4_t vmask_rp1 = vcgeq_s32(vld1q_s32(right_pad_idx), vdupq_n_s32(6 - w_in));
    uint32x4_t vmask_rp2 = vcgeq_s32(vld1q_s32(right_pad_idx + 4), vdupq_n_s32(6 - w_in));

    unsigned int vmask[8];
    vst1q_u32(vmask, vmask_rp1);
    vst1q_u32(vmask + 4, vmask_rp2);

    int size_in_channel = w_in * h_in;
    int size_out_channel = w_out * h_out;
    for (int n = 0; n < num; ++n) {
        const float *din_batch = din + n * ch_in * size_in_channel;
        float *dout_batch = dout + n * ch_in * size_out_channel;
#pragma omp parallel for
        for (int i = 0; i < ch_in; ++i) {
            float *dout_channel = dout_batch + i * size_out_channel;
            const float *din_channel = din_batch + i * size_in_channel;
            const float *weight_ptr = weights + i * 9;
            float32x4_t wr0 = vld1q_f32(weight_ptr);
            float32x4_t wr1 = vld1q_f32(weight_ptr + 3);
            float32x4_t wr2 = vld1q_f32(weight_ptr + 6);
            float32x4_t wbias;
            if (flag_bias) {
                wbias  = vdupq_n_f32(bias[i]);
            } else {
                wbias = vdupq_n_f32(0.f);
            }

            float out_buf1[4];
            float out_buf2[4];
            float trash_buf[4];

            float *doutr0 = dout_channel;
            float *doutr1 = dout_channel + w_out;

            for (int j = 0; j < h_out; j += 2){

                const float *dr0 = din_channel + j * w_in;
                const float *dr1 = dr0 + w_in;
                const float *dr2 = dr1 + w_in;
                const float *dr3 = dr2 + w_in;

                doutr0 = dout_channel + j * w_out;
                doutr1 = doutr0 + w_out;

                if (j + 3 >= h_in) {
                    switch (j + 3 - h_in) {
                        case 3:
                            dr1 = zero_ptr;
                        case 2:
                            dr2 = zero_ptr;
                        case 1:
                            dr3 = zero_ptr;
                            doutr1 = trash_buf;
                        case 0:
                            dr3 = zero_ptr;
                            doutr1 = trash_buf;
                        default:
                            break;
                    }
                }
#ifdef __aarch64__
                asm volatile(
                    "prfm pldl1keep, [%[din0]]\n"
                    "prfm pldl1keep, [%[din1]]\n"
                    "prfm pldl1keep, [%[din2]]\n"
                    "prfm pldl1keep, [%[din3]]\n"

                    "ld1 {v0.4s, v1.4s}, [%[din0]]\n"
                    "ld1 {v2.4s, v3.4s}, [%[din1]]\n"
                    "ld1 {v4.4s, v5.4s}, [%[din2]]\n"
                    "ld1 {v6.4s, v7.4s}, [%[din3]]\n"

                    "bif v0.16b, %[zero].16b, %[mask1].16b\n"  //d0_1234
                    "bif v1.16b, %[zero].16b, %[mask2].16b\n"  //d0_1234

                    "bif v2.16b, %[zero].16b, %[mask1].16b\n"  //d1_1234
                    "bif v3.16b, %[zero].16b, %[mask2].16b\n"  //d1_1234

                    "bif v4.16b, %[zero].16b, %[mask1].16b\n"  //d2_1234
                    "bif v5.16b, %[zero].16b, %[mask2].16b\n"  //d2_1234

                    "bif v6.16b, %[zero].16b, %[mask1].16b\n"  //d3_1234
                    "bif v7.16b, %[zero].16b, %[mask2].16b\n"  //d3_1234

                    "ext v8.16b, v0.16b, v1.16b, #4\n"  //d1_2345
                    "ext v9.16b, v0.16b, v1.16b, #8\n"  //d1_3450

                    "and  v12.16b, %[vbias].16b, %[vbias].16b  \n" //v12 = vbias
                    "and  v13.16b, %[vbias].16b, %[vbias].16b  \n" //v13 = vbias

                    //r0
                    "fmul v10.4s, v0.4s, %[wr0].s[0]\n"//d0_1234 * w0[0]
                    "fmul v11.4s, v8.4s, %[wr0].s[1]\n"//d1_2345 * w0[1]
                    "fmla v12.4s, v9.4s, %[wr0].s[2]\n"//d0_3456 * w0[2]

                    "ext v8.16b, v2.16b, v3.16b, #4\n"  //d1_2345
                    "ext v9.16b, v2.16b, v3.16b, #8\n"  //d1_3450

                    //r1
                    "fmul v14.4s, v2.4s, %[wr0].s[0]\n"//d0_1234 * w0[0]
                    "fmla v10.4s, v2.4s, %[wr1].s[0]\n"//d0_1234 * w0[0]

                    "fmul v15.4s, v8.4s, %[wr0].s[1]\n"//d1_2345 * w0[1]
                    "fmla v11.4s, v8.4s, %[wr1].s[1]\n"//d1_2345 * w0[1]

                    "fmla v13.4s, v9.4s, %[wr0].s[2]\n"//d0_3456 * w0[2]
                    "fmla v12.4s, v9.4s, %[wr1].s[2]\n"//d0_3456 * w0[2]

                    "ext v8.16b, v4.16b, v5.16b, #4\n"  //d1_2345
                    "ext v9.16b, v4.16b, v5.16b, #8\n"  //d1_3450

                    //r2
                    "fmla v14.4s, v4.4s, %[wr1].s[0]\n"//d0_1234 * w0[0]
                    "fmla v10.4s, v4.4s, %[wr2].s[0]\n"//d0_1234 * w0[0]

                    "fmla v15.4s, v8.4s, %[wr1].s[1]\n"//d1_2345 * w0[1]
                    "fmla v11.4s, v8.4s, %[wr2].s[1]\n"//d1_2345 * w0[1]

                    "fmla v13.4s, v9.4s, %[wr1].s[2]\n"//d0_3456 * w0[2]
                    "fmla v12.4s, v9.4s, %[wr2].s[2]\n"//d0_3456 * w0[2]

                    "ext v8.16b, v6.16b, v7.16b, #4\n"  //d1_2345
                    "ext v9.16b, v6.16b, v7.16b, #8\n"  //d1_3450

                    //r3
                    "fmla v14.4s, v6.4s, %[wr2].s[0]\n"//d0_1234 * w0[0]

                    "fmla v15.4s, v8.4s, %[wr2].s[1]\n"//d1_2345 * w0[1]

                    "fadd v12.4s, v12.4s, v10.4s\n"

                    "fmla v13.4s, v9.4s, %[wr2].s[2]\n"//d0_3456 * w0[2]

                    "fadd v12.4s, v12.4s, v11.4s\n" //out1
                    "fadd v13.4s, v13.4s, v14.4s\n" //out2
                    "fadd v13.4s, v13.4s, v15.4s\n" //out2

                    "prfm pldl1keep, [%[out1]]\n"
                    "prfm pldl1keep, [%[out2]]\n"

                    "st1 {v12.4s}, [%[out1]]\n"
                    "st1 {v13.4s}, [%[out2]]\n"
                : [din0] "+r"(dr0), [din1] "+r"(dr1), [din2] "+r"(dr2), [din3] "+r"(dr3)
                : [wr0]"w"(wr0), [wr1]"w"(wr1), [wr2]"w"(wr2), [vbias] "w" (wbias), \
                    [mask1] "w" (vmask_rp1), [mask2] "w" (vmask_rp2), [zero] "w" (vzero), \
                    [out1] "r" (out_buf1), [out2] "r" (out_buf2)
                :"cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", \
                      "v10", "v11", "v12", "v13", "v14", "v15"
                );
#else
                unsigned int* vmask_ptr = vmask;
                float bias_val = flag_bias ? bias[i] : 0.f;
                asm volatile(
                    "pld [%[din0]]\n"
                    "pld [%[din1]]\n"
                    "pld [%[din2]]\n"
                    "pld [%[din3]]\n"

                    "vld1.32  {d16-d18}, [%[din0]]    @ load din r0\n"
                    "vld1.32  {d20-d22}, [%[din1]]    @ load din r1\n"
                    "vld1.32  {d24-d26}, [%[din2]]    @ load din r2\n"
                    "vld1.32  {d28-d30}, [%[din3]]    @ load din r3\n"

                    "vdup.32 q4, %[bias_val]                            @ and \n" //q4 = vbias
                    "vdup.32 q5, %[bias_val]                            @ and \n" //q5 = vbias

                    "vld1.32  {d19}, [%[vmask]]!    @ load din r0\n"
                    "vld1.32  {d23}, [%[vmask]]!    @ load din r0\n"

                    "vld1.32  {d27}, [%[vmask]]!    @ load din r0\n"

                    "vbif d16, %e[vzero], d19              @ bit select, deal with right pad\n"
                    "vbif d20, %e[vzero], d19              @ bit select, deal with right pad\n"

                    "vbif d17, %e[vzero], d23              @ bit select, deal with right pad\n"
                    "vbif d21, %e[vzero], d23              @ bit select, deal with right pad\n"

                    "vbif d18, %e[vzero], d27             @ bit select, deal with right pad\n"
                    "vbif d22, %e[vzero], d27             @ bit select, deal with right pad\n"

                    "vext.32  q6, q8, q9, #1     @ 1234\n"
                    "vext.32  q7, q8, q9, #2     @ 2345\n"

                    //r0
                    "vmla.f32 q4, q8, %e[wr0][0]  @ q4 += 0123 * wr0[0]\n"

                    "vbif d24, %e[vzero], d19              @ bit select, deal with right pad\n"
                    "vbif d25, %e[vzero], d23              @ bit select, deal with right pad\n"
                    "vbif d26, %e[vzero], d27             @ bit select, deal with right pad\n"

                    "vmla.f32 q4, q6, %e[wr0][1]  @ q4 += 1234 * wr0[1]\n"

                    "vbif d28, %e[vzero], d19              @ bit select, deal with right pad\n"
                    "vbif d29, %e[vzero], d23              @ bit select, deal with right pad\n"
                    "vbif d30, %e[vzero], d27             @ bit select, deal with right pad\n"

                    "vmla.f32 q4, q7, %f[wr0][0]  @ q4 += 2345 * wr0[2]\n"

                    "vext.32  q6, q10, q11, #1     @ 1234\n"
                    "vext.32  q7, q10, q11, #2     @ 2345\n"

                    //r1
                    "vmla.f32 q5, q10, %e[wr0][0]  @ q4 += 1234 * wr0[0]\n"
                    "vmla.f32 q4, q10, %e[wr1][0]  @ q4 += 1234 * wr0[1]\n"

                    "vmul.f32 q8, q6, %e[wr0][1]  @ q4 += 1234 * wr0[1]\n"
                    "vmul.f32 q10, q6, %e[wr1][1]  @ q4 += 1234 * wr0[1]\n"

                    "vmul.f32 q9, q7, %f[wr0][0]  @ q4 += 1234 * wr0[1]\n"
                    "vmul.f32 q11, q7, %f[wr1][0]  @ q4 += 1234 * wr0[1]\n"

                    "vext.32  q6, q12, q13, #1     @ 1234\n"
                    "vext.32  q7, q12, q13, #2     @ 2345\n"

                    //r2
                    "vmla.f32 q5, q12, %e[wr1][0]  @ q4 += 1234 * wr0[0]\n"
                    "vmla.f32 q4, q12, %e[wr2][0]  @ q4 += 1234 * wr0[1]\n"

                    "vmla.f32 q8, q6, %e[wr1][1]  @ q4 += 1234 * wr0[1]\n"
                    "vmla.f32 q10, q6, %e[wr2][1]  @ q4 += 1234 * wr0[1]\n"

                    "vmla.f32 q9, q7, %f[wr1][0]  @ q4 += 1234 * wr0[1]\n"
                    "vmla.f32 q11, q7, %f[wr2][0]  @ q4 += 1234 * wr0[1]\n"

                    "vext.32  q6, q14, q15, #1     @ 1234\n"
                    "vext.32  q7, q14, q15, #2     @ 2345\n"

                    //r3
                    "vmla.f32 q5, q14, %e[wr2][0]  @ q4 += 0123 * wr0[0]\n"

                    "vmla.f32 q8, q6, %e[wr2][1]  @ q4 += 1234 * wr0[1]\n"
                    "vadd.f32 q4, q4, q10         @ q4 += q10 \n"

                    "pld [%[out1]]\n"
                    "pld [%[out2]]\n"

                    "vmla.f32 q9, q7, %f[wr2][0]  @ q4 += 2345 * wr0[2]\n"
                    "vadd.f32 q4, q4, q11         @ q4 += q10 \n"

                    "vadd.f32 q5, q5, q8         @ q4 += q10 \n"
                    "vadd.f32 q5, q5, q9         @ q4 += q10 \n"

                    "vst1.32  {d8-d9},   [%[out1]]  @ store result, add pointer\n"
                    "vst1.32  {d10-d11},   [%[out2]]  @ store result, add pointer\n"

                    :[din0] "+r" (dr0), [din1] "+r" (dr1), [din2] "+r" (dr2), [din3] "+r" (dr3), \
                      [vmask] "+r" (vmask_ptr)
                    :[wr0] "w" (wr0), [wr1] "w" (wr1), [wr2] "w" (wr2), [vzero] "w" (vzero), \
                       [bias_val] "r" (bias_val), [out1] "r" (out_buf1), [out2] "r" (out_buf2)
                    :"cc", "memory", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
                );
#endif //__aarch64__
                for (int w = 0; w < w_out; ++w){
                    *doutr0++ = out_buf1[w];
                    *doutr1++ = out_buf2[w];
                };
            }// end of processing heights
        } // end of processing channels
    } // end of processing batchs
}
/**
 * \brief depthwise convolution kernel 3x3, stride 2, width <= 4
 */

void conv_depthwise_3x3s2p0_bias_s(float* dout, const float* din, \
    const float* weights, const float* bias, bool flag_bias, \
    const int num, const int ch_in, const int h_in, const int w_in, \
    const int h_out, const int w_out, ARMContext* ctx) {

    int right_pad_idx[8] = {0, 2, 4, 6, 1, 3, 5, 7};
    int out_pad_idx[4] = {0, 1, 2, 3};
    float zeros[8] = {0.0f};

    uint32x4_t vmask_rp1 = vcgtq_s32(vdupq_n_s32(w_in), vld1q_s32(right_pad_idx));//0 2 4 6
    uint32x4_t vmask_rp2 = vcgtq_s32(vdupq_n_s32(w_in), vld1q_s32(right_pad_idx + 4));//1 3 5 7

    int size_in_channel = w_in * h_in;
    int size_out_channel = w_out * h_out;

    unsigned int dmask[8];
    vst1q_u32(dmask, vmask_rp1);
    vst1q_u32(dmask + 4, vmask_rp2);

    for (int n = 0; n < num; ++n) {
        const float *din_batch = din + n * ch_in * size_in_channel;
        float *dout_batch = dout + n * ch_in * size_out_channel;
#pragma omp parallel for
        for (int i = 0; i < ch_in; ++i) {
            const float* din_channel = din_batch + i * size_in_channel;
            float* dout_channel = dout_batch + i * size_out_channel;

            const float *weight_ptr = weights + i * 9;
            float32x4_t wr0 = vld1q_f32(weight_ptr);
            float32x4_t wr1 = vld1q_f32(weight_ptr + 3);
            float32x4_t wr2 = vld1q_f32(weight_ptr + 6);

            float bias_c = 0.f;

            if (flag_bias) {
                bias_c  = bias[i];
            }
            float32x4_t vbias = vdupq_n_f32(bias_c);
            float out_buf[4];
            const float *dr0 = din_channel;
            const float *dr1 = dr0 + w_in;
            const float *dr2 = dr1 + w_in;
            for (int j = 0; j < h_out; ++j){

                const float *din0_ptr = dr0;
                const float *din1_ptr = dr1;
                const float *din2_ptr = dr2;

                dr0 = dr2;
                dr1 = dr0 + w_in;
                dr2 = dr1 + w_in;

                unsigned int* mask_ptr = dmask;
#ifdef __aarch64__
                asm volatile (
                // Load up 12 elements (3 vectors) from each of 8 sources.
                        "movi v9.4s, #0                                 \n"
                        "ld1  {v6.4s, v7.4s}, [%[mask_ptr]], #32        \n"

                        "ld2  {v10.4s, v11.4s}, [%[din0_ptr]], #32      \n"//v10={0,2,4,6} v11={1,3,5,7}
                        "ld2  {v12.4s, v13.4s}, [%[din1_ptr]], #32      \n"//v13={0,2,4,6} v12={1,3,5,7}
                        "ld2  {v14.4s, v15.4s}, [%[din2_ptr]], #32      \n"//v14={0,2,4,6} v15={1,3,5,7}
                        "and  v4.16b, %[bias].16b, %[bias].16b  \n" //v10 = vbias

                        "bif v10.16b, v9.16b, v6.16b                    \n"
                        "bif v11.16b, v9.16b, v7.16b                    \n"
                        "bif v12.16b, v9.16b, v6.16b                    \n"
                        "bif v13.16b, v9.16b, v7.16b                    \n"
                        "bif v14.16b, v9.16b, v6.16b                    \n"
                        "bif v15.16b, v9.16b, v7.16b                    \n"

                        "ext v6.16b, v10.16b, v9.16b, #4               \n" //v6 = {2,4,6,8}
                        "ext v7.16b, v12.16b, v9.16b, #4               \n" //v6 = {2,4,6,8}
                        "ext v8.16b, v14.16b, v9.16b, #4               \n" //v6 = {2,4,6,8}

                        "fmla v4.4s, v10.4s, %[wr0].s[0]                \n"// 0246 * w00
                        "fmul v5.4s, v11.4s, %[wr0].s[1]                \n"// 1357 * w01
                        "fmul v16.4s, v6.4s,  %[wr0].s[2]                \n"// 2468  * w02

                        "fmla v4.4s, v12.4s, %[wr1].s[0]                \n"// v12 * w11
                        "fmla v5.4s, v13.4s, %[wr1].s[1]                \n"// v13 * w12
                        "fmla v16.4s, v7.4s,  %[wr1].s[2]                \n"// v7  * w10

                        "fmla v4.4s, v14.4s, %[wr2].s[0]                \n"// v14 * w20
                        "fmla v5.4s, v15.4s, %[wr2].s[1]                \n"// v15 * w21
                        "fmla v16.4s, v8.4s,  %[wr2].s[2]                \n"// v8  * w22

                        "fadd v4.4s, v4.4s, v5.4s                       \n"
                        "fadd v4.4s, v4.4s, v16.4s                       \n"

                        // "fadd v4.4s, v4.4s, %[bias].4s                  \n"
                        "st1 {v4.4s}, [%[out]]                          \n"
                : [din0_ptr] "+r"(din0_ptr), [din1_ptr] "+r"(din1_ptr), [din2_ptr] "+r"(din2_ptr), \
                    [mask_ptr] "+r"(mask_ptr)
                : [wr0] "w" (wr0), [wr1] "w" (wr1), [wr2] "w" (wr2), [bias] "w"(vbias), [out] "r" (out_buf)
                : "cc", "memory", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16"
                );

#else
               asm volatile (
                // Load up 12 elements (3 vectors) from each of 8 sources.
                        "vmov.u32 q9, #0                                \n"
                        "vld1.f32   {d12-d15}, [%[mask_ptr]]           @ load mask\n"
                        "vdup.32  q3, %[bias]                           @ and \n" //q3 = vbias

                        "vld2.32  {d20-d23}, [%[din0_ptr]]!             @ load din r0\n"//q10={0,2,4,6} q11={1,3,5,7}
                        "vld2.32  {d24-d27}, [%[din1_ptr]]!             @ load din r1\n"//q13={0,2,4,6} q12={1,3,5,7}
                        "vld2.32  {d28-d31}, [%[din2_ptr]]!             @ load din r2\n"//q14={0,2,4,6} q15={1,3,5,7}

                        "vbif q10, q9, q6                               @ bit select, deal with right pad\n"
                        "vbif q11, q9, q7                               @ bit select, deal with right pad\n"
                        "vbif q12, q9, q6                               @ bit select, deal with right pad\n"
                        "vbif q13, q9, q7                               @ bit select, deal with right pad\n"
                        "vbif q14, q9, q6                               @ bit select, deal with right pad\n"
                        "vbif q15, q9, q7                               @ bit select, deal with right pad\n"

                        "vext.32 q6, q10, q9, #1                        @ shift left 1 \n" //q6 = {2,4,6,0}
                        "vext.32 q7, q12, q9, #1                        @ shift left 1 \n" //q7 = {2,4,6,0}
                        "vext.32 q8, q14, q9, #1                        @ shift left 1 \n" //q8 = {2,4,6,0}

                        "vmul.f32 q4, q10, %e[wr0][0]                   @ mul weight 0, out0\n"// {0,2,4,6}
                        "vmul.f32 q5, q11, %e[wr0][1]                   @ mul weight 0, out0\n"// {1,3,5,7}
                        "vmla.f32 q3, q6,  %f[wr0][0]                   @ mul weight 0, out0\n"// {2,4,6,0}

                        "vmla.f32 q4, q12, %e[wr1][0]                   @ mul weight 1, out0\n"// q12 * w11
                        "vmla.f32 q5, q13, %e[wr1][1]                   @ mul weight 1, out0\n"// q13 * w12
                        "vmla.f32 q3, q7,  %f[wr1][0]                   @ mul weight 1, out0\n"// q7  * w10

                        "vmla.f32 q4, q14, %e[wr2][0]                   @ mul weight 2, out0\n"// q14 * w20
                        "vmla.f32 q5, q15, %e[wr2][1]                   @ mul weight 2, out0\n"// q15 * w21
                        "vmla.f32 q3, q8,  %f[wr2][0]                   @ mul weight 2, out0\n"// q8  * w22

                        "vadd.f32 q3, q3, q4                            @ add \n"
                        "vadd.f32 q3, q3, q5                            @ add \n"

                        "vst1.32 {d6-d7}, [%[out]]                            \n"
                : [din0_ptr] "+r"(din0_ptr), [din1_ptr] "+r"(din1_ptr), [din2_ptr] "+r"(din2_ptr)
                : [wr0] "w" (wr0), [wr1] "w" (wr1), [wr2] "w" (wr2), [bias] "r"(bias_c), [out] "r" (out_buf), \
                  [mask_ptr] "r"(dmask)
                : "cc", "memory", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
                );
#endif //__aarch64__
                for (int w = 0; w < w_out; ++w){
                    *dout_channel++ = out_buf[w];
                }
            }

        }
    }
}
/**
 * \brief depthwise convolution, kernel size 3x3, stride 1, pad 1, with bias, width <= 4
 */
void conv_depthwise_3x3s1p0_bias_s_relu(float* dout, const float* din, \
    const float* weights, const float* bias, bool flag_bias, \
    const int num, const int ch_in, const int h_in, const int w_in, \
    const int h_out, const int w_out, ARMContext* ctx) {
    //! 3x3s1 convolution, implemented by direct algorithm
    //! pad is done implicit
    //! for 4x6 convolution window
    const int right_pad_idx[8] = {5, 4, 3, 2, 1, 0, 0, 0};
    const float zero_ptr[4] = {0.f, 0.f, 0.f, 0.f};

    float32x4_t vzero = vdupq_n_f32(0.f);
    uint32x4_t vmask_rp1 = vcgeq_s32(vld1q_s32(right_pad_idx), vdupq_n_s32(6 - w_in));
    uint32x4_t vmask_rp2 = vcgeq_s32(vld1q_s32(right_pad_idx + 4), vdupq_n_s32(6 - w_in));

    unsigned int vmask[8];
    vst1q_u32(vmask, vmask_rp1);
    vst1q_u32(vmask + 4, vmask_rp2);

    int size_in_channel = w_in * h_in;
    int size_out_channel = w_out * h_out;
    for (int n = 0; n < num; ++n) {
        const float *din_batch = din + n * ch_in * size_in_channel;
        float *dout_batch = dout + n * ch_in * size_out_channel;
#pragma omp parallel for
        for (int i = 0; i < ch_in; ++i) {
            float *dout_channel = dout_batch + i * size_out_channel;
            const float *din_channel = din_batch + i * size_in_channel;
            const float *weight_ptr = weights + i * 9;
            float32x4_t wr0 = vld1q_f32(weight_ptr);
            float32x4_t wr1 = vld1q_f32(weight_ptr + 3);
            float32x4_t wr2 = vld1q_f32(weight_ptr + 6);
            float32x4_t wbias;
            if (flag_bias) {
                wbias  = vdupq_n_f32(bias[i]);
            } else {
                wbias = vdupq_n_f32(0.f);
            }

            float out_buf1[4];
            float out_buf2[4];
            float trash_buf[4];

            float *doutr0 = dout_channel;
            float *doutr1 = dout_channel + w_out;

            for (int j = 0; j < h_out; j += 2){

                const float *dr0 = din_channel + j * w_in;
                const float *dr1 = dr0 + w_in;
                const float *dr2 = dr1 + w_in;
                const float *dr3 = dr2 + w_in;

                doutr0 = dout_channel + j * w_out;
                doutr1 = doutr0 + w_out;

                if (j + 3 >= h_in) {
                    switch (j + 3 - h_in) {
                        case 3:
                            dr1 = zero_ptr;
                        case 2:
                            dr2 = zero_ptr;
                        case 1:
                            dr3 = zero_ptr;
                            doutr1 = trash_buf;
                        case 0:
                            dr3 = zero_ptr;
                            doutr1 = trash_buf;
                        default:
                            break;
                    }
                }
#ifdef __aarch64__
                asm volatile(
                    "prfm pldl1keep, [%[din0]]\n"
                    "prfm pldl1keep, [%[din1]]\n"
                    "prfm pldl1keep, [%[din2]]\n"
                    "prfm pldl1keep, [%[din3]]\n"

                    "ld1 {v0.4s, v1.4s}, [%[din0]]\n"
                    "ld1 {v2.4s, v3.4s}, [%[din1]]\n"
                    "ld1 {v4.4s, v5.4s}, [%[din2]]\n"
                    "ld1 {v6.4s, v7.4s}, [%[din3]]\n"

                    "bif v0.16b, %[zero].16b, %[mask1].16b\n"  //d0_1234
                    "bif v1.16b, %[zero].16b, %[mask2].16b\n"  //d0_1234

                    "bif v2.16b, %[zero].16b, %[mask1].16b\n"  //d1_1234
                    "bif v3.16b, %[zero].16b, %[mask2].16b\n"  //d1_1234

                    "bif v4.16b, %[zero].16b, %[mask1].16b\n"  //d2_1234
                    "bif v5.16b, %[zero].16b, %[mask2].16b\n"  //d2_1234

                    "bif v6.16b, %[zero].16b, %[mask1].16b\n"  //d3_1234
                    "bif v7.16b, %[zero].16b, %[mask2].16b\n"  //d3_1234

                    "ext v8.16b, v0.16b, v1.16b, #4\n"  //d1_2345
                    "ext v9.16b, v0.16b, v1.16b, #8\n"  //d1_3450

                    "and  v12.16b, %[vbias].16b, %[vbias].16b  \n" //v12 = vbias
                    "and  v13.16b, %[vbias].16b, %[vbias].16b  \n" //v13 = vbias

                    //r0
                    "fmul v10.4s, v0.4s, %[wr0].s[0]\n"//d0_1234 * w0[0]
                    "fmul v11.4s, v8.4s, %[wr0].s[1]\n"//d1_2345 * w0[1]
                    "fmla v12.4s, v9.4s, %[wr0].s[2]\n"//d0_3456 * w0[2]

                    "ext v8.16b, v2.16b, v3.16b, #4\n"  //d1_2345
                    "ext v9.16b, v2.16b, v3.16b, #8\n"  //d1_3450

                    //r1
                    "fmul v14.4s, v2.4s, %[wr0].s[0]\n"//d0_1234 * w0[0]
                    "fmla v10.4s, v2.4s, %[wr1].s[0]\n"//d0_1234 * w0[0]

                    "fmul v15.4s, v8.4s, %[wr0].s[1]\n"//d1_2345 * w0[1]
                    "fmla v11.4s, v8.4s, %[wr1].s[1]\n"//d1_2345 * w0[1]

                    "fmla v13.4s, v9.4s, %[wr0].s[2]\n"//d0_3456 * w0[2]
                    "fmla v12.4s, v9.4s, %[wr1].s[2]\n"//d0_3456 * w0[2]

                    "ext v8.16b, v4.16b, v5.16b, #4\n"  //d1_2345
                    "ext v9.16b, v4.16b, v5.16b, #8\n"  //d1_3450

                    //r2
                    "fmla v14.4s, v4.4s, %[wr1].s[0]\n"//d0_1234 * w0[0]
                    "fmla v10.4s, v4.4s, %[wr2].s[0]\n"//d0_1234 * w0[0]

                    "fmla v15.4s, v8.4s, %[wr1].s[1]\n"//d1_2345 * w0[1]
                    "fmla v11.4s, v8.4s, %[wr2].s[1]\n"//d1_2345 * w0[1]

                    "fmla v13.4s, v9.4s, %[wr1].s[2]\n"//d0_3456 * w0[2]
                    "fmla v12.4s, v9.4s, %[wr2].s[2]\n"//d0_3456 * w0[2]

                    "ext v8.16b, v6.16b, v7.16b, #4\n"  //d1_2345
                    "ext v9.16b, v6.16b, v7.16b, #8\n"  //d1_3450

                    //r3
                    "fmla v14.4s, v6.4s, %[wr2].s[0]\n"//d0_1234 * w0[0]

                    "fmla v15.4s, v8.4s, %[wr2].s[1]\n"//d1_2345 * w0[1]

                    "fadd v12.4s, v12.4s, v10.4s\n"

                    "fmla v13.4s, v9.4s, %[wr2].s[2]\n"//d0_3456 * w0[2]

                    "fadd v12.4s, v12.4s, v11.4s\n" //out1
                    "fadd v13.4s, v13.4s, v14.4s\n" //out2
                    "fadd v13.4s, v13.4s, v15.4s\n" //out2

                    "prfm pldl1keep, [%[out1]]\n"
                    "prfm pldl1keep, [%[out2]]\n"
                    "fmax v12.4s, v12.4s, %[zero].4s                       \n"
                    "fmax v13.4s, v13.4s, %[zero].4s                       \n"

                    "st1 {v12.4s}, [%[out1]]\n"
                    "st1 {v13.4s}, [%[out2]]\n"
                : [din0] "+r"(dr0), [din1] "+r"(dr1), [din2] "+r"(dr2), [din3] "+r"(dr3)
                : [wr0]"w"(wr0), [wr1]"w"(wr1), [wr2]"w"(wr2), [vbias] "w" (wbias), \
                    [mask1] "w" (vmask_rp1), [mask2] "w" (vmask_rp2), [zero] "w" (vzero), \
                    [out1] "r" (out_buf1), [out2] "r" (out_buf2)
                :"cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",\
                      "v10", "v11", "v12", "v13", "v14", "v15"
                );
#else
                unsigned int* vmask_ptr = vmask;
                float bias_val = flag_bias ? bias[i] : 0.f;
                asm volatile(
                    "pld [%[din0]]\n"
                    "pld [%[din1]]\n"
                    "pld [%[din2]]\n"
                    "pld [%[din3]]\n"

                    "vld1.32  {d16-d18}, [%[din0]]    @ load din r0\n"
                    "vld1.32  {d20-d22}, [%[din1]]    @ load din r1\n"
                    "vld1.32  {d24-d26}, [%[din2]]    @ load din r2\n"
                    "vld1.32  {d28-d30}, [%[din3]]    @ load din r3\n"

                    "vdup.32 q4, %[bias_val]                            @ and \n" //q4 = vbias
                    "vdup.32 q5, %[bias_val]                            @ and \n" //q5 = vbias

                    "vld1.32  {d19}, [%[vmask]]!    @ load din r0\n"
                    "vld1.32  {d23}, [%[vmask]]!    @ load din r0\n"

                    "vld1.32  {d27}, [%[vmask]]!    @ load din r0\n"

                    "vbif d16, %e[vzero], d19              @ bit select, deal with right pad\n"
                    "vbif d20, %e[vzero], d19              @ bit select, deal with right pad\n"

                    "vbif d17, %e[vzero], d23              @ bit select, deal with right pad\n"
                    "vbif d21, %e[vzero], d23              @ bit select, deal with right pad\n"

                    "vbif d18, %e[vzero], d27             @ bit select, deal with right pad\n"
                    "vbif d22, %e[vzero], d27             @ bit select, deal with right pad\n"

                    "vext.32  q6, q8, q9, #1     @ 1234\n"
                    "vext.32  q7, q8, q9, #2     @ 2345\n"

                    //r0
                    "vmla.f32 q4, q8, %e[wr0][0]  @ q4 += 0123 * wr0[0]\n"

                    "vbif d24, %e[vzero], d19              @ bit select, deal with right pad\n"
                    "vbif d25, %e[vzero], d23              @ bit select, deal with right pad\n"
                    "vbif d26, %e[vzero], d27             @ bit select, deal with right pad\n"

                    "vmla.f32 q4, q6, %e[wr0][1]  @ q4 += 1234 * wr0[1]\n"

                    "vbif d28, %e[vzero], d19              @ bit select, deal with right pad\n"
                    "vbif d29, %e[vzero], d23              @ bit select, deal with right pad\n"
                    "vbif d30, %e[vzero], d27             @ bit select, deal with right pad\n"

                    "vmla.f32 q4, q7, %f[wr0][0]  @ q4 += 2345 * wr0[2]\n"

                    "vext.32  q6, q10, q11, #1     @ 1234\n"
                    "vext.32  q7, q10, q11, #2     @ 2345\n"

                    //r1
                    "vmla.f32 q5, q10, %e[wr0][0]  @ q4 += 1234 * wr0[0]\n"
                    "vmla.f32 q4, q10, %e[wr1][0]  @ q4 += 1234 * wr0[1]\n"

                    "vmul.f32 q8, q6, %e[wr0][1]  @ q4 += 1234 * wr0[1]\n"
                    "vmul.f32 q10, q6, %e[wr1][1]  @ q4 += 1234 * wr0[1]\n"

                    "vmul.f32 q9, q7, %f[wr0][0]  @ q4 += 1234 * wr0[1]\n"
                    "vmul.f32 q11, q7, %f[wr1][0]  @ q4 += 1234 * wr0[1]\n"

                    "vext.32  q6, q12, q13, #1     @ 1234\n"
                    "vext.32  q7, q12, q13, #2     @ 2345\n"

                    //r2
                    "vmla.f32 q5, q12, %e[wr1][0]  @ q4 += 1234 * wr0[0]\n"
                    "vmla.f32 q4, q12, %e[wr2][0]  @ q4 += 1234 * wr0[1]\n"

                    "vmla.f32 q8, q6, %e[wr1][1]  @ q4 += 1234 * wr0[1]\n"
                    "vmla.f32 q10, q6, %e[wr2][1]  @ q4 += 1234 * wr0[1]\n"

                    "vmla.f32 q9, q7, %f[wr1][0]  @ q4 += 1234 * wr0[1]\n"
                    "vmla.f32 q11, q7, %f[wr2][0]  @ q4 += 1234 * wr0[1]\n"

                    "vext.32  q6, q14, q15, #1     @ 1234\n"
                    "vext.32  q7, q14, q15, #2     @ 2345\n"

                    //r3
                    "vmla.f32 q5, q14, %e[wr2][0]  @ q4 += 0123 * wr0[0]\n"

                    "vmla.f32 q8, q6, %e[wr2][1]  @ q4 += 1234 * wr0[1]\n"
                    "vadd.f32 q4, q4, q10         @ q4 += q10 \n"

                    "pld [%[out1]]\n"
                    "pld [%[out2]]\n"

                    "vmla.f32 q9, q7, %f[wr2][0]  @ q4 += 2345 * wr0[2]\n"
                    "vadd.f32 q4, q4, q11         @ q4 += q10 \n"

                    "vadd.f32 q5, q5, q8         @ q4 += q10 \n"
                    "vadd.f32 q5, q5, q9         @ q4 += q10 \n"
                    "vmax.f32 q4, q4, %q[vzero]   @ relu \n"
                    "vmax.f32 q5, q5, %q[vzero]   @ relu \n"

                    "vst1.32  {d8-d9},   [%[out1]]  @ store result, add pointer\n"
                    "vst1.32  {d10-d11},   [%[out2]]  @ store result, add pointer\n"

                    :[din0] "+r" (dr0), [din1] "+r" (dr1), [din2] "+r" (dr2), [din3] "+r" (dr3), \
                      [vmask] "+r" (vmask_ptr)
                    :[wr0] "w" (wr0), [wr1] "w" (wr1), [wr2] "w" (wr2), [vzero] "w" (vzero), \
                       [bias_val] "r" (bias_val), [out1] "r" (out_buf1), [out2] "r" (out_buf2)
                    :"cc", "memory", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
                );
#endif //__aarch64__
                for (int w = 0; w < w_out; ++w){
                    *doutr0++ = out_buf1[w];
                    *doutr1++ = out_buf2[w];
                };
                // doutr0 = doutr1;
                // doutr1 += w_out;
            }// end of processing heights
        } // end of processing channels
    } // end of processing batchs
}



/**
 * \brief depthwise convolution kernel 3x3, stride 2, width <= 7
 */
void conv_depthwise_3x3s2p0_bias_s_relu(float* dout, const float* din, \
    const float* weights, const float* bias, bool flag_bias, \
    const int num, const int ch_in, const int h_in, const int w_in, \
    const int h_out, const int w_out, ARMContext* ctx) {

    int right_pad_idx[8] = {0, 2, 4, 6, 1, 3, 5, 7};
    int out_pad_idx[4] = {0, 1, 2, 3};
    float zeros[8] = {0.0f};

    uint32x4_t vmask_rp1 = vcgtq_s32(vdupq_n_s32(w_in), vld1q_s32(right_pad_idx));//0 2 4 6
    uint32x4_t vmask_rp2 = vcgtq_s32(vdupq_n_s32(w_in), vld1q_s32(right_pad_idx + 4));//1 3 5 7

    int size_in_channel = w_in * h_in;
    int size_out_channel = w_out * h_out;

    unsigned int dmask[8];
    vst1q_u32(dmask, vmask_rp1);
    vst1q_u32(dmask + 4, vmask_rp2);

    for (int n = 0; n < num; ++n) {
        const float *din_batch = din + n * ch_in * size_in_channel;
        float *dout_batch = dout + n * ch_in * size_out_channel;
#pragma omp parallel for
        for (int i = 0; i < ch_in; ++i) {
            const float* din_channel = din_batch + i * size_in_channel;
            float* dout_channel = dout_batch + i * size_out_channel;

            const float *weight_ptr = weights + i * 9;
            float32x4_t wr0 = vld1q_f32(weight_ptr);
            float32x4_t wr1 = vld1q_f32(weight_ptr + 3);
            float32x4_t wr2 = vld1q_f32(weight_ptr + 6);

            float bias_c = 0.f;

            if (flag_bias) {
                bias_c  = bias[i];
            }
            float32x4_t vbias = vdupq_n_f32(bias_c);
            float out_buf[4];
            const float *dr0 = din_channel;
            const float *dr1 = dr0 + w_in;
            const float *dr2 = dr1 + w_in;
            for (int j = 0; j < h_out; ++j){

                const float *din0_ptr = dr0;
                const float *din1_ptr = dr1;
                const float *din2_ptr = dr2;

                dr0 = dr2;
                dr1 = dr0 + w_in;
                dr2 = dr1 + w_in;

                unsigned int* mask_ptr = dmask;
#ifdef __aarch64__
                asm volatile (
                // Load up 12 elements (3 vectors) from each of 8 sources.
                        "movi v9.4s, #0                                 \n"
                        "ld1  {v6.4s, v7.4s}, [%[mask_ptr]]        \n"

                        "ld2  {v10.4s, v11.4s}, [%[din0_ptr]], #32      \n"//v10={0,2,4,6} v11={1,3,5,7}
                        "ld2  {v12.4s, v13.4s}, [%[din1_ptr]], #32      \n"//v13={0,2,4,6} v12={1,3,5,7}
                        "ld2  {v14.4s, v15.4s}, [%[din2_ptr]], #32      \n"//v14={0,2,4,6} v15={1,3,5,7}
                        "and  v4.16b, %[bias].16b, %[bias].16b  \n" //v10 = vbias

                        "bif v10.16b, v9.16b, v6.16b                    \n"
                        "bif v11.16b, v9.16b, v7.16b                    \n"
                        "bif v12.16b, v9.16b, v6.16b                    \n"
                        "bif v13.16b, v9.16b, v7.16b                    \n"
                        "bif v14.16b, v9.16b, v6.16b                    \n"
                        "bif v15.16b, v9.16b, v7.16b                    \n"

                        "ext v6.16b, v10.16b, v9.16b, #4               \n" //v6 = {2,4,6,8}
                        "ext v7.16b, v12.16b, v9.16b, #4               \n" //v6 = {2,4,6,8}
                        "ext v8.16b, v14.16b, v9.16b, #4               \n" //v6 = {2,4,6,8}

                        "fmla v4.4s, v10.4s, %[wr0].s[0]                \n"// 0246 * w00
                        "fmul v5.4s, v11.4s, %[wr0].s[1]                \n"// 1357 * w01
                        "fmul v16.4s, v6.4s,  %[wr0].s[2]                \n"// 2468  * w02

                        "fmla v4.4s, v12.4s, %[wr1].s[0]                \n"// v12 * w11
                        "fmla v5.4s, v13.4s, %[wr1].s[1]                \n"// v13 * w12
                        "fmla v16.4s, v7.4s,  %[wr1].s[2]                \n"// v7  * w10

                        "fmla v4.4s, v14.4s, %[wr2].s[0]                \n"// v14 * w20
                        "fmla v5.4s, v15.4s, %[wr2].s[1]                \n"// v15 * w21
                        "fmla v16.4s, v8.4s,  %[wr2].s[2]                \n"// v8  * w22

                        "fadd v4.4s, v4.4s, v5.4s                       \n"
                        "fadd v4.4s, v4.4s, v16.4s                       \n"
                        "fmax v4.4s, v4.4s, v9.4s                       \n"

                        // "fadd v4.4s, v4.4s, %[bias].4s                  \n"
                        "st1 {v4.4s}, [%[out]]                          \n"
                : [din0_ptr] "+r"(din0_ptr), [din1_ptr] "+r"(din1_ptr), [din2_ptr] "+r"(din2_ptr)
                : [wr0] "w" (wr0), [wr1] "w" (wr1), [wr2] "w" (wr2), [bias] "w"(vbias), [out] "r" (out_buf), \
                  [mask_ptr] "r"(mask_ptr)
                : "cc", "memory", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16"
                );

#else
                asm volatile (
                // Load up 12 elements (3 vectors) from each of 8 sources.
                        "vmov.u32 q9, #0                                \n"
                        "vld1.f32   {d12-d15}, [%[mask_ptr]]           @ load mask\n"
                        "vdup.32  q3, %[bias]                           @ and \n" //q3 = vbias

                        "vld2.32  {d20-d23}, [%[din0_ptr]]!             @ load din r0\n"//q10={0,2,4,6} q11={1,3,5,7}
                        "vld2.32  {d24-d27}, [%[din1_ptr]]!             @ load din r1\n"//q13={0,2,4,6} q12={1,3,5,7}
                        "vld2.32  {d28-d31}, [%[din2_ptr]]!             @ load din r2\n"//q14={0,2,4,6} q15={1,3,5,7}

                        "vbif q10, q9, q6                               @ bit select, deal with right pad\n"
                        "vbif q11, q9, q7                               @ bit select, deal with right pad\n"
                        "vbif q12, q9, q6                               @ bit select, deal with right pad\n"
                        "vbif q13, q9, q7                               @ bit select, deal with right pad\n"
                        "vbif q14, q9, q6                               @ bit select, deal with right pad\n"
                        "vbif q15, q9, q7                               @ bit select, deal with right pad\n"

                        "vext.32 q6, q10, q9, #1                        @ shift left 1 \n" //q6 = {2,4,6,0}
                        "vext.32 q7, q12, q9, #1                        @ shift left 1 \n" //q7 = {2,4,6,0}
                        "vext.32 q8, q14, q9, #1                        @ shift left 1 \n" //q8 = {2,4,6,0}

                        "vmul.f32 q4, q10, %e[wr0][0]                   @ mul weight 0, out0\n"// {0,2,4,6}
                        "vmul.f32 q5, q11, %e[wr0][1]                   @ mul weight 0, out0\n"// {1,3,5,7}
                        "vmla.f32 q3, q6,  %f[wr0][0]                   @ mul weight 0, out0\n"// {2,4,6,0}

                        "vmla.f32 q4, q12, %e[wr1][0]                   @ mul weight 1, out0\n"// q12 * w11
                        "vmla.f32 q5, q13, %e[wr1][1]                   @ mul weight 1, out0\n"// q13 * w12
                        "vmla.f32 q3, q7,  %f[wr1][0]                   @ mul weight 1, out0\n"// q7  * w10

                        "vmla.f32 q4, q14, %e[wr2][0]                   @ mul weight 2, out0\n"// q14 * w20
                        "vmla.f32 q5, q15, %e[wr2][1]                   @ mul weight 2, out0\n"// q15 * w21
                        "vmla.f32 q3, q8,  %f[wr2][0]                   @ mul weight 2, out0\n"// q8  * w22

                        "vadd.f32 q3, q3, q4                            @ add \n"
                        "vadd.f32 q3, q3, q5                            @ add \n"

                        "vmax.f32 q3, q3, q9                            @ relu \n"

                        "vst1.32 {d6-d7}, [%[out]]                            \n"
                : [din0_ptr] "+r"(din0_ptr), [din1_ptr] "+r"(din1_ptr), [din2_ptr] "+r"(din2_ptr)
                : [wr0] "w" (wr0), [wr1] "w" (wr1), [wr2] "w" (wr2), [bias] "r"(bias_c), [out] "r" (out_buf), \
                  [mask_ptr] "r"(mask_ptr)
                : "cc", "memory", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
                );
#endif //__aarch64__
                for (int w = 0; w < w_out; ++w){
                    *dout_channel++ = out_buf[w];
                }
            }

        }
    }
}

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
