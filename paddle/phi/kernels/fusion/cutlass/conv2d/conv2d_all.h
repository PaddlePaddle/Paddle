

#include "cutlass/cutlass.h"
#include <map>
#include <vector>
#include <iostream>


namespace phi {
  namespace fusion {

#define check(status) \
if (status != cutlass::Status::kSuccess) {\
    printf("cutlass can not deal with this problem size, skip this kernel!\n");\
    return status;\
  }\

#define WARMUP 10
#define REPEATE 100

#define COMMON_CONV_PARAMS             \
const half *input,  const half *weight,\
const half *bias, half *output,\
int batch, int ic, int ih, int iw,\
int kh, int kw, int oc, int pad_h, int pad_w,\
int stride_h, int stride_w

#define COMMON_CONV_ARGS                                                       \
  input, weight, bias, output, batch, ic, ih, iw, kh, kw, oc, pad_h, pad_w, \
      stride_h, stride_w


#define CONV_RESIDUAL_PARAMS                                                      \
const half *input,  const half *weight,\
const half *bias, const half *residual, half *output,\
int batch, int ic, int ih, int iw,\
int kh, int kw, int oc, int pad_h, int pad_w,\
int stride_h, int stride_w


#define CONV_RESIDUAL_ARGS                                                         \
  input, weight, bias, residual, output, batch, ic, ih, iw, kh, kw, oc, pad_h, pad_w, \
      stride_h, stride_w


void cutlass_conv2d_bias_add_relu(CONV_RESIDUAL_PARAMS);
void cutlass_conv2d_bias_relu_few_channels(COMMON_CONV_PARAMS);
void cutlass_conv2d_bias_relu(COMMON_CONV_PARAMS);
void cutlass_conv2d_bias_silu(COMMON_CONV_PARAMS);
void cutlass_conv2d_bias(COMMON_CONV_PARAMS);

  } // namespace fusion
}// namespace phi

