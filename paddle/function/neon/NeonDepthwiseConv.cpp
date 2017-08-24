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

#include "neon_util.h"
#include "paddle/function/ConvOp.h"
#include "paddle/function/Im2Col.h"

namespace paddle {

namespace neon {

#if defined(__ARM_NEON__) || defined(__ARM_NEON)

template <int filterSize, int stride>
struct DepthwiseConvKernel {};

inline float32_t conv3x3(float32x4_t r0,
                         float32x4_t r1,
                         float32x4_t r2,
                         float32x4_t k0,
                         float32x4_t k1,
                         float32x4_t k2) {
  float32x4_t tmp;
  tmp = vmulq_f32(r0, k0);
  tmp = vmlaq_f32(tmp, r1, k1);
  tmp = vmlaq_f32(tmp, r2, k2);
  return vaddvq_f32(tmp);
}

inline float32_t conv4x4(float32x4_t r0,
                         float32x4_t r1,
                         float32x4_t r2,
                         float32x4_t r3,
                         float32x4_t k0,
                         float32x4_t k1,
                         float32x4_t k2,
                         float32x4_t k3) {
  float32x4_t tmp;
  tmp = vmulq_f32(r0, k0);
  tmp = vmlaq_f32(tmp, r1, k1);
  tmp = vmlaq_f32(tmp, r2, k2);
  tmp = vmlaq_f32(tmp, r3, k3);
  return vaddvq_f32(tmp);
}

/**
 * Each step calculates four elements of the output.
 * First step:
 *   R0[0, 1, 2, 3...] * K[0][0]
 *   R0[1, 2, 3, 4...] * K[0][1]
 *   R0[2, 3, 4, 5...] * K[0][2]
 *   R1[0, 1, 2, 3...] * K[1][0]
 *   R1[1, 2, 3, 4...] * K[1][1]
 *   R1[2, 3, 4, 5...] * K[1][2]
 *   R2[0, 1, 2, 3...] * K[2][0]
 *   R2[1, 2, 3, 4...] * K[2][1]
 * + R2[2, 3, 4, 5...] * K[2][2]
 * ------------------------------
 *     Output[0, 1, 2, 3]
 */
template <>
struct DepthwiseConvKernel<3, 1> {
  static void run(const float* inputData,
                  const float* filterData,
                  int inputHeight,
                  int inputWidth,
                  int outputChannels,
                  int outputHeight,
                  int outputWidth,
                  int filterMultiplier,
                  float* outputData) {
    const int steps = outputWidth >> 2;
    const int remain = outputWidth & 3;
    for (int c = 0; c < outputChannels; c++, filterData += 9) {
      // Load the filters
      float32x4_t k[3];
      k[0] = vld1q_f32(filterData);
      k[1] = vld1q_f32(filterData + 3);
      k[2] = vld1q_f32(filterData + 6);
      k[0] = vsetq_lane_f32(0.f, k[0], 3);
      k[1] = vsetq_lane_f32(0.f, k[1], 3);
      k[2] = vsetq_lane_f32(0.f, k[2], 3);

      const float* r0 =
          inputData + (c / filterMultiplier) * (inputHeight * inputWidth);
      const float* r1 = r0 + inputWidth;
      const float* r2 = r0 + inputWidth * 2;
      float32x4_t input[3][3];
      for (int h = 0; h < outputHeight; h++) {
        for (int s = 0; s < steps; s++) {
          // Load the inputs
          float32x4_t tmp;
          input[0][0] = vld1q_f32(r0);
          tmp = vld1q_f32(r0 + 4);
          input[0][1] = vextq_f32(input[0][0], tmp, 1);
          input[0][2] = vextq_f32(input[0][0], tmp, 2);
          input[1][0] = vld1q_f32(r1);
          tmp = vld1q_f32(r1 + 4);
          input[1][1] = vextq_f32(input[1][0], tmp, 1);
          input[1][2] = vextq_f32(input[1][0], tmp, 2);
          input[2][0] = vld1q_f32(r2);
          tmp = vld1q_f32(r2 + 4);
          input[2][1] = vextq_f32(input[2][0], tmp, 1);
          input[2][2] = vextq_f32(input[2][0], tmp, 2);

          float32x4_t tmp1 = vdupq_n_f32(0.f);
          float32x4_t tmp2 = vdupq_n_f32(0.f);
          tmp1 = vmlaq_laneq_f32(tmp1, input[0][0], k[0], 0);
          tmp2 = vmlaq_laneq_f32(tmp2, input[0][1], k[0], 1);
          tmp1 = vmlaq_laneq_f32(tmp1, input[0][2], k[0], 2);
          tmp2 = vmlaq_laneq_f32(tmp2, input[1][0], k[1], 0);
          tmp1 = vmlaq_laneq_f32(tmp1, input[1][1], k[1], 1);
          tmp2 = vmlaq_laneq_f32(tmp2, input[1][2], k[1], 2);
          tmp1 = vmlaq_laneq_f32(tmp1, input[2][0], k[2], 0);
          tmp2 = vmlaq_laneq_f32(tmp2, input[2][1], k[2], 1);
          tmp1 = vmlaq_laneq_f32(tmp1, input[2][2], k[2], 2);
          tmp1 = vaddq_f32(tmp1, tmp2);

          vst1q_f32(outputData, tmp1);
          r0 += 4;
          r1 += 4;
          r2 += 4;
          outputData += 4;
        }

        for (int r = 0; r < remain; r++) {
          float32x4_t i0 = vld1q_f32(r0);
          float32x4_t i1 = vld1q_f32(r1);
          float32x4_t i2 = vld1q_f32(r2);
          *outputData = conv3x3(i0, i1, i2, k[0], k[1], k[2]);
          r0++;
          r1++;
          r2++;
          outputData++;
        }

        r0 += 2;
        r1 += 2;
        r2 += 2;
      }
    }
  }
};

/**
 * Each step calculates four elements of the output.
 */
template <>
struct DepthwiseConvKernel<4, 1> {
  static void run(const float* inputData,
                  const float* filterData,
                  int inputHeight,
                  int inputWidth,
                  int outputChannels,
                  int outputHeight,
                  int outputWidth,
                  int filterMultiplier,
                  float* outputData) {
    const int steps = outputWidth >> 2;
    const int remain = outputWidth & 3;
    for (int c = 0; c < outputChannels; c++, filterData += 16) {
      // Load the filters
      float32x4_t k[4];
      k[0] = vld1q_f32(filterData);
      k[1] = vld1q_f32(filterData + 4);
      k[2] = vld1q_f32(filterData + 8);
      k[3] = vld1q_f32(filterData + 12);

      const float* r0 =
          inputData + (c / filterMultiplier) * (inputHeight * inputWidth);
      const float* r1 = r0 + inputWidth;
      const float* r2 = r0 + inputWidth * 2;
      const float* r3 = r0 + inputWidth * 3;
      float32x4_t input[4][4];
      for (int h = 0; h < outputHeight; h++) {
        for (int s = 0; s < steps; s++) {
          // Load the inputs
          float32x4_t tmp;
          input[0][0] = vld1q_f32(r0);
          tmp = vld1q_f32(r0 + 4);
          input[0][1] = vextq_f32(input[0][0], tmp, 1);
          input[0][2] = vextq_f32(input[0][0], tmp, 2);
          input[0][3] = vextq_f32(input[0][0], tmp, 3);

          input[1][0] = vld1q_f32(r1);
          tmp = vld1q_f32(r1 + 4);
          input[1][1] = vextq_f32(input[1][0], tmp, 1);
          input[1][2] = vextq_f32(input[1][0], tmp, 2);
          input[1][3] = vextq_f32(input[1][0], tmp, 3);

          input[2][0] = vld1q_f32(r2);
          tmp = vld1q_f32(r2 + 4);
          input[2][1] = vextq_f32(input[2][0], tmp, 1);
          input[2][2] = vextq_f32(input[2][0], tmp, 2);
          input[2][3] = vextq_f32(input[2][0], tmp, 3);

          input[3][0] = vld1q_f32(r3);
          tmp = vld1q_f32(r3 + 4);
          input[3][1] = vextq_f32(input[3][0], tmp, 1);
          input[3][2] = vextq_f32(input[3][0], tmp, 2);
          input[3][3] = vextq_f32(input[3][0], tmp, 3);

          float32x4_t tmp1 = vdupq_n_f32(0.f);
          float32x4_t tmp2 = vdupq_n_f32(0.f);
          tmp1 = vmlaq_laneq_f32(tmp1, input[0][0], k[0], 0);
          tmp2 = vmlaq_laneq_f32(tmp2, input[0][1], k[0], 1);
          tmp1 = vmlaq_laneq_f32(tmp1, input[0][2], k[0], 2);
          tmp2 = vmlaq_laneq_f32(tmp2, input[0][3], k[0], 3);
          tmp1 = vmlaq_laneq_f32(tmp1, input[1][0], k[1], 0);
          tmp2 = vmlaq_laneq_f32(tmp2, input[1][1], k[1], 1);
          tmp1 = vmlaq_laneq_f32(tmp1, input[1][2], k[1], 2);
          tmp2 = vmlaq_laneq_f32(tmp2, input[1][3], k[1], 3);
          tmp1 = vmlaq_laneq_f32(tmp1, input[2][0], k[2], 0);
          tmp2 = vmlaq_laneq_f32(tmp2, input[2][1], k[2], 1);
          tmp1 = vmlaq_laneq_f32(tmp1, input[2][2], k[2], 2);
          tmp2 = vmlaq_laneq_f32(tmp2, input[2][3], k[2], 3);
          tmp1 = vmlaq_laneq_f32(tmp1, input[3][0], k[3], 0);
          tmp2 = vmlaq_laneq_f32(tmp2, input[3][1], k[3], 1);
          tmp1 = vmlaq_laneq_f32(tmp1, input[3][2], k[3], 2);
          tmp2 = vmlaq_laneq_f32(tmp2, input[3][3], k[3], 3);
          tmp1 = vaddq_f32(tmp1, tmp2);

          vst1q_f32(outputData, tmp1);
          r0 += 4;
          r1 += 4;
          r2 += 4;
          r3 += 4;
          outputData += 4;
        }

        for (int r = 0; r < remain; r++) {
          float32x4_t i0 = vld1q_f32(r0);
          float32x4_t i1 = vld1q_f32(r1);
          float32x4_t i2 = vld1q_f32(r2);
          float32x4_t i3 = vld1q_f32(r3);
          *outputData = conv4x4(i0, i1, i2, i3, k[0], k[1], k[2], k[3]);
          r0++;
          r1++;
          r2++;
          r3++;
          outputData++;
        }

        r0 += 3;
        r1 += 3;
        r2 += 3;
        r3 += 3;
      }
    }
  }
};

template <DeviceType Device>
class NeonDepthwiseConvFunction : public ConvFunctionBase {
public:
  void init(const FuncConfig& config) override {
    ConvFunctionBase::init(config);
  }

  void check(const BufferArgs& inputs, const BufferArgs& outputs) override {
    const TensorShape& input = inputs[0].shape();
    const TensorShape& filter = inputs[1].shape();
    const TensorShape& output = outputs[0].shape();
    checkShape(input, filter, output);
  }

  void calc(const BufferArgs& inputs, const BufferArgs& outputs) override {
    CHECK_EQ(numInputs_, inputs.size());
    CHECK_EQ(numOutputs_, outputs.size());
    check(inputs, outputs);

    const TensorShape& input = inputs[0].shape();
    const TensorShape& filter = inputs[1].shape();
    const TensorShape& output = outputs[0].shape();

    size_t batchSize = input[0];
    size_t inputChannels = input[1];
    size_t inputHeight = input[2];
    size_t inputWidth = input[3];
    size_t filterHeight = getFilterHeight(filter);
    size_t filterWidth = getFilterWidth(filter);
    size_t outputChannels = output[1];
    size_t outputHeight = output[2];
    size_t outputWidth = output[3];
    size_t filterMultiplier = outputChannels / groups_;
    CHECK_EQ(inputChannels, groups_);

    // only support
    CHECK_EQ(strideH(), strideW());
    CHECK_EQ(filterHeight, filterWidth);
    CHECK_LT(strideH(), size_t(3));

    float* inputData = inputs[0].data<float>();
    float* filterData = inputs[1].data<float>();
    float* outputData = outputs[0].data<float>();

    // padding the input
    float* inputPadding = inputData;
    if (paddingH() > 0 || paddingW() > 0) {
      int newSize = batchSize * inputChannels * (inputHeight + 2 * paddingH()) *
                    (inputWidth + 2 * paddingW());
      resizeBuffer<Device>(newSize);
      inputPadding = reinterpret_cast<float*>(memory_->getBuf());
      Padding<float>::run(inputData,
                          inputPadding,
                          batchSize * inputChannels,
                          inputHeight,
                          inputWidth,
                          paddingH(),
                          paddingW());

      // height and width of padding data
      inputHeight += 2 * paddingH();
      inputWidth += 2 * paddingW();
    }

    for (size_t i = 0; i < batchSize; i++) {
      if (filterWidth == 3) {
        DepthwiseConvKernel<3, 1>::run(inputPadding,
                                       filterData,
                                       inputHeight,
                                       inputWidth,
                                       outputChannels,
                                       outputHeight,
                                       outputWidth,
                                       filterMultiplier,
                                       outputData);
      } else if (filterWidth == 4) {
        DepthwiseConvKernel<4, 1>::run(inputPadding,
                                       filterData,
                                       inputHeight,
                                       inputWidth,
                                       outputChannels,
                                       outputHeight,
                                       outputWidth,
                                       filterMultiplier,
                                       outputData);
      }

      inputPadding += inputChannels * inputHeight * inputWidth;
      outputData += outputChannels * outputHeight * outputWidth;
    }
  }
};

REGISTER_TYPED_FUNC(NeonDepthwiseConv, CPU, NeonDepthwiseConvFunction);

#endif

}  // namespace neon
}  // namespace paddle
