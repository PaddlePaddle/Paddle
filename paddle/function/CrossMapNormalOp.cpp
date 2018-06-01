/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "CrossMapNormalOp.h"
#include "paddle/math/Vector.h"

namespace paddle {

template <>
void CrossMapNormal<DEVICE_TYPE_CPU>(real* outputs,
                                     real* denoms,
                                     const real* inputs,
                                     size_t numSamples,
                                     size_t channels,
                                     size_t height,
                                     size_t width,
                                     size_t size,
                                     real scale,
                                     real pow) {
  size_t oneImage = height * width;
  size_t oneSample = channels * oneImage;

  CpuVector outputsV(numSamples * oneSample, outputs);
  CpuVector inputsV(numSamples * oneSample, const_cast<real*>(inputs));
  CpuVector denomsV(numSamples * oneSample, denoms);

  // f(x) = x * ( 1 + scale * SUM((x)^2) )^(-pow)
  // x represents inputs
  // f(x) represents outputs
  // denoms save the intermediate result for backward
  denomsV = denomsV.constant(1.0);
  const int start = -((int)size - 1) / 2;
  const int end = (int)size + start;
  for (size_t i = 0; i < numSamples; i++) {
    real* oneDenom = denoms + i * oneSample;
    real* oneInput = const_cast<real*>(inputs) + i * oneSample;
    for (int c = 0; c < (int)channels; c++) {
      CpuVector denom(oneImage, oneDenom + c * oneImage);
      for (int s = start; s < end; s++) {
        if (c + s >= 0 && c + s < (int)channels) {
          CpuVector input(oneImage, oneInput + (c + s) * oneImage);
          denom += input.square() * scale;
        }
      }
    }
  }

  outputsV = inputsV * denomsV.pow(-pow);
}

template <>
void CrossMapNormalGrad<DEVICE_TYPE_CPU>(real* inputsGrad,
                                         const real* inputsValue,
                                         const real* outputsValue,
                                         const real* outputsGrad,
                                         const real* denoms,
                                         size_t numSamples,
                                         size_t channels,
                                         size_t height,
                                         size_t width,
                                         size_t size,
                                         real scale,
                                         real pow) {
  size_t oneSample = channels * height * width;
  std::function<CpuVector(real*, size_t)> oneImage = [=](real* data,
                                                         size_t offset) {
    return CpuVector(height * width, data + offset);
  };

  const int start = -((int)size) / 2;
  const int end = (int)size + start;
  const real ratio = -(real)2 * scale * pow;
  for (size_t i = 0; i < numSamples; i++) {
    size_t sOffset = i * oneSample;
    real* oneInputGrad = inputsGrad + sOffset;
    real* oneInputValue = const_cast<real*>(inputsValue) + sOffset;
    real* oneDenom = const_cast<real*>(denoms) + sOffset;
    real* oneOutputGrad = const_cast<real*>(outputsGrad) + sOffset;
    real* oneOutputValue = const_cast<real*>(outputsValue) + sOffset;

    for (int c = 0; c < (int)channels; c++) {
      size_t cOffset = c * height * width;
      CpuVector inputGrad = oneImage(oneInputGrad, cOffset);
      CpuVector inputValue = oneImage(oneInputValue, cOffset);
      CpuVector denom = oneImage(oneDenom, cOffset);
      CpuVector outputGrad = oneImage(oneOutputGrad, cOffset);

      inputGrad = inputGrad + denom.pow(-pow) * outputGrad;
      for (int s = start; s < end; s++) {
        if (c + s >= 0 && c + s < (int)channels) {
          size_t offset = (c + s) * height * width;
          CpuVector output = oneImage(oneOutputValue, offset);
          CpuVector outputGrad = oneImage(oneOutputGrad, offset);
          CpuVector denom = oneImage(oneDenom, offset);

          inputGrad += ((outputGrad * output * ratio) / denom) * inputValue;
        }
      }
    }
  }
}

/**
 * \brief Normalization with across maps.
 *
 * This Function comes from the paper
 * "ImageNet Classification with Deep Convolutional Neural Networks".
 *
 * The original formula is:
 *
 *                                Input(i, x, y)
 * Output(i, x, y) = ----------------------------------------------
 *                                 -- upper
 *                    (k + alpha * >  (Input(j, x, y))^2) ^ (beta)
 *                                 -- j = lower
 *
 * upper is `min(C, c + N/2)`
 * lower if `max(0, c - N/2)`
 *
 * Function implementation:
 *
 * inputs and outpus is NCHW format, while input.shape.ndims() is equal 4.
 * And the meaning of each dimension(0-3) is respectively batch size,
 * feature maps, rows and columns.
 *
 * Input and Output in the above formula is for each map(i) of one image, and
 * Input(i, x, y), Output(i, x, y) represents an element in an image.
 *
 * C is the number of feature maps of one image, and N is a hyper-parameters
 * is configured when Function is initialized. The sum in the denominator
 * is the sum of the same position in the neighboring maps.
 *
 * In the implementation of Function, k is equal to 1,
 * so Function has no argument for k.
 *
 * Function Arguments:
 *
 * \param size_      represent N
 * \param scale_     represent alpha
 * \param pow_       represent beta
 * \param inputs[0]  represent Input
 * \param outputs[0] represent Output
 * \param outputs[1] represent The denominator in the formula(except beta)
 *
 * Note:
 * Save output[1] is to simplify the backward calculation.
 * TODO, if only consider the forward calculation, we can optimize to
 * remove the output[1].
 */
template <DeviceType Device>
class CrossMapNormalFunc : public FunctionBase {
 public:
  void init(const FuncConfig& config) override {
    // function arguments
    size_ = config.get<size_t>("size");
    scale_ = config.get<real>("scale");
    pow_ = config.get<real>("pow");

    // number of inputs and outputs
    numInputs_ = 1;
    numOutputs_ = 2;
  }

  void calc(const BufferArgs& inputs, const BufferArgs& outputs) override {
    check(inputs, outputs);
    // ArgType check still on here,
    // not sure whether it is better to put inside the check.
    CHECK_EQ(outputs[0].getArgType(), ASSIGN_TO);
    CHECK_EQ(outputs[1].getArgType(), ASSIGN_TO);
    size_t batchSize = inputs[0].shape()[0];
    size_t maps = inputs[0].shape()[1];
    size_t rows = inputs[0].shape()[2];
    size_t columns = inputs[0].shape()[3];

    CrossMapNormal<Device>(outputs[0].data<real>(),
                           outputs[1].data<real>(),
                           inputs[0].data<real>(),
                           batchSize,
                           maps,
                           rows,
                           columns,
                           size_,
                           scale_,
                           pow_);
  }

  void check(const BufferArgs& inputs, const BufferArgs& outputs) override {
    CHECK_EQ(numInputs_, inputs.size());
    CHECK_EQ(numOutputs_, outputs.size());

    CHECK_EQ(inputs[0].shape().ndims(), (size_t)4);
    CHECK(inputs[0].shape() == outputs[0].shape());
    CHECK(inputs[0].shape() == outputs[1].shape());
  }

  // Only need the shape of the input, can calculate the
  // floating-point operation.
  size_t ops(const BufferArgs& inputs, const BufferArgs& outputs) override {
    CHECK_EQ((size_t)numInputs_, inputs.size());
    size_t batchSize = inputs[0].shape()[0];
    size_t maps = inputs[0].shape()[1];
    size_t rows = inputs[0].shape()[2];
    size_t columns = inputs[0].shape()[3];

    // number of floating-point operations
    // an approximate value
    size_t ops = batchSize * maps * rows * columns * (size_ * 2 + 3);

    return ops;
  }

 private:
  size_t size_;
  real scale_;
  real pow_;
};

/**
 * \brief Backward calculation for normalization with across maps.
 *
 * Function implementation:
 *
 * The implementation of this Function is derived from the
 * CrossMapNormalFunc implementation.
 *
 * InputGrad = OutputGrad * denoms ^ (-beta)
 *    -- upper
 *  + > (OutputGrad * OutputValue * (-2 * alpha * beta) / denoms) * InputValue
 *    -- lower
 *
 * The data of inputs/outputs format is the same as the forward interface
 * and is NCHW.
 *
 * The upper and lower is the same as forward. The logic of the sum
 * is also the same as forward.
 *
 * Function Arguments:
 *
 * \param size_      represent N
 * \param scale_     represent alpha
 * \param pow_       represent beta
 * \param inputs[0]  represent InputValue, inputs[0] of CrossMapNormalFunc
 * \param inputs[1]  represent OutputValue, outputs[0] of CrossMapNormalFunc
 * \param inputs[2]  represent OutputGrad
 * \param inputs[3]  represent denoms, outputs[1] of CrossMapNormalFunc
 *                   This is the intermediate result that is
 *                   preserved in the forward calculation.
 * \param outputs[0] represent InputGrad
 */
template <DeviceType Device>
class CrossMapNormalGradFunc : public FunctionBase {
 public:
  void init(const FuncConfig& config) override {
    // function arguments
    size_ = config.get<size_t>("size");
    scale_ = config.get<real>("scale");
    pow_ = config.get<real>("pow");

    // number of inputs and outputs
    numInputs_ = 4;
    numOutputs_ = 1;
  }

  void calc(const BufferArgs& inputs, const BufferArgs& outputs) override {
    check(inputs, outputs);
    if (outputs[0].getArgType() != ADD_TO) {
      // Currently, some algorithm implementations are ASSIGN_TO mode,
      // if need to support the ADD_TO calculation, need to clear the output.
      typename Tensor<real, Device>::Vector tmp(
          outputs[0].shape().getElements(), outputs[0].data<real>());
      tmp.zero();
    }

    size_t batchSize = inputs[0].shape()[0];
    size_t maps = inputs[0].shape()[1];
    size_t rows = inputs[0].shape()[2];
    size_t columns = inputs[0].shape()[3];

    CrossMapNormalGrad<Device>(outputs[0].data<real>(),
                               inputs[0].data<real>(),
                               inputs[1].data<real>(),
                               inputs[2].data<real>(),
                               inputs[3].data<real>(),
                               batchSize,
                               maps,
                               rows,
                               columns,
                               size_,
                               scale_,
                               pow_);
  }

  void check(const BufferArgs& inputs, const BufferArgs& outputs) override {
    CHECK_EQ(numInputs_, inputs.size());
    CHECK_EQ(numOutputs_, outputs.size());

    CHECK_EQ(inputs[0].shape().ndims(), (size_t)4);
    CHECK(inputs[0].shape() == inputs[1].shape());
    CHECK(inputs[0].shape() == inputs[2].shape());
    CHECK(inputs[0].shape() == inputs[3].shape());
    CHECK(inputs[0].shape() == outputs[0].shape());
  }

  // Only need the shape of one input, can calculate the
  // floating-point operation.
  size_t ops(const BufferArgs& inputs, const BufferArgs& outputs) override {
    CHECK_LT((size_t)1, inputs.size());
    size_t batchSize = inputs[0].shape()[0];
    size_t maps = inputs[0].shape()[1];
    size_t rows = inputs[0].shape()[2];
    size_t columns = inputs[0].shape()[3];

    // number of floating-point operations
    // an approximate value
    size_t ops = batchSize * maps * rows * columns * (size_ * 4 + 2);

    return ops;
  }

 private:
  size_t size_;
  real scale_;
  real pow_;
};

REGISTER_TYPED_FUNC(CrossMapNormal, CPU, CrossMapNormalFunc);
REGISTER_TYPED_FUNC(CrossMapNormalGrad, CPU, CrossMapNormalGradFunc);
#ifdef PADDLE_WITH_CUDA
REGISTER_TYPED_FUNC(CrossMapNormal, GPU, CrossMapNormalFunc);
REGISTER_TYPED_FUNC(CrossMapNormalGrad, GPU, CrossMapNormalGradFunc);
#endif

}  // namespace paddle
