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

#include "Function.h"

namespace paddle {

/*
 * \brief Based on the ConvFunctionBase class, the forward calculation,
 *        backward input calculation and backward filter calculation
 *        of convolution operations can be implemented.
 *
 * Arguments of forward and backward calculation:
 *   1. Forward calculation of convolution.
 *      inputs = {INPUT, FILTER}, outputs = {OUTPUT}
 *      The first and second input arguments are input image and filter data.
 *      The output argument is output image.
 *
 *   2. Backward input calculation of convolution.
 *      inputs = {OUTPUT_GRAD, FILTER}, outputs = {INPUT_GRAD}
 *      The first and second input arguments are output grad image
 *      and filter data.
 *      The output argument is input grad image.
 *
 *   3. Backward filter calculation of convolution.
 *      inputs = {OUTPUT_GRAD, INPUT}, outputs = {FILTER_GRAD}
 *      The first and second input arguments are output grad image
 *      and input image.
 *      The output argument is filter grad.
 *
 * Arguments format of input, filter and output:
 *   1. Input image, output image, input image gradient, output image gradient
 *      are all NCHW format. Where N is batch size, C is the number of channels,
 *      H and W is the height and width of image or image gradient.
 *
 *   2. The format of the filter data is MCHW, where M is the number of
 *      output image channels, C is the number of input image channels,
 *      H and W is height and width of filter.
 */
class ConvFunctionBase : public FunctionBase {
public:
  void init(const FuncConfig& config) override {
    // function arguments
    strides_ = config.get<std::vector<size_t>>("strides");
    paddings_ = config.get<std::vector<size_t>>("paddings");
    groups_ = config.get<size_t>("groups");

    // number of inputs and outputs
    numInputs_ = 2;
    numOutputs_ = 1;
  }

  virtual void calc(const BufferArgs& inputs, const BufferArgs& outputs) {}

  // input can be INPUT and INPUT_GRAD
  // filter can be FILTER and FILTER_GRAD
  // output can be OUTPUT and OUTPUT_GRAD
  void check(const TensorShape& input,
             const TensorShape& filter,
             const TensorShape& output) {
    // inputs and outputs arguments should be 4-dimensional.
    CHECK_EQ(input.ndims(), (size_t)4);
    CHECK_EQ(filter.ndims(), (size_t)4);
    CHECK_EQ(output.ndims(), (size_t)4);

    // The batchSize of the input needs to be equal to
    // the batchSize of the output.
    CHECK_EQ(input[0], output[0]);

    // The input and output channel dimensions are the second and first
    // dimensions of the filter shape.
    CHECK_EQ(input[1] / groups_, filter[1]);
    CHECK_EQ(output[1], filter[0]);
  }

protected:
  std::vector<size_t> strides_;
  std::vector<size_t> paddings_;

  /// Group size, refer to grouped convolution in
  /// Alex Krizhevsky's paper: when group=2, the first half of the
  /// filters are only connected to the first half of the input channels,
  /// and the second half only connected to the second half.
  size_t groups_;

  inline int strideH() const { return strides_[0]; }

  inline int strideW() const { return strides_[1]; }

  inline int paddingH() const { return paddings_[0]; }

  inline int paddingW() const { return paddings_[1]; }

  // A temporary memory in convolution calculation.
  MemoryHandlePtr memory_;

  template <DeviceType Device>
  void resizeBuffer(size_t newSize) {
    if (!memory_ || newSize * sizeof(real) > memory_->getAllocSize()) {
      if (Device == DEVICE_TYPE_CPU) {
        memory_ = std::make_shared<CpuMemoryHandle>(newSize * sizeof(real));
      } else {
        memory_ = std::make_shared<GpuMemoryHandle>(newSize * sizeof(real));
      }
    }
  }
};

}  // namespace paddle
