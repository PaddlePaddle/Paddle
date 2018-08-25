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
 *   2. The format of the filter data is MCHW, where M is the number of output
 *      image channels, C is the number of input image channels,
 *      H and W is height and width of filter.
 *
 *      If `groups` is greater than 1, the filter's data format should be GMCHW,
 *      where G is the `groups`, and G * M is the number of output image
 *      channels, G * C is the number of input image channels,
 *      H and W is height and width of filter.
 */
class ConvFunctionBase : public FunctionBase {
 public:
  void init(const FuncConfig& config) override {
    // function arguments
    strides_ = config.get<std::vector<size_t>>("strides");
    paddings_ = config.get<std::vector<size_t>>("paddings");
    dilations_ = config.get<std::vector<size_t>>("dilations");
    groups_ = config.get<size_t>("groups");

    // number of inputs and outputs
    numInputs_ = 2;
    numOutputs_ = 1;
  }

  // input can be INPUT and INPUT_GRAD
  // filter can be FILTER and FILTER_GRAD
  // output can be OUTPUT and OUTPUT_GRAD
  void checkShape(const TensorShape& input,
                  const TensorShape& filter,
                  const TensorShape& output) {
    // inputs and outputs arguments should be 4-dimensional.
    CHECK_EQ(input.ndims(), (size_t)4);
    CHECK_EQ(output.ndims(), (size_t)4);
    // The batchSize of the input needs to be equal to
    // the batchSize of the output.
    CHECK_EQ(input[0], output[0]);

    if (filter.ndims() == (size_t)4) {
      // If the filter's dimension is 4, groups convolution is not supported.
      CHECK_EQ(groups_, (size_t)1);
      // The input and output channel dimensions are the second and first
      // dimensions of the filter shape.
      CHECK_EQ(input[1], filter[1]);
      CHECK_EQ(output[1], filter[0]);
    } else {
      // filter argument should be 5-dimensional.
      CHECK_EQ(filter.ndims(), (size_t)5);
      // The first dimension of the filter is the size of the group
      CHECK_EQ(filter[0], groups_);
      // The input and output channel dimensions are the third and second
      // dimensions of the filter shape.
      CHECK_EQ(input[1], filter[2] * groups_);
      CHECK_EQ(output[1], filter[1] * groups_);
    }
  }

 protected:
  size_t getFilterHeight(const TensorShape& filter) const {
    return filter[filter.ndims() - 2];
  }

  size_t getFilterWidth(const TensorShape& filter) const {
    return filter[filter.ndims() - 1];
  }

  // determine whether im2col needs to be performed
  inline bool isNeedIm2col(const TensorShape& filter) const {
    return !(getFilterHeight(filter) == 1 && getFilterWidth(filter) == 1 &&
             strideH() == 1 && strideW() == 1 && paddingH() == 0 &&
             paddingW() == 0);
  }

  std::vector<size_t> strides_;
  std::vector<size_t> paddings_;
  std::vector<size_t> dilations_;

  /// Group size, refer to grouped convolution in
  /// Alex Krizhevsky's paper: when group=2, the first half of the
  /// filters are only connected to the first half of the input channels,
  /// and the second half only connected to the second half.
  size_t groups_;

  inline int strideH() const { return strides_[0]; }

  inline int strideW() const { return strides_[1]; }

  inline int paddingH() const { return paddings_[0]; }

  inline int paddingW() const { return paddings_[1]; }

  inline int dilationH() const { return dilations_[0]; }

  inline int dilationW() const { return dilations_[1]; }

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
