/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

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

#include <vector>

#include "paddle/common/errors.h"
#include "paddle/common/layout.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/enforce.h"

namespace phi {
namespace funcs {

using DataLayout = phi::DataLayout;

/* The storage format of the coldata in the Im2ColFunctor and Col2ImFunctor. */
enum class ColFormat { kCFO = 0, kOCF = 1 };

/*
 * \brief Converts the image data of three dimensions(CHW) into a colData of
 *        five dimensions in the Im2ColFunctor calculation,
 *        And in the Col2ImFunctor calculation, it is reversed.
 *
 * \param imData   Image data.
 * \param imShape  The shape of imData,
 *                 [input_channels, input_height, input_width].
 * \param colData  Column data.
 * \param colShape The shape of colData.
 *
 * \param dilations    dilation data.
 * \param 2-dimension  [dilation_height, dilation_width].
 *
 * \param strides      stride data.
 * \param 2-dimension  [stride_height, stride_width].
 *
 * \param paddings     padding data.
 * \param 4-dimension  [up_pad, left_pad, down_pad, right_pad].
 *
 * If the template argument Format is kCFO, the shape of colData is:
 * [input_channels, filter_height, filter_width, output_height, output_width]
 * So, it is easy to reshape into a convolution matrix for convolution
 * calculation based on matrix multiplication.
 * The shape of convolution matrix is [height, width], where the height is equal
 * input_channels * filter_height * filter_width, and the width is equal
 * output_height * output_width.
 *
 * Reshape:
 *     shape of colData           shape of convolution matrix
 *     [input_channels,
 *      filter_height,
 *      filter_width,      ======>      [height, width]
 *      output_height,
 *      output_width]
 *
 * If the template argument Format is kOCF, the shape of colData is:
 * [output_height, output_width, input_channels, filter_height, filter_width]
 * So, it is easy to reshape into a sequence matrix for rnn calculation.
 * The shape of sequence matrix is [seq_length, step_size], where the seq_length
 * is equal output_height * output_width, and the step_size is equal
 * input_channels * filter_height * filter_width.
 *
 * Reshape:
 *     shape of colData             shape of sequence matrix
 *     [output_height,
 *      output_width,
 *      input_channels,    ======>    [seqLength, stepSize]
 *      filter_height,
 *      filter_width]
 *
 * \note The caller needs to ensure that imShape.inputChannels is equal to
 *       colShape.inputChannels.
 */
template <ColFormat Format, typename DeviceContext, typename T>
class Im2ColFunctor {
 public:
  void operator()(const DeviceContext& context,
                  const phi::DenseTensor& im,
                  const std::vector<int>& dilation,
                  const std::vector<int>& stride,
                  const std::vector<int>& padding,
                  phi::DenseTensor* col,
                  const DataLayout data_layout = DataLayout::kNCHW);
};

template <ColFormat Format, typename DeviceContext, typename T>
class Col2ImFunctor {
 public:
  void operator()(const DeviceContext& context,
                  const phi::DenseTensor& col,
                  const std::vector<int>& dilation,
                  const std::vector<int>& stride,
                  const std::vector<int>& padding,
                  phi::DenseTensor* im,
                  const DataLayout data_layout = DataLayout::kNCHW);
};

}  // namespace funcs
}  // namespace phi
