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

#include <vector>

#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/platform/device_context.h"

namespace paddle {
namespace operators {
namespace math {

using DataLayout = framework::DataLayout;

/*
 * \brief Converts the feature data of four dimensions(CDHW) into a colData of
 *        seven dimensions in the Vol2ColFunctor calculation,
 *        And in the Col2VolFunctor calculation, it is reversed.
 *
 * \param volData   Vol data.
 * \param volShape  The shape of volData,
 *                 [input_channels, input_depth, input_height, input_width].
 * \param colData  Column data.
 * \param colShape The shape of colData.
 *
 * \param dilations    dilation data.
 * \param 3-dimension  [dilation_depth, dilation_height, dilation_width].
 *
 * \param strides      stride data.
 * \param 3-dimension  [stride_depth, stride_height, stride_width].
 *
 * \param paddings     padding data.
 * \param 3-dimension  [d_pad, h_pad, w_pad].
 *
 * The shape of colData is:
 * [input_channels, filter_depth, filter_height, filter_width, output_depth,
 * output_height, output_width]
 * So, it is easy to reshape into a convolution matrix for convolution
 * calculation based on matrix multiplication.
 * The shape of convolution matrix is [height, width], where the height is equal
 * input_channels * filter_depth * filter_height * filter_width, and the width
 * is equal output_depth * output_height * output_width.
 *
 * Reshape:
 *     shape of colData           shape of convolution matrix
 *     [input_channels,
 *      filter_depth,
 *      filter_height,
 *      filter_width,      ======>      [height, width]
 *      output_depth,
 *      output_height,
 *      output_width]
 *
 * \note The caller needs to ensure that volShape.inputChannels is equal to
 *       colShape.inputChannels.
 */
template <typename DeviceContext, typename T>
class Vol2ColFunctor {
 public:
  void operator()(const DeviceContext& context,
                  const phi::DenseTensor& vol,
                  const std::vector<int>& dilations,
                  const std::vector<int>& strides,
                  const std::vector<int>& paddings,
                  phi::DenseTensor* col,
                  const DataLayout data_layout = DataLayout::kNCHW) const;
};

template <typename DeviceContext, typename T>
class Col2VolFunctor {
 public:
  void operator()(const DeviceContext& context,
                  const phi::DenseTensor& col,
                  const std::vector<int>& dilations,
                  const std::vector<int>& strides,
                  const std::vector<int>& paddings,
                  phi::DenseTensor* vol,
                  const DataLayout data_layout = DataLayout::kNCHW) const;
};

}  // namespace math
}  // namespace operators
}  // namespace paddle
