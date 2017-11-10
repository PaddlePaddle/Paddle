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

#include "paddle/framework/tensor.h"
#include "paddle/platform/device_context.h"

namespace paddle {
namespace operators {
namespace math {
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
template <typename Place, typename T>
class Vol2ColFunctor {
 public:
  void operator()(const platform::DeviceContext& context,
                  const framework::Tensor& vol, framework::Tensor& col,
                  int stride_depth, int stride_height, int stride_width,
                  int padding_depth, int padding_height,
                  int padding_width) const;
};

template <typename Place, typename T>
class Col2VolFunctor {
 public:
  void operator()(const platform::DeviceContext& context,
                  framework::Tensor& vol, const framework::Tensor& col,
                  int stride_depth, int stride_height, int stride_width,
                  int padding_depth, int padding_height,
                  int padding_width) const;
};

}  // namespace math
}  // namespace operators
}  // namespace paddle
