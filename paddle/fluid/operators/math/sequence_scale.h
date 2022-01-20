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

#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/platform/device_context.h"

namespace paddle {
namespace operators {
namespace math {

/*
 * \brief   Scale a sequence.
 *
 *  All sequences will be padded to the same length and stored in a transposed
 * shape.
 *  Example:
 *    Given:
 *      seq = (s0, s0, s0, s0; s1, s1; s2, s2, s2; s3)
 *      scales = (2, 3, 4, 5)
 *    then:
 *      result = (2*s0, 2*s0, 2*s0, 2*s0; 3*s1, 3*s1; 4*s2, 4*s2, 4*s2; 5*s3)

 *
 * \param context       Device context of this functor.
 * \param seq           LoDTensor which is stored in sequence format, the shape
 *                      is [total_sequence_length, sequence_width] where
 *                      total_sequence_length is the sum of all sequences'
 *                      length.
 * \param scales        Array<T>. The i-th sequence will be scaled by scales[i].

 * \param num_seq       Number of sequence
 *
 */
template <typename DeviceContext, typename T>
class ScaleLoDTensorFunctor {
 public:
  void operator()(const DeviceContext& context, const T* scales,
                  framework::LoDTensor* seq);
};

}  // namespace math
}  // namespace operators
}  // namespace paddle
