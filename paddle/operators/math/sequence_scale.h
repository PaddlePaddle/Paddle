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

#include "paddle/framework/lod_tensor.h"
#include "paddle/platform/device_context.h"

namespace paddle {
namespace operators {
namespace math {

/*
 * \brief   Scale a sequence.
 *
 *  All sequences will be padded to the same length and stored in a transposed
 * shape.
 *  Example:
 *    seq     (s0, s0, s0, s0; s1, s1; s2, s2, s2; s3)
 *    padding (s0, s1, s2, s3; s0, s1, s2, 0; s0, 0, s2, 0; s0, 0, 0, 0)
 *
 * \param context       device context of this functor.
 * \param seq           LoDTensor which is stored in sequence format, the shape
 *                      is [total_sequence_length, sequence_width] where
 *                      total_sequence_length is the sum of all sequences'
 *                      length.
 * \param padding       Tensor which is padded to the same length, the shape is
 *                      [max_sequence_length, num_sequences, sequence_width].
 * \param norm_by_times whether dividing sequence's length.
 *
 * \note  transposition is also done in this functor.
 */
template <typename DeviceContext, typename T>
class ScaleLoDTensorFunctor {
 public:
  void operator()(const DeviceContext& context, framework::LoDTensor& seq,
                  const T* scales, const size_t num_seq);
};

}  // namespace math
}  // namespace operators
}  // namespace paddle
