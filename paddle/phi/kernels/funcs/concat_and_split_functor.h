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

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/device_context.h"
#include "paddle/phi/core/utils/data_type.h"

// See Note [ Why still include the fluid headers? ]
#include "paddle/fluid/memory/memcpy.h"

namespace phi {
namespace funcs {

/*
 * \brief Concatenate the input tensors along the dimension axis.
 *  TODO(zcd): maybe it needs to be more detailed.
 *  Examples:
 *     Input[0] = [[1,2],[3,4]]
 *     Input[1] = [[5,6]]
 *     axis = 0
 *
 *     Output = [[1,2],
 *               [3,4],
 *               [5,6]]
 */
template <typename Context, typename T>
struct ConcatFunctor {
  void operator()(const Context& context,
                  const std::vector<phi::DenseTensor>& input,
                  int axis,
                  phi::DenseTensor* output);
};

/*
 * \brief Split the input tensors along the dimension axis into outputs.
 *  TODO(zcd): maybe it needs to be more detailed.
 *  Examples:
 *     Input = [[1,2],
 *              [3,4],
 *              [5,6]]
 *     axis = 0
 *
 *     Output[0] = [[1,2],[3,4]]
 *     Output[1] = [[5,6]]
 */
template <typename Context, typename T>
class SplitFunctor {
 public:
  void operator()(const Context& context,
                  const phi::DenseTensor& input,
                  const std::vector<const phi::DenseTensor*>& ref_inputs,
                  int axis,
                  std::vector<phi::DenseTensor*>* outputs);
};

}  // namespace funcs
}  // namespace phi

#define FOR_ALL_TYPES(macro)         \
  macro(int);                        \
  macro(float);                      \
  macro(double);                     \
  macro(bool);                       \
  macro(int64_t);                    \
  macro(int16_t);                    \
  macro(uint8_t);                    \
  macro(int8_t);                     \
  macro(phi::dtype::float16);        \
  macro(phi::dtype::bfloat16);       \
  macro(phi::dtype::complex<float>); \
  macro(phi::dtype::complex<double>);
