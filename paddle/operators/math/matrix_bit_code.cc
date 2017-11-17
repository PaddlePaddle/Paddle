/* Copyright (c) 2017 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "matrix_bit_code.h"

namespace paddle {
namespace operators {
namespace math {

/**
 * CodeTable class should support 3 functions:
 *
 * size_t size()
 *   return the number of codes
 *
 * int getMaxCodeLength()
 *   return the maximal code length
 *
 * Code operator()(size_t i)
 *   return the i-th code. Code class is descriebed below.
 *
 * Code class should support 3 functions:
 *
 * int getLength()
 *   return the length of the code
 *
 * bool calcIndex(int bit)
 *   bit ranges from 0 to getLength() - 1
 *   return the index for the (1+bit) level parent
 *
 * bool calcBit(int bit)
 *   return true if the bit level parent is the right child of (1+bit) level
 *   parent
 *
 */

/*
   for i:
     for j < codeLength:
       op(a(i, j), b(0, index(i, j)))
*/
template <class CodeTable, class Op, typename T>
static void AddByBitCodeT(Op op, CodeTable code_table,
                          const framework::Tensor& codes, framework::Tensor& a,
                          framework::Tensor& b) {
  size_t num_classes = code_table.size();
  size_t max_code_length = code_table.get_max_code_length();
  size_t num_sample = a.dims()[0].size();
  size_t width = a.dims()[1].size();

  for (size_t i = 0; i < num_sample; ++i) {
    auto code = code_table(codes.data<T>()[i]) int code_length =
        code.get_length();
    for (int j = 0; j < code_length; + j) {
      size_t index = code.calc_index(j);
      op(a<T>.data()[i * width + j], b<T>.data()[index]);
    }
  }
}

/* For j < codeLength:
   a(i, j) += b(0, index(i, j))
*/
template <typename T>
void AddByBitCode(size_t num_classes, const framework::Tensor& codes,
                  framework::Tensor& a, const framework::Tensor& b) {
  auto op = [](T& t, T& v) { t += v; };
  AddByBitCodeT<T>(op, SimpleCodeTable(num_classes), codes, a, b);
}

}  // namespace math
}  // namespace operators
}  // namespace paddle
