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
                          const framework::Tensor& b) {
  size_t num_classes = code_table.size();
  size_t max_code_length = code_table.get_max_code_length();
  size_t num_sample = a.dims()[0];
  size_t width = a.dims()[1];

  for (size_t i = 0; i < num_sample; ++i) {
    auto code = code_table(codes.data<T>()[i]);
    int code_length = code.get_length();
    for (int j = 0; j < code_length; + j) {
      size_t index = code.calc_index(j);
      op(a.data<T>()[i * width + j], b.data<T>()[index]);
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

template <class CodeTable, typename T>
void SumByBitCodeT(CodeTable code_table, const framework::Tensor& codes,
                   framework::Tensor& tmat, framework::Tensor& sum,
                   const T& scale_sum) {
  size_t max_code_length = code_table.get_max_code_length();
  size_t num_samples = tmat.dims()[0];
  size_t o_width = tmat.dims()[1];
  for (size_t i = 0; i < num_samples; ++i) {
    T sm = 0;
    auto code = code_table(codes.data<T>()[i]);
    int code_length = code.get_length();
    for (int j = 0; j < code_length; ++j) {
      if (code.calc_bit(j)) {
        sm += tmat.data<T>()[i * o_width + j];
      }
    }
    sum.data<T>()[i] = scale_sum * sm;
  }
}
/* For j < codeLength:
    sum(i, 0) = \sum_j bit(i, j) * input(i, j)
*/
template <typename T>
void SumByBitCode(size_t num_classes, const framework::Tensor& codes,
                  framework::Tensor& tmat, framework::Tensor& sum,
                  T scale_sum) {
  SumByBitCodeT(SimpleCodeTable(num_classes), codes, tmat, scale_sum);
}

template <class Op, class CodeTable, typename T>
void MulByBitCodeT(Op op, CodeTable code_table, const framework::Tensor& codes,
                   framework::Tensor& tmat, framework::Tensor& weight,
                   framework::Tensor& input) {
  size_t num_classes = code_table.size();
  size_t max_code_length = code_table.get_max_code_length();
  size_t num_samples = tmat.dims()[0];
  size_t input_dim = input.dims()[1];
  size_t o_width = tmat.dims()[1];

  for (size_t i = 0; i < num_samples; ++i) {
    auto code = code_table(codes.data<T>()[i]);
    int code_length = code.get_length();
    for (int j = 0; j < code_length; ++j) {
      size_t index = code.calc_index(j);
      op(tmat.data<T>()[i * o_width + j],
         weight.data<T>() + index * weight.dims()[1],
         input.data<T>() + i * input.dims()[1], input_dim);
    }
  }
}

template <typename T>
void MulByBitCode(size_t num_classes, const framework::Tensor& codes,
                  framework::Tensor& tmat, const framework::Tensor& weight,
                  const framework::Tensor& input) {
  auto op = [](T& t, const T* weight_row, const T* input_row,
               size_t input_dim) {
    T sum = 0;
    for (size_t k = 0; k < input_dim; ++k) {
      sum += weight_row[k] * input_row[k];
    }
    t += sum;
  };
  MulByBitCode(op, SimpleCodeTable(num_classes), codes, tmat, weight, input);
}
}  // namespace math
}  // namespace operators
}  // namespace paddle
