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
template <typename T, class CodeTable, class Op>
static void AddByBitCodeT(Op op, CodeTable code_table, const int64_t* codes,
                          const framework::Tensor& tmat,
                          const framework::Tensor& vec) {
  size_t num_sample = tmat.dims()[0];
  size_t width = vec.dims()[1];

  for (size_t i = 0; i < num_sample; ++i) {
    auto code = code_table(static_cast<size_t>(codes[i]));
    int code_length = code.get_length();
    for (int j = 0; j < code_length; ++j) {
      size_t index = code.calc_index(j);
      auto t = tmat.data<T>()[i * width + j];
      auto v = vec.data<T>()[index];
      op(t, v);
    }
  }
}

template <typename T, class CodeTable>
void SubByBitCodeT(CodeTable code_table, const int64_t* codes,
                   framework::Tensor& tmat) {
  // size_t max_code_length = code_table.get_max_code_length();
  size_t num_samples = tmat.dims()[0];
  size_t o_width = tmat.dims()[1];
  for (size_t i = 0; i < num_samples; ++i) {
    auto code = code_table(static_cast<size_t>(codes[i]));
    int code_length = code.get_length();
    for (int j = 0; j < code_length; ++j) {
      if (code.calc_bit(j)) {
        tmat.data<T>()[i * o_width + j] -= 1;
      }
    }
  }
}

template <typename T, class CodeTable>
void SumByBitCodeT(CodeTable code_table, const int64_t* codes,
                   framework::Tensor& tmat, framework::Tensor& sum,
                   const T& scale_sum) {
  // size_t max_code_length = code_table.get_max_code_length();
  size_t num_samples = tmat.dims()[0];
  size_t o_width = tmat.dims()[1];
  for (size_t i = 0; i < num_samples; ++i) {
    T sm = static_cast<T>(0.0);
    auto code = code_table(static_cast<size_t>(codes[i]));
    int code_length = code.get_length();
    for (int j = 0; j < code_length; ++j) {
      if (code.calc_bit(j)) {
        sm += tmat.data<T>()[i * o_width + j];
      }
    }
    sum.data<T>()[i] = scale_sum * sm;
  }
}

template <typename T>
void MatrixBitCodeFunctor<T>::Add(size_t num_classes, const int64_t* codes,
                                  framework::Tensor& tmat,
                                  const framework::Tensor& vec) {
  auto op = [](T& t, const T& v) { t += v; };
  AddByBitCodeT<T>(op, SimpleCodeTable(num_classes), codes, tmat, vec);
}

template <typename T>
void MatrixBitCodeFunctor<T>::AddGrad(size_t num_classes, const int64_t* codes,
                                      framework::Tensor& tmat,
                                      framework::Tensor& vec) {
  auto op = [](T& t, T& v) { v += t; };
  AddByBitCodeT<T>(op, SimpleCodeTable(num_classes), codes, tmat, vec);
}

template <typename T>
void MatrixBitCodeFunctor<T>::Sum(size_t num_classes, const int64_t* codes,
                                  framework::Tensor& tmat,
                                  framework::Tensor& sum, T scale_sum) {
  SumByBitCodeT<T>(SimpleCodeTable(num_classes), codes, tmat, sum, scale_sum);
}

template <typename T>
void MatrixBitCodeFunctor<T>::Mul(size_t num_classes, const int64_t* codes,
                                  framework::Tensor& tmat,
                                  const framework::Tensor& weight,
                                  const framework::Tensor& input) {
  size_t num_samples = tmat.dims()[0];
  size_t tmat_width = tmat.dims()[1];
  size_t input_width = input.dims()[1];
  size_t weight_width = weight.dims()[1];
  auto tmat_p = tmat.data<T>();
  auto weight_p = weight.data<T>();
  auto input_p = input.data<T>();
  auto code_table = SimpleCodeTable(num_classes);
  for (size_t i = 0; i < num_samples; ++i) {
    auto code = code_table(static_cast<size_t>(codes[i]));
    int code_length = code.get_length();
    for (int j = 0; j < code_length; ++j) {
      size_t index = code.calc_index(j);

      T sum = static_cast<T>(0.0);
      for (size_t k = 0; k < input_width; ++k) {
        sum +=
            weight_p[weight_width * index + k] * input_p[input_width * i + k];
      }
      std::cout << sum << std::endl;
      tmat_p[i * tmat_width + j] += sum;
    }
  }
}

template <typename T>
void MatrixBitCodeFunctor<T>::MulGradWeight(size_t num_classes,
                                            const int64_t* codes,
                                            const framework::Tensor& tmat,
                                            framework::Tensor& weight,
                                            const framework::Tensor& input) {
  size_t num_samples = tmat.dims()[0];
  size_t input_width = input.dims()[1];
  size_t weight_width = weight.dims()[1];
  auto tmat_p = tmat.data<T>();
  auto weight_p = weight.data<T>();
  auto input_p = input.data<T>();
  auto code_table = SimpleCodeTable(num_classes);
  for (size_t i = 0; i < num_samples; ++i) {
    auto code = code_table(static_cast<size_t>(codes[i]));
    int code_length = code.get_length();
    for (int j = 0; j < code_length; ++j) {
      size_t index = code.calc_index(j);

      for (size_t k = 0; k < input_width; ++k) {
        weight_p[weight_width * index * k] +=
            tmat_p[i * weight_width * j] * input_p[input_width * i + k];
      }
    }
  }
}

template <typename T>
void MatrixBitCodeFunctor<T>::MulGradError(size_t num_classes,
                                           const int64_t* codes,
                                           const framework::Tensor& tmat,
                                           const framework::Tensor& weight,
                                           framework::Tensor& input) {
  size_t num_samples = tmat.dims()[0];
  size_t input_width = input.dims()[1];
  size_t weight_width = weight.dims()[1];
  auto tmat_p = tmat.data<T>();
  auto weight_p = weight.data<T>();
  auto input_p = input.data<T>();
  auto code_table = SimpleCodeTable(num_classes);

  for (size_t i = 0; i < num_samples; ++i) {
    auto code = code_table(static_cast<size_t>(codes[i]));
    int code_length = code.get_length();
    for (int j = 0; j < code_length; ++j) {
      size_t index = code.calc_index(j);

      for (size_t k = 0; k < input_width; ++k) {
        input_p[weight_width * index * k] +=
            tmat_p[i * weight_width * j] * weight_p[weight_width * i + k];
      }
    }
  }
}

template <typename T>
void MatrixBitCodeFunctor<T>::Sub(size_t num_classes, const int64_t* codes,
                                  framework::Tensor& tmat) {
  SubByBitCodeT<T>(SimpleCodeTable(num_classes), codes, tmat);
}

template class MatrixBitCodeFunctor<float>;
template class MatrixBitCodeFunctor<double>;

}  // namespace math
}  // namespace operators
}  // namespace paddle
