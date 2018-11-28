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

#include "paddle/fluid/operators/math/matrix_bit_code.h"
#include <iostream>
namespace paddle {
namespace operators {
namespace math {

template <typename T>
void MatrixBitCodeFunctor<T>::Add(const framework::Tensor& vec,
                                  framework::Tensor* tmat) {
  size_t batch_size = tmat->dims()[0];
  size_t width = tmat->dims()[1];
  for (size_t i = 0; i < batch_size; ++i) {
    auto code = code_table_->get_code(i);
    int code_length = code->get_length();
    for (int j = 0; j < code_length; ++j) {
      size_t index = code->calc_index(j);
      tmat->data<T>()[i * width + j] += vec.data<T>()[index];
    }
  }
}

template <typename T>
void MatrixBitCodeFunctor<T>::AddGrad(const framework::Tensor& tmat,
                                      framework::Tensor* vec) {
  size_t batch_size = tmat.dims()[0];
  size_t width = tmat.dims()[1];
  for (size_t i = 0; i < batch_size; ++i) {
    auto code = code_table_->get_code(i);
    int code_length = code->get_length();
    for (int j = 0; j < code_length; ++j) {
      size_t index = code->calc_index(j);
      vec->data<T>()[index] += tmat.data<T>()[i * width + j];
    }
  }
}

template <typename T>
void MatrixBitCodeFunctor<T>::AddGrad(const framework::Tensor& tmat,
                                      framework::SelectedRows* vec) {
  size_t batch_size = tmat.dims()[0];
  size_t width = tmat.dims()[1];
  for (size_t i = 0; i < batch_size; ++i) {
    auto code = code_table_->get_code(i);
    int code_length = code->get_length();
    for (int j = 0; j < code_length; ++j) {
      size_t index = code->calc_index(j);
      int64_t row_index = vec->GetIndexFromId(static_cast<int64_t>(index));
      vec->mutable_value()->data<T>()[row_index] +=
          tmat.data<T>()[i * width + j];
    }
  }
}

template <typename T>
void MatrixBitCodeFunctor<T>::Sum(const framework::Tensor& tmat,
                                  framework::Tensor* sum, T scale_sum) {
  size_t num_samples = tmat.dims()[0];
  size_t o_width = tmat.dims()[1];
  for (size_t i = 0; i < num_samples; ++i) {
    T sm = static_cast<T>(0.0);
    auto code = code_table_->get_code(i);
    int code_length = code->get_length();
    for (int j = 0; j < code_length; ++j) {
      if (code->calc_bit(j)) {
        // calc_bit starts from right most bit, while data in tmat[i] is in the
        // reverse order.
        sm += tmat.data<T>()[i * o_width + j];
      }
    }
    sum->data<T>()[i] = scale_sum * sm;
  }
}

template <typename T>
void MatrixBitCodeFunctor<T>::Mul(framework::Tensor* tmat,
                                  const framework::Tensor& weight,
                                  const framework::Tensor& input) {
  size_t num_samples = tmat->dims()[0];
  size_t tmat_width = tmat->dims()[1];
  size_t input_width = input.dims()[1];
  size_t weight_width = weight.dims()[1];
  auto tmat_value = tmat->data<T>();
  auto weight_value = weight.data<T>();
  auto input_value = input.data<T>();
  for (size_t i = 0; i < num_samples; ++i) {
    auto code = code_table_->get_code(i);
    int code_length = code->get_length();
    for (int j = 0; j < code_length; ++j) {
      size_t index = code->calc_index(j);
      T sum = static_cast<T>(0.0);
      for (size_t k = 0; k < input_width; ++k) {
        sum += weight_value[weight_width * index + k] *
               input_value[input_width * i + k];
      }
      tmat_value[i * tmat_width + j] += sum;
    }
  }
}

template <typename T>
void MatrixBitCodeFunctor<T>::MulGradWeight(const framework::Tensor& tmat,
                                            framework::Tensor* weight,
                                            const framework::Tensor& input) {
  size_t num_samples = tmat.dims()[0];
  size_t input_width = input.dims()[1];
  size_t tmat_width = tmat.dims()[1];
  size_t weight_width = weight->dims()[1];
  auto tmat_value = tmat.data<T>();
  auto weight_value = weight->data<T>();
  auto input_value = input.data<T>();
  for (size_t i = 0; i < num_samples; ++i) {
    auto code = code_table_->get_code(i);
    int code_length = code->get_length();
    for (int j = 0; j < code_length; ++j) {
      size_t index = code->calc_index(j);

      for (size_t k = 0; k < input_width; ++k) {
        weight_value[weight_width * index + k] +=
            tmat_value[i * tmat_width + j] * input_value[input_width * i + k];
      }
    }
  }
}

template <typename T>
void MatrixBitCodeFunctor<T>::MulGradWeight(const framework::Tensor& tmat,
                                            framework::SelectedRows* weight,
                                            const framework::Tensor& input) {
  size_t num_samples = tmat.dims()[0];
  size_t input_width = input.dims()[1];
  size_t tmat_width = tmat.dims()[1];
  size_t weight_width = weight->value().dims()[1];
  auto tmat_value = tmat.data<T>();
  auto weight_value = weight->mutable_value()->data<T>();
  auto input_value = input.data<T>();
  for (size_t i = 0; i < num_samples; ++i) {
    auto code = code_table_->get_code(i);
    int code_length = code->get_length();
    for (int j = 0; j < code_length; ++j) {
      size_t index = code->calc_index(j);
      for (size_t k = 0; k < input_width; ++k) {
        int64_t row_index = weight->GetIndexFromId(static_cast<int64_t>(index));
        weight_value[row_index * weight_width + k] +=
            tmat_value[i * tmat_width + j] * input_value[input_width * i + k];
      }
    }
  }
}

template <typename T>
void MatrixBitCodeFunctor<T>::MulGradError(const framework::Tensor& tmat,
                                           const framework::Tensor& weight,
                                           framework::Tensor* input) {
  size_t num_samples = tmat.dims()[0];
  size_t tmat_width = tmat.dims()[1];
  size_t input_width = input->dims()[1];
  size_t weight_width = weight.dims()[1];
  auto tmat_value = tmat.data<T>();
  auto weight_value = weight.data<T>();
  auto input_value = input->data<T>();

  for (size_t i = 0; i < num_samples; ++i) {
    auto code = code_table_->get_code(i);
    int code_length = code->get_length();
    for (int j = 0; j < code_length; ++j) {
      size_t index = code->calc_index(j);

      for (size_t k = 0; k < input_width; ++k) {
        input_value[input_width * i + k] +=
            tmat_value[i * tmat_width + j] *
            weight_value[weight_width * index + k];
      }
    }
  }
}

template <typename T>
void MatrixBitCodeFunctor<T>::Sub(framework::Tensor* tmat) {
  size_t num_samples = tmat->dims()[0];
  size_t o_width = tmat->dims()[1];
  for (size_t i = 0; i < num_samples; ++i) {
    auto code = code_table_->get_code(i);
    int code_length = code->get_length();
    for (int j = 0; j < code_length; ++j) {
      if (code->calc_bit(j)) {
        tmat->data<T>()[i * o_width + j] -= 1;
      }
    }
  }
}

template class MatrixBitCodeFunctor<float>;
template class MatrixBitCodeFunctor<double>;

}  // namespace math
}  // namespace operators
}  // namespace paddle
