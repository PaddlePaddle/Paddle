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
  auto blas =
      GetBlas<platform::CPUDeviceContext, T>(platform::CPUDeviceContext());
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
    const T* input_row = input_value + input_width * i;
    for (int j = 0; j < code_length; ++j) {
      size_t index = code->calc_index(j);
      const T* weight_row = weight_value + weight_width * index;
      T sum = static_cast<T>(0.0);
      sum = blas.DOT(input_width, weight_row, input_row);
      tmat_value[i * tmat_width + j] += sum;
    }
  }
}

template <typename T>
void MatrixBitCodeFunctor<T>::MulGradWeight(const framework::Tensor& tmat,
                                            framework::Tensor* weight,
                                            const framework::Tensor& input) {
  auto blas =
      GetBlas<platform::CPUDeviceContext, T>(platform::CPUDeviceContext());
  size_t num_samples = tmat.dims()[0];
  size_t input_width = input.dims()[1];
  size_t tmat_width = tmat.dims()[1];
  size_t weight_width = weight->dims()[1];
  auto tmat_value = tmat.data<T>();
  auto weight_value = weight->data<T>();
  auto input_value = input.data<T>();

  std::unordered_map<int, std::vector<std::pair<T, const T*>>> ops;

  for (size_t i = 0; i < num_samples; ++i) {
    auto code = code_table_->get_code(i);
    int code_length = code->get_length();
    const T* input_value_row = input_value + input_width * i;
    const T* tmat_row = tmat_value + i * tmat_width;
    for (int j = 0; j < code_length; ++j) {
      ops[code->calc_index(j)].emplace_back(tmat_row[j], input_value_row);
    }
  }
  for (auto& op : ops) {
    auto& op_in_row = op.second;
    for (auto& pair : op_in_row) {
      auto& scale = pair.first;
      auto* input_row = pair.second;
      T* weight_row = weight_value + op.first * weight_width;
      blas.AXPY(input_width, scale, input_row, weight_row);
    }
  }
}

template <typename T>
void MatrixBitCodeFunctor<T>::MulGradWeight(const framework::Tensor& tmat,
                                            framework::SelectedRows* weight,
                                            const framework::Tensor& input) {
  auto blas =
      GetBlas<platform::CPUDeviceContext, T>(platform::CPUDeviceContext());
  size_t num_samples = tmat.dims()[0];
  size_t input_width = input.dims()[1];
  size_t tmat_width = tmat.dims()[1];
  size_t weight_width = weight->value().dims()[1];
  auto tmat_value = tmat.data<T>();
  auto weight_value = weight->mutable_value()->data<T>();
  auto input_value = input.data<T>();

  std::unordered_map<int, std::vector<std::pair<T, const T*>>> ops;
  ops.reserve(weight->rows().size());

  for (size_t i = 0; i < num_samples; ++i) {
    auto code = code_table_->get_code(i);
    int code_length = code->get_length();
    const T* input_value_row = input_value + input_width * i;
    const T* tmat_row = tmat_value + i * tmat_width;
    for (int j = 0; j < code_length; ++j) {
      ops[code->calc_index(j)].emplace_back(tmat_row[j], input_value_row);
    }
  }

  for (auto& row : weight->rows()) {
    auto& op_in_row = ops[row];
    for (auto& pair : op_in_row) {
      auto& scale = pair.first;
      auto* input_row = pair.second;
      blas.AXPY(input_width, scale, input_row, weight_value);
    }
    weight_value += weight_width;
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
