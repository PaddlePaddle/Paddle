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

namespace paddle {
namespace operators {
namespace math {

template <typename T>
struct MatrixBitCodeFunctorAdd {
  const phi::DenseTensor &vec_;
  phi::DenseTensor *tmat_;

  MatrixBitCodeFunctorAdd(const phi::DenseTensor &vec, phi::DenseTensor *tmat)
      : vec_(vec), tmat_(tmat) {}

  template <typename CodeTable>
  void operator()(const CodeTable &code_table) {
    size_t batch_size = tmat_->dims()[0];
    size_t width = tmat_->dims()[1];
    auto *tmat_data = tmat_->data<T>();
    auto *vec_data = vec_.data<T>();
    for (size_t i = 0; i < batch_size; ++i) {
      auto code = code_table.get_code(i);
      int code_length = code.get_length();
      for (int j = 0; j < code_length; ++j) {
        size_t index = code.calc_index(j);
        tmat_data[i * width + j] += vec_data[index];
      }
    }
  }
};

template <typename T>
void MatrixBitCodeFunctor<T>::Add(const phi::DenseTensor &vec,
                                  phi::DenseTensor *tmat) {
  MatrixBitCodeFunctorAdd<T> func(vec, tmat);
  paddle::visit(func, code_table_);
}

template <typename T>
struct MatrixBitCodeFunctorAddGrad {
  const phi::DenseTensor &tmat_;
  phi::DenseTensor *vec_;
  MatrixBitCodeFunctorAddGrad(const phi::DenseTensor &tmat,
                              phi::DenseTensor *vec)
      : tmat_(tmat), vec_(vec) {}

  template <typename CodeTable>
  void operator()(const CodeTable &table) {
    size_t batch_size = tmat_.dims()[0];
    size_t width = tmat_.dims()[1];
    auto *vec_data = vec_->data<T>();
    auto *tmat_data = tmat_.data<T>();
    for (size_t i = 0; i < batch_size; ++i) {
      auto code = table.get_code(i);
      int code_length = code.get_length();
      for (int j = 0; j < code_length; ++j) {
        size_t index = code.calc_index(j);
        vec_data[index] += tmat_data[i * width + j];
      }
    }
  }
};

template <typename T>
void MatrixBitCodeFunctor<T>::AddGrad(const phi::DenseTensor &tmat,
                                      phi::DenseTensor *vec) {
  MatrixBitCodeFunctorAddGrad<T> func(tmat, vec);
  paddle::visit(func, code_table_);
}

template <typename T>
struct MatrixBitCodeFunctorSum {
  const phi::DenseTensor &tmat_;
  phi::DenseTensor *sum_;
  T scale_sum_;

  MatrixBitCodeFunctorSum(const phi::DenseTensor &tmat,
                          phi::DenseTensor *sum,
                          T scale_sum)
      : tmat_(tmat), sum_(sum), scale_sum_(scale_sum) {}

  template <typename CodeTable>
  void operator()(const CodeTable &code_table) {
    size_t num_samples = tmat_.dims()[0];
    size_t o_width = tmat_.dims()[1];
    auto *tmat_data = tmat_.data<T>();
    auto *sum_data = sum_->data<T>();
    for (size_t i = 0; i < num_samples; ++i) {
      T sm = static_cast<T>(0.0);
      auto code = code_table.get_code(i);
      int code_length = code.get_length();
      for (int j = 0; j < code_length; ++j) {
        if (code.calc_bit(j)) {
          // calc_bit starts from right most bit, while data in tmat[i] is in
          // the
          // reverse order.
          sm += tmat_data[i * o_width + j];
        }
      }
      sum_data[i] = scale_sum_ * sm;
    }
  }
};

template <typename T>
void MatrixBitCodeFunctor<T>::Sum(const phi::DenseTensor &tmat,
                                  phi::DenseTensor *sum,
                                  T scale_sum) {
  MatrixBitCodeFunctorSum<T> func(tmat, sum, scale_sum);
  paddle::visit(func, code_table_);
}

template <typename T>
struct MatrixBitCodeFunctorMul {
  phi::DenseTensor *tmat_;
  const phi::DenseTensor &weight_;
  const phi::DenseTensor &input_;

  MatrixBitCodeFunctorMul(phi::DenseTensor *tmat,
                          const phi::DenseTensor &weight,
                          const phi::DenseTensor &input)
      : tmat_(tmat), weight_(weight), input_(input) {}

  template <typename CodeTable>
  void operator()(const CodeTable &code_table) {
    auto blas = phi::funcs::GetBlas<phi::CPUContext, T>(phi::CPUContext());
    size_t num_samples = tmat_->dims()[0];
    size_t tmat_width = tmat_->dims()[1];
    size_t input_width = input_.dims()[1];
    size_t weight_width = weight_.dims()[1];
    auto tmat_value = tmat_->data<T>();
    auto weight_value = weight_.data<T>();
    auto input_value = input_.data<T>();
    for (size_t i = 0; i < num_samples; ++i) {
      auto code = code_table.get_code(i);
      int code_length = code.get_length();
      const T *input_row = input_value + input_width * i;
      for (int j = 0; j < code_length; ++j) {
        size_t index = code.calc_index(j);
        const T *weight_row = weight_value + weight_width * index;
        T sum = blas.DOT(input_width, weight_row, input_row);
        tmat_value[i * tmat_width + j] += sum;
      }
    }
  }
};

template <typename T>
void MatrixBitCodeFunctor<T>::Mul(phi::DenseTensor *tmat,
                                  const phi::DenseTensor &weight,
                                  const phi::DenseTensor &input) {
  MatrixBitCodeFunctorMul<T> func(tmat, weight, input);
  paddle::visit(func, code_table_);
}

template <typename T, size_t N>
class ReservedVector : public std::vector<T> {
 public:
  ReservedVector() { this->reserve(N); }
};

template <typename T>
struct MatrixBitCodeFunctorMulGradWeight {
  const phi::DenseTensor &tmat_;
  phi::DenseTensor *weight_;
  const phi::DenseTensor &input_;
  MatrixBitCodeFunctorMulGradWeight(const phi::DenseTensor &tmat,
                                    phi::DenseTensor *weight,
                                    const phi::DenseTensor &input)
      : tmat_(tmat), weight_(weight), input_(input) {}
  template <typename CodeTable>
  void operator()(const CodeTable &code_table) {
    auto blas = phi::funcs::GetBlas<phi::CPUContext, T>(phi::CPUContext());
    size_t num_samples = tmat_.dims()[0];
    size_t input_width = input_.dims()[1];
    size_t tmat_width = tmat_.dims()[1];
    size_t weight_width = weight_->dims()[1];
    auto tmat_value = tmat_.data<T>();
    auto weight_value = weight_->data<T>();
    auto input_value = input_.data<T>();

    std::map<int, ReservedVector<std::pair<T, const T *>, 8u>> ops;
    for (size_t i = 0; i < num_samples; ++i) {
      auto code = code_table.get_code(i);
      int code_length = code.get_length();
      const T *input_value_row = input_value + input_width * i;
      const T *tmat_row = tmat_value + i * tmat_width;
      for (int j = 0; j < code_length; ++j) {
        ops[code.calc_index(j)].emplace_back(tmat_row[j], input_value_row);
      }
    }
    for (auto &op : ops) {
      auto &op_in_row = op.second;
      for (auto &pair : op_in_row) {
        auto &scale = pair.first;
        auto *input_row = pair.second;
        T *weight_row = weight_value + op.first * weight_width;
        blas.AXPY(input_width, scale, input_row, weight_row);
      }
    }
  }
};

template <typename T>
void MatrixBitCodeFunctor<T>::MulGradWeight(const phi::DenseTensor &tmat,
                                            phi::DenseTensor *weight,
                                            const phi::DenseTensor &input) {
  MatrixBitCodeFunctorMulGradWeight<T> func(tmat, weight, input);
  paddle::visit(func, code_table_);
}

template <typename T>
struct MatrixBitCodeFunctorMulGradWeightSR {
  const phi::DenseTensor &tmat_;
  phi::SelectedRows *weight_;
  const phi::DenseTensor &input_;

  MatrixBitCodeFunctorMulGradWeightSR(const phi::DenseTensor &tmat,
                                      phi::SelectedRows *weight,
                                      const phi::DenseTensor &input)
      : tmat_(tmat), weight_(weight), input_(input) {}

  template <typename CodeTable>
  void operator()(const CodeTable &code_table) {
    auto blas = phi::funcs::GetBlas<phi::CPUContext, T>(phi::CPUContext());
    size_t num_samples = tmat_.dims()[0];
    size_t input_width = input_.dims()[1];
    size_t tmat_width = tmat_.dims()[1];
    size_t weight_width = weight_->value().dims()[1];
    auto tmat_value = tmat_.data<T>();
    auto weight_value = weight_->mutable_value()->data<T>();
    auto input_value = input_.data<T>();

    std::unordered_map<int, std::vector<std::pair<T, const T *>>> ops;
    ops.reserve(weight_->rows().size());

    for (size_t i = 0; i < num_samples; ++i) {
      auto code = code_table.get_code(i);
      int code_length = code.get_length();
      const T *input_value_row = input_value + input_width * i;
      const T *tmat_row = tmat_value + i * tmat_width;
      for (int j = 0; j < code_length; ++j) {
        ops[code.calc_index(j)].emplace_back(tmat_row[j], input_value_row);
      }
    }

    for (auto &row : weight_->rows()) {
      auto &op_in_row = ops[row];
      for (auto &pair : op_in_row) {
        auto &scale = pair.first;
        auto *input_row = pair.second;
        blas.AXPY(input_width, scale, input_row, weight_value);
      }
      weight_value += weight_width;
    }
  }
};

template <typename T>
void MatrixBitCodeFunctor<T>::MulGradWeight(const phi::DenseTensor &tmat,
                                            phi::SelectedRows *weight,
                                            const phi::DenseTensor &input) {
  MatrixBitCodeFunctorMulGradWeightSR<T> func(tmat, weight, input);
  paddle::visit(func, code_table_);
}

template <typename T>
struct MatrixBitCodeFunctorMulGradError {
  const phi::DenseTensor &tmat_;
  const phi::DenseTensor &weight_;
  phi::DenseTensor *input_;

  MatrixBitCodeFunctorMulGradError(const phi::DenseTensor &tmat,
                                   const phi::DenseTensor &weight,
                                   phi::DenseTensor *input)
      : tmat_(tmat), weight_(weight), input_(input) {}
  template <typename CodeTable>
  void operator()(const CodeTable &code_table) {
    size_t num_samples = tmat_.dims()[0];
    size_t tmat_width = tmat_.dims()[1];
    size_t input_width = input_->dims()[1];
    size_t weight_width = weight_.dims()[1];
    auto tmat_value = tmat_.data<T>();
    auto weight_value = weight_.data<T>();
    auto input_value = input_->data<T>();

    for (size_t i = 0; i < num_samples; ++i) {
      auto code = code_table.get_code(i);
      int code_length = code.get_length();
      for (int j = 0; j < code_length; ++j) {
        size_t index = code.calc_index(j);

        for (size_t k = 0; k < input_width; ++k) {
          input_value[input_width * i + k] +=
              tmat_value[i * tmat_width + j] *
              weight_value[weight_width * index + k];
        }
      }
    }
  }
};

template <typename T>
void MatrixBitCodeFunctor<T>::MulGradError(const phi::DenseTensor &tmat,
                                           const phi::DenseTensor &weight,
                                           phi::DenseTensor *input) {
  MatrixBitCodeFunctorMulGradError<T> func(tmat, weight, input);
  paddle::visit(func, code_table_);
}

template <typename T>
struct MatrixBitCodeFunctorSub {
  phi::DenseTensor *tmat_;

  explicit MatrixBitCodeFunctorSub(phi::DenseTensor *tmat) : tmat_(tmat) {}

  template <typename CodeTable>
  void operator()(const CodeTable &code_table) {
    size_t num_samples = tmat_->dims()[0];
    size_t o_width = tmat_->dims()[1];
    auto *tmat_data = tmat_->data<T>();
    for (size_t i = 0; i < num_samples; ++i) {
      auto code = code_table.get_code(i);
      int code_length = code.get_length();
      for (int j = 0; j < code_length; ++j) {
        if (code.calc_bit(j)) {
          tmat_data[i * o_width + j] -= 1;
        }
      }
    }
  }
};

template <typename T>
void MatrixBitCodeFunctor<T>::Sub(phi::DenseTensor *tmat) {
  MatrixBitCodeFunctorSub<T> func(tmat);
  paddle::visit(func, code_table_);
}

template class MatrixBitCodeFunctor<float>;
template class MatrixBitCodeFunctor<double>;

}  // namespace math
}  // namespace operators
}  // namespace paddle
