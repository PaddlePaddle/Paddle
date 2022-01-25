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
struct MatrixBitCodeFunctorAdd : public boost::static_visitor<void> {
  const framework::Tensor &vec_;
  framework::Tensor *tmat_;

  MatrixBitCodeFunctorAdd(const framework::Tensor &vec, framework::Tensor *tmat)
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
void MatrixBitCodeFunctor<T>::Add(const framework::Tensor &vec,
                                  framework::Tensor *tmat) {
  MatrixBitCodeFunctorAdd<T> func(vec, tmat);
  code_table_.apply_visitor(func);
}

template <typename T>
struct MatrixBitCodeFunctorAddGrad : public boost::static_visitor<void> {
  const framework::Tensor &tmat_;
  framework::Tensor *vec_;
  MatrixBitCodeFunctorAddGrad(const framework::Tensor &tmat,
                              framework::Tensor *vec)
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
void MatrixBitCodeFunctor<T>::AddGrad(const framework::Tensor &tmat,
                                      framework::Tensor *vec) {
  MatrixBitCodeFunctorAddGrad<T> func(tmat, vec);
  code_table_.apply_visitor(func);
}

template <typename T>
struct MatrixBitCodeFunctorSum : public boost::static_visitor<void> {
  const framework::Tensor &tmat_;
  framework::Tensor *sum_;
  T scale_sum_;

  MatrixBitCodeFunctorSum(const framework::Tensor &tmat, framework::Tensor *sum,
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
void MatrixBitCodeFunctor<T>::Sum(const framework::Tensor &tmat,
                                  framework::Tensor *sum, T scale_sum) {
  MatrixBitCodeFunctorSum<T> func(tmat, sum, scale_sum);
  code_table_.apply_visitor(func);
}

template <typename T>
struct MatrixBitCodeFunctorMul : public boost::static_visitor<void> {
  framework::Tensor *tmat_;
  const framework::Tensor &weight_;
  const framework::Tensor &input_;

  MatrixBitCodeFunctorMul(framework::Tensor *tmat,
                          const framework::Tensor &weight,
                          const framework::Tensor &input)
      : tmat_(tmat), weight_(weight), input_(input) {}

  template <typename CodeTable>
  void operator()(const CodeTable &code_table) {
    auto blas =
        GetBlas<platform::CPUDeviceContext, T>(platform::CPUDeviceContext());
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
void MatrixBitCodeFunctor<T>::Mul(framework::Tensor *tmat,
                                  const framework::Tensor &weight,
                                  const framework::Tensor &input) {
  MatrixBitCodeFunctorMul<T> func(tmat, weight, input);
  code_table_.apply_visitor(func);
}

template <typename T, size_t N>
class ReservedVector : public std::vector<T> {
 public:
  ReservedVector() { this->reserve(N); }
};

template <typename T>
struct MatrixBitCodeFunctorMulGradWeight : public boost::static_visitor<void> {
  const framework::Tensor &tmat_;
  framework::Tensor *weight_;
  const framework::Tensor &input_;
  MatrixBitCodeFunctorMulGradWeight(const framework::Tensor &tmat,
                                    framework::Tensor *weight,
                                    const framework::Tensor &input)
      : tmat_(tmat), weight_(weight), input_(input) {}
  template <typename CodeTable>
  void operator()(const CodeTable &code_table) {
    auto blas =
        GetBlas<platform::CPUDeviceContext, T>(platform::CPUDeviceContext());
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
void MatrixBitCodeFunctor<T>::MulGradWeight(const framework::Tensor &tmat,
                                            framework::Tensor *weight,
                                            const framework::Tensor &input) {
  MatrixBitCodeFunctorMulGradWeight<T> func(tmat, weight, input);
  code_table_.apply_visitor(func);
}

template <typename T>
struct MatrixBitCodeFunctorMulGradWeightSR
    : public boost::static_visitor<void> {
  const framework::Tensor &tmat_;
  pten::SelectedRows *weight_;
  const framework::Tensor &input_;

  MatrixBitCodeFunctorMulGradWeightSR(const framework::Tensor &tmat,
                                      pten::SelectedRows *weight,
                                      const framework::Tensor &input)
      : tmat_(tmat), weight_(weight), input_(input) {}

  template <typename CodeTable>
  void operator()(const CodeTable &code_table) {
    auto blas =
        GetBlas<platform::CPUDeviceContext, T>(platform::CPUDeviceContext());
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
void MatrixBitCodeFunctor<T>::MulGradWeight(const framework::Tensor &tmat,
                                            pten::SelectedRows *weight,
                                            const framework::Tensor &input) {
  MatrixBitCodeFunctorMulGradWeightSR<T> func(tmat, weight, input);
  code_table_.apply_visitor(func);
}

template <typename T>
struct MatrixBitCodeFunctorMulGradError : public boost::static_visitor<void> {
  const framework::Tensor &tmat_;
  const framework::Tensor &weight_;
  framework::Tensor *input_;

  MatrixBitCodeFunctorMulGradError(const framework::Tensor &tmat,
                                   const framework::Tensor &weight,
                                   framework::Tensor *input)
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
void MatrixBitCodeFunctor<T>::MulGradError(const framework::Tensor &tmat,
                                           const framework::Tensor &weight,
                                           framework::Tensor *input) {
  MatrixBitCodeFunctorMulGradError<T> func(tmat, weight, input);
  code_table_.apply_visitor(func);
}

template <typename T>
struct MatrixBitCodeFunctorSub : public boost::static_visitor<void> {
  framework::Tensor *tmat_;

  explicit MatrixBitCodeFunctorSub(framework::Tensor *tmat) : tmat_(tmat) {}

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
void MatrixBitCodeFunctor<T>::Sub(framework::Tensor *tmat) {
  MatrixBitCodeFunctorSub<T> func(tmat);
  code_table_.apply_visitor(func);
}

template class MatrixBitCodeFunctor<float>;
template class MatrixBitCodeFunctor<double>;

}  // namespace math
}  // namespace operators
}  // namespace paddle
