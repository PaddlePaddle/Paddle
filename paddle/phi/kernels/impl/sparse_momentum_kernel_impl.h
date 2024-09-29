// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/compat/convert_utils.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/for_range.h"
#include "paddle/utils/optional.h"

#ifdef __NVCC__
#include "cub/cub.cuh"
#endif
#ifdef __HIPCC__
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#endif

namespace phi {

template <typename T>
using MultiPrecisionType = typename phi::dtype::MPTypeTrait<T>::Type;

enum class RegularizationType {
  kNONE = 0,
  kL1DECAY = 1,  // do not need support right now
  kL2DECAY = 2,
};

template <typename T>
struct NoNesterov {
  HOSTDEVICE inline T operator()(const T& grad,
                                 const T& velocity,
                                 const T& mu) const {
    return velocity;
  }
};

template <typename T>
struct UseNesterov {
  HOSTDEVICE inline T operator()(const T& grad,
                                 const T& velocity,
                                 const T& mu) const {
    return grad + velocity * mu;
  }
};

// The following code is from
// https://en.cppreference.com/w/cpp/algorithm/lower_bound
// https://en.cppreference.com/w/cpp/algorithm/upper_bound
template <typename T>
HOSTDEVICE inline void BinarySearchLowerUpperBound(const T* x,
                                                   int64_t num,
                                                   const T& value,
                                                   int64_t* lower_bound,
                                                   int64_t* upper_bound) {
  *lower_bound = -1;
  *upper_bound = -1;

  auto* first = x;
  int64_t count = static_cast<int64_t>(num);
  while (count > 0) {
    int64_t step = (count >> 1);
    auto* it = first + step;
    if (*it < value) {
      first = ++it;
      count -= (step + 1);
    } else {
      count = step;
    }
  }
  auto idx = static_cast<int64_t>(first - x);
  if ((idx > 0 && idx < num) || (idx == 0 && x[idx] == value)) {
    *lower_bound = idx;
  }

  if (*lower_bound >= 0) {
    first = x + idx;
    count = static_cast<int64_t>(num - idx);
    while (count > 0) {
      auto step = (count >> 1);
      auto* it = first + step;
      if (value < *it) {
        count = step;
      } else {
        first = ++it;
        count -= (step + 1);
      }
    }
    auto upper_idx = static_cast<int64_t>(first - x) - 1;
    if ((upper_idx >= 0 && upper_idx < num - 1) ||
        (upper_idx == num - 1 && x[upper_idx] == value)) {
      *upper_bound = upper_idx;
    }
  }
  return;
}

template <typename T>
class RangeFunctor {
 private:
  T* value_;

 public:
  explicit RangeFunctor(T* value) : value_(value) {}
  inline HOSTDEVICE void operator()(size_t i) { value_[i] = static_cast<T>(i); }
};

template <typename T, typename MT, typename IndexT, typename UpdateMethod>
class IndexMomentumFunctor {
 private:
  const T* param_;
  const T* grad_;
  const MT* velocity_;
  const MultiPrecisionType<MT>* lr_;
  const MT* master_param_;
  const MT mu_;
  const MT rescale_grad_;
  const IndexT* sorted_index_;
  const IndexT* grad_index_;
  const int64_t num_index_;
  const int axis_;
  const int64_t param_row_numel_;
  const int64_t grad_row_numel_;
  T* param_out_;
  MT* velocity_out_;
  MT* master_param_out_;
  const RegularizationType regularization_flag_;
  const MT regularization_coeff_;
  const UpdateMethod& update_method_;

 public:
  IndexMomentumFunctor(const T* param,
                       const T* grad,
                       const MT* velocity,
                       const MultiPrecisionType<MT>* lr,
                       const MT* master_param,
                       const MT mu,
                       const MT rescale_grad,
                       const IndexT* sorted_index,
                       const IndexT* grad_index,
                       int64_t num_index,
                       int axis,
                       int64_t param_row_numel,
                       int64_t grad_row_numel,
                       const RegularizationType regularization_flag,
                       const MT regularization_coeff,
                       const UpdateMethod& update_method,
                       T* param_out,
                       MT* velocity_out,
                       MT* master_param_out)
      : param_(param),
        grad_(grad),
        velocity_(velocity),
        lr_(lr),
        master_param_(master_param),
        mu_(mu),
        rescale_grad_(rescale_grad),
        sorted_index_(sorted_index),
        grad_index_(grad_index),
        num_index_(num_index),
        axis_(axis),
        param_row_numel_(param_row_numel),
        grad_row_numel_(grad_row_numel),
        param_out_(param_out),
        velocity_out_(velocity_out),
        master_param_out_(master_param_out),
        regularization_flag_(regularization_flag),
        regularization_coeff_(regularization_coeff),
        update_method_(update_method) {}

  inline HOSTDEVICE void operator()(size_t i) {
    MT grad = static_cast<MT>(0);
    size_t row = i / param_row_numel_;
    size_t col = i % param_row_numel_;
    if (axis_ == 0) {
      int64_t row_idx0, row_idx1;
      BinarySearchLowerUpperBound<IndexT>(
          sorted_index_, num_index_, row, &row_idx0, &row_idx1);
      if (row_idx0 >= 0 && row_idx1 >= 0) {
        for (int64_t row_idx = row_idx0; row_idx <= row_idx1; row_idx++) {
          size_t offset = grad_index_[row_idx] * param_row_numel_ + col;
          grad += static_cast<MT>(grad_[offset]) * rescale_grad_;
        }
      }
    } else if (axis_ == 1) {
      int64_t col_idx0, col_idx1;
      BinarySearchLowerUpperBound<IndexT>(
          sorted_index_, num_index_, col, &col_idx0, &col_idx1);
      if (col_idx0 >= 0 && col_idx1 >= 0) {
        for (int64_t col_idx = col_idx0; col_idx <= col_idx1; col_idx++) {
          size_t offset = row * grad_row_numel_ + grad_index_[col_idx];
          grad += static_cast<MT>(grad_[offset]) * rescale_grad_;
        }
      }
    }

    // put memory access in register
    const MT param =
        master_param_ ? master_param_[i] : static_cast<MT>(param_[i]);
    const MT lr = static_cast<MT>(lr_[0]);
    const MT velocity = velocity_[i];

    grad = regularization_flag_ == RegularizationType::kL2DECAY
               ? grad + regularization_coeff_ * param
               : grad;

    MT velocity_out = velocity * mu_ + grad;
    MT velocity_tmp = update_method_(grad, velocity_out, mu_);
    MT param_out = param - velocity_tmp * lr;
    // write register to memory
    velocity_out_[i] = velocity_out;
    param_out_[i] = static_cast<T>(param_out);
    if (master_param_out_) {
      master_param_out_[i] = param_out;
    }
  }
};

template <typename T,
          typename Context,
          typename MT,
          typename IndexT,
          typename UpdateMethod>
void InnerCompute(const Context& dev_ctx,
                  const DenseTensor& param_in,
                  const DenseTensor& grad_in,
                  const DenseTensor& velocity_in,
                  const DenseTensor& index_in,
                  const DenseTensor& learning_rate_in,
                  const paddle::optional<DenseTensor>& master_param_in,
                  float mu_in,
                  const Scalar& axis_in,
                  bool use_nesterov,
                  const std::string& regularization_method,
                  float regularization_coeff_in,
                  bool multi_precision,
                  float rescale_grad_in,
                  DenseTensor* param_out,
                  DenseTensor* velocity_out,
                  DenseTensor* master_param_out_out,
                  const UpdateMethod& update_method) {
  MT regularization_coeff = static_cast<MT>(regularization_coeff_in);
  RegularizationType regularization_flag{
      RegularizationType::kNONE};  // disable regularization
  if (regularization_method == "l2_decay") {
    regularization_flag = RegularizationType::kL2DECAY;
  }

  MT mu = static_cast<MT>(mu_in);
  MT rescale_grad = static_cast<MT>(rescale_grad_in);

  int axis = axis_in.to<int>();
  PADDLE_ENFORCE_EQ(
      axis == 0 || axis == 1,
      true,
      common::errors::InvalidArgument("The axis of sparse_momentum_op only "
                                      "support axis=0 or axis=1 now."));

  auto learning_rate = &learning_rate_in;
  auto param = &param_in;
  auto velocity = &velocity_in;
  auto index = &index_in;
  int64_t num_index = index->numel();

  // check index of shape 1-D
  if (index->dims().size() == 1) {
    PADDLE_ENFORCE_GT(index->dims()[0],
                      0,
                      common::errors::InvalidArgument(
                          "The index of sparse_momentum_op should not be empty"
                          "when the index's rank is 1."));
  } else if (index->dims().size() == 2) {
    PADDLE_ENFORCE_EQ(index->dims()[1],
                      1,
                      common::errors::InvalidArgument(
                          "If the index's rank of sparse_momentum_op is 2,"
                          " the second dimension should be 1."));
  }

  const phi::DenseTensor* master_param = nullptr;
  phi::DenseTensor* master_param_out = nullptr;
  if (multi_precision) {
    bool has_master = (master_param_in.get_ptr() != nullptr) &&
                      (master_param_out_out != nullptr);
    PADDLE_ENFORCE_EQ(has_master,
                      true,
                      common::errors::InvalidArgument(
                          "The Input(MasterParam) and Output(MasterParamOut) "
                          "should not be null when "
                          "the attr `multi_precision` is true"));
    master_param = master_param_in.get_ptr();
    master_param_out = master_param_out_out;
  }

  dev_ctx.template Alloc<T>(param_out);
  dev_ctx.template Alloc<MT>(velocity_out);
  const MT* master_in_data =
      multi_precision ? master_param->data<MT>() : nullptr;
  MT* master_out_data =
      multi_precision ? dev_ctx.template Alloc<MT>(master_param_out) : nullptr;

  auto grad = &grad_in;

  phi::funcs::ForRange<Context> for_range(static_cast<const Context&>(dev_ctx),
                                          param->numel());

  auto param_dims = param->dims();
  auto grad_dims = grad->dims();

  PADDLE_ENFORCE_EQ(
      param_dims.size(),
      2,
      common::errors::InvalidArgument("The Param's rank of sparse_momentum_op"
                                      " must be 2 now."));
  PADDLE_ENFORCE_EQ(
      grad_dims.size(),
      2,
      common::errors::InvalidArgument("The Grad's rank of sparse_momentum_op"
                                      " must be 2 now."));

  phi::DenseTensor sorted_index, grad_index, sort_value;
  sorted_index.Resize({num_index});
  grad_index.Resize({num_index});
  auto sorted_index_ptr = dev_ctx.template Alloc<IndexT>(&sorted_index);
  auto grad_index_ptr = dev_ctx.template Alloc<IndexT>(&grad_index);

  if (dev_ctx.GetPlace().GetType() == phi::AllocationType::GPU) {
#if defined(__NVCC__) || defined(__HIPCC__)
    sort_value.Resize({num_index});
    auto sort_value_ptr = dev_ctx.template Alloc<IndexT>(&sort_value);

    phi::funcs::ForRange<Context> for_range_index(dev_ctx, num_index);
    RangeFunctor<IndexT> range_functor(sort_value_ptr);
    for_range_index(range_functor);

    size_t temp_storage_bytes = 0;
    PADDLE_ENFORCE_GPU_SUCCESS((cub::DeviceRadixSort::SortPairs<IndexT, IndexT>(
        nullptr,
        temp_storage_bytes,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        static_cast<int>(num_index))));
    auto d_temp_storage =
        phi::memory_utils::Alloc(dev_ctx.GetPlace(), temp_storage_bytes);
    PADDLE_ENFORCE_GPU_SUCCESS((cub::DeviceRadixSort::SortPairs<IndexT, IndexT>(
        d_temp_storage->ptr(),
        temp_storage_bytes,
        index->data<IndexT>(),
        sorted_index_ptr,
        sort_value_ptr,
        grad_index_ptr,
        static_cast<int>(num_index),
        0,
        sizeof(IndexT) * 8,
        dev_ctx.stream())));
#endif
  } else if (dev_ctx.GetPlace().GetType() == phi::AllocationType::CPU) {
    std::vector<std::pair<IndexT, IndexT>> vec_tosort;
    auto index_ptr = index->data<IndexT>();
    for (IndexT i = 0; i < num_index; i++) {
      vec_tosort.push_back({index_ptr[i], i});
    }
    std::sort(vec_tosort.begin(),
              vec_tosort.end(),
              [](const std::pair<IndexT, IndexT>& k1,
                 const std::pair<IndexT, IndexT>& k2) {
                return k1.first < k2.first;
              });
    for (IndexT i = 0; i < num_index; i++) {
      sorted_index_ptr[i] = vec_tosort[i].first;
      grad_index_ptr[i] = vec_tosort[i].second;
    }
  } else {
    PADDLE_THROW(common::errors::Unimplemented(
        "sparse_momentum %s is not supported.", dev_ctx.GetPlace()));
  }
  using MPDType = MultiPrecisionType<T>;
  IndexMomentumFunctor<T, MT, IndexT, UpdateMethod> functor(
      param->data<T>(),
      grad->data<T>(),
      velocity->data<MT>(),
      learning_rate->data<MPDType>(),
      master_in_data,
      mu,
      rescale_grad,
      sorted_index_ptr,
      grad_index_ptr,
      num_index,
      axis,
      param_dims[1],
      grad_dims[1],
      regularization_flag,
      regularization_coeff,
      update_method,
      dev_ctx.template Alloc<T>(param_out),
      dev_ctx.template Alloc<MT>(velocity_out),
      master_out_data);
  for_range(functor);
}

template <typename T, typename Context>
void SparseMomentumOpKernel(const Context& dev_ctx,
                            const DenseTensor& param,
                            const DenseTensor& grad,
                            const DenseTensor& velocity,
                            const DenseTensor& index_in,
                            const DenseTensor& learning_rate,
                            const paddle::optional<DenseTensor>& master_param,
                            float mu,
                            const Scalar& axis,
                            bool use_nesterov,
                            const std::string& regularization_method,
                            float regularization_coeff,
                            bool multi_precision,
                            float rescale_grad,
                            DenseTensor* param_out,
                            DenseTensor* velocity_out,
                            DenseTensor* master_param_out) {
  using MPDType = MultiPrecisionType<T>;

  auto index = &index_in;
  const auto& index_type = index->dtype();
  if (multi_precision) {
    if (use_nesterov) {
      auto update_method = UseNesterov<MPDType>();
      if (index_type == phi::DataType::INT32) {
        InnerCompute<T, Context, MPDType, int, UseNesterov<MPDType>>(
            dev_ctx,
            param,
            grad,
            velocity,
            index_in,
            learning_rate,
            master_param,
            mu,
            axis,
            use_nesterov,
            regularization_method,
            regularization_coeff,
            multi_precision,
            rescale_grad,
            param_out,
            velocity_out,
            master_param_out,
            update_method);
      } else {
        InnerCompute<T, Context, MPDType, int64_t, UseNesterov<MPDType>>(
            dev_ctx,
            param,
            grad,
            velocity,
            index_in,
            learning_rate,
            master_param,
            mu,
            axis,
            use_nesterov,
            regularization_method,
            regularization_coeff,
            multi_precision,
            rescale_grad,
            param_out,
            velocity_out,
            master_param_out,
            update_method);
      }
    } else {
      auto update_method = NoNesterov<MPDType>();
      if (index_type == phi::DataType::INT32) {
        InnerCompute<T, Context, MPDType, int, NoNesterov<MPDType>>(
            dev_ctx,
            param,
            grad,
            velocity,
            index_in,
            learning_rate,
            master_param,
            mu,
            axis,
            use_nesterov,
            regularization_method,
            regularization_coeff,
            multi_precision,
            rescale_grad,
            param_out,
            velocity_out,
            master_param_out,
            update_method);
      } else {
        InnerCompute<T, Context, MPDType, int64_t, NoNesterov<MPDType>>(
            dev_ctx,
            param,
            grad,
            velocity,
            index_in,
            learning_rate,
            master_param,
            mu,
            axis,
            use_nesterov,
            regularization_method,
            regularization_coeff,
            multi_precision,
            rescale_grad,
            param_out,
            velocity_out,
            master_param_out,
            update_method);
      }
    }
  } else {
    if (use_nesterov) {
      auto update_method = UseNesterov<T>();
      if (index_type == phi::DataType::INT32) {
        InnerCompute<T, Context, T, int, UseNesterov<T>>(dev_ctx,
                                                         param,
                                                         grad,
                                                         velocity,
                                                         index_in,
                                                         learning_rate,
                                                         master_param,
                                                         mu,
                                                         axis,
                                                         use_nesterov,
                                                         regularization_method,
                                                         regularization_coeff,
                                                         multi_precision,
                                                         rescale_grad,
                                                         param_out,
                                                         velocity_out,
                                                         master_param_out,
                                                         update_method);
      } else {
        InnerCompute<T, Context, T, int64_t, UseNesterov<T>>(
            dev_ctx,
            param,
            grad,
            velocity,
            index_in,
            learning_rate,
            master_param,
            mu,
            axis,
            use_nesterov,
            regularization_method,
            regularization_coeff,
            multi_precision,
            rescale_grad,
            param_out,
            velocity_out,
            master_param_out,
            update_method);
      }
    } else {
      auto update_method = NoNesterov<T>();
      if (index_type == phi::DataType::INT32) {
        InnerCompute<T, Context, T, int, NoNesterov<T>>(dev_ctx,
                                                        param,
                                                        grad,
                                                        velocity,
                                                        index_in,
                                                        learning_rate,
                                                        master_param,
                                                        mu,
                                                        axis,
                                                        use_nesterov,
                                                        regularization_method,
                                                        regularization_coeff,
                                                        multi_precision,
                                                        rescale_grad,
                                                        param_out,
                                                        velocity_out,
                                                        master_param_out,
                                                        update_method);
      } else {
        InnerCompute<T, Context, T, int64_t, NoNesterov<T>>(
            dev_ctx,
            param,
            grad,
            velocity,
            index_in,
            learning_rate,
            master_param,
            mu,
            axis,
            use_nesterov,
            regularization_method,
            regularization_coeff,
            multi_precision,
            rescale_grad,
            param_out,
            velocity_out,
            master_param_out,
            update_method);
      }
    }
  }
}

}  // namespace phi
