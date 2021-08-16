// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/amp/fp16_type_traits.h"
#include "paddle/fluid/operators/math/algorithm.h"
#include "paddle/fluid/platform/float16.h"
#include "paddle/fluid/platform/for_range.h"

#ifdef __NVCC__
#include "cub/cub.cuh"
#endif
#ifdef __HIPCC__
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#endif

namespace paddle {
namespace operators {

using framework::Tensor;
struct NoNesterov;
struct UseNesterov;

template <typename T>
using MultiPrecisionType = typename details::MPTypeTrait<T>::Type;

enum class RegularizationType {
  kNONE = 0,
  kL1DECAY = 1,  // do not need support right now
  kL2DECAY = 2,
};

// The following code is from
// https://en.cppreference.com/w/cpp/algorithm/lower_bound
// https://en.cppreference.com/w/cpp/algorithm/upper_bound
template <typename T>
HOSTDEVICE inline void BinarySearchLowerUpperBound(const T* x, int64_t num,
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

class SparseMomentumOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override;
};

class SparseMomentumOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("Param"), true,
        platform::errors::NotFound(
            "Input(param) of SparseMomentum should not be null."));
    PADDLE_ENFORCE_EQ(ctx->HasInput("Grad"), true,
                      platform::errors::NotFound(
                          "Input(grad) of SparseMomentum should not be null."));
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("Velocity"), true,
        platform::errors::NotFound(
            "Input(velocity) of SparseMomentum should not be null."));
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("Index"), true,
        platform::errors::NotFound(
            "Input(Index) of SparseMomentum should not be null."));
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("LearningRate"), true,
        platform::errors::NotFound(
            "Input(LearningRate) of SparseMomentum should not be null."));
    PADDLE_ENFORCE_EQ(ctx->GetInputsVarType("Param").front(),
                      framework::proto::VarType::LOD_TENSOR,
                      platform::errors::InvalidArgument(
                          "The input var's type of SparseMomentum should be "
                          "LoDTensor, but the received is %s",
                          ctx->GetInputsVarType("Param").front()));

    PADDLE_ENFORCE_EQ(
        ctx->HasOutput("ParamOut"), true,
        platform::errors::NotFound(
            "Output(ParamOut) of SparseMomentum should not be null."));
    PADDLE_ENFORCE_EQ(
        ctx->HasOutput("VelocityOut"), true,
        platform::errors::NotFound(
            "Output(VelocityOut) of SparseMomentum should not be null."));

    auto lr_dims = ctx->GetInputDim("LearningRate");
    PADDLE_ENFORCE_NE(framework::product(lr_dims), 0,
                      platform::errors::InvalidArgument(
                          "Maybe the Input variable LearningRate has not "
                          "been initialized. You may need to confirm "
                          "if you put exe.run(startup_program) "
                          "after optimizer.minimize function."));
    PADDLE_ENFORCE_EQ(framework::product(lr_dims), 1,
                      platform::errors::InvalidArgument(
                          "Learning_rate should be a scalar. But Received "
                          "LearningRate's dim [%s]",
                          framework::product(lr_dims)));

    auto param_dim = ctx->GetInputDim("Param");
    if (ctx->GetInputsVarType("Grad")[0] ==
        framework::proto::VarType::LOD_TENSOR) {
      PADDLE_ENFORCE_EQ(
          param_dim, ctx->GetInputDim("Velocity"),
          platform::errors::InvalidArgument(
              "Param and Velocity of SparseMomentumOp should have the same "
              "dimension. But received Param's dim [%s] and Velocity [%s].",
              param_dim, ctx->GetInputDim("Velocity")));
    }

    ctx->SetOutputDim("ParamOut", param_dim);
    ctx->SetOutputDim("VelocityOut", param_dim);
    if (ctx->HasOutput("MasterParamOut")) {
      ctx->SetOutputDim("MasterParamOut", param_dim);
    }
  }

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto input_data_type =
        OperatorWithKernel::IndicateVarDataType(ctx, "Param");
    return framework::OpKernelType(input_data_type, ctx.GetPlace());
  }
};

template <typename T, typename MT, typename IndexT, typename UpdateMethod>
class IndexMomentumFunctor;

template <typename T, typename MT, typename IndexT>
class IndexMomentumFunctor<T, MT, IndexT, UseNesterov> {
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

 public:
  IndexMomentumFunctor(const T* param, const T* grad, const MT* velocity,
                       const MultiPrecisionType<MT>* lr, const MT* master_param,
                       const MT mu, const MT rescale_grad,
                       const IndexT* sorted_index, const IndexT* grad_index,
                       int64_t num_index, int axis, int64_t param_row_numel,
                       int64_t grad_row_numel,
                       const RegularizationType regularization_flag,
                       const MT regularization_coeff, T* param_out,
                       MT* velocity_out, MT* master_param_out)
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
        regularization_coeff_(regularization_coeff) {}

  inline HOSTDEVICE void operator()(size_t i) {
    MT grad = static_cast<MT>(0);
    size_t row = i / param_row_numel_;
    size_t col = i % param_row_numel_;
    if (axis_ == 0) {
      int64_t row_idx0, row_idx1;
      BinarySearchLowerUpperBound<IndexT>(sorted_index_, num_index_, row,
                                          &row_idx0, &row_idx1);
      if (row_idx0 >= 0 && row_idx1 >= 0) {
        for (int64_t row_idx = row_idx0; row_idx <= row_idx1; row_idx++) {
          size_t offset = grad_index_[row_idx] * param_row_numel_ + col;
          grad += static_cast<MT>(grad_[offset]) * rescale_grad_;
        }
      }
    } else if (axis_ == 1) {
      int64_t col_idx0, col_idx1;
      BinarySearchLowerUpperBound<IndexT>(sorted_index_, num_index_, col,
                                          &col_idx0, &col_idx1);
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
    MT param_out = param - (grad + velocity_out * mu_) * lr;
    // write reigster to memory
    velocity_out_[i] = velocity_out;
    param_out_[i] = static_cast<T>(param_out);
    if (master_param_out_) {
      master_param_out_[i] = param_out;
    }
  }
};

template <typename T, typename MT, typename IndexT>
class IndexMomentumFunctor<T, MT, IndexT, NoNesterov> {
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

 public:
  IndexMomentumFunctor(const T* param, const T* grad, const MT* velocity,
                       const MultiPrecisionType<MT>* lr, const MT* master_param,
                       const MT mu, const MT rescale_grad,
                       const IndexT* sorted_index, const IndexT* grad_index,
                       int64_t num_index, int axis, int64_t param_row_numel,
                       int64_t grad_row_numel,
                       const RegularizationType regularization_flag,
                       const MT regularization_coeff, T* param_out,
                       MT* velocity_out, MT* master_param_out)
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
        regularization_coeff_(regularization_coeff) {}

  inline HOSTDEVICE void operator()(size_t i) {
    MT grad = static_cast<MT>(0);
    size_t row = i / param_row_numel_;
    size_t col = i % param_row_numel_;
    if (axis_ == 0) {
      int64_t row_idx0, row_idx1;
      BinarySearchLowerUpperBound<IndexT>(sorted_index_, num_index_, row,
                                          &row_idx0, &row_idx1);
      if (row_idx0 >= 0 && row_idx1 >= 0) {
        for (int64_t row_idx = row_idx0; row_idx <= row_idx1; row_idx++) {
          size_t offset = grad_index_[row_idx] * param_row_numel_ + col;
          grad += static_cast<MT>(grad_[offset]) * rescale_grad_;
        }
      }
    } else if (axis_ == 1) {
      int64_t col_idx0, col_idx1;
      BinarySearchLowerUpperBound<IndexT>(sorted_index_, num_index_, col,
                                          &col_idx0, &col_idx1);
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
    MT param_out = param - velocity_out * lr;
    // write reigster to memory
    velocity_out_[i] = velocity_out;
    param_out_[i] = static_cast<T>(param_out);
    if (master_param_out_) {
      master_param_out_[i] = param_out;
    }
  }
};

template <typename DeviceContext, typename T>
class SparseMomentumOpKernel : public framework::OpKernel<T> {
  using MPDType = MultiPrecisionType<T>;

 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const bool multi_precision = ctx.Attr<bool>("multi_precision");
    if (multi_precision) {
      InnerCompute<MPDType>(ctx, multi_precision);
    } else {
      InnerCompute<T>(ctx, multi_precision);
    }
  }

 private:
  template <typename MT>
  void InnerCompute(const framework::ExecutionContext& ctx,
                    const bool multi_precision) const {
    std::string regularization_method =
        ctx.Attr<std::string>("regularization_method");
    MT regularization_coeff =
        static_cast<MT>(ctx.Attr<float>("regularization_coeff"));
    RegularizationType regularization_flag{
        RegularizationType::kNONE};  // disable regularization
    if (regularization_method == "l2_decay") {
      regularization_flag = RegularizationType::kL2DECAY;
    }

    MT mu = static_cast<MT>(ctx.Attr<float>("mu"));
    MT rescale_grad = static_cast<MT>(ctx.Attr<float>("rescale_grad"));
    bool use_nesterov = ctx.Attr<bool>("use_nesterov");

    int axis = ctx.Attr<int>("axis");
    // get axis from tensor
    if (ctx.HasInput("Axis")) {
      Tensor cpu_axis;
      const Tensor* axis_tensor = ctx.Input<Tensor>("Axis");
      framework::TensorCopy(*axis_tensor, platform::CPUPlace(), &cpu_axis);
      const auto& axis_type = axis_tensor->type();
      if (axis_type == framework::proto::VarType::INT32) {
        axis = static_cast<int>(cpu_axis.data<int32_t>()[0]);
      } else if (axis_type == framework::proto::VarType::INT64) {
        axis = static_cast<int>(cpu_axis.data<int64_t>()[0]);
      }
    }
    PADDLE_ENFORCE_EQ(
        axis == 0 || axis == 1, true,
        platform::errors::InvalidArgument("The axis of sparse_momentum_op only "
                                          "support axis=0 or axis=1 now."));

    auto learning_rate = ctx.Input<framework::Tensor>("LearningRate");
    auto param = ctx.Input<framework::Tensor>("Param");
    auto param_out = ctx.Output<framework::Tensor>("ParamOut");
    auto velocity = ctx.Input<framework::Tensor>("Velocity");
    auto velocity_out = ctx.Output<framework::Tensor>("VelocityOut");
    auto index = ctx.Input<framework::Tensor>("Index");
    const auto& index_type = index->type();
    int64_t num_index = index->numel();

    // check index of shape 1-D
    if (index->dims().size() == 1) {
      PADDLE_ENFORCE_GT(
          index->dims()[0], 0,
          platform::errors::InvalidArgument(
              "The index of sparse_momentum_op should not be empty"
              "when the index's rank is 1."));
    } else if (index->dims().size() == 2) {
      PADDLE_ENFORCE_EQ(index->dims()[1], 1,
                        platform::errors::InvalidArgument(
                            "If the index's rank of sparse_momentum_op is 2,"
                            " the second dimension should be 1."));
    }

    const framework::Tensor* master_param = nullptr;
    framework::Tensor* master_param_out = nullptr;
    if (multi_precision) {
      bool has_master =
          ctx.HasInput("MasterParam") && ctx.HasOutput("MasterParamOut");
      PADDLE_ENFORCE_EQ(has_master, true,
                        platform::errors::InvalidArgument(
                            "The Input(MasterParam) and Output(MasterParamOut) "
                            "should not be null when "
                            "the attr `multi_precision` is true"));
      master_param = ctx.Input<framework::Tensor>("MasterParam");
      master_param_out = ctx.Output<framework::Tensor>("MasterParamOut");
    }

    param_out->mutable_data<T>(ctx.GetPlace());
    velocity_out->mutable_data<MT>(ctx.GetPlace());
    const MT* master_in_data =
        multi_precision ? master_param->data<MT>() : nullptr;
    MT* master_out_data =
        multi_precision ? master_param_out->mutable_data<MT>(ctx.GetPlace())
                        : nullptr;

    auto grad = ctx.Input<framework::Tensor>("Grad");

    platform::ForRange<DeviceContext> for_range(
        static_cast<const DeviceContext&>(ctx.device_context()),
        param->numel());
    platform::ForRange<DeviceContext> for_range_index(
        static_cast<const DeviceContext&>(ctx.device_context()), num_index);

    auto param_dims = param->dims();
    auto grad_dims = grad->dims();

    PADDLE_ENFORCE_EQ(param_dims.size(), 2,
                      platform::errors::InvalidArgument(
                          "The Param's rank of sparse_momentum_op"
                          " must be 2 now."));
    PADDLE_ENFORCE_EQ(grad_dims.size(), 2,
                      platform::errors::InvalidArgument(
                          "The Grad's rank of sparse_momentum_op"
                          " must be 2 now."));

    if (use_nesterov) {
      if (index_type == framework::proto::VarType::INT32) {
        Tensor sorted_index, grad_index, sort_value;
        auto sorted_index_ptr =
            sorted_index.mutable_data<int>({num_index}, ctx.GetPlace());
        auto grad_index_ptr =
            grad_index.mutable_data<int>({num_index}, ctx.GetPlace());
        auto sort_value_ptr =
            sort_value.mutable_data<int>({num_index}, ctx.GetPlace());

        RangeFunctor<int> range_functor(sort_value_ptr);
        for_range_index(range_functor);

        if (platform::is_gpu_place(ctx.GetPlace())) {
#if defined(__NVCC__) || defined(__HIPCC__)
          size_t temp_storage_bytes = 0;
          PADDLE_ENFORCE_CUDA_SUCCESS(
              (cub::DeviceRadixSort::SortPairs<int, int>(
                  nullptr, temp_storage_bytes, nullptr, nullptr, nullptr,
                  nullptr, static_cast<int>(num_index))));
          auto d_temp_storage =
              memory::Alloc(ctx.GetPlace(), temp_storage_bytes);
          PADDLE_ENFORCE_CUDA_SUCCESS(
              (cub::DeviceRadixSort::SortPairs<int, int>(
                  d_temp_storage->ptr(), temp_storage_bytes, index->data<int>(),
                  sorted_index_ptr, sort_value_ptr, grad_index_ptr,
                  static_cast<int>(num_index), 0, sizeof(int) * 8,
                  ctx.cuda_device_context().stream())));
#endif
        } else if (platform::is_cpu_place(ctx.GetPlace())) {
          std::vector<std::pair<int, int>> vec_tosort;
          auto index_ptr = index->data<int>();
          for (int i = 0; i < num_index; i++) {
            vec_tosort.push_back({index_ptr[i], i});
          }
          std::sort(
              vec_tosort.begin(), vec_tosort.end(),
              [](const std::pair<int, int>& k1, const std::pair<int, int>& k2) {
                return k1.first < k2.first;
              });
          for (int i = 0; i < num_index; i++) {
            sorted_index_ptr[i] = vec_tosort[i].first;
            grad_index_ptr[i] = vec_tosort[i].second;
          }
        } else {
          PADDLE_THROW(platform::errors::Unimplemented(
              "sparse_momentum %s is not supported.", ctx.GetPlace()));
        }

        IndexMomentumFunctor<T, MT, int, UseNesterov> functor(
            param->data<T>(), grad->data<T>(), velocity->data<MT>(),
            learning_rate->data<MPDType>(), master_in_data, mu, rescale_grad,
            sorted_index_ptr, grad_index_ptr, num_index, axis, param_dims[1],
            grad_dims[1], regularization_flag, regularization_coeff,
            param_out->mutable_data<T>(ctx.GetPlace()),
            velocity_out->mutable_data<MT>(ctx.GetPlace()), master_out_data);
        for_range(functor);
      } else if (index_type == framework::proto::VarType::INT64) {
        Tensor sorted_index, grad_index, sort_value;
        auto sorted_index_ptr =
            sorted_index.mutable_data<int64_t>({num_index}, ctx.GetPlace());
        auto grad_index_ptr =
            grad_index.mutable_data<int64_t>({num_index}, ctx.GetPlace());
        auto sort_value_ptr =
            sort_value.mutable_data<int64_t>({num_index}, ctx.GetPlace());

        RangeFunctor<int64_t> range_functor(sort_value_ptr);
        for_range_index(range_functor);

        if (platform::is_gpu_place(ctx.GetPlace())) {
#if defined(__NVCC__) || defined(__HIPCC__)
          size_t temp_storage_bytes = 0;
          PADDLE_ENFORCE_CUDA_SUCCESS(
              (cub::DeviceRadixSort::SortPairs<int64_t, int64_t>(
                  nullptr, temp_storage_bytes, nullptr, nullptr, nullptr,
                  nullptr, static_cast<int>(num_index))));
          auto d_temp_storage =
              memory::Alloc(ctx.GetPlace(), temp_storage_bytes);
          PADDLE_ENFORCE_CUDA_SUCCESS(
              (cub::DeviceRadixSort::SortPairs<int64_t, int64_t>(
                  d_temp_storage->ptr(), temp_storage_bytes,
                  index->data<int64_t>(), sorted_index_ptr, sort_value_ptr,
                  grad_index_ptr, static_cast<int>(num_index), 0,
                  sizeof(int64_t) * 8, ctx.cuda_device_context().stream())));
#endif
        } else if (platform::is_cpu_place(ctx.GetPlace())) {
          std::vector<std::pair<int64_t, int64_t>> vec_tosort;
          auto index_ptr = index->data<int64_t>();
          for (int64_t i = 0; i < num_index; i++) {
            vec_tosort.push_back({index_ptr[i], i});
          }
          std::sort(vec_tosort.begin(), vec_tosort.end(),
                    [](const std::pair<int64_t, int64_t>& k1,
                       const std::pair<int64_t, int64_t>& k2) {
                      return k1.first < k2.first;
                    });
          for (int64_t i = 0; i < num_index; i++) {
            sorted_index_ptr[i] = vec_tosort[i].first;
            grad_index_ptr[i] = vec_tosort[i].second;
          }
        } else {
          PADDLE_THROW(platform::errors::Unimplemented(
              "sparse_momentum %s is not supported.", ctx.GetPlace()));
        }

        IndexMomentumFunctor<T, MT, int64_t, UseNesterov> functor(
            param->data<T>(), grad->data<T>(), velocity->data<MT>(),
            learning_rate->data<MPDType>(), master_in_data, mu, rescale_grad,
            sorted_index_ptr, grad_index_ptr, num_index, axis, param_dims[1],
            grad_dims[1], regularization_flag, regularization_coeff,
            param_out->mutable_data<T>(ctx.GetPlace()),
            velocity_out->mutable_data<MT>(ctx.GetPlace()), master_out_data);
        for_range(functor);
      }
    } else {
      if (index_type == framework::proto::VarType::INT32) {
        Tensor sorted_index, grad_index, sort_value;
        auto sorted_index_ptr =
            sorted_index.mutable_data<int>({num_index}, ctx.GetPlace());
        auto grad_index_ptr =
            grad_index.mutable_data<int>({num_index}, ctx.GetPlace());
        auto sort_value_ptr =
            sort_value.mutable_data<int>({num_index}, ctx.GetPlace());

        RangeFunctor<int> range_functor(sort_value_ptr);
        for_range_index(range_functor);

        if (platform::is_gpu_place(ctx.GetPlace())) {
#if defined(__NVCC__) || defined(__HIPCC__)
          size_t temp_storage_bytes = 0;
          PADDLE_ENFORCE_CUDA_SUCCESS(
              (cub::DeviceRadixSort::SortPairs<int, int>(
                  nullptr, temp_storage_bytes, nullptr, nullptr, nullptr,
                  nullptr, static_cast<int>(num_index))));
          auto d_temp_storage =
              memory::Alloc(ctx.GetPlace(), temp_storage_bytes);
          PADDLE_ENFORCE_CUDA_SUCCESS(
              (cub::DeviceRadixSort::SortPairs<int, int>(
                  d_temp_storage->ptr(), temp_storage_bytes, index->data<int>(),
                  sorted_index_ptr, sort_value_ptr, grad_index_ptr,
                  static_cast<int>(num_index), 0, sizeof(int) * 8,
                  ctx.cuda_device_context().stream())));
#endif
        } else if (platform::is_cpu_place(ctx.GetPlace())) {
          std::vector<std::pair<int, int>> vec_tosort;
          auto index_ptr = index->data<int>();
          for (int i = 0; i < num_index; i++) {
            vec_tosort.push_back({index_ptr[i], i});
          }
          std::sort(
              vec_tosort.begin(), vec_tosort.end(),
              [](const std::pair<int, int>& k1, const std::pair<int, int>& k2) {
                return k1.first < k2.first;
              });
          for (int i = 0; i < num_index; i++) {
            sorted_index_ptr[i] = vec_tosort[i].first;
            grad_index_ptr[i] = vec_tosort[i].second;
          }
        } else {
          PADDLE_THROW(platform::errors::Unimplemented(
              "sparse_momentum %s is not supported.", ctx.GetPlace()));
        }

        IndexMomentumFunctor<T, MT, int, NoNesterov> functor(
            param->data<T>(), grad->data<T>(), velocity->data<MT>(),
            learning_rate->data<MPDType>(), master_in_data, mu, rescale_grad,
            sorted_index_ptr, grad_index_ptr, num_index, axis, param_dims[1],
            grad_dims[1], regularization_flag, regularization_coeff,
            param_out->mutable_data<T>(ctx.GetPlace()),
            velocity_out->mutable_data<MT>(ctx.GetPlace()), master_out_data);
        for_range(functor);
      } else if (index_type == framework::proto::VarType::INT64) {
        Tensor sorted_index, grad_index, sort_value;
        auto sorted_index_ptr =
            sorted_index.mutable_data<int64_t>({num_index}, ctx.GetPlace());
        auto grad_index_ptr =
            grad_index.mutable_data<int64_t>({num_index}, ctx.GetPlace());
        auto sort_value_ptr =
            sort_value.mutable_data<int64_t>({num_index}, ctx.GetPlace());

        RangeFunctor<int64_t> range_functor(sort_value_ptr);
        for_range_index(range_functor);

        if (platform::is_gpu_place(ctx.GetPlace())) {
#if defined(__NVCC__) || defined(__HIPCC__)
          size_t temp_storage_bytes = 0;
          PADDLE_ENFORCE_CUDA_SUCCESS(
              (cub::DeviceRadixSort::SortPairs<int64_t, int64_t>(
                  nullptr, temp_storage_bytes, nullptr, nullptr, nullptr,
                  nullptr, static_cast<int>(num_index))));
          auto d_temp_storage =
              memory::Alloc(ctx.GetPlace(), temp_storage_bytes);
          PADDLE_ENFORCE_CUDA_SUCCESS(
              (cub::DeviceRadixSort::SortPairs<int64_t, int64_t>(
                  d_temp_storage->ptr(), temp_storage_bytes,
                  index->data<int64_t>(), sorted_index_ptr, sort_value_ptr,
                  grad_index_ptr, static_cast<int>(num_index), 0,
                  sizeof(int64_t) * 8, ctx.cuda_device_context().stream())));
#endif
        } else if (platform::is_cpu_place(ctx.GetPlace())) {
          std::vector<std::pair<int64_t, int64_t>> vec_tosort;
          auto index_ptr = index->data<int64_t>();
          for (int64_t i = 0; i < num_index; i++) {
            vec_tosort.push_back({index_ptr[i], i});
          }
          std::sort(vec_tosort.begin(), vec_tosort.end(),
                    [](const std::pair<int64_t, int64_t>& k1,
                       const std::pair<int64_t, int64_t>& k2) {
                      return k1.first < k2.first;
                    });
          for (int64_t i = 0; i < num_index; i++) {
            sorted_index_ptr[i] = vec_tosort[i].first;
            grad_index_ptr[i] = vec_tosort[i].second;
          }
        } else {
          PADDLE_THROW(platform::errors::Unimplemented(
              "sparse_momentum %s is not supported.", ctx.GetPlace()));
        }

        IndexMomentumFunctor<T, MT, int64_t, NoNesterov> functor(
            param->data<T>(), grad->data<T>(), velocity->data<MT>(),
            learning_rate->data<MPDType>(), master_in_data, mu, rescale_grad,
            sorted_index_ptr, grad_index_ptr, num_index, axis, param_dims[1],
            grad_dims[1], regularization_flag, regularization_coeff,
            param_out->mutable_data<T>(ctx.GetPlace()),
            velocity_out->mutable_data<MT>(ctx.GetPlace()), master_out_data);
        for_range(functor);
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle
