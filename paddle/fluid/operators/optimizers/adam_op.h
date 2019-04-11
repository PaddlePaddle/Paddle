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
#include <math.h>  // for sqrt in CPU and CUDA
#include <Eigen/Dense>
#include <unordered_map>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/threadpool.h"
#include "paddle/fluid/operators/detail/safe_ref.h"
#include "paddle/fluid/operators/math/algorithm.h"
#include "paddle/fluid/operators/math/selected_rows_functor.h"
#include "paddle/fluid/platform/for_range.h"

namespace paddle {
namespace operators {

namespace scatter = paddle::operators::math::scatter;

struct GPUAdam;
struct CPUAdam;

template <typename T, typename Flavour>
struct AdamFunctor;

template <typename T>
struct AdamFunctor<T, GPUAdam> {
  T beta1_;
  T beta2_;
  T epsilon_;

  const T* beta1_pow_;
  const T* beta2_pow_;
  const T* moment1_;
  T* moment1_out_;
  const T* moment2_;
  T* moment2_out_;
  const T* lr_;
  const T* grad_;
  const T* param_;
  T* param_out_;

  AdamFunctor(T beta1, T beta2, T epsilon, const T* beta1_pow,
              const T* beta2_pow, const T* mom1, T* mom1_out, const T* mom2,
              T* mom2_out, const T* lr, const T* grad, const T* param,
              T* param_out)
      : beta1_(beta1),
        beta2_(beta2),
        epsilon_(epsilon),
        beta1_pow_(beta1_pow),
        beta2_pow_(beta2_pow),
        moment1_(mom1),
        moment1_out_(mom1_out),
        moment2_(mom2),
        moment2_out_(mom2_out),
        lr_(lr),
        grad_(grad),
        param_(param),
        param_out_(param_out) {}

  inline HOSTDEVICE void operator()(size_t i) const {
    // Merge all memory access together.
    T g = grad_[i];
    T mom1 = moment1_[i];
    T mom2 = moment2_[i];
    T lr = *lr_;
    T beta1_pow = *beta1_pow_;
    T beta2_pow = *beta2_pow_;
    T p = param_[i];

    // Calculation
    lr *= sqrt(1 - beta2_pow) / (1 - beta1_pow);

    mom1 = beta1_ * mom1 + (1 - beta1_) * g;
    mom2 = beta2_ * mom2 + (1 - beta2_) * g * g;
    p -= lr * (mom1 / (sqrt(mom2) + epsilon_));

    // Write back to global memory
    moment1_out_[i] = mom1;
    moment2_out_[i] = mom2;
    param_out_[i] = p;
  }
};

template <typename T>
struct AdamFunctor<T, CPUAdam> {
  T beta1_;
  T beta2_;
  T epsilon_;

  const T* beta1_pow_;
  const T* beta2_pow_;
  const T* moment1_;
  T* moment1_out_;
  const T* moment2_;
  T* moment2_out_;
  const T* lr_;
  const T* grad_;
  const T* param_;
  T* param_out_;

  AdamFunctor(T beta1, T beta2, T epsilon, const T* beta1_pow,
              const T* beta2_pow, const T* mom1, T* mom1_out, const T* mom2,
              T* mom2_out, const T* lr, const T* grad, const T* param,
              T* param_out)
      : beta1_(beta1),
        beta2_(beta2),
        epsilon_(epsilon),
        beta1_pow_(beta1_pow),
        beta2_pow_(beta2_pow),
        moment1_(mom1),
        moment1_out_(mom1_out),
        moment2_(mom2),
        moment2_out_(mom2_out),
        lr_(lr),
        grad_(grad),
        param_(param),
        param_out_(param_out) {}

  void operator()(size_t numel) const {
    Eigen::Map<const Eigen::Array<T, 1, Eigen::Dynamic>> g{
        grad_, static_cast<Eigen::Index>(numel)};
    Eigen::Map<const Eigen::Array<T, 1, Eigen::Dynamic>> mom1{
        moment1_, static_cast<Eigen::Index>(numel)};
    Eigen::Map<const Eigen::Array<T, 1, Eigen::Dynamic>> mom2{
        moment2_, static_cast<Eigen::Index>(numel)};
    Eigen::Map<const Eigen::Array<T, 1, Eigen::Dynamic>> param{
        param_, static_cast<Eigen::Index>(numel)};

    Eigen::Map<Eigen::Array<T, 1, Eigen::Dynamic>> param_out{
        param_out_, static_cast<Eigen::Index>(numel)};
    Eigen::Map<Eigen::Array<T, 1, Eigen::Dynamic>> moment1_out{
        moment1_out_, static_cast<Eigen::Index>(numel)};
    Eigen::Map<Eigen::Array<T, 1, Eigen::Dynamic>> moment2_out{
        moment2_out_, static_cast<Eigen::Index>(numel)};

    T lr = *lr_;
    T beta1_pow = *beta1_pow_;
    T beta2_pow = *beta2_pow_;

    // Calculation
    lr *= sqrt(1 - beta2_pow) / (1 - beta1_pow);

    moment1_out = beta1_ * mom1 + (1 - beta1_) * g;
    moment2_out = beta2_ * mom2 + (1 - beta2_) * g * g;
    param_out = param - lr * (moment1_out / (moment2_out.sqrt() + epsilon_));
  }
};

template <typename T, typename Flavour>
struct SparseAdamFunctor;

template <typename T>
struct SparseAdamFunctor<T, GPUAdam> {
  T beta1_;
  T beta2_;
  T epsilon_;

  const T* beta1_pow_;
  const T* beta2_pow_;
  const T* moment1_;
  T* moment1_out_;
  const T* moment2_;
  T* moment2_out_;
  const T* lr_;
  const T* grad_;
  const T* param_;
  T* param_out_;

  const int64_t* rows_;
  int64_t row_numel_;
  int64_t row_count_;
  bool lazy_mode_;

  SparseAdamFunctor(T beta1, T beta2, T epsilon, const T* beta1_pow,
                    const T* beta2_pow, const T* mom1, T* mom1_out,
                    const T* mom2, T* mom2_out, const T* lr, const T* grad,
                    const T* param, T* param_out, const int64_t* rows,
                    int64_t row_numel, int64_t row_count, bool lazy_mode)
      : beta1_(beta1),
        beta2_(beta2),
        epsilon_(epsilon),
        beta1_pow_(beta1_pow),
        beta2_pow_(beta2_pow),
        moment1_(mom1),
        moment1_out_(mom1_out),
        moment2_(mom2),
        moment2_out_(mom2_out),
        lr_(lr),
        grad_(grad),
        param_(param),
        param_out_(param_out),
        rows_(rows),
        row_numel_(row_numel),
        row_count_(row_count),
        lazy_mode_(lazy_mode) {}

  inline HOSTDEVICE void adam_update(size_t i, T g) const {
    // The following code is the same as dense
    T mom1 = moment1_[i];
    T mom2 = moment2_[i];
    T lr = *lr_;
    T beta1_pow = *beta1_pow_;
    T beta2_pow = *beta2_pow_;
    T p = param_[i];

    // Calculation
    lr *= sqrt(1 - beta2_pow) / (1 - beta1_pow);

    mom1 = beta1_ * mom1 + (1 - beta1_) * g;
    mom2 = beta2_ * mom2 + (1 - beta2_) * g * g;
    p -= lr * (mom1 / (sqrt(mom2) + epsilon_));

    // Write back to global memory
    moment1_out_[i] = mom1;
    moment2_out_[i] = mom2;
    param_out_[i] = p;
  }

  inline HOSTDEVICE void operator()(size_t i) const {
    auto row_idx =
        math::BinarySearch<int64_t>(rows_, row_count_, i / row_numel_);
    if (lazy_mode_ && row_idx < 0) {
      return;
    } else {
      T g = row_idx >= 0 ? grad_[row_idx * row_numel_ + i % row_numel_] : 0;
      adam_update(i, g);
    }
  }
};

template <typename T>
struct SparseAdamFunctor<T, CPUAdam> {
  T beta1_;
  T beta2_;
  T epsilon_;

  const T* beta1_pow_;
  const T* beta2_pow_;
  const T* moment1_;
  T* moment1_out_;
  const T* moment2_;
  T* moment2_out_;
  const T* lr_;
  const T* grad_;
  const T* param_;
  T* param_out_;

  const int64_t* rows_;
  int64_t row_numel_;
  int64_t row_count_;

  SparseAdamFunctor(T beta1, T beta2, T epsilon, const T* beta1_pow,
                    const T* beta2_pow, const T* mom1, T* mom1_out,
                    const T* mom2, T* mom2_out, const T* lr, const T* grad,
                    const T* param, T* param_out, const int64_t* rows,
                    int64_t row_numel, int64_t row_count, bool lazy_mode)
      : beta1_(beta1),
        beta2_(beta2),
        epsilon_(epsilon),
        beta1_pow_(beta1_pow),
        beta2_pow_(beta2_pow),
        moment1_(mom1),
        moment1_out_(mom1_out),
        moment2_(mom2),
        moment2_out_(mom2_out),
        lr_(lr),
        grad_(grad),
        param_(param),
        param_out_(param_out),
        rows_(rows),
        row_numel_(row_numel),
        row_count_(row_count) {}

  inline HOSTDEVICE void adam_update(size_t i, T g) const {
    // The following code is the same as dense
    T mom1 = moment1_[i];
    T mom2 = moment2_[i];
    T lr = *lr_;
    T beta1_pow = *beta1_pow_;
    T beta2_pow = *beta2_pow_;
    T p = param_[i];

    // Calculation
    lr *= sqrt(1 - beta2_pow) / (1 - beta1_pow);

    mom1 = beta1_ * mom1 + (1 - beta1_) * g;
    mom2 = beta2_ * mom2 + (1 - beta2_) * g * g;
    p -= lr * (mom1 / (sqrt(mom2) + epsilon_));

    // Write back to global memory
    moment1_out_[i] = mom1;
    moment2_out_[i] = mom2;
    param_out_[i] = p;
  }

  inline void operator()(size_t numel) const {
    // lr could be reuse
    T lr = *lr_;
    T beta1_pow = *beta1_pow_;
    T beta2_pow = *beta2_pow_;
    lr *= sqrt(1 - beta2_pow) / (1 - beta1_pow);
    int64_t row_count = static_cast<int64_t>(numel / row_numel_);

    for (int64_t i = 0, j = 0; i != row_count; ++i) {
      if (i == *(rows_ + j)) {
        for (int64_t k = 0; k != row_numel_; ++k) {
          T g = grad_[j * row_numel_ + k];
          adam_update(i * row_numel_ + k, g);
        }
        ++j;
      } else {
        for (int64_t k = 0; k != row_numel_; ++k) {
          T mom1 = moment1_[i * row_numel_ + k];
          T mom2 = moment2_[i * row_numel_ + k];
          T p = param_[i * row_numel_ + k];

          mom1 = beta1_ * mom1;
          mom2 = beta2_ * mom2;

          p -= lr * (mom1 / (sqrt(mom2) + epsilon_));
          // Write back to global memory
          moment1_out_[i * row_numel_ + k] = mom1;
          moment2_out_[i * row_numel_ + k] = mom2;
          param_out_[i * row_numel_ + k] = p;
        }
      }
    }
  }
};

template <typename DeviceContext, typename T>
class AdamOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const auto* param_var = ctx.InputVar("Param");
    PADDLE_ENFORCE(param_var->IsType<framework::LoDTensor>(),
                   "The Var(%s)'s type should be LoDTensor, "
                   "but the received is %s",
                   ctx.Inputs("Param").front(),
                   framework::ToTypeName(param_var->Type()));

    using paddle::framework::LoDTensor;
    using paddle::operators::detail::Ref;

    int64_t min_row_size_to_use_multithread =
        ctx.Attr<int64_t>("min_row_size_to_use_multithread");
    bool lazy_mode = ctx.Attr<bool>("lazy_mode");
    T beta1 = static_cast<T>(ctx.Attr<float>("beta1"));
    T beta2 = static_cast<T>(ctx.Attr<float>("beta2"));
    T epsilon = static_cast<T>(ctx.Attr<float>("epsilon"));
    auto& param = Ref(ctx.Input<LoDTensor>("Param"), "Must set Param");
    // auto& grad = Ref(ctx.Input<LoDTensor>("Grad"), "Must set Grad");
    auto* grad_var = ctx.InputVar("Grad");
    auto& mom1 = Ref(ctx.Input<LoDTensor>("Moment1"), "Must set Moment1");
    auto& mom2 = Ref(ctx.Input<LoDTensor>("Moment2"), "Must set Moment2");
    auto& lr =
        Ref(ctx.Input<LoDTensor>("LearningRate"), "Must set LearningRate");

    auto& beta1_pow =
        Ref(ctx.Input<LoDTensor>("Beta1Pow"), "Must set Beta1Pow");
    auto& beta2_pow =
        Ref(ctx.Input<LoDTensor>("Beta2Pow"), "Must set Beta2Pow");

    auto& param_out =
        Ref(ctx.Output<LoDTensor>("ParamOut"), "Must set ParamOut");
    auto& mom1_out =
        Ref(ctx.Output<LoDTensor>("Moment1Out"), "Must set Moment1Out");
    auto& mom2_out =
        Ref(ctx.Output<LoDTensor>("Moment2Out"), "Must set Moment1Out");

    if (grad_var->IsType<framework::LoDTensor>()) {
      auto& grad = Ref(ctx.Input<LoDTensor>("Grad"), "Must set Grad");

      if (platform::is_cpu_place(ctx.GetPlace())) {
        AdamFunctor<T, CPUAdam> functor(
            beta1, beta2, epsilon, beta1_pow.template data<T>(),
            beta2_pow.template data<T>(), mom1.template data<T>(),
            mom1_out.template mutable_data<T>(ctx.GetPlace()),
            mom2.template data<T>(),
            mom2_out.template mutable_data<T>(ctx.GetPlace()),
            lr.template data<T>(), grad.template data<T>(),
            param.template data<T>(),
            param_out.template mutable_data<T>(ctx.GetPlace()));
        functor(param.numel());
      } else if (platform::is_gpu_place(ctx.GetPlace())) {
        AdamFunctor<T, GPUAdam> functor(
            beta1, beta2, epsilon, beta1_pow.template data<T>(),
            beta2_pow.template data<T>(), mom1.template data<T>(),
            mom1_out.template mutable_data<T>(ctx.GetPlace()),
            mom2.template data<T>(),
            mom2_out.template mutable_data<T>(ctx.GetPlace()),
            lr.template data<T>(), grad.template data<T>(),
            param.template data<T>(),
            param_out.template mutable_data<T>(ctx.GetPlace()));

        platform::ForRange<DeviceContext> for_range(
            static_cast<const DeviceContext&>(ctx.device_context()),
            param.numel());
        for_range(functor);
      }
    } else if (grad_var->IsType<framework::SelectedRows>()) {
      auto& grad =
          Ref(ctx.Input<framework::SelectedRows>("Grad"), "Must set Grad");
      if (grad.rows().size() == 0) {
        VLOG(3) << "grad row size is 0!!";
        return;
      }

      std::vector<int64_t> cpu_rows(grad.rows().begin(), grad.rows().end());
      bool is_strict_sorted = true;
      for (size_t i = 1; i < cpu_rows.size(); ++i) {
        if (cpu_rows[i - 1] >= cpu_rows[i]) {
          is_strict_sorted = false;
          break;
        }
      }

      framework::SelectedRows tmp_grad_merge;
      const framework::SelectedRows* grad_merge_ptr;
      if (is_strict_sorted) {
        grad_merge_ptr = &grad;
      } else {
        // merge duplicated rows if any.
        // The rows of grad_merge have been sorted inside MergeAdd functor
        scatter::MergeAdd<DeviceContext, T> merge_func;
        merge_func(ctx.template device_context<DeviceContext>(), grad,
                   &tmp_grad_merge, true);
        grad_merge_ptr = &tmp_grad_merge;
      }

      auto& grad_merge = *grad_merge_ptr;
      auto& grad_tensor = grad_merge.value();
      const T* grad_data = grad_tensor.template data<T>();
      const int64_t* rows = grad_merge.rows().Data(ctx.GetPlace());
      auto row_numel = grad_tensor.numel() / grad_merge.rows().size();

      if (platform::is_cpu_place(ctx.GetPlace())) {
        SparseAdamFunctor<T, CPUAdam> functor(
            beta1, beta2, epsilon, beta1_pow.template data<T>(),
            beta2_pow.template data<T>(), mom1.template data<T>(),
            mom1_out.template mutable_data<T>(ctx.GetPlace()),
            mom2.template data<T>(),
            mom2_out.template mutable_data<T>(ctx.GetPlace()),
            lr.template data<T>(), grad_data, param.template data<T>(),
            param_out.template mutable_data<T>(ctx.GetPlace()), rows, row_numel,
            grad_merge.rows().size(), lazy_mode);
        if (lazy_mode) {
          VLOG(3) << "run cpu lazy mode";
          size_t row_count = grad_merge.rows().size();
          std::vector<int64_t> cpu_rows(grad_merge.rows());
          for (size_t row_index = 0; row_index < row_count; ++row_index) {
            for (size_t offset = 0; offset < row_numel; ++offset) {
              size_t i = cpu_rows[row_index] * row_numel + offset;
              functor.adam_update(i, grad_data[row_index * row_numel + offset]);
            }
          }
        }
#ifndef _WIN32
        else if (FLAGS_inner_op_parallelism > 1 &&  // NOLINT
                 min_row_size_to_use_multithread > 0 &&
                 param.dims()[0] > min_row_size_to_use_multithread) {
          VLOG(3) << "use multi thread, inner_op_parallelism="
                  << FLAGS_inner_op_parallelism
                  << " min_row_size_to_use_multithread="
                  << min_row_size_to_use_multithread;
          if (FLAGS_inner_op_parallelism > 10) {
            VLOG(1) << "FLAGS_inner_op_parallelism "
                    << FLAGS_inner_op_parallelism << " is two large!";
          }
          auto& grad_rows = grad_merge.rows();
          std::unordered_map<size_t, int> row_id_to_grad_row_offset;
          size_t param_row_count = param.numel() / row_numel;
          if (param_row_count < 1000) {
            VLOG(1) << "param_row_count should be larger then 1000 to use "
                       "multi thread, currently "
                    << param_row_count;
          }
          for (size_t i = 0; i < grad_rows.size(); ++i) {
            row_id_to_grad_row_offset[grad_rows[i]] = i;
          }
          std::vector<std::future<void>> fs;
          int64_t line_in_each_thread =
              param_row_count / FLAGS_inner_op_parallelism + 1;
          for (int i = 0; i < FLAGS_inner_op_parallelism; ++i) {
            int64_t start = i * line_in_each_thread;
            int64_t end = (i + 1) * line_in_each_thread;
            if (start >= static_cast<int64_t>(param_row_count)) {
              break;
            }
            if (end > static_cast<int64_t>(param_row_count)) {
              end = static_cast<int64_t>(param_row_count);
            }
            fs.push_back(
                framework::Async([&functor, &row_id_to_grad_row_offset,
                                  &grad_data, row_numel, start, end]() {
                  for (int64_t row_id = start; row_id < end; ++row_id) {
                    auto iter = row_id_to_grad_row_offset.find(row_id);
                    if (iter != row_id_to_grad_row_offset.end()) {
                      for (size_t row_offset = 0U; row_offset < row_numel;
                           ++row_offset) {
                        functor.adam_update(
                            row_id * row_numel + row_offset,
                            grad_data[iter->second * row_numel + row_offset]);
                      }
                    } else {
                      for (size_t row_offset = 0U; row_offset < row_numel;
                           ++row_offset) {
                        functor.adam_update(row_id * row_numel + row_offset, 0);
                      }
                    }
                  }
                }));
          }
          for (size_t i = 0; i < fs.size(); ++i) fs[i].wait();
        }
#endif          // !_WIN32
        else {  // NOLINT
          functor(param.numel());
        }
      } else if (platform::is_gpu_place(ctx.GetPlace())) {
        SparseAdamFunctor<T, GPUAdam> functor(
            beta1, beta2, epsilon, beta1_pow.template data<T>(),
            beta2_pow.template data<T>(), mom1.template data<T>(),
            mom1_out.template mutable_data<T>(ctx.GetPlace()),
            mom2.template data<T>(),
            mom2_out.template mutable_data<T>(ctx.GetPlace()),
            lr.template data<T>(), grad_data, param.template data<T>(),
            param_out.template mutable_data<T>(ctx.GetPlace()), rows, row_numel,
            grad_merge.rows().size(), lazy_mode);

        // FIXME(minqiyang): remove BinarySearch in GPU later
        platform::ForRange<DeviceContext> for_range(
            static_cast<const DeviceContext&>(ctx.device_context()),
            param.numel());
        for_range(functor);
      }
    } else {
      PADDLE_THROW("Variable type not supported by adam_op");
    }
  }
};

}  // namespace operators
}  // namespace paddle
