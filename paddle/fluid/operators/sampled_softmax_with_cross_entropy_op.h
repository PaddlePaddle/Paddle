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

#include <vector>
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/operators/math/cross_entropy_multi_label.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/operators/math/sample_prob.h"
#include "paddle/fluid/operators/math/softmax.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenMatrix = framework::EigenMatrix<T, MajorType, IndexType>;

// UNDERSTAND: something like take_along_axis in numpy.
template <typename T>
static void CPUTakeAlongD1(const platform::DeviceContext& ctx,
                           const framework::Tensor& array,
                           const framework::Tensor& index,
                           framework::Tensor* value) {
  PADDLE_ENFORCE(platform::is_cpu_place(ctx.GetPlace()));
  // UNDERSTAND: check shape src(B, C), index(B, K), out should also be (B, K)
  PADDLE_ENFORCE(index.dims().size() == 2 && array.dims().size() == 2 &&
                 index.dims()[0] == array.dims()[0] &&
                 index.dims() == value->dims());

  const auto batch_size = index.dims()[0];
  const auto num_take = index.dims()[1];
  const auto array_dims = array.dims();
  const auto idx_dims = index.dims();

  // UNDERSTAND: no allocations here
  const T* p_array = array.data<T>();
  const int64_t* p_index = index.data<int64_t>();
  T* p_value = value->data<T>();

  // src slice size
  const auto array_slice_size = array_dims[1];

  // index slice size
  const auto idx_slice_size = idx_dims[1];
  const auto value_slice_size = idx_slice_size;

  for (int i = 0; i < batch_size; ++i) {
    for (int j = 0; j < num_take; ++j) {
      auto array_index = p_index[i * idx_slice_size + j];
      p_value[i * value_slice_size + j] =
          p_array[i * array_slice_size + array_index];
    }
  }
}

// UNDERSTAND: something like put_along_axis in numpy but if there is duplicate
// indices, scatter is done in += way.
template <typename T>
static void CPUPutAlongD1(const platform::DeviceContext& ctx,
                          framework::Tensor* array,
                          const framework::Tensor& index,
                          const framework::Tensor& value) {
  PADDLE_ENFORCE(platform::is_cpu_place(ctx.GetPlace()));
  // UNDERSTAND: check shape src(B, C), index(B, K), out should also be (B, K)
  PADDLE_ENFORCE(index.dims().size() == 2 && array->dims().size() == 2 &&
                 index.dims()[0] == array->dims()[0] &&
                 index.dims() == value.dims());
  const auto batch_size = index.dims()[0];
  const auto num_put = index.dims()[1];
  auto array_dims = array->dims();
  auto idx_dims = index.dims();

  // UNDERSTAND: no allocations here
  T* p_array = array->data<T>();
  const int64_t* p_index = index.data<int64_t>();
  const T* p_value = value.data<T>();

  // slice sizes
  const auto array_slice_size = array_dims[1];
  const auto idx_slice_size = idx_dims[1];
  const auto value_slice_size = idx_slice_size;

  for (int i = 0; i < batch_size; ++i) {
    for (int j = 0; j < num_put; ++j) {
      auto array_index = p_index[i * idx_slice_size + j];
      p_array[i * array_slice_size + array_index] +=
          p_value[i * value_slice_size + j];
    }
  }
}

// UNDERSTAND: compute accidentdal hits from samples and minus corresponding
// logits by a float max, here 1e20
template <typename T>
static void compute_remove_accidental_hits(const platform::DeviceContext& ctx,
                                           framework::Tensor* sampled_logits,
                                           const framework::Tensor& samples,
                                           const int num_true) {
  const auto batch_size = sampled_logits->dims()[0];
  const auto num_sampled_classes = sampled_logits->dims()[1];
  T* sampled_logits_data = sampled_logits->data<T>();
  const auto samples_data = samples.data<int64_t>();

  std::unordered_set<int64_t> tmp_true_labels;
  for (int i = 0; i < batch_size; ++i) {
    tmp_true_labels.clear();
    tmp_true_labels.insert(samples_data + i * num_sampled_classes,
                           samples_data + i * num_sampled_classes + num_true);
    for (int j = num_true; j < num_sampled_classes; ++j) {
      const auto idx = i * num_sampled_classes + j;
      if (tmp_true_labels.find(samples_data[idx]) != tmp_true_labels.end())
        sampled_logits_data[idx] -= 1e20;
    }
  }
}

template <typename T>
class SampledSoftmaxWithCrossEntropyKernel : public framework::OpKernel<T> {
 public:
  using Tensor = framework::Tensor;
  void Compute(const framework::ExecutionContext& context) const override {
    PADDLE_ENFORCE(platform::is_cpu_place(context.GetPlace()),
                   "This kernel only runs on CPU.");
    // get necessary inputs
    const Tensor* logits = context.Input<Tensor>("Logits");
    const Tensor* label = context.Input<Tensor>("Label");

    // get necessary outputs
    Tensor* samples = context.Output<Tensor>("Samples");
    Tensor* sampled_softmax = context.Output<Tensor>("SampledSoftmax");
    Tensor* loss = context.Output<Tensor>("Loss");

    // shapes
    const auto batch_size = logits->dims()[0];
    const auto num_classes = logits->dims()[1];
    const auto label_dim = label->dims();
    const auto num_true = label_dim[1];
    const auto samples_dim = samples->dims();

    // attrs
    const auto num_samples = context.Attr<int>("num_samples");
    const bool use_custom_samples = context.Attr<bool>("use_custom_samples");
    const bool remove_accidental_hits =
        context.Attr<bool>("remove_accidental_hits");

    // device contexts
    auto& dev_ctx =
        context.template device_context<platform::CPUDeviceContext>();

    // UNDERSTAND: allocate memories for temporaries
    Tensor probabilities_tmp;
    Tensor* probabilities = &probabilities_tmp;
    Tensor sampled_logits_tmp;
    Tensor* sampled_logits = &sampled_logits_tmp;
    sampled_logits->mutable_data<T>(samples_dim, context.GetPlace());
    Tensor shrinked_label_tmp;
    Tensor* shrinked_label = &shrinked_label_tmp;
    auto shrinked_lebel_data =
        shrinked_label->mutable_data<int64_t>(label_dim, context.GetPlace());
    for (int i = 0; i < batch_size; ++i)
      for (int j = 0; j < num_true; ++j)
        shrinked_lebel_data[i * num_true + j] = j;

    // UNDERSTAND: allocate memoroes for outputs
    sampled_softmax->mutable_data<T>(context.GetPlace());
    loss->mutable_data<T>(context.GetPlace());

    if (use_custom_samples) {
      const Tensor* custom_samples = context.Input<Tensor>("CustomSamples");
      const Tensor* custom_probabilities =
          context.Input<Tensor>("CustomProbabilities");
      samples->ShareDataWith(*custom_samples);
      probabilities->ShareDataWith(*custom_probabilities);
    } else {
      samples->mutable_data<int64_t>(context.GetPlace());
      probabilities->mutable_data<T>(samples_dim, context.GetPlace());
      // UNDERSTAND: sampling
      const auto seed = context.Attr<int>("seed");
      auto sampler_with_prob =
          math::SampleWithProb<platform::CPUDeviceContext, T>();
      sampler_with_prob(dev_ctx, math::LogUniformSampler(num_classes, seed),
                        num_samples, label, samples, probabilities);
    }

    // UNDERSTAND: gather sampled logits and remove accidental hits if needed
    CPUTakeAlongD1<T>(dev_ctx, *logits, *samples, sampled_logits);
    if (remove_accidental_hits) {
      compute_remove_accidental_hits<T>(dev_ctx, sampled_logits, *samples,
                                        num_true);
    }

    /* Debug
    const auto num_sampled_classes = samples_dim[1];
    std::cout << "Sampled Logits" << std::endl;
    const auto sampled_logits_data = sampled_logits->data<T>();
    for (int i = 0; i < sampled_logits->numel(); ++i) {
      std::cout << sampled_logits_data[i] << ", ";
      if ((i + 1) % num_sampled_classes == 0)
        std::cout << std::endl;
    }
    std::cout << std::endl;
    */
    /* Debug
    std::cout << "Samples" << std::endl;
    const auto samples_data = samples->data<int64_t>();
    for (int i = 0; i < samples->numel(); ++i) {
      std::cout << samples_data[i] << ", ";
      if ((i + 1) % num_sampled_classes == 0)
        std::cout << std::endl;
    }
    std::cout << std::endl;
    */
    /* Debug
    std::cout << "Probabilities" << std::endl;
    const auto probabilities_data = probabilities->data<T>();
    for (int i = 0; i < probabilities->numel(); ++i) {
      std::cout << probabilities_data[i] << ", ";
      if ((i + 1) % num_sampled_classes == 0)
        std::cout << std::endl;
    }
    std::cout << std::endl;
    */
    // subtracted sampled logits with logQ(y|x)
    auto probs = EigenMatrix<T>::From(*probabilities);
    auto smp_logits = EigenMatrix<T>::From(*sampled_logits);
    smp_logits.device(*dev_ctx.eigen_device()) =
        (smp_logits - probs.log().unaryExpr(math::TolerableValue<T>()))
            .unaryExpr(math::TolerableValue<T>());

    // UNDERSTAND: main logic here
    math::SoftmaxFunctor<platform::CPUDeviceContext, T, false>()(
        dev_ctx, sampled_logits, sampled_softmax);
    math::CrossEntropyMultiLabelFunctor<platform::CPUDeviceContext, T>()(
        dev_ctx, loss, sampled_softmax, shrinked_label);

    /* Debug
    std::cout << "Sampled Softmax" << std::endl;
    const auto sampled_softmax_data = sampled_softmax->data<T>();
    for (int i = 0; i < sampled_softmax->numel(); ++i) {
      std::cout << sampled_softmax_data[i] << ", ";
      if ((i + 1) % num_sampled_classes == 0)
        std::cout << std::endl;
    }
    std::cout << std::endl;
    */
  }
};

template <typename T>
class SampledSoftmaxWithCrossEntropyGradKernel : public framework::OpKernel<T> {
 public:
  using Tensor = framework::Tensor;
  void Compute(const framework::ExecutionContext& context) const override {
    const Tensor* out_grad =
        context.Input<Tensor>(framework::GradVarName("Loss"));
    const Tensor* label = context.Input<Tensor>("Label");
    const Tensor* samples = context.Input<Tensor>("Samples");
    auto logit_grad = context.Output<Tensor>(framework::GradVarName("Logits"));
    logit_grad->mutable_data<T>(context.GetPlace());

    auto& dev_ctx =
        context.template device_context<platform::CPUDeviceContext>();
    math::SetConstant<platform::CPUDeviceContext, T> set_zero;
    set_zero(dev_ctx, logit_grad, static_cast<T>(0));

    // shape
    const int batch_size = logit_grad->dims()[0];
    const int num_true = label->dims()[1];

    Tensor sampled_logits_grad_tmp;
    Tensor* sampled_logits_grad = &sampled_logits_grad_tmp;
    sampled_logits_grad->ShareDataWith(
        *context.Input<Tensor>("SampledSoftmax"));

    const int num_sampled_classes = sampled_logits_grad->dims()[1];
    auto out_grad_mat = EigenMatrix<T>::From(*out_grad);
    auto sampled_logits_grad_mat = EigenMatrix<T>::From(*sampled_logits_grad);
    auto& place = *context.template device_context<platform::CPUDeviceContext>()
                       .eigen_device();
    sampled_logits_grad_mat.device(place) =
        sampled_logits_grad_mat *
        out_grad_mat.broadcast(Eigen::DSizes<int, 2>(1, num_sampled_classes));

    T* sampled_logits_grad_data = sampled_logits_grad->data<T>();
    const T* out_grad_data = out_grad->data<T>();
    /* UNDERSTAND: still do it by loop */
    for (int i = 0; i < batch_size; ++i) {
      for (int j = 0; j < num_true; ++j) {
        sampled_logits_grad_data[i * num_sampled_classes + j] -=
            out_grad_data[i] / num_true;
      }
    }

    // UNDERSTAND: scatter it back to logit_grad
    CPUPutAlongD1<T>(dev_ctx, logit_grad, *samples, *sampled_logits_grad);
  }
};

}  // namespace operators
}  // namespace paddle
