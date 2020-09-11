/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/segment_ops/segment_pooling.h"
#include <string>
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/math_function.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename T, typename IndexT>
class SegmentPoolFunctor<platform::CPUDeviceContext, T, IndexT> {
 public:
  void operator()(const platform::CPUDeviceContext& context,
                  const framework::Tensor& input,
                  const framework::Tensor& segments, framework::Tensor* output,
                  framework::Tensor* index,
                  const std::string pooltype = "SUM") {
    const IndexT* segment_ids = segments.data<IndexT>();
    auto curent_id = segment_ids[0];
    int64_t last_idx = 0;
    int64_t w = input.numel() / input.dims()[0];
    auto& place = *context.eigen_device();
    for (int64_t idx = 1; idx <= segments.numel(); ++idx) {
      if (idx < segments.numel()) {
        if (segment_ids[idx] == curent_id) continue;
      }

      Tensor out_t = output->Slice(curent_id, curent_id + 1);
      Tensor in_t = input.Slice(last_idx, idx);

      int64_t h = idx - last_idx;
      auto in_e =
          framework::EigenMatrix<T>::From(in_t, framework::make_ddim({h, w}));
      auto out_e = framework::EigenVector<T>::Flatten(out_t);

      auto reduce_dim = Eigen::array<int, 1>({{0}});
      if (pooltype == "MEAN") {
        out_e.device(place) = in_e.mean(reduce_dim);
      } else if (pooltype == "SUM") {
        out_e.device(place) = in_e.sum(reduce_dim);
      } else if (pooltype == "MAX") {
        out_e.device(place) = in_e.maximum(reduce_dim);
      } else if (pooltype == "MIN") {
        out_e.device(place) = in_e.minimum(reduce_dim);
      } else {
        PADDLE_THROW("unsupported pooling pooltype");
      }

      last_idx = idx;
      if (idx < segments.numel()) curent_id = segment_ids[idx];
    }
  }
};

// grad function for max, min operation
template <typename T>
class IndexPoolGradFunctor {
 public:
  void operator()(const platform::CPUDeviceContext& context,
                  const framework::Tensor& out_grad,
                  const framework::Tensor& index, framework::Tensor* in_grad) {
    auto og_dims = out_grad.dims();
    auto ig_dims = in_grad->dims();
    auto idx_dims = index.dims();
    PADDLE_ENFORCE_GT(og_dims.size(), 1,
                      "The rank of output@Grad shall be greater than 1.");
    PADDLE_ENFORCE_GT(ig_dims.size(), 1,
                      "The rank of input@Grad shall be greater than 1.");
    for (int64_t i = 1; i < og_dims.size(); ++i) {
      PADDLE_ENFORCE_EQ(
          og_dims[i], ig_dims[i],
          "The dimension of input@Grad and output@Grad shall be same.");
    }
    PADDLE_ENFORCE_EQ(idx_dims, og_dims,
                      "The dimension of index and output@Grad shall be same.");

    const T* og_data = out_grad.data<T>();
    const int* pos_index = index.data<int>();
    T* ig_data = in_grad->data<T>();

    math::SetConstant<platform::CPUDeviceContext, T> set_zero;
    set_zero(context, in_grad, static_cast<T>(0.0));
    int64_t num_seq = og_dims[0];
    int64_t dim = out_grad.numel() / num_seq;
    for (int64_t i = 0; i < num_seq; ++i) {
      for (int64_t j = 0; j < dim; ++j) {
        int step_id = pos_index[i * dim + j];
        if (step_id == -1) continue;
        ig_data[step_id * dim + j] = og_data[i * dim + j];
      }
    }
  }
};

template <typename T, typename IndexT>
class SegmentPoolGradFunctor<platform::CPUDeviceContext, T, IndexT> {
 public:
  void operator()(const platform::CPUDeviceContext& context,
                  const framework::Tensor& out_grad,
                  const framework::Tensor& segments, framework::Tensor* in_grad,
                  const framework::Tensor* index = nullptr,
                  const std::string pooltype = "SUM") {
    if (pooltype == "MAX" || pooltype == "MIN") {
      IndexPoolGradFunctor<T> index_pool_grad;
      index_pool_grad(context, out_grad, *index, in_grad);
      return;
    }
    const IndexT* segment_ids = segments.data<IndexT>();
    auto& place = *context.eigen_device();
    auto curent_id = segment_ids[0];
    int64_t last_idx = 0;
    int64_t w = in_grad->numel() / in_grad->dims()[0];
    for (int64_t idx = 1; idx <= segments.numel(); ++idx) {
      if (idx < segments.numel()) {
        if (segment_ids[idx] == curent_id) continue;
      }

      Tensor out_g_t = out_grad.Slice(curent_id, curent_id + 1);
      Tensor in_g_t = in_grad->Slice(last_idx, idx);

      int64_t h = idx - last_idx;
      auto in_g_e = framework::EigenMatrix<T>::From(in_g_t, {h, w});
      auto out_g_e = framework::EigenMatrix<T>::From(out_g_t, {1, w});
      // auto out_g_e_v = framework::EigenVector<T>::Flatten(out_g_t);
      Eigen::DSizes<int, 2> bcast(h, 1);

      if (pooltype == "MEAN") {
        in_g_e.device(place) = (out_g_e / static_cast<T>(h)).broadcast(bcast);
      } else if (pooltype == "SUM") {
        in_g_e.device(place) = out_g_e.broadcast(bcast);
      } else {
        PADDLE_THROW(
            "unsupported segment pooling operation, only MEAN, SUM, MAX, MIN "
            "available.");
      }

      last_idx = idx;
      if (idx < segments.numel()) curent_id = segment_ids[idx];
    }
  }
};

template class SegmentPoolFunctor<platform::CPUDeviceContext, float, int>;
template class SegmentPoolFunctor<platform::CPUDeviceContext, float, int64_t>;
template class SegmentPoolFunctor<platform::CPUDeviceContext, double, int>;
template class SegmentPoolFunctor<platform::CPUDeviceContext, double, int64_t>;
template class SegmentPoolGradFunctor<platform::CPUDeviceContext, float, int>;
template class SegmentPoolGradFunctor<platform::CPUDeviceContext, float,
                                      int64_t>;
template class SegmentPoolGradFunctor<platform::CPUDeviceContext, double, int>;
template class SegmentPoolGradFunctor<platform::CPUDeviceContext, double,
                                      int64_t>;

}  // namespace operators
}  // namespace paddle
