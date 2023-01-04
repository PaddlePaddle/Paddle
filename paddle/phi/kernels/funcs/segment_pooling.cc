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

#include "paddle/phi/kernels/funcs/segment_pooling.h"

#include <string>

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"

namespace phi {
namespace funcs {

using Tensor = DenseTensor;

template <typename T, typename IndexT>
class SegmentPoolFunctor<phi::CPUContext, T, IndexT> {
 public:
  void operator()(const phi::CPUContext& dev_ctx,
                  const DenseTensor& input,
                  const DenseTensor& segments,
                  DenseTensor* output,
                  DenseTensor* index,
                  const std::string pooltype = "SUM") {
    const IndexT* segment_ids = segments.data<IndexT>();
    auto curent_id = segment_ids[0];
    int64_t last_idx = 0;
    int64_t w = input.numel() / input.dims()[0];
    auto& place = *dev_ctx.eigen_device();
    for (int64_t idx = 1; idx <= segments.numel(); ++idx) {
      if (idx < segments.numel()) {
        if (segment_ids[idx] == curent_id) continue;
        PADDLE_ENFORCE_GE(segment_ids[idx],
                          curent_id,
                          phi::errors::InvalidArgument(
                              "The segment ids should be sorted, but got "
                              "segment_ids[%d]:%d > segment_ids[%d]:%d.",
                              idx - 1,
                              curent_id,
                              idx,
                              segment_ids[idx]));
      }

      Tensor out_t = output->Slice(curent_id, curent_id + 1);
      Tensor in_t = input.Slice(last_idx, idx);

      int64_t h = idx - last_idx;
      auto in_e = EigenMatrix<T>::From(in_t, phi::make_ddim({h, w}));
      auto out_e = EigenVector<T>::Flatten(out_t);

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
        PADDLE_THROW(phi::errors::InvalidArgument(
            "Unsupported segment pooling type, only MEAN, SUM, MAX, MIN "
            "available, but got %s.",
            pooltype));
      }

      last_idx = idx;
      if (idx < segments.numel()) curent_id = segment_ids[idx];
    }
  }
};

template <typename T, typename IndexT>
class SegmentPoolGradFunctor<phi::CPUContext, T, IndexT> {
 public:
  void operator()(const phi::CPUContext& dev_ctx,
                  const DenseTensor& input,
                  const DenseTensor& output,
                  const DenseTensor& out_grad,
                  const DenseTensor& segments,
                  DenseTensor* in_grad,
                  const paddle::optional<DenseTensor>& index,
                  const std::string pooltype = "SUM") {
    const IndexT* segment_ids = segments.data<IndexT>();
    auto& place = *dev_ctx.eigen_device();
    auto curent_id = segment_ids[0];
    int64_t last_idx = 0;
    int64_t w = in_grad->numel() / in_grad->dims()[0];
    for (int64_t idx = 1; idx <= segments.numel(); ++idx) {
      if (idx < segments.numel()) {
        if (segment_ids[idx] == curent_id) continue;
        PADDLE_ENFORCE_GE(segment_ids[idx],
                          curent_id,
                          phi::errors::InvalidArgument(
                              "The segment ids should be sorted, but got "
                              "segment_ids[%d]:%d > segment_ids[%d]:%d.",
                              idx - 1,
                              curent_id,
                              idx,
                              segment_ids[idx]));
      }

      Tensor out_g_t = out_grad.Slice(curent_id, curent_id + 1);
      Tensor in_g_t = in_grad->Slice(last_idx, idx);

      int64_t h = idx - last_idx;
      auto in_g_e = EigenMatrix<T>::From(in_g_t, {h, w});
      auto out_g_e = EigenMatrix<T>::From(out_g_t, {1, w});
      Eigen::DSizes<int, 2> bcast(h, 1);

      if (pooltype == "MEAN") {
        in_g_e.device(place) = (out_g_e / static_cast<T>(h)).broadcast(bcast);
      } else if (pooltype == "SUM") {
        in_g_e.device(place) = out_g_e.broadcast(bcast);
      } else if (pooltype == "MAX" || pooltype == "MIN") {
        Tensor out_t = output.Slice(curent_id, curent_id + 1);
        Tensor in_t = input.Slice(last_idx, idx);
        auto in_e = EigenMatrix<T>::From(in_t, {h, w});
        auto out_e = EigenMatrix<T>::From(out_t, {1, w});
        in_g_e.device(place) =
            (in_e == out_e.broadcast(bcast)).template cast<T>() *
            out_g_e.broadcast(bcast);
      } else {
        PADDLE_THROW(phi::errors::InvalidArgument(
            "Unsupported segment pooling type, only MEAN, SUM, MAX, MIN "
            "available, but got %s.",
            pooltype));
      }

      last_idx = idx;
      if (idx < segments.numel()) curent_id = segment_ids[idx];
    }
  }
};

using CPU = phi::CPUContext;
using float16 = phi::dtype::float16;
template class SegmentPoolFunctor<CPU, float, int>;
template class SegmentPoolFunctor<CPU, float, int64_t>;
template class SegmentPoolFunctor<CPU, double, int>;
template class SegmentPoolFunctor<CPU, double, int64_t>;
template class SegmentPoolFunctor<CPU, int, int>;
template class SegmentPoolFunctor<CPU, int, int64_t>;
template class SegmentPoolFunctor<CPU, int64_t, int>;
template class SegmentPoolFunctor<CPU, int64_t, int64_t>;
template class SegmentPoolFunctor<CPU, float16, int>;
template class SegmentPoolFunctor<CPU, float16, int64_t>;

template class SegmentPoolGradFunctor<CPU, float, int>;
template class SegmentPoolGradFunctor<CPU, float, int64_t>;
template class SegmentPoolGradFunctor<CPU, double, int>;
template class SegmentPoolGradFunctor<CPU, double, int64_t>;
template class SegmentPoolGradFunctor<CPU, int, int>;
template class SegmentPoolGradFunctor<CPU, int, int64_t>;
template class SegmentPoolGradFunctor<CPU, int64_t, int>;
template class SegmentPoolGradFunctor<CPU, int64_t, int64_t>;
template class SegmentPoolGradFunctor<CPU, float16, int>;
template class SegmentPoolGradFunctor<CPU, float16, int64_t>;

}  // namespace funcs
}  // namespace phi
