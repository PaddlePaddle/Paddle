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

namespace phi::funcs {

template <typename T, typename IndexT>
class SegmentPoolFunctor<CPUContext, T, IndexT> {
 public:
  void operator()(const CPUContext& dev_ctx,
                  const DenseTensor& input,
                  const DenseTensor& segments,
                  DenseTensor* output,
                  DenseTensor* index UNUSED,
                  const std::string pooltype = "SUM") {
    const IndexT* segment_ids = segments.data<IndexT>();

    // Handle single element segment_ids (broadcast case)
    // If segment_ids has only 1 element, all input rows belong to segment 0
    if (segments.numel() == 1) {
      int64_t n = input.dims()[0];
      int64_t w = input.numel() / n;
      auto& place = *dev_ctx.eigen_device();

      // The output should have shape [1, w] (one segment)
      // Actually, for a single segment_id, it means all rows belong to one
      // segment So output[0] should be the pooled value across all rows
      auto in_e = EigenMatrix<T>::From(input);

      DenseTensor out_t = output->Slice(0, 1);
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
      }
      return;
    }

    // Original logic for multiple segment_ids
    auto current_id = segment_ids[0];
    int64_t last_idx = 0;
    int64_t w = input.numel() / input.dims()[0];
    auto& place = *dev_ctx.eigen_device();
    for (int64_t idx = 1; idx <= segments.numel(); ++idx) {
      if (idx < segments.numel()) {
        if (segment_ids[idx] == current_id) continue;
        PADDLE_ENFORCE_GE(segment_ids[idx],
                          current_id,
                          common::errors::InvalidArgument(
                              "The segment ids should be sorted, but got "
                              "segment_ids[%d]:%d > segment_ids[%d]:%d.",
                              idx - 1,
                              current_id,
                              idx,
                              segment_ids[idx]));
      }

      DenseTensor out_t = output->Slice(current_id, current_id + 1);
      DenseTensor in_t = input.Slice(last_idx, idx);

      int64_t h = idx - last_idx;
      auto in_e = EigenMatrix<T>::From(in_t, make_ddim({h, w}));
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
        PADDLE_THROW(common::errors::InvalidArgument(
            "Unsupported segment pooling type, only MEAN, SUM, MAX, MIN "
            "available, but got %s.",
            pooltype));
      }

      last_idx = idx;
      if (idx < segments.numel()) current_id = segment_ids[idx];
    }
  }
};

}  // namespace phi::funcs
