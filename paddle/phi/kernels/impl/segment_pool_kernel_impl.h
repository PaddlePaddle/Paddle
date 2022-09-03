// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include <string>

#include "paddle/fluid/framework/eigen.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/funcs/segment_pooling.h"

namespace phi {

template <typename Context, typename T, typename IndexT>
void SegmentKernelLaunchHelper(const Context& dev_ctx,
                               const DenseTensor& x,
                               const DenseTensor& segment_ids,
                               const std::string& pooltype,
                               DenseTensor* out,
                               DenseTensor* summed_ids) {
  int64_t num_indices = segment_ids.numel();
  PADDLE_ENFORCE_EQ(
      num_indices,
      x.dims()[0],
      phi::errors::InvalidArgument(
          "Segment_ids should be the same size as dimension 0 of input X."));
  PADDLE_ENFORCE_EQ(num_indices,
                    segment_ids.dims()[0],
                    phi::errors::InvalidArgument(
                        "Segment_ids should be 1-D tensor, or it's other "
                        "dimension size is 1. Segment_ids's shape is: [%s].",
                        segment_ids.dims()));

  if (x.numel() == 0 || segment_ids.numel() == 0) {
    return;
  }

  bool cpu_place = dev_ctx.GetPlace().GetType() == phi::AllocationType::CPU;
  if (cpu_place) {
    auto dims = x.dims();
    auto* segment_ids_ptr = segment_ids.data<IndexT>();
    dims[0] =
        static_cast<int64_t>(segment_ids_ptr[segment_ids.numel() - 1] + 1);
    PADDLE_ENFORCE_GT(
        dims[0],
        0,
        phi::errors::InvalidArgument(
            "Segment ids must be >= 0, but got last id %d", dims[0]));

    out->Resize({dims});
    dev_ctx.template Alloc<T>(out);

    phi::funcs::SetConstant<Context, T> set_zero;
    set_zero(dev_ctx, out, static_cast<T>(0));
  }
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  if (!cpu_place) {
    DenseTensor length;
    length.Resize(phi::make_ddim({1}));
    IndexT* length_data = dev_ctx.template HostAlloc<IndexT>(&length);

    const IndexT* segment_ids_ptr = segment_ids.data<IndexT>();

#ifdef PADDLE_WITH_HIP
    PADDLE_ENFORCE_GPU_SUCCESS(hipMemcpy(length_data,
                                         segment_ids_ptr + num_indices - 1,
                                         sizeof(IndexT),
                                         hipMemcpyDeviceToHost));
#else
    PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpy(length_data,
                                          segment_ids_ptr + num_indices - 1,
                                          sizeof(IndexT),
                                          cudaMemcpyDeviceToHost));
#endif

    IndexT length_host = length_data[0];
    length_host++;
    PADDLE_ENFORCE_GT(
        length_host,
        0,
        phi::errors::InvalidArgument(
            "Segment ids must be >= 0, but got last id %d", length_data[0]));
    auto dims = x.dims();
    dims[0] = static_cast<int64_t>(length_host);
    out->Resize({dims});
    dev_ctx.template Alloc<T>(out);

    T init_value = static_cast<T>(0);
    if (pooltype == "MAX") {
      init_value = static_cast<T>(-FLT_MAX);
    } else if (pooltype == "MIN") {
      init_value = static_cast<T>(FLT_MAX);
    }
    phi::funcs::SetConstant<Context, T> setconst;
    setconst(dev_ctx, out, static_cast<T>(init_value));
    // the gpu kernel of mean pool record the counts of segment_ids
    if (pooltype == "MEAN") {
      summed_ids->Resize({dims[0], 1});
      dev_ctx.template Alloc<T>(summed_ids);
      setconst(dev_ctx, summed_ids, static_cast<T>(1e-12));
    }
  }
#endif

  phi::funcs::SegmentPoolFunctor<Context, T, IndexT> pool;

  pool(dev_ctx, x, segment_ids, out, summed_ids, pooltype);
}

template <typename T, typename Context>
void SegmentPoolKernel(const Context& dev_ctx,
                       const DenseTensor& x,
                       const DenseTensor& segment_ids,
                       const std::string& pooltype,
                       DenseTensor* out,
                       DenseTensor* summed_ids) {
  auto index_type = segment_ids.dtype();
  if (index_type == DataType::INT32) {
    SegmentKernelLaunchHelper<Context, T, int>(
        dev_ctx, x, segment_ids, pooltype, out, summed_ids);
  } else if (index_type == DataType::INT64) {
    SegmentKernelLaunchHelper<Context, T, int64_t>(
        dev_ctx, x, segment_ids, pooltype, out, summed_ids);
  } else {
    PADDLE_THROW(phi::errors::InvalidArgument(
        "Unsupported index type, Expected int, int64, but got %s.",
        index_type));
  }
}

}  // namespace phi
