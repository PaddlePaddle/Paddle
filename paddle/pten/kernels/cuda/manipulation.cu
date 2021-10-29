//   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/memory/memcpy.h"
#include "paddle/fluid/operators/strided_memcpy.h"
#include "paddle/pten/infershape/unary.h"
#include "paddle/pten/kernels/cuda/manipulation.h"
#include "paddle/pten/kernels/cuda/utils.h"
#include "paddle/pten/kernels/functions/lod_utils.h"

#include "paddle/pten/kernels/cuda/concat_impl.h"

namespace pten {

template <typename T>
void Flatten(const CUDAContext& dev_ctx,
             const DenseTensor& x,
             int start_axis,
             int stop_axis,
             DenseTensor* out) {
  auto out_meta = FlattenInferShape(x.meta(), start_axis, stop_axis);
  pten::Copy(dev_ctx, x, out);
  out->set_lod(out_meta.lod);
  out->Resize(out_meta.dims);
}

// TODO(yuanrisheng): this kernel is for training and xshape is a Intermediate
// Output Tensor，
// is there a more flexible way to deal with this case?
template <typename T>
void FlattenWithXShape(const CUDAContext& dev_ctx,
                       const DenseTensor& x,
                       int start_axis,
                       int stop_axis,
                       DenseTensor* out,
                       DenseTensor* xshape) {
  Flatten<T>(dev_ctx, x, start_axis, stop_axis, out);
  const auto& in_dims = x.meta().dims;
  std::vector<int64_t> xshape_dims(in_dims.size() + 1);
  xshape_dims[0] = 0;
  for (int i = 0; i < in_dims.size(); ++i) {
    xshape_dims[i + 1] = in_dims[i];
  }
  xshape->Resize(paddle::framework::make_ddim(xshape_dims));
  xshape->set_lod(x.lod());
}

template <typename T>
void Concat(const CUDAContext& dev_ctx,
            const std::vector<DenseTensor>& x,
            int axis,
            DenseTensor* out) {
  axis = pten::ComputeAxis(axis, x[0].dims().size());

  // If axis is 0, the lod of the output is not the same as inputs.
  if (axis == 0 && x[0].lod().size() > 0) {
    size_t lod_size_0 = x[0].lod().size();
    size_t lod_size = lod_size_0;
    for (size_t i = 1; i < x.size(); ++i) {
      if (x[i].lod().size() > 0) {
        PADDLE_ENFORCE_EQ(
            x[i].lod().size(),
            lod_size_0,
            paddle::platform::errors::Unimplemented(
                "The lod level of all input LoDTensors should be same. "
                "Maybe different lod level of input LoDTensors can concat,"
                "it is not supported currently. The lod level of %dth input "
                "is %d and first input is %d.",
                i,
                x[i].lod().size(),
                lod_size_0));
      } else {
        lod_size = 0;
        break;
      }
    }
    if (lod_size) {
      auto out_lod = out->lod();
      for (size_t i = 1; i < x.size(); ++i) {
        auto in_lod = pten::ConvertToLengthBasedLoD(x[i].lod());
        pten::AppendLoD(&out_lod, in_lod);
      }
    }
  }

  // Sometimes direct copies will be faster, this maybe need deeply analysis.
  if (axis == 0 && x.size() < 10) {
    size_t output_offset = 0;
    for (auto& in : x) {
      if (in.numel() == 0UL) {
        continue;
      }

      auto in_stride = paddle::framework::stride_numel(in.dims());
      auto out_stride = paddle::framework::stride_numel(out->dims());
      paddle::operators::StridedNumelCopyWithAxis<T>(
          dev_ctx,
          axis,
          out->mutable_data<T>() + output_offset,
          out_stride,
          in.data<T>(),
          in_stride,
          in_stride[axis]);
      output_offset += in_stride[axis];
    }
  } else {
    // Note(chentianyu03): Old kernel will filter the numel()>0 tensor here.
    // because DensorTensor does not support copy constructor, we try to filter
    // in ConcatImpl.
    pten::detail::ConcatImpl<T>(dev_ctx, x, axis, out);
  }
}

}  // namespace pten

// TODO(chenweihang): replace by better impl
PT_REGISTER_MODULE(ManipulationCUDA);

using float16 = paddle::platform::float16;
// TODO(yuanrisheng): "flatten_contiguous_range" is compatible with old kernel
// architecture, kernel_name should be "flatten".
PT_REGISTER_KERNEL("flatten_contiguous_range",
                   CUDA,
                   ANY,
                   pten::Flatten,
                   float,
                   float16,
                   double,
                   uint8_t,
                   int8_t,
                   int,
                   int64_t) {}

PT_REGISTER_KERNEL("flatten_contiguous_range.mid",
                   CUDA,
                   ANY,
                   pten::FlattenWithXShape,
                   float,
                   double,
                   uint8_t,
                   int8_t,
                   int,
                   int64_t) {}

PT_REGISTER_KERNEL("concat",
                   CUDA,
                   ANY,
                   pten::Concat,
                   float,
                   double,
                   bool,
                   int64_t,
                   int,
                   uint8_t) {}
