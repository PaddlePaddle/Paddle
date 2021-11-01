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

#include "paddle/pten/kernels/cpu/manipulation.h"
#include "paddle/fluid/memory/memcpy.h"
#include "paddle/fluid/operators/strided_memcpy.h"
#include "paddle/pten/infershape/unary.h"
#include "paddle/pten/kernels/cpu/split_impl.h"
#include "paddle/pten/kernels/cpu/utils.h"
#include "paddle/pten/kernels/functions/lod_utils.h"
#include "paddle/pten/kernels/functions/utils.h"

namespace pten {

template <typename T>
void Flatten(const CPUContext& dev_ctx,
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
void FlattenWithXShape(const CPUContext& dev_ctx,
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
void Concat(const CPUContext& dev_ctx,
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
    std::vector<const DenseTensor*> input;
    for (size_t j = 0; j < x.size(); ++j) {
      if (x[j].numel() > 0) {
        input.push_back(&x[j]);
      }
    }

    size_t num = input.size();

    int64_t rows = 1;
    auto dim_0 = input[0]->dims();
    for (int i = 0; i < axis; ++i) {
      rows *= dim_0[i];
    }
    int64_t out_rows = rows, out_cols = 0;

    std::vector<int64_t> input_cols(input.size());
    for (size_t i = 0; i < num; ++i) {
      int64_t t_cols = input[i]->numel() / rows;
      out_cols += t_cols;
      input_cols[i] = t_cols;
    }
    auto cpu_place = dev_ctx.GetPlace();

    // computation
    auto output_data = out->mutable_data<T>();
    int64_t col_idx = 0;
    for (size_t j = 0; j < num; ++j) {
      int64_t col_len = input_cols[j];
      auto input_data = input[j]->data<T>();
      for (int64_t k = 0; k < out_rows; ++k) {
        paddle::memory::Copy(
            BOOST_GET_CONST(paddle::platform::CPUPlace, cpu_place),
            output_data + k * out_cols + col_idx,
            BOOST_GET_CONST(paddle::platform::CPUPlace, cpu_place),
            input_data + k * col_len,
            sizeof(T) * col_len);
      }
      col_idx += col_len;
    }
  }
}

template <typename T>
void ConcatAxisTensor(const CPUContext& dev_ctx,
                      const std::vector<DenseTensor>& x,
                      const DenseTensor& axis_tensor,
                      DenseTensor* out) {
  int axis = pten::GetDataFromTensor<int>(dev_ctx, &axis_tensor)[0];

  const size_t n = x.size();
  std::vector<DDim> ins_dims(n);
  for (size_t i = 0; i < n; ++i) {
    ins_dims[i] = x[i].dims();
  }

  DDim out_dims = pten::ComputeAndCheckShape(ins_dims, axis);
  out->Resize(out_dims);

  Concat<T>(dev_ctx, x, axis, out);
}

template <typename T>
void Split(const CPUContext& dev_ctx,
           const DenseTensor& x,
           const std::vector<int>& sections,
           int num,
           int axis,
           std::vector<DenseTensor*> out) {
  std::vector<const DenseTensor*> shape_refer;

  for (size_t j = 0; j < out.size(); ++j) {
    out[j]->mutable_data<T>();
    shape_refer.emplace_back(out[j]);
  }

  if (axis == 0 && out.size() < 10) {
    pten::StridedMemcpyWithAxis0<T>(dev_ctx, x, shape_refer, &out);
  } else {
    pten::detail::SplitImpl<T>(dev_ctx, x, shape_refer, axis, &out);
  }
}

}  // namespace pten

// TODO(chenweihang): replace by better impl
PT_REGISTER_MODULE(ManipulationCPU);

// TODO(yuanrisheng): "flatten_contiguous_range" is compatible with old kernel
// architecture, kernel_name should be "flatten".
PT_REGISTER_KERNEL("flatten_contiguous_range",
                   CPU,
                   ANY,
                   pten::Flatten,
                   float,
                   double,
                   uint8_t,
                   int8_t,
                   int,
                   int64_t) {}

PT_REGISTER_KERNEL("flatten_contiguous_range.mid",
                   CPU,
                   ANY,
                   pten::FlattenWithXShape,
                   float,
                   double,
                   uint8_t,
                   int8_t,
                   int,
                   int64_t) {}

PT_REGISTER_KERNEL("concat",
                   CPU,
                   ANY,
                   pten::Concat,
                   float,
                   double,
                   bool,
                   int64_t,
                   int,
                   uint8_t) {}

PT_REGISTER_KERNEL("concat.axisTensor",
                   CPU,
                   ANY,
                   pten::ConcatAxisTensor,
                   float,
                   double,
                   bool,
                   int64_t,
                   int,
                   uint8_t) {
  kernel->InputAt(1).SetBackend(pten::Backend::CPU);
}

PT_REGISTER_KERNEL("split",
                   CPU,
                   ANY,
                   pten::Split,
                   float,
                   double,
                   bool,
                   int64_t,
                   int,
                   uint8_t) {}
