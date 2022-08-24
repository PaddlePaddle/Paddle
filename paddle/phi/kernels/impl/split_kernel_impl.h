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

#include "paddle/fluid/operators/strided_memcpy.h"
#include "paddle/phi/common/int_array.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/infermeta/unary.h"
#include "paddle/phi/kernels/funcs/concat_and_split_functor.h"

namespace phi {

void SplitInferOutDims(const DenseTensor& x,
                       const IntArray& sections,
                       const Scalar& axis_scalar,
                       std::vector<DenseTensor*>* outs) {
  if (sections.FromTensor() || axis_scalar.FromTensor()) {
    std::vector<MetaTensor> out_metas;
    out_metas.reserve(outs->size());
    for (size_t i = 0; i < outs->size(); ++i) {
      out_metas.push_back((*outs)[i]);
    }

    // get axis_value
    int axis_value = GetSplitAxisValue(x, axis_scalar);

    auto input_axis_dim = x.dims().at(axis_value);
    auto sections_data = sections.GetData();
    // step1: get formated sections
    std::vector<int64_t> sections_vec;
    const int unknow_dim_val = -1;
    int unknow_dim_idx = -1;
    int num_of_unknow = 0;
    int sum_of_section = 0;

    for (size_t i = 0; i < sections_data.size(); ++i) {
      sections_vec.push_back(sections_data[i]);

      if (sections_data[i] == unknow_dim_val) {
        num_of_unknow++;
        unknow_dim_idx = i;
      } else {
        sum_of_section += sections_data[i];
      }
    }

    if (unknow_dim_idx != -1) {
      sections_vec[unknow_dim_idx] = input_axis_dim - sum_of_section;
    }
    // setp2: fill out dims
    std::vector<phi::DDim> out_dims(sections_vec.size(), x.dims());
    if (input_axis_dim > 0) {
      for (size_t i = 0; i < sections_vec.size(); ++i) {
        out_dims[i][axis_value] = sections_vec[i];
      }
    } else {
      for (size_t i = 0; i < sections_vec.size(); ++i) {
        out_dims[i][axis_value] = -1;
      }
    }

    for (size_t i = 0; i < sections_vec.size(); ++i) {
      if (axis_value != 0) {
        // Only pass LoD when not spliting along the first dim.
        out_metas[i].set_dtype(x.dtype());
        out_metas[i].set_dims(out_dims[i]);
        out_metas[i].set_layout(x.layout());
      } else {
        out_metas[i].set_dtype(x.dtype());
        out_metas[i].set_dims(out_dims[i]);
        out_metas[i].set_layout(x.layout());
        out_metas[i].share_lod(x);
      }
    }
  }
}

void SplitWithNumInferOutDims(const DenseTensor& x,
                              int num,
                              const Scalar& axis_scalar,
                              std::vector<DenseTensor*>* outs) {
  if (axis_scalar.FromTensor()) {
    std::vector<MetaTensor> out_metas;
    out_metas.reserve(outs->size());
    for (size_t i = 0; i < outs->size(); ++i) {
      out_metas.push_back((*outs)[i]);
    }
    int axis_value = GetSplitAxisValue(x, axis_scalar);
    auto input_axis_dim = x.dims().at(axis_value);
    // step1: get formated sections
    std::vector<int64_t> sections_vec;
    for (int i = 0; i < num; ++i) {
      sections_vec.push_back(input_axis_dim / num);
    }
    // setp2: fill out dims
    std::vector<phi::DDim> out_dims(sections_vec.size(), x.dims());
    if (input_axis_dim > 0) {
      for (size_t i = 0; i < sections_vec.size(); ++i) {
        out_dims[i][axis_value] = sections_vec[i];
      }
    } else {
      for (size_t i = 0; i < sections_vec.size(); ++i) {
        out_dims[i][axis_value] = -1;
      }
    }

    for (size_t i = 0; i < sections_vec.size(); ++i) {
      if (axis_value != 0) {
        // Only pass LoD when not spliting along the first dim.
        out_metas[i].set_dtype(x.dtype());
        out_metas[i].set_dims(out_dims[i]);
        out_metas[i].set_layout(x.layout());
      } else {
        out_metas[i].set_dtype(x.dtype());
        out_metas[i].set_dims(out_dims[i]);
        out_metas[i].set_layout(x.layout());
        out_metas[i].share_lod(x);
      }
    }
  }
}

template <typename T, typename Context>
void SplitKernel(const Context& dev_ctx,
                 const DenseTensor& x,
                 const IntArray& sections,
                 const Scalar& axis_scalar,
                 std::vector<DenseTensor*> outs) {
  // need to infershape output
  SplitInferOutDims(x, sections, axis_scalar, &outs);

  std::vector<const DenseTensor*> shape_refer;
  for (size_t j = 0; j < outs.size(); ++j) {
    dev_ctx.template Alloc<T>(outs[j]);
    shape_refer.emplace_back(outs[j]);
  }

  int axis = axis_scalar.to<int>();
  // Sometimes direct copies will be faster, this maybe need deeply analysis.
  if (axis == 0 && outs.size() < 10) {
    paddle::operators::StridedMemcpyWithAxis0<T>(
        dev_ctx, x, shape_refer, &outs);
  } else {
    phi::funcs::SplitFunctor<Context, T> functor;
    functor(dev_ctx, x, shape_refer, axis, &outs);
  }
}

template <typename T, typename Context>
void SplitWithNumKernel(const Context& dev_ctx,
                        const DenseTensor& x,
                        int num,
                        const Scalar& axis_scalar,
                        std::vector<DenseTensor*> outs) {
  // need to infershape output
  SplitWithNumInferOutDims(x, num, axis_scalar, &outs);

  std::vector<const DenseTensor*> shape_refer;
  for (size_t j = 0; j < outs.size(); ++j) {
    dev_ctx.template Alloc<T>(outs[j]);
    shape_refer.emplace_back(outs[j]);
  }

  int axis = axis_scalar.to<int>();
  // Sometimes direct copies will be faster, this maybe need deeply analysis.
  if (axis == 0 && outs.size() < 10) {
    paddle::operators::StridedMemcpyWithAxis0<T>(
        dev_ctx, x, shape_refer, &outs);
  } else {
    phi::funcs::SplitFunctor<Context, T> functor;
    functor(dev_ctx, x, shape_refer, axis, &outs);
  }
}

}  // namespace phi
