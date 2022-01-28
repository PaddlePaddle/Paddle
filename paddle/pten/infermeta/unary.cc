/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/pten/infermeta/unary.h"

#include <set>

#include "paddle/pten/core/infermeta_utils.h"

namespace pten {

void UnchangedInferMetaNew(MetaConfig config,
                           const MetaTensor& x,
                           MetaTensor* out) {
  out->set_dims(x.dims());
  out->share_lod(x);
}

DenseTensorMeta UnchangedInferMeta(const DenseTensorMeta& x_meta) {
  return x_meta;
}

DenseTensorMeta ReductionInferMeta(const DenseTensorMeta& x_meta) {
  const auto& out_dims = pten::framework::make_ddim({1});
  DenseTensorMeta return_meta(x_meta.dtype, out_dims, x_meta.layout);
  return return_meta;
}

DenseTensorMeta FlattenInferMeta(const DenseTensorMeta& x_meta,
                                 int start_axis,
                                 int stop_axis) {
  auto& x_dims = x_meta.dims;
  int in_dims_size = x_dims.size();
  if (start_axis < 0) {
    start_axis = start_axis + in_dims_size;
  }
  if (stop_axis < 0) {
    stop_axis = stop_axis + in_dims_size;
  }
  PADDLE_ENFORCE_GE(stop_axis,
                    start_axis,
                    paddle::platform::errors::InvalidArgument(
                        "The stop_axis should be greater"
                        "than or equal to start_axis."));

  int64_t outer = 1;
  std::vector<int32_t> out_shape;
  out_shape.reserve(in_dims_size - stop_axis + start_axis);

  for (int i = 0; i < start_axis; ++i) {
    out_shape.push_back(x_dims[i]);
  }
  for (int i = start_axis; i <= stop_axis; i++) {
    if (x_dims[i] == -1 || outer == -1) {
      outer = -1;
    } else {
      outer *= x_dims[i];
    }
  }
  out_shape.push_back(outer);
  for (int i = stop_axis + 1; i < in_dims_size; i++) {
    out_shape.push_back(x_dims[i]);
  }
  const auto& out_dims = pten::framework::make_ddim(out_shape);
  DenseTensorMeta return_meta(x_meta.dtype, out_dims, x_meta.layout);

  if (x_dims[0] == return_meta.dims[0]) {
    // Only pass LoD when the first dimension of output and Input(X)
    // are the same.
    return_meta.lod = x_meta.lod;
  }

  return return_meta;
}

DenseTensorMeta CastInferMeta(const DenseTensorMeta& x_meta,
                              const DataType out_dtype) {
  DenseTensorMeta out_meta(out_dtype, x_meta.dims, x_meta.layout);
  return out_meta;
}

DenseTensorMeta CreateLikeInferMeta(const DenseTensorMeta& x_meta,
                                    DataType dtype,
                                    DataLayout layout) {
  return {dtype == DataType::UNDEFINED ? x_meta.dtype : dtype,
          x_meta.dims,
          layout == DataLayout::UNDEFINED ? x_meta.layout : layout};
}

static pten::framework::DDim ValidateShape(
    const std::vector<int64_t> shape, const pten::framework::DDim& in_dims) {
  const int64_t in_size = pten::framework::product(in_dims);
  auto in_dims_vec = pten::framework::vectorize(in_dims);
  bool all_positive = std::all_of(in_dims_vec.cbegin(),
                                  in_dims_vec.cend(),
                                  [](int64_t i) { return i > 0; });
  // only one dimension can be set to -1, whose size will be automatically
  // infered.
  const int64_t unk_dim_val = -1;
  const int64_t copy_dim_val = 0;

  std::vector<int64_t> output_shape(shape.size(), 0);
  int64_t capacity = 1;
  int unk_dim_idx = -1;
  for (size_t i = 0; i < shape.size(); ++i) {
    if (shape[i] == unk_dim_val) {
      PADDLE_ENFORCE_EQ(
          unk_dim_idx,
          -1,
          paddle::platform::errors::InvalidArgument(
              "Only one dimension value of 'shape' in ReshapeOp can "
              "be -1. But received shape = [%s], shape[%d] is also -1.",
              pten::framework::make_ddim(shape),
              i));
      unk_dim_idx = i;
    } else if (shape[i] == copy_dim_val) {
      PADDLE_ENFORCE_LT(
          static_cast<int>(i),
          in_dims.size(),
          paddle::platform::errors::InvalidArgument(
              "The index of 0 in `shape` must be less than "
              "the input tensor X's dimensions. "
              "But received shape = [%s], shape[%d] = 0, X's shape = [%s], "
              "X's dimensions = %d.",
              pten::framework::make_ddim(shape),
              i,
              in_dims,
              in_dims.size()));
    } else {
      PADDLE_ENFORCE_GT(
          shape[i],
          0,
          paddle::platform::errors::InvalidArgument(
              "Each dimension value of 'shape' in ReshapeOp must not "
              "be negative except one unknown dimension. "
              "But received  shape = [%s], shape[%d] = %d.",
              pten::framework::make_ddim(shape),
              i,
              shape[i]));
    }

    // NOTE all non-zero values will be converted to True (include negative
    // value)
    capacity *= (shape[i] ? shape[i] : in_dims[i]);
    output_shape[i] = (shape[i] ? static_cast<int64_t>(shape[i]) : in_dims[i]);
  }

  if (unk_dim_idx != -1) {
    if (all_positive) {
      // in_size < 0 and is un-determinate in compile time, skip the check,
      // for example, in_dims = [-1, 8, 1, 1], shape = [-1, 3, 8],
      // capacity = -24, in_size = -8, output_shape[0] = 0
      // the following check will fail.
      output_shape[unk_dim_idx] = -in_size / capacity;
      PADDLE_ENFORCE_EQ(
          output_shape[unk_dim_idx] * capacity,
          -in_size,
          paddle::platform::errors::InvalidArgument(
              "The 'shape' attribute in ReshapeOp is invalid. "
              "The input tensor X'size must be divisible by known "
              "capacity of 'shape'. "
              "But received X's shape = [%s], X's size = %d, "
              "'shape' is [%s], known capacity of 'shape' is %d.",
              in_dims,
              in_size,
              pten::framework::make_ddim(shape),
              capacity));
    } else {
      output_shape[unk_dim_idx] = -1;
    }
  } else {
    if (all_positive) {
      PADDLE_ENFORCE_EQ(
          capacity,
          in_size,
          paddle::platform::errors::InvalidArgument(
              "The 'shape' in ReshapeOp is invalid. "
              "The input tensor X'size must be equal to the capacity of "
              "'shape'. "
              "But received X's shape = [%s], X's size = %d, 'shape' is "
              "[%s], the capacity of 'shape' is %d.",
              in_dims,
              in_size,
              pten::framework::make_ddim(shape),
              capacity));
    }
  }

  // support reshape with zero-input(input tensor with product(shape) == 0)
  // by now we require that if the input tensor is zero shape, the target
  // shape of output must be zero
  if (in_size == 0) {
    PADDLE_ENFORCE_LE(
        capacity,
        in_size,
        paddle::platform::errors::InvalidArgument(
            "The 'shape' in ReshapeOp is invalid. "
            "The input tensor X's shape = [%s], X's capacity = %d."
            "But the target shape of Out is [%s],  the "
            "capacity of 'Out' is %d.",
            in_dims,
            in_size,
            pten::framework::make_ddim(shape),
            capacity));
  }

  return pten::framework::make_ddim(output_shape);
}

DenseTensorMeta InferMetaFromVecValue(const DenseTensorMeta& x_meta,
                                      const std::vector<int64_t>& shape) {
  PADDLE_ENFORCE_EQ(!shape.empty(),
                    true,
                    paddle::platform::errors::InvalidArgument(
                        "The parameter 'shape' in ReshapeOp must be set. "
                        "But received 'shape' is empty."));
  auto x_dims = x_meta.dims;
  auto out_dims = ValidateShape(shape, x_dims);
  DenseTensorMeta return_meta(x_meta.dtype, out_dims, x_meta.layout);
  if (x_dims[0] == return_meta.dims[0]) {
    // Only pass LoD when the first dimension of output and Input(X)
    // are the same.
    return_meta.lod = x_meta.lod;
  }
  return return_meta;
}

DenseTensorMeta ReshapeInferMeta(const DenseTensorMeta& x_meta,
                                 const ScalarArray& shape) {
  return InferMetaFromVecValue(x_meta, shape.GetData());
}

/*  Why not use ReduceInferMeta directly?
    Because we need make InferMetaFunction's args follow the design of api.yaml
*/
DenseTensorMeta SumInferMeta(const DenseTensorMeta& x_meta,
                             const std::vector<int64_t>& axis,
                             DataType dtype,
                             bool keep_dim) {
  return ReduceInferMeta(x_meta, axis, keep_dim, dtype);
}

DenseTensorMeta ReduceInferMeta(const DenseTensorMeta& x_meta,
                                const std::vector<int64_t>& axis,
                                bool keep_dim,
                                DataType dtype) {
  bool reduce_all = true;
  std::set<int64_t> dims_set(axis.begin(), axis.end());
  for (int64_t i = 0; i < x_meta.dims.size(); ++i) {
    if (dims_set.find(i) == dims_set.end()) {
      reduce_all = false;
      break;
    }
  }

  std::vector<int64_t> out_dim_vector;
  if (keep_dim) {
    for (int64_t i = 0; i < x_meta.dims.size(); ++i) {
      if (reduce_all || dims_set.find(i) != dims_set.end()) {
        out_dim_vector.push_back(1);
      } else {
        out_dim_vector.push_back(x_meta.dims.at(i));
      }
    }
  } else {
    for (int64_t i = 0; i < x_meta.dims.size(); ++i) {
      if (reduce_all || dims_set.find(i) != dims_set.end()) {
        continue;
      } else {
        out_dim_vector.push_back(x_meta.dims.at(i));
      }
    }

    if (out_dim_vector.size() == 0) {
      out_dim_vector.push_back(1);
    }
  }
  DDim out_dim = pten::framework::make_ddim(out_dim_vector);

  DataType out_dtype;
  if (dtype != DataType::UNDEFINED) {
    out_dtype = dtype;
  } else {
    if (x_meta.dtype == DataType::BOOL || x_meta.dtype == DataType::INT32 ||
        x_meta.dtype == DataType::INT64) {
      out_dtype = DataType::INT64;
    } else {
      out_dtype = x_meta.dtype;
    }
  }

  DenseTensorMeta return_meta(out_dtype, out_dim, x_meta.layout);
  return return_meta;
}

std::vector<DenseTensorMeta> SplitInferMeta(const DenseTensorMeta& x_meta,
                                            const ScalarArray& num_or_sections,
                                            const Scalar& axis,
                                            bool is_runtime) {
  int axis_value = axis.to<int>();
  int rank = x_meta.dims.size();
  PADDLE_ENFORCE_EQ(
      axis_value >= -rank && axis_value < rank,
      true,
      paddle::platform::errors::InvalidArgument(
          "The axis is expected to be in range of [%d, %d), but got %d",
          -rank,
          rank,
          axis_value));
  if (axis_value < 0) {
    axis_value = axis_value + rank;
  }

  auto input_axis_dim = x_meta.dims.at(axis_value);
  auto num_or_sections_data = num_or_sections.GetData();
  // step1: get formated sections
  std::vector<int64_t> sections;
  // num_or_sections is a number
  if (num_or_sections_data.size() == 1) {
    int num = num_or_sections_data.at(0);

    PADDLE_ENFORCE_EQ(input_axis_dim % num,
                      0,
                      paddle::platform::errors::InvalidArgument(
                          "The input's size along the split dimension "
                          "must be evenly divisible by Attr(num_or_sections). "
                          "But received Attr(num_or_sections) "
                          "= %d, input(X)'s shape = [%s], Attr(dim) = %d.",
                          num,
                          x_meta.dims,
                          axis_value));

    for (int i = 0; i < num; ++i) {
      sections.push_back(input_axis_dim / num);
    }
  } else {
    // num_or_sections is a sections
    const int unknow_dim_val = -1;
    int unknow_dim_idx = -1;
    int num_of_unknow = 0;
    int sum_of_section = 0;

    for (size_t i = 0; i < num_or_sections_data.size(); ++i) {
      sections.push_back(num_or_sections_data[i]);

      if (num_or_sections_data[i] == unknow_dim_val) {
        num_of_unknow++;
        unknow_dim_idx = i;
      } else {
        sum_of_section += num_or_sections_data[i];
      }
    }

    if (is_runtime) {
      PADDLE_ENFORCE_LE(num_of_unknow,
                        1,
                        paddle::platform::errors::InvalidArgument(
                            "Only one dimension value of Attr(num_or_sections) "
                            "in SplitOp can be -1. "
                            "But received Attr(num_or_sections) = [%s].",
                            pten::framework::make_ddim(num_or_sections_data)));
    }

    if (unknow_dim_idx != -1) {
      // for example, input shape = [4 ,5], axis = 1, sections = [2, 3, -1].
      // input_axis_dim = 5, sum_of_sections = 5.
      // the following check will fail.
      PADDLE_ENFORCE_LT(
          sum_of_section,
          input_axis_dim,
          paddle::platform::errors::InvalidArgument(
              "Sum of Attr(num_or_sections) other than unknown section "
              "must be less than the input's "
              "size "
              "along the split dimension. But received Attr(num_or_sections) "
              "= [%s], input(X)'s shape = [%s], Attr(dim) = %d.",
              pten::framework::make_ddim(num_or_sections_data),
              x_meta.dims,
              axis_value));

      if (is_runtime) {
        sections[unknow_dim_idx] = input_axis_dim - sum_of_section;
      }
    } else {
      PADDLE_ENFORCE_EQ(
          sum_of_section,
          input_axis_dim,
          paddle::platform::errors::InvalidArgument(
              "Sum of Attr(num_or_sections) must be equal to the input's "
              "size "
              "along the split dimension. But received Attr(num_or_sections)"
              " = [%s], input(X)'s shape = [%s], Attr(dim) = %d.",
              pten::framework::make_ddim(num_or_sections_data),
              x_meta.dims,
              axis_value));
    }
  }

  // setp2: fill out dims
  std::vector<pten::DDim> out_dims(sections.size(), x_meta.dims);
  if (is_runtime || input_axis_dim > 0) {
    for (size_t i = 0; i < sections.size(); ++i) {
      out_dims[i][axis_value] = sections[i];
    }
  } else {
    for (size_t i = 0; i < sections.size(); ++i) {
      out_dims[i][axis_value] = -1;
    }
  }

  std::vector<DenseTensorMeta> out_metas;
  out_metas.reserve(sections.size());
  for (size_t i = 0; i < sections.size(); ++i) {
    DenseTensorMeta tmp_meta;
    if (axis_value != 0) {
      // Only pass LoD when not spliting along the first dim.
      tmp_meta = {x_meta.dtype, out_dims[i], x_meta.layout};
    } else {
      tmp_meta = {x_meta.dtype, out_dims[i], x_meta.layout, x_meta.lod};
    }
    out_metas.emplace_back(std::move(tmp_meta));
  }

  return out_metas;
}

}  // namespace pten

PT_REGISTER_INFER_META_FN(sign, pten::UnchangedInferMetaNew);
