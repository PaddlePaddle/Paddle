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

#include "paddle/phi/infermeta/unary.h"

#include <algorithm>
#include <set>
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/kernels/funcs/unfold_functor.h"

namespace phi {

void UnchangedInferMeta(const MetaTensor& x, MetaTensor* out) {
  out->share_meta(x);
}

void FlattenInferMeta(const MetaTensor& x,
                      int start_axis,
                      int stop_axis,
                      MetaTensor* out) {
  auto x_dims = x.dims();
  int in_dims_size = x_dims.size();
  if (start_axis < 0) {
    start_axis = start_axis + in_dims_size;
  }
  if (stop_axis < 0) {
    stop_axis = stop_axis + in_dims_size;
  }
  PADDLE_ENFORCE_GE(
      stop_axis,
      start_axis,
      phi::errors::InvalidArgument("The stop_axis should be greater"
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
  const auto& out_dims = phi::make_ddim(out_shape);
  out->set_dims(out_dims);
  out->set_dtype(x.dtype());
  out->set_layout(x.layout());

  if (x_dims[0] == out_dims[0]) {
    // Only pass LoD when the first dimension of output and Input(X)
    // are the same.
    out->share_lod(x);
  }
}

void CastInferMeta(const MetaTensor& x, DataType out_dtype, MetaTensor* out) {
  out->set_dims(x.dims());
  out->set_dtype(out_dtype);
  out->set_layout(x.layout());
}

void CholeskyInferMeta(const MetaTensor& x, bool upper, MetaTensor* out) {
  auto dims = x.dims();
  auto rank = dims.size();
  PADDLE_ENFORCE_GE(rank,
                    2,
                    errors::InvalidArgument(
                        "The Input(X) should have at least 2 dimensions. But "
                        "received a %d dimension tensor.",
                        rank));
  PADDLE_ENFORCE_EQ(
      dims[rank - 2],
      dims[rank - 1],
      errors::InvalidArgument(
          "The inner-most 2 dimensions of Input(X) all should be symmetric "
          "positive-definite matrices and have the same size. But received "
          "X's shape[-2] = %d and shape[-1] = %d.",
          dims[rank - 2],
          dims[rank - 1]));
  out->set_dims(x.dims());
  out->set_dtype(x.dtype());
}

void CopyToInferMeta(const MetaTensor& x,
                     Backend backend,
                     bool blocking,
                     MetaTensor* out) {
  UnchangedInferMeta(x, out);
}

void CreateLikeInferMeta(const MetaTensor& x, DataType dtype, MetaTensor* out) {
  out->set_dims(x.dims());
  out->set_dtype(dtype == DataType::UNDEFINED ? x.dtype() : dtype);
  out->set_layout(x.layout());
}

void IncrementInferMeta(const MetaTensor& x, float value, MetaTensor* out) {
  PADDLE_ENFORCE_EQ(
      product(x.dims()),
      1UL,
      errors::InvalidArgument("The number of elements in Input(X) should be 1."
                              "Now the number is %d.",
                              product(x.dims())));
  out->set_dims(x.dims());
  out->share_lod(x);
  out->set_dtype(x.dtype());
}

static phi::DDim ValidateShape(const std::vector<int64_t> shape,
                               const phi::DDim& in_dims) {
  const int64_t in_size = phi::product(in_dims);
  auto in_dims_vec = phi::vectorize(in_dims);
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
          phi::errors::InvalidArgument(
              "Only one dimension value of 'shape' in ReshapeOp can "
              "be -1. But received shape = [%s], shape[%d] is also -1.",
              phi::make_ddim(shape),
              i));
      unk_dim_idx = i;
    } else if (shape[i] == copy_dim_val) {
      PADDLE_ENFORCE_LT(
          static_cast<int>(i),
          in_dims.size(),
          phi::errors::InvalidArgument(
              "The index of 0 in `shape` must be less than "
              "the input tensor X's dimensions. "
              "But received shape = [%s], shape[%d] = 0, X's shape = [%s], "
              "X's dimensions = %d.",
              phi::make_ddim(shape),
              i,
              in_dims,
              in_dims.size()));
    } else {
      PADDLE_ENFORCE_GT(
          shape[i],
          0,
          phi::errors::InvalidArgument(
              "Each dimension value of 'shape' in ReshapeOp must not "
              "be negative except one unknown dimension. "
              "But received  shape = [%s], shape[%d] = %d.",
              phi::make_ddim(shape),
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
          phi::errors::InvalidArgument(
              "The 'shape' attribute in ReshapeOp is invalid. "
              "The input tensor X'size must be divisible by known "
              "capacity of 'shape'. "
              "But received X's shape = [%s], X's size = %d, "
              "'shape' is [%s], known capacity of 'shape' is %d.",
              in_dims,
              in_size,
              phi::make_ddim(shape),
              capacity));
    } else {
      output_shape[unk_dim_idx] = -1;
    }
  } else {
    if (all_positive) {
      PADDLE_ENFORCE_EQ(
          capacity,
          in_size,
          phi::errors::InvalidArgument(
              "The 'shape' in ReshapeOp is invalid. "
              "The input tensor X'size must be equal to the capacity of "
              "'shape'. "
              "But received X's shape = [%s], X's size = %d, 'shape' is "
              "[%s], the capacity of 'shape' is %d.",
              in_dims,
              in_size,
              phi::make_ddim(shape),
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
        phi::errors::InvalidArgument(
            "The 'shape' in ReshapeOp is invalid. "
            "The input tensor X's shape = [%s], X's capacity = %d."
            "But the target shape of Out is [%s],  the "
            "capacity of 'Out' is %d.",
            in_dims,
            in_size,
            phi::make_ddim(shape),
            capacity));
  }

  return phi::make_ddim(output_shape);
}

void InferMetaFromVecValue(const MetaTensor& x,
                           const std::vector<int64_t>& shape,
                           MetaTensor* out) {
  PADDLE_ENFORCE_EQ(!shape.empty(),
                    true,
                    phi::errors::InvalidArgument(
                        "The parameter 'shape' in ReshapeOp must be set. "
                        "But received 'shape' is empty."));
  auto x_dims = x.dims();
  auto out_dims = ValidateShape(shape, x_dims);
  out->set_dims(out_dims);
  out->set_dtype(x.dtype());
  out->set_layout(x.layout());
  if (x_dims[0] == out_dims[0]) {
    // Only pass LoD when the first dimension of output and Input(X)
    // are the same.
    out->share_lod(x);
  }
}

void MultinomialInferMeta(const MetaTensor& x,
                          int num_samples,
                          bool replacement,
                          MetaTensor* out) {
  auto x_dim = x.dims();
  int64_t x_rank = x_dim.size();
  PADDLE_ENFORCE_GT(x_rank,
                    0,
                    errors::InvalidArgument(
                        "The number of dimensions of the input probability "
                        "distribution should be > 0, but got %d.",
                        x_rank));
  PADDLE_ENFORCE_LE(x_rank,
                    2,
                    errors::InvalidArgument(
                        "The number of dimensions of the input probability "
                        "distribution should be <= 2, but got %d.",
                        x_rank));

  std::vector<int64_t> out_dims(x_rank);
  for (int64_t i = 0; i < x_rank - 1; i++) {
    out_dims[i] = x_dim[i];
  }

  PADDLE_ENFORCE_GT(
      num_samples,
      0,
      errors::InvalidArgument(
          "The number of samples should be > 0, but got %d.", num_samples));
  out_dims[x_rank - 1] = num_samples;

  out->set_dims(make_ddim(out_dims));
  out->set_dtype(DataType::INT64);
}

void ReshapeInferMeta(const MetaTensor& x,
                      const ScalarArray& shape,
                      MetaTensor* out,
                      MetaConfig config) {
  auto& shape_data = shape.GetData();
  PADDLE_ENFORCE_NOT_NULL(out,
                          phi::errors::InvalidArgument(
                              "Output(Out) of ReshapeOp should not be null."));
  if (!config.is_runtime && shape.FromTensor()) {
    out->set_dims(phi::make_ddim(shape_data));
    out->share_lod(x);
    return;
  }
  PADDLE_ENFORCE_GT(shape_data.size(),
                    0,
                    phi::errors::InvalidArgument(
                        "The shape's size in ReshapeOp can't be zero."));
  InferMetaFromVecValue(x, shape_data, out);
}

void ReshapeWithXShapeInferMeta(const MetaTensor& x,
                                const ScalarArray& shape,
                                MetaTensor* xshape,
                                MetaTensor* out,
                                MetaConfig config) {
  PADDLE_ENFORCE_NOT_NULL(
      xshape,
      phi::errors::InvalidArgument(
          "Output(XShape) of ReshapeOp should not be null."));
  const auto& x_dims = x.dims();
  std::vector<int64_t> xshape_dims(x_dims.size() + 1);
  xshape_dims[0] = 0;
  for (int i = 0; i < x_dims.size(); ++i) {
    xshape_dims[i + 1] = x_dims[i];
  }
  xshape->set_dims(phi::make_ddim(xshape_dims));
  xshape->share_lod(x);
  ReshapeInferMeta(x, shape, out, config);
}

/*  Why not use ReduceInferMeta directly?
    Because we need make InferMetaFunction's args follow the design of api.yaml
*/
void SumInferMeta(const MetaTensor& x,
                  const std::vector<int64_t>& axis,
                  DataType dtype,
                  bool keep_dim,
                  MetaTensor* out) {
  ReduceInferMetaBase(x, axis, keep_dim, dtype, out);
}

void ReduceInferMetaBase(const MetaTensor& x,
                         const std::vector<int64_t>& axis,
                         bool keep_dim,
                         DataType dtype,
                         MetaTensor* out) {
  bool reduce_all = true;
  std::set<int64_t> dims_set(axis.begin(), axis.end());
  for (int64_t i = 0; i < x.dims().size(); ++i) {
    if (dims_set.find(i) == dims_set.end()) {
      reduce_all = false;
      break;
    }
  }

  std::vector<int64_t> out_dim_vector;
  if (keep_dim) {
    for (int64_t i = 0; i < x.dims().size(); ++i) {
      if (reduce_all || dims_set.find(i) != dims_set.end()) {
        out_dim_vector.push_back(1);
      } else {
        out_dim_vector.push_back(x.dims().at(i));
      }
    }
  } else {
    for (int64_t i = 0; i < x.dims().size(); ++i) {
      if (reduce_all || dims_set.find(i) != dims_set.end()) {
        continue;
      } else {
        out_dim_vector.push_back(x.dims().at(i));
      }
    }

    if (out_dim_vector.size() == 0) {
      out_dim_vector.push_back(1);
    }
  }
  DDim out_dim = phi::make_ddim(out_dim_vector);

  DataType out_dtype;
  if (dtype != DataType::UNDEFINED) {
    out_dtype = dtype;
  } else {
    if (x.dtype() == DataType::BOOL || x.dtype() == DataType::INT32 ||
        x.dtype() == DataType::INT64) {
      out_dtype = DataType::INT64;
    } else {
      out_dtype = x.dtype();
    }
  }

  out->set_dims(out_dim);
  out->set_dtype(out_dtype);
  out->set_layout(x.layout());
}

void ReduceInferMeta(const MetaTensor& x,
                     const std::vector<int64_t>& axis,
                     bool keep_dim,
                     MetaTensor* out) {
  ReduceInferMetaBase(x, axis, keep_dim, DataType::UNDEFINED, out);
}

void TransferLayoutInferMeta(const MetaTensor& x,
                             DataLayout layout,
                             MetaTensor* out) {
  out->set_dims(x.dims());
  out->set_dtype(x.dtype());
  out->set_layout(layout);
}

void SplitInferMeta(const MetaTensor& x,
                    const ScalarArray& num_or_sections,
                    const Scalar& axis,
                    std::vector<MetaTensor>* out,
                    MetaConfig config) {
  int axis_value = axis.to<int>();
  int rank = x.dims().size();
  PADDLE_ENFORCE_EQ(
      axis_value >= -rank && axis_value < rank,
      true,
      phi::errors::InvalidArgument(
          "The axis is expected to be in range of [%d, %d), but got %d",
          -rank,
          rank,
          axis_value));
  if (axis_value < 0) {
    axis_value = axis_value + rank;
  }

  auto input_axis_dim = x.dims().at(axis_value);
  auto num_or_sections_data = num_or_sections.GetData();
  // step1: get formated sections
  std::vector<int64_t> sections;
  // num_or_sections is a number
  if (num_or_sections_data.size() == 1) {
    int num = num_or_sections_data.at(0);

    PADDLE_ENFORCE_EQ(input_axis_dim % num,
                      0,
                      phi::errors::InvalidArgument(
                          "The input's size along the split dimension "
                          "must be evenly divisible by Attr(num_or_sections). "
                          "But received Attr(num_or_sections) "
                          "= %d, input(X)'s shape = [%s], Attr(dim) = %d.",
                          num,
                          x.dims(),
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

    if (config.is_runtime) {
      PADDLE_ENFORCE_LE(num_of_unknow,
                        1,
                        phi::errors::InvalidArgument(
                            "Only one dimension value of Attr(num_or_sections) "
                            "in SplitOp can be -1. "
                            "But received Attr(num_or_sections) = [%s].",
                            phi::make_ddim(num_or_sections_data)));
    }

    if (unknow_dim_idx != -1) {
      // for example, input shape = [4 ,5], axis = 1, sections = [2, 3, -1].
      // input_axis_dim = 5, sum_of_sections = 5.
      // the following check will fail.
      PADDLE_ENFORCE_LT(
          sum_of_section,
          input_axis_dim,
          phi::errors::InvalidArgument(
              "Sum of Attr(num_or_sections) other than unknown section "
              "must be less than the input's "
              "size "
              "along the split dimension. But received Attr(num_or_sections) "
              "= [%s], input(X)'s shape = [%s], Attr(dim) = %d.",
              phi::make_ddim(num_or_sections_data),
              x.dims(),
              axis_value));

      if (config.is_runtime) {
        sections[unknow_dim_idx] = input_axis_dim - sum_of_section;
      }
    } else {
      PADDLE_ENFORCE_EQ(
          sum_of_section,
          input_axis_dim,
          phi::errors::InvalidArgument(
              "Sum of Attr(num_or_sections) must be equal to the input's "
              "size "
              "along the split dimension. But received Attr(num_or_sections)"
              " = [%s], input(X)'s shape = [%s], Attr(dim) = %d.",
              phi::make_ddim(num_or_sections_data),
              x.dims(),
              axis_value));
    }
  }

  // setp2: fill out dims
  std::vector<phi::DDim> out_dims(sections.size(), x.dims());
  if (config.is_runtime || input_axis_dim > 0) {
    for (size_t i = 0; i < sections.size(); ++i) {
      out_dims[i][axis_value] = sections[i];
    }
  } else {
    for (size_t i = 0; i < sections.size(); ++i) {
      out_dims[i][axis_value] = -1;
    }
  }

  for (size_t i = 0; i < sections.size(); ++i) {
    if (axis_value != 0) {
      // Only pass LoD when not spliting along the first dim.
      (*out)[i].set_dtype(x.dtype());
      (*out)[i].set_dims(out_dims[i]);
      (*out)[i].set_layout(x.layout());
    } else {
      (*out)[i].set_dtype(x.dtype());
      (*out)[i].set_dims(out_dims[i]);
      (*out)[i].set_layout(x.layout());
      (*out)[i].share_lod(x);
    }
  }
}

void UnbindInferMeta(const MetaTensor& x,
                     int axis,
                     std::vector<MetaTensor>* outs) {
  auto in_dims = x.dims();
  std::vector<int> out_dim;
  axis = axis < 0 ? in_dims.size() + axis : axis;
  for (int i = 0; i < in_dims.size(); ++i) {
    if (i != axis) out_dim.push_back(in_dims[i]);
  }
  auto out_dims = phi::make_ddim(out_dim);

  for (size_t i = 0; i < outs->size(); ++i) {
    (*outs)[i].set_dtype(x.dtype());
    (*outs)[i].set_dims(out_dims);
    (*outs)[i].set_layout(x.layout());
    (*outs)[i].share_lod(x);
  }
}

void TraceInferMeta(
    const MetaTensor& x, int offset, int axis1, int axis2, MetaTensor* out) {
  int dim1 = axis1;
  int dim2 = axis2;

  auto x_dims = x.dims();

  int dim1_ = dim1 < 0 ? x_dims.size() + dim1 : dim1;
  int dim2_ = dim2 < 0 ? x_dims.size() + dim2 : dim2;

  PADDLE_ENFORCE_GE(
      x_dims.size(),
      2,
      phi::errors::OutOfRange(
          "Input's dim is out of range (expected at least 2, but got %ld).",
          x_dims.size()));
  PADDLE_ENFORCE_LT(
      dim1_,
      x_dims.size(),
      phi::errors::OutOfRange(
          "Attr(dim1) is out of range (expected to be in range of [%ld, "
          "%ld], but got %ld).",
          -(x_dims.size()),
          (x_dims.size() - 1),
          dim1));
  PADDLE_ENFORCE_LT(
      dim2_,
      x_dims.size(),
      phi::errors::OutOfRange(
          "Attr(dim2) is out of range (expected to be in range of [%ld, "
          "%ld], but got %ld).",
          -(x_dims.size()),
          (x_dims.size() - 1),
          dim2));
  PADDLE_ENFORCE_NE(
      dim1_,
      dim2_,
      phi::errors::InvalidArgument("The dimensions should not be identical "
                                   "%ld vs %ld.",
                                   dim1,
                                   dim2));

  auto sizes = vectorize(x_dims);
  if (x_dims.size() == 2) {
    sizes.clear();
    sizes.push_back(1);
  } else {
    sizes.erase(sizes.begin() + std::max(dim1_, dim2_));
    sizes.erase(sizes.begin() + std::min(dim1_, dim2_));
  }
  out->set_dims(phi::make_ddim(sizes));
}

void UnfoldInferMeta(const MetaTensor& x,
                     const std::vector<int>& kernel_sizes,
                     const std::vector<int>& strides,
                     const std::vector<int>& paddings,
                     const std::vector<int>& dilations,
                     MetaTensor* out,
                     MetaConfig config) {
  auto in_dims = x.dims();
  // Only [N, C, H, W] input supported now
  PADDLE_ENFORCE_EQ(
      in_dims.size(),
      4,
      phi::errors::InvalidArgument(
          "Input should be 4-D tensor of format [N, C, H, W], but get %u",
          in_dims.size()));
  PADDLE_ENFORCE_EQ(
      in_dims.size() - kernel_sizes.size(),
      2U,
      phi::errors::InvalidArgument(
          "The dims of X should be larger than that of kernel_sizes "
          "by a number of 2, due to the batch size and input channel dim. "
          "But recieved dims(X:%u) - dims(kernel_sizes:%u) != 2",
          in_dims.size(),
          kernel_sizes.size()));
  PADDLE_ENFORCE_EQ(
      strides.size(),
      kernel_sizes.size(),
      phi::errors::InvalidArgument(
          "The dims of strides should be the same with that of kernel_sizes. "
          "But recieved dims(strides: %u) != dims(kernel_sizes: %u).",
          strides.size(),
          kernel_sizes.size()));
  PADDLE_ENFORCE_EQ(
      paddings.size(),
      2 * strides.size(),
      phi::errors::InvalidArgument(
          "The dims of paddings should be 2 times of that of strides. "
          "But recieved dims(paddings: %u) != 2*dims(strides: %u).",
          paddings.size(),
          strides.size()));
  PADDLE_ENFORCE_EQ(
      strides.size(),
      dilations.size(),
      phi::errors::InvalidArgument(
          "The dims of strides should be the same with that of dilations. "
          "But recieved dims(strides: %u) != dims(dilations: %u).",
          strides.size(),
          dilations.size()));

  // check kernel_sizes
  PADDLE_ENFORCE_GT(kernel_sizes[0],
                    0,
                    phi::errors::InvalidArgument(
                        "The `kernel_sizes` should be greater than zero, "
                        "but recieved kernel_height: %d kernel_width: %d.",
                        kernel_sizes[0],
                        kernel_sizes[1]));
  PADDLE_ENFORCE_GT(kernel_sizes[1],
                    0,
                    phi::errors::InvalidArgument(
                        "The `kernel_sizes` should be greater than zero, "
                        "but recieved kernel_height: %d kernel_width: %d.",
                        kernel_sizes[0],
                        kernel_sizes[1]));
  // check strides
  PADDLE_ENFORCE_GT(strides[0],
                    0,
                    phi::errors::InvalidArgument(
                        "The `strides` should be greater than zero, "
                        "but recieved strides_height: %d strides_width: %d.",
                        strides[0],
                        strides[1]));
  PADDLE_ENFORCE_GT(strides[1],
                    0,
                    phi::errors::InvalidArgument(
                        "The `strides` should be greater than zero, "
                        "but recieved strides_height: %d strides_width: %d.",
                        strides[0],
                        strides[1]));
  // check dilations
  PADDLE_ENFORCE_GT(
      dilations[0],
      0,
      phi::errors::InvalidArgument(
          "The `dilations` should be greater than zero, "
          "but recieved dilations_height: %d dilations_width: %d.",
          dilations[0],
          dilations[1]));
  PADDLE_ENFORCE_GT(
      dilations[1],
      0,
      phi::errors::InvalidArgument(
          "The `dilations` should be greater than zero, "
          "but recieved dilations_height: %d dilations_width: %d.",
          dilations[0],
          dilations[1]));

  std::vector<int> out_dims;
  out_dims.push_back(in_dims[0]);
  int output_channels = in_dims[1] * kernel_sizes[0] * kernel_sizes[1];
  out_dims.push_back(output_channels);

  int output_height = phi::funcs::CalcOutputSize(in_dims[2],
                                                 kernel_sizes[0],
                                                 dilations[0],
                                                 paddings[0],
                                                 paddings[2],
                                                 strides[0]);
  int output_width = phi::funcs::CalcOutputSize(in_dims[3],
                                                kernel_sizes[1],
                                                dilations[1],
                                                paddings[1],
                                                paddings[3],
                                                strides[1]);
  if (config.is_runtime) {
    // only check output height and width in runtime
    PADDLE_ENFORCE_GT(
        output_height,
        0,
        phi::errors::InvalidArgument(
            "The sliding blocks calculated from input spatial size "
            "(%d, %d), kernel_sizes (%d, %d), strides (%d, %d), "
            "dilations (%d, %d), is (%d, %d), which should be a "
            "positive integer.",
            in_dims[2],
            in_dims[3],
            kernel_sizes[0],
            kernel_sizes[1],
            strides[0],
            strides[1],
            dilations[0],
            dilations[1],
            output_height,
            output_width));
    PADDLE_ENFORCE_GT(
        output_width,
        0,
        phi::errors::InvalidArgument(
            "The sliding blocks calculated from input spatial size "
            "(%d, %d), kernel_sizes (%d, %d), strides (%d, %d), "
            "dilations (%d, %d), is (%d, %d), which should be a "
            "positive integer.",
            in_dims[2],
            in_dims[3],
            kernel_sizes[0],
            kernel_sizes[1],
            strides[0],
            strides[1],
            dilations[0],
            dilations[1],
            output_height,
            output_width));
  }
  int output_col_length = output_height * output_width;
  out_dims.push_back(output_col_length);
  out->set_dims(phi::make_ddim(out_dims));
}

void DiagInferMeta(const MetaTensor& x,
                   int offset,
                   float padding_value,
                   MetaTensor* out) {
  auto x_dims = x.dims();

  if (x_dims.size() == 1UL) {
    int64_t size_ = x_dims[0] + std::abs(offset);
    out->set_dims({size_, size_});
    out->set_dtype(x.dtype());
  } else if (x_dims.size() == 2UL) {
    int64_t size_ = 0;
    if (offset >= 0) {
      // Note(LutaoChu): Do not use std::min here, otherwise the calculation
      // of `size_` will have unexpected result on Windows Python3.8
      if (x_dims[0] < x_dims[1] - offset) {
        size_ = x_dims[0];
      } else {
        size_ = x_dims[1] - offset;
      }
    } else {
      // Note(LutaoChu): Do not use std::min here, otherwise the calculation
      // of `size_` will have unexpected result on Windows Python3.8
      if (x_dims[0] + offset < x_dims[1]) {
        size_ = x_dims[0] + offset;
      } else {
        size_ = x_dims[1];
      }
    }
    out->set_dims({size_});
    out->set_dtype(x.dtype());
  } else {
    PADDLE_THROW(phi::errors::InvalidArgument(
        "The input tensor X's dimensions of DiagV2Op should be either 1 or "
        "2, but received %d.",
        x_dims.size()));
  }
}

}  // namespace phi

PD_REGISTER_INFER_META_FN(copy_to, phi::CopyToInferMeta);
PD_REGISTER_INFER_META_FN(split, phi::SplitInferMeta);
