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

#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/type_traits.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/kernels/funcs/parse_qr_mode.h"
#include "paddle/phi/kernels/funcs/pooling.h"
#include "paddle/phi/kernels/funcs/slice_utils.h"
#include "paddle/phi/kernels/funcs/strided_slice.h"
#include "paddle/phi/kernels/funcs/unfold_functor.h"
#include "paddle/phi/kernels/funcs/unsqueeze.h"

namespace phi {

namespace detail {
// Used in MatrixRankInferMeta
static DDim CheckAndGetOutputDim(const DDim& dim_x) {
  auto x_vec = phi::vectorize(dim_x);
  if (x_vec.size() == 2) {
    return phi::make_ddim({1});
  }
  x_vec.erase(x_vec.end() - 2, x_vec.end());
  return phi::make_ddim(x_vec);
}
}  // namespace detail

void ArgMinMaxInferMeta(const MetaTensor& x,
                        int64_t axis,
                        bool keepdims,
                        bool flatten,
                        int dtype,
                        MetaTensor* out,
                        MetaConfig config) {
  const auto& x_dims = x.dims();

  PADDLE_ENFORCE_GE(
      axis,
      -x_dims.size(),
      phi::errors::InvalidArgument("'axis'(%d) must be greater than or equal to"
                                   " -Rank(X)(%d).",
                                   axis,
                                   -x_dims.size()));
  PADDLE_ENFORCE_LT(axis,
                    x_dims.size(),
                    phi::errors::InvalidArgument(
                        "'axis'(%d) must be less than Rank(X)(%d) of Input(X).",
                        axis,
                        x_dims.size()));

  PADDLE_ENFORCE_EQ(
      (dtype < 0 || dtype == 2 || dtype == 3),
      true,
      phi::errors::InvalidArgument(
          "The attribute of dtype in argmin/argmax must be [%s] or [%s], but "
          "received [%s]",
          paddle::framework::DataTypeToString(
              paddle::framework::proto::VarType::INT32),
          paddle::framework::DataTypeToString(
              paddle::framework::proto::VarType::INT64),
          paddle::framework::DataTypeToString(
              static_cast<paddle::framework::proto::VarType::Type>(dtype))));

  auto x_rank = x_dims.size();
  if (axis < 0) axis += x_rank;
  if (config.is_runtime) {
    if (dtype == paddle::framework::proto::VarType::INT32) {
      int64_t all_element_num = 0;
      if (flatten) {
        all_element_num = phi::product(x_dims);

      } else {
        all_element_num = x_dims[axis];
      }
      PADDLE_ENFORCE_LE(
          all_element_num,
          INT_MAX,
          phi::errors::InvalidArgument(
              "The element num of the argmin/argmax input at axis is "
              "%d, is larger than int32 maximum value:%d, you must "
              "set the dtype of argmin/argmax to 'int64'.",
              all_element_num,
              INT_MAX));
    }
  }
  std::vector<int64_t> vec;
  if (flatten) {
    vec.emplace_back(static_cast<int64_t>(1));
  } else {
    for (int64_t i = 0; i < axis; i++) vec.emplace_back(x_dims[i]);
    if (keepdims) {
      vec.emplace_back(static_cast<int64_t>(1));
    }
    for (int64_t i = axis + 1; i < x_rank; i++) vec.emplace_back(x_dims[i]);
  }
  out->set_dims(phi::make_ddim(vec));
  if (dtype == 2) {
    out->set_dtype(DataType::INT32);
  } else if (dtype == 3) {
    out->set_dtype(DataType::INT64);
  }
}

void ArgsortInferMeta(const MetaTensor& input,
                      int axis,
                      bool descending,
                      MetaTensor* output,
                      MetaTensor* indices) {
  auto in_dims = input.dims();
  auto num_dims = in_dims.size();
  PADDLE_ENFORCE_GE(
      axis,
      -num_dims,
      phi::errors::InvalidArgument("'axis'(%d) must be greater than or equal to"
                                   " -num_dims(%d).",
                                   axis,
                                   -num_dims));
  PADDLE_ENFORCE_LT(
      axis,
      num_dims,
      phi::errors::InvalidArgument(
          "'axis'(%d) must be less than num_dims(%d).", axis, num_dims));

  output->share_dims(input);
  output->set_dtype(input.dtype());
  indices->share_dims(input);
  indices->set_dtype(DataType::INT64);
  output->share_lod(input);
  indices->share_lod(input);
}

void BatchSizeLikeInferMeta(const MetaTensor& x,
                            const std::vector<int>& shape,
                            int x_batch_size_dim,
                            int out_batch_size_dim,
                            MetaTensor* out) {
  PADDLE_ENFORCE_GT(
      shape.size(),
      0UL,
      phi::errors::InvalidArgument(
          "Shape size must be larger than 0, but received: %s.", shape.size()));
  std::vector<int64_t> shape_int64(shape.size(), 0);
  std::transform(shape.begin(), shape.end(), shape_int64.begin(), [](int a) {
    return static_cast<int64_t>(a);
  });
  auto output_dim = phi::make_ddim(shape_int64);

  int input_dim_size = static_cast<int>(x.dims().size());
  PADDLE_ENFORCE_GE(
      x_batch_size_dim,
      0,
      phi::errors::InvalidArgument("Input dimension index must be larger "
                                   "equal than 0, but received: %s.",
                                   x_batch_size_dim));
  PADDLE_ENFORCE_GT(input_dim_size,
                    x_batch_size_dim,
                    phi::errors::InvalidArgument(
                        "Input dimension size must be larger than "
                        "input dimension index, but received input "
                        "dimension size: %s, input dimension index: %s.",
                        input_dim_size,
                        x_batch_size_dim));

  int output_dim_size = static_cast<int>(shape.size());
  PADDLE_ENFORCE_GE(
      out_batch_size_dim,
      0,
      phi::errors::InvalidArgument("Output dimension index must be larger "
                                   "equal than 0, but received: %s.",
                                   out_batch_size_dim));
  PADDLE_ENFORCE_GT(
      output_dim_size,
      out_batch_size_dim,
      phi::errors::InvalidArgument(
          "Output dimension size must be larger than output dimension index, "
          "but received output dimension size: %s, output dimension index: "
          "%s.",
          output_dim_size,
          out_batch_size_dim));

  output_dim[out_batch_size_dim] = x.dims()[x_batch_size_dim];
  out->set_dims(output_dim);
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

void CreateLikeInferMeta(const MetaTensor& x, DataType dtype, MetaTensor* out) {
  out->set_dims(x.dims());
  out->set_dtype(dtype == DataType::UNDEFINED ? x.dtype() : dtype);
  out->set_layout(x.layout());
}

void CumsumInferMeta(const MetaTensor& x,
                     int axis,
                     bool flatten,
                     bool exclusive,
                     bool reverse,
                     MetaTensor* out) {
  auto x_dims = x.dims();
  if (flatten) {
    out->set_dims(phi::make_ddim({phi::product(x_dims)}));
    out->set_dtype(x.dtype());
  } else {
    out->set_dims(x_dims);
    out->set_dtype(x.dtype());
  }

  out->share_lod(x);
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

void DiagonalInferMeta(const MetaTensor& input,
                       int offset,
                       int axis1,
                       int axis2,
                       MetaTensor* out) {
  auto x_dims = input.dims();
  int offset_ = offset;
  int axis1_ = axis1 < 0 ? x_dims.size() + axis1 : axis1;
  int axis2_ = axis2 < 0 ? x_dims.size() + axis2 : axis2;

  PADDLE_ENFORCE_GE(
      x_dims.size(),
      2,
      phi::errors::OutOfRange("Input's dim is out of range (expected at "
                              "least 2 dimensions, but got %ld).",
                              x_dims.size()));
  PADDLE_ENFORCE_LT(
      axis1_,
      x_dims.size(),
      phi::errors::OutOfRange(
          "Attr(axis1) is out of range (expected to be in range of [%ld, "
          "%ld], but got %ld).",
          -(x_dims.size()),
          (x_dims.size() - 1),
          axis1));
  PADDLE_ENFORCE_LT(
      axis2_,
      x_dims.size(),
      phi::errors::OutOfRange(
          "Attr(axis2) is out of range (expected to be in range of [%ld, "
          "%ld], but got %ld).",
          -(x_dims.size()),
          (x_dims.size() - 1),
          axis2));
  PADDLE_ENFORCE_NE(
      axis1_,
      axis2_,
      phi::errors::InvalidArgument("The dimensions should not be identical "
                                   "%d vs %d.",
                                   axis1,
                                   axis2));

  auto out_dims = vectorize(x_dims);
  // from out_dims get the dim size of axis1_.
  auto axis1_size = out_dims[axis1_];
  auto axis2_size = out_dims[axis2_];
  // delete two dims by attr axis1 and axis2 from out_dims.
  /* example:
     out_dim = [2, 3, 4];
     axis1 = 0;
     axis2 = 1;
     according to the attr of axis1 and axis2, we get:
     out_dim = [4].
  */
  out_dims.erase(out_dims.begin() + std::max(axis1_, axis2_));
  out_dims.erase(out_dims.begin() + std::min(axis1_, axis2_));

  if (offset_ == 0) {
    out_dims.push_back(std::min(axis1_size, axis2_size));
  } else if (offset_ > 0) {
    if ((axis2_size - offset_) > 0) {
      out_dims.push_back(std::min(axis1_size, axis2_size - offset_));
    } else {
      out_dims.push_back(0);
    }
  } else {
    if ((axis1_size + offset_) > 0) {
      out_dims.push_back(std::min(axis1_size + offset_, axis2_size));
    } else {
      out_dims.push_back(0);
    }
  }
  out->set_dims(phi::make_ddim(out_dims));
}

void EighInferMeta(const MetaTensor& x,
                   const std::string& uplo,
                   MetaTensor* out_w,
                   MetaTensor* out_v) {
  auto input_dim = x.dims();
  auto rank = input_dim.size();

  PADDLE_ENFORCE_GE(rank,
                    2,
                    phi::errors::InvalidArgument(
                        "The Input(X) should have at least 2 dimensions."
                        "But received a %d dimension tensor.",
                        rank));
  PADDLE_ENFORCE_EQ(
      input_dim[rank - 2],
      input_dim[rank - 1],
      phi::errors::InvalidArgument(
          "Eigh op is designed for square matrix, consequently"
          "inner-most 2 dimensions of Input(X) should be symmetric."
          "But received X's shape[-2] = %d and shape[-1] = %d.",
          input_dim[rank - 2],
          input_dim[rank - 1]));

  std::vector<int64_t> values_dim;

  for (auto i = 0; i < rank - 1; i++) {
    values_dim.emplace_back(input_dim[i]);
  }
  out_w->set_dims(phi::make_ddim(values_dim));
  out_v->set_dims(input_dim);
}

void ExpandInferMeta(const MetaTensor& x,
                     const IntArray& shape,
                     MetaTensor* out) {
#define MAX_RANK_SUPPORTED 6
  auto x_dims = x.dims();
  auto expand_shape = shape.GetData();

  if (expand_shape.size() == 0) {
    expand_shape = std::vector<int64_t>(x_dims.size(), -1);
  }

  PADDLE_ENFORCE_GE(
      expand_shape.size(),
      static_cast<size_t>(x_dims.size()),
      phi::errors::InvalidArgument(
          "The number of elements (%d) of 'shape' for "
          "expand_v2 op must be greater than or equal to the rank "
          "(%d) of the input.",
          expand_shape.size(),
          static_cast<size_t>(x_dims.size())));
  PADDLE_ENFORCE_LE(
      expand_shape.size(),
      MAX_RANK_SUPPORTED,
      phi::errors::InvalidArgument("The number of elements (%d) of 'shape' for "
                                   "must not be greater than %d.",
                                   expand_shape.size(),
                                   MAX_RANK_SUPPORTED));
  PADDLE_ENFORCE_GE(
      expand_shape.size(),
      1,
      phi::errors::InvalidArgument("The number of elements (%d) of 'shape' for "
                                   "must be a positive integer.",
                                   expand_shape.size()));

  auto out_rank =
      std::max(static_cast<size_t>(x_dims.size()), expand_shape.size());
  std::vector<int64_t> out_shape(out_rank);
  auto x_dim_vec = phi::vectorize<int>(x_dims);
  auto diff = expand_shape.size() - x_dim_vec.size();
  x_dim_vec.insert(x_dim_vec.begin(), diff, -1);
  for (size_t i = 0; i < expand_shape.size(); ++i) {
    if (x_dims[i] == -1) {
      out_shape[i] = -1;
    } else if (expand_shape[i] == -1) {
      if (static_cast<size_t>(x_dims.size()) > i) {
        out_shape[i] = x_dims[i];
      } else {
        out_shape[i] = -1;
      }
    } else if (expand_shape[i] == -2) {
      // We use -2 to represent the element in expand_shape is a var.
      out_shape[i] = -1;
    } else {
      PADDLE_ENFORCE_GT(
          expand_shape[i],
          0,
          phi::errors::InvalidArgument(
              "The %uth element of 'shape' for expand_v2 op must be "
              "greater than 0, but the value given is %d.",
              i,
              expand_shape[i]));
      out_shape[i] = expand_shape[i];
    }
  }

  out->set_dims(make_ddim(out_shape));
  out->set_dtype(x.dtype());
  if (out_shape[0] == x_dims[0]) {
    out->share_lod(x);
  }
}

void FlattenInferMeta(const MetaTensor& x,
                      int start_axis,
                      int stop_axis,
                      MetaTensor* out) {
  FlattenWithXShapeInferMeta(x, start_axis, stop_axis, out, nullptr);
}

void FlattenWithXShapeInferMeta(const MetaTensor& x,
                                int start_axis,
                                int stop_axis,
                                MetaTensor* out,
                                MetaTensor* xshape) {
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
  if (xshape == nullptr) return;
  std::vector<int64_t> xshape_dims(x_dims.size() + 1);
  xshape_dims[0] = 0;
  for (int i = 0; i < x_dims.size(); ++i) {
    xshape_dims[i + 1] = x_dims[i];
  }
  xshape->set_dims(phi::make_ddim(xshape_dims));
  xshape->share_lod(x);
}

void FlipInferMeta(const MetaTensor& x,
                   const std::vector<int>& axis,
                   MetaTensor* out) {
  auto x_dims = x.dims();
  auto flip_dims = axis;
  size_t flip_dims_size = axis.size();

  if (flip_dims_size > 0) {
    // check if dims axis within range
    auto min_max_d = std::minmax_element(flip_dims.begin(), flip_dims.end());
    PADDLE_ENFORCE_LT(*min_max_d.first,
                      x_dims.size(),
                      phi::errors::InvalidArgument(
                          "min(axes) should be less than the input tensor X's "
                          "axes of FlipOp. But received min(axes) = %d,  "
                          "X's axes = %d, X's shape = [%s]",
                          *min_max_d.first,
                          x_dims.size(),
                          x_dims));
    PADDLE_ENFORCE_GE(*min_max_d.first,
                      x_dims.size() * -1,
                      phi::errors::InvalidArgument(
                          "min(axes) should be greater than or equal to the "
                          "input tensor X's "
                          "axes of FlipOp times -1. But received "
                          "min(axes) = %d,  X's "
                          "axes = %d, X's shape = [%s]",
                          *min_max_d.first,
                          x_dims.size() * -1,
                          x_dims));
    PADDLE_ENFORCE_LT(*min_max_d.second,
                      x_dims.size(),
                      phi::errors::InvalidArgument(
                          "max(axes) should be less than the input tensor X's "
                          "axes of FlipOp. But received max(axes) = %d,  "
                          "X's axes = %d, X's shape = [%s]",
                          *min_max_d.second,
                          x_dims.size(),
                          x_dims));
    PADDLE_ENFORCE_GE(*min_max_d.second,
                      x_dims.size() * -1,
                      phi::errors::InvalidArgument(
                          "max(axes) should be greater than or equal to the "
                          "input tensor X's "
                          "axes of FlipOp times -1. But received "
                          "max(axes) = %d,  X's "
                          "axes = %d, X's shape = [%s]",
                          *min_max_d.second,
                          x_dims.size() * -1,
                          x_dims));

    // check duplicates in dims
    flip_dims.erase(std::unique(flip_dims.begin(), flip_dims.end()),
                    flip_dims.end());
    PADDLE_ENFORCE_EQ(flip_dims.size(),
                      flip_dims_size,
                      phi::errors::InvalidArgument(
                          "axes has duplicates, original flip axes size=%d, "
                          "but unique flip axes size=%d.)",
                          flip_dims_size,
                          flip_dims.size()));
  }

  VLOG(3) << "flip operator x.shape=" << x_dims;

  std::vector<int64_t> output_dims(x_dims.size());
  for (int i = 0; i < x_dims.size(); ++i) {
    output_dims[i] = x_dims[i];
  }

  out->set_dims(phi::make_ddim(output_dims));
  out->set_dtype(x.dtype());
  out->share_lod(x);
}

void FullBatchSizeLikeInferMeta(const MetaTensor& x,
                                const std::vector<int>& shape,
                                const Scalar& val,
                                DataType dtype,
                                int x_batch_size_dim,
                                int out_batch_size_dim,
                                MetaTensor* out) {
  BatchSizeLikeInferMeta(x, shape, x_batch_size_dim, out_batch_size_dim, out);
  out->set_dtype(dtype);
}

void GumbelSoftmaxInferMeta(const MetaTensor& x,
                            float temperature,
                            bool hard,
                            int axis,
                            MetaTensor* out) {
  UnchangedInferMetaCheckAxis(x, axis, out);
}

void HistogramInferMeta(
    const MetaTensor& input, int64_t bins, int min, int max, MetaTensor* out) {
  PADDLE_ENFORCE_GE(bins,
                    1,
                    phi::errors::InvalidArgument(
                        "The bins should be greater than or equal to 1."
                        "But received nbins is %d",
                        bins));
  PADDLE_ENFORCE_GE(
      max,
      min,
      phi::errors::InvalidArgument("max must be larger or equal to min."
                                   "But received max is %d, min is %d",
                                   max,
                                   min));

  out->set_dims({bins});
  out->share_lod(input);
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

void IsEmptyInferMeta(const MetaTensor& x, MetaTensor* out) {
  out->set_dims(phi::make_ddim({1}));
  out->set_dtype(DataType::BOOL);
}

void IsfiniteInferMeta(const MetaTensor& x, MetaTensor* out) {
  out->set_dims(x.dims());
  out->set_dtype(DataType::BOOL);
}

void KthvalueInferMeta(const MetaTensor& x,
                       int k,
                       int axis,
                       bool keepdim,
                       MetaTensor* out,
                       MetaTensor* indices,
                       MetaConfig config) {
  auto input_dims = x.dims();
  const int& dim_size = input_dims.size();
  PADDLE_ENFORCE_LT(axis,
                    dim_size,
                    phi::errors::InvalidArgument(
                        "the axis must be [-%d, %d), but received %d .",
                        dim_size,
                        dim_size,
                        axis));
  PADDLE_ENFORCE_GE(axis,
                    -dim_size,
                    phi::errors::InvalidArgument(
                        "the axis must be [-%d, %d), but received %d .",
                        dim_size,
                        dim_size,
                        axis));
  if (axis < 0) axis += dim_size;
  PADDLE_ENFORCE_GE(
      k,
      1,
      phi::errors::InvalidArgument(
          "the k in the kthvalue must >= 1, but received %d .", k));
  PADDLE_ENFORCE_GE(
      input_dims.size(),
      1,
      phi::errors::InvalidArgument("input of kthvalue must have >= 1d shape"));
  if (config.is_runtime) {
    PADDLE_ENFORCE_GE(
        input_dims[axis],
        k,
        phi::errors::InvalidArgument(
            "input of kthvalue must have >= %d columns in axis of %d",
            k,
            axis));
  }
  std::vector<int64_t> dimvec;
  for (int64_t i = 0; i < axis; i++) {
    dimvec.emplace_back(input_dims[i]);
  }
  if (keepdim) {
    dimvec.emplace_back(static_cast<int64_t>(1));
  }
  for (int64_t i = axis + 1; i < dim_size; i++) {
    dimvec.emplace_back(input_dims[i]);
  }
  DDim dims = phi::make_ddim(dimvec);
  out->set_dims(dims);
  out->share_lod(x);
  out->set_dtype(x.dtype());
  indices->set_dims(dims);
  indices->share_lod(x);
  indices->set_dtype(x.dtype());
}

void LogsumexpInferMeta(const MetaTensor& input,
                        const std::vector<int64_t>& axis,
                        bool keepdim,
                        bool reduce_all,
                        MetaTensor* out) {
  auto x_dims = input.dims();
  auto x_rank = x_dims.size();
  std::vector<int64_t> formated_axis = axis;
  PADDLE_ENFORCE_LE(x_rank,
                    4,
                    errors::InvalidArgument(
                        "The input tensor X's dimensions of logsumexp "
                        "should be less or equal than 4. But received X's "
                        "dimensions = %d, X's shape = [%s].",
                        x_rank,
                        x_dims));
  PADDLE_ENFORCE_GT(
      axis.size(),
      0,
      errors::InvalidArgument(
          "The size of axis of logsumexp "
          "should be greater than 0. But received the size of axis "
          "of logsumexp is %d.",
          axis.size()));

  for (size_t i = 0; i < axis.size(); i++) {
    PADDLE_ENFORCE_LT(axis[i],
                      x_rank,
                      errors::InvalidArgument(
                          "axis[%d] should be in the "
                          "range [-D, D), where D is the dimensions of X and "
                          "D is %d. But received axis[%d] = %d.",
                          i,
                          x_rank,
                          i,
                          axis[i]));
    PADDLE_ENFORCE_GE(axis[i],
                      -x_rank,
                      errors::InvalidArgument(
                          "axis[%d] should be in the "
                          "range [-D, D), where D is the dimensions of X and "
                          "D is %d. But received axis[%d] = %d.",
                          i,
                          x_rank,
                          i,
                          axis[i]));
    if (axis[i] < 0) {
      formated_axis[i] += x_rank;
    }
  }

  auto dims_vector = vectorize(x_dims);
  if (reduce_all) {
    if (keepdim)
      out->set_dims(phi::make_ddim(std::vector<int64_t>(x_rank, 1)));
    else
      out->set_dims({1});
  } else {
    auto dims_vector = vectorize(x_dims);
    if (keepdim) {
      for (size_t i = 0; i < formated_axis.size(); ++i) {
        dims_vector[formated_axis[i]] = 1;
      }
    } else {
      const int kDelFlag = -1;
      for (size_t i = 0; i < formated_axis.size(); ++i) {
        dims_vector[formated_axis[i]] = kDelFlag;
      }
      dims_vector.erase(
          std::remove(dims_vector.begin(), dims_vector.end(), kDelFlag),
          dims_vector.end());
    }
    if (!keepdim && dims_vector.size() == 0) {
      dims_vector.push_back(1);
    }
    auto out_dims = phi::make_ddim(dims_vector);
    out->set_dims(out_dims);
    if (formated_axis.size() > 0 && formated_axis[0] != 0) {
      // Only pass LoD when not reducing on the first dim.
      out->share_lod(input);
    }
  }
  out->set_dtype(input.dtype());
}

void MatrixPowerInferMeta(const MetaTensor& x, int n, MetaTensor* out) {
  auto dims = x.dims();
  auto n_dim = dims.size();
  PADDLE_ENFORCE_GE(n_dim,
                    2,
                    phi::errors::InvalidArgument(
                        "The Input(X) should have at least 2 dimensions. But "
                        "received a %d dimension tensor.",
                        n_dim));
  PADDLE_ENFORCE_EQ(dims[n_dim - 2],
                    dims[n_dim - 1],
                    phi::errors::InvalidArgument(
                        "The inner-most 2 dimensions of Input(X) all should "
                        "be square matrices "
                        "But received X's shape[-2] = %d and shape[-1] = %d.",
                        dims[n_dim - 2],
                        dims[n_dim - 1]));
  out->set_dims(dims);
  out->share_lod(x);
  out->set_dtype(x.dtype());
}

void MatrixRankInferMeta(const MetaTensor& x,
                         bool use_default_tol,
                         bool hermitian,
                         MetaTensor* out) {
  auto dim_x = x.dims();
  PADDLE_ENFORCE_GE(
      dim_x.size(),
      2,
      phi::errors::InvalidArgument("The dims of input must be greater than 2"));

  if (hermitian) {
    int rows = dim_x[dim_x.size() - 2];
    int cols = dim_x[dim_x.size() - 1];
    PADDLE_ENFORCE_EQ(rows,
                      cols,
                      phi::errors::InvalidArgument(
                          "if hermitian == true, matrix should be n*n"));
  }
  DDim dim_x_batch = detail::CheckAndGetOutputDim(dim_x);
  out->set_dims(dim_x_batch);
  out->share_lod(x);
}

void MaxOutInferMeta(const MetaTensor& x,
                     int groups,
                     int axis,
                     MetaTensor* out) {
  auto in_x_dims = x.dims();
  // check groups > 1
  PADDLE_ENFORCE_GT(
      groups,
      1,
      phi::errors::InvalidArgument("Attr(groups) of Op(maxout) should be "
                                   "larger than 1. But received %d.",
                                   groups));
  PADDLE_ENFORCE_EQ(
      axis == 1 || axis == -1 || axis == 3,
      true,
      phi::errors::InvalidArgument(
          "axis only supported 1, -1 or 3, but recevied axis is: %d", axis));
  PADDLE_ENFORCE_EQ(in_x_dims.size(),
                    4,
                    phi::errors::InvalidArgument(
                        "x's dims should be 4, but received x's dims is: %d",
                        in_x_dims.size()));

  if (axis < 0) {
    axis += in_x_dims.size();
  }
  PADDLE_ENFORCE_EQ(
      in_x_dims[axis] % groups,
      0,
      phi::errors::InvalidArgument(
          "The number of input channels for Op(maxout) "
          "should be divisible by Attr(groups). But received: the "
          "input's channels is [%d], the shape of input is [%s], "
          "the Attr(groups) is [%d], the Attr(axis) is [%d]. The "
          "error may come from wrong Attr(groups) or Attr(axis) setting.",
          in_x_dims[axis],
          in_x_dims,
          groups,
          axis));
  std::vector<int64_t> output_shape(
      {in_x_dims[0], in_x_dims[1], in_x_dims[2], in_x_dims[3]});
  output_shape[axis] = in_x_dims[axis] / groups;
  out->set_dims(phi::make_ddim(output_shape));
  out->set_dtype(x.dtype());
}

void MaxPoolWithIndexInferMeta(const MetaTensor& x,
                               const std::vector<int>& kernel_size,
                               const std::vector<int>& strides,
                               const std::vector<int>& paddings,
                               bool global_pooling,
                               bool adaptive,
                               MetaTensor* out,
                               MetaTensor* mask,
                               MetaConfig config) {
  std::vector<int> paddings_ = paddings;
  std::vector<int> kernel_size_ = kernel_size;

  auto x_dims = x.dims();

  PADDLE_ENFORCE(
      x_dims.size() == 4 || x_dims.size() == 5,
      errors::InvalidArgument(
          "Pooling intput should be 4-D or 5-D tensor but received %dD-Tensor",
          x_dims.size()));

  if (global_pooling) {
    kernel_size_.resize(static_cast<size_t>(x_dims.size()) - 2);
    for (size_t i = 0; i < kernel_size_.size(); ++i) {
      paddings_[i] = 0;
      kernel_size_[i] = static_cast<int>(x_dims[i + 2]);
    }
  }

  PADDLE_ENFORCE_EQ(
      x_dims.size() - kernel_size_.size(),
      2U,
      errors::InvalidArgument(
          "The input size %d minus the kernel size %d should equal to 2.",
          x_dims.size(),
          kernel_size_.size()));
  PADDLE_ENFORCE_EQ(
      kernel_size_.size(),
      strides.size(),
      errors::InvalidArgument(
          "Strides size %d and pooling size %d should be the same.",
          strides.size(),
          kernel_size_.size()));
  PADDLE_ENFORCE_EQ(
      kernel_size_.size(),
      paddings_.size(),
      errors::InvalidArgument(
          "Paddings size %d and pooling size %d should be the same.",
          paddings_.size(),
          kernel_size_.size()));

  std::vector<int64_t> output_shape({x_dims[0], x_dims[1]});
  if (adaptive) {
    output_shape.insert(
        output_shape.end(), kernel_size_.begin(), kernel_size_.end());
  } else {
    for (size_t i = 0; i < kernel_size_.size(); ++i) {
      if ((!config.is_runtime) && (x_dims[i + 2] < 0)) {
        output_shape.push_back(x_dims[i + 2]);
      } else {
        output_shape.push_back(funcs::MaxPoolOutputSize(
            x_dims[i + 2], kernel_size_[i], paddings_[i], strides[i]));
      }
    }
  }

  out->set_dims(make_ddim(output_shape));
  out->set_dtype(x.dtype());

  mask->set_dims(make_ddim(output_shape));
  mask->set_dtype(paddle::experimental::CppTypeToDataType<int>::Type());
}

void MeanAllInferMeta(const MetaTensor& x, MetaTensor* out) {
  out->set_dims(phi::make_ddim({1}));
  out->set_dtype(x.dtype());
  out->set_layout(x.layout());
}

void ModeInferMeta(const MetaTensor& x,
                   int axis,
                   bool keepdim,
                   MetaTensor* out,
                   MetaTensor* indices) {
  auto input_dims = x.dims();
  const int& dim_size = input_dims.size();
  PADDLE_ENFORCE_EQ(
      (axis < dim_size) && (axis >= (-1 * dim_size)),
      true,
      errors::InvalidArgument(
          "the axis of ModeOp must be [-%d, %d), but you set axis is %d",
          dim_size,
          dim_size,
          axis));
  PADDLE_ENFORCE_GE(
      input_dims.size(),
      1,
      errors::InvalidArgument("input of ModeOp must have >= 1d shape"));
  if (axis < 0) axis += dim_size;
  std::vector<int64_t> dimvec;
  for (int64_t i = 0; i < axis; i++) {
    dimvec.emplace_back(input_dims[i]);
  }
  if (keepdim) {
    dimvec.emplace_back(static_cast<int64_t>(1));
  }
  for (int64_t i = axis + 1; i < dim_size; i++) {
    dimvec.emplace_back(input_dims[i]);
  }
  DDim dims = phi::make_ddim(dimvec);
  PADDLE_ENFORCE_GE(input_dims.size(),
                    1,
                    errors::InvalidArgument("input shape should >= 1d"));
  out->set_dims(dims);
  out->share_lod(x);
  out->set_dtype(x.dtype());

  indices->set_dims(dims);
  indices->share_lod(x);
  indices->set_dtype(x.dtype());
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

void NormInferMeta(const MetaTensor& x,
                   int axis,
                   float epsilon,
                   bool is_test,
                   MetaTensor* out,
                   MetaTensor* norm) {
  auto xdim = x.dims();
  out->set_dims(x.dims());
  out->set_dtype(x.dtype());

  if (is_test == false) {
    if (axis < 0) axis = xdim.size() + axis;
    xdim[axis] = 1;
    norm->set_dims(xdim);
    norm->set_dtype(x.dtype());
  }
}

void PadInferMeta(const MetaTensor& input,
                  const std::vector<int>& paddings,
                  float pad_value,
                  MetaTensor* out,
                  MetaConfig config) {
  auto x_dim = input.dims();
  PADDLE_ENFORCE_EQ(
      static_cast<int>(paddings.size()),
      x_dim.size() * 2,
      phi::errors::InvalidArgument(
          "Size of 'paddings' dimension should be equal to 2 * size of "
          "Input(X)'s dimension, but received (size of 'paddings' dimension "
          "is) %d vs (2 * size of Input(X)'s dimension is) %d.",
          static_cast<int>(paddings.size()),
          x_dim.size() * 2));
  for (size_t i = 0; i < paddings.size(); ++i) {
    PADDLE_ENFORCE_GE(paddings[i],
                      0,
                      phi::errors::InvalidArgument(
                          "The element of 'paddings' should >= 0, but "
                          "received %d for index %d.",
                          paddings[i],
                          static_cast<int>(i)));
  }
  std::vector<int64_t> out_dims(x_dim.size());
  for (int i = 0; i < x_dim.size(); ++i) {
    if ((!config.is_runtime) && (x_dim[i] == -1)) {
      out_dims[i] = -1;
    } else {
      out_dims[i] = x_dim[i] + paddings[i * 2] + paddings[i * 2 + 1];
    }
  }
  out->set_dims(phi::make_ddim(out_dims));
  if (out_dims[0] == x_dim[0]) {
    // Only pass LoD when the first dimension is equal between
    // output and input.
    out->share_lod(input);
  }
  out->set_dtype(input.dtype());
}

void Pad3dInferMeta(const MetaTensor& x,
                    const IntArray& paddings_int_array,
                    const std::string& mode,
                    float value,
                    const std::string& data_format,
                    MetaTensor* out,
                    MetaConfig config) {
  auto x_dim = x.dims();
  PADDLE_ENFORCE_EQ(x_dim.size(),
                    5,
                    errors::InvalidArgument(
                        "The size of Input(X)'s dimension should be equal to "
                        "5, but received %d. ",
                        x_dim.size()));

  std::vector<int64_t> out_dims(x_dim.size(), -1);
  out_dims[0] = x_dim[0];
  auto& paddings = paddings_int_array.GetData();
  if (data_format == "NCDHW") {
    out_dims[1] = x_dim[1];
  } else {
    out_dims[4] = x_dim[4];
  }
  if (paddings_int_array.FromTensor()) {
    if (config.is_runtime) {
      PADDLE_ENFORCE_EQ(
          paddings.size(),
          6,
          errors::InvalidArgument("Shape of Input(Paddings) should be equal to "
                                  "[6], but received [%d].",
                                  paddings.size()));
      if (data_format == "NCDHW") {
        out_dims[2] = x_dim[2] + paddings[4] + paddings[5];
        out_dims[3] = x_dim[3] + paddings[2] + paddings[3];
        out_dims[4] = x_dim[4] + paddings[0] + paddings[1];
      } else {
        out_dims[1] = x_dim[1] + paddings[4] + paddings[5];
        out_dims[2] = x_dim[2] + paddings[2] + paddings[3];
        out_dims[3] = x_dim[3] + paddings[0] + paddings[1];
      }
    }
  } else {
    PADDLE_ENFORCE_EQ(
        paddings.size(),
        6,
        errors::InvalidArgument(
            "Size of paddings should be equal to 6, but received %d.",
            static_cast<int>(paddings.size())));
    if (data_format == "NCDHW") {
      out_dims[2] = ((!config.is_runtime) && (x_dim[2] < 0))
                        ? x_dim[2]
                        : (x_dim[2] + paddings[4] + paddings[5]);  // depth

      out_dims[3] = ((!config.is_runtime) && (x_dim[3] < 0))
                        ? x_dim[3]
                        : (x_dim[3] + paddings[2] + paddings[3]);  // height

      out_dims[4] = ((!config.is_runtime) && (x_dim[4] < 0))
                        ? x_dim[4]
                        : (x_dim[4] + paddings[0] + paddings[1]);  // width
    } else {                                                       // NDHWC
      out_dims[1] = ((!config.is_runtime) && (x_dim[1] < 0))
                        ? x_dim[1]
                        : (x_dim[1] + paddings[4] + paddings[5]);  // depth
      out_dims[2] = ((!config.is_runtime) && (x_dim[2] < 0))
                        ? x_dim[2]
                        : (x_dim[2] + paddings[2] + paddings[3]);  // height
      out_dims[3] = ((!config.is_runtime) && (x_dim[3] < 0))
                        ? x_dim[3]
                        : (x_dim[3] + paddings[0] + paddings[1]);  // width
    }
  }

  out->set_dims(phi::make_ddim(out_dims));
  out->set_dtype(x.dtype());
  out->share_lod(x);
}

void PixelShuffleInferMeta(const MetaTensor& x,
                           int upscale_factor,
                           const std::string& data_format,
                           MetaTensor* out) {
  auto input_dims = x.dims();
  PADDLE_ENFORCE_EQ(input_dims.size(),
                    4,
                    phi::errors::InvalidArgument(
                        "Input should be a 4-D tensor of format [N, C, H, W] "
                        "or [N, H, W, C], but got %u.",
                        input_dims.size()));

  const bool channel_last = (data_format == "NHWC");

  if (!channel_last) {
    PADDLE_ENFORCE_EQ(input_dims[1] % (upscale_factor * upscale_factor),
                      0,
                      phi::errors::InvalidArgument(
                          "The square of upscale_factor[%u] should divide the "
                          "number of channel[%u]",
                          upscale_factor * upscale_factor,
                          input_dims[1]));
  } else {
    PADDLE_ENFORCE_EQ(input_dims[3] % (upscale_factor * upscale_factor),
                      0,
                      phi::errors::InvalidArgument(
                          "The square of upscale_factor[%u] should divide the "
                          "number of channel[%u]",
                          upscale_factor * upscale_factor,
                          input_dims[3]));
  }
  auto output_dims = input_dims;
  output_dims[0] = input_dims[0];
  if (!channel_last) {
    output_dims[1] = input_dims[1] / (upscale_factor * upscale_factor);
    output_dims[2] = input_dims[2] * upscale_factor;
    output_dims[3] = input_dims[3] * upscale_factor;
  } else {
    output_dims[1] = input_dims[1] * upscale_factor;
    output_dims[2] = input_dims[2] * upscale_factor;
    output_dims[3] = input_dims[3] / (upscale_factor * upscale_factor);
  }
  out->set_dtype(x.dtype());
  out->set_dims(output_dims);
}

void PixelShuffleGradInferMeta(const MetaTensor& out_grad,
                               int upscale_factor,
                               const std::string& data_format,
                               MetaTensor* x_grad) {
  auto do_dims = out_grad.dims();
  PADDLE_ENFORCE_EQ(do_dims.size(),
                    4,
                    phi::errors::InvalidArgument(
                        "Input should be a 4-D tensor of format [N, C, H, W] "
                        "or [N, H, W, C], but got %u.",
                        do_dims.size()));

  const bool channel_last = (data_format == "NHWC");

  auto dx_dims = do_dims;
  dx_dims[0] = do_dims[0];

  if (!channel_last) {
    dx_dims[1] = do_dims[1] * (upscale_factor * upscale_factor);
    dx_dims[2] = do_dims[2] / upscale_factor;
    dx_dims[3] = do_dims[3] / upscale_factor;
  } else {
    dx_dims[1] = do_dims[1] / upscale_factor;
    dx_dims[2] = do_dims[2] / upscale_factor;
    dx_dims[3] = do_dims[3] * (upscale_factor * upscale_factor);
  }
  x_grad->set_dims(dx_dims);
  x_grad->set_dtype(out_grad.dtype());
}

void PNormInferMeta(const MetaTensor& x,
                    float porder,
                    int axis,
                    float epsilon,
                    bool keepdim,
                    bool asvector,
                    MetaTensor* out) {
  auto x_dim = x.dims();
  auto x_rank = x_dim.size();

  PADDLE_ENFORCE_GE(axis,
                    -x_rank,
                    errors::InvalidArgument(
                        "Attr(axis) value should be in range [-R, R-1], R is "
                        "the rank of Input(X). But received axis: %d, R: %d. "
                        "Current Input(X)'s shape is=[%s].",
                        axis,
                        x_rank,
                        x_dim));
  PADDLE_ENFORCE_LT(axis,
                    x_rank,
                    errors::InvalidArgument(
                        "Attr(axis) value should be in range [-R, R-1], R is "
                        "the rank of Input(X). But received axis: %d, R: %d. "
                        "Current Input(X)'s shape is=[%s].",
                        axis,
                        x_rank,
                        x_dim));

  std::vector<int> reduce_dims;
  if (asvector) {
    reduce_dims.emplace_back(1);
    if (keepdim) {
      for (int i = 1; i < x_dim.size(); ++i) {
        reduce_dims.emplace_back(1);
      }
      x_dim = phi::make_ddim(reduce_dims);
    }
  } else {
    if (axis < 0) axis = x_dim.size() + axis;
    for (int i = 0; i < x_dim.size(); ++i) {
      if (i != axis) reduce_dims.emplace_back(x_dim[i]);
    }
    if (reduce_dims.size() == 0) {
      reduce_dims.emplace_back(1);
    }
  }
  x_dim[axis] = 1;

  if (keepdim) {
    out->set_dims(x_dim);
  } else {
    out->set_dims(phi::make_ddim(reduce_dims));
  }
  out->set_dtype(x.dtype());
}

void PoolInferMeta(const MetaTensor& x,
                   const std::vector<int>& kernel_size,
                   const std::vector<int>& strides,
                   const std::vector<int>& paddings,
                   bool ceil_mode,
                   bool exclusive,
                   const std::string& data_format,
                   const std::string& pooling_type,
                   bool global_pooling,
                   bool adaptive,
                   const std::string& padding_algorithm,
                   MetaTensor* out,
                   MetaConfig config) {
  std::vector<int> paddings_ = paddings;
  std::vector<int> kernel_size_ = kernel_size;

  auto x_dims = x.dims();
  PADDLE_ENFORCE_EQ(
      x_dims.size() == 4 || x_dims.size() == 5,
      true,
      errors::InvalidArgument(
          "the input of Op(pool) should be 4-D or 5-D Tensor. But "
          "received: %u-D Tensor and it's shape is [%s].",
          x_dims.size(),
          x_dims));

  PADDLE_ENFORCE_EQ(x_dims.size() - kernel_size_.size(),
                    2U,
                    errors::InvalidArgument(
                        "the dimension of input minus the size of "
                        "Attr(kernel_size_) must be euqal to 2 in Op(pool). "
                        "But received: the dimension of input minus the size "
                        "of Attr(kernel_size_) is %d, the "
                        "input's dimension is %d, the shape of input "
                        "is [%s], the Attr(kernel_size_)'s size is %d, the "
                        "Attr(kernel_size_) is [%s].",
                        x_dims.size() - kernel_size_.size(),
                        x_dims.size(),
                        x_dims,
                        kernel_size_.size(),
                        make_ddim(kernel_size_)));

  PADDLE_ENFORCE_EQ(
      kernel_size_.size(),
      strides.size(),
      errors::InvalidArgument(
          "the size of Attr(kernel_size_) and Attr(strides) in "
          "Op(pool) must be equal. "
          "But received: Attr(kernel_size_)'s size is %d, Attr(strides)'s "
          "size is %d, Attr(kernel_size_) is [%s], Attr(strides)is [%s].",
          kernel_size_.size(),
          strides.size(),
          make_ddim(kernel_size_),
          make_ddim(strides)));

  // MKL-DNN Kernels are using NCHW order of dims description
  // so we ignore data_format consideration for MKL-DNN kernel
  const bool channel_last = (config.is_run_mkldnn_kernel == false) &&
                            (data_format == "NHWC" || data_format == "NDHWC");

  // update paddings if "SAME" or global_pooling
  DDim data_dims;
  if (channel_last) {
    data_dims = slice_ddim(x_dims, 1, x_dims.size() - 1);
  } else {
    data_dims = slice_ddim(x_dims, 2, x_dims.size());
  }
  funcs::UpdatePadding(&paddings_,
                       global_pooling,
                       adaptive,
                       padding_algorithm,
                       data_dims,
                       strides,
                       kernel_size_);

  if (global_pooling) {
    funcs::UpdateKernelSize(&kernel_size_, data_dims);
  }

  std::vector<int64_t> output_shape;
  if (adaptive) {
    output_shape.insert(
        output_shape.end(), kernel_size_.begin(), kernel_size_.end());
  } else {
    for (int i = 0; i < data_dims.size(); ++i) {
      if ((!config.is_runtime) && (data_dims[i] < 0)) {
        output_shape.push_back(data_dims[i]);
      } else {
        output_shape.push_back(funcs::PoolOutputSize(data_dims[i],
                                                     kernel_size_[i],
                                                     paddings_[2 * i],
                                                     paddings_[2 * i + 1],
                                                     strides[i],
                                                     ceil_mode));
      }
    }
  }

  // output_N = input_N
  output_shape.insert(output_shape.begin(), x_dims[0]);
  // output_C = input_C
  if (channel_last) {
    output_shape.push_back(x_dims[x_dims.size() - 1]);
  } else {
    output_shape.insert(output_shape.begin() + 1, x_dims[1]);
  }

  out->set_dims(make_ddim(output_shape));
  out->share_lod(x);
  out->set_dtype(x.dtype());
}

void RealAndImagInferMeta(const MetaTensor& x, MetaTensor* out) {
  out->set_dims(x.dims());
  out->set_dtype(dtype::ToReal(x.dtype()));
  out->set_layout(x.layout());
}

void QrInferMeta(const MetaTensor& x,
                 const std::string& mode,
                 MetaTensor* q,
                 MetaTensor* r) {
  auto x_dims = x.dims();
  int x_rank = x_dims.size();
  PADDLE_ENFORCE_GE(
      x_dims.size(),
      2,
      phi::errors::InvalidArgument("the rank of input must greater than 2"));
  bool compute_q;
  bool reduced_mode;
  int m = x_dims[x_rank - 2];
  int n = x_dims[x_rank - 1];
  int min_mn = std::min(m, n);
  std::tie(compute_q, reduced_mode) = phi::funcs::ParseQrMode(mode);

  if (compute_q) {
    int k = reduced_mode ? min_mn : m;
    auto q_dims_vec = phi::vectorize(x_dims);
    q_dims_vec[q_dims_vec.size() - 1] = k;
    q->set_dims(phi::make_ddim(q_dims_vec));
  } else {
    q->set_dims(phi::make_ddim({0}));
  }

  int k = reduced_mode ? min_mn : m;
  auto r_dims_vec = phi::vectorize(x_dims);
  r_dims_vec[r_dims_vec.size() - 2] = k;
  r_dims_vec[r_dims_vec.size() - 1] = n;
  r->set_dims(phi::make_ddim(r_dims_vec));

  q->share_lod(x);
  r->share_lod(x);
  q->set_dtype(x.dtype());
  r->set_dtype(x.dtype());
}

DDim ReduceInferDim(const MetaTensor& x,
                    const std::vector<int64_t>& axis,
                    bool keep_dim,
                    bool reduce_all) {
  auto x_rank = x.dims().size();

  std::vector<int64_t> formated_axis = axis;
  for (size_t i = 0; i < axis.size(); ++i) {
    PADDLE_ENFORCE_LT(axis[i],
                      x_rank,
                      errors::InvalidArgument(
                          "The reduce dim index %d should be in the "
                          "range [-dimension(X), dimension(X)] "
                          "which dimesion = %d. But received dim index = %d.",
                          i,
                          x_rank,
                          axis[i]));
    PADDLE_ENFORCE_GE(axis[i],
                      -x_rank,
                      errors::InvalidArgument(
                          "The reduce dim index %d should be in the "
                          "range [-dimension(X), dimension(X)] "
                          "which dimesion = %d. But received dim index = %d.",
                          i,
                          x_rank,
                          axis[i]));

    if (axis[i] < 0) {
      formated_axis[i] = axis[i] + x_rank;
    }
  }

  bool full_dim = true;
  std::set<int64_t> dims_set(formated_axis.begin(), formated_axis.end());
  for (int64_t i = 0; i < x.dims().size(); ++i) {
    if (dims_set.find(i) == dims_set.end()) {
      full_dim = false;
      break;
    }
  }
  reduce_all = reduce_all || full_dim;

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

  return out_dim;
}

void ReduceInferMeta(const MetaTensor& x,
                     const std::vector<int64_t>& axis,
                     bool keep_dim,
                     MetaTensor* out) {
  bool reduce_all = false;
  ReduceInferMetaBase(x, axis, keep_dim, reduce_all, out);
}

void ReduceInferMetaBase(const MetaTensor& x,
                         const std::vector<int64_t>& axis,
                         bool keep_dim,
                         bool reduce_all,
                         MetaTensor* out) {
  DDim out_dim = ReduceInferDim(x, axis, keep_dim, reduce_all);
  out->set_dims(out_dim);
  out->set_dtype(x.dtype());
  out->set_layout(x.layout());
}

void ReshapeInferMeta(const MetaTensor& x,
                      const IntArray& shape,
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
                                const IntArray& shape,
                                MetaTensor* out,
                                MetaTensor* xshape,
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

void ReverseInferMeta(const MetaTensor& x,
                      const std::vector<int>& axis,
                      MetaTensor* out) {
  PADDLE_ENFORCE_NE(axis.empty(),
                    true,
                    phi::errors::InvalidArgument("'axis' can not be empty."));
  const auto& x_dims = x.dims();
  for (int a : axis) {
    PADDLE_ENFORCE_LT(a,
                      x_dims.size(),
                      phi::errors::OutOfRange(
                          "The axis must be less than input tensor's rank. "
                          "but got %d >= %d",
                          a,
                          x_dims.size()));
    PADDLE_ENFORCE_GE(
        a,
        -x_dims.size(),
        phi::errors::OutOfRange(
            "The axis must be greater than the negative number of "
            "input tensor's rank, but got %d < %d",
            a,
            -x_dims.size()));
  }
  out->share_meta(x);
}

void RollInferMeta(const MetaTensor& x,
                   const IntArray& shifts,
                   const std::vector<int64_t>& axis,
                   MetaTensor* out) {
  auto shifts_data = shifts.GetData();

  if (axis.size() != 0) {
    PADDLE_ENFORCE_EQ(
        axis.size(),
        shifts_data.size(),
        phi::errors::InvalidArgument("When dims.size() != 0, dims.size() "
                                     "should be equal to "
                                     "shifts.size(). But received "
                                     "dims.size() = %d, shifts.size() = %d",
                                     axis.size(),
                                     shifts_data.size()));
  } else {
    PADDLE_ENFORCE_EQ(
        shifts_data.size(),
        1,
        phi::errors::InvalidArgument("When dims.size() == 0, shifts.size() "
                                     "should be equal to 1, But received "
                                     "shifts.size() = %d",
                                     shifts_data.size()));
  }

  out->set_dims(x.dims());
  out->share_lod(x);
  out->set_dtype(x.dtype());
}

void SetValueInferMeta(const MetaTensor& x, MetaTensor* out) {
  auto in_dims = x.dims();
  PADDLE_ENFORCE_LT(
      in_dims.size(),
      7,
      phi::errors::InvalidArgument(
          "The rank of input should be less than 7, but received %d.",
          in_dims.size()));
}

void ShapeInferMeta(const MetaTensor& input, MetaTensor* out) {
  auto in_dim = input.dims();
  out->set_dims(phi::make_ddim({in_dim.size()}));
  out->set_dtype(DataType::INT32);
}

void ShardIndexInferMeta(const MetaTensor& in,
                         int index_num,
                         int nshards,
                         int shard_id,
                         int ignore_value,
                         MetaTensor* out,
                         MetaConfig config) {
  auto x_dims = in.dims();
  PADDLE_ENFORCE_GE(
      x_dims.size(),
      2,
      phi::errors::InvalidArgument("Rank of Input(X) should be at least 2, "
                                   "but the value given is %d.",
                                   x_dims.size()));
  if (config.is_runtime || x_dims[x_dims.size() - 1] > 0) {
    PADDLE_ENFORCE_EQ(x_dims[x_dims.size() - 1],
                      1U,
                      phi::errors::InvalidArgument(
                          "The last dimension of Input(X) should be 1, "
                          "but the value given is %d.",
                          x_dims[x_dims.size() - 1]));
  }

  out->set_dims(x_dims);
  out->share_lod(in);
  out->set_dtype(in.dtype());
}

void SizeInferMeta(const MetaTensor& input, MetaTensor* out) {
  out->set_dtype(DataType::INT64);
  out->set_dims({1});
}

void SliceRawInferMeta(const MetaTensor& input,
                       const std::vector<int64_t>& axes,
                       const IntArray& starts_arr,
                       const IntArray& ends_arr,
                       const std::vector<int64_t>& infer_flags_t,
                       const std::vector<int64_t>& decrease_axis,
                       MetaTensor* out,
                       MetaConfig config) {
  auto in_dims = input.dims();
  PADDLE_ENFORCE_LT(
      in_dims.size(),
      7,
      phi::errors::InvalidArgument("The rank of input should be less than 7."));
  DDim out_dims(in_dims);

  std::vector<int64_t> infer_flags = infer_flags_t;
  if (infer_flags.empty()) {
    // Initialize infer_flags with 1.
    // To be compatible with other op tests in which infer_flags is not set.
    infer_flags = std::vector<int64_t>(axes.size(), 1);
  }

  // 2.1 Check attrs.
  std::vector<int64_t> starts = starts_arr.GetData();
  std::vector<int64_t> ends = ends_arr.GetData();

  phi::funcs::CheckAndUpdateSliceAttrs<int64_t>(
      in_dims, axes, &starts, &ends, nullptr, &infer_flags);

  auto slice_dims = phi::funcs::GetSliceDims<int64_t>(
      in_dims, axes, starts, ends, nullptr, &infer_flags);
  if (config.is_runtime) {
    out_dims = phi::funcs::GetDecreasedDims<int64_t>(
        slice_dims, decrease_axis, &infer_flags);
  } else {
    out_dims = phi::funcs::GetDecreasedDims<int64_t>(
        slice_dims, decrease_axis, nullptr);
  }

  out->set_dims(out_dims);
  if (axes.size() > 0 && axes[0] != 0) {
    out->share_lod(input);
  }
}

void SoftmaxInferMeta(const MetaTensor& x, int axis, MetaTensor* out) {
  auto dim_x = x.dims();
  auto rank_x = dim_x.size();
  PADDLE_ENFORCE_GE(axis,
                    -rank_x,
                    phi::errors::InvalidArgument(
                        "Attr(axis) value should be in range [-R, R-1], "
                        "R is the rank of Input(X)."));
  PADDLE_ENFORCE_LT(axis,
                    rank_x,
                    phi::errors::InvalidArgument(
                        "Attr(axis) value should be in range [-R, R-1], "
                        "R is the rank of Input(X)."));

  out->set_dims(x.dims());
  out->set_dtype(x.dtype());
  out->share_lod(x);
}

void SplitInferMeta(const MetaTensor& x,
                    const IntArray& num_or_sections,
                    const Scalar& axis,
                    std::vector<MetaTensor*> out,
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
      out[i]->set_dtype(x.dtype());
      out[i]->set_dims(out_dims[i]);
      out[i]->set_layout(x.layout());
    } else {
      out[i]->set_dtype(x.dtype());
      out[i]->set_dims(out_dims[i]);
      out[i]->set_layout(x.layout());
      out[i]->share_lod(x);
    }
  }
}

void SqueezeInferMeta(const MetaTensor& x,
                      const std::vector<int>& axes,
                      MetaTensor* xshape,
                      MetaTensor* out) {
  const auto& x_dims = x.dims();
  // Check input tensor dims (<6) Eigen limit.
  PADDLE_ENFORCE_LE(x_dims.size(),
                    6,
                    phi::errors::InvalidArgument(
                        "The dimensions of Input(X) "
                        "should be in the range of [1, 6] (Eigen limit)."
                        "But received X's dimensions = %d, X's shape = [%s].",
                        x_dims.size(),
                        x_dims));

  auto out_dims = funcs::GetOutputSqueezeShape(axes, x_dims, false);
  out->set_dims(out_dims);
  if (x_dims[0] == out_dims[0]) {
    // Only pass LoD when the first dimension of output and Input(X)
    // are the same.
    out->share_lod(x);
  }

  std::vector<int64_t> xshape_dims(x_dims.size() + 1);
  xshape_dims[0] = 0;
  for (int i = 0; i < x_dims.size(); ++i) {
    xshape_dims[i + 1] = x_dims[i];
  }
  xshape->set_dims(phi::make_ddim(xshape_dims));
  xshape->share_lod(x);
  xshape->set_dtype(x.dtype());
  out->set_dtype(x.dtype());
}

void StridedSliceRawInferMeta(const MetaTensor& x,
                              const std::vector<int>& axes,
                              const IntArray& starts,
                              const IntArray& ends,
                              const IntArray& strides,
                              const std::vector<int>& infer_flags,
                              const std::vector<int>& decrease_axis,
                              MetaTensor* out,
                              MetaConfig config) {
  auto in_dims = x.dims();
  PADDLE_ENFORCE_LT(
      in_dims.size(),
      7,
      errors::InvalidArgument(
          "The dimension of StridedSlice operator's input should be less "
          "than 7, but received dimension is %d.",
          in_dims.size()));

  auto starts_ = starts.GetData();
  auto ends_ = ends.GetData();
  auto strides_ = strides.GetData();

  auto starts_size = starts_.size();
  auto ends_size = ends_.size();
  auto strides_size = strides_.size();

  for (size_t i = 0; i < axes.size(); ++i) {
    PADDLE_ENFORCE_GE(
        axes[i],
        0,
        errors::InvalidArgument("The axis should be greater than or equal to 0."
                                "But received %d of axes[%d]",
                                axes[i],
                                i));
    PADDLE_ENFORCE_LT(
        axes[i],
        in_dims.size(),
        errors::InvalidArgument(
            "The axes should be less than or equal to input tensor's rank."
            "But received %d of axes[%d], input tensor shape [%d]",
            axes[i],
            i,
            in_dims.size()));
  }

  auto tensor_input = false;
  auto HasInput = [](const IntArray& arr) { return arr.FromTensor(); };
  if (HasInput(starts) || HasInput(ends) || HasInput(strides)) {
    tensor_input = true;
  }
  if (!HasInput(ends)) {
    PADDLE_ENFORCE_EQ(
        ends_size,
        axes.size(),
        errors::InvalidArgument(
            "The size of ends attribute in StridedSlice operator is not "
            "equal to the size of axes attribute. The ends attribute's size "
            "is %d, axes attribute's size is %d.",
            ends_size,
            axes.size()));
  }
  if (!HasInput(starts)) {
    PADDLE_ENFORCE_EQ(
        starts_size,
        axes.size(),
        errors::InvalidArgument(
            "The size of starts attribute in StridedSlice operator is not "
            "equal to the size of axes attribute. The starts attribute's "
            "size is %d, axes attribute's size is %d.",
            starts_size,
            axes.size()));
  }
  if (!HasInput(strides)) {
    PADDLE_ENFORCE_EQ(
        strides_size,
        axes.size(),
        errors::InvalidArgument(
            "The size of strides attribute in StridedSlice operator is not "
            "equal to the size of axes attribute. The strides attribute's "
            "size is %d, axes attribute's size is %d.",
            strides_size,
            axes.size()));
  }
  // we need to analysis strided slice op is valid for
  // the parameter that we get from python front
  std::vector<int64_t> out_dims_vector(in_dims.size(), -1);
  if (!tensor_input || config.is_runtime) {
    phi::funcs::StridedSliceOutDims(starts_,
                                    ends_,
                                    strides_,
                                    axes,
                                    infer_flags,
                                    in_dims,
                                    decrease_axis,
                                    out_dims_vector.data(),
                                    axes.size(),
                                    true);
  }
  DDim out_dims(phi::make_ddim(out_dims_vector));
  // generate new shape
  if (decrease_axis.size() > 0) {
    std::vector<int64_t> new_out_shape;
    for (size_t i = 0; i < decrease_axis.size(); ++i) {
      if (config.is_runtime && infer_flags[i] != -1) {
        PADDLE_ENFORCE_EQ(out_dims[decrease_axis[i]],
                          1,
                          errors::InvalidArgument(
                              "the size of decrease dimension should be 1, "
                              "but received %d.",
                              out_dims[decrease_axis[i]]));
      }
      out_dims[decrease_axis[i]] = 0;
    }

    for (int i = 0; i < out_dims.size(); ++i) {
      if (out_dims[i] != 0) {
        new_out_shape.push_back(out_dims[i]);
      }
    }
    if (new_out_shape.size() == 0) {
      new_out_shape.push_back(1);
    }
    out_dims = phi::make_ddim(new_out_shape);
  }
  VLOG(1) << "out_dims: " << out_dims;
  out->set_dims(out_dims);
  out->share_lod(x);
  out->set_dtype(x.dtype());
}

void StridedSliceInferMeta(const MetaTensor& x,
                           const std::vector<int>& axes,
                           const IntArray& starts,
                           const IntArray& ends,
                           const IntArray& strides,
                           MetaTensor* out,
                           MetaConfig config) {
  std::vector<int> infer_flags(axes.size(), 1);
  std::vector<int> decrease_axis;
  StridedSliceRawInferMeta(
      x, axes, starts, ends, strides, infer_flags, decrease_axis, out, config);
}

/*  Why not use SumRawInferMeta directly?
    Because we need make InferMetaFunction's args follow the design of api.yaml
*/
void SumInferMeta(const MetaTensor& x,
                  const std::vector<int64_t>& axis,
                  DataType dtype,
                  bool keep_dim,
                  MetaTensor* out) {
  bool reduce_all = false;
  SumRawInferMeta(x, axis, keep_dim, reduce_all, dtype, out);
}

void SumRawInferMeta(const MetaTensor& x,
                     const std::vector<int64_t>& axis,
                     bool keep_dim,
                     bool reduce_all,
                     DataType dtype,
                     MetaTensor* out) {
  DDim out_dim = ReduceInferDim(x, axis, keep_dim, reduce_all);

  DataType out_dtype;
  if (dtype != DataType::UNDEFINED) {
    out_dtype = dtype;
  } else {
    if (x.dtype() == DataType::BOOL || x.dtype() == DataType::INT32) {
      out_dtype = DataType::INT64;
    } else {
      out_dtype = x.dtype();
    }
  }

  out->set_dims(out_dim);
  out->set_dtype(out_dtype);
  out->set_layout(x.layout());
}

void TemporalShiftInferMeta(const MetaTensor& x,
                            int seg_num,
                            float shift_ratio,
                            const std::string& data_format,
                            MetaTensor* out,
                            MetaConfig config) {
  auto dim_x = x.dims();
  PADDLE_ENFORCE_EQ(dim_x.size(),
                    4,
                    phi::errors::InvalidArgument(
                        "Input(X) rank should be 4 in shape of [N*T, C, H, "
                        "W], but received X rank(%d)",
                        dim_x.size()));

  PADDLE_ENFORCE_GT(
      seg_num,
      0,
      phi::errors::InvalidArgument(
          "Attr(seg_num) should be greater than 0, but received %d", seg_num));
  PADDLE_ENFORCE_GT(
      shift_ratio,
      0.,
      phi::errors::InvalidArgument(
          "Attr(shift_ratio) should be greater than 0, but received %d",
          shift_ratio));
  PADDLE_ENFORCE_LT(
      shift_ratio,
      0.5,
      phi::errors::InvalidArgument(
          "Attr(shift_ratio) should be less than 0.5, but received %d",
          shift_ratio));

  if (config.is_runtime) {
    PADDLE_ENFORCE_EQ(dim_x[0] % seg_num,
                      0,
                      phi::errors::InvalidArgument(
                          "Input(X) dimension[0] should be divided exactly "
                          "by Attr(seg_num), but received X dimension[0](%d) "
                          "mod seg_num(%d) != 0",
                          dim_x[0],
                          seg_num));
  }

  out->share_meta(x);
}

void TileInferMeta(const MetaTensor& x,
                   const IntArray& repeat_times,
                   MetaTensor* out,
                   MetaConfig config) {
#define MAX_RANK_SUPPORTED 6

  auto repeat_times_data = repeat_times.GetData();
  auto x_dims = x.dims();
  if (repeat_times_data.size() == 0) {
    repeat_times_data = std::vector<int64_t>(x_dims.size(), -1);
  }

  PADDLE_ENFORCE_LE(
      x_dims.size(),
      MAX_RANK_SUPPORTED,
      errors::InvalidArgument(
          "The rank of the input 'x' for tile op "
          "must not be greater than %d, but the value received is %d.",
          MAX_RANK_SUPPORTED,
          x_dims.size()));
  PADDLE_ENFORCE_LE(
      repeat_times_data.size(),
      MAX_RANK_SUPPORTED,
      errors::InvalidArgument(
          "The size of the shape of input 'repeat_times' for tile op "
          "must not be greater than %d, but the value received is %d.",
          MAX_RANK_SUPPORTED,
          repeat_times_data.size()));
  PADDLE_ENFORCE_GE(
      repeat_times_data.size(),
      1,
      errors::InvalidArgument(
          "The size of the shape of input 'repeat_times' for tile op "
          "must be positive integers, but the value received is %d.",
          repeat_times_data.size()));

  auto out_rank =
      std::max(static_cast<size_t>(x_dims.size()), repeat_times_data.size());
  std::vector<int64_t> out_shape(out_rank);
  auto x_dim_vec = phi::vectorize<int>(x_dims);
  if (x_dim_vec.size() > repeat_times_data.size()) {
    auto diff = x_dim_vec.size() - repeat_times_data.size();
    repeat_times_data.insert(repeat_times_data.begin(), diff, -1);
  } else {
    auto diff = repeat_times_data.size() - x_dim_vec.size();
    x_dim_vec.insert(x_dim_vec.begin(), diff, -1);
  }
  for (size_t i = 0; i < repeat_times_data.size(); ++i) {
    if (x_dim_vec[i] == -1 || repeat_times_data[i] == -1) {
      out_shape[i] = -1;
    } else {
      PADDLE_ENFORCE_GT(
          repeat_times_data[i],
          0,
          errors::InvalidArgument(
              "Every element of the input 'repeat_times' for tile op must be "
              "greater than 0, but the value given is %d.",
              repeat_times_data[i]));
      out_shape[i] = x_dim_vec[i] * repeat_times_data[i];
    }
  }

  out->set_dims(phi::make_ddim(out_shape));
  if (out_shape[0] == x_dims[0]) {
    out->share_lod(x);
  }
  out->set_dtype(x.dtype());
}

void TopKInferMeta(const MetaTensor& x,
                   const Scalar& k_scalar,
                   int axis,
                   bool largest,
                   bool sorted,
                   MetaTensor* out,
                   MetaTensor* indices,
                   MetaConfig config) {
  auto input_dims = x.dims();
  const int& dim_size = input_dims.size();
  PADDLE_ENFORCE_EQ(
      (axis < dim_size) && (axis >= (-1 * dim_size)),
      true,
      phi::errors::InvalidArgument(
          "the axis of topk must be [-%d, %d), but you set axis is %d",
          dim_size,
          dim_size,
          axis));

  if (axis < 0) axis += dim_size;

  int k = k_scalar.to<int>();
  if (k_scalar.FromTensor()) {
    k = -1;
  } else {
    PADDLE_ENFORCE_EQ(k >= 1,
                      true,
                      phi::errors::InvalidArgument(
                          "the attribute of k in the topk must >= 1 or be a "
                          "Tensor, but received %d .",
                          k));
  }

  PADDLE_ENFORCE_GE(
      input_dims.size(),
      1,
      phi::errors::InvalidArgument("input of topk must have >= 1d shape"));

  phi::DDim dims = input_dims;

  dims[axis] = k;
  out->set_dims(dims);
  out->share_lod(x);
  out->set_dtype(x.dtype());
  indices->set_dims(dims);
  indices->share_lod(x);
  indices->set_dtype(DataType::INT64);
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
  out->set_dtype(x.dtype());
}

void TransferLayoutInferMeta(const MetaTensor& x,
                             DataLayout layout,
                             MetaTensor* out) {
  out->set_dims(x.dims());
  out->set_dtype(x.dtype());
  out->set_layout(layout);
}

void TransposeInferMeta(const MetaTensor& x,
                        const std::vector<int>& axis,
                        MetaTensor* out) {
  auto x_dims = x.dims();
  size_t x_rank = x_dims.size();
  size_t axis_size = axis.size();

  PADDLE_ENFORCE_EQ(
      x_rank,
      axis_size,
      errors::InvalidArgument("The input tensor's dimension "
                              "should be equal to the axis's size. "
                              "But received input tensor's dimension is %d, "
                              "axis's size is %d",
                              x_rank,
                              axis_size));

  std::vector<int> count(axis_size, 0);
  for (size_t i = 0; i < axis_size; i++) {
    PADDLE_ENFORCE_GE(
        axis[i],
        0,
        errors::InvalidArgument("The axis should be greater than or equal to 0."
                                "But received %d of axis[%d]",
                                axis[i],
                                i));

    PADDLE_ENFORCE_EQ(
        axis[i] < static_cast<int>(axis_size) && ++count[axis[i]] == 1,
        true,
        errors::InvalidArgument(
            "Each element of Attribute axis should "
            "be a unique value range from 0 to (dims - 1), "
            "where the dims is the axis's size, "
            "unique value means this axis value can appear only once. "
            "But received axis[%d] is %d, axis_size is %d, "
            "count[axis[%d]] is %d",
            i,
            axis[i],
            axis_size,
            i,
            count[axis[i]]));
  }

  phi::DDim out_dims(x_dims);
  for (size_t i = 0; i < axis_size; ++i) {
    out_dims[i] = x_dims[axis[i]];
  }

  out->set_dims(out_dims);
  out->set_dtype(x.dtype());
}

void TransposeGradInferMeta(const MetaTensor& x,
                            const std::vector<int>& axis,
                            MetaTensor* out) {
  std::vector<int> reversed_axis(axis);
  for (size_t i = 0; i < axis.size(); i++) {
    reversed_axis[axis[i]] = i;
  }

  TransposeInferMeta(x, reversed_axis, out);
}

void UnbindInferMeta(const MetaTensor& x,
                     int axis,
                     std::vector<MetaTensor*> outs) {
  auto in_dims = x.dims();
  std::vector<int> out_dim;
  axis = axis < 0 ? in_dims.size() + axis : axis;
  for (int i = 0; i < in_dims.size(); ++i) {
    if (i != axis) out_dim.push_back(in_dims[i]);
  }
  auto out_dims = phi::make_ddim(out_dim);

  for (size_t i = 0; i < outs.size(); ++i) {
    outs[i]->set_dtype(x.dtype());
    outs[i]->set_dims(out_dims);
    outs[i]->set_layout(x.layout());
    outs[i]->share_lod(x);
  }
}

void TrilTriuInferMeta(const MetaTensor& x,
                       int diagonal,
                       bool lower,
                       MetaTensor* out) {
  const auto& x_dims = x.dims();
  PADDLE_ENFORCE_GE(x_dims.size(),
                    2,
                    phi::errors::InvalidArgument(
                        "Input(X)'s rank must be at least 2 in TrilTriuOp."));
  out->set_dims(x.dims());
  out->share_lod(x);
  out->set_dtype(x.dtype());
}

void UnchangedInferMeta(const MetaTensor& x, MetaTensor* out) {
  out->share_meta(x);
}

// meta x -> out without change, check if axis in range [-Rank(x), Rank(x)-1]
void UnchangedInferMetaCheckAxis(const MetaTensor& x,
                                 int axis,
                                 MetaTensor* out) {
  auto rank = x.dims().size();
  PADDLE_ENFORCE_GE(
      axis,
      -rank,
      phi::errors::InvalidArgument(
          "Attr(axis) value should be in range [-R, R-1], "
          "R is the rank of Input(X). But received axis: %d, R: %d.",
          axis,
          rank));
  PADDLE_ENFORCE_LT(
      axis,
      rank,
      phi::errors::InvalidArgument(
          "Attr(axis) value should be in range [-R, R-1], "
          "R is the rank of Input(X). But received axis: %d, R: %d.",
          axis,
          rank));
  out->share_meta(x);
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

void UniqueInferMeta(const MetaTensor& x,
                     bool return_index,
                     bool return_inverse,
                     bool return_counts,
                     const std::vector<int>& axis,
                     DataType dtype,
                     MetaTensor* out,
                     MetaTensor* indices,
                     MetaTensor* index,
                     MetaTensor* counts) {
  bool is_sorted = true;
  UniqueRawInferMeta(x,
                     return_index,
                     return_inverse,
                     return_counts,
                     axis,
                     dtype,
                     is_sorted,
                     out,
                     indices,
                     index,
                     counts);
}

void UniqueRawInferMeta(const MetaTensor& x,
                        bool return_index,
                        bool return_inverse,
                        bool return_counts,
                        const std::vector<int>& axis,
                        DataType dtype,
                        bool is_sorted,
                        MetaTensor* out,
                        MetaTensor* indices,
                        MetaTensor* index,
                        MetaTensor* counts) {
  if (!is_sorted) {
    PADDLE_ENFORCE_EQ(
        x.dims().size(),
        1,
        phi::errors::InvalidArgument("The Input(X) should be 1-D Tensor, "
                                     "But now the dims of Input(X) is %d.",
                                     x.dims().size()));
    out->set_dims(phi::make_ddim({-1}));
    index->set_dims(x.dims());
    return;
  }

  if (axis.empty()) {
    out->set_dims(phi::make_ddim({-1}));
    if (return_inverse) {
      index->set_dims(phi::make_ddim({phi::product(x.dims())}));
    }
  } else {
    int axis_value = axis[0];
    if (axis_value < 0) {
      axis_value += x.dims().size();
    }
    PADDLE_ENFORCE_LT(
        axis_value,
        x.dims().size(),
        phi::errors::InvalidArgument("The axis(%d) should be less than "
                                     "the dimension size(%d) of x.",
                                     axis_value,
                                     x.dims().size()));
    auto out_dims = x.dims();
    out_dims[axis_value] = -1;
    out->set_dims(out_dims);
    if (return_inverse) {
      index->set_dims(phi::make_ddim({x.dims()[axis_value]}));
    }
  }
  if (return_index) {
    indices->set_dims(phi::make_ddim({-1}));
  }
  if (return_counts) {
    counts->set_dims(phi::make_ddim({-1}));
  }
}

void UnsqueezeInferMeta(const MetaTensor& x,
                        const IntArray& axes,
                        MetaTensor* xshape,
                        MetaTensor* out,
                        MetaConfig config) {
  const auto& x_dims = x.dims();
  // Validity Check: input tensor dims (<6).
  PADDLE_ENFORCE_LE(x_dims.size(),
                    6,
                    phi::errors::InvalidArgument(
                        "Invalid "
                        "dimensions, the rank of Input(X) "
                        "should be in the range of [1, 6] (Eigen limit)"));
  if (!config.is_runtime && axes.FromTensor()) {
    // compile time infershape.  set all elements to -1.
    int output_size = x.dims().size() + axes.GetData().size();
    std::vector<int64_t> vec_out_dims(output_size, -1);
    out->set_dtype(x.dtype());
    out->set_dims(phi::make_ddim(vec_out_dims));
  } else if (!axes.GetData().empty()) {
    std::vector<int32_t> tmp;
    tmp.reserve(axes.GetData().size());
    std::for_each(axes.GetData().begin(),
                  axes.GetData().end(),
                  [&tmp](const int64_t& t) { tmp.push_back(t); });
    auto out_dims = funcs::GetUnsqueezeShape(tmp, x_dims);
    out->set_dims(out_dims);
    if (x_dims[0] == out_dims[0]) {
      out->share_lod(x);
    }
    out->set_dtype(x.dtype());
  }
  // set xshape dims.
  std::vector<int64_t> xshape_dims(x_dims.size() + 1);
  xshape_dims[0] = 0;
  for (int i = 0; i < x_dims.size(); ++i) {
    xshape_dims[i + 1] = x_dims[i];
  }
  xshape->set_dims(phi::make_ddim(xshape_dims));
  xshape->share_lod(x);
  xshape->set_dtype(x.dtype());
}

void UnStackInferMeta(const MetaTensor& x,
                      int axis,
                      int num,
                      std::vector<MetaTensor*> outs) {
  auto x_dim = x.dims();
  int rank = x_dim.size();
  PADDLE_ENFORCE_GE(axis,
                    -rank,
                    phi::errors::InvalidArgument(
                        "The attribute axis is out of range, it must be "
                        "inside [-rank, rank), where rank = %d",
                        rank));
  PADDLE_ENFORCE_LT(axis,
                    rank,
                    phi::errors::InvalidArgument(
                        "The attribute axis is out of range, it must be "
                        "inside [-rank, rank), where rank = %d",
                        rank));
  if (axis < 0) axis += rank;

  size_t output_count = outs.size();
  PADDLE_ENFORCE_EQ(output_count,
                    static_cast<size_t>(num),
                    phi::errors::InvalidArgument(
                        "Number of Outputs(Y) is wrong. Got %d , but it must "
                        "equal to attribute num which is %d.",
                        output_count,
                        static_cast<size_t>(num)));
  if (x_dim[axis] > 0) {
    PADDLE_ENFORCE_EQ(
        num,
        x_dim[axis],
        phi::errors::InvalidArgument(
            "The number of attribute num is not equal to the length of the "
            "%d axis of Input(X). Expect %d but got %d.",
            axis,
            x_dim[axis],
            num));
  }
  auto vec = phi::vectorize<int>(x_dim);
  vec.erase(vec.begin() + axis);
  for (size_t i = 0; i < output_count; i++) {
    outs[i]->set_dims(phi::make_ddim(vec));
    outs[i]->set_dtype(x.dtype());
  }
}

void OneHotRawInferMeta(const MetaTensor& x,
                        int32_t depth,
                        DataType dtype,
                        bool allow_out_of_range,
                        MetaTensor* out) {
  auto x_dims = x.dims();
  PADDLE_ENFORCE_GE(
      x_dims.size(),
      1,
      phi::errors::InvalidArgument("Rank of Input(X) should be at least 1."));
  auto out_dims_vec = phi::vectorize(x_dims);
  out_dims_vec.push_back(depth);
  auto out_dims = phi::make_ddim(out_dims_vec);
  out->set_dims(out_dims);
  out->share_lod(x);
  out->set_dtype(dtype);
}

void OneHotInferMeta(const MetaTensor& x,
                     const Scalar& depth_t,
                     MetaTensor* out) {
  auto x_dims = x.dims();
  PADDLE_ENFORCE_GE(
      x_dims.size(),
      1,
      phi::errors::InvalidArgument("Rank of Input(X) should be at least 1."));

  int depth = depth_t.to<int>();
  auto out_dims_vec = phi::vectorize(x_dims);
  out_dims_vec.push_back(depth);
  auto out_dims = phi::make_ddim(out_dims_vec);
  out->set_dims(out_dims);
  out->share_lod(x);

  out->set_dtype(phi::DataType::FLOAT32);
}

void WhereIndexInferMeta(const MetaTensor& condition, MetaTensor* out) {
  auto rank = condition.dims().size();
  PADDLE_ENFORCE_GE(
      rank,
      1UL,
      phi::errors::InvalidArgument(
          "Input(Condition) should have number of dimension at least 1"));
  out->set_dims(phi::make_ddim({-1, rank}));
  out->set_dtype(DataType::INT64);
}

void ChannelShuffleInferMeta(const MetaTensor& x,
                             int groups,
                             const std::string& data_format,
                             MetaTensor* out) {
  auto input_dims = x.dims();
  PADDLE_ENFORCE_EQ(input_dims.size(),
                    4,
                    phi::errors::InvalidArgument(
                        "Input should be a 4-D tensor of format [N, C, H, W] "
                        "or [N, H, W, C], but got %u.",
                        input_dims.size()));
  PADDLE_ENFORCE_GE(
      groups,
      1,
      phi::errors::InvalidArgument("groups should be larger than 0."));
  PADDLE_ENFORCE_EQ(data_format == "NCHW" || data_format == "NHWC",
                    true,
                    phi::errors::InvalidArgument(
                        "data_format must be one of "
                        "NCHW and NHWC. But recevied data_format: %s",
                        data_format));

  const bool channel_last = (data_format == "NHWC");

  if (!channel_last) {
    PADDLE_ENFORCE_EQ(input_dims[1] % groups,
                      0,
                      phi::errors::InvalidArgument(
                          "The number of groups to divide channels in [%u] "
                          "should divide the number of channel [%u]",
                          groups,
                          input_dims[1]));
  } else {
    PADDLE_ENFORCE_EQ(input_dims[3] % groups,
                      0,
                      phi::errors::InvalidArgument(
                          "The number of groups to divide channels in [%u] "
                          "should divide the number of channel [%u]",
                          groups,
                          input_dims[3]));
  }
  auto output_dims = input_dims;
  out->set_dtype(x.dtype());
  out->set_dims(output_dims);
}

}  // namespace phi

PD_REGISTER_INFER_META_FN(flatten, phi::FlattenInferMeta);
PD_REGISTER_INFER_META_FN(split, phi::SplitInferMeta);
