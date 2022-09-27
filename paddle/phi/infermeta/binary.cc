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

#include "paddle/phi/infermeta/binary.h"

#include <algorithm>
#include <vector>

#include "glog/logging.h"

#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/layout.h"
#include "paddle/phi/common/type_traits.h"
#include "paddle/phi/core/ddim.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/kernels/cpu/conv_util.h"
#include "paddle/phi/kernels/funcs/axis_utils.h"
#include "paddle/phi/kernels/funcs/common_shape.h"

namespace phi {
namespace detail {

static void BinarySameInputDimsCheck(const MetaTensor& x,
                                     const MetaTensor& y,
                                     MetaConfig config) {
  auto input_dim = x.dims();
  auto other_dim = y.dims();
  PADDLE_ENFORCE_EQ(input_dim.size(),
                    other_dim.size(),
                    phi::errors::PreconditionNotMet(
                        "Input(Input) and Input(Other) must have the same "
                        "dimension size."));
  int n = input_dim.size();
  bool is_runtime = config.is_runtime;
  for (int i = 0; i < n; i++) {
    if (is_runtime) {
      PADDLE_ENFORCE_EQ(input_dim[i],
                        other_dim[i],
                        phi::errors::PreconditionNotMet(
                            "The value at dim %d of Input(Input) is not "
                            "equal to the Input(Other): %ld != %ld.",
                            i,
                            input_dim[i],
                            other_dim[i]));
    } else {
      if (!(input_dim[i] < 0 || other_dim[i] < 0)) {
        PADDLE_ENFORCE_EQ(input_dim[i],
                          other_dim[i],
                          phi::errors::PreconditionNotMet(
                              "The value at dim %d of Input(Input) is not "
                              "equal to the Input(Other): %ld != %ld.",
                              i,
                              input_dim[i],
                              other_dim[i]));
      }
    }
  }
}

// Used in MatrixRankTolInferMeta
static DDim CheckAndGetOutputDim(const DDim& dim_x) {
  auto x_vec = phi::vectorize(dim_x);
  if (x_vec.size() == 2) {
    return phi::make_ddim({1});
  }
  x_vec.erase(x_vec.end() - 2, x_vec.end());
  return phi::make_ddim(x_vec);
}

}  // namespace detail

void AllValueCompareInferMeta(const MetaTensor& x,
                              const MetaTensor& y,
                              MetaTensor* out,
                              MetaConfig config) {
  detail::BinarySameInputDimsCheck(x, y, config);

  out->set_dims(phi::make_ddim({1}));
  out->set_dtype(DataType::BOOL);
}

void KLDivInferMeta(const MetaTensor& x,
                    const MetaTensor& label,
                    const std::string& reduction,
                    MetaTensor* out,
                    MetaConfig config) {
  auto dim_x = x.dims();
  auto dim_target = label.dims();
  PADDLE_ENFORCE_EQ(dim_x.size(),
                    dim_target.size(),
                    phi::errors::InvalidArgument(
                        "Input(X) rank and Input(Target) rank should be "
                        "same, but received X rank(%d) != Target rank(%d)",
                        dim_x.size(),
                        dim_target.size()));
  for (int i = 0; i < dim_x.size(); i++) {
    if (config.is_runtime || (dim_x[i] > 0 && dim_target[i] > 0)) {
      PADDLE_ENFORCE_EQ(
          dim_x[i],
          dim_target[i],
          phi::errors::InvalidArgument(
              "Input(X) and Input(Target) should in same shape. but received "
              "X dimension[%d](%d) != Target dimension[%d](%d)",
              i,
              dim_x[i],
              i,
              dim_target[i]));
    }
  }

  auto reduction_valid = "mean" == reduction || "sum" == reduction ||
                         "batchmean" == reduction || "none" == reduction;
  PADDLE_ENFORCE_EQ(
      reduction_valid,
      true,
      phi::errors::InvalidArgument(
          "Attr(reduction) can only be 'none'|'batchmean'|'sum'|'mean'."));

  if ("none" == reduction) {
    out->set_dims(dim_x);
  } else {
    out->set_dims({1});
  }
  out->set_dtype(x.dtype());
}

void Atan2InferMeta(const MetaTensor& x, const MetaTensor& y, MetaTensor* out) {
  out->share_meta(x);
  if (x.dtype() == DataType::INT32 || x.dtype() == DataType::INT64 ||
      y.dtype() == DataType::INT32 || y.dtype() == DataType::INT64) {
    out->set_dtype(DataType::FLOAT64);
  }
}

void BCELossInferMeta(const MetaTensor& input,
                      const MetaTensor& label,
                      MetaTensor* out,
                      MetaConfig config) {
  auto input_dims = input.dims();
  auto label_dims = label.dims();

  int rank = input_dims.size();
  PADDLE_ENFORCE_EQ(rank,
                    label_dims.size(),
                    phi::errors::InvalidArgument(
                        "Input(X) and Input(Label) shall have the same rank."
                        "But received: the rank of Input(X) is [%d], "
                        "the rank of Input(Label) is [%d].",
                        rank,
                        label_dims.size()));

  bool check = true;
  if ((!config.is_runtime) &&
      (phi::product(input_dims) <= 0 || phi::product(label_dims) <= 0)) {
    check = false;
  }

  if (check) {
    PADDLE_ENFORCE_EQ(input_dims,
                      label_dims,
                      phi::errors::InvalidArgument(
                          "Input(X) and Input(Label) shall have the same "
                          "shape. But received: the shape of Input(X) is "
                          "[%s], the shape of Input(Label) is [%s].",
                          input_dims,
                          label_dims));
  }

  out->set_dims(input_dims);
  out->set_dtype(input.dtype());
  out->share_lod(input);
}

void BincountInferMeta(const MetaTensor& x,
                       const MetaTensor& weights,
                       const Scalar& minlength,
                       MetaTensor* out) {
  auto input_dim = x.dims();

  PADDLE_ENFORCE_EQ(
      input_dim.size(),
      1,
      phi::errors::InvalidArgument("The 'shape' of Input(X) must be 1-D tensor."
                                   "But the dimension of Input(X) is [%d]",
                                   input_dim.size()));

  VLOG(1) << "####### CHECK weights";
  if (weights) {
    auto weights_dim = weights.dims();
    VLOG(1) << "##### weights_dim " << weights_dim;
    PADDLE_ENFORCE_EQ(weights_dim.size(),
                      1,
                      phi::errors::InvalidArgument(
                          "The 'shape' of Input(Weights) must be 1-D tensor."
                          "But the dimension of Input(Weights) is [%d]",
                          weights_dim.size()));

    PADDLE_ENFORCE_EQ(
        weights_dim[0],
        input_dim[0],
        phi::errors::InvalidArgument(
            "The 'shape' of Input(Weights) must be equal to the 'shape' of "
            "Input(X)."
            "But received: the 'shape' of Input(Weights) is [%s],"
            "the 'shape' of Input(X) is [%s]",
            weights_dim,
            input_dim));
  }
  out->set_dims(phi::make_ddim({-1}));
  if (weights) {
    out->set_dtype(weights.dtype());
  } else {
    out->set_dtype(x.dtype());
  }

  out->share_lod(x);
}

void BmmInferMeta(const MetaTensor& x, const MetaTensor& y, MetaTensor* out) {
  std::vector<int64_t> x_dims = phi::vectorize(x.dims());
  std::vector<int64_t> y_dims = phi::vectorize(y.dims());
  std::size_t x_ndims = x_dims.size();
  std::size_t y_ndims = y_dims.size();

  PADDLE_ENFORCE_EQ(x_ndims,
                    3,
                    phi::errors::InvalidArgument(
                        "Input(X) of BmmOp must be 3-dimensional in BmmOp, "
                        "but received X's shape: [%s].",
                        x_ndims));
  PADDLE_ENFORCE_EQ(y_ndims,
                    3,
                    phi::errors::InvalidArgument(
                        "Input(Y) of BmmOp must be 3-dimensional in BmmOp, "
                        "but received Y's shape: [%s].",
                        y_ndims));
  PADDLE_ENFORCE_EQ(
      x_dims[0],
      y_dims[0],
      phi::errors::InvalidArgument(
          "Input(X) and Input(Y) must have the same batch size in BmmOp, "
          "but received X's batch size: [%s],"
          "Y's batch size [%s]",
          x_dims[0],
          y_dims[0]));
  PADDLE_ENFORCE_EQ(
      x_dims[2],
      y_dims[1],
      phi::errors::InvalidArgument(
          "Input(X)'s width must be equal with Input(Y)'s height in BmmOp,"
          "but receive X's width: [%s],"
          "Y's height: [%s].",
          x_dims[2],
          y_dims[1]));

  std::vector<int64_t> dim_out;
  dim_out.push_back(x_dims[0]);
  dim_out.push_back(x_dims[1]);
  dim_out.push_back(y_dims[2]);
  out->set_dims(phi::make_ddim(dim_out));
  out->share_lod(x);
  out->set_dtype(x.dtype());
  out->set_layout(x.layout());
}

void CholeskySolveInferMeta(const MetaTensor& x,
                            const MetaTensor& y,
                            bool upper,
                            MetaTensor* out) {
  auto x_dims = x.dims();
  auto y_dims = y.dims();

  auto x_dims_n = x_dims.size();
  auto y_dims_n = y_dims.size();

  PADDLE_ENFORCE_GE(x_dims_n,
                    2,
                    phi::errors::InvalidArgument(
                        "the rank of input Y must greater or equal to 2"));
  PADDLE_ENFORCE_GE(y_dims_n,
                    2,
                    phi::errors::InvalidArgument(
                        "the rank of input X must greater or equal to 2"));
  PADDLE_ENFORCE_EQ(
      y_dims[y_dims_n - 1],
      y_dims[y_dims_n - 2],
      phi::errors::InvalidArgument("input Matrix Y should be square matrix,"
                                   "But Got last shape of %ld x %ld",
                                   y_dims[y_dims_n - 1],
                                   y_dims[y_dims_n - 2]));
  PADDLE_ENFORCE_EQ(
      x_dims[x_dims_n - 2],
      y_dims[y_dims_n - 2],
      phi::errors::InvalidArgument("the first dim of Matrix X must be equal to "
                                   "the fisrt dim of Matrix Y,"
                                   "But Got %ld and %ld",
                                   x_dims[x_dims_n - 2],
                                   y_dims[y_dims_n - 2]));

  std::vector<int64_t> x_dims_vec = phi::vectorize(x_dims);
  std::vector<int64_t> y_dims_vec = phi::vectorize(y_dims);

  std::vector<int64_t> x_dims_vec_cut(x_dims_vec.begin(), x_dims_vec.end() - 2);
  std::vector<int64_t> y_dims_vec_cut(y_dims_vec.begin(), y_dims_vec.end() - 2);

  std::vector<int64_t> expand_batch_portion =
      funcs::MatrixGetBroadcastBatchPortion(x_dims_vec_cut, y_dims_vec_cut);

  std::vector<int64_t> x_broadcast_dims({expand_batch_portion});
  x_broadcast_dims.insert(x_broadcast_dims.end(),
                          {x_dims_vec[x_dims_n - 2], x_dims_vec[x_dims_n - 1]});

  // dim of 'out' is the same with 'X' after broadcast
  out->set_dims(phi::make_ddim(x_broadcast_dims));
  out->set_dtype(x.dtype());
  out->set_layout(x.layout());
  out->share_lod(x);
}

void CompareInferMeta(const MetaTensor& x,
                      const MetaTensor& y,
                      int axis,
                      MetaTensor* out) {
  auto dim_x = x.dims();
  auto dim_y = y.dims();

  if (dim_x == dim_y) {
    out->share_meta(x);
  } else {
    int max_dim = std::max(dim_x.size(), dim_y.size());
    int axis = std::abs(dim_x.size() - dim_y.size());
    std::vector<int> x_dims_array(max_dim);
    std::vector<int> y_dims_array(max_dim);
    std::vector<int> out_dims_array(max_dim);
    funcs::GetBroadcastDimsArrays(dim_x,
                                  dim_y,
                                  x_dims_array.data(),
                                  y_dims_array.data(),
                                  out_dims_array.data(),
                                  max_dim,
                                  axis);

    out->set_dims(make_ddim(out_dims_array));
    out->share_lod(x);
  }

  out->set_dtype(DataType::BOOL);
}

void CompareAllInferMeta(const MetaTensor& x,
                         const MetaTensor& y,
                         MetaTensor* out) {
  auto dim_x = x.dims();
  auto dim_y = y.dims();
  PADDLE_ENFORCE_GE(
      dim_x.size(),
      dim_y.size(),
      errors::InvalidArgument(
          "The size of dim_y should not be greater than dim_x's."));
  out->share_lod(x);
  out->set_dims(make_ddim({1}));
  out->set_dtype(DataType::BOOL);
}

void ComplexInferMeta(const MetaTensor& x,
                      const MetaTensor& y,
                      MetaTensor* out) {
  if (x.dims() == y.dims()) {
    auto sizes = vectorize(x.dims());
    out->set_dims(phi::make_ddim(sizes));
    out->set_dtype(dtype::ToComplex(x.dtype()));
    // NOTE(chenfeiyu): lod & broadcasting is intrinsically contradictory
    // so tensors with lod are not supported here
  } else {
    auto x_dims = x.dims();
    auto y_dims = y.dims();
    int max_dim = std::max(x_dims.size(), y_dims.size());

    // start align axis
    int axis = std::abs(x_dims.size() - y_dims.size());
    std::vector<int> x_dims_array(max_dim);
    std::vector<int> y_dims_array(max_dim);
    std::vector<int> out_dims_array(max_dim);
    phi::funcs::GetBroadcastDimsArrays(x_dims,
                                       y_dims,
                                       x_dims_array.data(),
                                       y_dims_array.data(),
                                       out_dims_array.data(),
                                       max_dim,
                                       axis);
    out->set_dims(phi::make_ddim(out_dims_array));
    out->set_dtype(dtype::ToComplex(x.dtype()));
  }
}

void ConvInferMeta(const MetaTensor& input,
                   const MetaTensor& filter,
                   const std::vector<int>& strides,
                   const std::vector<int>& paddings_t,
                   const std::string& padding_algorithm,
                   int groups,
                   const std::vector<int>& dilations_t,
                   const std::string& data_format,
                   bool use_addto,
                   int workspace_size_MB,
                   bool exhaustive_search,
                   MetaTensor* out,
                   MetaConfig config) {
  std::vector<int> paddings = paddings_t;
  std::vector<int> dilations = dilations_t;
  auto in_dims = input.dims();
  auto filter_dims = filter.dims();
  int dilation_size = dilations.size();
  for (int i = 0; i < dilation_size; ++i) {
    PADDLE_ENFORCE_GT(
        dilations[i],
        0,
        phi::errors::InvalidArgument(
            "The dilation of Op(Conv) should be larget than 0, but received "
            "dilation is %d.",
            dilations[i]));
  }
  const bool channel_last = (config.is_run_mkldnn_kernel == false) &&
                            (data_format == "NHWC" || data_format == "NDHWC");

  PADDLE_ENFORCE_EQ(
      in_dims.size() == 4 || in_dims.size() == 5,
      true,
      phi::errors::InvalidArgument(
          "The input of Op(Conv) should be a 4-D or 5-D Tensor. But "
          "received: input's dimension is %u, input's shape is [%s].",
          in_dims.size(),
          in_dims));

  PADDLE_ENFORCE_EQ(
      in_dims.size(),
      filter_dims.size(),
      phi::errors::InvalidArgument(
          "The input's dimension and filter's dimension of "
          "Op(Conv) should be equal. But received: the input's shape is [%s], "
          "the input's dimension is %d; the filter's shape is [%s],  "
          "the filter's dimension is %d.",
          in_dims,
          in_dims.size(),
          filter_dims,
          filter_dims.size()));

  int stride_size = strides.size();
  for (int i = 0; i < stride_size; ++i) {
    PADDLE_ENFORCE_GT(
        strides[i],
        0,
        phi::errors::InvalidArgument(
            "The stride of Op(Conv) should be larget than 0, but received "
            "stride is %d.",
            strides[i]));
  }

  int in_sub_stride_size = in_dims.size() - stride_size;
  PADDLE_ENFORCE_EQ(
      in_dims.size(),
      strides.size() + 2U,
      phi::errors::InvalidArgument(
          "The difference of input's dimension and Attr(strides)'s "
          "length must be euqal to 2 for Op(Conv). "
          "But received: input's dimension is %d, input's shape is [%s]; "
          "Attr(stride)'s length is %d, Attr(stride) is [%s]; "
          "difference of input's dimention and Attr(strides)'s length = %u.",
          in_dims.size(),
          in_dims,
          strides.size(),
          phi::make_ddim(strides),
          in_sub_stride_size));

  const auto input_channels =
      channel_last ? in_dims[in_dims.size() - 1] : in_dims[1];

  PADDLE_ENFORCE_EQ(
      input_channels,
      filter_dims[1] * groups,
      phi::errors::InvalidArgument(
          "The number of input's channels should be equal to filter's channels "
          "* groups for Op(Conv). But received: the input's channels is %d, "
          "the input's shape is [%s]; the filter's channels is %d, the "
          "filter's shape is [%s]; the groups is %d, the data_format is %s. "
          "The error may come from wrong data_format setting.",
          input_channels,
          in_dims,
          filter_dims[1],
          filter_dims,
          groups,
          data_format));
  PADDLE_ENFORCE_EQ(
      filter_dims[0] % groups,
      0,
      phi::errors::InvalidArgument(
          "The number of output's channels (filter's first dimension) of "
          "Op(Conv) should be divided by groups. But received: "
          "the output channels is %d, the filter's shape is [%s], "
          "the groups is %d.",
          filter_dims[0],
          filter_dims,
          groups));

  if (config.is_runtime) {
    PADDLE_ENFORCE_GT(
        filter_dims[0],
        0,
        phi::errors::InvalidArgument(
            "the size of filter at axis 0 should be greater than 0"));
  }

  DDim in_data_dims;
  if (channel_last) {
    in_data_dims = phi::slice_ddim(in_dims, 1, in_dims.size() - 1);
  } else {
    in_data_dims = phi::slice_ddim(in_dims, 2, in_dims.size());
  }

  DDim filter_data_dims = phi::slice_ddim(filter_dims, 2, filter_dims.size());

  std::vector<int> ksize = phi::vectorize<int>(filter_data_dims);
  phi::UpdatePaddingAndDilation(
      &paddings, &dilations, padding_algorithm, in_data_dims, strides, ksize);

  std::vector<int64_t> output_shape({in_dims[0]});
  if (!channel_last) {
    output_shape.push_back(filter_dims[0]);
  }
  for (int i = 0; i < in_data_dims.size(); ++i) {
    if ((!config.is_runtime) &&
        (in_data_dims[i] <= 0 || filter_dims[i + 2] <= 0)) {
      output_shape.push_back(-1);
    } else {
      const int dkernel = dilations[i] * (filter_data_dims[i] - 1) + 1;
      int output_size =
          (in_data_dims[i] + paddings[2 * i] + paddings[2 * i + 1] - dkernel) /
              strides[i] +
          1;
      output_shape.push_back(output_size);
    }
  }
  if (channel_last) {
    output_shape.push_back(filter_dims[0]);
  }

  out->set_dims(make_ddim(output_shape));
  out->set_dtype(input.dtype());
}

void ConvInferInferMeta(const MetaTensor& input,
                        const MetaTensor& filter,
                        const std::vector<int>& strides,
                        const std::vector<int>& paddings,
                        const std::string& paddding_algorithm,
                        int groups,
                        const std::vector<int>& dilations,
                        const std::string& data_format,
                        MetaTensor* out,
                        MetaConfig config) {
  ConvInferMeta(input,
                filter,
                strides,
                paddings,
                paddding_algorithm,
                groups,
                dilations,
                data_format,
                /*use_addto=*/false,
                /*workspace_size_MB=*/512,  // useless in infermeta
                /*exhaustive_search=*/false,
                out,
                config);
}

void ConvTransposeInferMeta(const MetaTensor& x,
                            const MetaTensor& filter,
                            const std::vector<int>& strides,
                            const std::vector<int>& paddings,
                            const std::vector<int>& output_padding,
                            const std::vector<int>& output_size,
                            const std::string& padding_algorithm,
                            int groups,
                            const std::vector<int>& dilations,
                            const std::string& data_format,
                            MetaTensor* out,
                            MetaConfig config) {
  auto x_dims = x.dims();
  auto filter_dims = filter.dims();

  std::vector<int> paddings_ = paddings;
  std::vector<int> dilations_ = dilations;

  const DataLayout data_layout =
      config.is_run_mkldnn_kernel
          ? DataLayout::kNCHW
          : paddle::framework::StringToDataLayout(data_format);

  PADDLE_ENFORCE_EQ(
      x_dims.size() == 4 || x_dims.size() == 5,
      true,
      errors::InvalidArgument("Input of Op(conv_transpose) should be 4-D or "
                              "5-D Tensor. But received: %u-D Tensor, "
                              "the shape of input is [%s]",
                              x_dims.size(),
                              x_dims));
  PADDLE_ENFORCE_EQ(
      x_dims.size(),
      filter_dims.size(),
      errors::InvalidArgument(
          "The input's dimension size and filter's dimension size of "
          "Op (conv_transpose) should be equal. But received: the shape of "
          "input is [%s], the dimension size of input is [%d], the shape "
          "of filter is [%s],  the dimension size of filter is [%d]. ",
          x_dims,
          x_dims.size(),
          filter_dims,
          filter_dims.size()));

  int stride_size = strides.size();
  for (int i = 0; i < stride_size; ++i) {
    PADDLE_ENFORCE_GT(
        strides[i],
        0,
        errors::InvalidArgument(
            "The stride of Op(Conv) should be larget than 0, but received "
            "stride is %d.",
            strides[i]));
  }

  int in_sub_stride_size = x_dims.size() - stride_size;

  PADDLE_ENFORCE_EQ(
      x_dims.size() - strides.size(),
      2U,
      errors::InvalidArgument(
          "The input's dimension size minus Attr(stride)'s size must "
          "be euqal to 2 for Op(conv_transpose). But received: [%d], the "
          "input's dimension size is [%d], the shape of input "
          "is [%s], the Attr(stride)'s size is [%d].",
          in_sub_stride_size,
          x_dims.size(),
          x_dims,
          strides.size()));
  if (output_size.size())
    PADDLE_ENFORCE_EQ(
        output_size.size(),
        strides.size(),
        errors::InvalidArgument(
            "The Attr(output_size) and Attr(stride) of Op(conv_transpose) "
            "should be the same."));
  if (output_padding.size())
    PADDLE_ENFORCE_EQ(
        output_padding.size(),
        strides.size(),
        errors::InvalidArgument(
            "The Attr(output_padding) and Attr(stride) of Op(conv_transpose) "
            "should be the same."));

  const int64_t C =
      (data_layout != DataLayout::kNHWC ? x_dims[1]
                                        : x_dims[x_dims.size() - 1]);
  PADDLE_ENFORCE_EQ(
      C,
      filter_dims[0],
      errors::InvalidArgument(
          "The number of input channels should be equal to filter channels "
          "for Op(conv_transpose). But received: the input's channels is "
          "[%d], the shape of input is [%s], the filter's channels is [%d], "
          "the shape of filter is [%s]. The data_format is %s."
          "The error may come from wrong data_format setting.",
          C,
          x_dims,
          filter_dims[0],
          filter_dims,
          data_format));

  DDim x_data_dims;
  if (data_layout != DataLayout::kNHWC) {
    x_data_dims = slice_ddim(x_dims, 2, x_dims.size());
  } else {
    x_data_dims = slice_ddim(x_dims, 1, x_dims.size() - 1);
  }
  DDim filter_data_dims = slice_ddim(filter_dims, 2, filter_dims.size());
  std::vector<int> ksize = vectorize<int>(filter_data_dims);
  UpdatePaddingAndDilation(
      &paddings_, &dilations_, padding_algorithm, x_data_dims, strides, ksize);

  std::vector<int64_t> output_shape({x_dims[0]});
  if (data_layout != DataLayout::kNHWC) {
    output_shape.push_back(filter_dims[1] * groups);
  }
  const int offset = (data_layout != DataLayout::kNHWC ? 2 : 1);
  for (size_t i = 0; i < strides.size(); ++i) {
    auto filter_extent = dilations_[i] * (filter_dims[i + 2] - 1) + 1;
    auto infer_shape = (config.is_runtime || x_dims[i + offset] > 0)
                           ? (x_dims[i + offset] - 1) * strides[i] -
                                 paddings_[2 * i] - paddings_[2 * i + 1] +
                                 filter_extent
                           : -1;
    if (output_size.size()) {
      if (config.is_runtime) {
        PADDLE_ENFORCE_GE(
            output_size[i],
            infer_shape,
            errors::InvalidArgument(
                "output_size of Op(ConvTransposeOp) should not be "
                "less than the infered output size. But received output_size = "
                "[%s], whose dim %d is less than the infered output size [%s]",
                make_ddim(output_size).to_str(),
                i,
                infer_shape));
        PADDLE_ENFORCE_LT(
            output_size[i],
            infer_shape + strides[i],
            errors::InvalidArgument(
                "output_size of Op(ConvTransposeOp) should be less "
                "than infered size + stride. But received output_size = [%s], "
                "whose dim %d is not less than the infered output size (%d) + "
                "stride (%d) = %d",
                make_ddim(output_size).to_str(),
                i,
                infer_shape,
                strides[i],
                infer_shape + strides[i]));
      }
      output_shape.push_back(output_size[i]);
    } else if (output_padding.size()) {
      if (config.is_runtime) {
        PADDLE_ENFORCE_GE(
            output_padding[i],
            0,
            errors::InvalidArgument(
                "output_padding of Op(ConvTransposeOp) should not be "
                "less than the 0. But received output_padding = "
                "[%s], whose dim %d is less than 0",
                make_ddim(output_padding).to_str(),
                i));
        PADDLE_ENFORCE_LT(
            output_padding[i],
            std::max(strides[i], dilations_[i]),
            errors::InvalidArgument(
                "output_padding of Op(ConvTransposeOp) should be less "
                "than either stride or dilation. But received output_size = "
                "[%s], "
                "whose dim %d is not less than either stride (%d)  or "
                "dilation (%d)",
                make_ddim(output_size).to_str(),
                i,
                strides[i],
                dilations_[i]));
      }
      output_shape.push_back((infer_shape + output_padding[i]));
    } else {
      output_shape.push_back(infer_shape);
    }
  }
  if (data_layout == DataLayout::kNHWC) {
    output_shape.push_back(filter_dims[1] * groups);
  }

  out->set_dims(make_ddim(output_shape));
  out->set_dtype(x.dtype());
}

void Conv2dTransposeInferMeta(const MetaTensor& x,
                              const MetaTensor& filter,
                              const std::vector<int>& strides,
                              const std::vector<int>& paddings,
                              const std::vector<int>& output_padding,
                              const IntArray& output_size,
                              const std::string& padding_algorithm,
                              int groups,
                              const std::vector<int>& dilations,
                              const std::string& data_format,
                              MetaTensor* out,
                              MetaConfig config) {
  std::vector<int32_t> vec_output_size(output_size.GetData().begin(),
                                       output_size.GetData().end());
  ConvTransposeInferMeta(x,
                         filter,
                         strides,
                         paddings,
                         output_padding,
                         vec_output_size,
                         padding_algorithm,
                         groups,
                         dilations,
                         data_format,
                         out,
                         config);
}

void CrossInferMeta(const MetaTensor& x,
                    const MetaTensor& y,
                    int axis,
                    MetaTensor* out) {
  auto x_dim = x.dims();
  auto y_dim = y.dims();
  auto dim = axis;

  bool dims_match = phi::funcs::CheckDims(x_dim, y_dim);
  PADDLE_ENFORCE_EQ(
      dims_match,
      true,
      phi::errors::InvalidArgument("The 'shape' of Input(X) should be equal to "
                                   "the 'shape' of Input(Y). But received "
                                   "Input(X).dimensions = [%s], "
                                   "Input(Y).dimensions = [%s]",
                                   x_dim,
                                   y_dim));

  if (dim != DDim::kMaxRank) {
    PADDLE_ENFORCE_EQ(
        dim < x_dim.size() && dim >= (0 - x_dim.size()),
        true,
        phi::errors::OutOfRange(
            "Attr(dim) is out of range, It's expected "
            "to be in range of [-%d, %d]. But received Attr(dim) = %d.",
            x_dim.size(),
            x_dim.size() - 1,
            dim));
    if (dim < 0) {
      dim += x_dim.size();
    }
    PADDLE_ENFORCE_EQ(x_dim[dim] == 3 && y_dim[dim] == 3,
                      true,
                      phi::errors::InvalidArgument(
                          "Input(X/Y).dims()[dim] should be equal to 3."
                          "But received Input(X/Y).dims()[dim] = %d.",
                          x_dim[dim]));
  }
  out->set_dims(x_dim);
  out->set_dtype(x.dtype());
  out->set_layout(x.layout());
  out->share_lod(x);
}

void CrossEntropyWithSoftmaxInferMeta(const MetaTensor& logits,
                                      const MetaTensor& label,
                                      bool soft_label,
                                      bool use_softmax,
                                      bool numeric_stable_mode,
                                      int ignore_index,
                                      int axis,
                                      MetaTensor* softmax,
                                      MetaTensor* loss,
                                      MetaConfig config) {
  auto logits_dims = logits.dims();
  auto labels_dims = label.dims();
  auto logits_rank = logits_dims.size();
  PADDLE_ENFORCE_GE(axis,
                    -logits_rank,
                    phi::errors::InvalidArgument(
                        "Attr(axis) value should be in range [-R, R-1], "
                        "R is the rank of Input(Logits)."));
  PADDLE_ENFORCE_LT(axis,
                    logits_rank,
                    phi::errors::InvalidArgument(
                        "Attr(axis) value should be in range [-R, R-1], "
                        "R is the rank of Input(Logits)."));

  axis = phi::funcs::CanonicalAxis(axis, logits_rank);
  for (int i = 0; i < logits_rank; i++) {
    if (i != axis) {
      if (config.is_runtime || (logits_dims[i] > 0 && labels_dims[i] > 0)) {
        PADDLE_ENFORCE_EQ(logits_dims[i],
                          labels_dims[i],
                          phi::errors::InvalidArgument(
                              "Input(Logits) and Input(Label) should in "
                              "same shape in dimensions except axis."));
      }
    }
  }

  if (axis != logits_rank - 1) {
    PADDLE_ENFORCE_EQ(
        numeric_stable_mode,
        true,
        phi::errors::InvalidArgument("Attr(axis) can only be -1 "
                                     "when not in numeric_stable_mode."));
  }

  if (soft_label) {
    if (config.is_runtime || (logits_dims[axis] > 0 && labels_dims[axis] > 0)) {
      PADDLE_ENFORCE_EQ(logits_dims[axis],
                        labels_dims[axis],
                        phi::errors::InvalidArgument(
                            "If Attr(soft_label) == true,  "
                            "the axis dimension of "
                            "Input(X) and Input(Label) should be equal."));
    }
  } else {
    if (config.is_runtime || labels_dims[axis] > 0) {
      PADDLE_ENFORCE_EQ(
          labels_dims[axis],
          1UL,
          phi::errors::InvalidArgument("If Attr(soft_label) == false, "
                                       "the axis dimension of "
                                       "Input(Label) should be 1."));
    }
  }

  softmax->set_dims(logits_dims);
  softmax->set_dtype(logits.dtype());

  logits_dims[axis] = 1;
  loss->set_dims(logits_dims);
  loss->set_dtype(logits.dtype());

  softmax->share_lod(logits);
  loss->share_lod(logits);
}

void DistInferMeta(const MetaTensor& x,
                   const MetaTensor& y,
                   float p,
                   MetaTensor* out) {
  auto x_dims = x.dims();
  auto y_dims = y.dims();

  PADDLE_ENFORCE_NE(phi::product(x_dims),
                    0,
                    phi::errors::InvalidArgument(
                        "The Input(X) has not been initialized properly. The "
                        "shape of Input(X) = [%s].",
                        x_dims));
  PADDLE_ENFORCE_NE(phi::product(y_dims),
                    0,
                    phi::errors::InvalidArgument(
                        "The Input(Y) has not been initialized properly. The "
                        "shape of Input(Y) = [%s].",
                        y_dims));
  out->set_dims({1});
  out->set_dtype(x.dtype());
}

void DistributeFpnProposalsInferMeta(
    const MetaTensor& fpn_rois,
    const MetaTensor& rois_num,
    int min_level,
    int max_level,
    int refer_level,
    int refer_scale,
    bool pixel_offset,
    std::vector<MetaTensor*> multi_fpn_rois,
    std::vector<MetaTensor*> multi_level_rois_num,
    MetaTensor* restore_index,
    MetaConfig config) {
  PADDLE_ENFORCE_GE(
      multi_fpn_rois.size(),
      1UL,
      errors::InvalidArgument("Outputs(MultiFpnRois) of "
                              "DistributeFpnProposalsOp should not be empty"));
  PADDLE_ENFORCE_GE(
      max_level,
      min_level,
      errors::InvalidArgument(
          "max_level must not lower than "
          "min_level. But received max_level = %d, min_level = %d",
          max_level,
          min_level));
  // Set the output shape
  for (size_t i = 0; i < multi_fpn_rois.size(); ++i) {
    DDim out_dim = {-1, 4};
    if (multi_fpn_rois[i] == nullptr) {
      continue;
    }
    multi_fpn_rois[i]->set_dims(out_dim);
    multi_fpn_rois[i]->set_dtype(fpn_rois.dtype());
  }
  restore_index->set_dims({-1, 1});
  restore_index->set_dtype(DataType::INT32);
  for (size_t i = 0; i < multi_level_rois_num.size(); ++i) {
    if (multi_level_rois_num[i] == nullptr) {
      continue;
    }
    multi_level_rois_num[i]->set_dims({-1});
    multi_level_rois_num[i]->set_dtype(DataType::INT32);
  }

  if (!config.is_runtime) {
    for (size_t i = 0; i < multi_fpn_rois.size(); ++i) {
      multi_fpn_rois[i]->share_lod(fpn_rois);
    }
  }
}

void DropoutInferMeta(const MetaTensor& x,
                      const MetaTensor& seed_tensor,
                      const Scalar& p,
                      bool is_test,
                      const std::string& mode,
                      int seed,
                      bool fix_seed,
                      MetaTensor* out,
                      MetaTensor* mask) {
  auto x_dims = x.dims();
  out->set_dims(x_dims);
  out->share_lod(x);
  out->set_dtype(x.dtype());

  if (mask != nullptr) {
    mask->set_dims(x_dims);
    mask->set_dtype(DataType::UINT8);
  }
}

void DropoutNdInferMeta(const MetaTensor& x,
                        const MetaTensor& seed_tensor,
                        const Scalar& p,
                        bool is_test,
                        const std::string& mode,
                        int seed,
                        bool fix_seed,
                        const std::vector<int>& axis,
                        MetaTensor* out,
                        MetaTensor* mask) {
  auto x_dims = x.dims();

  PADDLE_ENFORCE_LE(
      axis.size(),
      x_dims.size(),
      phi::errors::InvalidArgument(
          "The length of axis is expected to be less than or equal to the "
          "dimension size of x. But recieved the length of axis is %d, the "
          "dimension size of x is %d, x's shape is {%s}.",
          axis.size(),
          x_dims.size(),
          x_dims));
  for (size_t i = 0; i < axis.size(); ++i) {
    PADDLE_ENFORCE_EQ(
        axis[i] >= 0 && axis[i] <= x_dims.size() - 1,
        true,
        phi::errors::InvalidArgument(
            "The %d-th value of axis is expected to be greater ot "
            "equal to 0 and less than the dimensions of x. But "
            "recieved axis is {%s}, the dimension size of x is %d.",
            i,
            phi::make_ddim(axis),
            x_dims.size()));
  }

  out->set_dims(x_dims);
  out->share_lod(x);
  out->set_dtype(x.dtype());

  if (mask != nullptr) {
    std::vector<int64_t> mask_dims(x.dims().size(), 1);

    std::for_each(
        axis.begin(), axis.end(), [&mask_dims, &x_dims](const int64_t& t) {
          mask_dims[t] = x_dims[t];
        });

    mask->set_dims(make_ddim(mask_dims));
    mask->set_dtype(DataType::UINT8);
  }
}

void DotInferMeta(const MetaTensor& x, const MetaTensor& y, MetaTensor* out) {
  auto x_dims = x.dims();
  auto x_rank = static_cast<size_t>(x_dims.size());
  PADDLE_ENFORCE_EQ(true,
                    1 == x_rank || 2 == x_rank,
                    phi::errors::PreconditionNotMet(
                        "ShapeError: The dimensions of input tensor X (%s) "
                        "should be 1 or 2",
                        x_dims.to_str()));

  auto y_dims = y.dims();
  PADDLE_ENFORCE_EQ(
      true,
      x_rank == static_cast<size_t>(y_dims.size()),
      phi::errors::PreconditionNotMet(
          "ShapeError: The shape of input tensor Y: %s should match with "
          "input tenosr X: %s",
          y_dims.to_str(),
          x_dims.to_str()));
  bool shape_match = true;
  for (size_t i = 0; i < x_rank; ++i) {
    if (x_dims[i] != y_dims[i]) {
      shape_match = false;
      break;
    }
  }

  PADDLE_ENFORCE_EQ(true,
                    shape_match,
                    phi::errors::PreconditionNotMet(
                        "ShapeError: The shape of input tensor X: %s should "
                        "be exactly the same "
                        "with input tensor Y: %s",
                        x_dims.to_str(),
                        y_dims.to_str()));

  x_dims[x_dims.size() - 1] = 1;
  out->set_dims(x_dims);
  out->set_dtype(x.dtype());
  out->set_layout(x.layout());
}

void ElementwiseInferMeta(const MetaTensor& x,
                          const MetaTensor& y,
                          MetaTensor* out) {
  return ElementwiseRawInferMeta(x, y, -1, std::move(out));
}

void ElementwiseRawInferMeta(const MetaTensor& x,
                             const MetaTensor& y,
                             int axis,
                             MetaTensor* out) {
  if (x.dims() != y.dims()) {
    auto x_dims = x.dims();
    auto y_dims = y.dims();
    int max_dim = std::max(x_dims.size(), y_dims.size());
    if (x_dims.size() == y_dims.size()) {
      PADDLE_ENFORCE_EQ((axis == -1) || (axis == 0),
                        true,
                        phi::errors::InvalidArgument(
                            "axis should be -1 or 0 while the dimension of "
                            "tensor X (%s) is equal to the dimension of "
                            "tensor Y (%s), but received axis: %s",
                            x_dims.size(),
                            y_dims.size(),
                            axis));
    }
    PADDLE_ENFORCE_EQ((axis >= (-1 * max_dim)) && (axis < max_dim),
                      true,
                      phi::errors::InvalidArgument(
                          "The axis range must be [%s, %s), but axis is %s. "
                          "Please set the axis again.",
                          -1 * max_dim,
                          max_dim,
                          axis));
    axis = (axis < 0 ? (std::abs(x_dims.size() - y_dims.size()) + axis + 1)
                     : axis);
    std::vector<int> x_dims_array(max_dim);
    std::vector<int> y_dims_array(max_dim);
    std::vector<int> out_dims_array(max_dim);
    funcs::GetBroadcastDimsArrays(x_dims,
                                  y_dims,
                                  x_dims_array.data(),
                                  y_dims_array.data(),
                                  out_dims_array.data(),
                                  max_dim,
                                  axis);
    auto out_dims = phi::make_ddim(out_dims_array);
    out->set_dims(out_dims);
  } else {
    out->set_dims(x.dims());
  }

  out->set_dtype(x.dtype());
  out->set_layout(x.layout());
  out->share_lod(x);
}

void EmbeddingInferMeta(const MetaTensor& x,
                        const MetaTensor& weight,
                        int64_t padding_idx,
                        MetaTensor* out) {
  const auto& table_dims = weight.dims();
  const auto& ids_dims = x.dims();
  int ids_rank = ids_dims.size();
  VLOG(5) << "ids rank is " << ids_rank << std::endl;
  PADDLE_ENFORCE_EQ(
      table_dims.size(),
      2,
      phi::errors::InvalidArgument(
          "ShapeError: The dimensions of the 'lookup table' must be 2. "
          "But received lookup table's dimensions = %d, "
          "lookup table's shape = [%s].",
          table_dims.size(),
          table_dims));

  auto output_dims = phi::vectorize(ids_dims);
  output_dims.push_back(table_dims[1]);
  out->set_dims(phi::make_ddim(output_dims));
  out->set_dtype(weight.dtype());
  out->share_lod(x);
}

void ExpandAsInferMeta(const MetaTensor& x,
                       const MetaTensor& y,
                       const std::vector<int>& target_shape,
                       MetaTensor* out) {
#define MAX_RANK_SUPPORTED 6
  auto x_dims = x.dims();
  PADDLE_ENFORCE_GE(
      target_shape.size(),
      static_cast<size_t>(x_dims.size()),
      phi::errors::InvalidArgument(
          "The rank of target_shape must be greater than or equal "
          "to the rank of Input(X). But received Input(X): input "
          "rank %u; received target_shape: rank %u.",
          x_dims.size(),
          target_shape.size()));
  PADDLE_ENFORCE_LE(target_shape.size(),
                    MAX_RANK_SUPPORTED,
                    phi::errors::InvalidArgument(
                        "The rank of target_shape must be less than or equal "
                        "to %d. But received: rank %u.",
                        MAX_RANK_SUPPORTED,
                        target_shape.size()));
  out->set_dims(phi::make_ddim(target_shape));
  out->set_dtype(x.dtype());
#undef MAX_RANK_SUPPORTED
}

void FillDiagonalTensorInferMeta(const MetaTensor& x,
                                 const MetaTensor& y,
                                 int64_t offset,
                                 int dim1,
                                 int dim2,
                                 MetaTensor* out) {
  PADDLE_ENFORCE_NOT_NULL(out,
                          phi::errors::InvalidArgument(
                              "Output Tensor (out) should not be nullptr."));
  out->set_dims(x.dims());
  out->set_dtype(x.dtype());
}

void GatherInferMeta(const MetaTensor& x,
                     const MetaTensor& index,
                     const Scalar& axis,
                     MetaTensor* out) {
  auto index_dims = index.dims();

  if (index_dims.size() == 2) {
    PADDLE_ENFORCE_EQ(
        index_dims[1],
        1,
        phi::errors::InvalidArgument(
            "The last dim of index should be 1 when it is 2D, but we get %d",
            index_dims[1]));
  } else {
    PADDLE_ENFORCE_EQ(
        index_dims.size(),
        1,
        phi::errors::InvalidArgument(
            "The index should be 1D, when it is not 2D, but we get %d",
            index_dims.size()));
  }

  auto input_dim = x.dims();
  auto axis_v = axis.to<int>();
  if (axis.FromTensor() || axis_v == 0) {
    // if axis.FromTensor(), we can not obtain correct shape of output
    int batch_size = index_dims[0];
    phi::DDim output_dims(input_dim);
    output_dims[0] = batch_size;
    out->set_dims(output_dims);
    out->set_dtype(x.dtype());
    out->share_lod(x);
  } else {
    int index_size = index_dims[0];
    std::vector<int> out_dim_vec;
    for (int i = 0; i < axis_v; i++) {
      out_dim_vec.push_back(input_dim[i]);
    }
    out_dim_vec.push_back(index_size);
    for (int i = axis_v + 1; i < input_dim.size(); i++) {
      out_dim_vec.push_back(input_dim[i]);
    }
    auto output_dims = phi::make_ddim(out_dim_vec);
    out->set_dims(output_dims);
    out->set_dtype(x.dtype());
    out->share_lod(x);
  }
}

void GatherNdInferMeta(const MetaTensor& x,
                       const MetaTensor& index,
                       MetaTensor* out) {
  auto x_dims = x.dims();
  auto x_dims_size = x_dims.size();
  auto index_dims = index.dims();
  auto index_dims_size = index_dims.size();

  PADDLE_ENFORCE_LE(
      index_dims[index_dims_size - 1],
      x_dims_size,
      phi::errors::InvalidArgument(
          "Input(Index).shape[-1] should be no greater than Input(X).rank"));
  PADDLE_ENFORCE_GE(index_dims_size,
                    1UL,
                    phi::errors::InvalidArgument(
                        "The rank of Input(Index) should be greater than 1"));

  std::vector<int64_t> result_dims;
  // The result dims is
  //   Index.shape[:-1] + X.shape[Index.shape[-1]:]
  for (int i = 0; i < index_dims_size - 1; ++i) {
    result_dims.emplace_back(index_dims[i]);
  }
  for (int i = index_dims[index_dims_size - 1]; i < x_dims_size; ++i) {
    result_dims.emplace_back(x_dims[i]);
  }

  out->set_dims(phi::make_ddim(result_dims));
  out->share_lod(x);
  out->set_dtype(x.dtype());
}

void GatherTreeMeta(const MetaTensor& ids,
                    const MetaTensor& parents,
                    MetaTensor* out) {
  auto ids_dims = ids.dims();
  auto parents_dims = parents.dims();
  PADDLE_ENFORCE_EQ(ids_dims == parents_dims,
                    true,
                    phi::errors::InvalidArgument(
                        "The shape of Input(Parents) must be same with the "
                        "shape of Input(Ids)."));
  out->set_dims(ids_dims);
}

void GridSampleBaseInferMeta(const MetaTensor& x,
                             const MetaTensor& grid,
                             MetaTensor* out,
                             MetaConfig config) {
  auto x_dims = x.dims();
  auto grid_dims = grid.dims();
  PADDLE_ENFORCE_GE(x_dims.size(),
                    4,
                    phi::errors::InvalidArgument(
                        "Input(X) of GridSampleOp should be 4-D Tensor, but "
                        "received X dimension size(%d)",
                        x_dims.size()));
  PADDLE_ENFORCE_LE(x_dims.size(),
                    5,
                    phi::errors::InvalidArgument(
                        "Input(X) of GridSampleOp should be 4-D Tensor, but "
                        "received X dimension size(%d)",
                        x_dims.size()));
  PADDLE_ENFORCE_GE(grid_dims.size(),
                    4,
                    phi::errors::InvalidArgument(
                        "Input(Grid) of GridSampleOp should be 4-D Tensor, "
                        "but received X dimension size(%d)",
                        grid_dims.size()));
  PADDLE_ENFORCE_LE(grid_dims.size(),
                    5,
                    phi::errors::InvalidArgument(
                        "Input(Grid) of GridSampleOp should be 4-D Tensor, "
                        "but received X dimension size(%d)",
                        grid_dims.size()));
  if (grid_dims.size() == 4 && (config.is_runtime || grid_dims[3] > 0)) {
    PADDLE_ENFORCE_EQ(
        grid_dims[3],
        2,
        phi::errors::InvalidArgument(
            "Input(Grid) dimension[3] should be 2, but received %d",
            grid_dims[3]));
  }
  if (grid_dims.size() == 5 && (config.is_runtime || grid_dims[4] > 0)) {
    PADDLE_ENFORCE_EQ(
        grid_dims[4],
        3,
        phi::errors::InvalidArgument(
            "Input(Grid) dimension[4] should be 3, but received %d",
            grid_dims[4]));
  }
  if (config.is_runtime) {
    PADDLE_ENFORCE_EQ(
        grid_dims[0],
        x_dims[0],
        phi::errors::InvalidArgument(
            "Input(X) and Input(Grid) dimension[0] should be equal, but "
            "received X dimension[0](%d) != Grid dimension[0](%d)",
            x_dims[0],
            grid_dims[0]));
  }
  if (grid_dims.size() == 4) {
    out->set_dims({x_dims[0], x_dims[1], grid_dims[1], grid_dims[2]});
  } else {
    out->set_dims(
        {x_dims[0], x_dims[1], grid_dims[1], grid_dims[2], grid_dims[3]});
  }
  out->set_dtype(x.dtype());
  out->share_lod(x);
}

void HuberLossInferMeta(const MetaTensor& input,
                        const MetaTensor& label,
                        float delta,
                        MetaTensor* out,
                        MetaTensor* residual,
                        MetaConfig config) {
  auto input_dims = input.dims();
  auto label_dims = label.dims();

  PADDLE_ENFORCE_EQ(input_dims.size(),
                    label_dims.size(),
                    phi::errors::InvalidArgument(
                        "Input(input) rank and Input(label) rank should be "
                        "same, but received input rank(%d) != label rank(%d)",
                        input_dims.size(),
                        label_dims.size()));

  bool contain_unknown_dim = phi::contain_unknown_dim(input_dims) ||
                             phi::contain_unknown_dim(label_dims);
  if (config.is_runtime || !contain_unknown_dim) {
    PADDLE_ENFORCE_EQ(
        input_dims,
        label_dims,
        phi::errors::InvalidArgument(
            "The Input(input) and Input(label) should have the same "
            "shape, but received input shape [%s] != label shape [%s]",
            input_dims,
            label_dims));
  }

  auto out_dims = label_dims;
  residual->set_dims(out_dims);
  out->set_dims(out_dims);
  out->share_lod(input);
}

void IndexSampleInferMeta(const MetaTensor& x,
                          const MetaTensor& y,
                          MetaTensor* out,
                          MetaConfig config) {
  auto input_dims = x.dims();
  PADDLE_ENFORCE_EQ(input_dims.size(),
                    2,
                    errors::InvalidArgument(
                        "Inputs(X) shape of IndexSample op should be 2-D, but "
                        "got X's shape = [%s], please check X shape.",
                        input_dims));

  auto index_dims = y.dims();
  PADDLE_ENFORCE_EQ(
      index_dims.size(),
      2,
      errors::InvalidArgument(
          "Inputs(Index) shape of IndexSample op should be 2-D, but "
          "got Index's shape [%s] , please check index shape.",
          input_dims));
  if (config.is_runtime) {
    PADDLE_ENFORCE_EQ(input_dims[0],
                      index_dims[0],
                      errors::InvalidArgument(
                          "Inputs(X)'s value of dimension 0 must same with "
                          "Inputs(Index)'s value of dimension 0, but "
                          "got %d of Inputs(X), and got %d of Inputs(Index), "
                          "please check Inputs shape.",
                          input_dims[0],
                          index_dims[0]));
  }
  out->set_dtype(x.dtype());
  out->set_dims(index_dims);
  out->share_lod(y);
}

void IndexSelectInferMeta(const MetaTensor& x,
                          const MetaTensor& index,
                          int dim,
                          MetaTensor* output) {
  auto input_dim = x.dims();
  auto index_dim = index.dims();

  PADDLE_ENFORCE_EQ(
      dim < input_dim.size() && dim >= (0 - input_dim.size()),
      true,
      phi::errors::OutOfRange(
          "Attr(dim) is out of range, It's expected "
          "to be in range of [-%d, %d]. But received Attr(dim) = %d.",
          input_dim.size(),
          input_dim.size() - 1,
          dim));

  PADDLE_ENFORCE_EQ(
      index_dim.size() == 1 || (index_dim.size() == 2 && index_dim[1] == 1),
      true,
      phi::errors::InvalidArgument(
          "The 'shape' of Input(Index) must be 1-D tensor. "
          "But received: the 'shape' of Input(Index) is [%s], "
          "the dimension of Input(Index) is [%d].",
          index_dim,
          index_dim.size()));

  PADDLE_ENFORCE_EQ(
      index_dim[0] != 0,
      true,
      phi::errors::InvalidArgument("The length of Input(Index) can't be 0."));

  auto output_dim = phi::vectorize(input_dim);
  if (dim < 0) {
    dim += input_dim.size();
  }
  output_dim[dim] = index_dim[0];
  output->set_dims(phi::make_ddim(output_dim));
  output->set_dtype(x.dtype());
  output->set_layout(x.layout());
  output->share_lod(x);
}

void IndexAddInferMeta(const MetaTensor& x,
                       const MetaTensor& index,
                       const MetaTensor& add_value,
                       int axis,
                       MetaTensor* output) {
  auto input_dim = x.dims();
  auto index_dim = index.dims();
  auto add_value_dim = add_value.dims();

  PADDLE_ENFORCE_EQ(
      axis < input_dim.size() && axis >= (0 - input_dim.size()),
      true,
      phi::errors::OutOfRange(
          "Attr(dim) is out of range, It's expected "
          "to be in range of [-%d, %d]. But received Attr(axis) = %d.",
          input_dim.size(),
          input_dim.size() - 1,
          axis));

  int real_axis = axis >= 0 ? axis : axis + input_dim.size();

  PADDLE_ENFORCE_EQ(index_dim.size() == 1,
                    true,
                    phi::errors::InvalidArgument(
                        "The 'shape' of Input(Index) must be 1-D tensor. "
                        "But received: the 'shape' of Input(Index) is [%s], "
                        "the dimension of Input(Index) is [%d].",
                        index_dim,
                        index_dim.size()));

  PADDLE_ENFORCE_EQ(
      index_dim[0] != 0,
      true,
      phi::errors::InvalidArgument("The length of Input(Index) can't be 0."));

  // Note, add_value does not support broadcast now.
  PADDLE_ENFORCE_EQ(input_dim.size() == add_value_dim.size(),
                    true,
                    phi::errors::InvalidArgument(
                        "The add_value must be the same dimension as x."));
  for (int i = 0; i < input_dim.size(); i++) {
    if (i != real_axis) {
      PADDLE_ENFORCE_EQ(input_dim[i] == add_value_dim[i],
                        true,
                        phi::errors::InvalidArgument(
                            "The add_value parameter does not supported "
                            "broadcast, so input_dim[i] must be equal to "
                            "add_value_dim[i] when i != axis."));
    }
  }

  const auto& index_type = index.dtype();
  bool index_type_match =
      index_type == phi::DataType::INT64 || index_type == phi::DataType::INT32;
  PADDLE_ENFORCE_EQ(index_type_match,
                    true,
                    phi::errors::InvalidArgument(
                        "Input(Index) holds the wrong type, it holds %s, but "
                        "desires to be %s or %s",
                        index_type,
                        phi::DataType::INT32,
                        phi::DataType::INT64));

  output->set_dims(x.dims());
  output->set_dtype(x.dtype());
  output->set_layout(x.layout());
  output->share_lod(x);
}

void KronInferMeta(const MetaTensor& x, const MetaTensor& y, MetaTensor* out) {
  auto dim_x = x.dims();
  auto dim_y = y.dims();
  auto rank_x = dim_x.size();
  auto rank_y = dim_y.size();
  auto rank = (rank_x > rank_y) ? rank_x : rank_y;

  std::vector<int64_t> dim_out;
  dim_out.reserve(rank);
  for (int i = 0; i < rank; i++) {
    int64_t dim_xi = (i < rank - rank_x) ? 1 : dim_x.at(i - (rank - rank_x));
    int64_t dim_yi = (i < rank - rank_y) ? 1 : dim_y.at(i - (rank - rank_y));
    dim_out.push_back(dim_xi == -1 || dim_yi == -1 ? -1 : dim_xi * dim_yi);
  }
  out->set_dims(phi::make_ddim(dim_out));
  out->set_dtype(x.dtype());
}

void LogLossInferMeta(const MetaTensor& input,
                      const MetaTensor& label,
                      float epsilon,
                      MetaTensor* out,
                      MetaConfig config) {
  auto pred_dims = input.dims();
  auto label_dims = label.dims();

  if (config.is_runtime ||
      (phi::product(pred_dims) > 0 && phi::product(label_dims) > 0)) {
    PADDLE_ENFORCE_EQ(
        pred_dims,
        label_dims,
        phi::errors::InvalidArgument(
            "The dimensions of Input(Predicted) must be equal to the"
            "dimensions of Input(Labels), but received dimensions of "
            "Input(Predicted)"
            "is [%s], received dimensions of Input(Labels) is [%s].",
            pred_dims,
            label_dims));
  }
  PADDLE_ENFORCE_EQ(pred_dims.size(),
                    2,
                    phi::errors::InvalidArgument(
                        "The dimensions of Input(Predicted) must be 2,"
                        "But received dimensions of Input(Predicted)"
                        "is [%d]",
                        pred_dims.size()));
  if (config.is_runtime) {
    PADDLE_ENFORCE_EQ(pred_dims[1],
                      1,
                      phi::errors::InvalidArgument(
                          "Each row of Input(Predicted) contains a real value, "
                          "so the 2nd dimension of Input(X) must be 1,"
                          "But got [%d]",
                          pred_dims[1]));
  }
  out->set_dims({pred_dims[0], 1});
  out->set_dtype(input.dtype());
  out->share_lod(input);
}

void LUUnpackInferMeta(const MetaTensor& x,
                       const MetaTensor& pivots,
                       bool unpack_ludata,
                       bool unpack_pivots,
                       MetaTensor* pmat,
                       MetaTensor* l,
                       MetaTensor* u) {
  PADDLE_ENFORCE_NOT_NULL(
      pmat,
      phi::errors::InvalidArgument("Output(Pmat) should not be nullptr."));
  PADDLE_ENFORCE_NOT_NULL(
      l, phi::errors::InvalidArgument("Output(L) should not be nullptr."));
  PADDLE_ENFORCE_NOT_NULL(
      u, phi::errors::InvalidArgument("Output(U) should not be nullptr."));

  auto x_dims = x.dims();
  int x_rank = x_dims.size();
  PADDLE_ENFORCE_GE(
      x_rank,
      2,
      phi::errors::InvalidArgument("The rank of input must greater than 2."));

  int m = x_dims[x_rank - 1];
  int n = x_dims[x_rank - 2];
  int min_mn = std::min(m, n);
  if (unpack_ludata) {
    auto ldims = x_dims;
    auto udims = x_dims;
    if (m >= n) {
      udims[x_rank - 2] = min_mn;
    } else {
      ldims[x_rank - 1] = min_mn;
    }
    u->set_dims(udims);
    u->set_dtype(x.dtype());
    l->set_dims(ldims);
    l->set_dtype(x.dtype());
  }
  if (unpack_pivots) {
    auto pdims = x_dims;
    pdims[x_rank - 1] = m;
    pmat->set_dims(pdims);
    pmat->set_dtype(x.dtype());
  }
}

void MarginCrossEntropyInferMeta(const MetaTensor& logits,
                                 const MetaTensor& label,
                                 bool return_softmax,
                                 int ring_id,
                                 int rank,
                                 int nranks,
                                 float margin1,
                                 float margin2,
                                 float margin3,
                                 float scale,
                                 MetaTensor* softmax,
                                 MetaTensor* loss,
                                 MetaConfig config) {
  PADDLE_ENFORCE_NOT_NULL(
      logits,
      phi::errors::InvalidArgument("Input of logits should not be null."));
  PADDLE_ENFORCE_NOT_NULL(
      label,
      phi::errors::InvalidArgument("Input of label should not be null."));
  auto logits_dims = logits.dims();
  auto labels_dims = label.dims();

  auto logits_rank = logits_dims.size();
  auto axis = logits_rank - 1;
  for (int i = 0; i < logits_rank; i++) {
    if (i != axis) {
      if (config.is_runtime || (logits_dims[i] > 0 && labels_dims[i] > 0)) {
        PADDLE_ENFORCE_EQ(logits_dims[i],
                          labels_dims[i],
                          phi::errors::InvalidArgument(
                              "Input(Logits) and Input(Label) should in "
                              "same shape in dimensions except axis."));
      }
    }
  }

  if (labels_dims.size() > 1) {
    PADDLE_ENFORCE_EQ(
        labels_dims[logits_rank - 1],
        1UL,
        phi::errors::InvalidArgument(
            "the last dimension of Input(Label) should be 1."
            "But received: the last dimension of Input(Label) is [%d],"
            "the last dimension is [%d]",
            labels_dims[logits_rank - 1],
            logits_rank - 1));
  }

  softmax->set_dims(logits_dims);
  softmax->set_dtype(logits.dtype());

  logits_dims[axis] = 1;
  loss->set_dims(logits_dims);
  loss->set_dtype(logits.dtype());

  softmax->share_lod(logits);
  loss->share_lod(logits);
}

void MaskedSelectInferMeta(const MetaTensor& x,
                           const MetaTensor& mask,
                           MetaTensor* out) {
  out->set_dims({-1});  // can not infer
  out->set_dtype(x.dtype());
}

void MatmulInferMeta(const MetaTensor& x,
                     const MetaTensor& y,
                     bool trans_x,
                     bool trans_y,
                     MetaTensor* out) {
  std::vector<int64_t> dims_x = phi::vectorize(x.dims());
  std::vector<int64_t> dims_y = phi::vectorize(y.dims());
  auto ndims_x = dims_x.size();
  auto ndims_y = dims_y.size();
  PADDLE_ENFORCE_GT(ndims_x,
                    0UL,
                    phi::errors::InvalidArgument(
                        "The Input(x) dims size must be greater than 0,"
                        " but reviced dims size is 0. "));
  PADDLE_ENFORCE_GT(ndims_y,
                    0UL,
                    phi::errors::InvalidArgument(
                        "The Input(y) dims size must be greater than 0,"
                        " but reviced dims size is 0. "));

  bool x_broadcasted = false, y_broadcasted = false;
  if (ndims_x == 1) {
    dims_x.insert(dims_x.begin(), 1);
    ndims_x = 2;
    x_broadcasted = true;
  }

  if (ndims_y == 1) {
    dims_y.push_back(1);
    ndims_y = 2;
    y_broadcasted = true;
  }

  size_t M, N;
  if (trans_x) {
    M = dims_x[ndims_x - 1];
  } else {
    M = dims_x[ndims_x - 2];
  }
  if (trans_y) {
    N = dims_y[ndims_y - 2];
  } else {
    N = dims_y[ndims_y - 1];
  }

  std::vector<int64_t> new_dims;
  if (ndims_x > ndims_y) {
    new_dims.assign(dims_x.begin(), dims_x.end() - 2);
  } else if (ndims_x < ndims_y) {
    new_dims.assign(dims_y.begin(), dims_y.end() - 2);
  } else {
    new_dims.reserve(ndims_x);
    for (size_t i = 0; i < ndims_x - 2; ++i) {
      new_dims.push_back(std::max(dims_x[i], dims_y[i]));
    }
  }
  if (!x_broadcasted) {
    new_dims.push_back(M);
  }
  if (!y_broadcasted) {
    new_dims.push_back(N);
  }
  if (x_broadcasted && y_broadcasted) {
    new_dims.push_back(1);
  }

  auto ddim_out = phi::make_ddim(new_dims);

  out->set_dims(ddim_out);
  out->set_dtype(x.dtype());
  out->set_layout(x.layout());
}

void MatmulWithFlattenInferMeta(const MetaTensor& x,
                                const MetaTensor& y,
                                int x_num_col_dims,
                                int y_num_col_dims,
                                MetaTensor* out) {
  auto x_dims = x.dims();
  auto y_dims = y.dims();

  VLOG(3) << "mul operator x.shape=" << x_dims << " y.shape=" << y_dims
          << " x_num_col_dims=" << x_num_col_dims
          << " y_num_col_dims=" << y_num_col_dims;

  PADDLE_ENFORCE_NE(phi::product(y_dims),
                    0,
                    phi::errors::PreconditionNotMet(
                        "The Input variable Y has not "
                        "been initialized. You may need to confirm "
                        "if you put exe.run(startup_program) "
                        "after optimizer.minimize function."));
  PADDLE_ENFORCE_GT(
      x_dims.size(),
      x_num_col_dims,
      phi::errors::InvalidArgument(
          "The input tensor X's dimensions of MulOp "
          "should be larger than x_num_col_dims. But received X's "
          "dimensions = %d, X's shape = [%s], x_num_col_dims = %d.",
          x_dims.size(),
          x_dims,
          x_num_col_dims));
  PADDLE_ENFORCE_GT(
      y_dims.size(),
      y_num_col_dims,
      phi::errors::InvalidArgument(
          "The input tensor Y's dimensions of MulOp "
          "should be larger than y_num_col_dims. But received Y's "
          "dimensions = %d, Y's shape = [%s], y_num_col_dims = %d.",
          y_dims.size(),
          y_dims,
          y_num_col_dims));

  auto x_mat_dims = phi::flatten_to_2d(x_dims, x_num_col_dims);
  auto y_mat_dims = phi::flatten_to_2d(y_dims, y_num_col_dims);

  PADDLE_ENFORCE_EQ(
      x_mat_dims[1],
      y_mat_dims[0],
      phi::errors::InvalidArgument(
          "After flatten the input tensor X and Y to 2-D dimensions matrix "
          "X1 and Y1, the matrix X1's width must be equal with matrix Y1's "
          "height. But received X's shape = [%s], X1's shape = [%s], X1's "
          "width = %s; Y's shape = [%s], Y1's shape = [%s], Y1's height = "
          "%s.",
          x_dims,
          x_mat_dims,
          x_mat_dims[1],
          y_dims,
          y_mat_dims,
          y_mat_dims[0]));
  std::vector<int64_t> output_dims;
  output_dims.reserve(
      static_cast<size_t>(x_num_col_dims + y_dims.size() - y_num_col_dims));

  for (int i = 0; i < x_num_col_dims; ++i) {
    output_dims.push_back(x_dims[i]);
  }

  for (int i = y_num_col_dims; i < y_dims.size(); ++i) {
    output_dims.push_back(y_dims[i]);
  }

  out->set_dims(phi::make_ddim(output_dims));
  out->set_dtype(x.dtype());
  out->share_lod(x);
}

void MatrixNMSInferMeta(const MetaTensor& bboxes,
                        const MetaTensor& scores,
                        float score_threshold,
                        int nms_top_k,
                        int keep_top_k,
                        float post_threshold,
                        bool use_gaussian,
                        float gaussian_sigma,
                        int background_label,
                        bool normalized,
                        MetaTensor* out,
                        MetaTensor* index,
                        MetaTensor* roisnum,
                        MetaConfig config) {
  auto box_dims = bboxes.dims();
  auto score_dims = scores.dims();
  auto score_size = score_dims.size();

  if (config.is_runtime) {
    PADDLE_ENFORCE_EQ(
        score_size == 3,
        true,
        errors::InvalidArgument("The rank of Input(Scores) must be 3. "
                                "But received rank = %d.",
                                score_size));
    PADDLE_ENFORCE_EQ(
        box_dims.size(),
        3,
        errors::InvalidArgument("The rank of Input(BBoxes) must be 3."
                                "But received rank = %d.",
                                box_dims.size()));
    PADDLE_ENFORCE_EQ(box_dims[2] == 4,
                      true,
                      errors::InvalidArgument(
                          "The last dimension of Input (BBoxes) must be 4, "
                          "represents the layout of coordinate "
                          "[xmin, ymin, xmax, ymax]."));
    PADDLE_ENFORCE_EQ(
        box_dims[1],
        score_dims[2],
        errors::InvalidArgument(
            "The 2nd dimension of Input(BBoxes) must be equal to "
            "last dimension of Input(Scores), which represents the "
            "predicted bboxes."
            "But received box_dims[1](%s) != socre_dims[2](%s)",
            box_dims[1],
            score_dims[2]));
  }
  out->set_dims({box_dims[1], box_dims[2] + 2});
  out->set_dtype(bboxes.dtype());
  index->set_dims({box_dims[1], 1});
  index->set_dtype(phi::DataType::INT32);
  if (roisnum != nullptr) {
    roisnum->set_dims({-1});
    roisnum->set_dtype(phi::DataType::INT32);
  }
}

void MatrixRankTolInferMeta(const MetaTensor& x,
                            const MetaTensor& atol_tensor,
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
  auto dim_tol = atol_tensor.dims();
  if (dim_x_batch == dim_tol) {
    out->set_dims(dim_x_batch);
  } else {
    int max_dim = std::max(dim_x_batch.size(), dim_tol.size());
    int axis = std::abs(dim_x_batch.size() - dim_tol.size());
    std::vector<int> x_batch_dims_array(max_dim);
    std::vector<int> tol_dims_array(max_dim);
    std::vector<int> out_dims_array(max_dim);
    phi::funcs::GetBroadcastDimsArrays(dim_x_batch,
                                       dim_tol,
                                       x_batch_dims_array.data(),
                                       tol_dims_array.data(),
                                       out_dims_array.data(),
                                       max_dim,
                                       axis);
    out->set_dims(phi::make_ddim(out_dims_array));
  }
  out->share_lod(x);
}

void MvInferMeta(const MetaTensor& x, const MetaTensor& vec, MetaTensor* out) {
  auto dim_x = x.dims();
  auto dim_vec = vec.dims();
  PADDLE_ENFORCE_EQ(
      dim_x.size(),
      2,
      phi::errors::InvalidArgument("The rank of input X should be 2, but is %d",
                                   dim_x.size()));
  PADDLE_ENFORCE_EQ(
      dim_vec.size(),
      1,
      phi::errors::InvalidArgument(
          "The rank of input Vec should be 1, but is %d", dim_vec.size()));
  PADDLE_ENFORCE_EQ(dim_x[1],
                    dim_vec[0],
                    phi::errors::InvalidArgument(
                        "X's second dimension is expected to be equal to "
                        "Vec's first dimension"
                        "but received X'shape = [%s], Vec's shape = [%s]",
                        dim_x,
                        dim_vec));

  auto dim_out = phi::make_ddim({dim_x[0]});

  out->set_dims(dim_out);
  out->set_dtype(x.dtype());
  out->set_layout(x.layout());
  out->share_lod(x);
}

void PReluInferMeta(const MetaTensor& x,
                    const MetaTensor& alpha,
                    const std::string& data_format,
                    const std::string& mode,
                    MetaTensor* out,
                    MetaConfig config) {
  auto x_dim = x.dims();
  if (mode == "all") {
    PADDLE_ENFORCE_EQ(phi::product(alpha.dims()),
                      1,
                      phi::errors::InvalidArgument(
                          "For mode 'all', size of weight Alpha must be one. "
                          "But recevied alpha's size: %d.",
                          product(alpha.dims())));
  } else if (mode == "channel") {
    auto x_rank = x_dim.size();
    PADDLE_ENFORCE_GE(x_rank,
                      2,
                      phi::errors::InvalidArgument(
                          "For mode 'channel', rank of input X must be "
                          "equal or larger than 2. But recevied X's "
                          "rank: %d",
                          x_rank));
    PADDLE_ENFORCE_EQ(data_format == "NCHW" || data_format == "NHWC",
                      true,
                      phi::errors::InvalidArgument(
                          "For mode 'channel', data_format must be one of "
                          "NCHW and NHWC. But recevied data_format: %s",
                          data_format));
    if (data_format == "NCHW" || config.is_run_mkldnn_kernel) {
      PADDLE_ENFORCE_EQ(product(alpha.dims()) == x_dim[1],
                        true,
                        phi::errors::InvalidArgument(
                            "For mode 'channel', size of weight Alpha must be "
                            "equal to the number of channels of input(x). But "
                            "recevied alpha's size: %d, x_dim[1]: %d",
                            product(alpha.dims()),
                            x_dim[1]));
    } else {
      PADDLE_ENFORCE_EQ(product(alpha.dims()) == x_dim[x_rank - 1],
                        true,
                        phi::errors::InvalidArgument(
                            "For mode 'channel', size of weight Alpha must be "
                            "equal to the number of channels of input(x). But "
                            "recevied alpha's size: %d, x_dim[%d]: %d",
                            product(alpha.dims()),
                            x_rank - 1,
                            x_dim[x_rank - 1]));
    }
  } else if (mode == "element") {
    auto alpha_dim = alpha.dims();
    auto alpha_rank = alpha_dim.size();
    auto x_rank = x_dim.size();
    PADDLE_ENFORCE_GE(x_rank,
                      1,
                      phi::errors::InvalidArgument(
                          "For mode 'element', rank of input X must be "
                          "equal or larger than 2. But recevied X's "
                          "rank: %d",
                          x_rank));
    PADDLE_ENFORCE_EQ(
        alpha_rank,
        x_rank,
        phi::errors::InvalidArgument(
            "For mode 'element', rank of weight Alpha must be ",
            "equal to the rank of input(x). But recevied alpha's rank: %d, "
            "x's rank: %d.",
            alpha_rank,
            x_rank));
    size_t x_product = 1;
    size_t alpha_product = 1;
    for (int64_t i = x_rank - 1; i > 0; i--) {
      x_product *= x_dim[i];
      alpha_product *= alpha_dim[i];
    }
    PADDLE_ENFORCE_EQ(
        alpha_product,
        x_product,
        phi::errors::InvalidArgument(
            "For mode 'element', the size of weight Alpha must be "
            "equal to the size of input(x). But recevied alpha's size: %d, "
            "x's size: %d.",
            alpha_product,
            x_product));
  } else {
    PADDLE_THROW(phi::errors::InvalidArgument(
        "Attr(mode) of prelu must be one of 'all', 'channel', or 'element'. "
        "But recevied "
        "mode: '%s'.",
        mode));
  }
  out->set_dims(x_dim);
  out->set_dtype(x.dtype());
  out->set_layout(x.layout());
  out->share_lod(x);
}

inline void ExpandAspectRatios(const std::vector<float>& input_aspect_ratior,
                               bool flip,
                               std::vector<float>* output_aspect_ratior) {
  constexpr float epsilon = 1e-6;
  output_aspect_ratior->clear();
  output_aspect_ratior->push_back(1.0f);
  for (size_t i = 0; i < input_aspect_ratior.size(); ++i) {
    float ar = input_aspect_ratior[i];
    bool already_exist = false;
    for (size_t j = 0; j < output_aspect_ratior->size(); ++j) {
      if (fabs(ar - output_aspect_ratior->at(j)) < epsilon) {
        already_exist = true;
        break;
      }
    }
    if (!already_exist) {
      output_aspect_ratior->push_back(ar);
      if (flip) {
        output_aspect_ratior->push_back(1.0f / ar);
      }
    }
  }
}

void PriorBoxInferMeta(const MetaTensor& input,
                       const MetaTensor& image,
                       const std::vector<float>& min_sizes,
                       const std::vector<float>& aspect_ratios,
                       const std::vector<float>& variances,
                       const std::vector<float>& max_sizes,
                       bool flip,
                       bool clip,
                       float step_w,
                       float step_h,
                       float offset,
                       bool min_max_aspect_ratios_order,
                       MetaTensor* out,
                       MetaTensor* var) {
  auto image_dims = image.dims();
  auto input_dims = input.dims();

  PADDLE_ENFORCE_EQ(
      image_dims.size(),
      4,
      phi::errors::InvalidArgument(
          "The Input(Image) of Op(PriorBoxOp) should be a 4-D Tensor "
          "and data format is NCHW. But received Image's dimensions = %d, "
          "shape = [%s].",
          image_dims.size(),
          image_dims));
  PADDLE_ENFORCE_EQ(
      input_dims.size(),
      4,
      phi::errors::InvalidArgument(
          "The Input(Input) of Op(PriorBoxOp) should be a 4-D Tensor "
          "and data format is NCHW. But received Input's dimensions = %d, "
          "shape = [%s].",
          input_dims.size(),
          input_dims));

  std::vector<float> aspect_ratios_vec;
  ExpandAspectRatios(aspect_ratios, flip, &aspect_ratios_vec);

  size_t num_priors = aspect_ratios_vec.size() * min_sizes.size();
  if (max_sizes.size() > 0) {
    PADDLE_ENFORCE_EQ(
        max_sizes.size(),
        min_sizes.size(),
        phi::errors::InvalidArgument(
            "The length of min_size and "
            "max_size must be equal. But received: min_size's length is %d, "
            "max_size's length is %d.",
            min_sizes.size(),
            max_sizes.size()));
    num_priors += max_sizes.size();
    for (size_t i = 0; i < max_sizes.size(); ++i) {
      PADDLE_ENFORCE_GT(
          max_sizes[i],
          min_sizes[i],
          phi::errors::InvalidArgument(
              "max_size[%d] must be greater "
              "than min_size[%d]. But received: max_size[%d] is %f, "
              "min_size[%d] is %f.",
              i,
              i,
              i,
              max_sizes[i],
              i,
              min_sizes[i]));
    }
  }

  std::vector<int64_t> dim_vec(4);
  dim_vec[0] = input_dims[2];
  dim_vec[1] = input_dims[3];
  dim_vec[2] = num_priors;
  dim_vec[3] = 4;

  out->set_dtype(input.dtype());
  var->set_dtype(input.dtype());
  out->set_dims(phi::make_ddim(dim_vec));
  var->set_dims(phi::make_ddim(dim_vec));
}

void RepeatInterleaveWithTensorIndexInferMeta(const MetaTensor& x,
                                              const MetaTensor& repeats,
                                              int dim,
                                              MetaTensor* out) {
  const auto& input_dim = x.dims();
  auto output_dim = phi::vectorize(input_dim);
  PADDLE_ENFORCE_EQ(
      dim < input_dim.size() && dim >= (0 - input_dim.size()),
      true,
      phi::errors::OutOfRange(
          "Attr(dim) is out of range, It's expected "
          "to be in range of [-%d, %d]. But received Attr(dim) = %d.",
          input_dim.size(),
          input_dim.size() - 1,
          dim));

  auto repeats_dim = repeats.dims();

  PADDLE_ENFORCE_EQ(
      repeats_dim.size() == 1 ||
          (repeats_dim.size() == 2 && repeats_dim[1] == 1),
      true,
      phi::errors::InvalidArgument(
          "The 'shape' of Input(RepeatsTensor) must be 1-D tensor. "
          "But received: the 'shape' of Input(Index) is [%s], "
          "the dimension of Input(Index) is [%d].",
          repeats_dim,
          repeats_dim.size()));

  PADDLE_ENFORCE_EQ(repeats_dim[0] != 0,
                    true,
                    phi::errors::InvalidArgument(
                        "The length of Input(RepeatsTensor) can't be 0."));
  PADDLE_ENFORCE_NE(out,
                    nullptr,
                    phi::errors::InvalidArgument(
                        "repeat_interleave's output tensor can't be nullptr"));
  if (dim < 0) {
    dim += input_dim.size();
  }
  output_dim[dim] = -1;

  out->set_dims(phi::make_ddim(output_dim));
  out->share_lod(x);
  out->set_dtype(x.dtype());
}
void SearchsortedInferMeta(const MetaTensor& sorted_sequence,
                           const MetaTensor& value,
                           bool out_int32,
                           bool right,
                           MetaTensor* out) {
  auto sequences_dims = sorted_sequence.dims();
  auto values_dims = value.dims();

  bool flag = true;
  if (sequences_dims.size() != values_dims.size()) {
    flag = false;
  }
  const auto& sequences_dims_size = sequences_dims.size();
  for (int64_t dim = 0; dim < sequences_dims_size - 1; ++dim) {
    if (sequences_dims[dim] != values_dims[dim]) {
      flag = false;
      break;
    }
  }
  if (sequences_dims.size() != 1) {
    PADDLE_ENFORCE_EQ(
        flag,
        true,
        phi::errors::Unavailable(
            "The dimensions of sorted_sequence tensor ( %s ) and values "
            "tensor ( %s ) can not match. Because the input sorted_sequence "
            "tensor must be 1 dimension or the first N-1 dimensions of "
            "sorted_sequence tensor and input values tensor must match. "
            "Please input appropriate sorted_sequence and values again! ",
            sequences_dims,
            values_dims));
  }

  if (out_int32) {
    PADDLE_ENFORCE_LT(
        sequences_dims[sequences_dims.size() - 1],
        std::numeric_limits<int>::max(),
        phi::errors::Unavailable(
            "The size of sorted_sequence %d exceed the maximum limit d%. "
            "Because the size of sorted_sequence should be less than the "
            "output maximum value for int32 bit. Please set appropriate "
            "sorted_sequence to meet this requirement! ",
            sequences_dims[sequences_dims.size() - 1],
            std::numeric_limits<int>::max()));
  }

  out->set_dims(values_dims);
  if (out_int32) {
    out->set_dtype(DataType::INT32);
  } else {
    out->set_dtype(DataType::INT64);
  }
}

void SoftmaxMaskFuseInferMeta(const MetaTensor& x,
                              const MetaTensor& mask,
                              MetaTensor* out) {
  auto x_dims = x.dims();
  auto mask_dims = mask.dims();

  PADDLE_ENFORCE_EQ(
      x_dims.size(),
      4,
      phi::errors::InvalidArgument("Input x must be in 4D dimension but "
                                   "received the dimension of X is %d",
                                   x_dims.size()));
  PADDLE_ENFORCE_EQ(
      mask_dims.size(),
      4,
      phi::errors::InvalidArgument("Input mask must be in 4D dimension but "
                                   "received the dimension of mask is %d",
                                   mask_dims.size()));

  out->share_meta(x);
}

void SegmentPoolInferMeta(const MetaTensor& x,
                          const MetaTensor& segment_ids,
                          const std::string& pooltype,
                          MetaTensor* out,
                          MetaTensor* summed_ids,
                          MetaConfig config) {
  auto dims = x.dims();
  dims[0] = -1;
  out->set_dims(dims);
  out->set_dtype(x.dtype());
  out->set_layout(x.layout());

  if (pooltype == "MEAN") {
    summed_ids->set_dims({-1, 1});
    summed_ids->set_dtype(x.dtype());
    summed_ids->set_layout(x.layout());
  }
}

void SigmoidCrossEntropyWithLogitsInferMeta(const MetaTensor& x,
                                            const MetaTensor& label,
                                            bool normalize,
                                            int ignore_index,
                                            MetaTensor* out,
                                            MetaConfig config) {
  auto x_dims = x.dims();
  auto labels_dims = label.dims();
  int rank = x_dims.size();
  PADDLE_ENFORCE_EQ(rank,
                    labels_dims.size(),
                    phi::errors::InvalidArgument(
                        "Input(X) and Input(Label) shall have the same rank."
                        "But received: the rank of Input(X) is [%d], "
                        "the rank of Input(Label) is [%d].",
                        rank,
                        labels_dims.size()));

  bool check = true;
  if ((!config.is_runtime) &&
      (phi::product(x_dims) <= 0 || phi::product(labels_dims) <= 0)) {
    check = false;
  }

  if (check) {
    PADDLE_ENFORCE_EQ(
        phi::slice_ddim(x_dims, 0, rank),
        phi::slice_ddim(labels_dims, 0, rank),
        phi::errors::InvalidArgument(
            "Input(X) and Input(Label) shall have the same shape "
            "except the last dimension. But received: the shape of "
            "Input(X) is [%s], the shape of Input(Label) is [%s].",
            x_dims,
            labels_dims));
  }

  out->set_dims(x_dims);
  out->set_dtype(x.dtype());
  out->share_lod(x);
}

void TakeAlongAxisInferMeta(const MetaTensor& x,
                            const MetaTensor& index,
                            int axis,
                            MetaTensor* out) {
  auto input_dim = x.dims();
  auto index_dim = index.dims();

  PADDLE_ENFORCE_GT(input_dim.size(),
                    0,
                    phi::errors::InvalidArgument(
                        "Dimension of the input(Input) of TakeAlongAxisOp "
                        "should be greater than 0.",
                        input_dim));

  PADDLE_ENFORCE_GT(index_dim.size(),
                    0,
                    phi::errors::InvalidArgument(
                        "Dimension of the input(Index) of TakeAlongAxisOp "
                        "should be greater than 0.",
                        index_dim));

  out->set_dims(index_dim);
  out->set_dtype(x.dtype());
}

void TriangularSolveInferMeta(const MetaTensor& x,
                              const MetaTensor& y,
                              bool upper,
                              bool transpose,
                              bool unitriangular,
                              MetaTensor* out) {
  auto x_dims = x.dims();
  auto y_dims = y.dims();

  auto x_dims_n = x_dims.size();
  auto y_dims_n = y_dims.size();

  PADDLE_ENFORCE_GE(x_dims_n,
                    2,
                    phi::errors::InvalidArgument(
                        "The input tensor X's dimensions of TriangularSolveOp "
                        "should be >= 2. But received X's "
                        "dimensions = %d, X's shape = [%s]",
                        x_dims.size(),
                        x_dims));

  PADDLE_ENFORCE_GE(y_dims_n,
                    2,
                    phi::errors::InvalidArgument(
                        "The input tensor Y's dimensions of TriangularSolveOp "
                        "should be >=2. But received Y's "
                        "dimensions = %d, Y's shape = [%s]",
                        y_dims.size(),
                        y_dims));

  PADDLE_ENFORCE_EQ(x_dims[x_dims_n - 2],
                    x_dims[x_dims_n - 1],
                    phi::errors::InvalidArgument(
                        "The inner-most 2 dimensions of Input(X) all should "
                        "be square matrices "
                        "But received X's shape[-2] = %d and shape[-1] = %d.",
                        x_dims[x_dims_n - 2],
                        x_dims[x_dims_n - 1]));

  std::vector<int64_t> x_dims_vec = phi::vectorize(x_dims);
  std::vector<int64_t> y_dims_vec = phi::vectorize(y_dims);

  std::vector<int64_t> x_dims_vec_cut(x_dims_vec.begin(), x_dims_vec.end() - 2);
  std::vector<int64_t> y_dims_vec_cut(y_dims_vec.begin(), y_dims_vec.end() - 2);

  std::vector<int64_t> expand_batch_portion =
      funcs::MatrixGetBroadcastBatchPortion(x_dims_vec_cut, y_dims_vec_cut);

  std::vector<int64_t> y_broadcast_dims({expand_batch_portion});
  y_broadcast_dims.insert(y_broadcast_dims.end(),
                          {y_dims_vec[y_dims_n - 2], y_dims_vec[y_dims_n - 1]});

  // dim of 'out' is the same with 'Y' after broadcast
  out->set_dims(phi::make_ddim(y_broadcast_dims));
  out->set_dtype(y.dtype());
  out->set_layout(y.layout());
  out->share_lod(y);
}

void LstsqInferMeta(const MetaTensor& x,
                    const MetaTensor& y,
                    const Scalar& rcond,
                    const std::string& driver,
                    MetaTensor* solution,
                    MetaTensor* residuals,
                    MetaTensor* rank,
                    MetaTensor* singular_values) {
  auto x_dims = x.dims();
  auto y_dims = y.dims();
  int x_rank = x_dims.size();
  int y_rank = y_dims.size();

  int m = x_dims[x_rank - 2];
  int n = x_dims[x_rank - 1];
  int nrhs = y_dims[x_rank - 1];

  PADDLE_ENFORCE_GE(
      x_rank,
      2,
      phi::errors::InvalidArgument("Expects input tensor x to be not less than "
                                   "2 dimentions, but got dimention %d",
                                   x_rank));
  PADDLE_ENFORCE_GE(
      y_rank,
      2,
      phi::errors::InvalidArgument("Expects input tensor y to be not less than "
                                   "2 dimentions, but got dimention %d",
                                   y_rank));

  PADDLE_ENFORCE_EQ(
      x_rank,
      y_rank,
      phi::errors::InvalidArgument(
          "Expects input tensor x and y to have the same dimension "
          "but got x's dimention [%d] and y's dimention [%d]",
          x_rank,
          y_rank));

  std::vector<int> batch_dims_vec{};
  for (int i = 0; i < x_rank - 2; ++i) {
    PADDLE_ENFORCE_EQ(x_dims[i],
                      y_dims[i],
                      phi::errors::InvalidArgument(
                          "Expects input tensor x and y to have the same batch "
                          "dimension, but got x's batch dimention [%d] and "
                          "y's batch dimention [%d] in %d-th dim",
                          x_dims[i],
                          y_dims[i],
                          i));
    batch_dims_vec.emplace_back(x_dims[i]);
  }

  PADDLE_ENFORCE_EQ(
      m,
      y_dims[y_rank - 2],
      phi::errors::InvalidArgument(
          "Expects input tensor x and y to have the same row dimension "
          "of the inner-most 2-dims matrix, "
          "but got x's row dimention [%d] and y's row dimention [%d]",
          m,
          y_dims[y_rank - 2]));

  rank->set_dims(phi::make_ddim(batch_dims_vec));

  if (m > n) {
    batch_dims_vec.emplace_back(nrhs);
    residuals->set_dims(phi::make_ddim(batch_dims_vec));
    batch_dims_vec.pop_back();
  } else {
    residuals->set_dims(phi::make_ddim({0}));
  }
  residuals->set_dtype(y.dtype());

  batch_dims_vec.emplace_back(std::min(m, n));
  singular_values->set_dims(phi::make_ddim(batch_dims_vec));
  singular_values->set_dtype(y.dtype());

  batch_dims_vec[x_rank - 2] = n;
  batch_dims_vec.emplace_back(nrhs);
  solution->set_dims(phi::make_ddim(batch_dims_vec));
  solution->set_dtype(y.dtype());
}

void YoloBoxInferMeta(const MetaTensor& x,
                      const MetaTensor& img_size,
                      const std::vector<int>& anchors,
                      int class_num,
                      float conf_thresh,
                      int downsample_ratio,
                      bool clip_bbox,
                      float scale_x_y,
                      bool iou_aware,
                      float iou_aware_factor,
                      MetaTensor* boxes,
                      MetaTensor* scores,
                      MetaConfig config) {
  auto dim_x = x.dims();
  auto dim_imgsize = img_size.dims();
  int anchor_num = anchors.size() / 2;

  PADDLE_ENFORCE_EQ(
      dim_x.size(),
      4,
      phi::errors::InvalidArgument("Input(X) should be a 4-D tensor."
                                   "But received X dimension(%s)",
                                   dim_x.size()));
  if (iou_aware) {
    PADDLE_ENFORCE_EQ(
        dim_x[1],
        anchor_num * (6 + class_num),
        phi::errors::InvalidArgument(
            "Input(X) dim[1] should be equal to (anchor_mask_number * (6 "
            "+ class_num)) while iou_aware is true."
            "But received dim[1](%s) != (anchor_mask_number * "
            "(6+class_num)(%s).",
            dim_x[1],
            anchor_num * (6 + class_num)));
    PADDLE_ENFORCE_GE(
        iou_aware_factor,
        0,
        phi::errors::InvalidArgument(
            "Attr(iou_aware_factor) should greater than or equal to 0."
            "But received iou_aware_factor (%s)",
            iou_aware_factor));
    PADDLE_ENFORCE_LE(
        iou_aware_factor,
        1,
        phi::errors::InvalidArgument(
            "Attr(iou_aware_factor) should less than or equal to 1."
            "But received iou_aware_factor (%s)",
            iou_aware_factor));
  } else {
    PADDLE_ENFORCE_EQ(
        dim_x[1],
        anchor_num * (5 + class_num),
        phi::errors::InvalidArgument(
            "Input(X) dim[1] should be equal to (anchor_mask_number * (5 "
            "+ class_num))."
            "But received dim[1](%s) != (anchor_mask_number * "
            "(5+class_num)(%s).",
            dim_x[1],
            anchor_num * (5 + class_num)));
  }
  PADDLE_ENFORCE_EQ(
      dim_imgsize.size(),
      2,
      phi::errors::InvalidArgument("Input(ImgSize) should be a 2-D tensor."
                                   "But received Imgsize size(%s)",
                                   dim_imgsize.size()));
  if ((dim_imgsize[0] > 0 && dim_x[0] > 0) || config.is_runtime) {
    PADDLE_ENFORCE_EQ(
        dim_imgsize[0],
        dim_x[0],
        phi::errors::InvalidArgument(
            "Input(ImgSize) dim[0] and Input(X) dim[0] should be same."));
  }
  PADDLE_ENFORCE_EQ(
      dim_imgsize[1],
      2,
      phi::errors::InvalidArgument("Input(ImgSize) dim[1] should be 2."
                                   "But received imgsize dim[1](%s).",
                                   dim_imgsize[1]));
  PADDLE_ENFORCE_GT(anchors.size(),
                    0,
                    phi::errors::InvalidArgument(
                        "Attr(anchors) length should be greater than 0."
                        "But received anchors length(%s).",
                        anchors.size()));
  PADDLE_ENFORCE_EQ(anchors.size() % 2,
                    0,
                    phi::errors::InvalidArgument(
                        "Attr(anchors) length should be even integer."
                        "But received anchors length (%s)",
                        anchors.size()));
  PADDLE_ENFORCE_GT(class_num,
                    0,
                    phi::errors::InvalidArgument(
                        "Attr(class_num) should be an integer greater than 0."
                        "But received class_num (%s)",
                        class_num));

  int box_num;
  if ((dim_x[2] > 0 && dim_x[3] > 0) || config.is_runtime) {
    box_num = dim_x[2] * dim_x[3] * anchor_num;
  } else {
    box_num = -1;
  }
  std::vector<int64_t> dim_boxes({dim_x[0], box_num, 4});
  boxes->set_dims(phi::make_ddim(dim_boxes));
  boxes->set_dtype(x.dtype());

  std::vector<int64_t> dim_scores({dim_x[0], box_num, class_num});
  scores->set_dims(phi::make_ddim(dim_scores));
}

void ValueCompareInferMeta(const MetaTensor& x,
                           const MetaTensor& y,
                           MetaTensor* out,
                           MetaConfig config) {
  detail::BinarySameInputDimsCheck(x, y, config);

  out->set_dims(x.dims());
  out->set_dtype(DataType::BOOL);
}

void SolveInferMeta(const MetaTensor& x, const MetaTensor& y, MetaTensor* out) {
  auto x_dims = x.dims();
  auto y_dims = y.dims();

  std::vector<int64_t> x_dims_vec = phi::vectorize(x.dims());
  std::vector<int64_t> y_dims_vec = phi::vectorize(y.dims());

  auto x_dims_n = x_dims_vec.size();
  auto y_dims_n = y_dims_vec.size();

  PADDLE_ENFORCE_GT(
      x_dims_n,
      1,
      phi::errors::InvalidArgument("The input tensor X's dimensions of SolveOp "
                                   "should be larger than 1. But received X's "
                                   "dimensions = %d, X's shape = [%s]",
                                   x_dims_n,
                                   x_dims));

  PADDLE_ENFORCE_GE(y_dims_n,
                    1,
                    phi::errors::InvalidArgument(
                        "The input tensor Y's dimensions of SolveOp "
                        "should be larger than or equal 1. But received Y's "
                        "dimensions = %d, Y's shape = [%s]",
                        y_dims_n,
                        y_dims));

  PADDLE_ENFORCE_EQ(x_dims[x_dims_n - 2],
                    x_dims[x_dims_n - 1],
                    phi::errors::InvalidArgument(
                        "The inner-most 2 dimensions of Input(X) all should "
                        "be square matrices "
                        "But received X's shape[-2] = %d and shape[-1] = %d.",
                        x_dims[x_dims_n - 2],
                        x_dims[x_dims_n - 1]));

  bool x_broadcasted = false, y_broadcasted = false;
  bool trans_x = false, trans_y = false;
  if (x_dims_n == 1) {
    x_dims_vec.insert(x_dims_vec.begin(), 1);
    x_dims_n = 2;
    x_broadcasted = true;
  }

  if (y_dims_n == 1) {
    y_dims_vec.push_back(1);
    y_dims_n = 2;
    y_broadcasted = true;
  }

  size_t M, N;
  if (trans_x) {
    M = x_dims_vec[x_dims_n - 1];
  } else {
    M = x_dims_vec[x_dims_n - 2];
  }
  if (trans_y) {
    N = y_dims_vec[y_dims_n - 2];
  } else {
    N = y_dims_vec[y_dims_n - 1];
  }

  std::vector<int64_t> new_dims;
  if (x_dims_n >= y_dims_n) {
    new_dims.assign(x_dims_vec.begin(), x_dims_vec.end() - 2);
  } else {
    new_dims.assign(y_dims_vec.begin(), y_dims_vec.end() - 2);
  }
  if (!x_broadcasted) {
    new_dims.push_back(M);
  }
  if (!y_broadcasted) {
    new_dims.push_back(N);
  }
  if (x_broadcasted && y_broadcasted) {
    new_dims.push_back(1);
  }

  auto out_dims = phi::make_ddim(new_dims);

  out->set_dims(out_dims);
  out->set_dtype(x.dtype());
  out->set_layout(x.layout());
  out->share_lod(x);
}

void UnpoolInferMeta(const MetaTensor& x,
                     const MetaTensor& indices,
                     const std::vector<int>& ksize,
                     const std::vector<int>& strides,
                     const std::vector<int>& paddings,
                     const IntArray& output_size,
                     const std::string& data_format,
                     MetaTensor* out,
                     MetaConfig config) {
  auto in_x_dims = x.dims();
  auto in_y_dims = indices.dims();

  PADDLE_ENFORCE_EQ(in_x_dims.size() == 4,
                    true,
                    phi::errors::InvalidArgument(
                        "Unpool Intput(X) must be of 4-dimensional, but "
                        "received Input(X)'s dimensions is %d.",
                        in_x_dims.size()));
  PADDLE_ENFORCE_EQ(in_x_dims,
                    in_y_dims,
                    phi::errors::InvalidArgument(
                        "The dimensions of Input(X) must equal to be"
                        "the dimensions of Input(Indices), but received"
                        "dimensions of Input(X) is [%d], received dimensions"
                        "of Input(Indices) is [%d]",
                        in_x_dims,
                        in_y_dims));

  std::vector<int64_t> output_shape({in_x_dims[0], in_x_dims[1]});

  std::vector<int64_t> output_size_val(output_size.size(), -1);
  if (config.is_runtime || !output_size.FromTensor()) {
    output_size_val = output_size.GetData();
  }
  for (size_t i = 0; i < ksize.size(); ++i) {
    if (!config.is_runtime && in_x_dims[i + 2] <= 0) {
      output_shape.push_back(-1);
    } else {
      output_shape.push_back(output_size_val[i]);
    }
  }
  if (out != nullptr) {
    out->set_dims(phi::make_ddim(output_shape));
    out->set_dtype(x.dtype());
  }
}
void Unpool3dInferMeta(const MetaTensor& x,
                       const MetaTensor& indices,
                       const std::vector<int>& ksize,
                       const std::vector<int>& strides,
                       const std::vector<int>& paddings,
                       const std::vector<int>& output_size,
                       const std::string& data_format,
                       MetaTensor* out,
                       MetaConfig config) {
  auto in_x_dims = x.dims();
  auto in_y_dims = indices.dims();

  PADDLE_ENFORCE_EQ(in_x_dims.size() == 5,
                    true,
                    phi::errors::InvalidArgument(
                        "Unpool Intput(X) must be of 5-dimensional, but "
                        "received Input(X)'s dimensions is %d.",
                        in_x_dims.size()));
  PADDLE_ENFORCE_EQ(in_x_dims,
                    in_y_dims,
                    phi::errors::InvalidArgument(
                        "The dimensions of Input(X) must equal to be"
                        "the dimensions of Input(Indices), but received"
                        "dimensions of Input(X) is [%d], received dimensions"
                        "of Input(Indices) is [%d]",
                        in_x_dims,
                        in_y_dims));

  std::vector<int64_t> output_shape({in_x_dims[0], in_x_dims[1]});
  for (size_t i = 0; i < ksize.size(); ++i) {
    if (!config.is_runtime && in_x_dims[i + 2] <= 0) {
      output_shape.push_back(-1);
    } else {
      output_shape.push_back(output_size[i]);
    }
  }
  if (out != nullptr) {
    out->set_dims(phi::make_ddim(output_shape));
    out->set_dtype(x.dtype());
  }
}

}  // namespace phi

PD_REGISTER_INFER_META_FN(add_raw, phi::ElementwiseRawInferMeta);
PD_REGISTER_INFER_META_FN(conv2d_infer, phi::ConvInferInferMeta);
