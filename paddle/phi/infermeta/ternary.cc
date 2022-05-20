/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/infermeta/ternary.h"
#include "paddle/phi/core/ddim.h"
#include "paddle/phi/kernels/funcs/common_shape.h"

namespace phi {

void AccuracyInferMeta(const MetaTensor& out,
                       const MetaTensor& indice,
                       const MetaTensor& label,
                       MetaTensor* accuracy,
                       MetaTensor* correct,
                       MetaTensor* total,
                       MetaConfig config) {
  auto inference_dim = out.dims();
  auto label_dim = label.dims();
  // Assume indices has same shape as inference, because
  // it's the output of topk.
  PADDLE_ENFORCE_EQ(
      label_dim.size(),
      2,
      phi::errors::InvalidArgument(
          "ShapeError: label's dimensions of AccuracyOp must be 2. "
          "But received label's dimensions = %d, label's shape = [%s]",
          label_dim.size(),
          label_dim));
  if (config.is_runtime) {
    PADDLE_ENFORCE_EQ(label_dim[1],
                      1,
                      phi::errors::InvalidArgument(
                          "ShapeError: label's second dimension of "
                          "AccuracyOp must be 1. But received label's "
                          "second dimension is = %d, label's shape = [%s]",
                          label_dim[1],
                          label_dim));
    PADDLE_ENFORCE_EQ(
        inference_dim[0],
        label_dim[0],
        phi::errors::InvalidArgument(
            "ShapeError: the output's num_rows of AccuracyOp must be"
            " the same as label's num_rows. But received output's "
            "shape = [%s], label's shape = [%s], output's num_rows = %d, "
            "label's "
            "num_rows = %d",
            inference_dim,
            label_dim,
            inference_dim[0],
            label_dim[0]));
  }

  accuracy->set_dims({1});
  accuracy->set_dtype(out.dtype());
  correct->set_dims({1});
  correct->set_dtype(out.dtype());
  total->set_dims({1});
  total->set_dtype(out.dtype());
  accuracy->share_lod(out);
}

void AddmmInferMeta(const MetaTensor& input,
                    const MetaTensor& x,
                    const MetaTensor& y,
                    float alpha,
                    float beta,
                    MetaTensor* out) {
  auto input_dims = input.dims();
  auto x_dims = x.dims();
  auto y_dims = y.dims();

  auto ndim_input = input_dims.size();
  auto ndim_x = x_dims.size();
  auto ndim_y = y_dims.size();

  VLOG(3) << "addmm operator input.shape=" << input_dims
          << " x.shape=" << x_dims << " y.shape=" << y_dims << " beta=" << beta
          << " alpha=" << alpha << " ndim_input=" << ndim_input
          << " ndim_x=" << ndim_x << " ndim_y=" << ndim_y;

  PADDLE_ENFORCE_NE(
      product(input_dims),
      0,
      errors::PreconditionNotMet("The Input variable 'input' has not "
                                 "been initialized. You may need to confirm "
                                 "if you put exe.run(startup_program) "
                                 "after optimizer.minimize function."));

  PADDLE_ENFORCE_NE(
      product(x_dims),
      0,
      errors::PreconditionNotMet("The Input variable 'x' has not "
                                 "been initialized. You may need to confirm "
                                 "if you put exe.run(startup_program) "
                                 "after optimizer.minimize function."));

  PADDLE_ENFORCE_NE(
      product(y_dims),
      0,
      errors::PreconditionNotMet("The Input variable 'y' has not "
                                 "been initialized. You may need to confirm "
                                 "if you put exe.run(startup_program) "
                                 "after optimizer.minimize function."));
  // dim check
  PADDLE_ENFORCE_EQ(
      ndim_input,
      2,
      errors::InvalidArgument("The input tensor input's dimension must be 2. "
                              "But received input's dimension = [%s].",
                              ndim_input));
  PADDLE_ENFORCE_EQ(
      ndim_x,
      2,
      errors::InvalidArgument("The input tensor x's dimension must be 2. "
                              "But received x's dimension = [%s].",
                              ndim_x));
  PADDLE_ENFORCE_EQ(
      ndim_y,
      2,
      errors::InvalidArgument("The input tensor y's dimension must be 2. "
                              "But received y's dimension = [%s].",
                              ndim_y));

  std::vector<int64_t> output_dims;
  output_dims.push_back(x_dims[0]);
  output_dims.push_back(y_dims[1]);

  out->set_dims(make_ddim(output_dims));
  out->share_lod(input);
  out->set_dtype(input.dtype());
}

void ArangeInferMeta(const MetaTensor& start,
                     const MetaTensor& end,
                     const MetaTensor& step,
                     MetaTensor* out) {
  auto start_dims = start.dims();
  auto end_dims = end.dims();
  auto step_dims = step.dims();
  PADDLE_ENFORCE_EQ(
      start_dims.size(),
      1,
      phi::errors::InvalidArgument(
          "The dim of the shape of Input(Start) should be 1, but got %d",
          start_dims.size()));

  PADDLE_ENFORCE_EQ(start_dims[0],
                    1,
                    phi::errors::InvalidArgument(
                        "The first dim of the shape of Input(Start) should "
                        "be 1, but got %d",
                        start_dims[0]));
  PADDLE_ENFORCE_EQ(
      end_dims.size(),
      1,
      phi::errors::InvalidArgument(
          "The dim of the shape of Input(End) should be 1, but got %d",
          end_dims.size()));

  PADDLE_ENFORCE_EQ(
      end_dims[0],
      1,
      phi::errors::InvalidArgument("The first dim of the shape of "
                                   "Input(End) should be 1, but got %d",
                                   end_dims[0]));
  PADDLE_ENFORCE_EQ(
      step_dims.size(),
      1,
      phi::errors::InvalidArgument(
          "The dim of the shape of Input(Step) should be 1, but got %d",
          step_dims.size()));

  PADDLE_ENFORCE_EQ(step_dims[0],
                    1,
                    phi::errors::InvalidArgument(
                        "The first dim of the shape of Input(Step) should "
                        "be 1, but got %d",
                        step_dims[0]));
  out->set_dims({-1});
  out->set_dtype(start.dtype());
}

void GraphSendRecvInferMeta(const MetaTensor& x,
                            const MetaTensor& src_index,
                            const MetaTensor& dst_index,
                            const std::string& pool_type,
                            int64_t out_size,
                            MetaTensor* out,
                            MetaTensor* dst_count) {
  auto src_index_dims = src_index.dims();
  if (src_index_dims.size() == 2) {
    PADDLE_ENFORCE_EQ(src_index_dims[1],
                      1,
                      phi::errors::InvalidArgument(
                          "The last dim of Src_index should be 1 when it "
                          "is 2D, but we get %d",
                          src_index_dims[1]));
  } else {
    PADDLE_ENFORCE_EQ(
        src_index_dims.size(),
        1,
        phi::errors::InvalidArgument(
            "The Src_index should be 1D, when it is not 2D, but we get %d",
            src_index_dims.size()));
  }

  auto dst_index_dims = dst_index.dims();
  if (dst_index_dims.size() == 2) {
    PADDLE_ENFORCE_EQ(dst_index_dims[1],
                      1,
                      phi::errors::InvalidArgument(
                          "The last dim of Dst_index should be 1 when it "
                          "is 2D, but we get %d",
                          dst_index_dims[1]));
  } else {
    PADDLE_ENFORCE_EQ(
        dst_index_dims.size(),
        1,
        phi::errors::InvalidArgument("The Dst_index should be 1D, "
                                     "when it is not 2D, but we get %d",
                                     dst_index_dims.size()));
  }

  PADDLE_ENFORCE_EQ(src_index_dims[0],
                    dst_index_dims[0],
                    phi::errors::InvalidArgument(
                        "Src_index and Dst_index should have the same shape."));

  auto dims = x.dims();
  if (out_size <= 0) {
    out->set_dims(dims);
  } else {
    std::vector<int64_t> dims_ = phi::vectorize(dims);
    if (dims_.size() > 0) {
      dims_[0] = out_size;
    }
    out->set_dims(phi::make_ddim(dims_));
  }
  out->set_dtype(x.dtype());

  if (pool_type == "MEAN") {
    if (out_size <= 0) {
      dst_count->set_dims({dims[0]});
    } else {
      dst_count->set_dims({out_size});
    }
    dst_count->set_dtype(DataType::INT32);
  }
}

void LayerNormInferMeta(const MetaTensor& x,
                        paddle::optional<const MetaTensor&> scale,
                        paddle::optional<const MetaTensor&> bias,
                        float epsilon,
                        int begin_norm_axis,
                        bool is_test,
                        MetaTensor* out,
                        MetaTensor* mean,
                        MetaTensor* variance,
                        MetaConfig config) {
  auto x_dim = x.dims();
  PADDLE_ENFORCE_LT(
      begin_norm_axis,
      x_dim.size(),
      phi::errors::InvalidArgument(
          "'begin_norm_axis' must be less than the dimensions of X,"
          "But received 'begin_norm_axis' is [%d],"
          "received the dimensions of X is [%d].",
          begin_norm_axis,
          x_dim.size()));

  auto matrix_dim = phi::flatten_to_2d(x_dim, begin_norm_axis);
  int left = static_cast<int>(matrix_dim[0]);
  int right = static_cast<int>(matrix_dim[1]);
  if (scale.get_ptr() != nullptr) {
    PADDLE_ENFORCE_EQ(scale->dims().size(),
                      1,
                      phi::errors::InvalidArgument(
                          "The dimensions of Input(Scale) must be 1, but "
                          "received dimensions of"
                          "Input(Scale) is [%d]",
                          scale->dims().size()));
  }

  if (config.is_runtime && scale.get_ptr() != nullptr) {
    PADDLE_ENFORCE_EQ(
        scale->dims()[0],
        right,
        phi::errors::InvalidArgument(
            "The first dimension value of Input(Scale) must equal to be the"
            "second dimension value of the flattened 2D matrix of Input(X),"
            "But received the first dimension value of Input(Scale) is"
            "[%d], the second dimension value of the flattened 2D matrix of"
            " Input(Scale) is [%d].",
            scale->dims()[0],
            right));
  }
  if (bias.get_ptr() != nullptr) {
    PADDLE_ENFORCE_EQ(bias->dims().size(),
                      1,
                      phi::errors::InvalidArgument(
                          "The dimensions of Input(Bias) must be 1, but "
                          "received dimensions of"
                          "Input(Bias) is [%d]",
                          bias->dims().size()));
  }
  if (config.is_runtime && bias.get_ptr() != nullptr) {
    PADDLE_ENFORCE_EQ(
        bias->dims()[0],
        right,
        phi::errors::InvalidArgument(
            "The first dimension value of Input(Bias) must equal to be the"
            "second dimension value of the flattened 2D matrix of Input(X),"
            "But received the first dimension value of Input(Bias) is"
            "[%d], the second dimension value of the flattened 2D matrix of"
            " Input(Bias) is [%d].",
            bias->dims()[0],
            right));
  }

  out->set_dims(x_dim);
  if (mean) {
    mean->set_dims({left});
  }
  if (variance) {
    variance->set_dims({left});
  }
  out->share_lod(x);
}

void LayerNormGradInferMeta(const MetaTensor& x,
                            paddle::optional<const MetaTensor&> y,
                            paddle::optional<const MetaTensor&> z,
                            MetaTensor* dx,
                            MetaTensor* dy,
                            MetaTensor* dz) {
  if (dx) {
    dx->share_meta(x);
  }
  if (dy && (y.get_ptr() != nullptr)) {
    dy->share_meta(*y.get_ptr());
  }
  if (dz && (z.get_ptr() != nullptr)) {
    dz->share_meta(*z.get_ptr());
  }
}

void LerpInferMeta(const MetaTensor& x,
                   const MetaTensor& y,
                   const MetaTensor& weight,
                   MetaTensor* out) {
  auto x_dims = x.dims();
  auto y_dims = y.dims();
  auto w_dims = weight.dims();
  DDim out_dims;
  out_dims = funcs::GetOutputDims(x_dims, y_dims);
  if (w_dims.size() > 1 || w_dims[0] != 1) {
    out_dims = funcs::GetOutputDims(out_dims, w_dims);
  }
  out->set_dims(out_dims);
  out->set_dtype(x.dtype());
  out->share_lod(x);
}

void LinspaceRawInferMeta(const MetaTensor& start,
                          const MetaTensor& stop,
                          const MetaTensor& number,
                          MetaTensor* out) {
  auto s_dims = start.dims();
  PADDLE_ENFORCE_EQ(
      (s_dims.size() == 1) && (s_dims[0] == 1),
      true,
      phi::errors::InvalidArgument("The shape of Input(Start) must be [1],"
                                   "but received input shape is [%s].",
                                   s_dims));
  auto e_dims = stop.dims();
  PADDLE_ENFORCE_EQ(
      (e_dims.size() == 1) && (e_dims[0] == 1),
      true,
      phi::errors::InvalidArgument("The shape of Input(Stop) must be [1],"
                                   "but received input shape is [%s].",
                                   e_dims));
  auto step_dims = number.dims();
  PADDLE_ENFORCE_EQ(
      (step_dims.size() == 1) && (step_dims[0] == 1),
      true,
      phi::errors::InvalidArgument("The shape of Input(Num) must be [1],"
                                   "but received input shape is [%s].",
                                   step_dims));
  out->set_dims(phi::make_ddim({-1}));
  out->set_dtype(start.dtype());
}

void LinspaceInferMeta(const MetaTensor& start,
                       const MetaTensor& stop,
                       const MetaTensor& number,
                       DataType dtype,
                       MetaTensor* out) {
  LinspaceRawInferMeta(start, stop, number, out);
}

void NllLossRawInferMeta(const MetaTensor& input,
                         const MetaTensor& label,
                         paddle::optional<const MetaTensor&> weight,
                         int64_t ignore_index,
                         const std::string& reduction,
                         MetaTensor* out,
                         MetaTensor* total_weight,
                         MetaConfig config) {
  auto x_dims = input.dims();
  auto label_dims = label.dims();
  PADDLE_ENFORCE_EQ(x_dims.size() == 2 || x_dims.size() == 4,
                    true,
                    phi::errors::InvalidArgument(
                        "The tensor rank of Input(X) must be 2 or 4."));
  bool contain_unknown_dim =
      phi::contain_unknown_dim(x_dims) || phi::contain_unknown_dim(label_dims);
  bool check = config.is_runtime || !contain_unknown_dim;
  if (check) {
    PADDLE_ENFORCE_EQ(
        x_dims[0],
        label_dims[0],
        phi::errors::InvalidArgument(
            "ShapeError: Expected input batch_size to match label batch_size,"
            "But received: the Input(x) batch_size is [%s], the Input(label) "
            " batch_size is [%s].",
            x_dims[0],
            label_dims[0]));
    if (weight.get_ptr() != nullptr) {
      auto w_dims = weight->dims();
      PADDLE_ENFORCE_EQ(
          w_dims.size(),
          1,
          phi::errors::InvalidArgument("Input(Weight) should be a 1D tensor."));
      PADDLE_ENFORCE_EQ(
          x_dims[1],
          w_dims[0],
          phi::errors::InvalidArgument(
              "Expected input tensor Weight's size should equal "
              "to the first dimension of the input tensor X. But received "
              "Weight's "
              "size is %d, the first dimension of input X is %d",
              w_dims[0],
              x_dims[1]));
    }
  }
  if (x_dims.size() == 2) {
    if (reduction == "none") {
      out->set_dims({x_dims[0]});
    } else {
      out->set_dims({1});
    }
  } else if (x_dims.size() == 4) {
    PADDLE_ENFORCE_EQ(label_dims.size(),
                      3,
                      phi::errors::InvalidArgument(
                          "Expected Input(Lable) dimensions=3, received %d.",
                          label_dims.size()));
    auto input0 = x_dims[0];
    auto input2 = x_dims[2];
    auto input3 = x_dims[3];
    auto label0 = label_dims[0];
    auto label1 = label_dims[1];
    auto label2 = label_dims[2];
    PADDLE_ENFORCE_EQ(
        input0 == label0 && input2 == label1 && input3 == label2,
        true,
        phi::errors::InvalidArgument("Input(X) tensor shape should "
                                     "match to Input(Label) tensor "
                                     "shape."));
    if (reduction == "none") {
      out->set_dims({x_dims[0], x_dims[2], x_dims[3]});
    } else {
      out->set_dims({1});
    }
  }
  total_weight->set_dims({1});
  out->set_dtype(input.dtype());
  total_weight->set_dtype(input.dtype());
}

void PutAlongAxisInferMeta(const MetaTensor& x,
                           const MetaTensor& index,
                           const MetaTensor& value,
                           int axis,
                           const std::string& reduce,
                           MetaTensor* out) {
  out->set_dims(x.dims());
  out->set_dtype(x.dtype());
}

void RoiAlignInferMeta(const MetaTensor& x,
                       const MetaTensor& boxes,
                       paddle::optional<const MetaTensor&> boxes_num,
                       int pooled_height,
                       int pooled_width,
                       float spatial_scale,
                       int sampling_ratio,
                       bool aligned,
                       MetaTensor* out,
                       MetaConfig config) {
  auto input_dims = x.dims();
  auto boxes_dims = boxes.dims();

  if (boxes_num) {
    auto boxes_num_dims = boxes_num->dims();
    PADDLE_ENFORCE_EQ(
        boxes_num_dims.size(),
        1,
        phi::errors::InvalidArgument("The size of boxes_num should be 1"
                                     ", but received size = %d",
                                     boxes_num_dims.size()));
  }
  PADDLE_ENFORCE_EQ(input_dims.size(),
                    4,
                    phi::errors::InvalidArgument(
                        "The format of Input(x) in"
                        "RoiAlignOp is NCHW. And the rank of input must be 4. "
                        "But received rank = %d",
                        input_dims.size()));
  PADDLE_ENFORCE_EQ(boxes_dims.size(),
                    2,
                    phi::errors::InvalidArgument("The rank of Input(boxes) "
                                                 "in RoiAlignOp should be 2. "
                                                 "But the rank of boxes is %d",
                                                 boxes_dims.size()));
  if (config.is_runtime) {
    PADDLE_ENFORCE_EQ(boxes_dims[1],
                      4,
                      phi::errors::InvalidArgument(
                          "The second dimension "
                          "of Input(boxes) should be 4. But received the "
                          "dimension = %d",
                          boxes_dims[1]));
  }

  PADDLE_ENFORCE_GT(pooled_height,
                    0,
                    phi::errors::InvalidArgument(
                        "The 'pooled_height' attribute in RoiAlignOp is "
                        "invalid. The height must be greater than 0. But "
                        "received 'pooled_height' = %d",
                        pooled_height));
  PADDLE_ENFORCE_GT(pooled_width,
                    0,
                    phi::errors::InvalidArgument(
                        "The 'pooled_width' attribute in RoiAlignOp is "
                        "invalid. The width must be greater than 0. But "
                        "received 'pooled_width' = %d",
                        pooled_width));
  PADDLE_ENFORCE_GT(spatial_scale,
                    0.0f,
                    phi::errors::InvalidArgument(
                        "The 'spatial_scale' attribute in RoiAlignOp is "
                        "invalid. The scale must be greater than 0. But "
                        "received 'spatial_scale' = %f",
                        spatial_scale));

  auto out_dims = input_dims;
  out_dims[0] = boxes_dims[0];
  out_dims[1] = input_dims[1];
  out_dims[2] = pooled_height;
  out_dims[3] = pooled_width;

  out->set_dims(out_dims);
  out->set_dtype(x.dtype());
}

void RoiPoolInferMeta(const MetaTensor& x,
                      const MetaTensor& boxes,
                      paddle::optional<const MetaTensor&> boxes_num,
                      int pooled_height,
                      int pooled_width,
                      float spatial_scale,
                      MetaTensor* out,
                      MetaTensor* arg_max) {
  auto input_dims = x.dims();
  auto boxes_dims = boxes.dims();

  if (boxes_num) {
    auto boxes_num_dims = boxes_num->dims();
    PADDLE_ENFORCE_EQ(
        boxes_num_dims.size(),
        1,
        phi::errors::InvalidArgument("The second dimension of boxes_num should "
                                     "be 1, but received dimension is %d",
                                     boxes_num_dims.size()));
  }
  PADDLE_ENFORCE_EQ(input_dims.size(),
                    4,
                    phi::errors::InvalidArgument(
                        "The input data should be a four-dimensional "
                        "tensor with [N,C,H,W], but received input data with "
                        " %d dimension",
                        input_dims.size()));
  PADDLE_ENFORCE_EQ(
      boxes_dims.size(),
      2,
      phi::errors::InvalidArgument(
          "boxes should be a 2-D LoDTensor with shape (num_boxes, 4)"
          "given as [[x1, y1, x2, y2], ...], but received boxes is "
          "%d-dimensional LoDTensor",
          boxes_dims.size()));
  PADDLE_ENFORCE_EQ(
      boxes_dims[1],
      4,
      phi::errors::InvalidArgument(
          "boxes should be a 2-D LoDTensor with shape (num_boxes, 4)"
          "given as [[x1, y1, x2, y2], ...]. But the second dimension of  "
          "the received data is %d",
          boxes_dims[1]));

  PADDLE_ENFORCE_GT(
      pooled_height,
      0,
      phi::errors::OutOfRange("The pooled output height must be greater than 0"
                              "but received height is %d",
                              pooled_height));
  PADDLE_ENFORCE_GT(
      pooled_width,
      0,
      phi::errors::OutOfRange("The pooled output width must be greater than 0"
                              "but received width is %d",
                              pooled_width));
  PADDLE_ENFORCE_GT(
      spatial_scale,
      0.0f,
      phi::errors::OutOfRange("The spatial scale must be greater than 0, "
                              "but received spatial scale is %f",
                              spatial_scale));

  auto out_dims = input_dims;
  out_dims[0] = boxes_dims[0];
  out_dims[1] = input_dims[1];
  out_dims[2] = pooled_height;
  out_dims[3] = pooled_width;

  out->set_dims(out_dims);
  out->set_dtype(x.dtype());
  arg_max->set_dims(out_dims);
  arg_max->set_dtype(DataType::INT64);
}

void ScatterInferMeta(const MetaTensor& x,
                      const MetaTensor& index,
                      const MetaTensor& updates,
                      bool overwrite,
                      MetaTensor* out) {
  const auto& updates_dims = updates.dims();
  const auto& ref_dims = x.dims();
  const auto& index_dims = index.dims();
  PADDLE_ENFORCE_EQ(
      index_dims.size(),
      1,
      phi::errors::InvalidArgument(
          "The size of Input(Ids)'s shape should be equal to 1, but "
          "received the rank of Input(Ids) is %d.",
          index_dims.size()));
  PADDLE_ENFORCE_EQ(
      ref_dims.size(),
      updates_dims.size(),
      phi::errors::InvalidArgument(
          "Input(X) and Input(Updates) should have the same shape size, "
          "but received the size of Input(x)'s shape is %d, the size of "
          "Input(Updates)'s shape is %d.",
          ref_dims.size(),
          updates_dims.size()));
  PADDLE_ENFORCE_EQ(
      updates_dims[0],
      index_dims[0],
      phi::errors::InvalidArgument(
          "Input(Updates) and Input(Ids) should have same batch-size, but"
          " received Input(Updates)'s batch-size is %d, Input(Ids)'s "
          "batch-size is %d.",
          updates_dims[0],
          index_dims[0]));
  out->set_dims(ref_dims);
  out->share_lod(x);
  out->set_dtype(x.dtype());
}

void ScatterNdAddInferMeta(const MetaTensor& x,
                           const MetaTensor& index,
                           const MetaTensor& updates,
                           MetaTensor* out) {
  const auto& ref_dims = x.dims();
  auto ref_dims_size = ref_dims.size();
  const auto& index_dims = index.dims();
  auto index_dims_size = index_dims.size();
  const auto& updates_dims = updates.dims();
  auto updates_dims_size = updates_dims.size();

  PADDLE_ENFORCE_LE(
      index_dims[index_dims_size - 1],
      ref_dims_size,
      phi::errors::InvalidArgument(
          "The last dimension of Input(Index)'s shape should be no greater "
          "than the rank of Input(X), but received the last dimension of "
          "Input(Index)'s shape is %d, the rank of Input(X) is %d.",
          index_dims[index_dims_size - 1],
          ref_dims_size));
  PADDLE_ENFORCE_GE(index_dims_size,
                    2UL,
                    phi::errors::InvalidArgument(
                        "The rank of Input(Index) should be greater than 1, "
                        "but received the rank of Input(Index) is %d.",
                        index_dims_size));

  // update.shape = index.shape[:-1] + output.shape[index.shape[-1]:]
  std::vector<int64_t> r_updates_dims;
  for (int64_t i = 0; i < index_dims_size - 1; ++i) {
    r_updates_dims.emplace_back(index_dims[i]);
  }
  for (int64_t i = index_dims[index_dims_size - 1]; i < ref_dims_size; ++i) {
    r_updates_dims.emplace_back(ref_dims[i]);
  }

  PADDLE_ENFORCE_EQ(
      r_updates_dims.size(),
      updates_dims_size,
      phi::errors::InvalidArgument(
          "Updates has wrong shape. The shape of Updates and Input(Updates) "
          "should be same, but received the shape of Updates is %d, "
          "the shape of Input(Updates) is %d.",
          r_updates_dims.size(),
          updates_dims_size));

  for (int64_t i = 0; i < updates_dims_size; ++i) {
    PADDLE_ENFORCE_EQ(
        r_updates_dims[i],
        updates_dims[i],
        phi::errors::InvalidArgument(
            "Updates has wrong shape. The dimensions of Updates and "
            "Input(Updates) should match, but received Updates's"
            "%d-th dimension is %d, Input(Updates)'s %d-th "
            "dimension is %d.",
            i,
            r_updates_dims[i],
            i,
            updates_dims[i]));
  }
  out->set_dims(ref_dims);
  out->share_lod(x);
  out->set_dtype(x.dtype());
}

void ViterbiDecodeInferMeta(const MetaTensor& input,
                            const MetaTensor& transition,
                            const MetaTensor& length,
                            bool include_bos_eos_tag,
                            MetaTensor* scores,
                            MetaTensor* path,
                            MetaConfig config) {
  auto in_dims = input.dims();
  PADDLE_ENFORCE_EQ(in_dims.size(),
                    3,
                    phi::errors::InvalidArgument(
                        "The rank of Input in ViterbiDecode  must be 3. But "
                        "received Input's rank is %d.",
                        in_dims.size()));
  auto length_dims = length.dims();
  PADDLE_ENFORCE_EQ(length_dims.size(),
                    1,
                    phi::errors::InvalidArgument(
                        "The rank of Length in ViterbiDecode must be 1. But "
                        "received Length's rank is %d.",
                        length_dims.size()));
  auto transition_dims = transition.dims();
  PADDLE_ENFORCE_EQ(
      transition_dims.size(),
      2,
      phi::errors::InvalidArgument(
          "The rank of Transition in ViterbiDecode must be 2. But "
          "received Transition's rank is %d.",
          transition_dims.size()));
  if (config.is_runtime) {
    PADDLE_ENFORCE_EQ(
        in_dims[0],
        length_dims[0],
        phi::errors::InvalidArgument(
            "The batch size of Input and Length should be equal."));
    PADDLE_ENFORCE_EQ(in_dims[2],
                      transition_dims[0],
                      phi::errors::InvalidArgument(
                          "The number of tags of Input (%d) and Transition "
                          "(%d) should be equal.",
                          transition_dims[0],
                          in_dims[2]));
  }
  scores->set_dims(length_dims);
  scores->set_dtype(length.dtype());
}

}  // namespace phi
