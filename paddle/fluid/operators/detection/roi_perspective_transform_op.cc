/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <algorithm>
#include <memory>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;

template <typename T>
bool GT_E(T a, T b) {
  return (a > b) || fabs(a - b) < 1e-4;
}

template <typename T>
bool LT_E(T a, T b) {
  return (a < b) || fabs(a - b) < 1e-4;
}

template <typename T>
bool GT(T a, T b) {
  return (a - b) > 1e-4;
}

/*
*check if (x, y) is in the boundary of roi
*/
template <typename T>
bool in_quad(T x, T y, T roi_x[], T roi_y[]) {
  for (int i = 0; i < 4; i++) {
    T xs = roi_x[i];
    T ys = roi_y[i];
    T xe = roi_x[(i + 1) % 4];
    T ye = roi_y[(i + 1) % 4];
    if (fabs(ys - ye) < 1e-4) {
      if (fabs(y - ys) < 1e-4 && fabs(y - ye) < 1e-4 &&
          GT_E<T>(x, std::min(xs, xe)) && LT_E<T>(x, std::max(xs, xe))) {
        return true;
      }
    } else {
      T intersec_x = (y - ys) * (xe - xs) / (ye - ys) + xs;
      if (fabs(intersec_x - x) < 1e-4 && GT_E<T>(y, std::min(ys, ye)) &&
          LT_E<T>(y, std::max(ys, ye))) {
        return true;
      }
    }
  }

  int n_cross = 0;
  for (int i = 0; i < 4; i++) {
    T xs = roi_x[i];
    T ys = roi_y[i];
    T xe = roi_x[(i + 1) % 4];
    T ye = roi_y[(i + 1) % 4];
    if (fabs(ys - ye) < 1e-4) {
      continue;
    }
    if (LT_E<T>(y, std::min(ys, ye)) || GT<T>(y, std::max(ys, ye))) {
      continue;
    }
    T intersec_x = (y - ys) * (xe - xs) / (ye - ys) + xs;
    if (fabs(intersec_x - x) < 1e-4) {
      return true;
    }
    if (GT<T>(intersec_x, x)) {
      n_cross++;
    }
  }
  return (n_cross % 2 == 1);
}

/**
 * Get the matrix of perspective transform.
 *
 * dx1 = x1 - x2
 * dx2 = x3 - x2
 * dx3 = x0 - x1 + x2 - x3
 * dy1 = y1 - y2
 * dy2 = y3 - y2
 * dy3 = y0 - y1 + y2 - y3
 *
 * a11 = (x1 - x0 + a31 * (w - 1) * x1) / (w - 1)
 * a12 = (x3 - x0 + a32 * (h - 1) * x3) / (h - 1)
 * a13 = x0
 * a21 = (y1 - y0 + a31 * (w - 1) * y1) / (w - 1)
 * a22 = (y3 - y0 + a32 * (h - 1) * y3) / (h - 1)
 * a23 = y0
 * a31 = (dx3 * dy2 - dx2 * dy3) / (dx1 * dy2 - dx2 * dy1) / (w - 1)
 * a32 = (dx1 * dy3 - dx3 * dy1) / (dx1 * dy2 - dx2 * dy1) / (h - 1)
 * a33 = 1
 */
template <typename T>
void get_transform_matrix(const int transformed_width,
                          const int transformed_height, T roi_x[], T roi_y[],
                          T matrix[]) {
  T x0 = roi_x[0];
  T x1 = roi_x[1];
  T x2 = roi_x[2];
  T x3 = roi_x[3];
  T y0 = roi_y[0];
  T y1 = roi_y[1];
  T y2 = roi_y[2];
  T y3 = roi_y[3];

  // Estimate the height and width of RoI
  T len1 = sqrt((x0 - x1) * (x0 - x1) + (y0 - y1) * (y0 - y1));
  T len2 = sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
  T len3 = sqrt((x2 - x3) * (x2 - x3) + (y2 - y3) * (y2 - y3));
  T len4 = sqrt((x3 - x0) * (x3 - x0) + (y3 - y0) * (y3 - y0));
  T estimated_height = (len2 + len4) / 2.0;
  T estimated_width = (len1 + len3) / 2.0;

  // Get the normalized height and normalized width
  int normalized_height = std::max(2, transformed_height);
  int normalized_width =
      std::round(estimated_width * (normalized_height - 1) / estimated_height) +
      1;
  normalized_width = std::max(2, std::min(normalized_width, transformed_width));

  T dx1 = x1 - x2;
  T dx2 = x3 - x2;
  T dx3 = x0 - x1 + x2 - x3;
  T dy1 = y1 - y2;
  T dy2 = y3 - y2;
  T dy3 = y0 - y1 + y2 - y3;

  matrix[6] = (dx3 * dy2 - dx2 * dy3) / (dx1 * dy2 - dx2 * dy1 + 1e-5) /
              (normalized_width - 1);
  matrix[7] = (dx1 * dy3 - dx3 * dy1) / (dx1 * dy2 - dx2 * dy1 + 1e-5) /
              (normalized_height - 1);
  matrix[8] = 1;

  matrix[3] = (y1 - y0 + matrix[6] * (normalized_width - 1) * y1) /
              (normalized_width - 1);
  matrix[4] = (y3 - y0 + matrix[7] * (normalized_height - 1) * y3) /
              (normalized_height - 1);
  matrix[5] = y0;

  matrix[0] = (x1 - x0 + matrix[6] * (normalized_width - 1) * x1) /
              (normalized_width - 1);
  matrix[1] = (x3 - x0 + matrix[7] * (normalized_height - 1) * x3) /
              (normalized_height - 1);
  matrix[2] = x0;
}

/**
 * Get the source coordinates in the input feature map.
 *
 * (u, v, w)^matrix = matrix * (out_w, out_h, 1)^matrix
 *
 * in_w = u / w
 * in_h = v / w
 *
 */
template <typename T>
void get_source_coords(T matrix[], int out_w, int out_h, T* in_w, T* in_h) {
  T u = matrix[0] * out_w + matrix[1] * out_h + matrix[2];
  T v = matrix[3] * out_w + matrix[4] * out_h + matrix[5];
  T w = matrix[6] * out_w + matrix[7] * out_h + matrix[8];

  in_w[0] = u / w;
  in_h[0] = v / w;
}

/**
 * Perform bilinear interpolation in the input feature map.
 */
template <typename T>
void bilinear_interpolate(const T* in_data, const int channels, const int width,
                          const int height, int in_n, int in_c, T in_w, T in_h,
                          T* val) {
  // Deal with cases that source coords are out of feature map boundary
  if (GT_E<T>(-0.5, in_w) || GT_E<T>(in_w, width - 0.5) ||
      GT_E<T>(-0.5, in_h) || GT_E<T>(in_h, height - 0.5)) {
    // empty
    val[0] = 0.0;
    return;
  }

  if (GT_E<T>(0, in_w)) {
    in_w = 0;
  }
  if (GT_E<T>(0, in_h)) {
    in_h = 0;
  }

  int in_w_floor = floor(in_w);
  int in_h_floor = floor(in_h);
  int in_w_ceil;
  int in_h_ceil;

  if (GT_E<T>(in_w_floor, width - 1)) {
    in_w_ceil = in_w_floor = width - 1;
    in_w = static_cast<T>(in_w_floor);
  } else {
    in_w_ceil = in_w_floor + 1;
  }

  if (GT_E<T>(in_h_floor, height - 1)) {
    in_h_ceil = in_h_floor = height - 1;
    in_h = static_cast<T>(in_h_floor);
  } else {
    in_h_ceil = in_h_floor + 1;
  }
  T w_floor = in_w - in_w_floor;
  T h_floor = in_h - in_h_floor;
  T w_ceil = 1 - w_floor;
  T h_ceil = 1 - h_floor;
  const T* data = in_data + (in_n * channels + in_c) * height * width;
  // Do bilinear interpolation
  T v1 = data[in_h_floor * width + in_w_floor];
  T v2 = data[in_h_ceil * width + in_w_floor];
  T v3 = data[in_h_ceil * width + in_w_ceil];
  T v4 = data[in_h_floor * width + in_w_ceil];
  T w1 = w_ceil * h_ceil;
  T w2 = w_ceil * h_floor;
  T w3 = w_floor * h_floor;
  T w4 = w_floor * h_ceil;
  val[0] = w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4;
}

template <typename T>
class CPUROIPerspectiveTransformOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* in = ctx.Input<framework::Tensor>("X");
    auto* rois = ctx.Input<framework::LoDTensor>("ROIs");
    auto* out = ctx.Output<framework::Tensor>("Out");
    auto* mask = ctx.Output<framework::Tensor>("Mask");
    auto* out_transform_matrix =
        ctx.Output<framework::Tensor>("TransformMatrix");
    auto transformed_height = ctx.Attr<int>("transformed_height");
    auto transformed_width = ctx.Attr<int>("transformed_width");
    auto spatial_scale = ctx.Attr<float>("spatial_scale");

    auto in_dims = in->dims();
    int channels = in_dims[1];
    int in_height = in_dims[2];
    int in_width = in_dims[3];
    int rois_num = rois->dims()[0];

    const T* input_data = in->data<T>();
    int* mask_data = mask->mutable_data<int>(ctx.GetPlace());

    framework::Tensor roi2image;
    roi2image.Resize({rois_num});
    int* roi2image_data = roi2image.mutable_data<int>(ctx.GetPlace());
    auto lod = rois->lod().back();
    for (size_t i = 0; i < lod.size() - 1; ++i) {
      for (size_t j = lod[i]; j < lod[i + 1]; ++j) {
        roi2image_data[j] = i;
      }
    }

    T* output_data = out->mutable_data<T>(ctx.GetPlace());
    const T* rois_data = rois->data<T>();

    T* transform_matrix =
        out_transform_matrix->mutable_data<T>({rois_num, 9}, ctx.GetPlace());

    for (int n = 0; n < rois_num; ++n) {
      const T* n_rois = rois_data + n * 8;
      T roi_x[4];
      T roi_y[4];
      for (int k = 0; k < 4; ++k) {
        roi_x[k] = n_rois[2 * k] * spatial_scale;
        roi_y[k] = n_rois[2 * k + 1] * spatial_scale;
      }
      int image_id = roi2image_data[n];
      // Get transform matrix
      T matrix[9];
      get_transform_matrix<T>(transformed_width, transformed_height, roi_x,
                              roi_y, matrix);
      for (int i = 0; i < 9; i++) {
        transform_matrix[n * 9 + i] = matrix[i];
      }
      for (int c = 0; c < channels; ++c) {
        for (int out_h = 0; out_h < transformed_height; ++out_h) {
          for (int out_w = 0; out_w < transformed_width; ++out_w) {
            int out_index =
                n * channels * transformed_height * transformed_width +
                c * transformed_height * transformed_width +
                out_h * transformed_width + out_w;
            T in_w, in_h;
            get_source_coords<T>(matrix, out_w, out_h, &in_w, &in_h);
            if (in_quad<T>(in_w, in_h, roi_x, roi_y)) {
              if (GT_E<T>(-0.5, in_w) ||
                  GT_E<T>(in_w, static_cast<T>(in_width - 0.5)) ||
                  GT_E<T>(-0.5, in_h) ||
                  GT_E<T>(in_h, static_cast<T>(in_height - 0.5))) {
                output_data[out_index] = 0.0;
                mask_data[(n * transformed_height + out_h) * transformed_width +
                          out_w] = 0;
              } else {
                bilinear_interpolate(input_data, channels, in_width, in_height,
                                     image_id, c, in_w, in_h,
                                     output_data + out_index);
                mask_data[(n * transformed_height + out_h) * transformed_width +
                          out_w] = 1;
              }
            } else {
              output_data[out_index] = 0.0;
              mask_data[(n * transformed_height + out_h) * transformed_width +
                        out_w] = 0;
            }
          }
        }
      }
    }
  }
};

template <typename T>
T get_feature_gradient(T xs, T ys, int w, int h, const int width,
                       const int height) {
  if (GT_E<T>(-0.5, xs) || GT_E<T>(xs, width - 0.5) || GT_E<T>(-0.5, ys) ||
      GT_E<T>(ys, height - 0.5)) {
    return 0;
  }

  if (GT_E<T>(0, xs)) {
    xs = 0;
  }
  if (GT_E<T>(0, ys)) {
    ys = 0;
  }

  int xs_floor = floor(xs);
  int ys_floor = floor(ys);
  int xs_ceil;
  int ys_ceil;

  if (GT_E(xs_floor, width - 1)) {
    xs_ceil = xs_floor = width - 1;
    xs = static_cast<T>(xs_floor);
  } else {
    xs_ceil = xs_floor + 1;
  }

  if (GT_E(ys_floor, height - 1)) {
    ys_ceil = ys_floor = height - 1;
    ys = static_cast<T>(ys_floor);
  } else {
    ys_ceil = ys_floor + 1;
  }

  T weight = 0;
  if (w == xs_floor) {
    if (h == ys_floor) {
      weight = (w + 1 - xs) * (h + 1 - ys);
    } else if (h == ys_ceil) {
      weight = (w + 1 - xs) * (ys + 1 - h);
    }
  } else if (w == xs_ceil) {
    if (h == ys_floor) {
      weight = (xs + 1 - w) * (h + 1 - ys);
    } else if (h == ys_ceil) {
      weight = (xs + 1 - w) * (ys + 1 - h);
    }
  }
  return weight;
}

template <typename T>
class CPUROIPerspectiveTransformGradOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* in = ctx.Input<framework::Tensor>("X");
    auto* rois = ctx.Input<framework::LoDTensor>("ROIs");
    auto* out_grad =
        ctx.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto* in_grad = ctx.Output<framework::Tensor>(framework::GradVarName("X"));

    auto transformed_height = ctx.Attr<int>("transformed_height");
    auto transformed_width = ctx.Attr<int>("transformed_width");
    auto spatial_scale = ctx.Attr<float>("spatial_scale");

    auto in_dims = in->dims();
    int batch_size = in_dims[0];
    int channels = in_dims[1];
    int in_height = in_dims[2];
    int in_width = in_dims[3];
    int rois_num = rois->dims()[0];

    T* in_grad_data = in_grad->mutable_data<T>(ctx.GetPlace());
    const T* out_grad_data = out_grad->data<T>();
    const T* rois_data = rois->data<T>();

    framework::Tensor roi2image;
    roi2image.Resize({rois_num});
    int* roi2image_data = roi2image.mutable_data<int>(ctx.GetPlace());
    auto lod = rois->lod().back();
    for (size_t i = 0; i < lod.size() - 1; ++i) {
      for (size_t j = lod[i]; j < lod[i + 1]; ++j) {
        roi2image_data[j] = i;
      }
    }

    for (int n = 0; n < batch_size; ++n) {
      for (int c = 0; c < channels; ++c) {
        for (int in_h = 0; in_h < in_height; ++in_h) {
          for (int in_w = 0; in_w < in_width; ++in_w) {
            T gradient = 0.0;
            for (size_t roi_idx = lod[n]; roi_idx < lod[n + 1]; ++roi_idx) {
              const T* rois = rois_data + roi_idx * 8;
              T roi_x[4];
              T roi_y[4];
              for (int k = 0; k < 4; ++k) {
                roi_x[k] = rois[2 * k] * spatial_scale;
                roi_y[k] = rois[2 * k + 1] * spatial_scale;
              }

              // Get transform matrix
              T matrix[9];
              get_transform_matrix<T>(transformed_width, transformed_height,
                                      roi_x, roi_y, matrix);
              const T* out_grad_ptr = out_grad_data +
                                      (roi_idx * channels + c) *
                                          transformed_height *
                                          transformed_width;
              for (int out_h = 0; out_h < transformed_height; ++out_h) {
                for (int out_w = 0; out_w < transformed_width; ++out_w) {
                  T src_w;
                  T src_h;
                  get_source_coords<T>(matrix, out_w, out_h, &src_w, &src_h);
                  if (in_quad<T>(src_w, src_h, roi_x, roi_y)) {
                    if (GT_E<T>(-0.5, src_w) ||
                        GT_E<T>(src_w, static_cast<T>(in_width - 0.5)) ||
                        GT_E<T>(-0.5, src_h) ||
                        GT_E<T>(src_h, static_cast<T>(in_height - 0.5))) {
                      continue;
                    }
                    T weight = get_feature_gradient<T>(src_w, src_h, in_w, in_h,
                                                       in_width, in_height);
                    gradient +=
                        out_grad_ptr[out_h * transformed_width + out_w] *
                        weight;
                  }
                }
              }
            }
            int out_idx = (n * channels + c) * in_height * in_width +
                          in_h * in_width + in_w;
            in_grad_data[out_idx] = gradient;
          }
        }
      }
    }
  }
};

class ROIPerspectiveTransformOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X",
                   "roi_perspective_transform");
    OP_INOUT_CHECK(ctx->HasInput("ROIs"), "Input", "ROIs",
                   "roi_perspective_transform");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Ountput", "Out",
                   "roi_perspective_transform");

    auto input_dims = ctx->GetInputDim("X");
    auto rois_dims = ctx->GetInputDim("ROIs");

    PADDLE_ENFORCE_EQ(input_dims.size(), 4,
                      platform::errors::InvalidArgument(
                          "The format of input tensor must be NCHW. But "
                          "received input dims is %d.",
                          input_dims.size()));
    PADDLE_ENFORCE_EQ(
        rois_dims.size(), 2,
        platform::errors::InvalidArgument(
            "ROIs should be a 2-D LoDTensor of shape (num_rois, 8)"
            "given as [[x0, y0, x1, y1, x2, y2, x3, y3], ...]. But received "
            "rois dims is %d",
            rois_dims.size()));
    PADDLE_ENFORCE_EQ(
        rois_dims[1], 8,
        platform::errors::InvalidArgument(
            "ROIs should be a 2-D LoDTensor of shape (num_rois, 8)"
            "given as [[x0, y0, x1, y1, x2, y2, x3, y3], ...]. But received %d",
            rois_dims[1]));

    int transformed_height = ctx->Attrs().Get<int>("transformed_height");
    int transformed_width = ctx->Attrs().Get<int>("transformed_width");
    float spatial_scale = ctx->Attrs().Get<float>("spatial_scale");

    PADDLE_ENFORCE_GT(
        transformed_height, 0,
        platform::errors::InvalidArgument("The transformed output height must "
                                          "greater than 0. But received %d.",
                                          transformed_height));
    PADDLE_ENFORCE_GT(
        transformed_width, 0,
        platform::errors::InvalidArgument("The transformed output width must "
                                          "greater than 0. But received %d.",
                                          transformed_width));
    PADDLE_ENFORCE_GT(
        spatial_scale, 0.0f,
        platform::errors::InvalidArgument(
            "The spatial scale must greater than 0. But received %f.",
            spatial_scale));
    std::vector<int64_t> out_dims_v({rois_dims[0],   // num_rois
                                     input_dims[1],  // channels
                                     static_cast<int64_t>(transformed_height),
                                     static_cast<int64_t>(transformed_width)});
    auto out_dims = phi::make_ddim(out_dims_v);

    std::vector<int64_t> mask_dims_v({rois_dims[0],  // num_rois
                                      1,             // channels
                                      static_cast<int64_t>(transformed_height),
                                      static_cast<int64_t>(transformed_width)});
    auto mask_dims = phi::make_ddim(mask_dims_v);

    std::vector<int64_t> matrix_dims_v({rois_dims[0], 9});
    auto matrix_dims = phi::make_ddim(matrix_dims_v);

    ctx->SetOutputDim("Out", out_dims);
    ctx->SetOutputDim("Mask", mask_dims);
    ctx->SetOutputDim("TransformMatrix", matrix_dims);
    ctx->SetOutputDim("Out2InIdx", out_dims);
    ctx->SetOutputDim("Out2InWeights", out_dims);
    ctx->ShareLoD("ROIs", /*->*/ "Out");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"),
        ctx.device_context());
  }
};

class ROIPerspectiveTransformGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Out")), "Input",
                   "Out@Grad", "roi_perspective_transform_grad");
    OP_INOUT_CHECK(ctx->HasOutputs(framework::GradVarName("X")), "Output",
                   "X@Grad", "roi_perspective_transform_grad");

    ctx->SetOutputsDim(framework::GradVarName("X"), ctx->GetInputsDim("X"));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"),
        ctx.device_context());
  }
};

class ROIPerspectiveTransformOpMaker
    : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "(Tensor), "
             "the input of ROIPerspectiveTransformOp. "
             "The format of input tensor is NCHW. Where N is batch size, "
             "C is the number of input channels, "
             "H is the height of the feature, and "
             "W is the width of the feature.");
    AddInput("ROIs",
             "(LoDTensor), "
             "ROIs (Regions of Interest) to be transformed. "
             "should be a 2-D LoDTensor of shape (num_rois, 8)"
             "given as [[x1, y1, x2, y2, x3, y3, x4, y4], ...]."
             "(x1, y1) is the top left coordinates, and "
             "(x2, y2) is the top right coordinates, and"
             "(x3, y3) is the bottom right coordinates, and"
             "(x4, y4) is the bottom left coordinates.");
    AddOutput(
        "Out",
        "(Tensor), "
        "The output of ROIPerspectiveTransformOp is a 4-D tensor with shape "
        "(num_rois, channels, transformed_h, transformed_w).");
    AddOutput("Mask",
              "(Tensor), "
              "The output mask of ROIPerspectiveTransformOp is a 4-D tensor "
              "with shape "
              "(num_rois, 1, transformed_h, transformed_w).");
    AddOutput("TransformMatrix",
              "(Tensor), "
              "The output transform matrix of ROIPerspectiveTransformOp is a "
              "1-D tensor with shape "
              "(num_rois, 9).");
    AddOutput("Out2InIdx",
              "(Tensor), "
              "An intermediate tensor used to map indexes of input feature map "
              "and indexes of output feature map."
              "The shape of the tensor is [out_size, 4] and out_size is the "
              "number of elements in output feature map.")
        .AsIntermediate();
    AddOutput("Out2InWeights",
              "(Tensor), "
              "An intermediate tensor used to record the weights of bilinear "
              "interpolatein for each element in output. The shape of the "
              "tensor is [out_size, 4] and out_size is the number of elements "
              "in output feature map.")
        .AsIntermediate();
    AddAttr<float>("spatial_scale",
                   "(float, default 1.0), "
                   "Spatial scale factor to scale ROI coords.")
        .SetDefault(1.0);
    AddAttr<int>("transformed_height",
                 "(int, default 1), "
                 "The height of transformed output.")
        .SetDefault(1);
    AddAttr<int>("transformed_width",
                 "(int, default 1), "
                 "The width of transformed output.")
        .SetDefault(1);
    AddComment(R"DOC(
**ROIPerspectiveTransform Operator**

    )DOC");
  }
};

template <typename T>
class ROIPerspectiveTransformGradMaker
    : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("roi_perspective_transform_grad");
    op->SetInput("X", this->Input("X"));
    op->SetInput("ROIs", this->Input("ROIs"));
    op->SetInput("Out2InIdx", this->Output("Out2InIdx"));
    op->SetInput("Out2InWeights", this->Output("Out2InWeights"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetAttrMap(this->Attrs());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(
    roi_perspective_transform, ops::ROIPerspectiveTransformOp,
    ops::ROIPerspectiveTransformOpMaker,
    ops::ROIPerspectiveTransformGradMaker<paddle::framework::OpDesc>,
    ops::ROIPerspectiveTransformGradMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(roi_perspective_transform_grad,
                  ops::ROIPerspectiveTransformGradOp);
REGISTER_OP_CPU_KERNEL(roi_perspective_transform,
                       ops::CPUROIPerspectiveTransformOpKernel<float>);
REGISTER_OP_CPU_KERNEL(roi_perspective_transform_grad,
                       ops::CPUROIPerspectiveTransformGradOpKernel<float>);
