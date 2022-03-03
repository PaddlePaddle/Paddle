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
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/device/gpu/gpu_primitives.h"
#include "paddle/fluid/platform/float16.h"
#include "paddle/phi/kernels/funcs/math_function.h"

using paddle::platform::PADDLE_CUDA_NUM_THREADS;
using paddle::platform::float16;

namespace paddle {
namespace operators {

// CUDA: index helpers
#define idx4_4(index, d1, d2, d3, d4) (index % d4)
#define idx4_3(index, d1, d2, d3, d4) ((index / d4) % d3)
#define idx4_2(index, d1, d2, d3, d4) ((index / d4 / d3) % d2)
#define idx4_1(index, d1, d2, d3, d4) ((index / d4 / d3 / d2) % d1)

template <typename T>
__device__ bool GT_E(T a, T b) {
  return (a > b) || Eigen::numext::abs(a - b) < 1e-4;
}

template <typename T>
__device__ bool LT_E(T a, T b) {
  return (a < b) || Eigen::numext::abs(a - b) < 1e-4;
}

template <typename T>
__device__ bool GT(T a, T b) {
  return (a - b) > 1e-4;
}

template <typename T>
__device__ T max(T a, T b) {
  return a > b ? a : b;
}

template <typename T>
__device__ T min(T a, T b) {
  return a < b ? a : b;
}

/*
* check if (x, y) is in the boundary of roi
*/
template <typename T>
__device__ bool in_quad(T x, T y, T roi_x[], T roi_y[]) {
  for (int i = 0; i < 4; i++) {
    T start_w = roi_x[i];
    T start_h = roi_y[i];
    T end_w = roi_x[(i + 1) % 4];
    T end_h = roi_y[(i + 1) % 4];
    if (fabs(start_h - end_h) < 1e-4) {
      if (fabs(y - start_h) < 1e-4 && fabs(y - end_h) < 1e-4 &&
          GT_E<T>(x, min<T>(start_w, end_w)) &&
          LT_E<T>(x, max<T>(start_w, end_w))) {
        return true;
      }
    } else {
      T intersec_x =
          (y - start_h) * (end_w - start_w) / (end_h - start_h) + start_w;
      if (fabs(intersec_x - x) < 1e-4 && GT_E(y, min<T>(start_h, end_h)) &&
          LT_E<T>(y, max<T>(start_h, end_h))) {
        return true;
      }
    }
  }

  int n_cross = 0;
  for (int i = 0; i < 4; i++) {
    T start_w = roi_x[i];
    T start_h = roi_y[i];
    T end_w = roi_x[(i + 1) % 4];
    T end_h = roi_y[(i + 1) % 4];
    if (fabs(start_h - end_h) < 1e-4) {
      continue;
    }
    if (LT_E<T>(y, min<T>(start_h, end_h)) ||
        GT<T>(y, max<T>(start_h, end_h))) {
      continue;
    }
    T intersec_x =
        (y - start_h) * (end_w - start_w) / (end_h - start_h) + start_w;
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
 * Perform bilinear interpolation in the input feature map.
 */
template <typename T>
__device__ void bilinear_interpolate(const T* in_data, const int channels,
                                     const int width, const int height,
                                     int in_n, int in_c, T in_w, T in_h, T* val,
                                     int out_idx, int* out2in_idx,
                                     T* out2in_w) {
  // Deal with cases that source coords are out of feature map boundary
  if (GT_E<T>(-0.5, in_w) || GT_E<T>(in_w, width - 0.5) ||
      GT_E<T>(-0.5, in_h) || GT_E<T>(in_h, height - 0.5)) {
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

  int base_idx = (in_n * channels + in_c) * height * width;
  out2in_idx[out_idx * 4] = base_idx + in_h_floor * width + in_w_floor;
  out2in_idx[out_idx * 4 + 1] = base_idx + in_h_ceil * width + in_w_floor;
  out2in_idx[out_idx * 4 + 2] = base_idx + in_h_ceil * width + in_w_ceil;
  out2in_idx[out_idx * 4 + 3] = base_idx + in_h_floor * width + in_w_ceil;
  out2in_w[out_idx * 4] = w1;
  out2in_w[out_idx * 4 + 1] = w2;
  out2in_w[out_idx * 4 + 2] = w3;
  out2in_w[out_idx * 4 + 3] = w4;
}

/**
 * Get the source coordinates in the input feature map.
 *
 * (u, v, w)^matrix = T * (out_w, out_h, 1)^matrix
 *
 * in_w = u / w
 * in_h = v / w
 *
 */
template <typename T>
__device__ void get_source_coords(T matrix[], int out_w, int out_h, T* in_w,
                                  T* in_h) {
  T u = matrix[0] * out_w + matrix[1] * out_h + matrix[2];
  T v = matrix[3] * out_w + matrix[4] * out_h + matrix[5];
  T w = matrix[6] * out_w + matrix[7] * out_h + matrix[8];

  in_w[0] = u / w;
  in_h[0] = v / w;
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
 *
 */
template <typename T>
__device__ void get_transform_matrix(const int transformed_width,
                                     const int transformed_height, T roi_x[],
                                     T roi_y[], T matrix[]) {
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
  int normalized_height = max(2, transformed_height);
  int normalized_width =
      round(estimated_width * (normalized_height - 1) / estimated_height) + 1;
  normalized_width = max(2, min(normalized_width, transformed_width));

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

template <typename T>
__global__ void RoiTransformKernel(const float* input_data,
                                   const float* rois_data,
                                   const int* roi2image_data, int num_rois,
                                   int in_height, int in_width, int channels,
                                   int transformed_height,
                                   int transformed_width, float spatial_scale,
                                   T* output_data, int* out2in_idx, T* out2in_w,
                                   int* mask, T* transform_matrix) {
  int output_size =
      num_rois * transformed_height * transformed_width * channels;
  CUDA_KERNEL_LOOP(index, output_size) {
    // (n, c, out_h, out_w) is an element in the transformed output
    int out_w = idx4_4(index, num_rois, channels, transformed_height,
                       transformed_width);
    int out_h = idx4_3(index, num_rois, channels, transformed_height,
                       transformed_width);
    int c = idx4_2(index, num_rois, channels, transformed_height,
                   transformed_width);
    int n = idx4_1(index, num_rois, channels, transformed_height,
                   transformed_width);

    auto bottom_rois = rois_data + n * 8;
    int roi_batch_ind = bottom_rois[0];
    T roi_x[4];
    T roi_y[4];
    for (int k = 0; k < 4; ++k) {
      roi_x[k] = bottom_rois[2 * k] * spatial_scale;
      roi_y[k] = bottom_rois[2 * k + 1] * spatial_scale;
    }

    // Get transform matrix
    T matrix[9];
    get_transform_matrix<T>(transformed_width, transformed_height, roi_x, roi_y,
                            matrix);
    for (int i = 0; i < 9; i++) {
      transform_matrix[n * 9 + i] = matrix[i];
    }
    // Get source coords
    T in_w;
    T in_h;
    get_source_coords<T>(matrix, out_w, out_h, &in_w, &in_h);

    if (in_quad<T>(in_w, in_h, roi_x, roi_y)) {
      if (GT_E<T>(-0.5, in_w) ||
          GT_E<T>(in_w, static_cast<T>(in_width - 0.5)) ||
          GT_E<T>(-0.5, in_h) ||
          GT_E<T>(in_h, static_cast<T>(in_height - 0.5))) {
        // Skip if source coords is not in input image
        output_data[index] = 0.0;
        mask[(n * transformed_height + out_h) * transformed_width + out_w] = 0;
      } else {
        // Perform bilinear interpolation
        int in_n = roi2image_data[n];
        bilinear_interpolate<T>(input_data, channels, in_width, in_height, in_n,
                                c, in_w, in_h, output_data + index, index,
                                out2in_idx, out2in_w);
        mask[(n * transformed_height + out_h) * transformed_width + out_w] = 1;
      }

    } else {
      // Skip if source coords is not in quad
      output_data[index] = 0.0;
      mask[(n * transformed_height + out_h) * transformed_width + out_w] = 0;
    }
  }
}

template <typename T>
class CUDAROIPerspectiveTransformOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* in = ctx.Input<framework::Tensor>("X");
    auto* rois = ctx.Input<framework::LoDTensor>("ROIs");
    auto* out = ctx.Output<framework::Tensor>("Out");
    auto* out2in_idx = ctx.Output<framework::Tensor>("Out2InIdx");
    auto* out2in_w = ctx.Output<framework::Tensor>("Out2InWeights");
    auto* mask = ctx.Output<framework::Tensor>("Mask");
    auto* out_transform_matrix =
        ctx.Output<framework::Tensor>("TransformMatrix");

    int* mask_data = mask->mutable_data<int>(ctx.GetPlace());
    int* out2in_idx_data =
        out2in_idx->mutable_data<int>({out->numel(), 4}, ctx.GetPlace());
    T* out2in_w_data =
        out2in_w->mutable_data<T>({out->numel(), 4}, ctx.GetPlace());

    phi::funcs::SetConstant<platform::CUDADeviceContext, int> init;
    init(ctx.cuda_device_context(), out2in_idx, static_cast<int>(-1));

    auto transformed_height = ctx.Attr<int>("transformed_height");
    auto transformed_width = ctx.Attr<int>("transformed_width");
    auto spatial_scale = ctx.Attr<float>("spatial_scale");

    auto in_dims = in->dims();
    int batch_size = in_dims[0];
    int channels = in_dims[1];
    int in_height = in_dims[2];
    int in_width = in_dims[3];
    int rois_num = rois->dims()[0];

    const T* input_data = in->data<T>();
    T* output_data = out->mutable_data<T>(ctx.GetPlace());
    const T* rois_data = rois->data<T>();

    framework::Tensor roi2image;
    framework::Tensor roi2image_dev;
    roi2image.Resize({rois_num});
    int* roi2image_data = roi2image.mutable_data<int>(platform::CPUPlace());
    auto lod = rois->lod().back();
    for (size_t i = 0; i < lod.size() - 1; ++i) {
      for (size_t j = lod[i]; j < lod[i + 1]; ++j) {
        roi2image_data[j] = i;
      }
    }
    paddle::framework::TensorCopySync(roi2image, ctx.GetPlace(),
                                      &roi2image_dev);

    int out_size = rois_num * transformed_height * transformed_width * channels;
    auto stream = ctx.cuda_device_context().stream();
    int block = 512;
    int grid = (out_size + block - 1) / block;

    // Get transform matrix
    T* matrix =
        out_transform_matrix->mutable_data<T>({rois_num, 9}, ctx.GetPlace());

    RoiTransformKernel<T><<<grid, block, 0, stream>>>(
        input_data, rois_data, roi2image_dev.data<int>(), rois_num, in_height,
        in_width, channels, transformed_height, transformed_width,
        spatial_scale, output_data, out2in_idx_data, out2in_w_data, mask_data,
        matrix);
  }
};

template <typename T>
__device__ T get_feature_gradient(T xs, T ys, int w, int h, const int width,
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

  if (GT_E<T>(xs_floor, width - 1)) {
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
__global__ void RoiTransformGradKernel(int out_size, const int* out2in_idx_data,
                                       const T* out2in_w_data,
                                       const T* out_grad_data,
                                       T* in_grad_data) {
  CUDA_KERNEL_LOOP(index, out_size * 4) {
    int in_idx = out2in_idx_data[index];
    if (in_idx >= 0) {
      int out_idx = index / 4;
      atomicAdd(in_grad_data + in_idx,
                out_grad_data[out_idx] * out2in_w_data[index]);
    }
  }
}

template <typename T>
class CUDAROIPerspectiveTransformGradOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* out2in_idx = ctx.Input<framework::LoDTensor>("Out2InIdx");
    auto* out2in_w = ctx.Input<framework::LoDTensor>("Out2InWeights");
    auto* out_grad =
        ctx.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto* in_grad = ctx.Output<framework::Tensor>(framework::GradVarName("X"));

    T* in_grad_data = in_grad->mutable_data<T>(ctx.GetPlace());

    phi::funcs::SetConstant<platform::CUDADeviceContext, T> set_zero;
    set_zero(ctx.cuda_device_context(), in_grad, static_cast<T>(0));

    const T* out_grad_data = out_grad->data<T>();
    const int* out2in_idx_data = out2in_idx->data<int>();
    const T* out2in_w_data = out2in_w->data<T>();

    int out_size = out_grad->numel();
    auto stream = ctx.cuda_device_context().stream();
    int block = 512;
    int grid = (out_size * 4 + block - 1) / block;

    RoiTransformGradKernel<T><<<grid, block, 0, stream>>>(
        out_size, out2in_idx_data, out2in_w_data, out_grad_data, in_grad_data);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(roi_perspective_transform,
                        ops::CUDAROIPerspectiveTransformOpKernel<float>);
REGISTER_OP_CUDA_KERNEL(roi_perspective_transform_grad,
                        ops::CUDAROIPerspectiveTransformGradOpKernel<float>);
