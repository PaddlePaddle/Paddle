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

#include <vector>
#include "paddle/fluid/memory/memory.h"
#include "paddle/fluid/operators/bezier_align_op.h"
#include "paddle/fluid/platform/cuda_primitives.h"
#include "paddle/fluid/platform/gpu_launch_config.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;

static constexpr int kNumCUDAThreads = 512;
static constexpr int kNumMaxinumNumBlocks = 4096;

static inline int NumBlocks(const int N) {
  return std::min((N + kNumCUDAThreads - 1) / kNumCUDAThreads,
                  kNumMaxinumNumBlocks);
}

template <typename T>
__device__ T bezier_curve_cu(const T p0, const T p1, const T p2, const T p3,
                             const T u) {
  return ((1. - u) * (1. - u) * (1. - u) * p0 +
          3. * u * (1. - u) * (1. - u) * p1 + 3. * u * u * (1. - u) * p2 +
          u * u * u * p3);
}

template <typename T>
__device__ T bilinear_interpolate(const T* bottom_data, const int height,
                                  const int width, T y, T x,
                                  const int index /* index for debug only*/) {
  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > height || x < -1.0 || x > width) {
    // empty
    return 0;
  }

  if (y <= 0) y = 0;
  if (x <= 0) x = 0;

  int y_low = static_cast<int>(y);
  int x_low = static_cast<int>(x;) int y_high;
  int x_high;

  if (y_low >= height - 1) {
    y_high = y_low = height - 1;
    y = (T)y_low;
  } else {
    y_high = y_low + 1;
  }

  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = (T)x_low;
  } else {
    x_high = x_low + 1;
  }

  T ly = y - y_low;
  T lx = x - x_low;
  T hy = 1. - ly, hx = 1. - lx;
  // do bilinear interpolation
  T v1 = bottom_data[y_low * width + x_low];
  T v2 = bottom_data[y_low * width + x_high];
  T v3 = bottom_data[y_high * width + x_low];
  T v4 = bottom_data[y_high * width + x_high];
  T w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

  T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

  return val;
}

template <typename T>
__global__ void GPUBezierAlignPoolForward(
    const int nthreads, const T* input_data, const T* input_rois,
    const float spatial_scale, const int input_channels, const int height,
    const int width, const int output_channels, const int pooled_height,
    const int pooled_width, const int* rois_batch_id_data, T* output_data,
    const bool aligned, const int sampling_ratio) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int offset = blockDim.x * gridDim.x;

  for (size_t i = index; i < nthreads; i += offset) {
    // The output is in order (n, c, ph, pw)
    int pw = i % pooled_width;
    int ph = (i / pooled_width) % pooled_height;
    int c = (i / pooled_width / pooled_height) % output_channels;
    int n = i / pooled_width / pooled_height / output_channels;

    // set roi_batch_id
    int roi_batch_id = rois_batch_id_data[n];
    // [start, end) interval for spatial sampling, beziers have size Nx(8*2) =
    // Nx16
    const T* offset_bottom_rois = input_rois + n * 16;

    // Do not use rounding; this implementation detail is critical
    T offset = aligned ? (T)0.5 : (T)0.0;

    // avoid this by using parallel annotation, for good
    T p0_x = offset_bottom_rois[0] * spatial_scale;
    T p0_y = offset_bottom_rois[1] * spatial_scale;
    T p1_x = offset_bottom_rois[2] * spatial_scale;
    T p1_y = offset_bottom_rois[3] * spatial_scale;
    T p2_x = offset_bottom_rois[4] * spatial_scale;
    T p2_y = offset_bottom_rois[5] * spatial_scale;
    T p3_x = offset_bottom_rois[6] * spatial_scale;
    T p3_y = offset_bottom_rois[7] * spatial_scale;
    T p4_x = offset_bottom_rois[14] * spatial_scale;
    T p4_y = offset_bottom_rois[15] * spatial_scale;
    T p5_x = offset_bottom_rois[12] * spatial_scale;
    T p5_y = offset_bottom_rois[13] * spatial_scale;
    T p6_x = offset_bottom_rois[10] * spatial_scale;
    T p6_y = offset_bottom_rois[11] * spatial_scale;
    T p7_x = offset_bottom_rois[8] * spatial_scale;
    T p7_y = offset_bottom_rois[9] * spatial_scale;

    // compute the coords
    const T u = pw / static_cast<T>(pooled_width);
    const T v = ph / static_cast<T>(pooled_height);
    const T x0 = bezier_curve_cu(p0_x, p1_x, p2_x, p3_x, u);
    const T y0 = bezier_curve_cu(p0_y, p1_y, p2_y, p3_y, u);
    const T x1 = bezier_curve_cu(p4_x, p5_x, p6_x, p7_x, u);
    const T y1 = bezier_curve_cu(p4_y, p5_y, p6_y, p7_y, u);
    const T x_center = x1 * v + x0 * (1. - v) - offset;
    const T y_center = y1 * v + y0 * (1. - v) - offset;

    T roi_width = max(abs(p0_x - p3_x), abs(p4_x - p7_x));
    T roi_height = max(abs(p0_y - p3_y), abs(p4_y - p7_y));
    if (!aligned) {  // for backward-compatibility only
      roi_width = max(roi_width, (T)1.);
      roi_height = max(roi_height, (T)1.);
    }
    T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
    T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

    const T* offset_input_data =
        input_data + (roi_batch_id * input_channels + c) * height * width;

    int roi_bin_grid_h = (sampling_ratio > 0)
                             ? sampling_ratio
                             : ceil(roi_height / pooled_height);  // e.g., = 2
    int roi_bin_grid_w =
        (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / pooled_width);

    // We do average (integral) pooling inside a bin
    // When the grid is empty, output zeros == 0/1, instead of NaN.
    const T count = max(roi_bin_grid_h * roi_bin_grid_w, 1);  // e.g. = 4

    T output_val = 0.;
    for (int iy = 0; iy < roi_bin_grid_h; iy++) {
      // e.g., iy = 0, 1
      const T y = y_center - (T)0.5 * bin_size_h +
                  static_cast<T>(iy + .5f) * bin_size_h /
                      static_cast<T>(roi_bin_grid_h);  // e.g., 0.5, 1.5
      for (int ix = 0; ix < roi_bin_grid_w; ix++) {
        const T x = x_center - (T)0.5 * bin_size_w +
                    static_cast<T>(ix + .5f) * bin_size_w /
                        static_cast<T>(roi_bin_grid_w);

        T val =
            bilinear_interpolate(offset_input_data, height, width, y, x, index);
        output_val += val;
      }
    }
    output_val /= count;

    output_data[index] = output_val;
  }  // end for
}

template <typename T>
__device__ void bilinear_interpolate_gradient_cu(
    const int height, const int width, T y, T x, T& w1, T& w2, T& w3, T& w4,
    int& x_low, int& x_high, int& y_low, int& y_high, const int index) {
  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > height || x < -1.0 || x > width) {
    // empty
    w1 = w2 = w3 = w4 = 0.;
    x_low = x_high = y_low = y_high = -1;
    return;
  }

  if (y <= 0) y = 0;
  if (x <= 0) x = 0;

  y_low = static_cast<int> y;
  x_low = static_cast<int> x;

  if (y_low >= height - 1) {
    y_high = y_low = height - 1;
    y = (T)y_low;
  } else {
    y_high = y_low + 1;
  }

  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = (T)x_low;
  } else {
    x_high = x_low + 1;
  }

  T ly = y - y_low;
  T lx = x - x_low;
  T hy = 1. - ly, hx = 1. - lx;

  // reference in forward
  // T v1 = bottom_data[y_low * width + x_low];
  // T v2 = bottom_data[y_low * width + x_high];
  // T v3 = bottom_data[y_high * width + x_low];
  // T v4 = bottom_data[y_high * width + x_high];
  // T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

  w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

  return;
}

template <typename T>
__global__ void GPUBezierAlignPoolBackward(
    const int nthreads, const T* input_rois, const T* output_grad_data,
    const float spatial_scale, const int input_channels, const int height,
    const int width, const int output_channels, const int pooled_height,
    const int pooled_width, const int* rois_batch_id_data, T* input_grad_data,
    const bool aligned, const int sampling_ratio) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int offset = blockDim.x * gridDim.x;
  for (int i = index; i < nthreads; i += offset) {
    // The output is in order (n, c, ph, pw)
    int pw = i % pooled_width;
    int ph = (i / pooled_width) % pooled_height;
    int c = (i / pooled_width / pooled_height) % output_channels;
    int n = i / pooled_width / pooled_height / output_channels;

    // beziers have size Nx(8*2) = Nx16
    const T* offset_bottom_rois = input_rois + n * 16;
    // set roi_batch_id
    int roi_batch_id = rois_batch_id_data[n];

    // Do not use rounding; this implementation detail is critical
    T offset = aligned ? (T)0.5 : (T)0.0;
    T p0_x = offset_bottom_rois[0] * spatial_scale;
    T p0_y = offset_bottom_rois[1] * spatial_scale;
    T p1_x = offset_bottom_rois[2] * spatial_scale;
    T p1_y = offset_bottom_rois[3] * spatial_scale;
    T p2_x = offset_bottom_rois[4] * spatial_scale;
    T p2_y = offset_bottom_rois[5] * spatial_scale;
    T p3_x = offset_bottom_rois[6] * spatial_scale;
    T p3_y = offset_bottom_rois[7] * spatial_scale;
    T p4_x = offset_bottom_rois[14] * spatial_scale;
    T p4_y = offset_bottom_rois[15] * spatial_scale;
    T p5_x = offset_bottom_rois[12] * spatial_scale;
    T p5_y = offset_bottom_rois[13] * spatial_scale;
    T p6_x = offset_bottom_rois[10] * spatial_scale;
    T p6_y = offset_bottom_rois[11] * spatial_scale;
    T p7_x = offset_bottom_rois[8] * spatial_scale;
    T p7_y = offset_bottom_rois[9] * spatial_scale;

    // compute the coords
    const T u = pw / static_cast<T>(pooled_width);
    const T v = ph / static_cast<T>(pooled_height);
    const T x0 = bezier_curve_cu(p0_x, p1_x, p2_x, p3_x, u);
    const T y0 = bezier_curve_cu(p0_y, p1_y, p2_y, p3_y, u);
    const T x1 = bezier_curve_cu(p4_x, p5_x, p6_x, p7_x, u);
    const T y1 = bezier_curve_cu(p4_y, p5_y, p6_y, p7_y, u);
    const T x_center = x1 * v + x0 * (1. - v) - offset;
    const T y_center = y1 * v + y0 * (1. - v) - offset;

    T roi_width = max(abs(p0_x - p3_x), abs(p4_x - p7_x));
    T roi_height = max(abs(p0_y - p3_y), abs(p4_y - p7_y));
    if (!aligned) {  // for backward-compatibility only
      roi_width = max(roi_width, (T)1.);
      roi_height = max(roi_height, (T)1.);
    }
    T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
    T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

    T* offset_bottom_diff =
        input_grad_data + (roi_batch_id * input_channels + c) * height * width;

    int top_offset = (n * output_channels + c) * pooled_height * pooled_width;
    const T* offset_top_diff = output_grad_data + top_offset;
    const T top_diff_this_bin = offset_top_diff[ph * pooled_width + pw];

    // We use roi_bin_grid to sample the grid and mimic integral
    int roi_bin_grid_h = (sampling_ratio > 0)
                             ? sampling_ratio
                             : ceil(roi_height / pooled_height);  // e.g., = 2
    int roi_bin_grid_w =
        (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / pooled_width);

    // We do average (integral) pooling inside a bin
    const T count = roi_bin_grid_h * roi_bin_grid_w;  // e.g. = 4

    for (int iy = 0; iy < roi_bin_grid_h; iy++) {  // e.g., iy = 0, 1
      const T y = y_center - (T)0.5 * bin_size_h +
                  static_cast<T>(iy + .5f) * bin_size_h /
                      static_cast<T>(roi_bin_grid_h);  // e.g., 0.5, 1.5
      for (int ix = 0; ix < roi_bin_grid_w; ix++) {
        const T x = x_center - (T)0.5 * bin_size_w +
                    static_cast<T>(ix + .5f) * bin_size_w /
                        static_cast<T>(roi_bin_grid_w);

        T w1, w2, w3, w4;
        int x_low, x_high, y_low, y_high;

        bilinear_interpolate_gradient_cu(height, width, y, x, w1, w2, w3, w4,
                                         x_low, x_high, y_low, y_high, index);

        T g1 = top_diff_this_bin * w1 / count;
        T g2 = top_diff_this_bin * w2 / count;
        T g3 = top_diff_this_bin * w3 / count;
        T g4 = top_diff_this_bin * w4 / count;

        if (x_low >= 0 && x_high >= 0 && y_low >= 0 && y_high >= 0) {
          platform::CudaAtomicAdd(offset_bottom_diff + y_low * width + x_low,
                                  static_cast<T>(g1));
          platform::CudaAtomicAdd(offset_bottom_diff + y_low * width + x_high,
                                  static_cast<T>(g2));
          platform::CudaAtomicAdd(offset_bottom_diff + y_high * width + x_low,
                                  static_cast<T>(g3));
          platform::CudaAtomicAdd(offset_bottom_diff + y_high * width + x_high,
                                  static_cast<T>(g4));
        }  // if
      }    // ix
    }      // iy
  }
}

template <typename Place, typename T>
class GPUBezierAlignOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* in = ctx.Input<framework::Tensor>("X");
    auto* rois = ctx.Input<framework::Tensor>("ROIs");
    auto* out = ctx.Output<framework::Tensor>("Out");

    auto pooled_height = ctx.Attr<int>("pooled_height");
    auto pooled_width = ctx.Attr<int>("pooled_width");
    auto spatial_scale = ctx.Attr<float>("spatial_scale");
    auto sampling_ratio = ctx.Attr<float>("sampling_ratio");
    auto aligned = ctx.Attr<bool>("aligned");

    auto in_dims = in->dims();
    int batch_size = in_dims[0];
    int input_channels = in_dims[1];
    auto output_channels = input_channels;
    int height = in_dims[2];
    int width = in_dims[3];

    int rois_num = rois->dims()[0];
    if (rois_num == 0) return;

    const T* input = in->data<T>();

    T* output = out->mutable_data<T>(ctx.GetPlace());

    // get roi batch id
    int rois_batch_size;
    framework::Tensor roi_batch_id_list;
    roi_batch_id_list.Resize({rois_num});

    const T* input_rois = rois->data<T>();

    auto cplace = platform::CPUPlace();
    int* roi_batch_id_data = roi_batch_id_list.mutable_data<int>(cplace);
    auto gplace = BOOST_GET_CONST(platform::CUDAPlace, ctx.GetPlace());
    if (ctx.HasInput("RoisNum")) {
      auto* rois_num_t = ctx.Input<Tensor>("RoisNum");
      int rois_batch_size = rois_num_t->numel();
      PADDLE_ENFORCE_EQ(
          rois_batch_size, batch_size,
          platform::errors::InvalidArgument(
              "The rois_batch_size and imgs "
              "batch_size must be the same. But received rois_batch_size = %d, "
              "batch_size = %d",
              rois_batch_size, batch_size));

      std::vector<int> rois_num_list(rois_batch_size);
      memory::Copy(cplace, rois_num_list.data(), gplace,
                   rois_num_t->data<int>(), sizeof(int) * rois_batch_size, 0);

      int start = 0;
      for (int n = 0; n < rois_batch_size; ++n) {
        for (int i = start; i < start + rois_num_list[n]; ++i) {
          roi_batch_id_data[i] = n;
        }
        start += rois_num_list[n];
      }
    }

    int output_size = rois_num * pooled_height * pooled_width * input_channels;
    int blocks = NumBlocks(output_size);
    int threads = kNumCUDAThreads;

    auto& dev_ctx = ctx.cuda_device_context();
    int bytes = roi_batch_id_list.numel() * sizeof(int);
    auto roi_ptr = memory::Alloc(dev_ctx, bytes);
    int* roi_id_data = reinterpret_cast<int*>(roi_ptr->ptr());
    memory::Copy(gplace, roi_id_data, cplace, roi_batch_id_data, bytes,
                 dev_ctx.stream());

    GPUBezierAlignPoolForward<T><<<blocks, threads, 0, dev_ctx.stream()>>>(
        output_size, in->data<T>(), rois->data<T>(), spatial_scale,
        input_channels, height, width, output_channels, pooled_height,
        pooled_width, roi_id_data, out->mutable_data<T>(ctx.GetPlace()),
        aligned, sampling_ratio);
  }
};

template <typename DeviceContext, typename T>
class GPUBezierAlignGradOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* in = ctx.Input<framework::Tensor>("X");
    auto* rois = ctx.Input<framework::Tensor>("ROIs");
    auto* out_grad =
        ctx.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto* in_grad = ctx.Output<framework::Tensor>(framework::GradVarName("X"));

    auto pooled_height = ctx.Attr<int>("pooled_height");
    auto pooled_width = ctx.Attr<int>("pooled_width");
    auto spatial_scale = ctx.Attr<float>("spatial_scale");
    auto sampling_ratio = ctx.Attr<float>("sampling_ratio");
    auto in_dims = in->dims();
    auto aligned = ctx.Attr<bool>("aligned");

    int channels = in_dims[1];
    int height = in_dims[2];
    int width = in_dims[3];
    int rois_num = rois->dims()[0];

    if (!in_grad) {
      return;
    }
    framework::Tensor roi_batch_id_list;
    roi_batch_id_list.Resize({rois_num});
    int* roi_batch_id_data =
        roi_batch_id_list.mutable_data<int>(ctx.GetPlace());

    auto cplace = platform::CPUPlace();
    auto& dev_ctx = ctx.cuda_device_context();
    auto gplace = BOOST_GET_CONST(platform::CUDAPlace, ctx.GetPlace());
    if (ctx.HasInput("RoisNum")) {
      auto* rois_num_t = ctx.Input<Tensor>("RoisNum");
      int rois_batch_size = rois_num_t->numel();
      std::vector<int> rois_num_list(rois_batch_size);
      memory::Copy(cplace, rois_num_list.data(), gplace,
                   rois_num_t->data<int>(), sizeof(int) * rois_batch_size, 0);
      int start = 0;
      for (int n = 0; n < rois_batch_size; ++n) {
        for (size_t i = start; i < start + rois_num_list[n]; ++i) {
          roi_batch_id_data[i] = n;
        }
        start += rois_num_list[n];
      }
    }

    auto roi_ptr =
        memory::Alloc(dev_ctx, roi_batch_id_list.numel() * sizeof(int));
    int* roi_id_data = reinterpret_cast<int*>(roi_ptr->ptr());
    int bytes = roi_batch_id_list.numel() * sizeof(int);
    memory::Copy(gplace, roi_id_data, cplace, roi_batch_id_data, bytes,
                 dev_ctx.stream());

    in_grad->mutable_data<T>(ctx.GetPlace());

    math::SetConstant<DeviceContext, T> set_zero;
    set_zero(dev_ctx, in_grad, static_cast<T>(0));

    int output_grad_size = out_grad->numel();
    if ((!out_grad->IsInitialized()) || (output_grad_size <= 0)) {
      return;
    }

    const T* input_rois = rois->data<T>();
    const T* output_grad = out_grad->data<T>();

    T* input_grad_data = in_grad->mutable_data<T>(ctx.GetPlace());

    int blocks = NumBlocks(output_grad_size);
    int threads = kNumCUDAThreads;

    if (output_grad_size > 0) {
      GPUBezierAlignPoolBackward<T><<<blocks, threads, 0, dev_ctx.stream()>>>(
          output_grad_size, input_rois, output_grad, spatial_scale, channels,
          height, width, channels, pooled_height, pooled_width,
          roi_batch_id_data, in_grad->mutable_data<T>(ctx.GetPlace()), aligned,
          sampling_ratio);
    }  // end if
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    bezier_align,
    ops::GPUBezierAlignOpKernel<paddle::platform::CUDADeviceContext, float>,
    ops::GPUBezierAlignOpKernel<paddle::platform::CUDADeviceContext, double>);
REGISTER_OP_CUDA_KERNEL(
    bezier_align_grad,
    ops::GPUBezierAlignGradOpKernel<paddle::platform::CUDADeviceContext, float>,
    ops::GPUBezierAlignGradOpKernel<paddle::platform::CUDADeviceContext,
                                    double>);
