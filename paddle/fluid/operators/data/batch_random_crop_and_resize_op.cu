// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/operators/data/batch_random_crop_and_resize_op.h"
#include "paddle/fluid/operators/reader/lod_tensor_blocking_queue.h"
#include "paddle/fluid/platform/device/gpu/gpu_launch_config.h"
#include "paddle/fluid/platform/device/gpu/gpu_primitives.h"

namespace paddle {
namespace operators {
namespace data {

using DataLayout = framework::DataLayout;
using LoDTensorBlockingQueueHolder =
    operators::reader::LoDTensorBlockingQueueHolder;

template <typename T>
__global__ void KeNearestNeighborInterpFw(
    const T* in, const size_t in_img_h, const size_t in_img_w,
    const size_t input_h, const size_t input_w, T* out, const size_t out_img_h,
    const size_t out_img_w, const size_t output_h, const size_t output_w,
    const size_t num_channels, const float ratio_h, const float ratio_w,
    const size_t idx_h, const size_t idx_w, const bool align_corners,
    const DataLayout data_format) {
  int nthreads = output_h * output_w;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (; tid < nthreads; tid += stride) {
    // batch size
    int out_id_h = tid / output_w;
    // single image's index
    int out_id_w = tid % output_w;
    // input_w or output_w = c * h * w, img_size = h * w
    int in_img_size = input_w / num_channels;
    int out_img_size = output_w / num_channels;

    // get output c, h, w index
    int channel_id, out_img_idy, out_img_idx;
    if (data_format == DataLayout::kNCHW) {
      channel_id = out_id_w / out_img_size;
      out_img_idy = (out_id_w % out_img_size) / out_img_w;
      out_img_idx = tid % out_img_w;
    } else {
      out_img_idy = out_id_w / (out_img_w * num_channels);
      out_img_idx = out_id_w % (out_img_w * num_channels) / num_channels;
      channel_id = tid % num_channels;
    }

    // get input h index with offset
    int in_img_idy = (align_corners)
                         ? static_cast<int>(ratio_h * out_img_idy + 0.5)
                         : static_cast<int>(ratio_h * out_img_idy);
    in_img_idy += idx_h;
    // get input w index with offset
    int in_img_idx = (align_corners)
                         ? static_cast<int>(ratio_w * out_img_idx + 0.5)
                         : static_cast<int>(ratio_w * out_img_idx);
    in_img_idx += idx_w;

    if (data_format == DataLayout::kNCHW) {
      out[tid] = in[out_id_h * input_w + channel_id * in_img_size +
                    in_img_idy * in_img_w + in_img_idx];
    } else {
      out[tid] = in[out_id_h * input_w + in_img_idy * in_img_w * num_channels +
                    in_img_idx * num_channels + channel_id];
    }
  }
}

template <typename T>
__global__ void KeBilinearInterpFw(
    const T* in, const size_t in_img_h, const size_t in_img_w,
    const size_t input_h, const size_t input_w, T* out, const size_t out_img_h,
    const size_t out_img_w, const size_t output_h, const size_t output_w,
    const size_t num_channels, const float ratio_h, const float ratio_w,
    const size_t idx_h, const size_t idx_w, const bool align_corners,
    const int align_mode, const DataLayout data_format) {
  int nthreads = output_h * output_w;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  bool align_flag = (align_mode == 0 && !align_corners);
  for (; tid < nthreads; tid += stride) {
    // batch size
    int out_id_h = tid / output_w;
    // single image's index
    int out_id_w = tid % output_w;
    // input_w or output_w = c * h * w, img_size = h * w
    int in_img_size = input_w / num_channels;
    int out_img_size = output_w / num_channels;

    // get output c, h, w index
    int channel_id, out_img_idy, out_img_idx;
    if (data_format == DataLayout::kNCHW) {
      channel_id = out_id_w / out_img_size;
      out_img_idy = (out_id_w % out_img_size) / out_img_w;
      out_img_idx = tid % out_img_w;
    } else {
      out_img_idy = out_id_w / (out_img_w * num_channels);
      out_img_idx = out_id_w % (out_img_w * num_channels) / num_channels;
      channel_id = tid % num_channels;
    }

    // get input h index with offset
    int in_img_idy = align_flag
                         ? static_cast<int>(ratio_h * (out_img_idy + 0.5) - 0.5)
                         : static_cast<int>(ratio_h * out_img_idy);
    in_img_idy = (in_img_idy > 0) ? in_img_idy + idx_h : idx_h;
    int h_id = (in_img_idy < in_img_h + idx_h - 1) ? 1 : 0;
    T src_h = ratio_h * (out_img_idy + 0.5) - 0.5;
    src_h = (src_h > 0) ? src_h + idx_h : idx_h;
    T h1lambda = align_flag ? src_h - in_img_idy
                            : ratio_h * out_img_idy + idx_h - in_img_idy;
    T h2lambda = 1.f - h1lambda;

    // get input w index with offset
    int in_img_idx = align_flag
                         ? static_cast<int>(ratio_w * (out_img_idx + 0.5) - 0.5)
                         : static_cast<int>(ratio_w * out_img_idx);
    in_img_idx = (in_img_idx > 0) ? in_img_idx + idx_w : idx_w;
    int w_id = (in_img_idx < in_img_w + idx_w - 1) ? 1 : 0;
    T src_w = ratio_w * (out_img_idx + 0.5) - 0.5;
    src_w = (src_w > 0) ? src_w + idx_w : idx_w;
    T w1lambda = align_flag ? src_w - in_img_idx
                            : ratio_w * out_img_idx + idx_w - in_img_idx;
    T w2lambda = 1.f - w1lambda;

    if (data_format == DataLayout::kNCHW) {
      const T* in_pos = &in[out_id_h * input_w + channel_id * in_img_size +
                            in_img_idy * in_img_w + in_img_idx];

      // bilinear interpolation
      out[out_id_h * output_w + out_id_w] =
          h2lambda * (w2lambda * in_pos[0] + w1lambda * in_pos[w_id]) +
          h1lambda * (w2lambda * in_pos[h_id * in_img_w] +
                      w1lambda * in_pos[h_id * in_img_w + w_id]);
    } else {
      const T* in_pos =
          &in[out_id_h * input_w + in_img_idy * in_img_w * num_channels +
              in_img_idx * num_channels + channel_id];

      // bilinear interpolation
      out[out_id_h * output_w + out_id_w] =
          h2lambda *
              (w2lambda * in_pos[0] + w1lambda * in_pos[w_id * num_channels]) +
          h1lambda * (w2lambda * in_pos[h_id * in_img_w * num_channels] +
                      w1lambda * in_pos[h_id * in_img_w * num_channels +
                                        w_id * num_channels]);
    }
  }
}

template <typename T>
static void BatchRandomCropAndResizeFwd(
    const framework::ExecutionContext& ctx, const framework::Tensor& input,
    framework::Tensor* output, const std::vector<int64_t> out_size,
    const std::string interp_method, const bool align_corners,
    const int align_mode, const int img_h, const int img_w, const int c,
    const int idx_h, const int idx_w, const int crop_h, const int crop_w,
    const DataLayout data_format) {
  auto input_data = input.template data<T>();
  int out_h = static_cast<int>(out_size[0]);
  int out_w = static_cast<int>(out_size[1]);

  framework::DDim dim_out;
  if (data_format == DataLayout::kNCHW) {
    dim_out = {c, out_h, out_w};
  } else {
    dim_out = {out_h, out_w, c};
  }
  auto output_data = output->data<T>();

  if (img_h == crop_h && img_w == crop_w) {
    framework::TensorCopy(input, ctx.GetPlace(), output);
    return;
  }

  float ratio_h = 0.f;
  float ratio_w = 0.f;
  if (out_h > 1) {
    ratio_h = (align_corners) ? static_cast<float>(crop_h - 1) / (out_h - 1)
                              : static_cast<float>(crop_h) / out_h;
  }
  if (out_w > 1) {
    ratio_w = (align_corners) ? static_cast<float>(crop_w - 1) / (out_w - 1)
                              : static_cast<float>(crop_w) / out_w;
  }

  int in_chw = c * crop_h * crop_w;
  int out_chw = c * out_h * out_w;

  platform::GpuLaunchConfig config =
      platform::GetGpuLaunchConfig1D(ctx.cuda_device_context(), out_chw);

  if ("nearest" == interp_method) {
    KeNearestNeighborInterpFw<
        T><<<config.block_per_grid, config.thread_per_block, 0,
             ctx.cuda_device_context().stream()>>>(
        input_data, crop_h, crop_w, 1, in_chw, output_data, out_h, out_w, 1,
        out_chw, c, ratio_h, ratio_w, idx_h, idx_w, align_corners, data_format);
  } else if ("bilinear" == interp_method) {
    KeBilinearInterpFw<T><<<config.block_per_grid, config.thread_per_block, 0,
                            ctx.cuda_device_context().stream()>>>(
        input_data, crop_h, crop_w, 1, in_chw, output_data, out_h, out_w, 1,
        out_chw, c, ratio_h, ratio_w, idx_h, idx_w, align_corners, align_mode,
        data_format);
  }
}

template <typename T>
class BatchRandomCropAndResizeCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE_EQ(
        platform::is_gpu_place(ctx.GetPlace()), true,
        platform::errors::NotFound("This kernel only runs on GPU device."));
    // get input, output
    auto x = ctx.MultiInput<framework::Tensor>("X");
    PADDLE_ENFORCE_GT(x.size(), 0,
                      platform::errors::InvalidArgument(
                          "The size of X must be greater than 0."));
    auto* out = ctx.Output<framework::Tensor>("Out");

    auto aspect_ratio_min = ctx.Attr<float>("aspect_ratio_min");
    auto aspect_ratio_max = ctx.Attr<float>("aspect_ratio_max");
    AspectRatioRange aspect_ratio_range{aspect_ratio_min, aspect_ratio_max};

    auto area_min = ctx.Attr<float>("area_min");
    auto area_max = ctx.Attr<float>("area_max");
    AreaRange area_range{area_min, area_max};

    auto* generators = GeneratorManager::Instance()->GetGenerators(
        x.size(), x.size(), aspect_ratio_range, area_range);

    const std::vector<int64_t> size = ctx.Attr<std::vector<int64_t>>("size");

    // get data_format
    const std::string data_format_str = ctx.Attr<std::string>("data_format");
    const DataLayout data_format =
        framework::StringToDataLayout(data_format_str);
    // get interpolation method
    const std::string interp_method = ctx.Attr<std::string>("interp_method");
    bool align_corners = ctx.Attr<bool>("align_corners");
    int align_mode = ctx.Attr<int>("align_mode");

    auto* img = x.at(0);
    int64_t img_c =
        data_format == DataLayout::kNCHW ? img->dims()[0] : img->dims()[2];

    std::vector<int64_t> out_dim;
    if (data_format == DataLayout::kNCHW) {
      out_dim = {static_cast<int64_t>(x.size()), img_c, size[0], size[1]};
    } else {
      out_dim = {static_cast<int64_t>(x.size()), size[0], size[1], img_c};
    }
    out->Resize(phi::make_ddim(out_dim));
    out->mutable_data<T>(ctx.GetPlace());

    int img_h, img_w, idx_h, idx_w, crop_h, crop_w;
    for (int i = 0; i < x.size(); i++) {
      img = x.at(i);
      img_h =
          data_format == DataLayout::kNCHW ? img->dims()[1] : img->dims()[0];
      img_w =
          data_format == DataLayout::kNCHW ? img->dims()[2] : img->dims()[1];
      ROI roi;
      generators->at(i)->GenerateRandomROI(img_w, img_h, &roi);

      auto out_tensor = out->Slice(i, i + 1);
      BatchRandomCropAndResizeFwd<T>(ctx, *img, &out_tensor, size,
                                     interp_method, align_corners, align_mode,
                                     img_h, img_w, img_c, roi.y, roi.x, roi.h,
                                     roi.w, data_format);
    }
  }
};

}  // namespace data
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(batch_random_crop_and_resize,
                        ops::data::BatchRandomCropAndResizeCUDAKernel<uint8_t>,
                        ops::data::BatchRandomCropAndResizeCUDAKernel<float>,
                        ops::data::BatchRandomCropAndResizeCUDAKernel<double>);
