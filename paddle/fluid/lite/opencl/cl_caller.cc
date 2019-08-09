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

#include "paddle/fluid/lite/opencl/cl_caller.h"
#include <string>
#include "paddle/fluid/lite/core/compatible_tensor.h"
#include "paddle/fluid/lite/opencl/cl_engine.h"
#include "paddle/fluid/lite/opencl/cl_helper.h"
#include "paddle/fluid/lite/opencl/cl_image.h"
#include "paddle/fluid/lite/opencl/cl_tool.h"
#include "paddle/fluid/lite/utils/string.h"

namespace paddle {
namespace lite {
static void CopyImageData(CLHelper* helper, const CLImage& cl_image,
                          float* out) {
  int width = cl_image.image_dims()[0];
  int height = cl_image.image_dims()[1];

  float* image_data = new float[height * width * 4];
  cl::Image* image = cl_image.cl_image();
  const std::array<size_t, 3> origin{0, 0, 0};
  const std::array<size_t, 3> region{static_cast<size_t>(width),
                                     static_cast<size_t>(height), 1};
  cl_int err = helper->OpenCLCommandQueue().enqueueReadImage(
      *image, CL_TRUE, origin, region, 0, 0, image_data, nullptr, nullptr);
  CL_CHECK_ERRORS(err);

  auto* converter = cl_image.image_converter();
  converter->ImageToNCHW(image_data, out, cl_image.image_dims(),
                         cl_image.tensor_dims());

  delete[] image_data;
}

bool InitOpenCLEngine(std::string cl_path) {
  auto* engine = CLEngine::Global();
  engine->set_cl_path(cl_path);
  return engine->IsInitSuccess();
}

void elementwise_add(CLHelper* helper, const float* in, const DDim& in_dim,
                     const float* bias, const DDim& bias_dim, float* out,
                     const DDim& out_dim) {
  if (!(bias_dim.size() == 1 || bias_dim.size() == 4)) {
    LOG(FATAL) << "Error: bias dims is error";
    return;
  }
  auto kernel = bias_dim.size() == 1 ? helper->GetKernel("channel_add")
                                     : helper->GetKernel("elementwise_add");
  CLImage in_image;
  in_image.set_tensor_data(in, in_dim);
  in_image.InitNormalCLImage(helper->OpenCLContext());
  VLOG(3) << " --- Inpu image: " << in_image << " --- ";
  CLImage bias_image;
  bias_image.set_tensor_data(bias, bias_dim);
  bias_image.InitCLImage(helper->OpenCLContext());
  VLOG(3) << " --- Bias image: " << bias_image << " --- ";
  CLImage out_image;
  out_image.InitEmptyImage(helper->OpenCLContext(), out_dim);
  cl_int status;
  status = kernel.setArg(0, *in_image.cl_image());
  CL_CHECK_ERRORS(status);
  status = kernel.setArg(1, *bias_image.cl_image());
  CL_CHECK_ERRORS(status);
  status = kernel.setArg(2, *out_image.cl_image());
  CL_CHECK_ERRORS(status);

  if (bias_dim.size() == 1) {
    int tensor_w = in_dim[3];
    status = kernel.setArg(3, tensor_w);
    CL_CHECK_ERRORS(status);
  }
  size_t width = in_image.ImageWidth();
  size_t height = in_image.ImageHeight();
  auto global_work_size = cl::NDRange{width, height};
  status = helper->OpenCLCommandQueue().enqueueNDRangeKernel(
      kernel, cl::NullRange, global_work_size, cl::NullRange, nullptr, nullptr);
  CL_CHECK_ERRORS(status);

  status = helper->OpenCLCommandQueue().finish();
  CL_CHECK_ERRORS(status);
  VLOG(3) << " --- Out image: " << out_image << " --- ";
  CopyImageData(helper, out_image, out);
}

void pool(CLHelper* helper, const std::string pooling_type, const int pad_h,
          const int pad_w, const int stride_h, const int stride_w,
          const int ksize_h, const int ksize_w, const float* in,
          const DDim& in_dim, float* out, const DDim& out_dim) {
  auto kernel =
      helper->GetKernel(string_format("pool_%s", pooling_type.c_str()));
  CLImage in_image;
  in_image.set_tensor_data(in, in_dim);
  in_image.InitNormalCLImage(helper->OpenCLContext());
  VLOG(3) << " --- Inpu image: " << in_image << " --- ";
  CLImage out_image;
  out_image.InitEmptyImage(helper->OpenCLContext(), out_dim);
  auto global_work_size = helper->DefaultWorkSize(out_image);
  auto* in_converter =
      dynamic_cast<CLImageConverterNormal*>(in_image.image_converter());
  auto* out_converter =
      dynamic_cast<CLImageConverterNormal*>(out_image.image_converter());
  const int in_height = in_converter->HeightOfOneBlock();
  const int in_width = in_converter->WidthOfOneBlock();
  const int out_height = out_converter->HeightOfOneBlock();
  const int out_width = out_converter->WidthOfOneBlock();
  cl_int status;
  status = kernel.setArg(0, in_height);
  CL_CHECK_ERRORS(status);
  status = kernel.setArg(1, in_width);
  CL_CHECK_ERRORS(status);
  status = kernel.setArg(2, out_height);
  CL_CHECK_ERRORS(status);
  status = kernel.setArg(3, out_width);
  CL_CHECK_ERRORS(status);
  status = kernel.setArg(4, pad_h);
  CL_CHECK_ERRORS(status);
  status = kernel.setArg(5, pad_w);
  CL_CHECK_ERRORS(status);
  status = kernel.setArg(6, stride_h);
  CL_CHECK_ERRORS(status);
  status = kernel.setArg(7, stride_w);
  CL_CHECK_ERRORS(status);
  status = kernel.setArg(8, ksize_h);
  CL_CHECK_ERRORS(status);
  status = kernel.setArg(9, ksize_w);
  CL_CHECK_ERRORS(status);
  status = kernel.setArg(10, *in_image.cl_image());
  CL_CHECK_ERRORS(status);
  status = kernel.setArg(11, *out_image.cl_image());
  CL_CHECK_ERRORS(status);

  status = helper->OpenCLCommandQueue().enqueueNDRangeKernel(
      kernel, cl::NullRange, global_work_size, cl::NullRange, nullptr, nullptr);
  CL_CHECK_ERRORS(status);

  status = helper->OpenCLCommandQueue().finish();
  CL_CHECK_ERRORS(status);
  VLOG(3) << " --- Out image: " << out_image << " --- ";
  CopyImageData(helper, out_image, out);
}

}  // namespace lite
}  // namespace paddle
