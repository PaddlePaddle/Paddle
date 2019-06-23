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
#include "paddle/fluid/lite/opencl/cl_context.h"
#include "paddle/fluid/lite/opencl/cl_engine.h"
#include "paddle/fluid/lite/opencl/cl_helper.h"
#include "paddle/fluid/lite/opencl/cl_image.h"
#include "paddle/fluid/lite/opencl/cl_tool.h"

namespace paddle {
namespace lite {
static void CopyImageData(const CLImage& cl_image, float* out) {
  int width = cl_image.image_dims()[0];
  int height = cl_image.image_dims()[1];

  half_t* image_data = new half_t[height * width * 4];
  cl::Image* image = cl_image.cl_image();
  const std::array<size_t, 3> origin{0, 0, 0};
  const std::array<size_t, 3> region{static_cast<size_t>(width),
                                     static_cast<size_t>(height), 1};
  cl_int err = CLEngine::Global()->command_queue().enqueueReadImage(
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

void elementwise_add(CLContext* context, float* in, const DDim& in_dim,
                     float* bias, const DDim& bias_dim, float* out,
                     const DDim& out_dim) {
  CLHelper helper(context);
  helper.AddKernel("elementwise_add", "elementwise_add_kernel.cl");
  auto kernel = helper.KernelAt(0);
  CLImage in_image;
  in_image.set_tensor_data(in, in_dim);
  in_image.InitNormalCLImage(helper.OpenCLContext());
  VLOG(3) << " --- Inpu image: " << in_image << " --- ";
  CLImage bias_image;
  bias_image.set_tensor_data(bias, bias_dim);
  bias_image.InitNormalCLImage(helper.OpenCLContext());
  VLOG(3) << " --- Bias image: " << bias_image << " --- ";
  CLImage out_image;
  out_image.InitEmptyImage(helper.OpenCLContext(), out_dim);
  cl_int status;
  status = kernel.setArg(0, *in_image.cl_image());
  CL_CHECK_ERRORS(status);
  status = kernel.setArg(1, *bias_image.cl_image());
  CL_CHECK_ERRORS(status);
  status = kernel.setArg(2, *out_image.cl_image());
  CL_CHECK_ERRORS(status);
  size_t width = in_image.ImageWidth();
  size_t height = in_image.ImageHeight();
  auto global_work_size = cl::NDRange{width, height};
  status = helper.OpenCLCommandQueue().enqueueNDRangeKernel(
      kernel, cl::NullRange, global_work_size, cl::NullRange, nullptr, nullptr);
  CL_CHECK_ERRORS(status);

  VLOG(3) << " --- Out image: " << out_image << " --- ";

  CopyImageData(out_image, out);
}

}  // namespace lite
}  // namespace paddle
