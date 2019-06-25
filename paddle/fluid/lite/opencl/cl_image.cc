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

#include "paddle/fluid/lite/opencl/cl_image.h"
#include <glog/logging.h>
#include <array>
#include "paddle/fluid/lite/opencl/cl_engine.h"
#include "paddle/fluid/lite/opencl/cl_tool.h"

namespace paddle {
namespace lite {

std::ostream& operator<<(std::ostream& os, const CLImage& cl_image) {
  int width = cl_image.image_dims_[0];
  int height = cl_image.image_dims_[1];

  float* image_data = new float[height * width * 4];
  cl::Image* image = cl_image.cl_image();
  const std::array<size_t, 3> origin{0, 0, 0};
  const std::array<size_t, 3> region{static_cast<size_t>(width),
                                     static_cast<size_t>(height), 1};
  cl_int err = CLEngine::Global()->command_queue().enqueueReadImage(
      *image, CL_TRUE, origin, region, 0, 0, image_data, nullptr, nullptr);
  CL_CHECK_ERRORS(err);

  float* tensor_data = new float[cl_image.numel()];
  auto* converter = cl_image.image_converter();
  converter->ImageToNCHW(image_data, tensor_data, cl_image.image_dims_,
                         cl_image.tensor_dims_);
  int stride = cl_image.numel() / 20;
  stride = stride > 0 ? stride : 1;

  os << " dims: " << cl_image.tensor_dims_ << "\n";
  for (int i = 0; i < cl_image.numel(); i += stride) {
    os << tensor_data[i] << " ";
  }

  delete[] tensor_data;
  delete[] image_data;

  return os;
}

void CLImage::set_tensor_data(const float* tensor_data, const DDim& dim) {
#ifdef LITE_WITH_LIGHT_WEIGHT_FRAMEWORK
  auto numel = dim.product();
#else
  auto numel = dim.production();
#endif
  tensor_data_.reset(new float[numel]);
  memcpy(tensor_data_.get(), tensor_data, numel * sizeof(float));
  tensor_dims_ = dim;
}

void CLImage::InitCLImage(const cl::Context& context) {
  CHECK(tensor_data_ != nullptr) << " Please call "
                                    "set_tensohelper->DefaultWorkSize(out_"
                                    "image)r_data first!";
  image_converter_.reset(new CLImageConverterFolder);
  InitCLImage(context, image_converter_.get());
}

void CLImage::InitNormalCLImage(const cl::Context& context) {
  CHECK(tensor_data_ != nullptr) << " Please call set_tensor_data first!";
  image_converter_.reset(new CLImageConverterNormal);
  InitCLImage(context, image_converter_.get());
}

void CLImage::InitNImage(const cl::Context& context) {
  CHECK(tensor_data_ != nullptr) << " Please call set_tensor_data first!";
  CHECK(tensor_dims_.size() == 4) << " Tensor dim is not 4.";
  image_converter_.reset(new CLImageConverterNWBlock);
  InitCLImage(context, image_converter_.get());
}

void CLImage::InitDWImage(const cl::Context& context) {
  CHECK(tensor_data_ != nullptr) << " Please call set_tensor_data first!";
  CHECK(tensor_dims_.size() == 4) << " Tensor dim is not 4.";
  image_converter_.reset(new CLImageConverterDWBlock);
  InitCLImage(context, image_converter_.get());
}

void CLImage::InitEmptyImage(const cl::Context& context, const DDim& dim) {
  CHECK(tensor_data_ == nullptr)
      << " Empty image tensor data shouldn't have value";

  tensor_dims_ = dim;
  image_converter_.reset(new CLImageConverterNormal);

  VLOG(3) << " to get image dims ";
  image_dims_ = image_converter_->InitImageDimInfoWith(tensor_dims_);
  VLOG(3) << " end get image dims " << image_dims_;

  InitCLImage(context, image_dims_[0], image_dims_[1], nullptr);

  cl_event_ = CLEngine::Global()->CreateEvent(context);
  initialized_ = true;
  VLOG(3) << " end init cl image ";
}

void CLImage::InitEmptyWithImageDim(const cl::Context& context,
                                    const DDim& image_dims) {
  VLOG(3) << " to get image dims ";
  image_dims_ = image_dims;
  VLOG(3) << " end get image dims " << image_dims_;

  InitCLImage(context, image_dims_[0], image_dims_[1], nullptr);

  cl_event_ = CLEngine::Global()->CreateEvent(context);
  initialized_ = true;
  VLOG(3) << " end init cl image";
}

void CLImage::InitCLImage(const cl::Context& context,
                          CLImageConverterBase* converter) {
  CHECK(tensor_data_ != nullptr) << " Please call set_tensor_data first!";

  VLOG(3) << " begin init cl image ";
  image_dims_ = converter->InitImageDimInfoWith(tensor_dims_);

#ifdef LITE_WITH_LIGHT_WEIGHT_FRAMEWORK
  float* image_data = new float[image_dims_.product() * 4];
#else
  float* image_data = new float[image_dims_.production() * 4];
#endif

  VLOG(3) << " convert to image ";
  converter->NCHWToImage(tensor_data_.get(), image_data, tensor_dims_);
  VLOG(3) << " end convert to image ";

  InitCLImage(context, image_dims_[0], image_dims_[1], image_data);

  delete[] image_data;
  tensor_data_ = nullptr;
  cl_event_ = CLEngine::Global()->CreateEvent(context);
  initialized_ = true;
  VLOG(3) << " end init cl image ";
}

void CLImage::InitCLImage(const cl::Context& context, int width, int height,
                          void* data) {
  cl::ImageFormat img_format(CL_RGBA, CL_FLOAT);
  cl_int err;
  cl_image_.reset(new cl::Image2D(
      context, CL_MEM_READ_WRITE | (data ? CL_MEM_COPY_HOST_PTR : 0),
      img_format, width, height, 0, data, &err));
  CL_CHECK_ERRORS(err);
  CHECK(err == CL_SUCCESS) << " Create image 2d error.";
}

}  // namespace lite
}  // namespace paddle
