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

#include "paddle/fluid/lite/opencl/cl_helper.h"
#include <glog/logging.h>
#include <string>
#include <utility>
#include <vector>

namespace paddle {
namespace lite {

void CLHelper::set_context(CLContext *context) { context_ = context; }

void CLHelper::AddKernel(const std::string &kernel_name,
                         const std::string &file_name,
                         const std::string &options) {
  CHECK(context_ != nullptr) << "Please use set_context first!";
  VLOG(3) << " --- begin to add kernel ---";
  auto kernel = context_->GetKernel(kernel_name, file_name, options);
  kernels_.emplace_back(std::move(kernel));
  kernel_offset_[kernel_name] = kernels_.size() - 1;
  VLOG(3) << " --- end to add kernel --- ";
}

cl::Kernel &CLHelper::GetKernel(const int index) {
  VLOG(3) << " --- kernel count: " << kernels_.size() << " --- ";
  CHECK(static_cast<size_t>(index) < kernels_.size())
      << "The index must be less than the size of kernels.";
  CHECK(kernels_[index] != nullptr)
      << "The target kernel pointer cannot be null.";
  return *(kernels_[index]);
}

cl::CommandQueue &CLHelper::OpenCLCommandQueue() {
  CHECK(context_ != nullptr) << "Please use set_context first!";
  return context_->GetCommandQueue();
}

cl::Context &CLHelper::OpenCLContext() {
  CHECK(context_ != nullptr) << "Please use set_context first!";
  return context_->GetContext();
}

cl::NDRange CLHelper::DefaultWorkSize(const CLImage &image) {
  // n c h w
  auto image_dim = image.tensor_dims();
  if (image_dim.size() == 4) {
    auto n = image_dim[0];
    auto h = image_dim[2];
    auto w = image_dim[3];
    auto image_width = image.ImageWidth();
    auto work_size_0 = image_width / w;
    auto work_size_1 = w;
    auto work_size_2 = n * h;
    return cl::NDRange{static_cast<size_t>(work_size_0),
                       static_cast<size_t>(work_size_1),
                       static_cast<size_t>(work_size_2)};
  } else if (image_dim.size() == 2) {
    return cl::NDRange{static_cast<size_t>(1),
                       static_cast<size_t>(image.ImageWidth()),
                       static_cast<size_t>(image.ImageHeight())};
  } else if (image_dim.size() == 1) {
    return cl::NDRange{static_cast<size_t>(1),
                       static_cast<size_t>(image.ImageWidth()),
                       static_cast<size_t>(1)};
  } else if (image_dim.size() == 3) {
    auto c = image_dim[0];
    auto h = image_dim[1];
    auto w = image_dim[2];
    return cl::NDRange{static_cast<size_t>((c + 3) / 4), static_cast<size_t>(w),
                       static_cast<size_t>(h)};
  } else {
    LOG(FATAL) << "Not support this dimension, need to be implemented!";
    return cl::NDRange{};
  }
}

}  // namespace lite
}  // namespace paddle
