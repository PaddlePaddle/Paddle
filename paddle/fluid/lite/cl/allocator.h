// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once
#include <memory>
#include "paddle/fluid/lite/cl/cl2_header.h"
#include "paddle/fluid/lite/cl/cl_context.h"
#include "paddle/fluid/lite/cl/helper.h"
#include "paddle/fluid/lite/core/compatible_tensor.h"

namespace paddle {
namespace lite {

cl_channel_type DataTypeToCLChannelType(const PrecisionType t) {
  switch (t) {
    case PRECISION(kFloat):
      return CL_FLOAT;
    case PRECISION(kInt8):
      return CL_SIGNED_INT8;
    default:
      LOG(FATAL) << "Opencl Image doesn't support the data type: "
                 << PrecisionToStr(t);
      return 0;
  }
}

/*
 * The OpenclAllocator helps to manage OpenCL memory.
 */
class OpenclAllocator {
 public:
  void* New(size_t nbytes) const {
    if (nbytes == 0) {
      return nullptr;
    }

    cl_int error;
    cl::Buffer* buffer = new cl::Buffer(
        *opencl_context_->context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
        nbytes, nullptr, &error);
    if (error != CL_SUCCESS) {
      LOG(WARNING) << "Allocate OpenCL Buffer with " << nbytes
                   << " bytes failed: " << OpenclErrorToStr(error);
      delete buffer;
      return nullptr;
    } else {
      return buffer;
    }
  }

  void* Delete(void* buf) const {
    VLOG(3) << "Free OpenCL buffer";
    if (buf != nullptr) {
      cl::Buffer* cl_buffer = static_cast<cl::Buffer*>(buf);
      delete cl_buffer;
    }
  }

  void* NewImage(lite::DDim shape, PrecisionType dtype) {
    CHECK_EQ(shape.size(), 2UL) << "Image shape's size must equal 2";
    VLOG(3) << "Allocate OpenCL image: " << shape[0] << ", " << shape[1];

    cl::ImageFormat img_format(CL_RGBA, DataTypeToCLChannelType(dtype));

    cl_int error;
    cl::Image2D* cl_image = new cl::Image2D(
        *opencl_context_->context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
        img_format, shape[0], shape[1], 0, nullptr, &error);
    if (error != CL_SUCCESS) {
      LOG(WARNING) << "Allocate OpenCL image with shape: [" << shape[0] << ", "
                   << shape[1] << "] failed because of "
                   << OpenclErrorToStr(error);
      // Many users have doubts at CL_INVALID_IMAGE_SIZE, add some tips.
      if (error == CL_INVALID_IMAGE_SIZE) {
        auto max_2d_size = opencl_context_->GetMaxImage2DSize();
        LOG(WARNING) << "The allowable OpenCL image size is: " << max_2d_size[0]
                     << "x" << max_2d_size[1];
      }
      delete cl_image;
      return nullptr;
    } else {
      return cl_image;
    }
  }

  void* DeleteImage(void* buf) {
    VLOG(3) << "Free OpenCL image";
    if (buf) {
      auto* cl_image = static_cast<cl::Image2D*>(buf);
      delete cl_image;
    }
  }

 private:
  std::shared_ptr<OpenclContext> opencl_context_;
};

}  // namespace lite
}  // namespace paddle
