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

#pragma once

#include <iostream>
#include <memory>
#include <vector>
#include "paddle/fluid/lite/core/compatible_tensor.h"
#include "paddle/fluid/lite/opencl/cl_image_converter.h"
#include "paddle/fluid/lite/opencl/cl_include.h"

namespace paddle {
namespace lite {

class CLImage {
  // For debug
  friend std::ostream& operator<<(std::ostream& os, const CLImage& image);

 public:
  CLImage() = default;
  /*
   * Will not hold input tensor data, memcpy in this method.
   * */
  void set_tensor_data(const float* tensor_data, const DDim& dim);

  bool IsInit() { return initialized_; }
  /*
   * Need call set_tensor_data first.
   * Folder when one dim or two dim.
   * */
  void InitCLImage(const cl::Context& context);

  void InitNormalCLImage(const cl::Context& context);

  void InitNImage(const cl::Context& context);

  void InitDWImage(const cl::Context& context);

  void InitEmptyImage(const cl::Context& context, const DDim& dim);

  void InitEmptyWithImageDim(const cl::Context& context,
                             const DDim& image_dims);

  cl::Image* cl_image() const { return cl_image_.get(); }

  const DDim& image_dims() const { return image_dims_; }

  inline size_t ImageWidth() const { return image_dims_[0]; }

  inline size_t ImageHeight() const { return image_dims_[1]; }

  const DDim& tensor_dims() const { return tensor_dims_; }

  /*with_da
   * Resize original tensor dim.
   * */
  inline CLImage& Resize(const DDim& dims) {
    tensor_dims_ = dims;
    return *this;
  }

  template <typename T>
  T* data() const {
    CHECK(!initialized_) << "CL image has initialized, tensor data has been "
                            "deleted, can't use tensor data!";
    return reinterpret_cast<T*>(tensor_data_);
  }

  /*
   *  Numel of tensor dim
   * */
  inline int64_t numel() const {
#ifdef LITE_WITH_LIGHT_WEIGHT_FRAMEWORK
    return tensor_dims_.product();
#else
    return tensor_dims_.production();
#endif
  }

  /*
   *  Original tensor dim
   * */

  cl::UserEvent& cl_event() const { return *cl_event_; }

  CLImageConverterBase* image_converter() const {
    return image_converter_.get();
  }

 private:
  void InitCLImage(const cl::Context& context, CLImageConverterBase* converter);

  void InitCLImage(const cl::Context& context, int width, int height,
                   void* data);

  bool initialized_ = false;
  std::unique_ptr<cl::Image2D> cl_image_{nullptr};
  std::unique_ptr<cl::UserEvent> cl_event_{nullptr};
  DDim tensor_dims_;
  DDim image_dims_;
  std::unique_ptr<float> tensor_data_{nullptr};
  std::unique_ptr<CLImageConverterBase> image_converter_{nullptr};
};

}  // namespace lite
}  // namespace paddle
