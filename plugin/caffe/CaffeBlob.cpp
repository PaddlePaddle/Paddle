/* Copyright (c) 2016 Baidu, Inc. All Rights Reserve.

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

#include <paddle/parameter/Argument.h>

#include <caffe/layer.hpp>
#include <caffe/blob.hpp>

namespace paddle {

std::vector<int> ArgShape2Vector(const Argument& arg) {
  std::vector<int> shape;
  shape.push_back(arg.getBatchSize());
  int frameHeight = arg.getFrameHeight();
  int frameWidth = arg.getFrameWidth();
  int dim = 0;
  if (arg.value) dim = arg.value->getWidth();
  if (arg.grad) dim = arg.grad->getWidth();
  CHECK(dim);
  // Paddle only support 4 dimension at most.
  // s1 means channel number for convolution layer.
  // s1 means hidden dimension for other layers.
  int s1 = 0;
  if (frameHeight && frameWidth) {
    s1 = dim / frameHeight / frameWidth;
    CHECK(s1);
    CHECK_EQ(dim, s1 * frameHeight * frameWidth);
  }
  shape.push_back(s1);
  if (frameHeight) shape.push_back(frameHeight);
  if (frameWidth) shape.push_back(frameWidth);
  return shape;
}

void SetDataToBlob(const Argument& arg,
                   ::caffe::Blob<real>* blob,
                   bool useGpu) {
  std::vector<int> shape = ArgShape2Vector(arg);
  blob->Reshape(shape);
  if (useGpu) {
    blob->set_gpu_data(arg.value->getData());
  } else {
    blob->set_cpu_data(arg.value->getData());
  }
}

}  // namespace paddle
