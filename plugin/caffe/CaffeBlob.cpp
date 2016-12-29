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

#include <caffe/blob.hpp>
#include <caffe/layer.hpp>

namespace paddle {

// After reconstructing the data shape in paddle,
// this functions can be simplified.
std::vector<int> layerConfig2BlobShape(const batch, const LayerConfig& config) {
  std::vector<int> shape;
  shape.push_back(batch);
  int h = config.height();
  int w = config.width();
  int size = config.size();
  if (h && w) {
    int c = size / h / w;
    CHECK_EQ(c * h * w, size);
    shape.push(c);
    shape.push(h);
    shape.push(w);
  } else {
    shape.push(size);
  }
}

std::vector<int> argShape2Vector(const Argument& arg) {
  std::vector<int> shape;
  shape.push_back(arg.getBatchSize());
  int frameHeight = arg.getFrameHeight();
  int frameWidth = arg.getFrameWidth();
  int dim = 0;
  if (arg.value) {
    dim = arg.value->getWidth();
  } else if (arg.grad) {
    dim = arg.grad->getWidth();
  }
  CHECK(dim);
  // Paddle only support 4 dimension at most.
  // s1 means channel number for convolution layer,
  // means hidden dimension for other layers.
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

void setDataToBlob(const Argument& arg,
                   ::caffe::Blob<real>* blob,
                   bool useGpu) {
  std::vector<int> shape = argShape2Vector(arg);
  blob->Reshape(shape);
  if (useGpu) {
    blob->set_gpu_data(arg.grad->getData());
  } else {
    blob->set_cpu_data(arg.grad->getData());
  }
}

void setGradToBlob(const Argument& arg,
                   ::caffe::Blob<real>* blob,
                   bool useGpu) {
  std::vector<int> shape = argShape2Vector(arg);
  blob->Reshape(shape);
  if (useGpu) {
    blob->set_gpu_diff(arg.value->getData());
  } else {
    blob->set_cpu_diff(arg.value->getData());
  }
}

void setDataToArg(::caffe::Blob<real>* blob, const Argument& arg, bool useGpu) {
  auto& shape = blob->shape();
  int h = shape(0);
  int w = blob->count(1);
  if (shape.size() == 4) {
    arg.setFrameHeight(shape[3]);
    arg.setFrameWidth(shape[3]);
  }
  CHECK_LE(shape.size(), 4) << "Now only support 4-dimension at most";
  real* data = useGpu ? blob->gpu_data() : blob->cpu_data();
  arg.value = Matrix::create(data, h, w, false, useGpu);
}

void setGradToArg(const Argument& arg, ::caffe::Blob<real>* blob, bool useGpu) {
  auto& shape = blob->shape();
  int h = shape(0);
  int w = blob->count(1);
  if (shape.size() == 4) {
    arg.setFrameHeight(shape[3]);
    arg.setFrameWidth(shape[3]);
  }
  CHECK_LE(shape.size(), 4) << "Now only support 4-dimension at most";
  real* data = useGpu ? blob->gpu_diff() : blob->cpu_diff();
  arg.grad = Matrix::create(data, h, w, false, useGpu);
}

}  // namespace paddle
