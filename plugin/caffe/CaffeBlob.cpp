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

#include "CaffeBlob.h"
#include "paddle/parameter/Argument.h"

#include <caffe/blob.hpp>
#include <caffe/layer.hpp>

namespace paddle {

std::vector<int> layerConfigToBlobShape(const int batch,
                                        const LayerConfig& config) {
  std::vector<int> shape;
  shape.push_back(batch);
  int h = config.height();
  int w = config.width();
  int size = config.size();
  if (h > 1 || w > 1) {
    int c = size / h / w;
    CHECK_EQ(c * h * w, size);
    shape.push_back(c);
    shape.push_back(h);
    shape.push_back(w);
  } else {
    shape.push_back(size);
  }
  return shape;
}

std::vector<int> argShapeToVector(const Argument& arg) {
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
  int s1 = dim;
  if (frameHeight > 1 || frameWidth > 1) {
    s1 = dim / frameHeight / frameWidth;
    CHECK(s1);
    CHECK_EQ(dim, s1 * frameHeight * frameWidth);
  }
  shape.push_back(s1);
  if (frameHeight) shape.push_back(frameHeight);
  if (frameWidth) shape.push_back(frameWidth);
  return shape;
}

void setBlob(MemoryTypes memType,
             ::caffe::Blob<real>* blob,
             real* d,
             bool useGpu) {
  if (memType == VALUE) {
    if (useGpu) {
      blob->set_gpu_data(d);
    } else {
      blob->set_cpu_data(d);
    }
  } else {
    if (useGpu) {
      blob->set_gpu_diff(d);
    } else {
      blob->set_cpu_diff(d);
    }
  }
}

void argToBlob(MemoryTypes memType,
               const Argument& arg,
               ::caffe::Blob<real>* blob,
               bool useGpu) {
  std::vector<int> shape = argShapeToVector(arg);
  blob->Reshape(shape);
  auto& mat = memType == VALUE ? arg.value : arg.grad;
  CHECK(mat);
  setBlob(memType, blob, mat->getData(), useGpu);
}

void blobToArg(MemoryTypes memType,
               ::caffe::Blob<real>* blob,
               Argument& arg,
               bool useGpu) {
  auto& shape = blob->shape();
  int h = shape[0];
  int w = blob->count(1);
  if (shape.size() == 4) {
    arg.setFrameHeight(shape[2]);
    arg.setFrameWidth(shape[3]);
  }
  CHECK_LE(shape.size(), 4) << "Now only support 4-dimension at most";
  if (memType == VALUE) {
    real* data = useGpu ? blob->mutable_gpu_data() : blob->mutable_cpu_data();
    arg.value = Matrix::create(data, h, w, false, useGpu);
  } else {
    real* data = useGpu ? blob->mutable_gpu_diff() : blob->mutable_cpu_diff();
    arg.grad = Matrix::create(data, h, w, false, useGpu);
  }
}

void copyBlobToParameter(MemoryTypes memType,
                         ::caffe::Blob<real>* blob,
                         ParameterPtr para,
                         bool useGpu) {
  int size = blob->count();
  if (memType == VALUE) {
    real* d = useGpu ? blob->mutable_gpu_data() : blob->mutable_cpu_data();
    para->getBuf(PARAMETER_VALUE)->copyFrom(d, size);
  } else {
    real* d = useGpu ? blob->mutable_gpu_diff() : blob->mutable_cpu_diff();
    para->getBuf(PARAMETER_GRADIENT)->copyFrom(d, size);
  }
}

void parameterToBlob(MemoryTypes memType,
                     ParameterPtr para,
                     ::caffe::Blob<real>* blob,
                     const std::vector<int>& shape,
                     bool useGpu) {
  blob->Reshape(shape);
  auto& buf = memType == VALUE ? para->getBuf(PARAMETER_VALUE)
                               : para->getBuf(PARAMETER_GRADIENT);
  setBlob(memType, blob, buf->getData(), useGpu);
}

}  // namespace paddle
