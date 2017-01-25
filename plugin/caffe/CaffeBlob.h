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

#include "CaffeUtil.h"
#include "paddle/parameter/Argument.h"

#include <caffe/blob.hpp>
#include <caffe/layer.hpp>

namespace paddle {

std::vector<int> layerConfigToBlobShape(const int batch,
                                        const LayerConfig& preConfig);
std::vector<int> argShapeToVector(const Argument& arg);

void setBlob(MemoryTypes memType,
             ::caffe::Blob<real>* blob,
             real* d,
             bool useGpu);

void argToBlob(MemoryTypes memType,
               const Argument& arg,
               ::caffe::Blob<real>* blob,
               bool useGpu);

void blobToArg(MemoryTypes memType,
               ::caffe::Blob<real>* blob,
               Argument& arg,
               bool useGpu);

void copyBlobToParameter(MemoryTypes memType,
                         ::caffe::Blob<real>* blob,
                         ParameterPtr para,
                         bool useGpu);

void parameterToBlob(MemoryTypes memType,
                     ParameterPtr para,
                     ::caffe::Blob<real>* blob,
                     const std::vector<int>& shape,
                     bool useGpu);

}  // namespace paddle
