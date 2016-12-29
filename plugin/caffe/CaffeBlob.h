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

std::vector<int> layerConfig2BlobShape(const batch,
                                       const LayerConfig& preConfig);
std::vector<int> argShape2Vector(const Argument& arg);

void setDataToBlob(Argument& arg, ::caffe::Blob<real>* blob);
void setGradToBlob(Argument& arg, ::caffe::Blob<real>* blob);

void setDataToArg(::caffe::Blob<real>* blob, Argument& arg);
void setGradToArg(::caffe::Blob<real>* blob, Argument& arg);

}  // namespace paddle
