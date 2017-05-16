/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

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
#include <functional>
#include "BufferArgs.h"
#include "FuncConfig.h"
#include "paddle/topology/AttributeMap.h"
#include "paddle/utils/Error.h"
namespace paddle {
namespace function {

//! Kernel Function type of Paddle.
//! Each layer will invoke this KernelType when forward/backward.
typedef std::function<Error(const BufferArgs& inputs,
                            const BufferArgs& outputs)>
    Function;

class FunctionList : public std::vector<Function> {
public:
  void add(const std::string& name,
           const function::Config& config,
           bool useGPU);
};

}  // namespace function
}  // namespace paddle
