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

#include "paddle/framework/ddim.h"

namespace paddle {
namespace framework {

class InferShapeContextBase;

using ShapeInferenceFn =
    std::function<void(const framework::InferShapeContextBase& ctx)>;

class InferShapeContextBase {
 public:
  virtual ~InferShapeContextBase() {}
  virtual framework::DDim get_input_dim(const std::string& name) const = 0;
  virtual void set_input_dim(const std::string& name,
                             const framework::DDim& dim) const = 0;
  virtual framework::DDim get_output_dim(const std::string& name) const = 0;
  virtual void set_output_dim(const std::string& name,
                              const DDim& dim) const = 0;
  virtual AttrReader attrs() const = 0;

 protected:
  virtual framework::DDim get_dim(const std::string& name) const = 0;
  virtual void set_dim(const std::string& name,
                       const framework::DDim& dim) const = 0;
};

inline void NonFn(const framework::InferShapeContextBase& ctx){};

}  // namespace framework
}  // namespace paddle
