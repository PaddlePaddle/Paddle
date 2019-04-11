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

#include <Python.h>
#include <string>
#include <vector>
#include "paddle/fluid/imperative/layer.h"
#include "paddle/fluid/imperative/nccl_context.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

namespace paddle {
namespace pybind {

class Layer : public imperative::Layer {
 public:
  using imperative::Layer::Layer;  // Inherit constructors

  std::vector<imperative::VarBase> Forward(
      const std::vector<imperative::VarBase>& inputs) override {
    PYBIND11_OVERLOAD(std::vector<imperative::VarBase>, Layer, Forward,
                      inputs);  // NOLINT
  }
};

class PYBIND11_HIDDEN PyOpBase : public imperative::OpBase {
 public:
  using imperative::OpBase::OpBase;  // Inherit constructors

  PyOpBase(const std::string& name) : OpBase(name) {}
};

class PyVarBase : public imperative::VarBase {
 public:
  using imperative::VarBase::VarBase;  // Inherit constructors
};

void BindImperative(pybind11::module* m);

}  // namespace pybind
}  // namespace paddle
