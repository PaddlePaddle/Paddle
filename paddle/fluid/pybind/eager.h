/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/eager/hooks.h"
#include "paddle/fluid/eager/pylayer/py_layer_node.h"
#include "paddle/phi/core/dense_tensor.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

namespace paddle {
namespace pybind {

typedef struct {
  PyObject_HEAD paddle::experimental::Tensor tensor;
  // Weak references
  PyObject* weakrefs;
} TensorObject;

typedef struct {
  PyObject_HEAD PyObject* container;
  bool container_be_packed;
  std::shared_ptr<egr::UnPackHookBase> unpack_hook;
  PyObject* non_differentiable;
  PyObject* not_inplace_tensors;
  bool materialize_grads;
  std::vector<bool> forward_input_tensor_is_duplicable;
  std::vector<bool> forward_output_tensor_is_duplicable;
  std::weak_ptr<egr::GradNodePyLayer> grad_node;
} PyLayerObject;

void BindEager(pybind11::module* m);
void BindEagerStringTensor(pybind11::module* module);
void BindFunctions(PyObject* module);
void BindEagerPyLayer(PyObject* module);
void BindEagerOpFunctions(pybind11::module* module);
void BindFinalStateEagerOpFunctions(pybind11::module* module);
}  // namespace pybind
}  // namespace paddle
