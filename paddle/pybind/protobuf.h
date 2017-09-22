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

#include <Python.h>
#include <fstream>
#include <vector>
#include "paddle/framework/op_registry.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

namespace py = pybind11;

namespace paddle {
namespace framework {

template <typename T>
inline std::vector<T> RepeatedToVector(
    const google::protobuf::RepeatedField<T>& repeated_field) {
  std::vector<T> ret;
  ret.reserve(repeated_field.size());
  std::copy(
      repeated_field.begin(), repeated_field.end(), std::back_inserter(ret));
  return ret;
}

template <typename T, typename RepeatedField>
inline void VectorToRepeated(const std::vector<T>& vec,
                             RepeatedField* repeated_field) {
  repeated_field->Reserve(vec.size());
  for (auto& elem : vec) {
    *repeated_field->Add() = elem;
  }
}

void bind_program_desc(py::module& m);
void bind_block_desc(py::module& m);
void bind_var_dses(py::module& m);
void bind_op_desc(py::module& m);
}  // namespace framework
}  // namespace paddle
