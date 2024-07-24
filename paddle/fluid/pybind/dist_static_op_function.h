// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "paddle/fluid/pir/dialect/distributed/ir/dist_api.h"
#include "paddle/fluid/pir/dialect/operator/ir/api_builder.h"
#include "paddle/fluid/pybind/eager_utils.h"
#include "paddle/fluid/pybind/exception.h"
#include "paddle/fluid/pybind/pir.h"
#include "paddle/phi/core/enforce.h"

namespace paddle {

namespace pybind {

static PyObject *static_api_shard_tensor(PyObject *self,
                                         PyObject *args,
                                         PyObject *kwargs) {
  try {
    VLOG(6) << "Add shard_tensor op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *input_obj = PyTuple_GET_ITEM(args, 0);
    auto input = CastPyArg2Value(input_obj, "shard_tensor", 0);

    PyObject *process_mesh_obj = PyTuple_GET_ITEM(args, 1);
    auto process_mesh = CastPyArg2ProcessMesh(process_mesh_obj, 1);

    PyObject *placements_obj = PyTuple_GET_ITEM(args, 2);
    auto placements = CastPyArg2VectorOfPlacement(placements_obj, 2);

    int64_t ndim = GetValueDims(input).size();
    auto res = CvtPlacements(placements, ndim);

    // Call ir static api
    auto static_api_out = paddle::dialect::shard_tensor(
        input, process_mesh, std::get<0>(res), std::get<1>(res));

    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *static_api_reshard(PyObject *self,
                                    PyObject *args,
                                    PyObject *kwargs) {
  try {
    VLOG(6) << "Add reshard op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // Get Value from args
    PyObject *input_obj = PyTuple_GET_ITEM(args, 0);
    auto input = CastPyArg2Value(input_obj, "reshard", 0);

    PyObject *process_mesh_obj = PyTuple_GET_ITEM(args, 1);
    auto process_mesh = CastPyArg2ProcessMesh(process_mesh_obj, 1);

    PyObject *placements_obj = PyTuple_GET_ITEM(args, 2);
    auto placements = CastPyArg2VectorOfPlacement(placements_obj, 2);

    int64_t ndim = GetValueDims(input).size();
    auto res = CvtPlacements(placements, ndim);

    // Call ir static api
    auto static_api_out = paddle::dialect::reshard(
        input, process_mesh, std::get<0>(res), std::get<1>(res));

    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyMethodDef DistOpsAPI[] = {
    {"shard_tensor",
     (PyCFunction)(void (*)(void))static_api_shard_tensor,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for shard_tensor."},
    {"reshard",
     (PyCFunction)(void (*)(void))static_api_reshard,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for reshard."},
    {nullptr, nullptr, 0, nullptr}};

}  // namespace pybind

}  // namespace paddle
