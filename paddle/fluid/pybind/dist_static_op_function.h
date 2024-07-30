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

static PyObject *static_api_local_tensors_from_dist(PyObject *self,
                                                    PyObject *args,
                                                    PyObject *kwargs) {
  try {
    VLOG(6) << "Add local_tensors_from_dist op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // input dist tensor
    PyObject *input_obj = PyTuple_GET_ITEM(args, 0);
    auto input = CastPyArg2Value(input_obj, "local_tensors_from_dist", 0);

    // local process_mesh list
    PyObject *local_mesh_list_obj = PyTuple_GET_ITEM(args, 1);
    std::vector<phi::distributed::ProcessMesh> local_mesh_list =
        CastPyArg2VectorOfProcessMesh(local_mesh_list_obj, 1);

    // local placements
    PyObject *local_placements_obj = PyTuple_GET_ITEM(args, 2);
    auto local_placements =
        CastPyArg2VectorOfPlacement(local_placements_obj, 2);

    // global process_mesh
    PyObject *global_mesh_obj = PyTuple_GET_ITEM(args, 3);
    auto global_mesh = CastPyArg2ProcessMesh(global_mesh_obj, 3);

    // global placements
    PyObject *global_placements_obj = PyTuple_GET_ITEM(args, 4);
    auto global_placements =
        CastPyArg2VectorOfPlacement(global_placements_obj, 4);

    int64_t ndim = GetValueDims(input).size();
    auto local_res = CvtPlacements(local_placements, ndim);
    auto global_res = CvtPlacements(global_placements, ndim);

    // Call ir static api
    auto static_api_out =
        paddle::dialect::local_tensors_from_dist(input,
                                                 local_mesh_list,
                                                 std::get<0>(local_res),
                                                 std::get<1>(local_res),
                                                 global_mesh,
                                                 std::get<0>(global_res),
                                                 std::get<1>(global_res));

    VLOG(6) << "End of adding local_tensors_from_dist op into program";
    return ToPyObject(static_api_out);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject *static_api_dtensor_from_local_list(PyObject *self,
                                                    PyObject *args,
                                                    PyObject *kwargs) {
  try {
    VLOG(6) << "Add dtensor_from_local_list op into program";
    VLOG(8) << "args count: " << (PyTuple_Size(args) / 2);

    // input local tensor list
    PyObject *input_obj = PyTuple_GET_ITEM(args, 0);
    auto input =
        CastPyArg2VectorOfValue(input_obj, "dtensor_from_local_list", 0);

    // local process_mesh list
    PyObject *local_mesh_list_obj = PyTuple_GET_ITEM(args, 1);
    std::vector<phi::distributed::ProcessMesh> local_mesh_list =
        CastPyArg2VectorOfProcessMesh(local_mesh_list_obj, 1);

    // local placements
    PyObject *local_placements_obj = PyTuple_GET_ITEM(args, 2);
    auto local_placements =
        CastPyArg2VectorOfPlacement(local_placements_obj, 2);

    // global process_mesh
    PyObject *global_mesh_obj = PyTuple_GET_ITEM(args, 3);
    auto global_mesh = CastPyArg2ProcessMesh(global_mesh_obj, 3);

    // global placements
    PyObject *global_placements_obj = PyTuple_GET_ITEM(args, 4);
    auto global_placements =
        CastPyArg2VectorOfPlacement(global_placements_obj, 4);

    // global shape
    PyObject *global_shape_obj = PyTuple_GET_ITEM(args, 5);
    auto global_shape = CastPyArg2VectorOfInt64(global_shape_obj, 5);

    int64_t ndim = GetValueDims(input[0]).size();
    auto local_res = CvtPlacements(local_placements, ndim);
    auto global_res = CvtPlacements(global_placements, ndim);

    // Call ir static api
    auto static_api_out =
        paddle::dialect::dist_tensor_from_locals(input,
                                                 local_mesh_list,
                                                 std::get<0>(local_res),
                                                 std::get<1>(local_res),
                                                 global_mesh,
                                                 std::get<0>(global_res),
                                                 std::get<1>(global_res),
                                                 global_shape);

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
    {"local_tensors_from_dist",
     (PyCFunction)(void (*)(void))static_api_local_tensors_from_dist,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for local_tensors_from_dist."},
    {"dtensor_from_local_list",
     (PyCFunction)(void (*)(void))static_api_dtensor_from_local_list,
     METH_VARARGS | METH_KEYWORDS,
     "C++ interface function for dtensor_from_local_list."},
    {nullptr, nullptr, 0, nullptr}};

}  // namespace pybind

}  // namespace paddle
