// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include <Python.h>
#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/framework/scope_guard.h"
#include "paddle/fluid/operators/utils.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/compat/convert_utils.h"
#include "paddle/phi/core/dense_tensor.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

namespace py = pybind11;

namespace paddle {
namespace pybind {

static bool PyCheckTensor(PyObject* obj);
static Py_ssize_t GetSliceIndexFromPyObject(PyObject* obj);
// Slice related methods
static bool PyCheckInteger(PyObject* obj) {
#if PY_VERSION_HEX < 0x03000000
  return (PyLong_Check(obj) || PyInt_Check(obj)) && !PyBool_Check(obj);
#else
  return PyLong_Check(obj) && !PyBool_Check(obj);
#endif
}

static bool IsNumpyType(PyObject* obj) {
  // It is not a good way to judge the type of obj by its type'name. Maybe using
  // `PyArray_IsScalar` will be better. However, this interface cannot be used
  // by including pybind11, and it needs to compile with numpy.
  auto type_name = std::string(Py_TYPE(obj)->tp_name);
  return type_name == "numpy.int64" || type_name == "numpy.longlong" ||
         type_name == "numpy.int32" || type_name == "numpy.int16";
}

static Py_ssize_t GetSliceIndexFromTensor(const phi::DenseTensor& tensor) {
  if (tensor.numel() == 1) {
    if (framework::TransToProtoVarType(tensor.type()) ==
        framework::proto::VarType::INT32) {
      return static_cast<Py_ssize_t>(operators::GetValue<int32_t>(&tensor));
    } else if (framework::TransToProtoVarType(tensor.type()) ==
               framework::proto::VarType::INT64) {
      return static_cast<Py_ssize_t>(operators::GetValue<int64_t>(&tensor));
    } else {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Currently, the type of tensor in slice indices only allows "
          "int32 and int64, please check the type of index tensor."));
    }
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "Currently, tensor in slice indices only allows 1 element, "
        "but received %d.",
        tensor.numel()));
  }
}

// NOTE(zhiqiu): Revised version of PySlice_GetIndices. From:
// https://github.com/python/cpython/blob/8d21aa21f2cbc6d50aab3f420bb23be1d081dac4/Objects/sliceobject.c#L103
// Original PySlice_GetIndices return wrong result when
// slice_item contains long int, such as arr[:180L].
// NOT sure why this happens !!!
// Besides, PySlice_GetIndices cannot raise error when float in slice item.
// So, I make a revised version of PySlice_GetIndices, named to
// _PySlice_GetIndices. Try to use _PySlice_Unpack which is more robust than
// PySlice_GetIndices in the future.
static int _PySlice_GetIndices(PySliceObject* r, Py_ssize_t length,
                               Py_ssize_t* start, Py_ssize_t* stop,
                               Py_ssize_t* step) {
  /* XXX support long ints */
  if (r->step == Py_None) {
    *step = 1;
  } else {
    if (PyCheckInteger(r->step) || IsNumpyType(r->step)) {
      *step = PyLong_AsLong(r->step);
    } else if (PyCheckTensor(r->step)) {
      *step = GetSliceIndexFromPyObject(r->step);
    } else {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Currently, slice indices only allows None, integers, "
          "tensor(int) and numpy(int) in slice item, but received %s.",
          std::string(Py_TYPE(r->step)->tp_name)));
    }
  }
  if (r->start == Py_None) {
    *start = *step < 0 ? length - 1 : 0;
  } else {
    if (PyCheckInteger(r->start) || IsNumpyType(r->start)) {
      *start = PyLong_AsLong(r->start);
    } else if (PyCheckTensor(r->start)) {
      *start = GetSliceIndexFromPyObject(r->start);
    } else {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Currently, slice indices only allows None, integers, "
          "tensor(int) and numpy(int) in slice item, but received %s.",
          std::string(Py_TYPE(r->start)->tp_name)));
    }
    if (*start < 0) *start += length;
    *start = std::max(*start, static_cast<Py_ssize_t>(0));
  }
  if (r->stop == Py_None) {
    *stop = *step < 0 ? -1 : length;
  } else {
    if (PyCheckInteger(r->stop) || IsNumpyType(r->stop)) {
      *stop = PyLong_AsLong(r->stop);
    } else if (PyCheckTensor(r->stop)) {
      *stop = GetSliceIndexFromPyObject(r->stop);
    } else {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Currently, slice indices only allows None, integers, "
          "tensor(int) and numpy(int) in slice item, but received %s.",
          std::string(Py_TYPE(r->stop)->tp_name)));
    }
    if (0 < *step && *stop < 0) *stop += length;
    *stop = std::min(*stop, length);
  }
  if (*stop > length) return -1;
  if (*start >= length) return -1;
  if (*step == 0) return -1;
  return 0;
}

static void ParseIndexingSlice(
    framework::LoDTensor* tensor, PyObject* _index,
    std::vector<int>* slice_axes, std::vector<int>* slice_starts,
    std::vector<int>* slice_ends, std::vector<int>* slice_strides,
    std::vector<int>* decrease_axis, std::vector<int>* none_axes,
    std::vector<int>* infer_flags, std::vector<int>* list_select_idxs,
    bool* list_select_flag) {
  // We allow indexing by Integers, Slices, Ellipsis, None, tuples of those
  // types, and list of Bool and Integers.
  // wrap to tuple

  // NOTE(zhiqiu): PyTuple_Pack increases refcount.
  PyObject* index = !PyTuple_Check(_index) ? PyTuple_Pack(1, _index) : _index;
  DEFINE_PADDLE_SCOPE_GUARD([index, _index]() {
    if (!PyTuple_Check(_index)) {
      Py_DECREF(index);
      VLOG(4) << "Call Py_DECREF";
    }
  });
  PADDLE_ENFORCE_EQ(
      tensor->IsInitialized(), true,
      platform::errors::InvalidArgument("tensor has not been initialized"));
  const auto& shape = tensor->dims();
  const int rank = shape.size();
  const int size = PyTuple_GET_SIZE(index);

  // specified_dims is the number of dimensions which indexed by Interger,
  // Slices.
  int specified_dims = 0;
  int ell_count = 0;
  for (int dim = 0; dim < size; ++dim) {
    PyObject* slice_item = PyTuple_GetItem(index, dim);
    if (PyCheckInteger(slice_item) || PySlice_Check(slice_item)) {
      specified_dims++;
    } else if (slice_item == Py_Ellipsis) {
      ell_count++;
    }
  }

  PADDLE_ENFORCE_LE(ell_count, 1,
                    platform::errors::InvalidArgument(
                        "An index can only have a single ellipsis ('...')"));
  int none_count = 0;
  for (int i = 0, dim = 0; i < size; ++i) {
    PyObject* slice_item = PyTuple_GetItem(index, i);

    infer_flags->push_back(1);
    int dim_len = shape[dim];
    if (PyCheckInteger(slice_item) || IsNumpyType(slice_item)) {
      // integer, PyLong_AsLong supports both int and long
      int start = static_cast<int>(PyLong_AsLong(slice_item));
      auto s_t = start;
      start = start < 0 ? start + dim_len : start;

      PADDLE_ENFORCE(
          0 <= start && start < dim_len,
          platform::errors::OutOfRange("The starting index %d of slice is out "
                                       "of bounds in tensor %d-th axis, it "
                                       "shound be in the range of [%d, %d).",
                                       s_t, dim, -dim_len, dim_len));

      slice_axes->push_back(dim);
      slice_starts->push_back(start);
      slice_ends->push_back(start + 1);
      slice_strides->push_back(1);
      decrease_axis->push_back(dim);
      dim++;
    } else if (PySlice_Check(slice_item)) {
      // slice item
      Py_ssize_t start, end, step;
      PySliceObject* p = reinterpret_cast<PySliceObject*>(slice_item);
      _PySlice_GetIndices(p, dim_len, &start, &end, &step);

      // :: or : or 0:dim_len:1
      if (start == 0 && end == dim_len && step == 1) {
        dim++;
        continue;
      }
      slice_axes->push_back(dim);
      slice_starts->push_back(start);
      slice_ends->push_back(end);
      slice_strides->push_back(step);
      dim++;
    } else if (slice_item == Py_Ellipsis) {
      dim += rank - specified_dims;
    } else if (slice_item == Py_None) {
      none_axes->push_back(dim + none_count);
      none_count++;
    } else if (PyList_Check(slice_item)) {
      *list_select_flag = true;
      PADDLE_ENFORCE_EQ(
          size, 1,
          platform::errors::InvalidArgument(
              "When index contains a list, its length is excepted to 1, "
              "but received %d",
              size));
      bool all_bool = true;
      int list_size = PyList_GET_SIZE(slice_item);
      for (int j = 0; j < list_size; ++j) {
        PyObject* list_item = PyList_GetItem(slice_item, j);
        if (PyCheckInteger(list_item)) {
          all_bool = false;
        } else if (!PyBool_Check(list_item)) {
          PADDLE_THROW(platform::errors::InvalidArgument(
              "Only support int or bool in index list."));
        }
      }
      if (all_bool) {
        PADDLE_ENFORCE_EQ(
            list_size, shape[0],
            platform::errors::InvalidArgument(
                "The dimension of bool index doesn't match indexed array along "
                "dimension 0, the target dimension is %d, but received %d.",
                shape[0], list_size));

        for (int j = 0; j < list_size; ++j) {
          PyObject* list_item = PyList_GetItem(slice_item, j);
          if (list_item == Py_True) {
            list_select_idxs->push_back(j);
          }
        }
      } else {
        for (int j = 0; j < list_size; ++j) {
          PyObject* list_item = PyList_GetItem(slice_item, j);
          if (PyCheckInteger(list_item)) {
            list_select_idxs->push_back(
                static_cast<int>(PyLong_AsLong(list_item)));
          } else if (list_item == Py_True) {
            list_select_idxs->push_back(1);
          } else {
            list_select_idxs->push_back(0);
          }
        }
      }

    } else {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Currently, Tensor.__indices__() only allows indexing "
          "by Integers, Slices, Ellipsis, None, tuples of these types "
          "and list of Bool and Integers, but received "
          "%s in %dth slice item",
          std::string(Py_TYPE(slice_item)->tp_name), i + 1));
    }
  }

  // valid_index is the number of dimensions exclude None index
  const int valid_indexs = size - none_axes->size() - ell_count;
  PADDLE_ENFORCE_EQ(valid_indexs <= rank, true,
                    platform::errors::InvalidArgument(
                        "Too many indices (%d) for tensor of dimension %d.",
                        valid_indexs, rank));
}

}  // namespace pybind
}  // namespace paddle
