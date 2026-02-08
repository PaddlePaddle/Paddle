// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/fluid/framework/python_headers.h"

namespace paddle {
namespace operators {

size_t AppendPythonCallableObjectAndReturnId(const ::pybind11::object &py_obj);

}  // namespace operators
}  // namespace paddle
// C++示例：使用AppendPythonCallableObjectAndReturnId
// 功能：注册Python可调用对象并获取ID
#include <pybind11/pybind11.h>
namespace py = pybind11;

// 示例1：注册简单Python函数
py::object py_add_func = py::module::import("__main__").attr("add");
size_t add_func_id = paddle::operators::AppendPythonCallableObjectAndReturnId(py_add_func);
// 示例2：注册带参数的Python函数
py::object py_multiply_func = py::module::import("__main__").attr("multiply");
size_t multiply_func_id = paddle::operators::AppendPythonCallableObjectAndReturnId(py_multiply_func);
