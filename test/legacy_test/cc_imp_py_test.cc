//  Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <gtest/gtest.h>
#include <iostream>
#include "Python.h"

TEST(CC, IMPORT_PY) {
  // Initialize python environment
  Py_Initialize();
  ASSERT_TRUE(Py_IsInitialized());

  // 1. C/C++ Run Python simple string
  // ASSERT_FALSE(PyRun_SimpleString("import paddle"));
  // ASSERT_FALSE(PyRun_SimpleString("print(paddle.to_tensor(1))"));

  // 2. C/C++ Run Python function
  PyRun_SimpleString("import sys");
  PyRun_SimpleString("import os");
  PyRun_SimpleString("sys.path.append(os.getcwd())");
  // PyObject* pModule = PyImport_ImportModule("test_install_check");
  // ASSERT_TRUE(pModule != NULL);

  // PyObject* pTestInt = PyObject_GetAttrString(pModule, "TestInt");
  // ASSERT_TRUE(pTestInt != NULL);
  // PyObject* pArg1 = PyObject_CallObject(pTestInt, NULL);
  // ASSERT_TRUE(pArg1 != NULL);
  // int result;
  // ASSERT_TRUE(PyArg_Parse(pArg1, "i", &result));
  // ASSERT_EQ(result, 100);

  // PyObject* pTestString = PyObject_GetAttrString(pModule, "TestString");
  // ASSERT_TRUE(pTestString != NULL);
  // PyObject* pArg2 = PyObject_CallObject(pTestString, NULL);
  // ASSERT_TRUE(pArg2 != NULL);
  // char* cwd;
  // ASSERT_TRUE(PyArg_Parse(pArg2, "s", &cwd));

  // 3. C/C++ Run Python file
  // std::string file_name(cwd);
  // file_name.append("/test_install_check.py");
  // PyObject* obj = Py_BuildValue("s", file_name.c_str());
  // FILE* fp = _Py_fopen_obj(obj, "r+");
  // ASSERT_TRUE(fp != NULL);
  // ASSERT_FALSE(PyRun_SimpleFile(fp, file_name.c_str()));

  // // Uninitialize python environment
  // Py_Finalize();
  // ASSERT_FALSE(Py_IsInitialized());
}
