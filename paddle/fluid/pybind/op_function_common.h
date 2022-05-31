// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#if defined(_MSC_VER)
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#endif

#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>

#include <memory>
#include <string>
#include <vector>

#include "paddle/fluid/framework/attribute.h"
#include "paddle/fluid/framework/op_info.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/imperative/tracer.h"
#include "paddle/fluid/imperative/type_defs.h"
#include "paddle/fluid/pybind/imperative.h"

namespace py = pybind11;
namespace paddle {
namespace pybind {

bool PyObject_CheckBool(PyObject** obj);

bool PyObject_CheckLongOrToLong(PyObject** obj);

bool PyObject_CheckFloatOrToFloat(PyObject** obj);

bool PyObject_CheckString(PyObject* obj);

bool CastPyArg2Boolean(PyObject* obj, const std::string& op_type,
                       ssize_t arg_pos);
int CastPyArg2Int(PyObject* obj, const std::string& op_type, ssize_t arg_pos);
int64_t CastPyArg2Long(PyObject* obj, const std::string& op_type,
                       ssize_t arg_pos);
float CastPyArg2Float(PyObject* obj, const std::string& op_type,
                      ssize_t arg_pos);
double CastPyArg2Double(PyObject* obj, const std::string& op_type,
                        ssize_t arg_pos);
std::string CastPyArg2String(PyObject* obj, const std::string& op_type,
                             ssize_t arg_pos);
std::vector<bool> CastPyArg2Booleans(PyObject* obj, const std::string& op_type,
                                     ssize_t arg_pos);
std::vector<int> CastPyArg2Ints(PyObject* obj, const std::string& op_type,
                                ssize_t arg_pos);
std::vector<int64_t> CastPyArg2Longs(PyObject* obj, const std::string& op_type,
                                     ssize_t arg_pos);
std::vector<float> CastPyArg2Floats(PyObject* obj, const std::string& op_type,
                                    ssize_t arg_pos);
std::vector<double> CastPyArg2Float64s(PyObject* obj,
                                       const std::string& op_type,
                                       ssize_t arg_pos);
std::vector<std::string> CastPyArg2Strings(PyObject* obj,
                                           const std::string& op_type,
                                           ssize_t arg_pos);

void CastPyArg2AttrBoolean(PyObject* obj,
                           paddle::framework::AttributeMap& attrs,  // NOLINT
                           const std::string& key, const std::string& op_type,
                           ssize_t arg_pos);

void CastPyArg2AttrInt(PyObject* obj,
                       paddle::framework::AttributeMap& attrs,  // NOLINT
                       const std::string& key, const std::string& op_type,
                       ssize_t arg_pos);

void CastPyArg2AttrLong(PyObject* obj,
                        paddle::framework::AttributeMap& attrs,  // NOLINT
                        const std::string& key, const std::string& op_type,
                        ssize_t arg_pos);

void CastPyArg2AttrFloat(PyObject* obj,
                         paddle::framework::AttributeMap& attrs,  // NOLINT
                         const std::string& key, const std::string& op_type,
                         ssize_t arg_pos);

void CastPyArg2AttrString(PyObject* obj,
                          paddle::framework::AttributeMap& attrs,  // NOLINT
                          const std::string& key, const std::string& op_type,
                          ssize_t arg_pos);

void CastPyArg2AttrBooleans(PyObject* obj,
                            paddle::framework::AttributeMap& attrs,  // NOLINT
                            const std::string& key, const std::string& op_type,
                            ssize_t arg_pos);

void CastPyArg2AttrInts(PyObject* obj,
                        paddle::framework::AttributeMap& attrs,  // NOLINT
                        const std::string& key, const std::string& op_type,
                        ssize_t arg_pos);

void CastPyArg2AttrLongs(PyObject* obj,
                         paddle::framework::AttributeMap& attrs,  // NOLINT
                         const std::string& key, const std::string& op_type,
                         ssize_t arg_pos);

void CastPyArg2AttrFloats(PyObject* obj,
                          paddle::framework::AttributeMap& attrs,  // NOLINT
                          const std::string& key, const std::string& op_type,
                          ssize_t arg_pos);

void CastPyArg2AttrFloat64s(PyObject* obj,
                            paddle::framework::AttributeMap& attrs,  // NOLINT
                            const std::string& key, const std::string& op_type,
                            ssize_t arg_pos);

void CastPyArg2AttrStrings(PyObject* obj,
                           paddle::framework::AttributeMap& attrs,  // NOLINT
                           const std::string& key, const std::string& op_type,
                           ssize_t arg_pos);

void CastPyArg2AttrBlock(PyObject* obj,
                         paddle::framework::AttributeMap& attrs,  // NOLINT
                         const std::string& key, const std::string& op_type,
                         ssize_t arg_pos);

void ConstructAttrMapFromPyArgs(
    const std::string& op_type, PyObject* args, ssize_t attr_start,
    ssize_t attr_end,
    paddle::framework::AttributeMap& attrs);  // NOLINT

std::shared_ptr<imperative::VarBase> GetVarBaseFromArgs(
    const std::string& op_type, const std::string& arg_name, PyObject* args,
    ssize_t arg_idx, bool dispensable = false);

std::vector<std::shared_ptr<imperative::VarBase>> GetVarBaseListFromArgs(
    const std::string& op_type, const std::string& arg_name, PyObject* args,
    ssize_t arg_idx, bool dispensable = false);

unsigned long GetUnsignedLongFromArgs(  // NOLINT
    const std::string& op_type, const std::string& arg_name, PyObject* args,
    ssize_t arg_idx, bool dispensable = false);

void InitOpsAttrTypeMap();

ssize_t GetIdxFromCoreOpsInfoMap(
    const std::unordered_map<std::string, std::vector<std::string>>&
        core_ops_info_map,
    const std::string& op_type, const std::string& name);

}  // namespace pybind
}  // namespace paddle
