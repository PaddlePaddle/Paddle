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

#include "paddle/fluid/pybind/compatible.h"

#include <memory>
#include <string>

#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/pybind/pybind_variant_caster.h"

namespace py = pybind11;

using paddle::framework::compatible::OpAttrInfo;
using paddle::framework::compatible::OpAttrVariantT;
using paddle::framework::compatible::OpBugfixInfo;
using paddle::framework::compatible::OpCheckpoint;
using paddle::framework::compatible::OpInputOutputInfo;
using paddle::framework::compatible::OpUpdateBase;
using paddle::framework::compatible::OpUpdateInfo;
using paddle::framework::compatible::OpUpdateType;
using paddle::framework::compatible::OpVersion;
using paddle::framework::compatible::OpVersionDesc;

namespace paddle::pybind {

namespace {
using paddle::framework::compatible::PassVersionCheckerRegistrar;
void BindPassVersionChecker(py::module *m) {
  py::class_<PassVersionCheckerRegistrar>(*m, "PassVersionChecker")
      .def_static("IsCompatible", [](const std::string &name) -> bool {
        auto instance = PassVersionCheckerRegistrar::GetInstance();
        return instance.IsPassCompatible(name);
      });
}

void BindPassCompatible(py::module *m) { BindPassVersionChecker(m); }

void BindOpUpdateInfo(py::module *m) {
  py::class_<OpUpdateInfo>(*m, "OpUpdateInfo").def(py::init<>());
}

void BindOpAttrInfo(py::module *m) {
  py::class_<OpAttrInfo, OpUpdateInfo>(*m, "OpAttrInfo")
      .def(py::init<const std::string &,
                    const std::string &,
                    const OpAttrVariantT &>())
      .def(py::init<const OpAttrInfo &>())
      .def("name", &OpAttrInfo::name)
      .def("default_value", &OpAttrInfo::default_value)
      .def("remark", &OpAttrInfo::remark);
}

void BindOpInputOutputInfo(py::module *m) {
  py::class_<OpInputOutputInfo, OpUpdateInfo>(*m, "OpInputOutputInfo")
      .def(py::init<const std::string &, const std::string &>())
      .def(py::init<const OpInputOutputInfo &>())
      .def("name", &OpInputOutputInfo::name)
      .def("remark", &OpInputOutputInfo::remark);
}

void BindOpBugfixInfo(py::module *m) {
  py::class_<OpBugfixInfo, OpUpdateInfo>(*m, "OpBugfixInfo")
      .def(py::init<const std::string &>())
      .def(py::init<const OpBugfixInfo &>())
      .def("remark", &OpBugfixInfo::remark);
}

void BindOpCompatible(py::module *m) {
  BindOpUpdateInfo(m);
  BindOpAttrInfo(m);
  BindOpInputOutputInfo(m);
  BindOpBugfixInfo(m);
}

void BindOpUpdateType(py::module *m) {
  py::enum_<OpUpdateType>(*m, "OpUpdateType")
      .value("kInvalid", OpUpdateType::kInvalid)
      .value("kModifyAttr", OpUpdateType::kModifyAttr)
      .value("kNewAttr", OpUpdateType::kNewAttr)
      .value("kNewInput", OpUpdateType::kNewInput)
      .value("kNewOutput", OpUpdateType::kNewOutput)
      .value("kBugfixWithBehaviorChanged",
             OpUpdateType::kBugfixWithBehaviorChanged);
}

void BindOpUpdateBase(py::module *m) {
  py::class_<OpUpdateBase>(*m, "OpUpdateBase")
      .def("info", &OpUpdateBase::info, py::return_value_policy::reference)
      .def("type", &OpUpdateBase::type);
}

void BindOpVersionDesc(py::module *m) {
  py::class_<OpVersionDesc>(*m, "OpVersionDesc")
      // Pybind11 does not yet support the transfer of `const
      // std::vector<std::unique_ptr<T>>&` type objects.
      .def("infos", [](const OpVersionDesc &obj) {
        auto pylist = py::list();
        for (const auto &ptr : obj.infos()) {
          auto pyobj = py::cast(*ptr, py::return_value_policy::reference);
          pylist.append(pyobj);
        }
        return pylist;
      });
}

void BindOpCheckpoint(py::module *m) {
  py::class_<OpCheckpoint>(*m, "OpCheckpoint")
      .def("note", &OpCheckpoint::note, py::return_value_policy::reference)
      .def("version_desc",
           &OpCheckpoint::version_desc,
           py::return_value_policy::reference);
}

void BindOpVersion(py::module *m) {
  py::class_<OpVersion>(*m, "OpVersion")
      .def("version_id",
           &OpVersion::version_id,
           py::return_value_policy::reference)
      .def("checkpoints",
           &OpVersion::checkpoints,
           py::return_value_policy::reference);
  // At least pybind v2.3.0 is required because of bug #1603 of pybind11.
  m->def("get_op_version_map",
         &framework::compatible::get_op_version_map,
         py::return_value_policy::reference);
}

}  // namespace

void BindCompatible(py::module *m) {
  BindPassCompatible(m);
  BindOpCompatible(m);
  BindOpUpdateType(m);
  BindOpUpdateBase(m);
  BindOpVersionDesc(m);
  BindOpCheckpoint(m);
  BindOpVersion(m);
}

}  // namespace paddle::pybind
