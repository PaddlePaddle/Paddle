// Copyright (c) 2022 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/auto_schedule/cost_model/xgb_cost_model.h"

#include <dirent.h>
#include <glog/logging.h>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <atomic>
#include <cassert>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <regex>
#include <string>
#include <utility>
#include <vector>

#include "paddle/cinn/common/python_interpreter_guard.h"

namespace cinn {
namespace auto_schedule {

std::atomic<int> XgbCostModel::xgb_cost_model_count_(0);

// Convert 1D vector to py numpy
template <typename Dtype>
pybind11::array VectorToNumpy(const std::vector<Dtype>& vec) {
  return pybind11::array(pybind11::cast(vec));
}

// Convert 2D vector to py numpy
template <typename Dtype>
pybind11::array VectorToNumpy(const std::vector<std::vector<Dtype>>& vec) {
  if (vec.size() == 0) {
    return pybind11::array(pybind11::dtype::of<Dtype>(), {0, 0});
  }

  std::vector<size_t> shape{vec.size(), vec[0].size()};
  pybind11::array ret(pybind11::dtype::of<Dtype>(), shape);

  Dtype* py_data = static_cast<Dtype*>(ret.mutable_data());
  for (size_t i = 0; i < vec.size(); ++i) {
    assert(vec[i].size() == shape[1] &&
           "Sub vectors must have same size in VectorToNumpy");
    memcpy(py_data + (shape[1] * i), vec[i].data(), shape[1] * sizeof(Dtype));
  }
  return ret;
}

// the Pybind default Python interpreter doesn't contain some paths in
// sys.path, so we have to add it.
//
// Note: the Pybind default Python interpreter only uses default Python.
// Something may be wrong when users use virtual Python environment.
void AddDistPkgToPythonSysPath() {
  pybind11::module sys_py_mod = pybind11::module::import("sys");
  // short version such as "3.7", "3.8", ...
  std::string py_short_version =
      sys_py_mod.attr("version").cast<std::string>().substr(0, 3);

  std::string site_pkg_str =
      "/usr/local/lib/python" + py_short_version + "/dist-packages";
  sys_py_mod.attr("path").attr("append")(site_pkg_str);

  // TODO(zhhsplendid): warning to users if setuptools hasn't been installed
  DIR* site_pkg_dir = opendir(site_pkg_str.c_str());
  if (site_pkg_dir != nullptr) {
    std::regex setuptool_regex("setuptools-.*-py" + py_short_version +
                               "\\.egg");
    struct dirent* entry = nullptr;
    while ((entry = readdir(site_pkg_dir)) != nullptr) {
      if (std::regex_match(entry->d_name, setuptool_regex)) {
        sys_py_mod.attr("path").attr("append")(site_pkg_str + "/" +
                                               entry->d_name);
      }
    }
    closedir(site_pkg_dir);
  }
}

XgbCostModel::XgbCostModel() {
  common::PythonInterpreterGuard::Guard();
  int previous = xgb_cost_model_count_.fetch_add(1);
  if (previous == 0) {
    AddDistPkgToPythonSysPath();
  }
  xgb_module_ = pybind11::module::import("xgboost");
  xgb_booster_ = xgb_module_.attr("Booster")();
}

void XgbCostModel::Train(const std::vector<std::vector<float>>& samples,
                         const std::vector<float>& labels) {
  update_samples_ = samples;
  update_labels_ = labels;
  pybind11::array np_samples = VectorToNumpy<float>(samples);
  pybind11::array np_labels = VectorToNumpy<float>(labels);

  pybind11::object dmatrix = xgb_module_.attr("DMatrix")(np_samples, np_labels);
  xgb_booster_ = xgb_module_.attr("train")(
      pybind11::dict(), dmatrix, pybind11::int_(kTrainRound_));
}

std::vector<float> XgbCostModel::Predict(
    const std::vector<std::vector<float>>& samples) const {
  pybind11::array np_samples = VectorToNumpy<float>(samples);
  pybind11::object dmatrix = xgb_module_.attr("DMatrix")(np_samples);
  pybind11::array py_result = xgb_booster_.attr("predict")(dmatrix);
  return py_result.cast<std::vector<float>>();
}

void XgbCostModel::Update(const std::vector<std::vector<float>>& samples,
                          const std::vector<float>& labels) {
  update_samples_.insert(update_samples_.end(), samples.begin(), samples.end());
  update_labels_.insert(update_labels_.end(), labels.begin(), labels.end());
  pybind11::array np_samples = VectorToNumpy<float>(update_samples_);
  pybind11::array np_labels = VectorToNumpy<float>(update_labels_);

  pybind11::object dmatrix = xgb_module_.attr("DMatrix")(np_samples, np_labels);
  xgb_booster_ = xgb_module_.attr("train")(
      pybind11::dict(), dmatrix, pybind11::int_(kTrainRound_));
}

void XgbCostModel::Save(const std::string& path) {
  xgb_booster_.attr("save_model")(pybind11::str(path));
}

void XgbCostModel::Load(const std::string& path) {
  xgb_booster_.attr("load_model")(pybind11::str(path));
}

}  // namespace auto_schedule
}  // namespace cinn
