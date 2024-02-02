// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include <algorithm>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "paddle_infer_declare.h"  // NOLINT

/// \namespace paddle

namespace paddle {

enum PassType {
  Close = 0,  ///
  Open,       ///
  Default,    ///
};

enum PassCtrlMode {
  DefaultMode = 0,
  RadicalMode,
};

struct PD_INFER_DECL VersionCtrlState {
  VersionCtrlState() = default;

  explicit VersionCtrlState(const std::string& ctrl_name, PassType ctrl_state) {
    ctrl_name_ = ctrl_name;
    version_ctrl_state_ = ctrl_state;
  }

  void SetGtTrtVersion(int64_t gt_trt_version) {
    gt_trt_version_ = gt_trt_version;
  }

  void SetLtTrtVersion(int64_t lt_trt_version) {
    lt_trt_version_ = lt_trt_version;
  }

  void SetGtCudnnVersion(int64_t gt_cudnn_version) {
    gt_cudnn_version_ = gt_cudnn_version;
  }

  void SetLtCudnnVersion(int64_t lt_cudnn_version) {
    lt_cudnn_version_ = lt_cudnn_version;
  }

  bool UseVersionCtrlStatus();

  std::string ctrl_name_;
  int64_t gt_trt_version_{0};
  int64_t lt_trt_version_{0};
  int64_t gt_cudnn_version_{0};
  int64_t lt_cudnn_version_{0};
  PassType version_ctrl_state_{PassType::Default};
};

struct PD_INFER_DECL PassState {
  PassState() = default;

  explicit PassState(PassType pass_default_state) {
    pass_default_state_ = pass_default_state;
  }

  void SetPassDefaultState(PassType pass_default_state) {
    pass_default_state_ = pass_default_state;
  }

  std::map<std::string, std::vector<VersionCtrlState>> version_ctrl_state_;

  PassType pass_default_state_{PassType::Default};
};

struct PD_INFER_DECL SinglePassContorller {
  SinglePassContorller() = default;

  explicit SinglePassContorller(const std::string& pass_name) {
    pass_name_ = pass_name;
  }

  PassType GetPassStatus(const std::string& pass_runtime_status,
                         PassCtrlMode pass_ctrl_mode);

  void SetUserPassStatus(PassType user_pass_status) {
    user_pass_status_ = user_pass_status;
  }

  void SetRunTimeList(const std::vector<std::string>& runtime_status_list) {
    runtime_status_list_ = runtime_status_list;
  }

  bool HasRuntimeStatue(const std::string& runtime_status) {
    auto has_runtime_status = std::find(runtime_status_list_.begin(),
                                        runtime_status_list_.end(),
                                        runtime_status);
    return has_runtime_status != runtime_status_list_.end();
  }

  std::string pass_name_;
  std::vector<std::string> runtime_status_list_;  // runtime_status
  std::map<std::string, PassState>
      pass_state_map_;  // runtime_status->PassState
  PassType user_pass_status_{PassType::Default};
};

class PD_INFER_DECL PassContorller {
 public:
  PassContorller() = default;

  PassContorller(const PassContorller& other) {
    pass_ctrl_map_ = other.pass_ctrl_map_;
    ctrl_passes_ = other.ctrl_passes_;
  }

  bool LoadDefaultPassCtrl();

  bool LoadDefaultConfig();

  void SetPassStatus(const std::string& pass_name, const int64_t pass_status);

  const std::vector<std::string> GetCtrlPassList(
      const std::vector<std::string> passes,
      const int64_t mixed_precision_mode,
      const int64_t tensorrt_precision_mode,
      const bool use_gpu,
      const bool use_trt);

 protected:
  // pass->SinglePassContorller
  std::map<std::string, SinglePassContorller> pass_ctrl_map_;
  std::vector<std::string> ctrl_passes_;
};
};  // namespace paddle
