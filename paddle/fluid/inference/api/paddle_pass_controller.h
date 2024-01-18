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

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "paddle_infer_declare.h"  // NOLINT
///
/// \file paddle_pass_manager.h
///

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
  VersionCtrlState() {}
  explicit VersionCtrlState(std::string ctrl_name, PassType ctrl_state) {
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
  PassState() {}
  explicit PassState(std::string status_name) { status_name_ = status_name; }
  void SetPassDefaultState(PassType pass_default_state) {
    pass_default_state_ = pass_default_state;
  }
  std::string status_name_;
  std::map<std::string, std::vector<VersionCtrlState>> version_ctrl_state_;
  PassType pass_default_state_{PassType::Default};
};

struct PD_INFER_DECL PassContorller {
  PassType GetPassStatus(std::string pass_runtime_status,
                         PassCtrlMode pass_ctrl_mode);
  PassContorller() {}
  explicit PassContorller(std::string pass_name) { pass_name_ = pass_name; }
  void SetUserPassStatus(PassType user_pass_status) {
    user_pass_status_ = user_pass_status;
  }
  void SetSupportCategories(std::vector<std::string> support_categories) {
    support_categories_ = support_categories;
  }
  std::string pass_name_;
  std::vector<std::string> support_categories_;
  std::map<std::string, PassState> pass_state_map_;
  PassType user_pass_status_{PassType::Default};
};

struct PD_INFER_DECL PaddlePassContorller {
  enum class Precision {
    kFloat32 = 0,  ///< fp32
    kInt8,         ///< int8
    kHalf,         ///< fp16
    kBf16,         ///< bf16
  };
  PaddlePassContorller() {}
  explicit PaddlePassContorller(const PaddlePassContorller& other) {
    pass_ctrl_map_ = other.pass_ctrl_map_;
    ctrl_passes_ = other.ctrl_passes_;
  }
  void LoadDefaultPassCtrl();
  void LoadDefaultConfig();
  void SetPassStatus(const std::string& pass_name, const int64_t pass_status);
  const std::vector<std::string> GetCtrlPassList(
      const std::vector<std::string> passes,
      const int64_t mixed_precision_mode,
      const int64_t tensorrt_precision_mode,
      const bool use_gpu,
      const bool use_trt);

 protected:
  // pass->PassContorller
  std::map<std::string, PassContorller> pass_ctrl_map_;
  std::vector<std::string> ctrl_passes_;
};
};  // namespace paddle
