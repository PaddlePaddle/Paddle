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

struct PD_INFER_DECL PassState {
  virtual ~PassState() = default;
  virtual void SetGtTrtVersion(int64_t gt_trt_version) {}
  virtual void SetLtTrtVersion(int64_t lt_trt_version) {}
  virtual void SetTrtPassCtrlState(PassType trt_pass_ctrl_state) {}
  virtual void SetGtCudnnVersion(int64_t gt_cudnn_version) {}
  virtual void SetLtCudnnVersion(int64_t lt_cudnn_version) {}
  virtual void SetCudnnPassCtrlState(PassType cudnn_pass_ctrl_state) {}
  virtual void SetPassDefaultState(PassType pass_default_state) {}
};

struct PD_INFER_DECL TrtPassState : public PassState {
  explicit TrtPassState(std::string version_ctrl_name) {
    version_ctrl_name_ = version_ctrl_name;
  }
  void SetGtTrtVersion(int64_t gt_trt_version) override {
    gt_trt_version_ = gt_trt_version;
  }
  void SetLtTrtVersion(int64_t lt_trt_version) override {
    lt_trt_version_ = lt_trt_version;
  }
  void SetTrtPassCtrlState(PassType trt_pass_ctrl_state) override {
    trt_pass_ctrl_state_ = trt_pass_ctrl_state;
  }
  void SetPassDefaultState(PassType pass_default_state) override {
    pass_default_state_ = pass_default_state;
  }
  std::string version_ctrl_name_;
  int64_t gt_trt_version_{0};
  int64_t lt_trt_version_{0};
  PassType trt_pass_ctrl_state_{PassType::Default};
  PassType pass_default_state_{PassType::Default};
};
struct PD_INFER_DECL GpuPassState : public PassState {
  explicit GpuPassState(std::string version_ctrl_name) {
    version_ctrl_name_ = version_ctrl_name;
  }
  void SetGtCudnnVersion(int64_t gt_cudnn_version) override {
    gt_cudnn_version_ = gt_cudnn_version;
  }
  void SetLtCudnnVersion(int64_t lt_cudnn_version) override {
    lt_cudnn_version_ = lt_cudnn_version;
  }
  void SetCudnnPassCtrlState(PassType cudnn_pass_ctrl_state) override {
    cudnn_pass_ctrl_state_ = cudnn_pass_ctrl_state;
  }
  void SetPassDefaultState(PassType pass_default_state) override {
    pass_default_state_ = pass_default_state;
  }
  std::string version_ctrl_name_;
  int64_t gt_cudnn_version_{0};
  int64_t lt_cudnn_version_{0};
  PassType cudnn_pass_ctrl_state_{PassType::Default};
  PassType pass_default_state_{PassType::Default};
};

struct PD_INFER_DECL PassContorller {
  PassType GetPassStatus(std::string pass_runtime_status,
                         PassCtrlMode pass_ctrl_mode);
  explicit PassContorller(std::string pass_name) { pass_name_ = pass_name; }
  void SetUserPassStatus(PassType user_pass_status) {
    user_pass_status_ = user_pass_status;
  }
  std::string pass_name_;
  std::vector<std::string> support_categories_;
  std::map<std::string, std::unique_ptr<PassState>> pass_state_map_;
  PassType user_pass_status_{PassType::Default};
};

struct PD_INFER_DECL PaddlePassContorller {
  enum class Precision {
    kFloat32 = 0,  ///< fp32
    kInt8,         ///< int8
    kHalf,         ///< fp16
    kBf16,         ///< bf16
  };
  PaddlePassContorller();
  PaddlePassContorller(std::string pass_runtime_status,
                       PassCtrlMode pass_ctrl_mode = PassCtrlMode::DefaultMode);
  void LoadDefaultPassCtrl();
  void LoadDefaultConfig();
  void SetPassStatus(const std::string& pass_name, const int64_t& pass_status);
  void SetPassCtrlMode(PassCtrlMode pass_ctrl_mode) {
    pass_ctrl_mode_ = pass_ctrl_mode;
  }
  std::string GetPassRuntimeStatus() { return pass_runtime_status_; }
  PassCtrlMode GetPassCtrlMode() { return pass_ctrl_mode_; }
  void SetPassRuntimeStatus(std::string pass_runtime_status) {
    pass_runtime_status_ = pass_runtime_status;
  }
  std::vector<std::string> GetCtrlPassList(
      const std::vector<std::string> passes);

 protected:
  std::map<std::string, std::unique_ptr<PassContorller>> pass_ctrl_map_;
  std::vector<std::string> ctrl_passes_;
  std::string pass_runtime_status_{"gpu"};
  PassCtrlMode pass_ctrl_mode_{PassCtrlMode::DefaultMode};
};
};  // namespace paddle
