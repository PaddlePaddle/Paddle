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

struct PD_INFER_DECL PassVersionContorl {
  virtual ~PassVersionContorl() = default;
  std::string version_ctrl_name_;  // cudnn版本控制
};

struct PD_INFER_DECL TrtPassVersionContorl : public PassVersionContorl {
  TrtPassVersionContorl(std::string version_ctrl_name,
                        int64_t gt_trt_version,
                        int64_t lt_trt_version,
                        PassType trt_pass_ctrl_state) {
    version_ctrl_name_ = version_ctrl_name;
    gt_trt_version_ = gt_trt_version;
    lt_trt_version_ = lt_trt_version;
    trt_pass_ctrl_state_ = trt_pass_ctrl_state;
  }
  int64_t gt_trt_version_;
  int64_t lt_trt_version_;
  PassType trt_pass_ctrl_state_{PassType::Default};
};
struct PD_INFER_DECL CudnnPassVersionContorl : public PassVersionContorl {
  CudnnPassVersionContorl(std::string version_ctrl_name,
                          int64_t gt_cudnn_version,
                          int64_t lt_cudnn_version,
                          PassType cudnn_pass_ctrl_state) {
    version_ctrl_name_ = version_ctrl_name;
    gt_cudnn_version_ = gt_cudnn_version;
    lt_cudnn_version_ = lt_cudnn_version;
    cudnn_pass_ctrl_state_ = cudnn_pass_ctrl_state;
  }
  int64_t gt_cudnn_version_;
  int64_t lt_cudnn_version_;
  PassType cudnn_pass_ctrl_state_{PassType::Default};
};
struct PD_INFER_DECL PassStatus {
  PassStatus(std::string categorie, PassType pass_default_state) {
    categorie_ = categorie;
    pass_default_state_ = pass_default_state;
  }
  std::string categorie_;
  PassType pass_default_state_{PassType::Default};
};

struct PD_INFER_DECL PassContorl {
  PassType GetPassStatus(std::string pass_runtime_status);
  PassContorl(
      std::string pass_name,
      std::vector<std::string> support_categories,
      std::map<std::string, int64_t> pass_default_status_map,
      std::vector<std::map<std::string, std::string>> version_contorl_vector);

  std::string pass_name_;
  std::vector<std::string> support_categories_;
  std::map<std::string, std::unique_ptr<PassStatus>> pass_default_status_map_;
  std::map<std::string, std::unique_ptr<PassVersionContorl>>
      version_contorl_map_;
  PassType user_pass_status_{PassType::Default};
};

struct PD_INFER_DECL PaddlePassContorl {
  enum class Precision {
    kFloat32 = 0,  ///< fp32
    kInt8,         ///< int8
    kHalf,         ///< fp16
    kBf16,         ///< bf16
  };
  PaddlePassContorl(int64_t mixed_precision_mode,
                    int64_t tensorrt_precision_mode,
                    bool use_gpu,
                    bool use_trt);
  PaddlePassContorl& SetPassStatus(const std::string& pass_name,
                                   const int64_t& pass_status);
  std::vector<std::string> GetCtrlPassList(
      const std::vector<std::string> passes);

 protected:
  std::map<std::string, std::unique_ptr<PassContorl>> pass_contorl_map_;
  std::vector<std::string> ctrl_passes_;
  std::string pass_runtime_status_{"gpu"};
};
};  // namespace paddle
