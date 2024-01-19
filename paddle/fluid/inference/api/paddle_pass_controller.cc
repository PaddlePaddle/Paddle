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

#include "paddle/fluid/inference/api/paddle_pass_controller.h"
#ifdef PADDLE_WITH_CUDA
#include "paddle/phi/backends/gpu/gpu_info.h"
#endif
#ifdef PADDLE_WITH_TENSORRT
#include "paddle/fluid/inference/tensorrt/helper.h"
#endif
#include <fstream>
#include <iostream>
#include "glog/logging.h"
#include "paddle/fluid/platform/flags.h"

PADDLE_DEFINE_EXPORTED_string(pass_controller_config_path,
                              "",
                              "Enable pass controller for manage passes");
PADDLE_DEFINE_EXPORTED_bool(pass_controller_radical_mode,
                            false,
                            "Enable switch to radical mode");

namespace paddle {
int GetTrtVersion() {
#ifdef PADDLE_WITH_TENSORRT
  auto trt_runtime_version_ =
      paddle::inference::tensorrt::GetTrtRuntimeVersion();
  return std::get<0>(trt_runtime_version_) * 1000 +
         std::get<1>(trt_runtime_version_) * 100 +
         std::get<2>(trt_runtime_version_);
#else
  return 0;
#endif
}
int GetCudnnVersion() {
#ifdef PADDLE_WITH_CUDA
  auto cudnn_version_ = phi::backends::gpu::DnnVersion();
  return cudnn_version_;
#else
  return 0;
#endif
}
bool VersionCtrlState::UseVersionCtrlStatus() {
  if (ctrl_name_ == "trt") {
    auto trt_runtime_version = GetTrtVersion();
    if (lt_trt_version_ != 0 && gt_trt_version_ != 0) {
      return trt_runtime_version >= gt_trt_version_ &&
             trt_runtime_version < lt_trt_version_;
    } else if (lt_trt_version_ != 0) {
      return trt_runtime_version < lt_trt_version_;
    } else if (gt_trt_version_ != 0) {
      return trt_runtime_version >= gt_trt_version_;
    }
  } else if (ctrl_name_ == "cudnn") {
    auto cudnn_version = GetCudnnVersion();
    if (gt_cudnn_version_ != 0 && lt_cudnn_version_ != 0) {
      return cudnn_version >= gt_cudnn_version_ &&
             cudnn_version < lt_cudnn_version_;
    } else if (lt_trt_version_ != 0) {
      return cudnn_version < lt_cudnn_version_;
    } else if (gt_trt_version_ != 0) {
      return cudnn_version >= gt_cudnn_version_;
    }
  }
  return false;
}
PassType getVersionCtrlRes(bool has_trt,
                           bool has_cudnn,
                           bool meet2trt_version,
                           bool meet2cudnn_version,
                           PassType trt_ctrl_state,
                           PassType cudnn_ctrl_state) {
  if (has_trt && has_cudnn && meet2trt_version && meet2cudnn_version) {
    return trt_ctrl_state == cudnn_ctrl_state ? trt_ctrl_state
                                              : PassType::Close;
  } else if (has_trt && meet2trt_version) {
    return trt_ctrl_state;
  } else if (has_cudnn && meet2cudnn_version) {
    return cudnn_ctrl_state;
  } else if (has_trt || has_cudnn) {
    return PassType::Default;
  } else {
    return PassType::Default;
  }
}
PassType PassContorller::GetPassStatus(std::string pass_runtime_status,
                                       PassCtrlMode pass_ctrl_mode) {
  if (user_pass_status_ != PassType::Default) {
    return user_pass_status_;
  }
  try {
    if (pass_state_map_.count(pass_runtime_status) == 0) {
      return PassType::Default;
    }
    if (pass_ctrl_mode == PassCtrlMode::RadicalMode &&
        pass_state_map_[pass_runtime_status]
                .version_ctrl_state_["trt"]
                .size() != 0) {
      return PassType::Open;
    }

    auto has_trt_version_ctrl =
        pass_state_map_[pass_runtime_status].version_ctrl_state_["trt"].size() >
                0
            ? true
            : false;
    auto has_cudnn_version_ctrl = pass_state_map_[pass_runtime_status]
                                              .version_ctrl_state_["cudnn"]
                                              .size() > 0
                                      ? true
                                      : false;

    auto use_cudnn_version_state = false;
    auto use_trt_version_state = false;
    auto trt_version_state = PassType::Default;
    auto cudnn_version_state = PassType::Default;
    if (has_trt_version_ctrl) {
      trt_version_state = pass_state_map_[pass_runtime_status]
                              .version_ctrl_state_["trt"][0]
                              .version_ctrl_state_;
      for (auto version_ctrl_info :
           pass_state_map_[pass_runtime_status].version_ctrl_state_["trt"]) {
        use_trt_version_state =
            use_trt_version_state || version_ctrl_info.UseVersionCtrlStatus();
      }
    }
    if (has_cudnn_version_ctrl) {
      cudnn_version_state = pass_state_map_[pass_runtime_status]
                                .version_ctrl_state_["cudnn"][0]
                                .version_ctrl_state_;
      for (auto version_ctrl_info :
           pass_state_map_[pass_runtime_status].version_ctrl_state_["cudnn"]) {
        use_cudnn_version_state =
            use_cudnn_version_state || version_ctrl_info.UseVersionCtrlStatus();
      }
    }
    if (has_trt_version_ctrl || has_cudnn_version_ctrl) {
      auto version_state_res = getVersionCtrlRes(has_trt_version_ctrl,
                                                 has_cudnn_version_ctrl,
                                                 use_trt_version_state,
                                                 use_cudnn_version_state,
                                                 trt_version_state,
                                                 cudnn_version_state);
      if (version_state_res == PassType::Default) {
        return pass_state_map_[pass_runtime_status].pass_default_state_;
      } else {
        return version_state_res;
      }
    } else {
      return pass_state_map_[pass_runtime_status].pass_default_state_;
    }
  } catch (const std::exception& e) {
    LOG(ERROR) << "[pass controller] Error:" << e.what()
               << ",so paddle controller return default";
    return PassType::Default;
  }
}

void PaddlePassContorller::LoadDefaultConfig() {
  LOG(INFO) << "[PaddlePassController]Load default configuration ！";
  ctrl_passes_.assign({"preln_residual_bias_fuse_pass",
                       "trt_skip_layernorm_fuse_pass",
                       "vit_attention_fuse_pass",
                       "layernorm_shift_partition_fuse_pass",
                       "reverse_roll_fuse_pass",
                       "preln_layernorm_x_fuse_pass",
                       "split_layernorm_to_math_ops_pass",
                       "add_support_int8_pass",
                       "merge_layernorm_fuse_pass",
                       "elementwiseadd_transpose_pass"});
  for (auto ctrl_pass_name : ctrl_passes_) {
    LOG(INFO) << "Load pass[" << ctrl_pass_name << "] default config!";
    pass_ctrl_map_.emplace(ctrl_pass_name, PassContorller(ctrl_pass_name));
    if (ctrl_pass_name == "preln_residual_bias_fuse_pass" ||
        ctrl_pass_name == "reverse_roll_fuse_pass" ||
        ctrl_pass_name == "preln_layernorm_x_fuse_pass" ||
        ctrl_pass_name == "layernorm_shift_partition_fuse_pass" ||
        ctrl_pass_name == "split_layernorm_to_math_ops_pass" ||
        ctrl_pass_name == "merge_layernorm_fuse_pass" ||
        ctrl_pass_name == "elementwiseadd_transpose_pass") {
      pass_ctrl_map_[ctrl_pass_name].SetSupportCategories(
          std::vector<std::string>{"trt"});
      pass_ctrl_map_[ctrl_pass_name].pass_state_map_["trt"] = PassState("trt");
      pass_ctrl_map_[ctrl_pass_name].pass_state_map_["trt"].SetPassDefaultState(
          static_cast<PassType>(1));
      VersionCtrlState trtversion =
          VersionCtrlState("trt", static_cast<PassType>(0));
      trtversion.SetGtTrtVersion(8600);
      pass_ctrl_map_[ctrl_pass_name]
          .pass_state_map_["trt"]
          .version_ctrl_state_["trt"]
          .push_back(trtversion);
    }

    if (ctrl_pass_name == "vit_attention_fuse_pass") {
      pass_ctrl_map_[ctrl_pass_name].support_categories_ =
          std::vector<std::string>{"trt", "gpu"};
      pass_ctrl_map_[ctrl_pass_name].pass_state_map_["trt"] = PassState("trt");
      pass_ctrl_map_[ctrl_pass_name].pass_state_map_["trt"].SetPassDefaultState(
          static_cast<PassType>(1));
      pass_ctrl_map_[ctrl_pass_name].pass_state_map_["gpu"] = PassState("gpu");
      pass_ctrl_map_[ctrl_pass_name].pass_state_map_["gpu"].SetPassDefaultState(
          static_cast<PassType>(1));

      VersionCtrlState trtversion =
          VersionCtrlState("trt", static_cast<PassType>(0));
      trtversion.SetGtTrtVersion(8600);
      pass_ctrl_map_[ctrl_pass_name]
          .pass_state_map_["trt"]
          .version_ctrl_state_["trt"]
          .push_back(trtversion);
    }

    if (ctrl_pass_name == "add_support_int8_pass") {
      pass_ctrl_map_[ctrl_pass_name].support_categories_ =
          std::vector<std::string>{"trtint8"};
      pass_ctrl_map_[ctrl_pass_name].pass_state_map_["trtint8"] =
          PassState("trt");
      pass_ctrl_map_[ctrl_pass_name]
          .pass_state_map_["trtint8"]
          .SetPassDefaultState(static_cast<PassType>(1));
    }

    if (ctrl_pass_name == "trt_skip_layernorm_fuse_pass") {
      pass_ctrl_map_[ctrl_pass_name].support_categories_ =
          std::vector<std::string>{"trt", "trtlow"};
      pass_ctrl_map_[ctrl_pass_name].pass_state_map_["trt"] = PassState("trt");
      pass_ctrl_map_[ctrl_pass_name].pass_state_map_["trt"].SetPassDefaultState(
          static_cast<PassType>(1));

      pass_ctrl_map_[ctrl_pass_name].pass_state_map_["trtlow"] =
          PassState("trt");
      pass_ctrl_map_[ctrl_pass_name]
          .pass_state_map_["trtlow"]
          .SetPassDefaultState(static_cast<PassType>(1));

      VersionCtrlState trtversion =
          VersionCtrlState("trt", static_cast<PassType>(0));
      trtversion.SetGtTrtVersion(8600);
      pass_ctrl_map_[ctrl_pass_name]
          .pass_state_map_["trtlow"]
          .version_ctrl_state_["trt"]
          .push_back(trtversion);
    }
  }
}

void safeStoll(const std::string& str, int64_t& result) {  // NOLINT
  if (str.empty() ||
      str.find_first_not_of("0123456789-") != std::string::npos) {
    return;
  }

  try {
    result = static_cast<int64_t>(std::stoll(str));
  } catch (const std::out_of_range& e) {
    LOG(INFO) << "[PassController] inputs is invaild!";
    return;
  }

  return;
}
void PaddlePassContorller::SetPassStatus(const std::string& pass_name,
                                         const int64_t pass_status) {
  if (pass_ctrl_map_.count(pass_name) != 0) {
    pass_ctrl_map_[pass_name].SetUserPassStatus(
        static_cast<PassType>(pass_status));
  }
}
std::vector<VersionCtrlState> GetVersionRange(std::string range,
                                              std::string category_name,
                                              int64_t ctrl_state) {
  std::vector<VersionCtrlState> res;
  std::vector<std::string> version_range_list;
  std::istringstream value_Stream(range);
  std::string token;
  while (std::getline(value_Stream, token, ',')) {
    version_range_list.push_back(token);
  }
  for (auto version_range : version_range_list) {
    size_t delimiterPos = version_range.find('-');
    int64_t start = 0;
    int64_t end = 0;
    if (delimiterPos != std::string::npos) {
      std::string startStr = range.substr(0, delimiterPos);
      std::string endStr = range.substr(delimiterPos + 1);
      if (startStr.empty()) {
        safeStoll(endStr, end);
      } else if (endStr.empty()) {
        safeStoll(startStr, start);
      } else {
        safeStoll(endStr, end);
        safeStoll(startStr, start);
      }
    }
    if (category_name == "trt") {
      VersionCtrlState trtversion =
          VersionCtrlState(category_name, static_cast<PassType>(ctrl_state));
      trtversion.SetGtTrtVersion(start);
      trtversion.SetLtTrtVersion(end);
      res.push_back(trtversion);
    }
    if (category_name == "cudnn") {
      VersionCtrlState cudnnversion =
          VersionCtrlState(category_name, static_cast<PassType>(ctrl_state));
      cudnnversion.SetGtCudnnVersion(start);
      cudnnversion.SetLtCudnnVersion(end);
      res.push_back(cudnnversion);
    }
  }
  return res;
}
void PaddlePassContorller::LoadDefaultPassCtrl() {
  if (FLAGS_pass_controller_config_path == "") {
    LoadDefaultConfig();
  } else {
    LOG(INFO) << "[PaddlePassController]Load  user configured!";
    std::ifstream configFile(FLAGS_pass_controller_config_path);
    if (configFile.is_open()) {
      std::map<std::string,
               std::map<std::string, std::map<std::string, std::string>>>
          pass_status_config;
      std::map<std::string, std::map<std::string, std::string>>
          pass_info_config;
      std::map<
          std::string,
          std::map<std::string,
                   std::map<std::string, std::map<std::string, std::string>>>>
          pass_version_ctrl;

      std::string line;
      std::string pass;
      while (std::getline(configFile, line)) {
        if (line.empty() || line[0] == ';') {
          continue;
        }
        if (line[0] == '[' && line.back() == ']') {
          pass = line.substr(1, line.size() - 2);
          pass_ctrl_map_.emplace(pass, PassContorller(pass));
        } else {
          size_t equalsPos = line.find('=');
          if (equalsPos != std::string::npos) {
            std::string key = line.substr(0, equalsPos);
            std::string value = line.substr(equalsPos + 1);

            if (!pass.empty()) {
              size_t dotPos = key.find('.');
              if (dotPos != std::string::npos) {
                std::string outerKey = key.substr(0, dotPos);
                std::string innerKey = key.substr(dotPos + 1);
                size_t mPos = value.find(':');
                if (mPos != std::string::npos) {
                  std::string innervalue1 = value.substr(0, mPos);
                  std::string innervalue2 = value.substr(mPos + 1);
                  pass_status_config[pass][outerKey][innerKey] = innervalue1;
                  pass_version_ctrl[pass][outerKey][innerKey][innervalue1] =
                      innervalue2;
                } else {
                  pass_status_config[pass][outerKey][innerKey] = value;
                }
              } else {
                pass_info_config[pass][key] = value;
              }
            }
          }
        }
      }
      configFile.close();
      for (auto& pass_ctrl_map : pass_ctrl_map_) {
        auto pass_name = pass_ctrl_map.first;
        auto pass_info_map = pass_info_config[pass_name];
        if (pass_info_map.count("SupportedCategories") != 0) {
          auto categories_value = pass_info_map["SupportedCategories"];
          std::vector<std::string> supportedcategories;
          std::istringstream value_Stream(categories_value);
          std::string token;
          while (std::getline(value_Stream, token, ',')) {
            supportedcategories.push_back(token);
          }
          pass_ctrl_map.second.SetSupportCategories(supportedcategories);
        }
        if (pass_info_map.count("pass_status") != 0) {
          int64_t user_pass_status = 2;
          safeStoll(pass_info_map["pass_status"], user_pass_status);
          SetPassStatus(pass_name, user_pass_status);
        }
        for (const auto& passStatus : pass_status_config[pass_name]) {
          auto pass_category = "trt";
          if (passStatus.first == "TrtLowPassState") {
            pass_category = "trtlow";
          } else if (passStatus.first == "GpuPassState") {
            pass_category = "gpu";
          } else if (passStatus.first == "GpuLowPassState") {
            pass_category = "gpulow";
          } else if (passStatus.first == "TrtInt8PassState") {
            pass_category = "trtint8";
          }
          pass_ctrl_map_[pass_name].pass_state_map_[pass_category] =
              PassState(pass_category);
          auto pass_status_info_map = passStatus.second;
          for (const auto& pass_status_info : pass_status_info_map) {
            if (pass_status_info.first == "pass_default_state") {
              int64_t pass_default_state = 0;
              safeStoll(pass_status_info.second, pass_default_state);
              pass_ctrl_map_[pass_name]
                  .pass_state_map_[pass_category]
                  .SetPassDefaultState(
                      static_cast<PassType>(pass_default_state));
            } else if (pass_status_info.first == "trt_version_range") {
              auto ctrl_state = pass_version_ctrl[pass_name][passStatus.first]
                                                 ["trt_version_range"]
                                                 [pass_status_info.second];
              int64_t trt_pass_ctrl_state = 2;
              safeStoll(ctrl_state, trt_pass_ctrl_state);
              pass_ctrl_map_[pass_name]
                  .pass_state_map_[pass_category]
                  .version_ctrl_state_["trt"] = GetVersionRange(
                  pass_status_info.second, "trt", trt_pass_ctrl_state);
            } else if (pass_status_info.first == "cudnn_version_range") {
              auto ctrl_state = pass_version_ctrl[pass_name][passStatus.first]
                                                 ["cudnn_version_range"]
                                                 [pass_status_info.second];
              int64_t cudnn_pass_ctrl_state = 2;
              safeStoll(ctrl_state, cudnn_pass_ctrl_state);
              pass_ctrl_map_[pass_name]
                  .pass_state_map_[pass_category]
                  .version_ctrl_state_["cudnn"] = GetVersionRange(
                  pass_status_info.second, "cudnn", cudnn_pass_ctrl_state);
            }
          }
        }
      }
    } else {
      LOG(ERROR) << "ini Path is error paddle contorller is invalid!";
    }
  }
}

std::string GetCtrlRuntimeStatus(int64_t mixed_precision_mode,
                                 int64_t tensorrt_precision_mode,
                                 bool use_gpu,
                                 bool use_trt) {
  auto pass_runtime_status = "gpu";
  if (use_gpu && use_trt) {
    // fp32
    if (tensorrt_precision_mode == 0) {
      pass_runtime_status = "trt";
    } else if (tensorrt_precision_mode == 2) {
      // fp16
      pass_runtime_status = "trtlow";
    } else if (tensorrt_precision_mode == 1) {
      pass_runtime_status = "trtint8";
    }
  } else if (use_gpu) {
    if (mixed_precision_mode == 0) {
      pass_runtime_status = "gpu";
    } else if (mixed_precision_mode == 2) {
      pass_runtime_status = "gpulow";
    }
  }
  return pass_runtime_status;
}

const std::vector<std::string> PaddlePassContorller::GetCtrlPassList(
    const std::vector<std::string> passes,
    const int64_t mixed_precision_mode,
    const int64_t tensorrt_precision_mode,
    const bool use_gpu,
    const bool use_trt) {
  if (pass_ctrl_map_.empty()) {
    LoadDefaultPassCtrl();
  }
  auto pass_runtime_status = GetCtrlRuntimeStatus(
      mixed_precision_mode, tensorrt_precision_mode, use_gpu, use_trt);
  auto pass_ctrl_mode = FLAGS_pass_controller_radical_mode
                            ? PassCtrlMode::RadicalMode
                            : PassCtrlMode::DefaultMode;
  auto new_passes = std::vector<std::string>();
  for (auto const pass : passes) {
    if (pass_ctrl_map_.count(pass) != 0) {
      if (pass_ctrl_map_[pass].GetPassStatus(
              pass_runtime_status, pass_ctrl_mode) == PassType::Open) {
        new_passes.push_back(pass);
        continue;
      } else {
        LOG(INFO) << "[pass controller] pass[" << pass << "] is Closed!";
        continue;
      }
    }
    new_passes.push_back(pass);
  }
  return new_passes;
}
}  // namespace paddle
