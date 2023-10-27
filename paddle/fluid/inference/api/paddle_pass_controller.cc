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
#include <iostream>
#include "glog/logging.h"
#include "paddle/fluid/platform/flags.h"

PADDLE_DEFINE_EXPORTED_string(pass_controller_config_path,
                              "",
                              "Enable pass controller for manage passes");

namespace paddle {
int GetTrtVersion() {
#ifdef PADDLE_WITH_TENSORRT
  auto trt_cmp_version_ = paddle::inference::tensorrt::GetTrtCompileVersion();
  return std::get<0>(trt_cmp_version_) * 1000 +
         std::get<1>(trt_cmp_version_) * 100 + std::get<2>(trt_cmp_version_);
#else
  return 0;
#endif
}
int GetCudnnVersion() {
#ifdef PADDLE_WITH_CUDA
  auto trt_cudnn_version_ = phi::backends::gpu::DnnVersion();
  return trt_cudnn_version_;
#else
  return 0;
#endif
}

PassType PassContorller::GetPassStatus(std::string pass_runtime_status,
                                       PassCtrlMode pass_ctrl_mode) {
  if (user_pass_status_ != PassType::Default) {
    return user_pass_status_;
  }
  try {
    if (pass_runtime_status == "trt") {
      TrtPassState* trt_pass_ctrl =
          dynamic_cast<TrtPassState*>(pass_state_map_["trt"].get());
      if (trt_pass_ctrl) {
        auto gt_trt_version = trt_pass_ctrl->gt_trt_version_;
        auto lt_trt_version = trt_pass_ctrl->gt_trt_version_;
        auto trt_compile_version = GetTrtVersion();
        if ((gt_trt_version != 0 || lt_trt_version != 0) &&
            pass_ctrl_mode == PassCtrlMode::RadicalMode) {
          return PassType::Open;
        }
        if (gt_trt_version != 0 && lt_trt_version != 0 &&
            (trt_compile_version >= gt_trt_version &&
             trt_compile_version < lt_trt_version)) {
          return trt_pass_ctrl->trt_pass_ctrl_state_;
        }
        if (gt_trt_version != 0 && (trt_compile_version >= gt_trt_version)) {
          return trt_pass_ctrl->trt_pass_ctrl_state_;
        }
        if (lt_trt_version != 0 && (trt_compile_version < lt_trt_version)) {
          return trt_pass_ctrl->trt_pass_ctrl_state_;
        }
        return trt_pass_ctrl->pass_default_state_;
      } else {
        throw std::bad_cast();
      }
    } else if (pass_runtime_status == "gpu") {
      GpuPassState* gpu_pass_ctrl =
          dynamic_cast<GpuPassState*>(pass_state_map_["trt"].get());
      if (gpu_pass_ctrl) {
        auto gt_cudnn_version = gpu_pass_ctrl->gt_cudnn_version_;
        auto lt_cudnn_version = gpu_pass_ctrl->lt_cudnn_version_;
        auto cudnn_version = GetCudnnVersion();
        if (gt_cudnn_version != 0 && lt_cudnn_version != 0 &&
            (cudnn_version >= gt_cudnn_version &&
             cudnn_version < lt_cudnn_version)) {
          return gpu_pass_ctrl->cudnn_pass_ctrl_state_;
        }
        if (gt_cudnn_version != 0 && (cudnn_version >= gt_cudnn_version)) {
          return gpu_pass_ctrl->cudnn_pass_ctrl_state_;
        }
        if (lt_cudnn_version != 0 && (cudnn_version < lt_cudnn_version)) {
          return gpu_pass_ctrl->cudnn_pass_ctrl_state_;
        }
        return gpu_pass_ctrl->pass_default_state_;
      }
    } else {
      throw std::bad_cast();
    }
  } catch (const std::bad_cast& e) {
    LOG(ERROR) << "[pass controller] Error:" << e.what()
               << "Load config.ini error,so paddle controller return default";
    return PassType::Default;
  } catch (const std::exception& e) {
    LOG(ERROR) << "[pass controller] Error:" << e.what()
               << ",so paddle controller return default";
    return PassType::Default;
  }
  return PassType::Default;
}

PaddlePassContorller::PaddlePassContorller(std::string pass_runtime_status,
                                           PassCtrlMode pass_ctrl_mode) {
  ctrl_passes_.assign({
      "preln_residual_bias_fuse_pass",
      "trt_skip_layernorm_fuse_pass",
      "vit_attention_fuse_pass",
      "layernorm_shift_partition_fuse_pass",
      "reverse_roll_fuse_pass",
      "preln_layernorm_x_fuse_pass",
      "split_layernorm_to_math_ops_pass",
      "add_support_int8_pass",
  });
  pass_runtime_status_ = pass_runtime_status;
  pass_ctrl_mode_ = pass_ctrl_mode;
  if (pass_ctrl_mode == PassCtrlMode::RadicalMode) {
    LOG(INFO) << "[PaddlePassContorller] is RadicalMode!";
  } else {
    LOG(INFO) << "[PaddlePassContorller] is DefaultMode!";
  }
}
void PaddlePassContorller::LoadDefaultConfig() {
  LOG(INFO) << "[PaddlePassController]Load default configuration ！";
  for (auto ctrl_pass_name : ctrl_passes_) {
    LOG(INFO) << "Load pass[" << ctrl_pass_name << "] default config!";
    pass_ctrl_map_.emplace(ctrl_pass_name,
                           std::make_unique<PassContorller>(ctrl_pass_name));
    if (ctrl_pass_name == "preln_residual_bias_fuse_pass") {
      pass_ctrl_map_[ctrl_pass_name]->support_categories_ =
          std::vector<std::string>{"trt"};
      pass_ctrl_map_[ctrl_pass_name]->pass_state_map_["trt"] =
          std::make_unique<TrtPassState>("trt");
      pass_ctrl_map_[ctrl_pass_name]
          ->pass_state_map_["trt"]
          ->SetPassDefaultState(static_cast<PassType>(1));
      pass_ctrl_map_[ctrl_pass_name]->pass_state_map_["trt"]->SetGtTrtVersion(
          8600);
      pass_ctrl_map_[ctrl_pass_name]
          ->pass_state_map_["trt"]
          ->SetTrtPassCtrlState(static_cast<PassType>(0));
    }
    if (ctrl_pass_name == "trt_skip_layernorm_fuse_pass") {
      pass_ctrl_map_[ctrl_pass_name]->support_categories_ =
          std::vector<std::string>{"trt"};
      pass_ctrl_map_[ctrl_pass_name]->pass_state_map_["trt"] =
          std::make_unique<TrtPassState>("trt");
      pass_ctrl_map_[ctrl_pass_name]
          ->pass_state_map_["trt"]
          ->SetPassDefaultState(static_cast<PassType>(1));
      pass_ctrl_map_[ctrl_pass_name]->pass_state_map_["trt"]->SetGtTrtVersion(
          8600);
      pass_ctrl_map_[ctrl_pass_name]
          ->pass_state_map_["trt"]
          ->SetTrtPassCtrlState(static_cast<PassType>(0));
    }
    if (ctrl_pass_name == "vit_attention_fuse_pass") {
      pass_ctrl_map_[ctrl_pass_name]->support_categories_ =
          std::vector<std::string>{"trt", "gpu"};
      pass_ctrl_map_[ctrl_pass_name]->pass_state_map_["trt"] =
          std::make_unique<TrtPassState>("trt");
      pass_ctrl_map_[ctrl_pass_name]
          ->pass_state_map_["trt"]
          ->SetPassDefaultState(static_cast<PassType>(1));
      pass_ctrl_map_[ctrl_pass_name]->pass_state_map_["trt"]->SetGtTrtVersion(
          8600);
      pass_ctrl_map_[ctrl_pass_name]
          ->pass_state_map_["trt"]
          ->SetTrtPassCtrlState(static_cast<PassType>(0));
      pass_ctrl_map_[ctrl_pass_name]->pass_state_map_["gpu"] =
          std::make_unique<GpuPassState>("gpu");
      pass_ctrl_map_[ctrl_pass_name]
          ->pass_state_map_["gpu"]
          ->SetPassDefaultState(static_cast<PassType>(1));
    }
    if (ctrl_pass_name == "layernorm_shift_partition_fuse_pass") {
      pass_ctrl_map_[ctrl_pass_name]->support_categories_ =
          std::vector<std::string>{"trt"};
      pass_ctrl_map_[ctrl_pass_name]->pass_state_map_["trt"] =
          std::make_unique<TrtPassState>("trt");
      pass_ctrl_map_[ctrl_pass_name]
          ->pass_state_map_["trt"]
          ->SetPassDefaultState(static_cast<PassType>(1));
    }
    if (ctrl_pass_name == "reverse_roll_fuse_pass") {
      pass_ctrl_map_[ctrl_pass_name]->pass_state_map_["trt"] =
          std::make_unique<TrtPassState>("trt");
      pass_ctrl_map_[ctrl_pass_name]
          ->pass_state_map_["trt"]
          ->SetPassDefaultState(static_cast<PassType>(1));
      pass_ctrl_map_[ctrl_pass_name]->pass_state_map_["trt"]->SetGtTrtVersion(
          8600);
      pass_ctrl_map_[ctrl_pass_name]
          ->pass_state_map_["trt"]
          ->SetTrtPassCtrlState(static_cast<PassType>(0));
    }
    if (ctrl_pass_name == "preln_layernorm_x_fuse_pass") {
      pass_ctrl_map_[ctrl_pass_name]->support_categories_ =
          std::vector<std::string>{"trt"};
      pass_ctrl_map_[ctrl_pass_name]->pass_state_map_["trt"] =
          std::make_unique<TrtPassState>("trt");
      pass_ctrl_map_[ctrl_pass_name]
          ->pass_state_map_["trt"]
          ->SetPassDefaultState(static_cast<PassType>(1));
    }
    if (ctrl_pass_name == "split_layernorm_to_math_ops_pass") {
      pass_ctrl_map_[ctrl_pass_name]->support_categories_ =
          std::vector<std::string>{"trt"};
      pass_ctrl_map_[ctrl_pass_name]->pass_state_map_["trt"] =
          std::make_unique<TrtPassState>("trt");
      pass_ctrl_map_[ctrl_pass_name]
          ->pass_state_map_["trt"]
          ->SetPassDefaultState(static_cast<PassType>(0));
    }
    if (ctrl_pass_name == "add_support_int8_pass") {
      pass_ctrl_map_[ctrl_pass_name]->support_categories_ =
          std::vector<std::string>{"trt"};
      pass_ctrl_map_[ctrl_pass_name]->pass_state_map_["trt"] =
          std::make_unique<TrtPassState>("trt");
      pass_ctrl_map_[ctrl_pass_name]
          ->pass_state_map_["trt"]
          ->SetPassDefaultState(static_cast<PassType>(0));
    }
  }
}

void safeStoll(const std::string& str, const int64_t& result) {
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
                                         const int64_t& pass_status) {
  if (pass_ctrl_map_.count(pass_name) != 0) {
    pass_ctrl_map_[pass_name]->SetUserPassStatus(
        static_cast<PassType>(pass_status));
  }
}

void PaddlePassContorller::LoadDefaultPassCtrl() {
  if (FLAGS_pass_controller_config_path == "") {
    LoadDefaultConfig();
  } else {
    LOG(INFO) << "[PaddlePassController]Load  user configured！";
    std::ifstream configFile(FLAGS_pass_controller_config_path);
    if (configFile.is_open()) {
      std::map<std::string,
               std::map<std::string, std::map<std::string, std::string>>>
          pass_status_config;
      std::map<std::string, std::map<std::string, std::string>>
          pass_info_config;
      std::string line;
      std::string pass;
      while (std::getline(configFile, line)) {
        if (line.empty() || line[0] == ';') {
          continue;
        }
        if (line[0] == '[' && line.back() == ']') {
          pass = line.substr(1, line.size() - 2);
          pass_ctrl_map_.emplace(pass, std::make_unique<PassContorller>(pass));
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
                pass_status_config[pass][outerKey][innerKey] = value;
              } else {
                pass_info_config[pass][key] = value;
              }
            }
          }
        }
      }
      configFile.close();
      for (const auto& pass_ctrl_map : pass_ctrl_map_) {
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
          pass_ctrl_map.second->support_categories_ = supportedcategories;
        }
        if (pass_info_map.count("pass_status") != 0) {
          int64_t user_pass_status = 2;
          safeStoll(pass_info_map["pass_status"], user_pass_status);
          SetPassStatus(pass_name, user_pass_status);
        }
        for (const auto& passStatus : pass_status_config[pass_name]) {
          if (passStatus.first == "TrtPassState") {
            pass_ctrl_map_[pass_name]->pass_state_map_["trt"] =
                std::make_unique<TrtPassState>("trt");
            auto pass_status_info_map = passStatus.second;
            for (const auto& pass_status_info : pass_status_info_map) {
              if (pass_status_info.first == "pass_default_state") {
                int64_t pass_default_state = 0;
                safeStoll(pass_status_info.second, pass_default_state);
                pass_ctrl_map_[pass_name]
                    ->pass_state_map_["trt"]
                    ->SetPassDefaultState(
                        static_cast<PassType>(pass_default_state));
              } else if (pass_status_info.first == "gt_trt_version") {
                int64_t gt_trt_version = 0;
                safeStoll(pass_status_info.second, gt_trt_version);
                pass_ctrl_map_[pass_name]
                    ->pass_state_map_["trt"]
                    ->SetGtTrtVersion(static_cast<PassType>(gt_trt_version));
              } else if (pass_status_info.first == "lt_trt_version") {
                int64_t lt_trt_version = 0;
                safeStoll(pass_status_info.second, lt_trt_version);
                pass_ctrl_map_[pass_name]
                    ->pass_state_map_["trt"]
                    ->SetLtTrtVersion(static_cast<PassType>(lt_trt_version));
              } else if (pass_status_info.first == "trt_pass_ctrl_state") {
                int64_t trt_pass_ctrl_state = 2;
                safeStoll(pass_status_info.second, trt_pass_ctrl_state);
                pass_ctrl_map_[pass_name]
                    ->pass_state_map_["trt"]
                    ->SetTrtPassCtrlState(
                        static_cast<PassType>(trt_pass_ctrl_state));
              }
            }
          } else if (passStatus.first == "GpuPassState") {
            pass_ctrl_map_[pass_name]->pass_state_map_["gpu"] =
                std::make_unique<GpuPassState>("gpu");
            auto pass_status_info_map = passStatus.second;
            for (const auto& pass_status_info : pass_status_info_map) {
              if (pass_status_info.first == "pass_default_state") {
                int64_t pass_default_state = 0;
                safeStoll(pass_status_info.second, pass_default_state);
                pass_ctrl_map_[pass_name]
                    ->pass_state_map_["gpu"]
                    ->SetPassDefaultState(
                        static_cast<PassType>(pass_default_state));
              } else if (pass_status_info.first == "gt_cudnn_version") {
                int64_t gt_cudnn_version = 0;
                safeStoll(pass_status_info.second, gt_cudnn_version);
                pass_ctrl_map_[pass_name]
                    ->pass_state_map_["trt"]
                    ->SetGtTrtVersion(static_cast<PassType>(gt_cudnn_version));
              } else if (pass_status_info.first == "lt_cudnn_version") {
                int64_t lt_cudnn_version = 0;
                safeStoll(pass_status_info.second, lt_cudnn_version);
                pass_ctrl_map_[pass_name]
                    ->pass_state_map_["trt"]
                    ->SetLtTrtVersion(static_cast<PassType>(lt_cudnn_version));
              } else if (pass_status_info.first == "gpu_pass_ctrl_state") {
                int64_t gpu_pass_ctrl_state = 2;
                safeStoll(pass_status_info.second, gpu_pass_ctrl_state);
                pass_ctrl_map_[pass_name]
                    ->pass_state_map_["trt"]
                    ->SetTrtPassCtrlState(
                        static_cast<PassType>(gpu_pass_ctrl_state));
              }
            }
          }
        }
      }
    } else {
      LOG(ERROR) << "ini Path is error paddle contorller is invalid!";
    }
  }
}

std::vector<std::string> PaddlePassContorller::GetCtrlPassList(
    const std::vector<std::string> passes) {
  if (pass_ctrl_map_.empty()) {
    LoadDefaultPassCtrl();
  }
  auto new_passes = std::vector<std::string>();
  for (auto const pass : passes) {
    if (pass_ctrl_map_.count(pass) != 0) {
      if (pass_ctrl_map_[pass]->GetPassStatus(
              pass_runtime_status_, pass_ctrl_mode_) == PassType::Open) {
        new_passes.push_back(pass);
        LOG(INFO) << "[pass controller]  pass[" << pass << "] is Opened!";
        continue;
      } else if (pass_ctrl_map_[pass]->GetPassStatus(pass_runtime_status_,
                                                     pass_ctrl_mode_) ==
                 PassType::Close) {
        LOG(INFO) << "[pass controller] pass[" << pass << "] is Closed!";
        continue;
      }
    }
    LOG(INFO) << "[pass controller]  pass[" << pass << "] is Opened!";
    new_passes.push_back(pass);
  }
  return new_passes;
}
}  // namespace paddle
