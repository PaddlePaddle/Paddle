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
#include <cudnn.h>
#endif
#ifdef PADDLE_WITH_TENSORRT
#include "paddle/fluid/inference/tensorrt/helper.h"
#endif

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
  auto trt_cudnn_version_ = GetCudnnMajorVersion();
  return std::get<0>(trt_cudnn_version_) * 1000 +
         std::get<1>(trt_cudnn_version_) * 100 +
         std::get<2>(trt_cudnn_version_);
#else
  return 0;
#endif
}
PassContorl::PassContorl(std::string pass_name,
                         std::vector<std::string> support_categories,
                         std::map<std::string, int64_t> pass_default_status_map,
                         std::vector<std::map<std::string, std::string>>
                             version_contorl_vector = {}) {
  pass_name_ = pass_name;
  support_categories_ = support_categories;

  if (pass_default_status_map.find("trt") != pass_default_status_map.end()) {
    auto pass_default_state = pass_default_status_map["trt"];
    pass_default_status_map_["trt"] = std::make_unique<PassStatus>(
        "trt", static_cast<PassType>(pass_default_state));
  }
  if (pass_default_status_map.find("gpu") != pass_default_status_map.end()) {
    auto pass_default_state = pass_default_status_map["gpu"];
    pass_default_status_map_["gpu"] = std::make_unique<PassStatus>(
        "gpu", static_cast<PassType>(pass_default_state));
  }

  for (auto version_ctrl_map : version_contorl_vector) {
    auto ctrl_name = version_ctrl_map["ctrl_name"];
    if (ctrl_name == "trt") {
      auto gt_trt_version = std::stoll(version_ctrl_map["gt_trt_version"]);
      auto lt_trt_version = std::stoll(version_ctrl_map["lt_trt_version"]);
      auto trt_pass_ctrl_state =
          std::stoll(version_ctrl_map["trt_pass_ctrl_state"]);
      version_contorl_map_["trt"] = std::make_unique<TrtPassVersionContorl>(
          pass_name,
          gt_trt_version,
          lt_trt_version,
          static_cast<PassType>(trt_pass_ctrl_state));
    } else if (ctrl_name == "cudnn") {
      auto gt_cudnn_version = std::stoll(version_ctrl_map["gt_cudnn_version"]);
      auto lt_cudnn_version = std::stoll(version_ctrl_map["lt_cudnn_version"]);
      auto cudnn_pass_ctrl_state =
          std::stoll(version_ctrl_map["cudnn_pass_ctrl_state"]);
      version_contorl_map_["cudnn"] = std::make_unique<CudnnPassVersionContorl>(
          pass_name,
          gt_cudnn_version,
          lt_cudnn_version,
          static_cast<PassType>(cudnn_pass_ctrl_state));
    }
  }
}
PassType PassContorl::GetPassStatus(std::string pass_runtime_status) {
  // 1.用户设置的status
  if (user_pass_status_ == PassType::Default) {
    return user_pass_status_;
  }
  // 2.版本控制的状态
  if (pass_runtime_status == "trt") {
    TrtPassVersionContorl* trt_pass_ctrl =
        dynamic_cast<TrtPassVersionContorl*>(version_contorl_map_["trt"].get());
    auto gt_trt_version = trt_pass_ctrl->gt_trt_version_;
    auto lt_trt_version = trt_pass_ctrl->gt_trt_version_;
    auto trt_compile_version = GetTrtVersion();
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
  } else if (pass_runtime_status == "gpu") {
    CudnnPassVersionContorl* cudnn_pass_ctrl =
        dynamic_cast<CudnnPassVersionContorl*>(
            version_contorl_map_["trt"].get());
    auto gt_cudnn_version = cudnn_pass_ctrl->gt_cudnn_version_;
    auto lt_cudnn_version = cudnn_pass_ctrl->lt_cudnn_version_;
    auto cudnn_version = GetCudnnVersion();
    if (gt_cudnn_version != 0 && lt_cudnn_version != 0 &&
        (cudnn_version >= gt_cudnn_version &&
         cudnn_version < lt_cudnn_version)) {
      return cudnn_pass_ctrl->cudnn_pass_ctrl_state_;
    }
    if (gt_cudnn_version != 0 && (cudnn_version >= gt_cudnn_version)) {
      return cudnn_pass_ctrl->cudnn_pass_ctrl_state_;
    }
    if (lt_cudnn_version != 0 && (cudnn_version < lt_cudnn_version)) {
      return cudnn_pass_ctrl->cudnn_pass_ctrl_state_;
    }
  }
  // 3.默认的状态
  return user_pass_status_;
}

void LoadDefaultPassCtrl(
    std::map<std::string, std::unique_ptr<PassContorl>>* pass_contorl_map) {
  // preln_residual_bias_fuse_pass
  pass_contorl_map->emplace(
      "preln_residual_bias_fuse_pass",
      std::make_unique<PassContorl>(
          "preln_residual_bias_fuse_pass",
          std::vector<std::string>{"trt"},
          std::map<std::string, int64_t>{{"trt", 1}},
          std::vector<std::map<std::string, std::string>>{
              std::map<std::string, std::string>{{"ctrl_name", "trt"},
                                                 {"gt_trt_version", "8600"},
                                                 {"trt_pass_ctrl_state", "0"}},
          }));

  // trt_skip_layernorm_fuse_pass
  pass_contorl_map->emplace(
      "trt_skip_layernorm_fuse_pass",
      std::make_unique<PassContorl>(
          "trt_skip_layernorm_fuse_pass",
          std::vector<std::string>{"trt"},
          std::map<std::string, int64_t>{{"trt", 0}},
          std::vector<std::map<std::string, std::string>>{
              std::map<std::string, std::string>{{"ctrl_name", "trt"},
                                                 {"gt_trt_version", "8600"},
                                                 {"trt_pass_ctrl_state", "0"}},
          }));

  // vit_attention_fuse_pass
  pass_contorl_map->emplace(
      "vit_attention_fuse_pass",
      std::make_unique<PassContorl>(
          "vit_attention_fuse_pass",
          std::vector<std::string>{"gpu", "trt"},
          std::map<std::string, int64_t>{{"trt", 0}, {"gpu", 1}},
          std::vector<std::map<std::string, std::string>>{
              std::map<std::string, std::string>{{"ctrl_name", "trt"},
                                                 {"gt_trt_version", "8600"},
                                                 {"trt_pass_ctrl_state", "0"}},
          }));

  // layernorm_shift_partition_fuse_pass
  pass_contorl_map->emplace(
      "layernorm_shift_partition_fuse_pass",
      std::make_unique<PassContorl>("layernorm_shift_partition_fuse_pass",
                                    std::vector<std::string>{"trt"},
                                    std::map<std::string, int64_t>{
                                        {"trt", 1},
                                    }));

  // reverse_roll_fuse_pass
  pass_contorl_map->emplace(
      "reverse_roll_fuse_pass",
      std::make_unique<PassContorl>(
          "reverse_roll_fuse_pass",
          std::vector<std::string>{"trt"},
          std::map<std::string, int64_t>{
              {"trt", 1},
          },
          std::vector<std::map<std::string, std::string>>{
              std::map<std::string, std::string>{{"ctrl_name", "trt"},
                                                 {"gt_trt_version", "8600"},
                                                 {"trt_pass_ctrl_state", "0"}},
          }));

  // preln_layernorm_x_fuse_pass
  pass_contorl_map->emplace(
      "preln_layernorm_x_fuse_pass",
      std::make_unique<PassContorl>("preln_layernorm_x_fuse_pass",
                                    std::vector<std::string>{"trt"},
                                    std::map<std::string, int64_t>{
                                        {"trt", 1},
                                    }));

  // split_layernorm_to_math_ops_pass
  pass_contorl_map->emplace(
      "split_layernorm_to_math_ops_pass",
      std::make_unique<PassContorl>("split_layernorm_to_math_ops_pass",
                                    std::vector<std::string>{"trt"},
                                    std::map<std::string, int64_t>{
                                        {"trt", 0},
                                    }));

  // add_support_int8_pass
  pass_contorl_map->emplace(
      "add_support_int8_pass",
      std::make_unique<PassContorl>("add_support_int8_pass",
                                    std::vector<std::string>{"trt"},
                                    std::map<std::string, int64_t>{
                                        {"trt", 0},
                                    }));
}

PaddlePassContorl::PaddlePassContorl(int64_t mixed_precision_mode,
                                     int64_t tensorrt_precision_mode,
                                     bool use_gpu,
                                     bool use_trt) {
  LoadDefaultPassCtrl(&pass_contorl_map_);
  if (use_trt) {
    if (tensorrt_precision_mode == static_cast<int64_t>(Precision::kFloat32)) {
      pass_runtime_status_ = "trt";
    } else if (tensorrt_precision_mode ==
               static_cast<int64_t>(Precision::kHalf)) {
      pass_runtime_status_ = "trt_low";
    } else if (tensorrt_precision_mode ==
               static_cast<int64_t>(Precision::kInt8)) {
      pass_runtime_status_ = "int8";
    }
  } else if (use_gpu) {
    if (mixed_precision_mode == static_cast<int64_t>(Precision::kFloat32)) {
      pass_runtime_status_ = "gpu";
    } else if (mixed_precision_mode == static_cast<int64_t>(Precision::kHalf)) {
      pass_runtime_status_ = "gpu_low";
    }
  }
}
std::vector<std::string> PaddlePassContorl::GetCtrlPassList(
    const std::vector<std::string> passes) {
  auto new_passes = std::vector<std::string>();
  for (auto const pass : passes) {
    if (pass_contorl_map_.find(pass) != pass_contorl_map_.end()) {
      if (pass_contorl_map_[pass]->GetPassStatus(pass_runtime_status_) ==
          PassType::Open) {
        new_passes.push_back(pass);
      }
    }
  }
  return new_passes;
}
}  // namespace paddle
