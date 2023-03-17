// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
#include <unordered_set>
#include "paddle/fluid/platform/device/gpu/gpu_info.h"
namespace paddle {
namespace framework {
namespace ir {

class CutlassTeller {
 public:
  static CutlassTeller *Instance() {
    static CutlassTeller global;
    return &global;
  }

  // 判断这个NCHW conv2d_fusion是否可以转成NHWC 让 cutlass 来处理！
  bool Conv2dFusionCanSupport(ir::Node *conv2d_fusion_node,
                              Scope *scope,
                              int device_id) {
    auto op_desc = conv2d_fusion_node->Op();
    if (op_desc->Type() != "conv2d_fusion") return false;
    auto data_format = op_desc->GetAttrIfExists<std::string>("data_format");
    if (data_format != "NCHW") return false;

    auto filter_names = op_desc->Input("Filter");

    for (const auto &filter_name : filter_names) {
      auto *filter_var = scope->FindLocalVar(filter_name);
      const auto &filter_tensor = filter_var->Get<phi::DenseTensor>();
      CHECK_EQ(filter_tensor.dims().size() == 4UL, true);
      auto groups = op_desc->GetAttrIfExists<int>("groups");
      int oc = filter_tensor.dims()[0];
      int kc = filter_tensor.dims()[1];
      int kh = filter_tensor.dims()[2];
      int kw = filter_tensor.dims()[3];
      auto act_type = op_desc->GetAttrIfExists<std::string>("activation");
      bool has_residual = op_desc->Input("ResidualData").size() >= 1UL;
      if (!Conv2dCanSupport(
              oc, kc, kh, kw, groups, act_type, device_id, has_residual)) {
        return false;
      }
    }
    return true;
  }

  // 判断这个conv+bias能否和act融合，然后让cutlass来处理！
  bool Conv2dCanSupport(int oc,
                        int kc,
                        int kh,
                        int kw,
                        int groups,
                        std::string activation,
                        int device_id,
                        bool has_residual = false) {
    int sm_version = platform::GetGPUComputeCapability(device_id);
    int ic = kc * groups;
    if (!cutlass_sm.count(sm_version)) {
      return false;
    }
    if (oc % CUTLASS_NHWC_ALIGNMENT != 0) {
      return false;
    }

    if (ic % CUTLASS_NHWC_ALIGNMENT != 0) {
      return false;
    }

    if (groups == 1) {
      if (!has_residual && !cba_act_set.count(activation)) {
        return false;
      }
      if (has_residual && !cbaa_act_set.count(activation)) {
        return false;
      }
    } else if (groups == ic && ic == oc) {
      if (has_residual) {
        return false;
      }
      // conv2d_depthwise
      if (!cdba_act_set.count(activation)) {
        return false;
      }
    }
    return true;
  }

  std::unordered_set<std::string> CbaAct(int device_id) {
    int sm_version = platform::GetGPUComputeCapability(device_id);
    if (cutlass_sm.count(sm_version)) {
      return cba_act_set;
    } else {
      return {};
    }
  }

  static const int CUTLASS_NHWC_ALIGNMENT = 8;
  const std::unordered_set<int> cutlass_sm = {
      75,
  };
  const std::unordered_set<std::string> cba_act_set = {
      "relu", "swish", "identity", "leaky_relu", "sigmoid"};
  const std::unordered_set<std::string> cdba_act_set = {
      "identity", "relu", "swish", "sigmoid"};
  const std::unordered_set<std::string> cbaa_act_set = {"relu"};
};

// const std::unordered_set<std::string> CutlassTeller::cutlass_cba_act_set =
// {"relu", "swish", "identity", "leaky_relu"}; const
// std::unordered_set<std::string> CutlassTeller::cutlass_cbaa_act_set =
// {"relu"};

}  // namespace ir
}  // namespace framework
}  // namespace paddle
