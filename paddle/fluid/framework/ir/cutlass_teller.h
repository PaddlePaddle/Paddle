// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.

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

typedef enum {
  cba,     // This servers for conv_elementwise_add_fuse_pass
  cbaa,    // This servers for conv_elementwise_add2_act_fuse_pass
  cbaele,  // This servers for conv2d_fusion_cutlass_elementwise
} CutlassFusionType;

class CutlassTeller {
 public:
  static CutlassTeller *Instance() {
    static CutlassTeller global;
    return &global;
  }

#if defined(PADDLE_WITH_CUTLASS)
  // Determine this NCHW conv2d + bias can be fused with activation by cutlass?
  // This servers for conv_elementwise_add_fuse_pass.
  // will not set or change any attribute in op_desc
  bool CbaCanSupport(OpDesc *op_desc,
                     Scope *scope,
                     std::string act_type,
                     int device_id) {
    auto strides = op_desc->GetAttrIfExists<std::vector<int>>("strides");
    auto dilations = op_desc->GetAttrIfExists<std::vector<int>>("dilations");
    PADDLE_ENFORCE_EQ(strides.size(),
                      2UL,
                      phi::errors::InvalidArgument(
                          "The 'strides' attribute in conv2d should be a "
                          "vector of size 2, but received size %d.",
                          strides.size()));
    PADDLE_ENFORCE_EQ(dilations.size(),
                      2UL,
                      phi::errors::InvalidArgument(
                          "The 'dilations' attribute in conv2d should be a "
                          "vector of size 2, but received size %d.",
                          dilations.size()));
    int stride_h = strides[0];
    int stride_w = strides[1];
    int dilation_h = dilations[0];
    int dilation_w = dilations[1];

    auto filter_names = op_desc->Input("Filter");

    for (const auto &filter_name : filter_names) {
      auto *filter_var = scope->FindLocalVar(filter_name);
      const auto &filter_tensor = filter_var->Get<phi::DenseTensor>();
      PADDLE_ENFORCE_EQ(filter_tensor.dims().size(),
                        4UL,
                        phi::errors::InvalidArgument(
                            "The 'Filter' tensor in conv2d should have 4 "
                            "dimensions, but received dimensions %d.",
                            filter_tensor.dims().size()));
      auto groups = op_desc->GetAttrIfExists<int>("groups");
      int oc = filter_tensor.dims()[0];
      int kc = filter_tensor.dims()[1];
      int kh = filter_tensor.dims()[2];
      int kw = filter_tensor.dims()[3];

      // For convience, we only support EXPLICIT
      auto padding_algorithm =
          op_desc->GetAttrIfExists<std::string>("padding_algorithm");
      if (padding_algorithm != "EXPLICIT") {
        return false;
      }

      if (!Conv2dCanSupport(oc,
                            kc,
                            kh,
                            kw,
                            stride_h,
                            stride_w,
                            dilation_h,
                            dilation_w,
                            groups,
                            act_type,
                            device_id,
                            CutlassFusionType::cba)) {
        return false;
      }
    }
    return true;
  }

  // Determine this NCHW conv2d + bias + elewise_add + act can be fused by
  // cutlass?, this is for conv_elementwise_add_fuse_pass
  // will not set or change any attribute in op_desc
  bool CbaaCanSupport(OpDesc *op_desc,
                      Scope *scope,
                      std::string act_type,
                      int device_id) {
    auto strides = op_desc->GetAttrIfExists<std::vector<int>>("strides");
    auto dilations = op_desc->GetAttrIfExists<std::vector<int>>("dilations");
    PADDLE_ENFORCE_EQ(strides.size(),
                      2UL,
                      phi::errors::InvalidArgument(
                          "The 'strides' attribute in conv2d should be a "
                          "vector of size 2, but received size %d.",
                          strides.size()));
    PADDLE_ENFORCE_EQ(dilations.size(),
                      2UL,
                      phi::errors::InvalidArgument(
                          "The 'dilations' attribute in conv2d should be a "
                          "vector of size 2, but received size %d.",
                          dilations.size()));
    int stride_h = strides[0];
    int stride_w = strides[1];
    int dilation_h = dilations[0];
    int dilation_w = dilations[1];

    auto filter_names = op_desc->Input("Filter");

    for (const auto &filter_name : filter_names) {
      auto *filter_var = scope->FindLocalVar(filter_name);
      const auto &filter_tensor = filter_var->Get<phi::DenseTensor>();
      PADDLE_ENFORCE_EQ(filter_tensor.dims().size(),
                        4UL,
                        phi::errors::InvalidArgument(
                            "The 'Filter' tensor in conv2d should have 4 "
                            "dimensions, but received dimensions %d.",
                            filter_tensor.dims().size()));
      auto groups = op_desc->GetAttrIfExists<int>("groups");
      int oc = filter_tensor.dims()[0];
      int kc = filter_tensor.dims()[1];
      int kh = filter_tensor.dims()[2];
      int kw = filter_tensor.dims()[3];

      // For convience, we only support EXPLICIT
      auto padding_algorithm =
          op_desc->GetAttrIfExists<std::string>("padding_algorithm");
      if (padding_algorithm != "EXPLICIT") {
        return false;
      }

      if (!Conv2dCanSupport(oc,
                            kc,
                            kh,
                            kw,
                            stride_h,
                            stride_w,
                            dilation_h,
                            dilation_w,
                            groups,
                            act_type,
                            device_id,
                            CutlassFusionType::cbaa)) {
        return false;
      }
    }
    return true;
  }

  // Determine this NCHW conv2d_fusion + elewise_op + act1 can be fused by
  // cutlass?
  //  This servers for conv2d_fusion_cutlass_elementwise.
  // will not set or change any attribute in op_desc
  bool CbaeleCanSupport(OpDesc *op_desc,
                        Scope *scope,
                        std::string ele_type,
                        std::string act1_type,
                        int device_id) {
    auto strides = op_desc->GetAttrIfExists<std::vector<int>>("strides");
    auto dilations = op_desc->GetAttrIfExists<std::vector<int>>("dilations");
    PADDLE_ENFORCE_EQ(strides.size(),
                      2UL,
                      phi::errors::InvalidArgument(
                          "The 'strides' attribute in conv2d should be a "
                          "vector of size 2, but received size %d.",
                          strides.size()));
    PADDLE_ENFORCE_EQ(dilations.size(),
                      2UL,
                      phi::errors::InvalidArgument(
                          "The 'dilations' attribute in conv2d should be a "
                          "vector of size 2, but received size %d.",
                          dilations.size()));
    int stride_h = strides[0];
    int stride_w = strides[1];
    int dilation_h = dilations[0];
    int dilation_w = dilations[1];
    auto act_type = op_desc->GetAttrIfExists<std::string>("activation");

    // Do not allow conv2d_fusion already have residual input.
    if (op_desc->Input("ResidualData").size() >= 1) {
      return false;
    }

    auto filter_names = op_desc->Input("Filter");

    for (const auto &filter_name : filter_names) {
      auto *filter_var = scope->FindLocalVar(filter_name);
      const auto &filter_tensor = filter_var->Get<phi::DenseTensor>();
      PADDLE_ENFORCE_EQ(filter_tensor.dims().size(),
                        4UL,
                        phi::errors::InvalidArgument(
                            "The 'Filter' tensor in conv2d should have 4 "
                            "dimensions, but received dimensions %d.",
                            filter_tensor.dims().size()));
      auto groups = op_desc->GetAttrIfExists<int>("groups");
      int oc = filter_tensor.dims()[0];
      int kc = filter_tensor.dims()[1];
      int kh = filter_tensor.dims()[2];
      int kw = filter_tensor.dims()[3];

      // For convience, we only support EXPLICIT
      auto padding_algorithm =
          op_desc->GetAttrIfExists<std::string>("padding_algorithm");
      if (padding_algorithm != "EXPLICIT") {
        return false;
      }

      if (!Conv2dCanSupport(oc,
                            kc,
                            kh,
                            kw,
                            stride_h,
                            stride_w,
                            dilation_h,
                            dilation_w,
                            groups,
                            act_type,
                            device_id,
                            CutlassFusionType::cbaele,
                            act1_type,
                            ele_type)) {
        return false;
      }
    }
    return true;
  }

  // Determine whether this conv can be fused with the activation by cutlass
  // backend.
  bool Conv2dCanSupport(int oc,
                        int kc,
                        int kh,
                        int kw,
                        int stride_h,
                        int stride_w,
                        int dilation_h,
                        int dilation_w,
                        int groups,
                        std::string activation,
                        int device_id,
                        CutlassFusionType fuse_type,
                        // below two are used by cbaele
                        std::string activation1 = "identity",
                        std::string elemenstwise_type = "elementwise_add") {
    int sm_version = platform::GetGPUComputeCapability(device_id);
    int ic = kc * groups;
    if (!cutlass_sm.count(sm_version)) {
      return false;
    }

    // To prevent generating too many cutlass code,
    // we only allow oc and ic is divisible by CUTLASS_NHWC_ALIGNMENT
    if (groups == 1) {
      if (oc % CUTLASS_NHWC_ALIGNMENT != 0 ||
          ic % CUTLASS_NHWC_ALIGNMENT != 0) {
        return false;
      }
      // conv + bias + act
      if (fuse_type == CutlassFusionType::cba &&
          !cba_act_set.count(activation)) {
        return false;
      }
      // conv + bias + elementwise_add + act
      if (fuse_type == CutlassFusionType::cbaa &&
          !cbaa_act_set.count(activation)) {
        return false;
      }

      // conv + bias + act + elementwise_op
      if (fuse_type == CutlassFusionType::cbaele &&
          !cbaele_act_set.count(activation + "_" + elemenstwise_type + "_" +
                                activation1)) {
        return false;
      }

    } else if (groups == ic && ic == oc) {
      // return false;
      //  conv2d_depthwise not support residual input
      if (fuse_type != CutlassFusionType::cba) {
        return false;
      }

      // Now we only 3x3s1s2, 5x5s1s2
      if (!(kh == 3 && kw == 3) || (kh == 5 && kw == 5)) {
        return false;
      }

      if (!(stride_h == 1 || stride_h == 2)) {
        return false;
      }

      if (stride_h != stride_w) {
        return false;
      }

      if (dilation_h != 1) {
        return false;
      }

      if (dilation_w != 1) {
        return false;
      }

      // Now we only allow ic % 8 == 0, because of cutlass.
      if (ic % 8 != 0) {
        return false;
      }

      // conv2d_depthwise + bias + act
      if (!cdba_act_set.count(activation)) {
        return false;
      }
    } else {
      // only support groups == 1 or conv2d_depthwise
      return false;
    }
    return true;
  }
  // Return the supported activation set by cutlass conv + bias + act pattern
  std::unordered_set<std::string> CbaAct(int device_id) {
    int sm_version = platform::GetGPUComputeCapability(device_id);
    if (cutlass_sm.count(sm_version)) {
      return cba_act_set;
    } else {
      return {};
    }
  }
  // Return the supported activation set by cutlass conv + bias + act pattern
  std::unordered_set<std::string> CbaaAct(int device_id) {
    int sm_version = platform::GetGPUComputeCapability(device_id);
    if (cutlass_sm.count(sm_version)) {
      return cbaa_act_set;
    } else {
      return {};
    }
  }
#else

  bool CbaaCanSupport(OpDesc *op_desc,
                      Scope *scope,
                      std::string act_type,
                      int device_id) {
    return false;
  }

  bool CbaCanSupport(OpDesc *op_desc,
                     Scope *scope,
                     std::string act_type,
                     int device_id) {
    return false;
  }

  bool CbaeleCanSupport(OpDesc *op_desc,
                        Scope *scope,
                        std::string ele_type,
                        std::string act1_type,
                        int device_id) {
    return false;
  }

  bool Conv2dCanSupport(int oc,
                        int kc,
                        int kh,
                        int kw,
                        int stride_h,
                        int stride_w,
                        int dilation_h,
                        int dilation_w,
                        int groups,
                        std::string activation,
                        int device_id,
                        CutlassFusionType fuse_type,
                        // below two are used by cbaele
                        std::string activation1 = "identity",
                        std::string elemenstwise_type = "elementwise_add") {
    return false;
  }
  std::unordered_set<std::string> CbaAct(int device_id) { return {}; }
  std::unordered_set<std::string> CbaaAct(int device_id) { return {}; }
#endif
  static const int CUTLASS_NHWC_ALIGNMENT = 8;
  const std::unordered_set<int> cutlass_sm = {
      75,
      80,
      85,
      86,
  };
  const std::unordered_set<std::string> cba_act_set = {
      "relu", "swish", "identity", "leaky_relu", "sigmoid"};

  // conv2d_depthwise act
  const std::unordered_set<std::string> cdba_act_set = {
      "identity", "relu", "swish", "sigmoid"};
  const std::unordered_set<std::string> cbaa_act_set = {"relu"};
  const std::unordered_set<std::string> cbaele_act_set = {
      "identity_elementwise_add_identity",
      "swish_elementwise_add_identity",
  };
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle
