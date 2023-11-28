/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/platform/device/gcu/utils/types.h"

namespace paddle {
namespace platform {
namespace gcu {
using TensorPtr = std::shared_ptr<phi::DenseTensor>;

static std::map<std::string, std::vector<std::string>> g_channel_last_kernels =
    {{"conv2d", {"Filter"}},
     {"conv2d_grad", {"Filter", "Filter@GRAD"}},
     {"conv3d", {"Filter"}},
     {"conv3d_grad", {"Filter", "Filter@GRAD"}},
     {"depthwise_conv2d", {"Filter"}},
     {"depthwise_conv2d_grad", {"Filter", "Filter@GRAD"}}};

static std::set<std::string> g_channellast_trans_impl_ops = {
    "conv2d",
    "conv3d",
    "batch_norm",
    "relu",
    "elementwise_add",
    "elementwise_sub",
    "elementwise_mul",
    "elementwise_div",
    "pool2d",
    "bilinear_interp_v2",
    "nearest_interp",
    "nearest_interp_v2"};

// for layout transfer
enum class GcuLayout : int { NCHW, NHWC, HWCN, NCDHW, NDHWC, DHWCN };

struct GcuTransInfo {
  GcuLayout srs_layout;
  GcuLayout dst_layout;
  GcuShape src_shape;
  GcuShape dst_shape;
  size_t element_bytes = 4;
  TensorPtr src_tensor = nullptr;
  TensorPtr dst_tensor = nullptr;
  uint8_t* src_data = nullptr;
  uint8_t* dst_data = nullptr;
  bool has_transed = false;

  GcuTransInfo& operator=(const GcuTransInfo& info) = default;
  GcuTransInfo() = default;
  GcuTransInfo(GcuLayout layout_srs,
               GcuLayout layout_dst,
               GcuShape shape_src,
               GcuShape shape_dst,
               size_t width,
               uint8_t* data_src,
               uint8_t* data_dst)
      : srs_layout(layout_srs),
        dst_layout(layout_dst),
        src_shape(shape_src),
        dst_shape(shape_dst),
        element_bytes(width),
        src_tensor(nullptr),
        dst_tensor(nullptr),
        src_data(nullptr),
        dst_data(nullptr) {}
  GcuTransInfo(const GcuTransInfo& info)
      : srs_layout(info.srs_layout),
        dst_layout(info.dst_layout),
        src_shape(info.src_shape),
        dst_shape(info.dst_shape),
        element_bytes(info.element_bytes),
        src_tensor(info.src_tensor),
        dst_tensor(info.dst_tensor),
        src_data(info.src_data),
        dst_data(info.dst_data) {}
  ~GcuTransInfo() = default;
};

class GcuTransfer {
 public:
  GcuTransInfo Trans(const GcuTransInfo& args, bool only_trans_shape);
  const char* ToString(const GcuLayout& layout);
  static std::vector<int64_t> GetPermByFormat(const std::string& src_format,
                                              const std::string& dst_format);

 private:
  GcuShape TransShape(const GcuShape& src_shape,
                      const std::vector<int64_t>& perm);
  void Transpose(GcuTransInfo& args,  // NOLINT
                 const std::vector<int64_t>& perm);
};

}  // namespace gcu
}  // namespace platform
}  // namespace paddle
