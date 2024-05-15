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

#include "paddle/fluid/platform/device/gcu/utils/layout.h"

#include <string.h>

#include <functional>
#include <map>
#include <memory>
#include <numeric>

#include "paddle/fluid/platform/device/gcu/gcu_info.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace platform {
namespace gcu {
namespace {
const size_t k4D = 4;
const size_t k5D = 5;
}  // namespace

// src layout, dst layout, perm_list
static std::map<GcuLayout, std::map<GcuLayout, const std::vector<int64_t>>>
    kPermTable{
        {GcuLayout::NCHW,
         {{GcuLayout::NHWC, {0, 2, 3, 1}}, {GcuLayout::HWCN, {2, 3, 1, 0}}}},
        {GcuLayout::NHWC, {{GcuLayout::NCHW, {0, 3, 1, 2}}}},
        {GcuLayout::HWCN, {{GcuLayout::NCHW, {3, 2, 0, 1}}}},
        {GcuLayout::NCDHW,
         {{GcuLayout::NDHWC, {0, 2, 3, 4, 1}},
          {GcuLayout::DHWCN, {2, 3, 4, 1, 0}}}},
        {GcuLayout::NDHWC, {{GcuLayout::NCDHW, {0, 4, 1, 2, 3}}}},
        {GcuLayout::DHWCN, {{GcuLayout::NCDHW, {4, 3, 0, 1, 2}}}},
    };

static std::map<GcuLayout, const std::string> kLayoutToString = {
    {GcuLayout::NCHW, "NCHW"},
    {GcuLayout::NHWC, "NHWC"},
    {GcuLayout::HWCN, "HWCN"},
    {GcuLayout::NCDHW, "NCDHW"},
    {GcuLayout::NDHWC, "NDHWC"},
    {GcuLayout::DHWCN, "DHWCN"}};

static std::map<const std::string, GcuLayout> kStringToLayout = {
    {"NCHW", GcuLayout::NCHW},
    {"NHWC", GcuLayout::NHWC},
    {"HWCN", GcuLayout::HWCN},
    {"NCDHW", GcuLayout::NCDHW},
    {"NDHWC", GcuLayout::NDHWC},
    {"DHWCN", GcuLayout::DHWCN}};

std::vector<int64_t> GcuTransfer::GetPermByFormat(
    const std::string& src_format, const std::string& dst_format) {
  auto it = kStringToLayout.find(src_format);
  PADDLE_ENFORCE_NE(
      it == kStringToLayout.end(),
      true,
      platform::errors::Fatal("Unsupported to get perm for src layout %s",
                              src_format.c_str()));
  auto it_2 = kStringToLayout.find(dst_format);
  PADDLE_ENFORCE_NE(
      it_2 == kStringToLayout.end(),
      true,
      platform::errors::Fatal("Unsupported to get perm for src layout %s",
                              dst_format.c_str()));
  auto iter_src = kPermTable.find(it->second);
  PADDLE_ENFORCE_NE(iter_src == kPermTable.end(),
                    true,
                    platform::errors::Fatal("can not get perm from %s to %s ",
                                            src_format.c_str(),
                                            dst_format.c_str()));
  auto iter_dst = iter_src->second.find(it_2->second);
  PADDLE_ENFORCE_NE(iter_dst == iter_src->second.end(),
                    true,
                    platform::errors::Fatal("can not get perm from %s to %s ",
                                            src_format.c_str(),
                                            dst_format.c_str()));
  return iter_dst->second;
}

std::vector<int64_t> GenHeads(const std::vector<int64_t>& shape) {
  std::vector<int64_t> heads(shape.size());
  bool first = true;
  for (auto i = static_cast<int64_t>(shape.size() - 1); i >= 0; --i) {
    if (first) {
      heads[i] = 1;
      first = false;
    } else {
      heads[i] = shape[i + 1] * heads[i + 1];
    }
  }
  return heads;
}

int64_t GenOffset(const std::vector<int64_t>& offsets,
                  const std::vector<int64_t>& indexes) {
  int64_t offset = 0;
  for (size_t i = 0; i < indexes.size(); ++i) {
    offset += offsets[i] * indexes[i];
  }
  return offset;
}

void AddOne(const std::vector<int64_t>& shape,
            std::vector<int64_t>& indexes) {  // NOLINT
  size_t i = indexes.size() - 1;
  indexes[i]++;
  while (i > 0) {
    if (indexes[i] >= shape[i]) {
      indexes[i] = 0;
      indexes[i - 1]++;
      --i;
    } else {
      break;
    }
  }
}

void GcuTransfer::Transpose(GcuTransInfo& args,
                            const std::vector<int64_t>& perm) {
  PADDLE_ENFORCE_NE(args.src_data,
                    nullptr,
                    platform::errors::Fatal(
                        "input src data is null when do format transfer!"));
  PADDLE_ENFORCE_NE(
      args.dst_data,
      nullptr,
      platform::errors::Fatal("output data is null when do format transfer!"));
  auto dst_shape = TransShape(args.src_shape, perm);
  auto src_origin_ordered_heads = GenHeads(args.src_shape);
  auto src_heads = TransShape(src_origin_ordered_heads, perm);
  int64_t data_size = args.element_bytes;
  int64_t dst_ele_num = std::accumulate(
      dst_shape.begin(), dst_shape.end(), 1, std::multiplies<int>());
  int64_t dst_index = 0;
  std::vector<int64_t> dst_indexes(dst_shape.size());
  while (dst_index < dst_ele_num) {
    auto src_offset = GenOffset(src_heads, dst_indexes) * data_size;
    auto dst_offset_bytes = dst_index * data_size;
    (void)memcpy(args.dst_data + dst_offset_bytes,  // NOLINT
                 args.src_data + src_offset,
                 static_cast<size_t>(data_size));
    AddOne(dst_shape, dst_indexes);
    ++dst_index;
  }
}

const char* GcuTransfer::ToString(const GcuLayout& layout) {
  auto it = kLayoutToString.find(layout);
  PADDLE_ENFORCE_EQ(it != kLayoutToString.end(),
                    true,
                    platform::errors::NotFound("gcu does not impl layout %d",
                                               static_cast<int>(layout)));
  return (it->second).c_str();
}

GcuShape GcuTransfer::TransShape(const GcuShape& src_shape,
                                 const std::vector<int64_t>& perm) {
  GcuShape out_shape;
  for (const auto idx : perm) {
    out_shape.push_back(src_shape.at(idx));
  }
  return out_shape;
}

GcuTransInfo GcuTransfer::Trans(const GcuTransInfo& args,
                                bool only_trans_shape) {
  GcuTransInfo result = args;
  // Trans Shape Firstly
  size_t rank = args.src_shape.size();
  bool is_valid = rank == k4D || rank == k5D;
  PADDLE_ENFORCE_EQ(
      is_valid,
      true,
      platform::errors::Unimplemented(
          "when do layout trans, the input rank must be %zu or %zu", k4D, k5D));
  auto it = kPermTable.find(args.srs_layout);
  PADDLE_ENFORCE_EQ(it != kPermTable.end(),
                    true,
                    platform::errors::NotFound("gcu does not impl layout %s",
                                               ToString(args.srs_layout)));
  auto it_2 = it->second.find(args.dst_layout);
  PADDLE_ENFORCE_EQ(
      it_2 != it->second.end(),
      true,
      platform::errors::NotFound("gcu does not impl from layout %s to %s",
                                 ToString(args.srs_layout),
                                 ToString(args.dst_layout)));
  result.dst_shape = TransShape(args.src_shape, it_2->second);
  if (only_trans_shape) {
    return result;
  }
  // Trans Data Secondly
  Transpose(result, it_2->second);
  return result;
}
}  // namespace gcu
}  // namespace platform
}  // namespace paddle
