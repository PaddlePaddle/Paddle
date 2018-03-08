/* Copyright (c) 2017 PaddlePaddle Authors. All Rights Reserved.

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

#include <mkldnn.hpp>

#include "paddle/fluid/framework/operator.h"

namespace paddle {
namespace platform {

using MKLDNNStream = mkldnn::stream;
using MKLDNNEngine = mkldnn::engine;
using MKLDNNMemory = mkldnn::memory;
using MKLDNNMemoryDescriptor = mkldnn::memory::desc;
using MKLDNNPrimitive = mkldnn::primitive;
using MKLDNNPrimitiveDesc = mkldnn::handle<mkldnn_primitive_desc_t>;

typedef std::unique_ptr<MKLDNNStream> MKLDNNStreamPtr;
typedef std::unique_ptr<MKLDNNEngine> MKLDNNEnginePtr;
typedef std::unique_ptr<MKLDNNMemory> MKLDNNMemoryPtr;
typedef std::unique_ptr<MKLDNNPrimitive> MKLDNNPrimitivePtr;
typedef std::unique_ptr<MKLDNNPrimitiveDesc> MKLDNNPrimitiveDescPtr;

inline mkldnn::memory::desc MKLDNNMemDesc(const std::vector<int>& dims,
                                          mkldnn::memory::data_type data_type,
                                          mkldnn::memory::format format) {
  mkldnn::memory::dims tz = dims;
  return mkldnn::memory::desc({tz}, data_type, format);
}

inline bool CanMKLDNNBeUsed(const framework::ExecutionContext& ctx) {
  bool use_mkldnn = ctx.Attr<bool>("use_mkldnn");
  return use_mkldnn && platform::is_cpu_place(ctx.GetPlace());
}

}  // namespace platform
}  // namespace paddle
