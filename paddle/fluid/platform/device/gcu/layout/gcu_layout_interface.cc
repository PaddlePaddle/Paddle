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

#include "paddle/fluid/platform/device/gcu/layout/gcu_layout_interface.h"

#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "dtu/hlir/dispatch.h"
#include "dtu/hlir/types.h"
#include "paddle/fluid/platform/device/gcu/compiler/tops_compiler.h"
#include "paddle/fluid/platform/device/gcu/runtime/gcu_rt_interface.h"
#include "paddle/fluid/platform/device/gcu/runtime/rt_context.h"
#include "paddle/fluid/platform/device/gcu/runtime/rt_executable.h"
#include "paddle/fluid/platform/device/gcu/runtime/rt_utils.h"
#include "paddle/fluid/platform/device/gcu/utils/utils.h"
#include "paddle/phi/core/tensor_utils.h"

namespace paddle {
namespace platform {
namespace gcu {

using GcuOp = ::builder::Op;
using GcuOpPtr = std::shared_ptr<GcuOp>;
using GcuPrimitiveType = builder::PrimitiveType;
using GcuType = builder::Type;
using GcuShape = std::vector<int64_t>;
using GcuBuilder = builder::Builder;
using GcuBuilderPtr = std::shared_ptr<builder::Builder>;
using GcuGraphPtr = std::shared_ptr<hlir::Module>;
using GcuMemPtr = std::shared_ptr<paddle::platform::gcu::runtime::Memory>;
using ExecutablePtr = paddle::platform::gcu::runtime::ExecutablePtr;

static std::map<size_t, ExecutablePtr> map_str_to_exec;

// src layout, dst layout, perm_list
static std::map<Layout, std::map<Layout, const std::vector<int64_t>>>
    kPermTable{
        {Layout::NCHW,
         {{Layout::NHWC, {0, 2, 3, 1}}, {Layout::HWCN, {2, 3, 1, 0}}}},
        {Layout::NHWC, {{Layout::NCHW, {0, 3, 1, 2}}}},
        {Layout::HWCN, {{Layout::NCHW, {3, 2, 0, 1}}}},
        {Layout::NCDHW,
         {{Layout::NDHWC, {0, 2, 3, 4, 1}}, {Layout::DHWCN, {2, 3, 4, 1, 0}}}},
        {Layout::NDHWC, {{Layout::NCDHW, {0, 4, 1, 2, 3}}}},
        {Layout::DHWCN, {{Layout::NCDHW, {4, 3, 0, 1, 2}}}},
    };

static std::map<Layout, const std::string> kLayoutToString = {
    {Layout::NCHW, "NCHW"},
    {Layout::NHWC, "NHWC"},
    {Layout::HWCN, "HWCN"},
    {Layout::NCDHW, "NCDHW"},
    {Layout::NDHWC, "NDHWC"},
    {Layout::DHWCN, "DHWCN"}};

static std::map<const std::string, Layout> kStringToLayout = {
    {"NCHW", Layout::NCHW},
    {"NHWC", Layout::NHWC},
    {"HWCN", Layout::HWCN},
    {"NCDHW", Layout::NCDHW},
    {"NDHWC", Layout::NDHWC},
    {"DHWCN", Layout::DHWCN}};

std::vector<int64_t> GetPermByFormat(const std::string& src_format,
                                     const std::string& dst_format) {
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

size_t GetElementSize(const builder::PrimitiveType& dtype) {
  if (dtype == builder::PrimitiveType::PRED()) {
    return 1;
  } else if (dtype == builder::PrimitiveType::S8()) {
    return 1;
  } else if (dtype == builder::PrimitiveType::S16()) {
    return 2;
  } else if (dtype == builder::PrimitiveType::S32()) {
    return 4;
  } else if (dtype == builder::PrimitiveType::S64()) {
    return 8;
  } else if (dtype == builder::PrimitiveType::F16()) {
    return 2;
  } else if (dtype == builder::PrimitiveType::F32()) {
    return 4;
  } else if (dtype == builder::PrimitiveType::F64()) {
    return 8;
  } else if (dtype == builder::PrimitiveType::U8()) {
    return 1;
  } else if (dtype == builder::PrimitiveType::U16()) {
    return 2;
  } else if (dtype == builder::PrimitiveType::U32()) {
    return 4;
  } else if (dtype == builder::PrimitiveType::U64()) {
    return 8;
  } else {
    return 0;
  }
}

std::string DtypeToString(const builder::PrimitiveType& dtype) {
  if (dtype == builder::PrimitiveType::PRED()) {
    return "bool";
  } else if (dtype == builder::PrimitiveType::S8()) {
    return "int8";
  } else if (dtype == builder::PrimitiveType::S16()) {
    return "int16";
  } else if (dtype == builder::PrimitiveType::S32()) {
    return "int32";
  } else if (dtype == builder::PrimitiveType::S64()) {
    return "int64";
  } else if (dtype == builder::PrimitiveType::F16()) {
    return "f16";
  } else if (dtype == builder::PrimitiveType::F32()) {
    return "f32";
  } else if (dtype == builder::PrimitiveType::F64()) {
    return "f64";
  } else if (dtype == builder::PrimitiveType::U8()) {
    return "uint8";
  } else if (dtype == builder::PrimitiveType::U16()) {
    return "uint16";
  } else if (dtype == builder::PrimitiveType::U32()) {
    return "uint32";
  } else if (dtype == builder::PrimitiveType::U64()) {
    return "uint64";
  } else {
    return "none";
  }
}

Layout StringToLayout(const std::string& layout) {
  auto it = kStringToLayout.find(layout);
  if (it == kStringToLayout.end()) {
    PADDLE_THROW(platform::errors::InvalidArgument("Unsupport gcu layout %s",
                                                   layout.c_str()));
  }
  return it->second;
}

std::string LayoutToString(const Layout& layout) {
  auto it = kLayoutToString.find(layout);
  if (it == kLayoutToString.end()) {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "Unsupport gcu layout %d", static_cast<int32_t>(layout)));
  }
  return it->second;
}

Layout GetFormatByPdShape(const std::vector<int64_t>& pdshape) {
  if (pdshape.size() <= 4) {
    return Layout::NCHW;
  } else if (pdshape.size() == 5) {
    return Layout::NCDHW;
  } else {
    VLOG(1) << "[warn] undefined format when shape is rank[%zu]; return NCHW "
               "as default.It may be cause something wrong!"
            << pdshape.size();
    return Layout::NCHW;
  }
}

std::vector<int64_t> TransShapeByFormat(const std::vector<int64_t>& shape,
                                        const std::string& src_format,
                                        const std::string& dst_format) {
  auto perm = GetPermByFormat(src_format, dst_format);
  std::vector<int64_t> transed_shape(shape.size(), 0);
  for (size_t idx = 0; idx < perm.size(); idx++) {
    transed_shape[idx] = shape[perm[idx]];
  }
  return transed_shape;
}

GcuTensor::GcuTensor(const std::vector<int64_t>& shape,
                     Layout format,
                     builder::PrimitiveType dtype)
    : shape_(shape), format_(format), dtype_(dtype) {
  numel_ = 1;
  std::for_each(
      shape.begin(), shape.end(), [&](int64_t dim) { numel_ *= dim; });
  size_ = numel_ * GetElementSize(dtype);
}

void GcuTensor::SetShape(const std::vector<int64_t>& shape) {
  shape_ = shape;
  numel_ = 1;
  std::for_each(
      shape.begin(), shape.end(), [&](int64_t dim) { numel_ *= dim; });
  size_ = numel_ * GetElementSize(dtype_);
}

void GcuTensor::SetDataType(const builder::PrimitiveType& dtype) {
  dtype_ = dtype;
  size_ = numel_ * GetElementSize(dtype_);
}

static size_t GenerateKey(const GcuTensor& in,
                          const std::vector<int64_t>& perm) {
  std::hash<std::string> hasher;
  std::ostringstream os;
  os << "transpose_in_dims:" << in.GetShapeStr()
     << "_in_type:" << DtypeToString(in.GetDataType()) << "_perm:"
     << "[";
  for (const auto& dim : perm) {
    os << dim << ",";
  }
  os << "]";
  return hasher(os.str());
}

static void RunTranspose(const ExecutablePtr& exec,
                         const GcuTensor& in,
                         GcuTensor& out) {  // NOLINT
  int device_id = runtime::GcuGetCurrentDevice();
  auto ctx = runtime::GcuGetContext(device_id);
  PADDLE_ENFORCE_NE(
      ctx, nullptr, platform::errors::NotFound("create runtime ctx failed"));
  auto stream = ctx->default_exe_stream;
  PADDLE_ENFORCE_NE(
      stream, nullptr, platform::errors::NotFound("create stream failed"));
  if (exec == nullptr) {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "input exec is null when do transpose on gcu"));
  }
  stream->RunExecutableAsync(exec, {in.Data()}, {out.Data()});

  // sync for run
  stream->Synchronize();
}

void GcuTranspose(const GcuTensor& in,
                  GcuTensor& out,  // NOLINT
                  const std::vector<int64_t>& perm) {
  auto executable_key = GenerateKey(in, perm);
  auto iter = map_str_to_exec.find(executable_key);
  if (iter != map_str_to_exec.end()) {
    RunTranspose(iter->second, in, out);
    return;
  }
  // compile first
  auto builder = std::make_shared<GcuBuilder>();
  builder->SetShapeInference(true);
  // construct transpose module
  builder::Type input_type(in.GetShape(), in.GetDataType());
  auto x = builder->CreateInput(input_type);
  auto transpose = builder::Transpose(x, perm);
  builder->SetOutput({transpose});
  // compile hlir module
  auto exec = CompileExecutable(builder->GetModule());
  // create runtime gcu_executable
  auto gcu_executable =
      std::shared_ptr<runtime::Executable>(new runtime::Executable(exec));
  map_str_to_exec[executable_key] = gcu_executable;
  // Run transpose in gcu
  return RunTranspose(gcu_executable, in, out);
}
}  // namespace gcu
}  // namespace platform
}  // namespace paddle
