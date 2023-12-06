// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/inference/api/helper.h"

#include "paddle/fluid/framework/custom_operator.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/platform/init.h"
#include "paddle/phi/api/ext/op_meta_info.h"
#include "paddle/phi/core/flags.h"
#include "paddle/pir/core/ir_context.h"

PHI_DECLARE_bool(enable_pir_in_executor);

namespace paddle {
namespace inference {

template <>
std::string to_string<std::vector<float>>(
    const std::vector<std::vector<float>> &vec) {
  std::stringstream ss;
  for (const auto &piece : vec) {
    ss << to_string(piece) << "\n";
  }
  return ss.str();
}

template <>
std::string to_string<std::vector<std::vector<float>>>(
    const std::vector<std::vector<std::vector<float>>> &vec) {
  std::stringstream ss;
  for (const auto &line : vec) {
    for (const auto &rcd : line) {
      ss << to_string(rcd) << ";\t";
    }
    ss << '\n';
  }
  return ss.str();
}

void RegisterAllCustomOperator() {
  auto &op_meta_info_map = OpMetaInfoMap::Instance();
  const auto &meta_info_map = op_meta_info_map.GetMap();
  for (auto &pair : meta_info_map) {
    if (FLAGS_enable_pir_in_executor) {
      ::pir::IrContext *ctx = ::pir::IrContext::Instance();
      auto *custom_dialect =
          ctx->GetOrRegisterDialect<paddle::dialect::CustomOpDialect>();
      if (custom_dialect->HasRegistered(pair.first)) {
        LOG(INFO) << "The operator `" << pair.first
                  << "` has been registered. "
                     "Therefore, we will not repeat the registration here.";
        continue;
      }
      for (const auto &meta_info : pair.second) {
        LOG(INFO) << "register pir custom op :" << pair.first;
        custom_dialect->RegisterCustomOp(meta_info);
      }
    }
    const auto &all_op_kernels{framework::OperatorWithKernel::AllOpKernels()};
    if (all_op_kernels.find(pair.first) == all_op_kernels.end()) {
      framework::RegisterOperatorWithMetaInfo(pair.second);
    } else {
      LOG(INFO) << "The operator `" << pair.first
                << "` has been registered. "
                   "Therefore, we will not repeat the registration here.";
    }
  }
}

void InitGflagsFromEnv() {
  // support set gflags from environment.
  std::vector<std::string> gflags;
  const phi::ExportedFlagInfoMap &env_map = phi::GetExportedFlagInfoMap();
  std::ostringstream os;
  for (auto &pair : env_map) {
    os << pair.second.name << ",";
  }
  std::string tryfromenv_str = os.str();
  if (!tryfromenv_str.empty()) {
    tryfromenv_str.pop_back();
    tryfromenv_str = "--tryfromenv=" + tryfromenv_str;
    gflags.push_back(tryfromenv_str);
  }
  framework::InitGflags(gflags);
}

}  // namespace inference
}  // namespace paddle
