// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/feed_hook.h"
#include <fstream>
#include <limits>
#include <random>
#include <sstream>
#include "paddle/common/flags.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/pir/include/core/program.h"

COMMON_DECLARE_string(logging_pir_py_code_dir);
COMMON_DECLARE_bool(logging_trunc_pir_py_code);

namespace paddle::framework {

namespace {

std::optional<std::string> GetLoggingFilePath() {
  if (FLAGS_logging_pir_py_code_dir.empty()) return std::nullopt;
  const std::string file_path =
      FLAGS_logging_pir_py_code_dir + "/programs_example_input_tensor_meta.py";
  return file_path;
}

void TryTruncateLoggingFile() {
  if (!FLAGS_logging_trunc_pir_py_code) return;
  std::optional<std::string> file_path = GetLoggingFilePath();
  if (!file_path.has_value()) return;
  static std::once_flag once_flag;
  std::call_once(once_flag, [&] {
    std::ofstream ofs;
    ofs.open(file_path.value().c_str(), std::ios::out | std::ios::trunc);
    ofs.close();
  });
}

template <typename DoEachFeadNameT>
void VisitFeedName(const pir::Program& program,
                   const DoEachFeadNameT& DoEachFeadName) {
  auto module_op = program.module_op();
  const auto& block = module_op.block();
  const auto& IsDataOp = [](const pir::Operation& op) -> bool {
    return op.isa<paddle::dialect::DataOp>();
  };
  const auto& GetDataOpName = [](const pir::Operation& op) -> std::string {
    return op.attributes().at("name").dyn_cast<pir::StrAttribute>().AsString();
  };
  for (const auto& op : block) {
    if (IsDataOp(op)) {
      DoEachFeadName(GetDataOpName(op));
    }
  }
  for (const auto& [name, _] : block.kwargs()) {
    DoEachFeadName(name);
  }
}

std::string GetLoggingShapeOrDataForName(int64_t program_id,
                                         const std::string& name,
                                         const phi::DenseTensor& tensor) {
  int64_t random_id = [&] {
    std::random_device rd{};
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<int64_t> dis(
        0, std::numeric_limits<int64_t>::max());
    return dis(gen);
  }();
  std::ostringstream ss;
  ss << "class PirProgram_example_input_tensor_meta_" << random_id << ":";
  ss << "\n\tprogram_id = " << program_id;
  ss << "\n\tinput_name = " << std::quoted(name);
  ss << "\n\tshape = [";
  int i = 0;
  for (int dim : ::common::vectorize<int64_t>(tensor.dims())) {
    if (i++ > 0) {
      ss << ", ";
    }
    ss << dim;
  }
  ss << "]";
  ss << "\n\n";
  return ss.str();
}

void AppendToLoggingFile(const std::string& logging_str) {
  std::optional<std::string> file_path = GetLoggingFilePath();
  if (!file_path.has_value()) return;
  std::ofstream ofs;
  ofs.open(file_path.value().c_str(), std::ios::out | std::ios::app);
  if (!ofs.is_open()) return;
  ofs << logging_str << std::endl;
  ofs.close();
}

void AppendLoggingShapeOrDataForName(int64_t uid,
                                     const std::string& name,
                                     const phi::DenseTensor& tensor) {
  static std::mutex mutex;
  std::unique_lock<std::mutex> lock(mutex);
  using Name2OnceFlag = std::unordered_map<std::string, std::once_flag>;
  static std::unordered_map<int64_t, Name2OnceFlag> once_flags;
  std::call_once(once_flags[uid][name], [&] {
    AppendToLoggingFile(GetLoggingShapeOrDataForName(uid, name, tensor));
  });
}

void SaveLoggingShapeOrData(const pir::Program& program, const Scope& scope) {
  if (FLAGS_logging_pir_py_code_dir.empty()) return;
  TryTruncateLoggingFile();
  VisitFeedName(program, [&](const std::string& name) {
    Variable* variable = scope.FindVar(name);
    if (variable == nullptr) return;
    if (!variable->IsType<phi::DenseTensor>()) return;
    const phi::DenseTensor& tensor = variable->Get<phi::DenseTensor>();
    AppendLoggingShapeOrDataForName(program.id(), name, tensor);
  });
}

}  // namespace

void RunFeedHooks(const pir::Program& program, const Scope& scope) {
  SaveLoggingShapeOrData(program, scope);
}

}  // namespace paddle::framework
