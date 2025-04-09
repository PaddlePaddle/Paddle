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

#include "paddle/fluid/jit/serializer.h"

#include <set>

#include "paddle/fluid/framework/var_desc.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/jit/engine/pir_interpreter_engine.h"
#include "paddle/phi/core/platform/device_context.h"

#include "paddle/common/flags.h"
#include "paddle/fluid/jit/engine/interpreter_engine.h"
#include "paddle/fluid/jit/engine/predictor_engine.h"
#include "paddle/fluid/jit/layer.h"
#include "paddle/fluid/jit/property.h"
#include "paddle/fluid/jit/serializer_utils.h"

COMMON_DECLARE_string(jit_engine_type);
COMMON_DECLARE_bool(enable_pir_api);
namespace paddle {
namespace jit {

using BaseFunctionInfoMap =
    std::unordered_map<std::string, std::shared_ptr<BaseFunctionInfo>>;
Layer Deserializer::operator()(const std::string& path,
                               const phi::Place& place) {
  const auto& pdmodel_paths = utils::ModelFilePaths(path);
  // set is ordered
  std::set<std::string> param_names_set;
  BaseFunctionInfoMap info_map;
  // PirFunctionInfo pir_info_map;
  for (auto& it : pdmodel_paths) {
    auto& func_name = it.first;
    std::vector<std::string> persist_var_names;
    if (FLAGS_enable_pir_api) {
      auto pir_program = LoadPirProgram(it.second);
      auto module_op = pir_program->module_op();
      auto& block = module_op.block();
      const auto& ops = block.ops();
      for (auto* op : ops) {
        auto values = op->results();
        for (auto& value : values) {
          if (utils::IsPersistable(&value) &&
              value.defining_op()->attributes().count("parameter_name")) {
            const auto& value_name = value.defining_op()
                                         ->attributes()
                                         .at("parameter_name")
                                         .dyn_cast<pir::StrAttribute>();
            persist_var_names.emplace_back(value_name.AsString());
          }
        }
      }
      info_map[func_name] = std::make_shared<PirFunctionInfo>(
          func_name, persist_var_names, pir_program);
    } else {
      auto program_desc = LoadProgram(it.second);

      auto all_var_desc = program_desc.Block(0).AllVars();
      for (auto* desc_ptr : all_var_desc) {
        if (utils::IsPersistable(desc_ptr)) {
          persist_var_names.emplace_back(desc_ptr->Name());
        }
      }
      info_map[func_name] = std::make_shared<FunctionInfo>(
          func_name, persist_var_names, program_desc);
    }

    param_names_set.insert(persist_var_names.begin(), persist_var_names.end());

    info_map[func_name]->SetProgramFilePath(it.second);
  }

  auto params_dict = std::make_shared<VariableMap>();
  auto attrs_dict = std::make_shared<VariableMap>();
  ReadTensorData(path + PDPARAMS_SUFFIX, param_names_set, place, params_dict);

  if (utils::FileExists(path + PROPERTY_SUFFIX)) {
    ReadAttributeData(path + PROPERTY_SUFFIX, attrs_dict);
    VLOG(3) << "Read Property Success!";
  }

  Layer layer = Layer(params_dict, attrs_dict, info_map, place);

  for (auto& map_item : info_map) {
    const std::string& func_name = map_item.first;
    auto& base_info = map_item.second;
    VLOG(3) << "Add function type: " << FLAGS_jit_engine_type
            << " Function name: " << func_name;
    if (FLAGS_enable_pir_api) {
      auto pir_info = std::dynamic_pointer_cast<PirFunctionInfo>(base_info);
      layer.SetEngine(func_name,
                      utils::MakePirEngine<PirInterpreterEngine>(
                          pir_info, params_dict, place, pir_info->Program()));
    } else {
      auto info = std::dynamic_pointer_cast<FunctionInfo>(base_info);
      if (FLAGS_jit_engine_type == "New") {
        layer.SetEngine(
            func_name,
            utils::MakeEngine<InterpreterEngine>(info, params_dict, place));
      } else if (FLAGS_jit_engine_type == "Predictor") {
        layer.SetEngine(
            info->FunctionName(),
            utils::MakeEngine<PredictorEngine>(info, params_dict, place));
      } else {
        PD_THROW("Invalid JitLayer engine type.");
      }
    }
  }
  return layer;
}

void Deserializer::ReadTensorData(
    const std::string& file_name,
    const std::set<std::string>& var_name,
    const phi::Place& place,
    std::shared_ptr<VariableMap> params_dict) const {
  VLOG(3) << "ReadTensorData from: " << file_name;
  std::ifstream fin(file_name, std::ios::binary);
  phi::DeviceContextPool& pool = phi::DeviceContextPool::Instance();
  auto& dev_ctx = *pool.Get(place);
  for (const auto& item : var_name) {
    VLOG(3) << "load Tensor: " << item;
    Variable v;
    // TODO(dev): Support framework::Vocab
    DenseTensor* dense_tensor = v.GetMutable<DenseTensor>();
    phi::DeserializeFromStream(fin, dense_tensor, dev_ctx);
    (*params_dict)[item] = std::make_shared<Variable>(v);
  }
}

void Deserializer::ReadAttributeData(
    const std::string& file_path,
    std::shared_ptr<VariableMap> attrs_dict) const {
  VLOG(3) << "ReadPropertyData from: " << file_path;
  Property p;
  p.Deserialization(file_path);
  for (auto& it : p.Values()) {
    attrs_dict->emplace(it.first, it.second);
  }
  return;
}

framework::ProgramDesc Deserializer::LoadProgram(const std::string& file_name) {
  VLOG(3) << "LoadProgram from: " << file_name;
  std::ifstream fin(file_name, std::ios::in | std::ios::binary);
  fin.seekg(0, std::ios::end);
  std::string buffer(fin.tellg(), ' ');
  fin.seekg(0, std::ios::beg);
  fin.read(&buffer[0], buffer.size());  // NOLINT
  fin.close();
  return framework::ProgramDesc(buffer);
}

std::shared_ptr<pir::Program> Deserializer::LoadPirProgram(
    const std::string& file_name) {
  VLOG(3) << "LoadPirProgram from: " << file_name;
  auto pir_program_ =
      std::make_shared<pir::Program>(pir::IrContext::Instance());
  pir::ReadModule(file_name, pir_program_.get(), 1);
  return pir_program_;
}

Layer Load(const std::string& file_path, const phi::Place& place) {
  auto deserializer = Deserializer();
  return deserializer(file_path, place);
}

}  // namespace jit
}  // namespace paddle
