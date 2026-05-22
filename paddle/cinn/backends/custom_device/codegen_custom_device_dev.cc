// Copyright (c) 2026 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/backends/custom_device/codegen_custom_device_dev.h"
#include <string>
#include <vector>
#include "paddle/cinn/runtime/custom_device/custom_device_backend_api.h"
#include "paddle/phi/backends/device_manager.h"
#include "paddle/phi/common/place.h"

namespace cinn {
namespace backends {
namespace custom_device {

CodeGenCustomDevice::CodeGenCustomDevice(Target target)
    : CodeGenGpuDev(target) {}

void CodeGenCustomDevice::PrintIncludes() {
  // 1. Basic macro definitions
  str_ += "#define CINN_WITH_CUSTOM_DEVICE\n";
  str_ += "#include \"float16.h\"\n";
  str_ += "#include \"bfloat16.h\"\n";
  str_ += "using cinn::common::float16;\n";
  str_ += "using cinn::common::bfloat16;\n";

  // 2. Dynamically retrieve CustomDevice Runtime Source
  // Logic: Identify current system's Custom Device types -> Get Plugin ->
  // Extract Source
  std::string dev_type = "";
  auto devs = phi::DeviceManager::GetAllCustomDeviceTypes();
  if (!devs.empty()) {
    // Default to the first registered custom device
    // Notice: Multi-vendor Environment not supported yet
    dev_type = devs[0];
  } else {
    LOG(WARNING)
        << "No custom device found, skipping runtime source injection.";
    return;
  }

  // Get the plugin instance
  auto place = phi::CustomPlace(dev_type, 0);
  try {
    auto& plugin =
        cinn::runtime::custom_device::CinnCustomDevicePlugin::GetInstance(
            place);

    // 3. Extract runtime source from Toolchain and append to the generated
    // Kernel
    std::string runtime_src = plugin.GetToolchain()->GetRuntimeSource();
    if (runtime_src.empty()) {
      LOG(WARNING) << "Custom Device [" << dev_type
                   << "] returned empty runtime source.";
    }
    str_ += "\n// ----- Custom Device Runtime Source (Begin) -----\n";
    str_ += runtime_src;
    str_ += "\n// ----- Custom Device Runtime Source (End) -----\n";
  } catch (const std::exception& e) {
    LOG(ERROR) << "Failed to get CinnCustomDevicePlugin: " << e.what();
  }
}

const std::string& CodeGenCustomDevice::GetSourceHeader() {
  static std::string empty_header = "";
  return empty_header;
}

void CodeGenCustomDevice::Visit(const ir::Min* op) {
  str_ += "std::min(";
  ir::Expr a = op->a(), b = op->b();
  auto [unify_bit, both_dyn] =
      common::UnifiedOperandTypeBits(&this->DynamicShapeMap(), op);
  this->ProcessMinMaxOperand(&a, &b, unify_bit, both_dyn);
  IrPrinter::Visit(a);
  str_ += ", ";
  IrPrinter::Visit(b);
  str_ += ")";
}

void CodeGenCustomDevice::Visit(const ir::Max* op) {
  str_ += "std::max(";
  ir::Expr a = op->a(), b = op->b();
  auto [unify_bit, both_dyn] =
      common::UnifiedOperandTypeBits(&this->DynamicShapeMap(), op);
  this->ProcessMinMaxOperand(&a, &b, unify_bit, both_dyn);
  IrPrinter::Visit(a);
  str_ += ", ";
  IrPrinter::Visit(b);
  str_ += ")";
}

}  // namespace custom_device
}  // namespace backends
}  // namespace cinn
