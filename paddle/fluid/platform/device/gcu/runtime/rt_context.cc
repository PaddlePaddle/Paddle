/* Copyright (c) 2023 Enflame. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/platform/device/gcu/runtime/rt_context.h"

#include <dtu/driver/device_manager.h>
#include <gcu/umd/device_ids.h>

#include <string>

#include "paddle/fluid/platform/device/gcu/runtime/rt_stream.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace platform {
namespace gcu {
namespace runtime {
namespace {
ChipType ParseChipType() {
  ChipType type = ChipType::UNKNOW;
  if (dtu::driver::DeviceManager::instance()->IsDorado()) {
    type = ChipType::DORADO;
    if (dtu::driver::DeviceManager::instance()->device_info().clusters_num ==
        2) {
      type = ChipType::DORADO_2C;
    } else {
      VLOG(1) << "[WARN] Paddle now only suport dorado_2c in dorado platform!";
    }
  } else if (dtu::driver::DeviceManager::instance()->IsScorpio()) {
    type = ChipType::SCORPIO;
  } else if (dtu::driver::DeviceManager::instance()->IsPavo()) {
    type = ChipType::PAVO;
  }
  PADDLE_ENFORCE_NE(
      type,
      ChipType::UNKNOW,
      phi::errors::Unavailable("unknown chip type is not support!"));
  return type;
}

std::string GetChipTypeStr(ChipType type) {
  switch (type) {
    case ChipType::LEO:
      return "leo";
    case ChipType::PAVO:
      return "pavo";
    case ChipType::PAVO_1C:
      return "pavo_1c";
    case ChipType::DORADO:
      return "dorado";
    case ChipType::DORADO_2C:
      return "dorado_2c";
    case ChipType::DORADO_3PG:
      return "dorado_3pg";
    case ChipType::LIBRA:
      return "libra";
    case ChipType::SCORPIO:
      return "scorpio";
    default:
      return "unknown";
  }
}

topsDeviceInfo GetDeviceInfo() {
  static const auto dev_info =
      dtu::driver::DeviceManager::instance()->device_info();
  return dev_info;
}
}  // namespace

int Context::VisibleDeviceCount() {
  int count;
  RT_CHECK(topsGetDeviceCount(&count));
  return count;
}

std::string Context::GlobalTargetName() {
  auto chip_type = GlobalChipType();
  return GetChipTypeStr(chip_type);
}

ChipType Context::GlobalChipType() { return ParseChipType(); }

std::shared_ptr<Context> Context::CreateContext(int device) {
  return std::make_shared<Context>(device);
}

Context::Context(int device)
    : device(device), default_exe_stream(nullptr), default_dma_stream(nullptr) {
  Init(device);
}

Context::~Context() {}

void Context::Init(int device) {
  RT_CHECK(topsSetDevice(device));
  auto place = paddle::platform::CustomPlace("gcu", device);
  auto* device_ctx = static_cast<platform::CustomDeviceContext*>(
      platform::DeviceContextPool::Instance().Get(place));
  auto stream = reinterpret_cast<topsStream_t>(device_ctx->stream());
  default_exe_stream = Stream::CreateStream(this, stream);
  default_dma_stream = Stream::CreateStream(this);
}

ChipType Context::GetChipType() const { return GlobalChipType(); }

void Context::Synchronize() {
  GcuDeviceGuard guard(device);
  RT_CHECK(topsDeviceSynchronize());
}

std::string Context::GetName() const { return "GCU" + std::to_string(device); }

std::string Context::GetTargetName() const { return GlobalTargetName(); }

}  // namespace runtime
}  // namespace gcu
}  // namespace platform
}  // namespace paddle
