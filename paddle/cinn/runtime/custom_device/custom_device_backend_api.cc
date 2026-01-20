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

// paddle/cinn/runtime/custom_device/custom_device_backend_api.cc

#include "paddle/cinn/runtime/custom_device/custom_device_backend_api.h"
#include "glog/logging.h"
#include "paddle/phi/backends/device_ext.h"
#include "paddle/phi/backends/device_manager.h"

#ifdef CINN_WITH_CUSTOM_DEVICE
namespace cinn {
namespace runtime {
namespace custom_device {
  void ForceRegisterCinnCustomDeviceHostAPI();
  void ForceRegisterCinnCustomDeviceIntrinsics(); 
  void ForceRegisterCustomDeviceIntrinsicsReduce();
  void ForceRegisterCustomDeviceIntrinsicsFloat16();

// ============================================================
// 匿名命名空间：定义具体的默认实现类 (不对外暴露)
// ============================================================
namespace {
// 0. 具体的 Module 实现类
// 默认的 CustomDeviceModule 实现(连接 module_unload 和 get_kernel_address)
class DefaultCustomDeviceModule : public cinn::runtime::CustomModule {
 public:
  DefaultCustomDeviceModule(void* handle, C_CinnInterface* cif)
      : handle_(handle), cif_(cif) {}

  // RAII: 析构时自动调用 module_unload
  ~DefaultCustomDeviceModule() override {
    if (handle_ && cif_ && cif_->module_unload) {
      // 传递厂商上下文 dev_ptr 和模块句柄
      cif_->module_unload(cif_->dev_ptr, handle_);
    }
  }

  // 实现基类的 GetFunction
  void* GetFunction(const std::string& func_name) override {
    if (handle_ && cif_ && cif_->get_kernel_address) {
      void* func_ptr = nullptr;
      // 调用 C 接口查找符号
      C_Status status = cif_->get_kernel_address(
          cif_->dev_ptr, handle_, func_name.c_str(), &func_ptr);

      if (status == C_SUCCESS) {
        return func_ptr;
      } else {
        LOG(WARNING) << "Failed to get kernel address for: " << func_name;
      }
    }
    return nullptr;
  }

 private:
  void* handle_;          // 模块句柄
  C_CinnInterface* cif_;  // 接口指针 (用于回调)
};

// 1. 编译工具链接口：负责调用外部编译器 (如 mxcc)
// 默认编译工具链实现
class DefaultCompilerToolchain : public CustomCompilerToolchain {
 public:
  explicit DefaultCompilerToolchain(C_CinnInterface* cif) : cif_(cif) {}

  // 1. 实现 Compile：调用 C 接口
  std::string Compile(const std::string& code) override {
    if (cif_ && cif_->compile) {
      // TODO(Plugin): 这里需要按照具体的 C 接口协议调用 compile
      // void* handle = nullptr;
      char output_path[1024] = {0};
      C_Status status = cif_->compile(
          cif_->dev_ptr, code.c_str(), output_path, sizeof(output_path));
      if (status == C_SUCCESS) {
        VLOG(3) << "Calling Custom Device compile_kernel...";
        return std::string(output_path);
      }
    }
    LOG(ERROR) << "compile_kernel interface not implemented by vendor.";
    return "";
  }

  // 2. 实现 GetRuntimeSource：调用 C 接口
  std::string GetRuntimeSource() override {
    if (cif_ && cif_->get_runtime_source) {
      // 获取厂商内置的 Runtime 源码字符串
      const char* src = cif_->get_runtime_source(cif_->dev_ptr);
      return src ? std::string(src) : "";
    }
    // 如果厂商没提供，可能返回一个空的，或者基础的通用定义
    return "";
  }

 private:
  C_CinnInterface* cif_;
};

// 2. 运行时策略接口：负责加载和启动 Kernel
// 默认运行时策略实现
class DefaultRuntimeStrategy : public CustomRuntimeStrategy {
 public:
  explicit DefaultRuntimeStrategy(C_CinnInterface* cif) : cif_(cif) {}

  std::unique_ptr<cinn::runtime::CustomModule> LoadModule(
      const std::string& path) override {
    if (cif_ && cif_->module_load) {
      void* handle = nullptr;
      C_Status status = cif_->module_load(cif_->dev_ptr, path.c_str(), &handle);

      if (status == C_SUCCESS && handle != nullptr) {
        // 创建 DefaultCustomDeviceModule 并移交所有权
        return std::make_unique<DefaultCustomDeviceModule>(handle, cif_);
      }
    }
    LOG(ERROR) << "Failed to load custom device module from path: " << path;
    return nullptr;
  }

  void LaunchKernel(void* func_ptr,
                    const std::string& func_name,
                    void** args,
                    int num_args,
                    int grid_x,
                    int grid_y,
                    int grid_z,
                    int block_x,
                    int block_y,
                    int block_z,
                    int shared_mem,
                    void* stream) override {
    if (cif_ && cif_->launch_kernel) {
      // 调用 C 接口
      cif_->launch_kernel(cif_->dev_ptr,
                          func_ptr,
                          args,
                          num_args,
                          grid_x,
                          grid_y,
                          grid_z,
                          block_x,
                          block_y,
                          block_z,
                          shared_mem,
                          stream);
      return;
    }
    LOG(ERROR) << "launch_kernel interface not implemented by vendor.";
  }

 private:
  C_CinnInterface* cif_;
};

// 3. 编译优化接口：负责厂商自定义的 Fusion/Schedule/Pass
// 默认编译策略
class DefaultCompileStrategy : public CustomCompileStrategy {
  // 目前使用基类默认实现 (return false)
};

}  // namespace

// ============================================================
// CinnCustomDevicePlugin 实现
// ============================================================

// 1. 实现 InitWrappers：将 C 接口转换为 C++ 策略对象
void CinnCustomDevicePlugin::InitWrappers(C_CinnInterface* cif) {
  // 使用上面定义的 Default 实现类
  toolchain_ = std::make_unique<DefaultCompilerToolchain>(cif);
  runtime_strategy_ = std::make_unique<DefaultRuntimeStrategy>(cif);
  compile_strategy_ = std::make_unique<DefaultCompileStrategy>();
}

// 2. 实现 GetInstance
CinnCustomDevicePlugin& CinnCustomDevicePlugin::GetInstance(
    const phi::Place& place) {
  static std::unordered_map<std::string,
                            std::unique_ptr<CinnCustomDevicePlugin>>
      instances;
  std::string device_type = place.GetDeviceType();

  if (instances.find(device_type) == instances.end()) {
    // A. 获取基础设备指针
    auto* device_base = phi::DeviceManager::GetDeviceWithPlace(place);
    PADDLE_ENFORCE_NOT_NULL(
        device_base,
        phi::errors::NotFound("Device for %s not found.", place.DebugString()));

    // B. 转换为 CustomDevice 并获取 CINN 专属 C 接口
    C_CinnInterface* cif = device_base->GetCinnInterface();

    // C. 检查接口是否存在
    if (cif == nullptr) {
      LOG(FATAL) << "Custom Device [" << device_type
                 << "] does not support CINN (C_CinnInterface is null).";
    }

    // D. 创建并初始化插件
    auto plugin_ptr =
        std::make_unique<CinnCustomDevicePlugin>();  // 调用默认构造
    plugin_ptr->InitWrappers(cif);

    instances[device_type] = std::move(plugin_ptr);
  }

  return *instances[device_type];
}

// ============================================================
// CustomBackendAPI Implementation
// ============================================================

CustomBackendAPI* CustomBackendAPI::Global() {
  static CustomBackendAPI instance;
  return &instance;
}

void CustomBackendAPI::set_device(int device_id) {
  auto dev_types = phi::DeviceManager::GetAllCustomDeviceTypes();
  if (dev_types.empty()) {
    LOG(WARNING) << "No custom device types found when calling set_device.";
    return;
  }
  // Set the device for the first available custom device type
  // Note: CINN usually assumes one active backend type at a time
  phi::DeviceManager::SetDevice(dev_types[0], static_cast<size_t>(device_id));
}

int CustomBackendAPI::get_device() {
  auto dev_types = phi::DeviceManager::GetAllCustomDeviceTypes();
  if (dev_types.empty()) return 0;

  // Return the device ID of the current active device for this type
  return phi::DeviceManager::GetDevice(dev_types[0]);
}

int CustomBackendAPI::get_device_property(DeviceProperty device_property,
                                          std::optional<int> device_id) {
  auto dev_types = phi::DeviceManager::GetAllCustomDeviceTypes();
  if (dev_types.empty()) return 0;

  // Use current device ID if not provided
  size_t id = device_id.has_value() ? static_cast<size_t>(device_id.value())
                                    : static_cast<size_t>(get_device());
  std::string dev_type = dev_types[0];
  phi::Place place = phi::CustomPlace(dev_type, id);

  switch (device_property) {
    case DeviceProperty::MaxSharedMemoryPerBlock:
      return phi::DeviceManager::GetMaxSharedMemPerBlock(place);
    case DeviceProperty::MaxThreadsPerBlock:
      return phi::DeviceManager::GetMaxThreadsPerBlock(place);
    case DeviceProperty::MaxThreadsPerSM:
      return phi::DeviceManager::GetMaxThreadsPerMultiProcessor(place);
    case DeviceProperty::MultiProcessorCount:
      return phi::DeviceManager::GetMultiProcessors(place);
    case DeviceProperty::MaxBlocksPerSM:
      return phi::DeviceManager::GetMaxBlocksPerMultiProcessor(place);
    case DeviceProperty::MaxGridDimX:
      return phi::DeviceManager::GetMaxGridDimSize(place)[0];
    case DeviceProperty::MaxGridDimY:
      return phi::DeviceManager::GetMaxGridDimSize(place)[1];
    case DeviceProperty::MaxGridDimZ:
      return phi::DeviceManager::GetMaxGridDimSize(place)[2];
    case DeviceProperty::MaxBlockDimX:
      return phi::DeviceManager::GetMaxBlockDimSize(place)[0];
    case DeviceProperty::MaxBlockDimY:
      return phi::DeviceManager::GetMaxBlockDimSize(place)[1];
    case DeviceProperty::MaxBlockDimZ:
      return phi::DeviceManager::GetMaxBlockDimSize(place)[2];
    default:
      LOG(WARNING) << "Not supported device property: "
                   << static_cast<int>(device_property);
      return 0;
  }
}

void* CustomBackendAPI::malloc(size_t numBytes) {
  auto dev_types = phi::DeviceManager::GetAllCustomDeviceTypes();
  if (dev_types.empty()) return nullptr;

  int device_id = get_device();
  auto place = phi::CustomPlace(dev_types[0], device_id);

  // Use DeviceManager::GetDeviceWithPlace to access memory allocation
  return phi::DeviceManager::GetDeviceWithPlace(place)->MemoryAllocate(
      numBytes);
}

void CustomBackendAPI::free(void* data) {
  auto dev_types = phi::DeviceManager::GetAllCustomDeviceTypes();
  if (dev_types.empty()) return;

  int device_id = get_device();
  auto place = phi::CustomPlace(dev_types[0], device_id);

  // Note: Standard Device interface requires size for deallocation.
  // Since BackendAPI::free only provides the pointer, we might need a
  // workaround or rely on the specific device implementation ignoring the size
  // if possible, OR use a CINN-specific allocator that tracks sizes. For now,
  // we pass 0 as size, assuming underlying implementation handles it or CINN
  // fixes this API.
  phi::DeviceManager::GetDeviceWithPlace(place)->MemoryDeallocate(data, 0);
}

void CustomBackendAPI::memset(void* data, int value, size_t numBytes) {
  auto dev_types = phi::DeviceManager::GetAllCustomDeviceTypes();
  if (dev_types.empty()) return;

  int device_id = get_device();
  auto place = phi::CustomPlace(dev_types[0], device_id);

  // Device::MemorySet takes uint8_t value
  phi::DeviceManager::GetDeviceWithPlace(place)->MemorySet(
      data, static_cast<uint8_t>(value), numBytes);
}

void CustomBackendAPI::memcpy(void* dest,
                              const void* src,
                              size_t numBytes,
                              MemcpyType type) {
  auto dev_types = phi::DeviceManager::GetAllCustomDeviceTypes();
  if (dev_types.empty()) return;

  int device_id = get_device();
  auto place = phi::CustomPlace(dev_types[0], device_id);
  auto* device = phi::DeviceManager::GetDeviceWithPlace(place);

  // Map CINN MemcpyType to Phi Device methods
  switch (type) {
    case MemcpyType::HostToDevice:
      device->MemoryCopyH2D(dest, src, numBytes, nullptr);
      break;
    case MemcpyType::DeviceToHost:
      device->MemoryCopyD2H(dest, src, numBytes, nullptr);
      break;
    case MemcpyType::DeviceToDevice:
      device->MemoryCopyD2D(dest, src, numBytes, nullptr);
      break;
  }
}

void CustomBackendAPI::device_sync() {
  auto dev_types = phi::DeviceManager::GetAllCustomDeviceTypes();
  if (dev_types.empty()) return;

  int device_id = get_device();
  auto place = phi::CustomPlace(dev_types[0], device_id);

  phi::DeviceManager::SynchronizeDevice(place);
}

void CustomBackendAPI::stream_sync(void* stream) {
  auto dev_types = phi::DeviceManager::GetAllCustomDeviceTypes();
  if (dev_types.empty()) return;

  int device_id = get_device();
  auto place = phi::CustomPlace(dev_types[0], device_id);

  if (stream) {
    // Convert void* to phi::stream::stream_t (which is void*) and sync
    phi::DeviceManager::GetDeviceWithPlace(place)->SynchronizeStream(
        static_cast<phi::stream::stream_t>(stream));
  }
}

std::array<int, 3> CustomBackendAPI::get_max_grid_dims(
    std::optional<int> device_id) {
  auto dev_types = phi::DeviceManager::GetAllCustomDeviceTypes();
  if (dev_types.empty()) return {0, 0, 0};

  size_t id = device_id.has_value() ? static_cast<size_t>(device_id.value())
                                    : static_cast<size_t>(get_device());
  auto place = phi::CustomPlace(dev_types[0], id);

  auto dims = phi::DeviceManager::GetMaxGridDimSize(place);
  return {static_cast<int>(dims[0]),
          static_cast<int>(dims[1]),
          static_cast<int>(dims[2])};
}

std::array<int, 3> CustomBackendAPI::get_max_block_dims(
    std::optional<int> device_id) {
  auto dev_types = phi::DeviceManager::GetAllCustomDeviceTypes();
  if (dev_types.empty()) return {0, 0, 0};

  size_t id = device_id.has_value() ? static_cast<size_t>(device_id.value())
                                    : static_cast<size_t>(get_device());
  auto place = phi::CustomPlace(dev_types[0], id);

  auto dims = phi::DeviceManager::GetMaxBlockDimSize(place);
  return {static_cast<int>(dims[0]),
          static_cast<int>(dims[1]),
          static_cast<int>(dims[2])};
}

namespace {
    struct CinnCustomDeviceStaticInitializer {
        CinnCustomDeviceStaticInitializer() {
            // 这一行日志非常关键，证明库被加载了
            VLOG(0) << "!!! STATIC INIT: Triggering CINN Custom Device Registration !!!";
            
            // 强制执行注册
            ForceRegisterCinnCustomDeviceHostAPI();
            ForceRegisterCinnCustomDeviceIntrinsics();
            ForceRegisterCustomDeviceIntrinsicsReduce();
            ForceRegisterCustomDeviceIntrinsicsFloat16();
        }
    };

    // 定义一个静态实例，让它在库加载时自动构造
    static CinnCustomDeviceStaticInitializer __global_initializer_instance;
}

}  // namespace custom_device
}  // namespace runtime
}  // namespace cinn
#endif  // CINN_WITH_CUSTOM_DEVICE
