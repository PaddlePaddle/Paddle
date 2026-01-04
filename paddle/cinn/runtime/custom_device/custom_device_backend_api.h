// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#pragma once

#include <array>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "paddle/cinn/runtime/backend_api.h"
#include "paddle/phi/backends/device_ext.h"
#include "paddle/phi/common/place.h"

#ifdef CINN_WITH_CUSTOM_DEVICE
namespace cinn {
namespace runtime {
namespace custom_device {

// ============================================================
// 第一部分：CINN 编译与运行策略抽象接口
// ============================================================

// 1. 编译工具链接口：负责调用外部编译器 (如 mxcc)
class CustomCompilerToolchain {
 public:
  virtual ~CustomCompilerToolchain() = default;
  virtual std::string Compile(const std::string& code) = 0;
};

// 2. 运行时策略接口：负责加载和启动 Kernel
class CustomRuntimeStrategy {
 public:
  virtual ~CustomRuntimeStrategy() = default;
  virtual void* LoadModule(const std::string& path) = 0;
  virtual void LaunchKernel(void* module_handle,
                            const std::string& func_name,
                            void** args,
                            int num_args,
                            void* stream) = 0;
};

// 3. 编译优化接口：负责厂商自定义的 Fusion/Schedule/Pass
class CustomCompileStrategy {
 public:
  virtual ~CustomCompileStrategy() = default;
  virtual bool ApplyCustomPass(void* ir_module) { return false; }
  // 可以在这里增加 GetHeaderSource 等接口获取硬件特定头文件内容
};

// ============================================================
// 第二部分：插件管理类 (单例)
// ============================================================
// 4. 顶层插件管理类
class CinnCustomDevicePlugin {
 public:
  // 禁用构造，统一通过 GetInstance 访问
  CinnCustomDevicePlugin() = default;
  ~CinnCustomDevicePlugin() = default;

  // 按 Place 获取对应的单例插件实例
  static CinnCustomDevicePlugin& GetInstance(const phi::Place& place);

  // 暴露给 Compiler/Codegen 调用的包装接口
  CustomCompilerToolchain* GetToolchain() { return toolchain_.get(); }
  CustomRuntimeStrategy* GetRuntime() { return runtime_strategy_.get(); }
  CustomCompileStrategy* GetCompileStrategy() {
    return compile_strategy_.get();
  }

  // 内部初始化，由 .cc 中的 GetInstance 调用
  void InitWrappers(C_CinnInterface* cif);

 private:
  // 具体的包装器实例
  std::unique_ptr<CustomCompilerToolchain> toolchain_;
  std::unique_ptr<CustomRuntimeStrategy> runtime_strategy_;
  std::unique_ptr<CustomCompileStrategy> compile_strategy_;

  // 禁止拷贝
  CinnCustomDevicePlugin(const CinnCustomDevicePlugin&) = delete;
  CinnCustomDevicePlugin& operator=(const CinnCustomDevicePlugin&) = delete;
};

// ============================================================
// 第三部分：BackendAPI 实现 (核心运行时接口)
// ============================================================
class CustomBackendAPI final : public BackendAPI {
 public:
  CustomBackendAPI() = default;
  ~CustomBackendAPI() = default;

  // 全局访问点
  static CustomBackendAPI* Global();

  // --- 必须实现的虚函数 (来自 BackendAPI) ---
  void set_device(int device_id) override;
  int get_device() override;

  // 内存管理
  void* malloc(size_t numBytes) override;
  void free(void* data) override;
  void memset(void* data, int value, size_t numBytes) override;
  void memcpy(void* dest,
              const void* src,
              size_t numBytes,
              MemcpyType type) override;

  // 同步
  void device_sync() override;
  void stream_sync(void* stream) override;

  // 属性查询 (这些通常在 Target 中也有，但 Runtime 有时需要直接调用)
  int get_device_property(DeviceProperty device_property,
                          std::optional<int> device_id = std::nullopt) override;

  std::array<int, 3> get_max_grid_dims(
      std::optional<int> device_id = std::nullopt) override;
  std::array<int, 3> get_max_block_dims(
      std::optional<int> device_id = std::nullopt) override;
};

}  // namespace custom_device
}  // namespace runtime
}  // namespace cinn
#endif  // CINN_WITH_CUSTOM_DEVICE
