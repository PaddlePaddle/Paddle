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

class CustomModule {
 public:
  virtual ~CustomModule() = default;

  virtual void* GetFunction(const std::string& func_name) = 0;
};

namespace custom_device {

// ============================================================
// Part 1: CINN Compilation and Runtime Strategy Abstract Interfaces
// ============================================================

// Compiler Toolchain Interface: Responsible for invoking external compilers
class CustomCompilerToolchain {
 public:
  virtual ~CustomCompilerToolchain() = default;
  virtual std::string Compile(const std::string& code) = 0;
  virtual std::string GetRuntimeSource() = 0;
};

// Runtime Strategy Interface: Responsible for loading and launching Kernels
class CustomRuntimeStrategy {
 public:
  virtual ~CustomRuntimeStrategy() = default;
  virtual std::unique_ptr<cinn::runtime::CustomModule> LoadModule(
      const std::string& path) = 0;
  virtual void LaunchKernel(void* func_ptr,
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
                            void* stream) = 0;
};

// Compilation Optimization Interface: Responsible for vendor-specific
// Fusion/Schedule/Pass
class CustomCompileStrategy {
 public:
  virtual ~CustomCompileStrategy() = default;
  virtual bool ApplyCustomPass(void* ir_module) { return false; }
};

// ============================================================
// Part 2: Plugin Management Class
// ============================================================

/// @brief Top-level CINN plugin management class for a custom hardware device.
///
/// CinnCustomDevicePlugin acts as the single entry point through which CINN's
/// compiler and runtime subsystems access vendor-provided capabilities. Each
/// custom device type (identified by its phi::Place) owns exactly one
/// singleton instance.
///
/// A vendor plugin initialises the instance by calling InitWrappers() with a
/// C_CinnInterface pointer obtained from C_DeviceInterface::cinn_interface.
/// After initialisation, callers can retrieve the three strategy objects:
///
///   - CustomCompilerToolchain – invokes the vendor compiler to translate
///     device-specific IR/source into an executable module.
///   - CustomRuntimeStrategy   – loads a compiled module and dispatches
///     kernel launches to the device.
///   - CustomCompileStrategy   – applies vendor-specific fusion/schedule/pass
///     optimisations at the CINN IR level.
///
/// Usage example:
/// @code
///   auto& plugin = CinnCustomDevicePlugin::GetInstance(place);
///   plugin.GetToolchain()->Compile(source_code);
///   plugin.GetRuntime()->LaunchKernel(...);
/// @endcode
///
/// @note This class is non-copyable.  All three strategy objects are lazily
///       constructed during InitWrappers(); if the vendor does not implement
///       a given C hook the corresponding strategy pointer may be null.
class PADDLE_API CinnCustomDevicePlugin {
 public:
  CinnCustomDevicePlugin() = default;
  ~CinnCustomDevicePlugin() = default;

  /// @brief Returns the per-place singleton CinnCustomDevicePlugin instance.
  ///
  /// @param[in] place  The phi::Place identifying the custom device type and
  ///                   device index for which the plugin is requested.
  /// @return Reference to the singleton plugin for the given place.
  static CinnCustomDevicePlugin& GetInstance(const phi::Place& place);

  /// @brief Returns the compiler toolchain strategy for this device.
  ///
  /// The toolchain is responsible for invoking the vendor compiler and
  /// providing runtime source headers required by generated kernels.
  /// Returns nullptr if the vendor plugin does not implement the compile hook.
  CustomCompilerToolchain* GetToolchain() { return toolchain_.get(); }

  /// @brief Returns the runtime strategy for this device.
  ///
  /// The runtime strategy handles module loading and kernel launch dispatch.
  /// Returns nullptr if the vendor plugin does not implement runtime hooks.
  CustomRuntimeStrategy* GetRuntime() { return runtime_strategy_.get(); }

  /// @brief Returns the compile-time optimisation strategy for this device.
  ///
  /// The compile strategy allows vendors to inject custom IR passes (e.g.,
  /// hardware-specific fusion rules) into the CINN compilation pipeline.
  /// Returns nullptr if the vendor plugin does not implement the pass hook.
  CustomCompileStrategy* GetCompileStrategy() {
    return compile_strategy_.get();
  }

  /// @brief Initialises the three strategy wrappers from a C_CinnInterface.
  ///
  /// Called once during plugin registration (triggered by
  /// C_DeviceInterface::cinn_interface).  Constructs internal wrapper objects
  /// that delegate to the function pointers provided by @p cif.
  ///
  /// @param[in] cif  Pointer to the vendor-supplied C_CinnInterface struct.
  ///                 Must remain valid for the lifetime of this plugin
  ///                 instance.  Passing nullptr is a no-op.
  void InitWrappers(C_CinnInterface* cif);

 private:
  std::unique_ptr<CustomCompilerToolchain> toolchain_;
  std::unique_ptr<CustomRuntimeStrategy> runtime_strategy_;
  std::unique_ptr<CustomCompileStrategy> compile_strategy_;

  // Disable copying
  CinnCustomDevicePlugin(const CinnCustomDevicePlugin&) = delete;
  CinnCustomDevicePlugin& operator=(const CinnCustomDevicePlugin&) = delete;
};

// ============================================================
// Part 3: BackendAPI Implementation (Core Runtime Interface)
// ============================================================
class CustomBackendAPI final : public BackendAPI {
 public:
  CustomBackendAPI() = default;
  ~CustomBackendAPI() = default;

  // Global access point
  static CustomBackendAPI* Global();

  // Device Management
  void set_device(int device_id) override;
  int get_device() override;

  // Memory Management
  void* malloc(size_t numBytes) override;
  void free(void* data) override;
  void memset(void* data, int value, size_t numBytes) override;
  void memcpy(void* dest,
              const void* src,
              size_t numBytes,
              MemcpyType type) override;

  // Synchronization
  void device_sync() override;
  void stream_sync(void* stream) override;

  // Property Query
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
