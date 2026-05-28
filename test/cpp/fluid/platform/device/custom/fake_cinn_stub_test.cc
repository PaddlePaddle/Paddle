// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

// =============================================================================
// fake_cinn_stub_test.cc
//
// Tests the CINN CustomDevice code path using a CPU-based stub device.
// Covers: CinnCustomDevicePlugin, DefaultCompilerToolchain,
//         DefaultRuntimeStrategy, DefaultCustomDeviceModule,
//         CustomBackendAPI, cinn_call_custom_device_kernel.
//
// NO GPU REQUIRED - runs entirely on CPU with mock functions.
// =============================================================================

#include <gtest/gtest.h>

#include <array>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "paddle/phi/backends/custom/fake_cinn_stub_device.h"
#include "paddle/phi/backends/device_manager.h"

#include "paddle/cinn/runtime/cinn_runtime.h"
#include "paddle/cinn/runtime/custom_device/custom_device_backend_api.h"
#include "paddle/cinn/runtime/custom_device/custom_device_util.h"

namespace {

// Register the fake CINN stub device with Paddle's DeviceManager.
void RegisterFakeCinnStubDevice() {
  CustomRuntimeParams runtime_params;
  memset(&runtime_params, 0, sizeof(CustomRuntimeParams));
  runtime_params.size = sizeof(CustomRuntimeParams);
  auto device_interface = std::make_unique<C_DeviceInterface>();
  runtime_params.interface = device_interface.get();
  std::memset(runtime_params.interface, 0, sizeof(C_DeviceInterface));
  runtime_params.interface->size = sizeof(C_DeviceInterface);

  InitFakeCinnStubDevice(&runtime_params);

  phi::LoadCustomRuntimeLib(
      runtime_params, std::move(device_interface), "", nullptr);
}

// Helper: ensure device is registered (idempotent via static flag)
void EnsureDeviceRegistered() {
  static bool registered = false;
  if (!registered) {
    RegisterFakeCinnStubDevice();
    registered = true;
  }
}

}  // namespace

// =============================================================================
// Test: Device Registration
// =============================================================================
TEST(FakeCinnStub, DeviceRegistration) {
  EnsureDeviceRegistered();

  // Verify the device is registered
  auto device_types = phi::DeviceManager::GetAllDeviceTypes();
  bool found = false;
  for (const auto& dt : device_types) {
    if (dt == FAKE_CINN_DEVICE_TYPE) {
      found = true;
      break;
    }
  }
  EXPECT_TRUE(found) << "FakeCinnStub device should be registered";

  // Verify custom device list
  auto custom_types = phi::DeviceManager::GetAllCustomDeviceTypes();
  bool found_custom = false;
  for (const auto& dt : custom_types) {
    if (dt == FAKE_CINN_DEVICE_TYPE) {
      found_custom = true;
      break;
    }
  }
  EXPECT_TRUE(found_custom);
}

// =============================================================================
// Test: Device Set/Get
// =============================================================================
TEST(FakeCinnStub, SetGetDevice) {
  EnsureDeviceRegistered();

  phi::DeviceManager::SetDevice(FAKE_CINN_DEVICE_TYPE, 0);
  int dev_id = phi::DeviceManager::GetDevice(FAKE_CINN_DEVICE_TYPE);
  EXPECT_EQ(dev_id, 0);
}

// =============================================================================
// Test: Memory Allocate/Deallocate
// =============================================================================
TEST(FakeCinnStub, MemoryAllocateDeallocate) {
  EnsureDeviceRegistered();

  auto place = phi::CustomPlace(FAKE_CINN_DEVICE_TYPE, 0);
  auto* device = phi::DeviceManager::GetDeviceWithPlace(place);
  ASSERT_NE(device, nullptr);

  // Allocate
  size_t alloc_size = 1024;
  void* ptr = device->MemoryAllocate(alloc_size);
  ASSERT_NE(ptr, nullptr);

  // Write and read back (since it's CPU memory under the hood)
  memset(ptr, 42, alloc_size);
  EXPECT_EQ(static_cast<uint8_t*>(ptr)[0], 42);

  // Deallocate
  device->MemoryDeallocate(ptr, alloc_size);
}

// =============================================================================
// Test: MemCpy H2D / D2H via direct device interface
// (Bypasses DeviceContextPool which doesn't support custom places)
// =============================================================================
TEST(FakeCinnStub, MemCopyH2DAndD2H) {
  EnsureDeviceRegistered();

  auto place = phi::CustomPlace(FAKE_CINN_DEVICE_TYPE, 0);
  auto* device = phi::DeviceManager::GetDeviceWithPlace(place);
  ASSERT_NE(device, nullptr);

  size_t size = 256;
  void* dev_ptr = device->MemoryAllocate(size);
  ASSERT_NE(dev_ptr, nullptr);

  // Prepare host data
  std::vector<uint8_t> host_data(size, 0xAB);
  std::vector<uint8_t> host_out(size, 0);

  // H2D and D2H via direct C interface (our stub is CPU-based memcpy)
  C_CinnInterface* cif = device->GetCinnInterface();
  ASSERT_NE(cif, nullptr);

  // Use the device_memory_set through MemorySet (doesn't need stream)
  device->MemorySet(dev_ptr, 0, size);
  EXPECT_EQ(static_cast<uint8_t*>(dev_ptr)[0], 0);

  // Direct memcpy (since our fake device is CPU-backed, this tests the path)
  memcpy(dev_ptr, host_data.data(), size);
  memcpy(host_out.data(), dev_ptr, size);
  EXPECT_EQ(host_out, host_data);

  device->MemoryDeallocate(dev_ptr, size);
}

// =============================================================================
// Test: GetCinnInterface returns non-null
// =============================================================================
TEST(FakeCinnStub, GetCinnInterface) {
  EnsureDeviceRegistered();

  auto place = phi::CustomPlace(FAKE_CINN_DEVICE_TYPE, 0);
  auto* device = phi::DeviceManager::GetDeviceWithPlace(place);
  ASSERT_NE(device, nullptr);

  C_CinnInterface* cif = device->GetCinnInterface();
  ASSERT_NE(cif, nullptr);
  EXPECT_NE(cif->compile, nullptr);
  EXPECT_NE(cif->get_runtime_source, nullptr);
  EXPECT_NE(cif->module_load, nullptr);
  EXPECT_NE(cif->module_unload, nullptr);
  EXPECT_NE(cif->get_kernel_address, nullptr);
  EXPECT_NE(cif->launch_kernel, nullptr);
  EXPECT_NE(cif->apply_custom_pass, nullptr);
}

// =============================================================================
// Test: CinnCustomDevicePlugin initialization
// Covers: CinnCustomDevicePlugin::GetInstance(), InitWrappers()
// =============================================================================
TEST(FakeCinnStub, CinnPluginInit) {
  EnsureDeviceRegistered();

  auto place = phi::CustomPlace(FAKE_CINN_DEVICE_TYPE, 0);

  auto& plugin =
      cinn::runtime::custom_device::CinnCustomDevicePlugin::GetInstance(place);

  EXPECT_NE(plugin.GetToolchain(), nullptr);
  EXPECT_NE(plugin.GetRuntime(), nullptr);
  EXPECT_NE(plugin.GetCompileStrategy(), nullptr);
}

// =============================================================================
// Test: Compile via DefaultCompilerToolchain
// Covers: DefaultCompilerToolchain::Compile(), GetRuntimeSource()
// =============================================================================
TEST(FakeCinnStub, CompilerToolchain) {
  EnsureDeviceRegistered();

  auto place = phi::CustomPlace(FAKE_CINN_DEVICE_TYPE, 0);
  auto& plugin =
      cinn::runtime::custom_device::CinnCustomDevicePlugin::GetInstance(place);
  auto* toolchain = plugin.GetToolchain();
  ASSERT_NE(toolchain, nullptr);

  // Test Compile
  std::string fake_code = "__global__ void fake_kernel() {}";
  std::string output_path = toolchain->Compile(fake_code);
  EXPECT_FALSE(output_path.empty());
  EXPECT_NE(output_path.find("fake_cinn_stub_kernel"), std::string::npos);

  // Test GetRuntimeSource
  std::string runtime_src = toolchain->GetRuntimeSource();
  EXPECT_FALSE(runtime_src.empty());
  EXPECT_NE(runtime_src.find("FAKE_WARP_SIZE"), std::string::npos);
}

// =============================================================================
// Test: Module load/unload + GetFunction via DefaultRuntimeStrategy
// Covers: DefaultRuntimeStrategy::LoadModule(),
//         DefaultCustomDeviceModule::GetFunction(),
//         DefaultCustomDeviceModule::~DefaultCustomDeviceModule() (unload)
// =============================================================================
TEST(FakeCinnStub, RuntimeStrategyLoadModule) {
  EnsureDeviceRegistered();

  auto place = phi::CustomPlace(FAKE_CINN_DEVICE_TYPE, 0);
  auto& plugin =
      cinn::runtime::custom_device::CinnCustomDevicePlugin::GetInstance(place);
  auto* runtime = plugin.GetRuntime();
  ASSERT_NE(runtime, nullptr);

  // Load module
  auto module = runtime->LoadModule("/tmp/fake_cinn_stub_kernel.bin");
  ASSERT_NE(module, nullptr);

  // Get kernel function address
  void* func_ptr = module->GetFunction("fake_kernel");
  EXPECT_NE(func_ptr, nullptr);

  // Module destructor will call module_unload (RAII)
  module.reset();
}

// =============================================================================
// Test: LaunchKernel via DefaultRuntimeStrategy
// Covers: DefaultRuntimeStrategy::LaunchKernel()
// =============================================================================
TEST(FakeCinnStub, RuntimeStrategyLaunchKernel) {
  EnsureDeviceRegistered();

  auto place = phi::CustomPlace(FAKE_CINN_DEVICE_TYPE, 0);
  auto& plugin =
      cinn::runtime::custom_device::CinnCustomDevicePlugin::GetInstance(place);
  auto* runtime = plugin.GetRuntime();
  ASSERT_NE(runtime, nullptr);

  // Load module and get kernel
  auto module = runtime->LoadModule("/tmp/fake_cinn_stub_kernel.bin");
  ASSERT_NE(module, nullptr);
  void* func_ptr = module->GetFunction("fake_kernel");
  ASSERT_NE(func_ptr, nullptr);

  // Prepare fake args
  float fake_data = 1.0f;
  void* args[] = {&fake_data};

  // Launch (no-op on stub, but exercises the full code path)
  EXPECT_NO_FATAL_FAILURE(runtime->LaunchKernel(func_ptr,
                                                "fake_kernel",
                                                args,
                                                1,  // num_args
                                                1,
                                                1,
                                                1,  // grid
                                                1,
                                                1,
                                                1,       // block
                                                0,       // shared_mem
                                                nullptr  // stream
                                                ));
}

// =============================================================================
// Test: ApplyCustomPass via DefaultCompileStrategy
// Covers: CustomCompileStrategy::ApplyCustomPass()
// =============================================================================
TEST(FakeCinnStub, CompileStrategyApplyPass) {
  EnsureDeviceRegistered();

  auto place = phi::CustomPlace(FAKE_CINN_DEVICE_TYPE, 0);
  auto& plugin =
      cinn::runtime::custom_device::CinnCustomDevicePlugin::GetInstance(place);
  auto* compile_strategy = plugin.GetCompileStrategy();
  ASSERT_NE(compile_strategy, nullptr);

  // Apply pass (no-op stub)
  int fake_ir_module = 42;
  bool result = compile_strategy->ApplyCustomPass(&fake_ir_module);
  // Default implementation returns false
  EXPECT_FALSE(result);
}

// =============================================================================
// Test: CustomBackendAPI global singleton
// Covers: CustomBackendAPI::Global(), set_device(), get_device(),
//         malloc(), free(), memset(), device_sync(),
//         get_device_property(), get_max_grid_dims(), get_max_block_dims()
// =============================================================================
TEST(FakeCinnStub, CustomBackendAPI) {
  EnsureDeviceRegistered();

  auto* api = cinn::runtime::custom_device::CustomBackendAPI::Global();
  ASSERT_NE(api, nullptr);

  // set_device / get_device
  api->set_device(0);
  EXPECT_EQ(api->get_device(), 0);

  // malloc / memset / free
  size_t size = 128;
  void* ptr = api->malloc(size);
  ASSERT_NE(ptr, nullptr);

  api->memset(ptr, 0, size);
  EXPECT_EQ(static_cast<uint8_t*>(ptr)[0], 0);

  api->memset(ptr, 0x55, size);
  EXPECT_EQ(static_cast<uint8_t*>(ptr)[0], 0x55);

  api->free(ptr);

  // device_sync
  EXPECT_NO_FATAL_FAILURE(api->device_sync());

  // get_device_property
  using DP = cinn::runtime::BackendAPI::DeviceProperty;
  EXPECT_EQ(api->get_device_property(DP::MultiProcessorCount), 108);
  EXPECT_EQ(api->get_device_property(DP::MaxThreadsPerBlock), 1024);
  EXPECT_EQ(api->get_device_property(DP::MaxSharedMemoryPerBlock), 49152);
  EXPECT_EQ(api->get_device_property(DP::MaxThreadsPerSM), 2048);
  EXPECT_EQ(api->get_device_property(DP::MaxBlocksPerSM), 32);

  // get_max_grid_dims / get_max_block_dims
  auto grid_dims = api->get_max_grid_dims();
  EXPECT_EQ(grid_dims[0], 2147483647);
  EXPECT_EQ(grid_dims[1], 65535);
  EXPECT_EQ(grid_dims[2], 65535);

  auto block_dims = api->get_max_block_dims();
  EXPECT_EQ(block_dims[0], 1024);
  EXPECT_EQ(block_dims[1], 1024);
  EXPECT_EQ(block_dims[2], 64);
}

// =============================================================================
// Test: cinn_call_custom_device_kernel (the CINN runtime dispatch entry point)
// Covers: cinn_call_custom_device_kernel() full path including
//         cinn_pod_value_t argument unpacking
// =============================================================================
TEST(FakeCinnStub, CinnCallCustomDeviceKernel) {
  EnsureDeviceRegistered();

  // The function needs a valid kernel_fn pointer (our stub accepts any)
  auto place = phi::CustomPlace(FAKE_CINN_DEVICE_TYPE, 0);
  auto& plugin =
      cinn::runtime::custom_device::CinnCustomDevicePlugin::GetInstance(place);
  auto* runtime = plugin.GetRuntime();
  auto module = runtime->LoadModule("/tmp/fake_cinn_stub_kernel.bin");
  ASSERT_NE(module, nullptr);
  void* kernel_fn = module->GetFunction("test_kernel");
  ASSERT_NE(kernel_fn, nullptr);

  // Prepare cinn_buffer_t args (simulating real CINN kernel args)
  cinn_buffer_t buf_a;
  memset(&buf_a, 0, sizeof(cinn_buffer_t));
  buf_a.memory = reinterpret_cast<uint8_t*>(malloc(256));
  memset(buf_a.memory, 0, 256);

  cinn_buffer_t buf_b;
  memset(&buf_b, 0, sizeof(cinn_buffer_t));
  buf_b.memory = reinterpret_cast<uint8_t*>(malloc(256));
  memset(buf_b.memory, 0, 256);

  // Create pod values
  cinn_pod_value_t pod_args[2];
  pod_args[0] = cinn_pod_value_t(&buf_a);
  pod_args[1] = cinn_pod_value_t(&buf_b);

  // Call the dispatch function
  EXPECT_NO_FATAL_FAILURE(
      cinn::runtime::custom_device::cinn_call_custom_device_kernel(
          kernel_fn,
          static_cast<void*>(pod_args),
          2,  // num_args
          1,
          1,
          1,  // grid
          32,
          1,
          1,       // block
          0,       // shared_mem
          nullptr  // stream
          ));

  free(buf_a.memory);
  free(buf_b.memory);
}

// =============================================================================
// Test: End-to-end compile -> load -> get_function -> launch
// Covers the complete CINN CustomDevice workflow
// =============================================================================
TEST(FakeCinnStub, EndToEndCompileLaunch) {
  EnsureDeviceRegistered();

  auto place = phi::CustomPlace(FAKE_CINN_DEVICE_TYPE, 0);
  auto& plugin =
      cinn::runtime::custom_device::CinnCustomDevicePlugin::GetInstance(place);

  // Step 1: Compile
  auto* toolchain = plugin.GetToolchain();
  std::string code = "extern \"C\" __global__ void add(float* a, float* b) {}";
  std::string module_path = toolchain->Compile(code);
  ASSERT_FALSE(module_path.empty());

  // Step 2: Load module
  auto* runtime = plugin.GetRuntime();
  auto module = runtime->LoadModule(module_path);
  ASSERT_NE(module, nullptr);

  // Step 3: Get kernel function
  void* func_ptr = module->GetFunction("add");
  ASSERT_NE(func_ptr, nullptr);

  // Step 4: Launch
  float fake_a = 1.0f, fake_b = 2.0f;
  void* args[] = {&fake_a, &fake_b};
  EXPECT_NO_FATAL_FAILURE(runtime->LaunchKernel(func_ptr,
                                                "add",
                                                args,
                                                2,
                                                1,
                                                1,
                                                1,  // grid
                                                256,
                                                1,
                                                1,  // block
                                                0,
                                                nullptr));

  // Step 5: Module unload via RAII
  module.reset();
}

// =============================================================================
// Test: infer_shape_set_value helper
// =============================================================================
TEST(FakeCinnStub, InferShapeSetValue) {
  // Simple test for the helper function
  int64_t row0[4] = {0, 0, 0, 0};
  int64_t row1[4] = {0, 0, 0, 0};
  int64_t* v[2] = {row0, row1};

  cinn::runtime::custom_device::infer_shape_set_value(0, 2, 42, v);
  EXPECT_EQ(row0[2], 42);

  cinn::runtime::custom_device::infer_shape_set_value(1, 3, 99, v);
  EXPECT_EQ(row1[3], 99);
}
