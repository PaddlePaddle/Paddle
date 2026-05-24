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

// fake_nvgpu_cinn_test.cc
//
// CI coverage test for the CINN CustomDevice code path using real NVIDIA GPU
// hardware accessed via dlopen (no WITH_GPU=ON required in Paddle build).
//
// What is exercised:
//   1. RegisterDevice        – CustomRuntimeParams fill + LoadCustomRuntimeLib
//   2. CinnCompilerToolchain – C_CinnInterface::compile (NVRTC → PTX file)
//   3. CinnRuntimeStrategy   – module_load / get_kernel_address / launch_kernel
//   4. CustomBackendAPI      – malloc / free / memcpy / device_sync /
//   stream_sync
//   5. DeviceInterface       – Allocate/Deallocate, MemCpy, Stream, Event
//   6. CinnCustomDevicePlugin – GetInstance, GetRuntime, GetToolchain
//
// The test uses a trivial __global__ kernel (elementwise float add) so that
// the entire compile→load→launch chain runs on real CUDA hardware while the
// CustomDevice abstraction layer is exercised end-to-end.
//
// Build requirements: WITH_CUSTOM_DEVICE=ON, WITH_CINN=ON (WITH_GPU=OFF)
// Runtime requirements: NVIDIA GPU + driver (libcuda.so, libcudart.so,
// libnvrtc.so)

#include <gtest/gtest.h>

#include <array>
#include <cstring>
#include <string>
#include <vector>

#include "paddle/phi/backends/context_pool.h"
#include "paddle/phi/backends/custom/fake_nvgpu_device.h"
#include "paddle/phi/backends/device_manager.h"
#include "paddle/phi/backends/event.h"
#include "paddle/phi/backends/stream.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/memory/allocation/allocator_facade.h"

#include "paddle/fluid/platform/init.h"

// CINN headers needed to exercise the plugin interface directly.
#include "paddle/cinn/runtime/custom_device/custom_device_backend_api.h"

namespace {

// ---------------------------------------------------------------------------
// Helper: register the FakeNVGPU device (idempotent in practice).
// ---------------------------------------------------------------------------
bool g_device_registered = false;

void RegisterFakeNVGPU() {
  if (g_device_registered) return;

  ASSERT_TRUE(FakeNVGPU_CudaAPI::Instance().Load())
      << "CUDA libraries (libcuda.so, libcudart.so, libnvrtc.so) not found. "
         "This test requires NVIDIA GPU + driver at runtime.";

  // Initialize memory subsystem (required before DeviceContextPool::Init).
  paddle::framework::InitMemoryMethod();

  CustomRuntimeParams runtime_params;
  memset(&runtime_params, 0, sizeof(CustomRuntimeParams));
  runtime_params.size = sizeof(CustomRuntimeParams);
  auto device_interface = std::make_unique<C_DeviceInterface>();
  runtime_params.interface = device_interface.get();
  std::memset(runtime_params.interface, 0, sizeof(C_DeviceInterface));
  runtime_params.interface->size = sizeof(C_DeviceInterface);

  InitFakeNVGPUDevice(&runtime_params);
  phi::LoadCustomRuntimeLib(
      runtime_params, std::move(device_interface), "", nullptr);

  // Init DeviceContextPool so SynchronizeDevice/CustomBackendAPI work.
  auto place = phi::CustomPlace(FAKE_NVGPU_DEVICE_TYPE, 0);
  std::vector<phi::Place> places = {place};
  phi::DeviceContextPool::Init(places);

  g_device_registered = true;
}

// Return the CustomPlace for device 0.
phi::Place FakeNVGPUPlace() {
  return phi::CustomPlace(FAKE_NVGPU_DEVICE_TYPE, 0);
}

}  // namespace

// ============================================================
// Test 1: Device registration and basic device manager queries
// ============================================================
TEST(FakeNVGPU, DeviceRegistration) {
  RegisterFakeNVGPU();

  auto dev_types = phi::DeviceManager::GetAllDeviceTypes();
  bool found = false;
  for (const auto& t : dev_types) {
    if (t == FAKE_NVGPU_DEVICE_TYPE) {
      found = true;
      break;
    }
  }
  EXPECT_TRUE(found) << "FakeNVGPU device type not registered";

  EXPECT_GT(phi::DeviceManager::GetDeviceCount(FAKE_NVGPU_DEVICE_TYPE), 0UL);

  auto place = FakeNVGPUPlace();
  auto* device = phi::DeviceManager::GetDeviceWithPlace(place);
  EXPECT_NE(device, nullptr);
}

// ============================================================
// Test 2: Device interface – SetDevice / GetDevice
// ============================================================
TEST(FakeNVGPU, SetGetDevice) {
  RegisterFakeNVGPU();

  auto place = FakeNVGPUPlace();
  phi::DeviceManager::SetDevice(place);
  int dev_id = phi::DeviceManager::GetDevice(FAKE_NVGPU_DEVICE_TYPE);
  EXPECT_EQ(dev_id, place.GetDeviceId());
}

// ============================================================
// Test 3: Device memory allocate / deallocate (via DeviceManager)
// ============================================================
TEST(FakeNVGPU, MemoryAllocateDeallocate) {
  RegisterFakeNVGPU();

  auto place = FakeNVGPUPlace();
  auto* device = phi::DeviceManager::GetDeviceWithPlace(place);
  ASSERT_NE(device, nullptr);

  constexpr size_t kSize = 4096;
  void* ptr = device->MemoryAllocate(kSize);
  EXPECT_NE(ptr, nullptr);
  device->MemoryDeallocate(ptr, kSize);
}

// ============================================================
// Test 4: H2D / D2H memory copy via Device interface
// ============================================================
TEST(FakeNVGPU, MemCopyH2DAndD2H) {
  RegisterFakeNVGPU();

  auto place = FakeNVGPUPlace();
  auto* device = phi::DeviceManager::GetDeviceWithPlace(place);
  ASSERT_NE(device, nullptr);

  constexpr int kN = 16;
  constexpr size_t kBytes = kN * sizeof(float);

  // Allocate device buffer
  void* d_buf = device->MemoryAllocate(kBytes);
  ASSERT_NE(d_buf, nullptr);

  // Prepare host data
  std::vector<float> h_src(kN), h_dst(kN, 0.0f);
  for (int i = 0; i < kN; ++i) h_src[i] = static_cast<float>(i);

  // H2D
  device->MemoryCopyH2D(d_buf, h_src.data(), kBytes);
  phi::DeviceManager::SynchronizeDevice(place);

  // D2H
  device->MemoryCopyD2H(h_dst.data(), d_buf, kBytes);
  phi::DeviceManager::SynchronizeDevice(place);

  for (int i = 0; i < kN; ++i) {
    EXPECT_FLOAT_EQ(h_dst[i], static_cast<float>(i))
        << "Mismatch at index " << i;
  }

  device->MemoryDeallocate(d_buf, kBytes);
}

// ============================================================
// Test 5: Stream create / sync / destroy
// ============================================================
TEST(FakeNVGPU, StreamLifecycle) {
  RegisterFakeNVGPU();

  auto place = FakeNVGPUPlace();

  phi::stream::Stream stream;
  stream.Init(place);
  EXPECT_NE(stream.raw_stream(), nullptr);

  stream.Synchronize();
  stream.Destroy();
}

// ============================================================
// Test 6: Event create / record / sync / destroy
// ============================================================
TEST(FakeNVGPU, EventLifecycle) {
  RegisterFakeNVGPU();

  auto place = FakeNVGPUPlace();

  phi::stream::Stream stream;
  stream.Init(place);

  phi::event::Event event;
  event.Init(place);

  event.Record(&stream);
  event.Synchronize();
  event.Destroy();
  stream.Destroy();
}

// ============================================================
// Test 7: Memory stats query
// ============================================================
TEST(FakeNVGPU, DeviceMemStats) {
  RegisterFakeNVGPU();

  auto place = FakeNVGPUPlace();

  size_t total = 0, free_mem = 0;
  phi::DeviceManager::MemoryStats(place, &total, &free_mem);
  EXPECT_GT(total, 0UL);
  EXPECT_GT(free_mem, 0UL);
  EXPECT_GE(total, free_mem);
}

// ============================================================
// Test 8: CinnCustomDevicePlugin initialisation
//         Exercises: GetInstance → device_base->GetCinnInterface → InitWrappers
// ============================================================
TEST(FakeNVGPU, CinnPluginInit) {
  RegisterFakeNVGPU();

  auto place = FakeNVGPUPlace();

  // GetInstance should not throw / LOG(FATAL).
  EXPECT_NO_THROW({
    auto& plugin =
        cinn::runtime::custom_device::CinnCustomDevicePlugin::GetInstance(
            place);
    EXPECT_NE(plugin.GetToolchain(), nullptr);
    EXPECT_NE(plugin.GetRuntime(), nullptr);
    EXPECT_NE(plugin.GetCompileStrategy(), nullptr);
  });
}

// ============================================================
// Test 9: End-to-end CINN path
//          compile() → module_load() → get_kernel_address() → launch_kernel()
//
// Kernel: elementwise float add, c[i] = a[i] + b[i], N=256
// ============================================================
TEST(FakeNVGPU, CinnCompileAndLaunch) {
  RegisterFakeNVGPU();

  auto place = FakeNVGPUPlace();

  // ---- 9a: obtain toolchain and runtime via CinnCustomDevicePlugin --------
  auto& plugin =
      cinn::runtime::custom_device::CinnCustomDevicePlugin::GetInstance(place);
  auto* toolchain = plugin.GetToolchain();
  auto* runtime = plugin.GetRuntime();
  ASSERT_NE(toolchain, nullptr);
  ASSERT_NE(runtime, nullptr);

  // ---- 9b: CUDA source code for the kernel --------------------------------
  const std::string kKernelSrc = R"cuda(
extern "C" __global__ void fake_add(const float* a,
                                     const float* b,
                                     float*       c,
                                     int          n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) c[idx] = a[idx] + b[idx];
}
)cuda";

  // ---- 9c: Compile CUDA source → PTX file ---------------------------------
  std::string ptx_path = toolchain->Compile(kKernelSrc);
  ASSERT_FALSE(ptx_path.empty()) << "NVRTC compile returned an empty path";

  // ---- 9d: Load module from PTX file ---------------------------------------
  auto module = runtime->LoadModule(ptx_path);
  ASSERT_NE(module, nullptr) << "cuModuleLoad failed for " << ptx_path;

  // ---- 9e: Get function pointer -------------------------------------------
  void* fn_ptr = module->GetFunction("fake_add");
  ASSERT_NE(fn_ptr, nullptr) << "cuModuleGetFunction('fake_add') failed";

  // ---- 9f: Allocate and initialise device buffers -------------------------
  constexpr int kN = 256;
  constexpr size_t kBytes = kN * sizeof(float);

  auto* device = phi::DeviceManager::GetDeviceWithPlace(place);
  ASSERT_NE(device, nullptr);

  std::vector<float> h_a(kN), h_b(kN), h_c(kN, 0.0f);
  for (int i = 0; i < kN; ++i) {
    h_a[i] = static_cast<float>(i);
    h_b[i] = static_cast<float>(i * 2);
  }

  void* d_a = device->MemoryAllocate(kBytes);
  void* d_b = device->MemoryAllocate(kBytes);
  void* d_c = device->MemoryAllocate(kBytes);
  ASSERT_NE(d_a, nullptr);
  ASSERT_NE(d_b, nullptr);
  ASSERT_NE(d_c, nullptr);

  device->MemoryCopyH2D(d_a, h_a.data(), kBytes);
  device->MemoryCopyH2D(d_b, h_b.data(), kBytes);

  // ---- 9g: Create stream and launch kernel --------------------------------
  phi::stream::Stream launch_stream;
  launch_stream.Init(place);
  phi::stream::stream_t raw_stream = launch_stream.raw_stream();

  // cuLaunchKernel expects void** where each element is a pointer-to-argument.
  int n_val = kN;
  void* args[] = {&d_a, &d_b, &d_c, &n_val};

  constexpr int kBlockSize = 256;
  constexpr int kGridSize = (kN + kBlockSize - 1) / kBlockSize;

  runtime->LaunchKernel(fn_ptr,
                        "fake_add",
                        args,
                        4,
                        kGridSize,
                        1,
                        1,
                        kBlockSize,
                        1,
                        1,
                        0,
                        raw_stream);

  launch_stream.Synchronize();

  // ---- 9h: Copy result back and verify ------------------------------------
  device->MemoryCopyD2H(h_c.data(), d_c, kBytes);

  for (int i = 0; i < kN; ++i) {
    float expected = static_cast<float>(i) + static_cast<float>(i * 2);
    EXPECT_FLOAT_EQ(h_c[i], expected) << "Wrong result at index " << i;
  }

  // ---- 9i: Cleanup --------------------------------------------------------
  launch_stream.Destroy();
  device->MemoryDeallocate(d_a, kBytes);
  device->MemoryDeallocate(d_b, kBytes);
  device->MemoryDeallocate(d_c, kBytes);
}

// ============================================================
// Test 10: CustomBackendAPI facade
//          Exercises the CINN-internal singleton API surface.
// ============================================================
TEST(FakeNVGPU, CustomBackendAPI) {
  RegisterFakeNVGPU();

  auto* api = cinn::runtime::custom_device::CustomBackendAPI::Global();
  ASSERT_NE(api, nullptr);

  // set/get device
  api->set_device(0);
  EXPECT_EQ(api->get_device(), 0);

  // device properties
  int sm_count =
      api->get_device_property(cinn::runtime::custom_device::CustomBackendAPI::
                                   DeviceProperty::MultiProcessorCount);
  EXPECT_GT(sm_count, 0);

  int max_threads =
      api->get_device_property(cinn::runtime::custom_device::CustomBackendAPI::
                                   DeviceProperty::MaxThreadsPerBlock);
  EXPECT_GT(max_threads, 0);

  // malloc / memset / free
  constexpr size_t kN = 64;
  void* ptr = api->malloc(kN * sizeof(float));
  ASSERT_NE(ptr, nullptr);

  api->memset(ptr, 0, kN * sizeof(float));

  // memcpy H2D
  std::vector<float> h_src(kN, 3.14f);
  api->memcpy(
      ptr,
      h_src.data(),
      kN * sizeof(float),
      cinn::runtime::custom_device::CustomBackendAPI::MemcpyType::HostToDevice);

  api->device_sync();
  api->free(ptr);

  // grid/block dims
  auto grid_dims = api->get_max_grid_dims();
  EXPECT_GT(grid_dims[0], 0);
  auto block_dims = api->get_max_block_dims();
  EXPECT_GT(block_dims[0], 0);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
