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
// fake_cinn_compiler_test.cc
//
// Exercises CINN_WITH_CUSTOM_DEVICE code paths in:
//   1. paddle/cinn/ir/group_schedule/config/group_tile_config.cc
//      - BuildScheduleConfig() with CustomDeviceArch target
//   2. paddle/cinn/backends/custom_device/compiler_custom_device.cc
//      - cdrtc::Compiler operator() (invokes plugin Compile toolchain)
//   3. paddle/cinn/common/target.cc
//      - DefaultCustomDeviceTarget() and device property queries
//
// Uses the CPU-based FakeCinnStub device (no GPU required).
// =============================================================================

#include <gtest/gtest.h>

#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "paddle/phi/backends/custom/fake_cinn_stub_device.h"
#include "paddle/phi/backends/device_manager.h"

// CINN headers
#include "paddle/cinn/backends/compiler.h"
#include "paddle/cinn/backends/custom_device/codegen_custom_device_dev.h"
#include "paddle/cinn/backends/custom_device/compiler_custom_device.h"
#include "paddle/cinn/common/target.h"
#include "paddle/cinn/hlir/framework/pir/trivial_op_impl.h"
#include "paddle/cinn/ir/group_schedule/config/group_tile_config.h"
#include "paddle/cinn/ir/module.h"
#include "paddle/cinn/ir/stmt.h"
#include "paddle/cinn/runtime/custom_device/custom_device_backend_api.h"

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

void EnsureDeviceRegistered() {
  static bool registered = false;
  if (!registered) {
    RegisterFakeCinnStubDevice();
    registered = true;
  }
}

}  // namespace

// =============================================================================
// Test: DefaultCustomDeviceTarget
// Covers: paddle/cinn/common/target.cc CINN_WITH_CUSTOM_DEVICE
// =============================================================================
TEST(FakeCinnCompiler, DefaultCustomDeviceTarget) {
  EnsureDeviceRegistered();

  const auto& target = cinn::common::DefaultCustomDeviceTarget();

  // Should have CustomDeviceArch
  bool is_custom =
      target.arch.Match([](cinn::common::CustomDeviceArch) { return true; },
                        [](auto) { return false; });
  ASSERT_TRUE(is_custom) << "DefaultCustomDeviceTarget should have "
                            "CustomDeviceArch";

  // Should detect our registered device type
  std::string device_type;
  target.arch.Match(
      [&](cinn::common::CustomDeviceArch impl) {
        device_type = impl.device_type;
      },
      [](auto) {});
  EXPECT_EQ(device_type, FAKE_CINN_DEVICE_TYPE);

  // Target properties should reflect our stub device
  EXPECT_GT(target.max_num_threads(), 0);
  EXPECT_GT(target.get_multi_processor_count(), 0);
  EXPECT_GT(target.get_max_threads_per_sm(), 0);
  EXPECT_GT(target.get_max_blocks_per_sm(), 0);
}

// =============================================================================
// Test: BuildScheduleConfig with static reduce
// Covers: group_tile_config.cc CINN_WITH_CUSTOM_DEVICE branches in
//   GetWarpSize, GetMaxRegistersPerSM, BuildPureStaticShapeConfig,
//   CalculateWarpNums, UpdateWarpNumsInDifferentCase, register calc
// =============================================================================
TEST(FakeCinnCompiler, BuildScheduleConfig_StaticReduce) {
  EnsureDeviceRegistered();

  const auto& target = cinn::common::DefaultCustomDeviceTarget();

  // Create FusionGroupInfo: spatial=1024, reduce=512 (last dim = "R")
  auto group_info =
      std::make_shared<cinn::hlir::framework::pir::FusionGroupInfo>();
  group_info->loop_ranges = {1024, 512};
  group_info->loop_strides = {512, 1};
  group_info->reduce_axis = {1};
  group_info->reduce_var_name = {"rv0"};
  group_info->can_apply_grid_reduce = false;

  auto result = cinn::ir::BuildScheduleConfig(group_info, target);

  // Should produce at least one bucket config
  ASSERT_FALSE(result.empty())
      << "BuildScheduleConfig should return non-empty map";

  // Verify TileConfig values are reasonable
  for (const auto& [bucket, config] : result) {
    EXPECT_GT(config.tile_config.warp_num, 0);
    EXPECT_GT(config.tile_config.warp_size, 0);
    EXPECT_EQ(config.tile_config.warp_size, 32);  // Our stub returns 32
    EXPECT_GT(config.tile_config.tree_reduce_num, 0);
  }
}

// =============================================================================
// Test: BuildScheduleConfig with static spatial (dynamic reduce)
// Covers: group_tile_config.cc BuildStaticSpatialConfig path
// =============================================================================
TEST(FakeCinnCompiler, BuildScheduleConfig_DynamicReduce) {
  EnsureDeviceRegistered();

  const auto& target = cinn::common::DefaultCustomDeviceTarget();

  // Create FusionGroupInfo: spatial=1024, reduce=-1 (dynamic)
  auto group_info =
      std::make_shared<cinn::hlir::framework::pir::FusionGroupInfo>();
  group_info->loop_ranges = {1024, -1};
  group_info->loop_strides = {1, 1};
  group_info->reduce_axis = {1};
  group_info->reduce_var_name = {"rv0"};
  group_info->can_apply_grid_reduce = false;

  auto result = cinn::ir::BuildScheduleConfig(group_info, target);
  ASSERT_FALSE(result.empty())
      << "BuildScheduleConfig (dynamic reduce) should return non-empty map";
}

// =============================================================================
// Test: BuildScheduleConfig with dynamic spatial (static reduce)
// Covers: group_tile_config.cc BuildStaticReduceConfig path
// =============================================================================
TEST(FakeCinnCompiler, BuildScheduleConfig_DynamicSpatial) {
  EnsureDeviceRegistered();

  const auto& target = cinn::common::DefaultCustomDeviceTarget();

  // Create FusionGroupInfo: spatial=-1 (dynamic), reduce=256
  auto group_info =
      std::make_shared<cinn::hlir::framework::pir::FusionGroupInfo>();
  group_info->loop_ranges = {-1, 256};
  group_info->loop_strides = {256, 1};
  group_info->reduce_axis = {1};
  group_info->reduce_var_name = {"rv0"};
  group_info->can_apply_grid_reduce = false;

  auto result = cinn::ir::BuildScheduleConfig(group_info, target);
  ASSERT_FALSE(result.empty())
      << "BuildScheduleConfig (dynamic spatial) should return non-empty map";
}

// =============================================================================
// Test: BuildScheduleConfig with both dynamic
// Covers: group_tile_config.cc BuildDynamicShapeConfig path
// =============================================================================
TEST(FakeCinnCompiler, BuildScheduleConfig_BothDynamic) {
  EnsureDeviceRegistered();

  const auto& target = cinn::common::DefaultCustomDeviceTarget();

  auto group_info =
      std::make_shared<cinn::hlir::framework::pir::FusionGroupInfo>();
  group_info->loop_ranges = {-1, -1};
  group_info->loop_strides = {1, 1};
  group_info->reduce_axis = {1};
  group_info->reduce_var_name = {"rv0"};
  group_info->can_apply_grid_reduce = false;

  auto result = cinn::ir::BuildScheduleConfig(group_info, target);
  ASSERT_FALSE(result.empty())
      << "BuildScheduleConfig (both dynamic) should return non-empty map";
}

// =============================================================================
// Test: BuildScheduleConfig pure spatial (no reduce, last dim = "S")
// Covers: the "last_dim == S" branches in group_tile_config.cc
// =============================================================================
TEST(FakeCinnCompiler, BuildScheduleConfig_PureSpatial) {
  EnsureDeviceRegistered();

  const auto& target = cinn::common::DefaultCustomDeviceTarget();

  // Pure spatial, no reduce axis
  auto group_info =
      std::make_shared<cinn::hlir::framework::pir::FusionGroupInfo>();
  group_info->loop_ranges = {4096};
  group_info->loop_strides = {1};
  group_info->reduce_axis = {};
  group_info->reduce_var_name = {};
  group_info->can_apply_grid_reduce = false;

  auto result = cinn::ir::BuildScheduleConfig(group_info, target);
  ASSERT_FALSE(result.empty())
      << "BuildScheduleConfig (pure spatial) should return non-empty map";
}

// =============================================================================
// Test: cdrtc::Compiler (compiler_custom_device.cc)
// Covers: paddle/cinn/backends/custom_device/compiler_custom_device.cc
//         This is the CINN → plugin toolchain bridge
// =============================================================================
TEST(FakeCinnCompiler, CdrtcCompiler) {
  EnsureDeviceRegistered();

  const auto& target = cinn::common::DefaultCustomDeviceTarget();

  // cdrtc::Compiler wraps the plugin's Compile toolchain
  cinn::backends::cdrtc::Compiler compiler(target);

  // Pass a simple kernel source code
  std::string source = R"(
extern "C" __global__ void test_kernel(float* a, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) a[idx] = 1.0f;
}
)";

  // Call the compiler - this invokes CinnCustomDevicePlugin::GetInstance()
  // then plugin.GetToolchain()->Compile(code)
  std::string output_path = compiler(source);

  // FakeCinnStub's Compile writes to /tmp/fake_cinn_stub_kernel.bin
  ASSERT_FALSE(output_path.empty()) << "cdrtc::Compiler should return a path";
  EXPECT_NE(output_path.find("fake_cinn_stub"), std::string::npos)
      << "Output path should be from our fake device: " << output_path;
}

// =============================================================================
// Test: BuildScheduleConfig with grid reduce enabled
// Covers: the can_apply_grid_reduce branches with large reduce
// =============================================================================
TEST(FakeCinnCompiler, BuildScheduleConfig_GridReduce) {
  EnsureDeviceRegistered();

  const auto& target = cinn::common::DefaultCustomDeviceTarget();

  // Large reduce with grid reduce enabled
  auto group_info =
      std::make_shared<cinn::hlir::framework::pir::FusionGroupInfo>();
  group_info->loop_ranges = {256, 65536};
  group_info->loop_strides = {65536, 1};
  group_info->reduce_axis = {1};
  group_info->reduce_var_name = {"rv0"};
  group_info->can_apply_grid_reduce = true;

  auto result = cinn::ir::BuildScheduleConfig(group_info, target);
  ASSERT_FALSE(result.empty())
      << "BuildScheduleConfig (grid reduce) should return non-empty map";
}

// =============================================================================
// Test: BuildVectorizeConfig path (line 686)
// Covers: group_tile_config.cc BuildVectorizeConfig, CalculateWarpNums (320),
//   UpdateWarpNumsInDifferentCase (369/376/382), CheckPerformanceLimit (586),
//   GetMaxRegistersPerSM (73)
// =============================================================================
TEST(FakeCinnCompiler, BuildScheduleConfig_Vectorize_Reduce) {
  EnsureDeviceRegistered();

  const auto& target = cinn::common::DefaultCustomDeviceTarget();

  // Static [S, R] shape that triggers vectorization path
  // Need meet_vectorization_condition=true and has_if_else_op=true for line 369
  auto group_info =
      std::make_shared<cinn::hlir::framework::pir::FusionGroupInfo>();
  group_info->loop_ranges = {1024, 512};
  group_info->loop_strides = {512, 1};
  group_info->reduce_axis = {1};
  group_info->reduce_var_name = {"rv0"};
  group_info->can_apply_grid_reduce = false;

  // Enable vectorization
  group_info->vectorize_info.meet_vectorization_condition = true;
  group_info->vectorize_info.has_if_else_op = true;

  auto result = cinn::ir::BuildScheduleConfig(group_info, target);
  ASSERT_FALSE(result.empty())
      << "BuildScheduleConfig (vectorize reduce) should return non-empty map";
}

TEST(FakeCinnCompiler, BuildScheduleConfig_Vectorize_Spatial) {
  EnsureDeviceRegistered();

  const auto& target = cinn::common::DefaultCustomDeviceTarget();

  // Pure spatial [S] that triggers CalculateWarpNums (line 320)
  auto group_info =
      std::make_shared<cinn::hlir::framework::pir::FusionGroupInfo>();
  group_info->loop_ranges = {4096};
  group_info->loop_strides = {1};
  group_info->reduce_axis = {};
  group_info->reduce_var_name = {};
  group_info->can_apply_grid_reduce = false;

  // Enable vectorization for spatial path
  group_info->vectorize_info.meet_vectorization_condition = true;
  group_info->vectorize_info.has_if_else_op = false;

  auto result = cinn::ir::BuildScheduleConfig(group_info, target);
  // This might return empty if vectorize fails checks, but still covers
  // BuildVectorizeConfig + CalculateWarpNums + UpdateWarpNumsInDifferentCase
  // Even if empty, the code paths were exercised
  (void)result;
}

TEST(FakeCinnCompiler, BuildScheduleConfig_Vectorize_Broadcast) {
  EnsureDeviceRegistered();

  const auto& target = cinn::common::DefaultCustomDeviceTarget();

  // Pure spatial [S] with broadcast args → triggers line 376
  auto group_info =
      std::make_shared<cinn::hlir::framework::pir::FusionGroupInfo>();
  group_info->loop_ranges = {4096};
  group_info->loop_strides = {1};
  group_info->reduce_axis = {};
  group_info->reduce_var_name = {};
  group_info->can_apply_grid_reduce = false;

  group_info->vectorize_info.meet_vectorization_condition = true;
  group_info->vectorize_info.has_if_else_op = false;
  // Add broadcast axis info to trigger line 376
  group_info->vectorize_info.args_broadcast_axis_info["input_0"] = {{true}};

  auto result = cinn::ir::BuildScheduleConfig(group_info, target);
  (void)result;
}

// =============================================================================
// Test: BuildStaticSpatialConfig with grid reduce (line 990)
// Covers: group_tile_config.cc:990 (grid reduce path in static spatial)
// =============================================================================
TEST(FakeCinnCompiler, BuildScheduleConfig_StaticSpatial_GridReduce) {
  EnsureDeviceRegistered();

  const auto& target = cinn::common::DefaultCustomDeviceTarget();

  // Need: static spatial, dynamic reduce, last_dim == "R"
  // AND sm_count / sp_block_num > 1, so spatial_numel should be small
  // Our stub has sm_count=80, so spatial=4 → rd_block_num=FloorPow2(80/4)=16>1
  auto group_info =
      std::make_shared<cinn::hlir::framework::pir::FusionGroupInfo>();
  group_info->loop_ranges = {4, -1};
  group_info->loop_strides = {1, 1};
  group_info->reduce_axis = {1};
  group_info->reduce_var_name = {"rv0"};
  group_info->can_apply_grid_reduce = true;

  auto result = cinn::ir::BuildScheduleConfig(group_info, target);
  ASSERT_FALSE(result.empty())
      << "BuildScheduleConfig (static spatial grid reduce) should return "
         "non-empty map";
}

// =============================================================================
// Test: BuildStaticSpatialConfig with last_dim == "S" (line 1036)
// Covers: group_tile_config.cc:1036
// =============================================================================
TEST(FakeCinnCompiler, BuildScheduleConfig_StaticSpatial_LastDimS) {
  EnsureDeviceRegistered();

  const auto& target = cinn::common::DefaultCustomDeviceTarget();

  // Static spatial, dynamic reduce, last_dim == "S"
  // To get last_dim == "S": reduce must come before spatial in stride order
  // loop_ranges={128, -1}, strides={1, 128}, reduce_axis={1}
  // Sorted by stride desc: index1(stride128), index0(stride1)
  // index1 is reduce → "R", index0 is spatial → "S"
  // iter_space_type = [("R","dynamic"), ("S","static")], last_dim = "S"
  auto group_info =
      std::make_shared<cinn::hlir::framework::pir::FusionGroupInfo>();
  group_info->loop_ranges = {128, -1};
  group_info->loop_strides = {1, 128};
  group_info->reduce_axis = {1};
  group_info->reduce_var_name = {"rv0"};
  group_info->can_apply_grid_reduce = false;

  auto result = cinn::ir::BuildScheduleConfig(group_info, target);
  ASSERT_FALSE(result.empty())
      << "BuildScheduleConfig (static spatial last_dim S) should return "
         "non-empty map";
}

// =============================================================================
// Test: BuildStaticReduceConfig with last_dim == "S" (line 1179)
// Covers: group_tile_config.cc:1179
// =============================================================================
TEST(FakeCinnCompiler, BuildScheduleConfig_StaticReduce_LastDimS) {
  EnsureDeviceRegistered();

  const auto& target = cinn::common::DefaultCustomDeviceTarget();

  // Dynamic spatial, static reduce, last_dim == "S"
  // loop_ranges={-1, 128}, strides={1, 128}, reduce_axis={0}
  // Sorted by stride desc: index1(stride128), index0(stride1)
  // index1 is spatial → "S" (not in reduce_axis), index0 is reduce → "R"
  // Wait: reduce_axis={0} means index 0 is reduce.
  // After sorting: index1 first (stride 128, not reduce → "S"),
  //                index0 second (stride 1, reduce → "R")
  // iter_space_type = [("S","static"), ("R","dynamic")], last_dim = "R"
  // That's wrong. We need last_dim == "S".
  //
  // To get last_dim == "S" with dynamic spatial + static reduce:
  // loop_ranges={256, -1}, strides={1, 256}, reduce_axis={0}
  // Sorted: index1 (stride 256, spatial, dynamic), index0 (stride 1, reduce)
  // iter_space_type = [("S","dynamic"), ("R","static")], last_dim = "R"
  // Still wrong.
  //
  // Actually for last_dim == "S", the last (lowest stride) dimension must be
  // spatial. Example:
  // loop_ranges={-1, 64, 128}, strides={8192, 128, 1}, reduce_axis={0}
  // Sorted: index0(stride 8192,reduce), index1(stride128,spatial),
  //         index2(stride1,spatial)
  // iter_space_type = [("R","dynamic"), ("S","static")], last_dim = "S"
  // spatial_numel = 64*128 = 8192, reduce_numel = -1 (dynamic)
  // This gives us BuildStaticReduceConfig? NO - spatial is static here too.
  //
  // For BuildStaticReduceConfig: need spatial dynamic, reduce static.
  // loop_ranges = {-1, 128}, strides = {128, 1}, reduce_axis = {0}
  // Sorted: index0(stride128, reduce, dynamic), index1(stride1, spatial, 128)
  // iter_space_type = [("R","dynamic"), ("S","static")], last_dim = "S"
  // reduce_numel = -1 (dynamic), spatial_numel = 128 (static)
  // This should trigger BuildStaticReduceConfig (reduce is dynamic? NO)
  //
  // Actually let me re-read the dispatch logic:
  // BuildStaticSpatialConfig: spatial static, reduce dynamic
  // BuildStaticReduceConfig: spatial dynamic, reduce static
  // For BuildStaticReduceConfig with last_dim=="S":
  // Need reduce static (last element is in reduce_axis and static), spatial
  // dynamic, and the LAST dim in iter_space_type is "S".
  //
  // loop_ranges = {-1, 64}, strides = {64, 1}, reduce_axis = {0}
  // Sorted: index0 (stride64, reduce, dynamic → "R" "dynamic")
  //         index1 (stride1, spatial, 64 → "S" "static")
  // iter_space_type = [("R","dynamic"), ("S","static")]
  // spatial_numel = 64, reduce_numel = -1 → spatial=static, reduce=dynamic
  // → BuildStaticSpatialConfig (not what we want)
  //
  // Need: spatial dynamic AND reduce static AND last_dim == "S"
  // loop_ranges = {256, -1}, strides = {1, 256}, reduce_axis = {0}
  // Sorted: index1(stride256, spatial, dynamic), index0(stride1, reduce, 256)
  // iter_space_type = [("S","dynamic"), ("R","static")]
  // last_dim = "R"... still wrong.
  //
  // It seems like for last_dim == "S" we need spatial to be the lowest-stride
  // dimension AND not be merged with a reduce dim that follows.
  // loop_ranges = {256, -1}, strides = {256, 1}, reduce_axis = {0}
  // Sorted: index0(stride256, reduce, 256), index1(stride1, spatial, dynamic)
  // iter_space_type = [("R","static"), ("S","dynamic")]
  // spatial = dynamic, reduce = static, last_dim = "S" ✓
  // This triggers BuildStaticReduceConfig with last_dim == "S"!
  auto group_info =
      std::make_shared<cinn::hlir::framework::pir::FusionGroupInfo>();
  group_info->loop_ranges = {256, -1};
  group_info->loop_strides = {256, 1};
  group_info->reduce_axis = {0};
  group_info->reduce_var_name = {"rv0"};
  group_info->can_apply_grid_reduce = false;

  auto result = cinn::ir::BuildScheduleConfig(group_info, target);
  ASSERT_FALSE(result.empty())
      << "BuildScheduleConfig (static reduce last_dim S) should return "
         "non-empty map";
}

// =============================================================================
// Test: BuildDynamicShapeConfig with last_dim == "S" (line 1270)
// Covers: group_tile_config.cc:1270
// =============================================================================
TEST(FakeCinnCompiler, BuildScheduleConfig_BothDynamic_LastDimS) {
  EnsureDeviceRegistered();

  const auto& target = cinn::common::DefaultCustomDeviceTarget();

  // Both dynamic, last_dim == "S"
  // loop_ranges = {-1, -1}, strides = {256, 1}, reduce_axis = {0}
  // Sorted: index0(stride256, reduce, dynamic), index1(stride1, spatial,
  // dynamic)
  // iter_space_type = [("R","dynamic"), ("S","dynamic")], last_dim = "S"
  auto group_info =
      std::make_shared<cinn::hlir::framework::pir::FusionGroupInfo>();
  group_info->loop_ranges = {-1, -1};
  group_info->loop_strides = {256, 1};
  group_info->reduce_axis = {0};
  group_info->reduce_var_name = {"rv0"};
  group_info->can_apply_grid_reduce = false;

  auto result = cinn::ir::BuildScheduleConfig(group_info, target);
  ASSERT_FALSE(result.empty())
      << "BuildScheduleConfig (both dynamic last_dim S) should return "
         "non-empty map";
}

// =============================================================================
// Test: CodeGenCustomDevice (codegen_custom_device_dev.cc)
// Covers: paddle/cinn/backends/custom_device/codegen_custom_device_dev.cc
//   - CodeGenCustomDevice::PrintIncludes() → GetRuntimeSource from plugin
//   - This is the same codegen that compiler.cc line 357/737 calls
// =============================================================================
TEST(FakeCinnCompiler, CodeGenCustomDevice) {
  EnsureDeviceRegistered();

  const auto& target = cinn::common::DefaultCustomDeviceTarget();

  // Create codegen and directly call PrintIncludes() to exercise:
  //   - CodeGenCustomDevice::PrintIncludes() → DeviceManager →
  //     CinnCustomDevicePlugin → plugin.GetToolchain()->GetRuntimeSource()
  // Note: CodeGenGpuDev::Compile() does not call PrintIncludes() directly,
  // but this covers the codegen_custom_device_dev.cc code path.
  cinn::backends::custom_device::CodeGenCustomDevice codegen(target);

  // Also call Compile with empty module to exercise the base class path
  cinn::ir::Module::Builder builder("test_codegen_device", target);
  auto module = builder.Build();
  std::string source = codegen.Compile(module);

  // Compile produces the extern "C" wrapper even with no functions
  EXPECT_FALSE(source.empty()) << "Compile should produce non-empty output";

  // Now call PrintIncludes() directly - this is the key coverage target
  // It accesses DeviceManager, CinnCustomDevicePlugin, and GetRuntimeSource
  codegen.PrintIncludes();
}
