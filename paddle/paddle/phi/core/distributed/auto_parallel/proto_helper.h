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

#pragma once
#include "paddle/phi/core/distributed/auto_parallel/auto_parallel.pb.h"
namespace phi {
namespace distributed {
class TensorDistAttr;
class ProcessMesh;
namespace auto_parallel {
struct DeviceCapability;
class Device;
struct LinkCapability;
class Link;
class DeviceMesh;
class DistributedMapper;
}  // namespace auto_parallel
auto_parallel::TensorDistAttrProto to_proto(const TensorDistAttr& dist_attr);
auto_parallel::ProcessMeshProto to_proto(const ProcessMesh& dist_attr);

auto_parallel::DeviceCapabilityProto to_proto(
    const auto_parallel::DeviceCapability& device_capability);
auto_parallel::DeviceProto to_proto(const auto_parallel::Device& device);
auto_parallel::LinkCapabilityProto to_proto(
    const auto_parallel::LinkCapability& link_capability);
auto_parallel::LinkProto to_proto(const auto_parallel::Link& link);
auto_parallel::DeviceMeshProto to_proto(const auto_parallel::DeviceMesh& link);
auto_parallel::DistributedMapperProto to_proto(
    const auto_parallel::DistributedMapper& dist_mapper);

}  // namespace distributed
}  // namespace phi
