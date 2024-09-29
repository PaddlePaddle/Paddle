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

// Note: cinn::runtime::BackendAPI is a temporary implementation.
// It will be replaced by paddle::phi::backends after resolving circular
// dependencies.

#pragma once
#include <array>
#include <optional>
#include <variant>
#include "paddle/cinn/common/target.h"

namespace cinn {
namespace runtime {
class BackendAPI {
 public:
  BackendAPI() {}
  virtual ~BackendAPI() {}
  enum class MemcpyType : int {
    HostToHost = 0,
    HostToDevice,
    DeviceToHost,
    DeviceToDevice,
  };
  enum class DeviceProperty : int {
    MaxGridDimX = 0,
    MaxGridDimY,
    MaxGridDimZ,
    MaxBlockDimX,
    MaxBlockDimY,
    MaxBlockDimZ,
    MultiProcessorCount,
    MaxThreadsPerSM,
    MaxThreadsPerBlock,
    MaxBlocksPerSM,
    WarpSize,
    MaxSharedMemoryPerBlock,
  };
  /*!
   * \brief Get BackendAPI by target.
   * \param target
   * \return The corresponding BackendAPI.
   */
  static BackendAPI* get_backend(const common::Target target);
  /*!
   * \brief Get BackendAPI by arch.
   * \param arch
   * \return The corresponding BackendAPI.
   */
  static BackendAPI* get_backend(common::Arch arch);
  /*!
   * \brief Set device by device_id
   * \param device_id
   */
  virtual void set_device(int device_id) = 0;
  /*!
   * \brief get the current device_id
   * \return device_id
   */
  virtual int get_device() = 0;
  /*!
   * \brief Set active device by device_ids
   * \param device_ids
   */
  // virtual void set_active_devices(std::vector<int> device_ids) =0;
  /*!
   * \brief Get device property
   * \param device_property
   * \param device_id optional, default is now device which set by set_device.
   * \return result value
   */
  virtual int get_device_property(
      DeviceProperty device_property,
      std::optional<int> device_id = std::nullopt) = 0;
  /*!
   * \brief malloc memory in the idth device
   * \param numBytes
   * \param device_id
   * \return pointer to memory
   */
  virtual void* malloc(size_t numBytes) = 0;
  /*!
   * \brief free memory in the idth device
   * \param data pointer to memory
   * \param device_id
   */
  virtual void free(void* data) = 0;
  /*!
   * \brief  in the idth device
   * \param data pointer to memory
   * \param device_id
   */
  virtual void memset(void* data, int value, size_t numBytes) = 0;
  virtual void memcpy(void* dest,
                      const void* src,
                      size_t numBytes,
                      MemcpyType type) = 0;
  /*!
   * \brief synchronize now device
   */
  virtual void device_sync() = 0;
  /*!
   * \brief synchronize the stream
   */
  virtual void stream_sync(void* stream) = 0;
  virtual std::array<int, 3> get_max_grid_dims(
      std::optional<int> device_id = std::nullopt) = 0;
  virtual std::array<int, 3> get_max_block_dims(
      std::optional<int> device_id = std::nullopt) = 0;
};
}  // namespace runtime
}  // namespace cinn
