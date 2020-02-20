// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#ifndef _WIN32

#include "paddle/fluid/memory/allocation/mmap_allocator.h"

#include <fcntl.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <random>
#include <string>

namespace paddle {
namespace memory {
namespace allocation {

MemoryMapWriterAllocation::~MemoryMapWriterAllocation() {
  munmap(this->ptr(), this->size());
}

MemoryMapReaderAllocation::~MemoryMapReaderAllocation() {
  munmap(this->ptr(), this->size());
  shm_unlink(this->ipc_name().c_str());
}

std::string MemoryMapAllocator::GetIPCName() {
  static std::random_device rd;
  std::string handle = "/paddle_";
#ifdef _WIN32
  handle += std::to_string(GetCurrentProcessId());
#else
  handle += std::to_string(getpid());
#endif
  handle += "_";
  handle += std::to_string(rd());
  return handle;
}

std::shared_ptr<MemoryMapWriterAllocation> MemoryMapAllocator::Allocate(
    size_t size) {
  std::string ipc_name = GetIPCName();
  int flags = O_RDWR | O_CREAT;

  int fd = shm_open(ipc_name.c_str(), flags, 0644);
  PADDLE_ENFORCE_NE(
      fd, -1, platform::errors::Unavailable("File descriptor %s open failed",
                                            ipc_name.c_str()));
  PADDLE_ENFORCE_EQ(ftruncate(fd, size), 0,
                    platform::errors::Unavailable(
                        "Fruncate a file to a specified length failed!"));

  void* ptr = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  PADDLE_ENFORCE_NE(ptr, MAP_FAILED,
                    platform::errors::Unavailable(
                        "Memory map failed when create shared memory."));
  close(fd);

  return std::make_shared<MemoryMapWriterAllocation>(ptr, size, ipc_name);
}

std::shared_ptr<MemoryMapReaderAllocation> MemoryMapAllocator::Rebuild(
    std::string ipc_name, size_t size) {
  int fd = shm_open(ipc_name.c_str(), O_RDONLY, 0644);
  PADDLE_ENFORCE_NE(
      fd, -1, platform::errors::Unavailable("File descriptor %s open failed",
                                            ipc_name.c_str()));

  void* ptr = mmap(NULL, size, PROT_READ, MAP_SHARED, fd, 0);
  PADDLE_ENFORCE_NE(ptr, MAP_FAILED,
                    platform::errors::Unavailable(
                        "Memory map failed when rebuild shared memory."));
  close(fd);
  return std::make_shared<MemoryMapReaderAllocation>(ptr, size, ipc_name);
}

void MemoryMapAllocator::Free(MemoryMapWriterAllocation* mmap_allocation) {
  void* ptr = mmap_allocation->ptr();
  munmap(ptr, mmap_allocation->size());
  shm_unlink(mmap_allocation->ipc_name().c_str());
  delete mmap_allocation;
}

}  // namespace allocation
}  // namespace memory
}  // namespace paddle

#endif
