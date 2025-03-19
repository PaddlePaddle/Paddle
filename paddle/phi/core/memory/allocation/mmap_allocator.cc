// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/core/memory/allocation/mmap_allocator.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <cstdlib>

#include <atomic>
#include <random>
#include <string>

#include "glog/logging.h"
#include "paddle/common/flags.h"
#include "paddle/phi/core/enforce.h"

COMMON_DECLARE_bool(use_shm_cache);

namespace paddle::memory::allocation {

std::string GetIPCName() {
  static std::random_device rd;
  static std::atomic<uint64_t> counter{0};
  std::string handle = "/paddle_";
#ifdef _WIN32
  handle += std::to_string(GetCurrentProcessId());
#else
  handle += std::to_string(getpid());
#endif
  handle += "_";
  handle += std::to_string(counter.fetch_add(1));
  handle += "_";
  handle += std::to_string(rd());
  return handle;
}

struct CountInfo {
  std::atomic<int> refcount;
};

void AllocateMemoryMap(std::string filename,
                       int *shared_fd,
                       int flags,
                       size_t size,
                       void **map_ptr_) {
  // TODO(@ZHUI): support win32
  int file_flags = 0;
  int fd = *shared_fd;
  if (flags & MAPPED_SHAREDMEM) {
    file_flags = O_RDWR | O_CREAT;
  } else {
    file_flags = O_RDONLY;
  }
  if (flags & MAPPED_EXCLUSIVE) {
    file_flags |= O_EXCL;
  }
  if (flags & MAPPED_NOCREATE) {
    file_flags &= ~O_CREAT;
  }

  if (!(flags & MAPPED_FROMFD) && fd == -1) {
    if (flags & MAPPED_SHAREDMEM) {
      fd = shm_open(filename.c_str(), file_flags, (mode_t)0600);
      PADDLE_ENFORCE_NE(
          fd,
          -1,
          common::errors::Unavailable(
              "File descriptor %s open failed, unable in read-write mode",
              filename.c_str()));
      VLOG(6) << "shm_open: " << filename;
      MemoryMapFdSet::Instance().Insert(filename);
    }
  }

  PADDLE_ENFORCE_EQ(ftruncate(fd, size),
                    0,
                    common::errors::Unavailable(
                        "Truncate a file to a specified length failed!"));

  if (flags & MAPPED_SHAREDMEM) {
    *map_ptr_ = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  } else {
    *map_ptr_ = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0);
  }

  if (flags & MAPPED_UNLINK) {
    VLOG(6) << "shm_unlink: " << filename;
    shm_unlink(filename.c_str());
  }

  PADDLE_ENFORCE_NE(*map_ptr_,
                    MAP_FAILED,
                    common::errors::Unavailable(
                        "Memory map failed when create shared memory."));
  if (flags & MAPPED_KEEPFD) {
    *shared_fd = fd;
    VLOG(6) << "keep fd: " << *shared_fd;
  } else {
    PADDLE_ENFORCE_NE(::close(fd),
                      -1,
                      common::errors::Unavailable(
                          "Error closing memory mapped file <", filename, ">"));

    *shared_fd = -1;
  }
}

std::shared_ptr<RefcountedMemoryMapAllocation>
AllocateRefcountedMemoryMapAllocation(std::string filename,
                                      int shared_fd,
                                      int flags,
                                      size_t size,
                                      int buffer_id) {
  int fd = shared_fd;
  void *base_ptr = nullptr;
  if (buffer_id == -1) {
    AllocateMemoryMap(filename, &fd, flags, size + mmap_alignment, &base_ptr);
    VLOG(4) << "Create and mmap a new shm: " << filename;
  } else {
    base_ptr = MemoryMapAllocationPool::Instance().GetById(buffer_id).mmap_ptr_;
    VLOG(4) << "Get a cached shm " << filename;
  }
  void *aligned_base_ptr =
      static_cast<void *>(static_cast<char *>(base_ptr) + mmap_alignment);
  return std::make_shared<RefcountedMemoryMapAllocation>(
      aligned_base_ptr, size, filename, fd, flags, buffer_id);
}

RefcountedMemoryMapAllocation::RefcountedMemoryMapAllocation(
    void *ptr,
    size_t size,
    std::string ipc_name,
    int fd,
    int flags,
    int buffer_id)
    : MemoryMapAllocation(ptr, size, ipc_name, fd, flags) {
  // must reset base ptr first.
  buffer_id_ = buffer_id;
  fd_ = fd;
  flags_ = flags;
  resetBaseptr();
  initializeRefercount();
}

void MemoryMapAllocation::close() {
  if (!closed_fd_) {
    closed_fd_ = true;
    if (flags_ & MAPPED_KEEPFD) {
      PADDLE_ENFORCE_NE(::close(fd_),
                        -1,
                        common::errors::Unavailable(
                            "Error closing file descriptor <", fd_, ">"));
    }
  }
  if (closed_) {
    return;
  }
  closed_ = true;
}

MemoryMapAllocation::~MemoryMapAllocation() { close(); }  // NOLINT

void RefcountedMemoryMapAllocation::incref() {
  CountInfo *info = static_cast<CountInfo *>(map_ptr_);
  ++info->refcount;
}

int RefcountedMemoryMapAllocation::decref() {
  CountInfo *info = static_cast<CountInfo *>(map_ptr_);
  return --info->refcount == 0;
}

void RefcountedMemoryMapAllocation::resetBaseptr() {
  map_ptr_ =
      static_cast<void *>(static_cast<char *>(map_ptr_) - mmap_alignment);
  map_size_ = map_size_ + mmap_alignment;
}

void RefcountedMemoryMapAllocation::initializeRefercount() {
  CountInfo *info = reinterpret_cast<CountInfo *>(map_ptr_);

  if (flags_ & MAPPED_EXCLUSIVE) {
    new (&info->refcount) std::atomic<int>(1);
  } else {
    info->refcount++;
  }
}

void RefcountedMemoryMapAllocation::close() {
  VLOG(4) << "Close a RefcountedMemoryMapAllocation: " << ipc_name_;
  if (closed_) {
    return;
  }
  closed_ = true;
  void *data = map_ptr_;
  CountInfo *info = reinterpret_cast<CountInfo *>(data);
  --info->refcount;
  if (flags_ & MAPPED_KEEPFD) {
    closed_fd_ = true;
    PADDLE_ENFORCE_NE(::close(fd_),
                      -1,
                      common::errors::Unavailable(
                          "Error closing file descriptor <", fd_, ">"));
    VLOG(6) << "close fd: " << fd_;
  }

  if (FLAGS_use_shm_cache && buffer_id_ != -1) {
    return;
  } else {
    if (FLAGS_use_shm_cache &&
        MemoryMapAllocationPool::Instance().BufferSize() <
            static_cast<size_t>(
                MemoryMapAllocationPool::Instance().MaxPoolSize())) {
      MemoryMapAllocationPool::Instance().Insert(MemoryMapInfo(
          flags_, map_size_ - mmap_alignment, ipc_name_, map_ptr_));
    } else {
      if (info->refcount == 0) {
        shm_unlink(ipc_name_.c_str());
        VLOG(6) << "shm_unlink file: " << ipc_name_;
      }

      PADDLE_ENFORCE_NE(munmap(map_ptr_, map_size_),
                        -1,
                        common::errors::Unavailable(
                            "could not unmap the shared memory file: ",
                            strerror(errno),
                            " (",
                            errno,
                            ")"));
    }
  }
}

MemoryMapWriterAllocation::~MemoryMapWriterAllocation() {
  if (munmap(this->ptr(), this->size()) == -1) {
    common::errors::Unavailable("could not unmap the shared memory file %s",
                                this->ipc_name());
  }
}

MemoryMapReaderAllocation::~MemoryMapReaderAllocation() {
  if (munmap(this->ptr(), this->size()) == -1) {
    common::errors::Unavailable("could not unmap the shared memory file %s",
                                this->ipc_name());
  }

  /* Here we do not pay attention to the result of shm_unlink,
     because the memory mapped file may have been cleared due to the
     MemoryMapFdSet::Clear() */

  // Code of DataLoader subprocess:
  //
  //    core._array_to_share_memory_tensor(b)
  //    out_queue.put((idx, tensor_list, structure))
  //    core._remove_tensor_list_mmap_fds(tensor_list)

  /* If the tensor in already in the send queue, the tensor will be
   * deconstructed by the function. If the tensor not send yet, it
   * will be cleared by MemoryMapFdSet::Clear().
   * If the `_remove_tensor_list_mmap_fds` have be interrupted, the
   * tensor will be cleared by both methods.
   * */

  shm_unlink(this->ipc_name().c_str());
  MemoryMapFdSet::Instance().Remove(this->ipc_name());
  VLOG(3) << "~MemoryMapReaderAllocation: " << this->ipc_name();
}

std::shared_ptr<MemoryMapWriterAllocation> AllocateMemoryMapWriterAllocation(
    size_t size) {
  const std::string &ipc_name = GetIPCName();
  int flags = O_RDWR | O_CREAT;
  int fd = shm_open(ipc_name.c_str(), flags, 0600);

  PADDLE_ENFORCE_NE(fd,
                    -1,
                    common::errors::Unavailable(
                        "File descriptor %s open failed", ipc_name.c_str()));
  PADDLE_ENFORCE_EQ(ftruncate(fd, size),
                    0,
                    common::errors::Unavailable(
                        "Truncate a file to a specified length failed!"));

  void *ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  PADDLE_ENFORCE_NE(ptr,
                    MAP_FAILED,
                    common::errors::Unavailable(
                        "Memory map failed when create shared memory."));
  close(fd);

  return std::make_shared<MemoryMapWriterAllocation>(ptr, size, ipc_name);
}

std::shared_ptr<MemoryMapReaderAllocation> RebuildMemoryMapReaderAllocation(
    const std::string &ipc_name, size_t size) {
  int flags = O_RDWR | O_CREAT;
  flags &= ~O_CREAT;
  int fd = shm_open(ipc_name.c_str(), flags, 0600);
  PADDLE_ENFORCE_NE(fd,
                    -1,
                    common::errors::Unavailable(
                        "File descriptor %s open failed", ipc_name.c_str()));
  void *ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  PADDLE_ENFORCE_NE(ptr,
                    MAP_FAILED,
                    common::errors::Unavailable(
                        "Memory map failed when rebuild shared memory."));
  close(fd);
  return std::make_shared<MemoryMapReaderAllocation>(ptr, size, ipc_name);
}

MemoryMapFdSet &MemoryMapFdSet::Instance() {  // NOLINT
  static MemoryMapFdSet set;
  return set;
}

void MemoryMapFdSet::Insert(const std::string &ipc_name) {
  std::lock_guard<std::mutex> guard(mtx_);
  fd_set_.emplace(ipc_name);
  VLOG(3) << "PID: " << getpid() << ", MemoryMapFdSet: insert " << ipc_name
          << ", set size: " << fd_set_.size();
}

void MemoryMapFdSet::Remove(const std::string &ipc_name) {
  std::lock_guard<std::mutex> guard(mtx_);
  fd_set_.erase(ipc_name);
  VLOG(3) << "PID: " << getpid() << ", MemoryMapFdSet: erase " << ipc_name
          << ", set size: " << fd_set_.size();
}

void MemoryMapFdSet::Clear() {
  VLOG(3) << "PID: " << getpid() << ", MemoryMapFdSet: set size - "
          << fd_set_.size();
  std::lock_guard<std::mutex> guard(mtx_);
  for (auto const &fd : fd_set_) {
    int rlt = shm_unlink(fd.c_str());
    if (rlt == 0) {
      VLOG(3) << "PID: " << getpid() << ", MemoryMapFdSet: clear " << fd;
    }
  }
  fd_set_.clear();
}

MemoryMapFdSet::~MemoryMapFdSet() { Clear(); }

MemoryMapAllocationPool *MemoryMapAllocationPool::pool_ = nullptr;

void MemoryMapAllocationPool::Insert(const MemoryMapInfo &memory_map) {
  std::lock_guard<std::mutex> guard(mtx_);
  memory_map_allocations_.push_back(memory_map);
  VLOG(4) << this << "Insert a new shm: " << memory_map.file_name_;
}

int MemoryMapAllocationPool::FindFromCache(const int &flag,
                                           const size_t &data_size,
                                           const std::string &file_name,
                                           bool check_refcount) {
  std::lock_guard<std::mutex> guard(mtx_);
  for (int idx = 0; idx < static_cast<int>(memory_map_allocations_.size());
       idx++) {
    if (memory_map_allocations_.at(idx).flags_ == flag &&
        memory_map_allocations_.at(idx).data_size_ == data_size) {
      if (file_name.empty() ||
          memory_map_allocations_.at(idx).file_name_ == file_name) {
        if (!check_refcount || reinterpret_cast<CountInfo *>(
                                   memory_map_allocations_.at(idx).mmap_ptr_)
                                       ->refcount == 0) {
          VLOG(4) << "Match at: " << idx;
          return idx;
        }
      }
    }
  }
  return -1;
}

const MemoryMapInfo &MemoryMapAllocationPool::GetById(int id) {
  std::lock_guard<std::mutex> guard(mtx_);
  return memory_map_allocations_.at(id);
}

void MemoryMapAllocationPool::SetMaxPoolSize(const int &size) {
  max_pool_size_ = size;
  VLOG(4) << this << "Set max pool size is: " << max_pool_size_;
}

void MemoryMapAllocationPool::Clear() {
  std::lock_guard<std::mutex> guard(mtx_);
  for (auto const &mmap : memory_map_allocations_) {
    int rlt = shm_unlink(mmap.file_name_.c_str());
    if (rlt == 0) {
      VLOG(4) << "MemoryMapAllocationPool: clear " << mmap.file_name_;
    }
    PADDLE_ENFORCE_NE(
        munmap(mmap.mmap_ptr_, mmap.data_size_ + mmap_alignment),
        -1,
        common::errors::Unavailable("could not unmap the shared memory file: ",
                                    strerror(errno),
                                    " (",
                                    errno,
                                    ")"));
  }
  memory_map_allocations_.clear();
}

MemoryMapAllocationPool::~MemoryMapAllocationPool() { Clear(); }  // NOLINT

}  // namespace paddle::memory::allocation

#endif
