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

#include "paddle/phi/core/memory/allocation/mmap_allocator.h"

#include <stdint.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

#include <cstdlib>

#include <atomic>
#include <random>
#include <string>

#include "glog/logging.h"
#include "paddle/common/flags.h"
#include "paddle/phi/core/enforce.h"

#ifdef _WIN32
static inline int GetPid() { return static_cast<int>(GetCurrentProcessId()); }
#else
static inline int GetPid() { return getpid(); }
#endif

#ifdef _WIN32
#include <io.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#endif

COMMON_DECLARE_bool(use_shm_cache);

namespace paddle::memory::allocation {

std::string GetIPCName() {
  static std::random_device rd;
  static std::atomic<uint64_t> counter{0};
  std::string handle = "/paddle_";
#ifdef _WIN32
  handle += std::to_string(GetCurrentProcessId());
#else
  handle += std::to_string(GetPid());
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

void AllocateMemoryMap(std::string *fname_ptr,
                       intptr_t *shared_fd,
                       int flags,
                       size_t size,
                       void **map_ptr_) {
  std::string &fname = *fname_ptr;
#ifdef _WIN32
  DWORD protect = (flags & MAPPED_SHAREDMEM) ? PAGE_READWRITE : PAGE_READONLY;
  // For reader path (MAPPED_NOCREATE): open existing section, never create.
  if (flags & MAPPED_NOCREATE) {
    HANDLE hMap = OpenFileMappingA(FILE_MAP_ALL_ACCESS, FALSE, fname.c_str());
    PADDLE_ENFORCE_NE(hMap,
                      nullptr,
                      common::errors::Unavailable(
                          "OpenFileMappingA failed for %s, error: %lu",
                          fname.c_str(),
                          GetLastError()));
    void *ptr = MapViewOfFile(hMap, FILE_MAP_ALL_ACCESS, 0, 0, size);
    if (ptr == nullptr) {
      DWORD err = GetLastError();
      CloseHandle(hMap);
      PADDLE_THROW(common::errors::Unavailable(
          "MapViewOfFile for reader %s failed, error: %lu",
          fname.c_str(),
          err));
    }
    *shared_fd = reinterpret_cast<intptr_t>(hMap);
    *map_ptr_ = ptr;
    return;
  }
  // Retry loop for exclusive mode: on Windows, CreateFileMapping silently
  // returns the existing mapping if the name already exists, unlike Linux's
  // shm_open + O_EXCL which fails atomically. We must check explicitly.
  for (int attempt = 0; attempt < 100; attempt++) {
    HANDLE hMap = CreateFileMappingA(INVALID_HANDLE_VALUE,
                                     NULL,
                                     protect,
                                     static_cast<DWORD>(size >> 32),
                                     static_cast<DWORD>(size & 0xffffffffULL),
                                     fname.c_str());

    if (hMap == NULL) {
      DWORD err = GetLastError();
      // Write diagnostic to stderr before throwing
      fprintf(stderr,
              "[PADDLE_MMAP] PID=%lu CreateFileMapping FAILED "
              "name=%s size=%zu flags=0x%x attempt=%d err=%lu\n",
              GetCurrentProcessId(),
              fname.c_str(),
              size,
              flags,
              attempt,
              err);
      fflush(stderr);
      PADDLE_THROW(common::errors::Unavailable(
          "CreateFileMapping failed for %s, error: %lu", fname.c_str(), err));
    }

    if ((flags & MAPPED_EXCLUSIVE) && GetLastError() == ERROR_ALREADY_EXISTS) {
      CloseHandle(hMap);
      VLOG(3) << "[PADDLE_MMAP] PID=" << GetCurrentProcessId()
              << " name collision, retrying attempt=" << attempt
              << " name=" << fname;
      fname = GetIPCName();  // name collision; retry with fresh name
      continue;
    }

    DWORD access = FILE_MAP_ALL_ACCESS;
    *map_ptr_ = MapViewOfFile(hMap, access, 0, 0, size);
    if (*map_ptr_ == nullptr) {
      DWORD err = GetLastError();
      CloseHandle(hMap);
      fprintf(stderr,
              "[PADDLE_MMAP] PID=%lu MapViewOfFile FAILED "
              "name=%s size=%zu err=%lu\n",
              GetCurrentProcessId(),
              fname.c_str(),
              size,
              err);
      fflush(stderr);
      PADDLE_THROW(common::errors::Unavailable(
          "MapViewOfFile failed for %s, error: %lu", fname.c_str(), err));
    }

    // On Windows, always keep the HANDLE so the section stays alive
    // after the caller's MapViewOfFile. The handle is closed in
    // RefcountedMemoryMapAllocation::close(). Without this, the section
    // would be destroyed when the last view is unmapped (i.e. when the
    // worker's tensor is GC'd), before the reader opens it -- this is
    // the root cause of "Blocking queue is killed" with large data.
    // Linux doesn't have this issue because munmap never destroys the
    // shared memory file (only shm_unlink does).
    *shared_fd = reinterpret_cast<intptr_t>(hMap);

    if (flags & MAPPED_UNLINK) {
      VLOG(6) << "CreateFileMapping (unlink mode): " << fname;
    }

    // Caller (AllocateRefcountedMemoryMapAllocation) handles Insert
    return;
  }

  PADDLE_THROW(common::errors::Unavailable(
      "Failed to allocate exclusive shared memory after 100 retries"));
#else
  // Linux implementation using shm_open + mmap
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
      fd = shm_open(fname.c_str(), file_flags, (mode_t)0600);
      PADDLE_ENFORCE_NE(
          fd,
          -1,
          common::errors::Unavailable(
              "File descriptor %s open failed, unable in read-write mode",
              fname.c_str()));
      VLOG(6) << "shm_open: " << fname;
      MemoryMapFdSet::Instance().Insert(fname);
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
    VLOG(6) << "shm_unlink: " << fname;
    shm_unlink(fname.c_str());
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
                          "Error closing memory mapped file %s", fname));
    *shared_fd = -1;
  }
#endif
}

std::shared_ptr<RefcountedMemoryMapAllocation>
AllocateRefcountedMemoryMapAllocation(std::string fname,
                                      intptr_t shared_fd,
                                      int flags,
                                      size_t size,
                                      int buffer_id) {
  intptr_t fd = shared_fd;
  void *base_ptr = nullptr;
  if (buffer_id == -1) {
    AllocateMemoryMap(&fname, &fd, flags, size + mmap_alignment, &base_ptr);
    VLOG(4) << "Create and mmap a new shm: " << fname;
  } else {
    base_ptr = MemoryMapAllocationPool::Instance().GetById(buffer_id).mmap_ptr_;
    fd = shared_fd;
    VLOG(4) << "Get a cached shm " << fname;
  }
  void *aligned_base_ptr =
      static_cast<void *>(static_cast<char *>(base_ptr) + mmap_alignment);
  MemoryMapFdSet::Instance().Insert(fname);
  return std::make_shared<RefcountedMemoryMapAllocation>(
      aligned_base_ptr, size, fname, fd, flags, buffer_id);
}

RefcountedMemoryMapAllocation::RefcountedMemoryMapAllocation(
    void *ptr,
    size_t size,
    std::string ipc_name,
    intptr_t fd,
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
#ifdef _WIN32
    // On Windows, the HANDLE from CreateFileMapping is always kept
    // (never closed in AllocateMemoryMap). Close it unconditionally
    // to prevent handle leaks.
    if (fd_ != -1) {
      CloseHandle(reinterpret_cast<HANDLE>(fd_));
    }
#else
    if (flags_ & MAPPED_KEEPFD) {
      PADDLE_ENFORCE_NE(::close(fd_),
                        -1,
                        common::errors::Unavailable(
                            "Error closing file descriptor <%d>", fd_));
    }
#endif
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

  if (FLAGS_use_shm_cache && buffer_id_ != -1) {
#ifdef _WIN32
    // Prevent base class MemoryMapAllocation::close() from closing the HANDLE
    // (which would destroy the named section before the reader opens it).
    closed_fd_ = true;
#endif
    return;
  } else {
    if (FLAGS_use_shm_cache &&
        MemoryMapAllocationPool::Instance().BufferSize() <
            static_cast<size_t>(
                MemoryMapAllocationPool::Instance().MaxPoolSize())) {
      MemoryMapAllocationPool::Instance().Insert(MemoryMapInfo(
          flags_, map_size_ - mmap_alignment, ipc_name_, map_ptr_));
#ifdef _WIN32
      // Prevent base class destructor from closing the HANDLE.
      // When the buffer is later retrieved, a new RefcountedMMap is
      // constructed from the same map_ptr_ and increments refcount — the
      // HANDLE must stay alive so the named section survives until then.
      closed_fd_ = true;
#endif
    } else {
      if (info->refcount == 0) {
#ifdef _WIN32
        // Only close the handle AND unmap when refcount reaches 0.
        // On Windows, CloseHandle destroys the named file mapping section.
        // Closing it early (while refcount > 0) would delete the section
        // before the reader process has a chance to OpenFileMappingA it.
        if (fd_ != -1 && !closed_fd_) {
          closed_fd_ = true;
          CloseHandle(reinterpret_cast<HANDLE>(fd_));
          VLOG(6) << "close handle: " << fd_;
        }
        VLOG(6) << "UnmapViewOfFile: " << ipc_name_;
        UnmapViewOfFile(map_ptr_);
#else
        if (flags_ & MAPPED_KEEPFD) {
          closed_fd_ = true;
          PADDLE_ENFORCE_NE(::close(fd_),
                            -1,
                            common::errors::Unavailable(
                                "Error closing file descriptor <%d>", fd_));
          VLOG(6) << "close fd: " << fd_;
        }
        shm_unlink(ipc_name_.c_str());
        VLOG(6) << "shm_unlink file: " << ipc_name_;
#endif
      } else {
#ifdef _WIN32
        if (info->refcount > 1) {
          // Refcount > 1 means the reader has already opened this section
          // (initializeRefercount was called after MapViewOfFile). The
          // reader's mapping keeps the section alive, so we can safely
          // CloseHandle immediately — no accumulation needed.
          VLOG(6) << "close handle (reader already opened): " << ipc_name_;
          CloseHandle(reinterpret_cast<HANDLE>(fd_));
          fd_ = -1;
          closed_fd_ = true;
          VLOG(6) << "UnmapViewOfFile: " << ipc_name_;
          UnmapViewOfFile(map_ptr_);
        } else {
          // Refcount == 1: reader hasn't opened yet. Transfer both HANDLE
          // and map_ptr_ to WindowsHandleKeeper so the named section stays
          // alive until the reader opens it. The keeper will UnmapViewOfFile
          // + CloseHandle when refcount reaches 0 (via SweepClosedMappings).
          WindowsHandleKeeper::Instance().Insert(
              ipc_name_, fd_, map_ptr_, map_size_);
          fd_ = -1;
          closed_fd_ = true;
        }
#endif
      }
#ifndef _WIN32
      // On Linux, munmap is always safe since it only unmaps virtual memory;
      // the shared memory file in /dev/shm persists until shm_unlink.
      PADDLE_ENFORCE_NE(munmap(map_ptr_, map_size_),
                        -1,
                        common::errors::Unavailable(
                            "could not unmap the shared memory file: %s (%d)",
                            strerror(errno),
                            errno));
#endif
    }
  }
}

MemoryMapWriterAllocation::~MemoryMapWriterAllocation() {
#ifdef _WIN32
  UnmapViewOfFile(this->ptr());
  if (fd_ != -1) {
    CloseHandle(reinterpret_cast<HANDLE>(fd_));
  }
#else
  if (munmap(this->ptr(), this->size()) == -1) {
    common::errors::Unavailable("could not unmap the shared memory file %s",
                                this->ipc_name());
  }
#endif
}

MemoryMapReaderAllocation::~MemoryMapReaderAllocation() {
#ifdef _WIN32
  UnmapViewOfFile(this->ptr());
  // On Windows, named file mapping is auto-destroyed when final handle is
  // closed.
  MemoryMapFdSet::Instance().Remove(this->ipc_name());
  VLOG(3) << "~MemoryMapReaderAllocation: " << this->ipc_name();
#else
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
#endif
}

std::shared_ptr<MemoryMapWriterAllocation> AllocateMemoryMapWriterAllocation(
    size_t size) {
  const std::string &ipc_name = GetIPCName();
#ifdef _WIN32
  HANDLE hMap = CreateFileMappingA(INVALID_HANDLE_VALUE,
                                   NULL,
                                   PAGE_READWRITE,
                                   static_cast<DWORD>(size >> 32),
                                   static_cast<DWORD>(size & 0xffffffffULL),
                                   ipc_name.c_str());
  PADDLE_ENFORCE_NE(
      hMap,
      nullptr,
      common::errors::Unavailable("CreateFileMapping for writer %s failed",
                                  ipc_name.c_str()));

  void *ptr = MapViewOfFile(hMap, FILE_MAP_ALL_ACCESS, 0, 0, size);
  if (ptr == nullptr) {
    DWORD err = GetLastError();
    CloseHandle(hMap);
    PADDLE_THROW(common::errors::Unavailable(
        "MapViewOfFile for writer %s failed, error: %lu",
        ipc_name.c_str(),
        err));
  }
  // Keep the HANDLE open until ~MemoryMapWriterAllocation (where
  // UnmapViewOfFile runs first, then CloseHandle). Closing it here
  // would destroy the named section as soon as the writer unmaps,
  // before the reader has a chance to OpenFileMappingA it.
  return std::make_shared<MemoryMapWriterAllocation>(
      ptr, size, ipc_name, reinterpret_cast<intptr_t>(hMap));
#else
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
#endif
}

std::shared_ptr<MemoryMapReaderAllocation> RebuildMemoryMapReaderAllocation(
    const std::string &ipc_name, size_t size) {
#ifdef _WIN32
  HANDLE hMap = OpenFileMappingA(FILE_MAP_ALL_ACCESS, FALSE, ipc_name.c_str());
  PADDLE_ENFORCE_NE(hMap,
                    nullptr,
                    common::errors::Unavailable(
                        "OpenFileMappingA for reader %s failed, error: %lu",
                        ipc_name.c_str(),
                        GetLastError()));

  void *ptr = MapViewOfFile(hMap, FILE_MAP_ALL_ACCESS, 0, 0, size);
  if (ptr == nullptr) {
    DWORD err = GetLastError();
    CloseHandle(hMap);
    PADDLE_THROW(common::errors::Unavailable(
        "MapViewOfFile for reader %s failed, error: %lu",
        ipc_name.c_str(),
        err));
  }
  CloseHandle(hMap);
  return std::make_shared<MemoryMapReaderAllocation>(ptr, size, ipc_name);
#else
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
#endif
}

MemoryMapFdSet &MemoryMapFdSet::Instance() {  // NOLINT
  static MemoryMapFdSet set;
  return set;
}

void MemoryMapFdSet::Insert(const std::string &ipc_name) {
  std::lock_guard<std::mutex> guard(mtx_);
  fd_set_.emplace(ipc_name);
  VLOG(3) << "PID: " << GetPid() << ", MemoryMapFdSet: insert " << ipc_name
          << ", set size: " << fd_set_.size();
}

void MemoryMapFdSet::Remove(const std::string &ipc_name) {
  std::lock_guard<std::mutex> guard(mtx_);
  fd_set_.erase(ipc_name);
  VLOG(3) << "PID: " << GetPid() << ", MemoryMapFdSet: erase " << ipc_name
          << ", set size: " << fd_set_.size();
}

void MemoryMapFdSet::Clear() {
  VLOG(7) << "PID: " << GetPid() << ", MemoryMapFdSet: set size - "
          << fd_set_.size();
  std::lock_guard<std::mutex> guard(mtx_);
  for (auto const &fd : fd_set_) {
#ifdef _WIN32
    // On Windows, named file mappings are auto-destroyed when the last view
    // is unmapped. Nothing to unlink explicitly.
    VLOG(7) << "PID: " << GetPid() << ", MemoryMapFdSet: clear " << fd;
#else
    int rlt = shm_unlink(fd.c_str());
    if (rlt == 0) {
      VLOG(7) << "PID: " << GetPid() << ", MemoryMapFdSet: clear " << fd;
    }
#endif
  }
  fd_set_.clear();
#ifdef _WIN32
  // Close pending HANDLEs that were kept open for cross-process section
  // lifecycle (refcount > 0 path in RefcountedMemoryMapAllocation::close).
  WindowsHandleKeeper::Instance().CloseAll();
#endif
}

MemoryMapFdSet::~MemoryMapFdSet() { Clear(); }

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
#ifdef _WIN32
    VLOG(4) << "MemoryMapAllocationPool: clear " << mmap.file_name_;
    UnmapViewOfFile(mmap.mmap_ptr_);
#else
    int rlt = shm_unlink(mmap.file_name_.c_str());
    if (rlt == 0) {
      VLOG(4) << "MemoryMapAllocationPool: clear " << mmap.file_name_;
    }
    PADDLE_ENFORCE_NE(munmap(mmap.mmap_ptr_, mmap.data_size_ + mmap_alignment),
                      -1,
                      common::errors::Unavailable(
                          "could not unmap the shared memory file: %s (%d)",
                          strerror(errno),
                          errno));
#endif
  }
  memory_map_allocations_.clear();
}

MemoryMapAllocationPool::~MemoryMapAllocationPool() { Clear(); }  // NOLINT

#ifdef _WIN32
WindowsHandleKeeper &WindowsHandleKeeper::Instance() {
  // Leaky singleton: heap-allocated, never destructed, to avoid static
  // destruction ordering conflicts with MemoryMapFdSet. Normal cleanup
  // is driven by MemoryMapFdSet::Clear() / _cleanup_mmap_fds(); on
  // abnormal process exit the OS reclaims the memory.
  static auto *keeper = new WindowsHandleKeeper();
  return *keeper;
}

void WindowsHandleKeeper::Insert(const std::string &ipc_name,
                                 intptr_t fd,
                                 void *map_ptr,
                                 size_t map_size) {
  std::lock_guard<std::mutex> lock(mtx_);
  SweepClosedMappingsLocked();
  handles_[ipc_name] = {fd, map_ptr};
}

void WindowsHandleKeeper::SweepClosedMappingsLocked() {
  for (auto it = handles_.begin(); it != handles_.end();) {
    // CountInfo::refcount is the first field at map_ptr. A value of 0 means
    // all references (including the reader's) have been released — the
    // section is no longer in use and can be cleaned up.
    auto *refcnt =
        static_cast<volatile std::atomic<uint32_t> *>(it->second.map_ptr);
    if (refcnt->load(std::memory_order_acquire) == 0) {
      VLOG(6) << "WindowsHandleKeeper sweeping: " << it->first;
      UnmapViewOfFile(it->second.map_ptr);
      CloseHandle(reinterpret_cast<HANDLE>(it->second.fd));
      it = handles_.erase(it);
    } else {
      ++it;
    }
  }
}

void WindowsHandleKeeper::SweepClosedMappings() {
  std::lock_guard<std::mutex> lock(mtx_);
  SweepClosedMappingsLocked();
}

void WindowsHandleKeeper::CloseAll() {
  std::lock_guard<std::mutex> lock(mtx_);
  for (auto &pair : handles_) {
    if (pair.second.fd != -1) {
      UnmapViewOfFile(pair.second.map_ptr);
      CloseHandle(reinterpret_cast<HANDLE>(pair.second.fd));
    }
  }
  handles_.clear();
}

WindowsHandleKeeper::~WindowsHandleKeeper() { CloseAll(); }  // NOLINT
#endif

}  // namespace paddle::memory::allocation
