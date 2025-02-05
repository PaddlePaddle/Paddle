// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/core/operators/reader/buffered_reader.h"

#include "paddle/phi/core/framework/convert_utils.h"
#include "paddle/phi/core/platform/device/device_wrapper.h"
#include "paddle/phi/core/platform/profiler.h"
#include "paddle/phi/core/platform/profiler/event_tracing.h"

#include "paddle/phi/backends/device_guard.h"
#include "paddle/phi/backends/device_manager.h"
#include "paddle/phi/common/memory_utils.h"

namespace paddle::operators::reader {
BufferedReader::~BufferedReader() {
  VLOG(1) << "~BufferedReader";
  reader_->Shutdown();
  while (!position_.empty()) {
    auto &front = position_.front();
    if (front.valid()) {
      front.wait();
    }
    position_.pop();
  }
}

BufferedReader::BufferedReader(
    const std::shared_ptr<framework::ReaderBase> &reader,
    const phi::Place &place,
    size_t buffer_size,
    bool pin_memory)
    : framework::DecoratedReader(reader),
      thread_pool_(1),
      place_(place),
      buffer_size_(buffer_size),
      pin_memory_(pin_memory),
      position_(),
      cpu_buffer_(),
      cuda_buffer_(),
      xpu_buffer_(),
      custom_device_buffer_() {
  VLOG(1) << "BufferedReader";
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  if (place_.GetType() == phi::AllocationType::GPU && !pin_memory) {
    int dev_idx = place_.device;  // NOLINT
    compute_stream_ =
        ((phi::GPUContext *)(phi::DeviceContextPool::Instance().Get(place_)))
            ->stream();
    events_.resize(buffer_size);
    for (auto &event : events_) {
      event = platform::CudaEventResourcePool::Instance().New(dev_idx);
    }
    stream_ = platform::CudaStreamResourcePool::Instance().New(dev_idx);
  }
#endif

#ifdef PADDLE_WITH_XPU
  if (place_.GetType() == phi::AllocationType::XPU) {
    int dev_idx = place_.device;
    compute_stream_ =
        ((phi::XPUContext *)(phi::DeviceContextPool::Instance().Get(place_)))
            ->stream();
    events_.resize(buffer_size);
    for (auto &event : events_) {
      event = platform::XpuEventResourcePool::Instance().New(dev_idx);
    }
    stream_ = platform::XpuStreamResourcePool::Instance().New(dev_idx);
  }
#endif

#ifdef PADDLE_WITH_CUSTOM_DEVICE
  if (place_.GetType() == phi::AllocationType::CUSTOM) {
    auto stream =
        ((phi::CustomContext *)(phi::DeviceContextPool::Instance().Get(place_)))
            ->stream();
    custom_device_compute_stream_ =
        std::make_shared<phi::stream::Stream>(place_, stream);

    custom_device_events_.resize(buffer_size);
    for (auto &event : custom_device_events_) {
      event = std::make_shared<phi::event::Event>();
      event->Init(place_);
    }
    custom_device_stream_ = std::make_shared<phi::stream::Stream>();
    custom_device_stream_->Init(place_);
  }
#endif

  cpu_buffer_.resize(buffer_size);
  cuda_buffer_.resize(buffer_size);
  xpu_buffer_.resize(buffer_size);
  custom_device_buffer_.resize(buffer_size);
  ReadTillBufferFullAsync();
}

void BufferedReader::ReadTillBufferFullAsync() {
  for (size_t i = 0; i < buffer_size_; ++i) {
    ReadAsync(i);
  }
}

void BufferedReader::ReadAsync(size_t i) {
  position_.emplace(thread_pool_.enqueue([this, i]() -> size_t {
    TensorVec &cpu = cpu_buffer_[i];
    reader_->ReadNext(&cpu);

    if (cpu.empty()) {
      return -1UL;
    }

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)  // @{ Group GPU Place
    if (place_.GetType() == phi::AllocationType::GPU) {
      auto dev_ctx_gpu =
          (phi::GPUContext *)(phi::DeviceContextPool::Instance().Get(place_));
      TensorVec &cuda = cuda_buffer_[i];
      if (cuda.empty()) {
        cuda.resize(cpu.size());
      } else {
        PADDLE_ENFORCE_EQ(
            cuda.size(),
            cpu.size(),
            common::errors::InvalidArgument(
                "Input tensor number on GPU and CPU devices are not matched."));
      }
      if (pin_memory_) {
        // NOTE: [Copy processing of different input devices]
        // We may accept input tensor in three different devices:
        //   - CPUPlace
        //   - CUDAPinnedPlace
        //   - CUDAPlace
        // CUDA Stream Synchronizing is slow, in order to avoid Synchronizing
        // in BufferedReader thread, we do data copy as follows:
        //   - If src Tensor on CPU memory, we copy it to CUDAPinned memory
        //   - IF src Tensor on CUDAPinned memory, we use it directly
        //   - IF src Tensor on CUDA memory, we use it directly
        phi::GPUPinnedPlace cuda_pinned_place;
        std::vector<void *> cuda_pinned_ptrs;
        cuda_pinned_ptrs.reserve(cpu.size());
        phi::RecordEvent record_event(
            "BufferedReader:MemoryCopy", phi::TracerEventType::UserDefined, 1);
        // NODE(chenweihang): When we use CUDAPinned Memory, we need call
        // cudaHostAlloc, that is a CUDA API, calling CUDA API need load
        // cuda lib into device, it will cost hundreds of MB of GPU memory.
        // If we don't set Device here, which will use CUDAPlace(0) default.
        platform::SetDeviceId(place_.device);
        for (size_t i = 0; i < cpu.size(); ++i) {
          if (cpu[i].place().GetType() == phi::AllocationType::CPU) {
            cuda[i].Resize(cpu[i].dims());
            cuda[i].set_layout(cpu[i].layout());
            cuda_pinned_ptrs[i] =
                dev_ctx_gpu->Alloc(&cuda[i],
                                   cpu[i].type(),
                                   0,
                                   true);  // pinned=true to use GPUPinnedPlace
            auto size = cpu[i].numel() * phi::SizeOf(cpu[i].dtype());

            phi::memory_utils::Copy(cuda_pinned_place,
                                    cuda_pinned_ptrs[i],
                                    cpu[i].place(),
                                    cpu[i].data(),
                                    size);

            cuda[i].set_lod(cpu[i].lod());
          } else {
            // Here the cpu[i]'s place may be CUDAPlace, CUDAPinnedPlace, or
            // others, we don't copy the memory of it to CUDAPinnedPlace, but
            // we should share tensor data to cuda[i]
            cuda[i].ShareDataWith(cpu[i]);
          }
        }
      } else {
        // NOTE(liangdun): using async copy instead of TensorCopySync
        // TensorCopySync would block other stream, because TensorCopySync
        // issues the copying command to the default stream, it will make two
        // commands from different streams cannot run concurrently.
        std::vector<void *> gpu_ptrs;
        gpu_ptrs.reserve(cpu.size());
        for (size_t i = 0; i < cpu.size(); ++i) {
          cuda[i].Resize(cpu[i].dims());
          cuda[i].set_layout(cpu[i].layout());
          gpu_ptrs.emplace_back(dev_ctx_gpu->Alloc(&cuda[i], cpu[i].type()));
        }

        // NOTE(zjl): cudaStreamWaitEvent() must be called after all
        // Alloc(cuda[i]) is called, since some ops release
        // cuda memory immediately without waiting cuda kernel ends
        platform::SetDeviceId(place_.device);
#ifdef PADDLE_WITH_HIP
        PADDLE_ENFORCE_GPU_SUCCESS(
            hipEventRecord(events_[i].get(), compute_stream_));
        PADDLE_ENFORCE_GPU_SUCCESS(
            hipStreamWaitEvent(stream_.get(), events_[i].get(), 0));
#else
        PADDLE_ENFORCE_GPU_SUCCESS(
            cudaEventRecord(events_[i].get(), compute_stream_));
        PADDLE_ENFORCE_GPU_SUCCESS(
            cudaStreamWaitEvent(stream_.get(), events_[i].get(), 0));
#endif

        phi::RecordEvent record_event(
            "BufferedReader:MemoryCopy", phi::TracerEventType::UserDefined, 1);
        for (size_t i = 0; i < cpu.size(); ++i) {
          auto cpu_place = cpu[i].place();
          auto cpu_ptr = cpu[i].data();
          auto gpu_ptr = gpu_ptrs[i];
          auto size = cpu[i].numel() * phi::SizeOf(cpu[i].dtype());
          if (cpu_place.GetType() == phi::AllocationType::GPUPINNED ||
              cpu_place.GetType() == phi::AllocationType::GPU) {
            phi::memory_utils::Copy(
                place_, gpu_ptr, cpu_place, cpu_ptr, size, stream_.get());
          } else {
            phi::GPUPinnedPlace cuda_pinned_place;
            phi::DenseTensor cuda_pinned_tensor;
            cuda_pinned_tensor.Resize(cpu[i].dims());
            auto cuda_pinned_ptr =
                dev_ctx_gpu->Alloc(&cuda_pinned_tensor,
                                   cpu[i].type(),
                                   0,
                                   true);  // pinned=true to use GPUPinnedPlace
            phi::memory_utils::Copy(
                cuda_pinned_place, cuda_pinned_ptr, cpu_place, cpu_ptr, size);
            phi::memory_utils::Copy(place_,
                                    gpu_ptr,
                                    cuda_pinned_place,
                                    cuda_pinned_ptr,
                                    size,
                                    stream_.get());

            phi::backends::gpu::GpuStreamSync(stream_.get());
          }
          cuda[i].set_lod(cpu[i].lod());
        }
        phi::backends::gpu::GpuStreamSync(stream_.get());
      }
    }
#endif

#ifdef PADDLE_WITH_XPU
    if (place_.GetType() == phi::AllocationType::XPU) {
      auto dev_ctx_xpu =
          (phi::XPUContext *)(phi::DeviceContextPool::Instance().Get(place_));
      TensorVec &xpu = xpu_buffer_[i];
      if (xpu.empty()) {
        xpu.resize(cpu.size());
      } else {
        PADDLE_ENFORCE_EQ(
            xpu.size(),
            cpu.size(),
            common::errors::InvalidArgument(
                "Input tensor number on XPU and CPU devices are not matched. "
                "The number on XPU is %d, on CPU is %d",
                xpu.size(),
                cpu.size()));
      }

      std::vector<void *> xpu_ptrs;
      xpu_ptrs.reserve(cpu.size());
      for (size_t i = 0; i < cpu.size(); ++i) {
        xpu[i].Resize(cpu[i].dims());
        xpu[i].set_layout(cpu[i].layout());
        xpu_ptrs.emplace_back(dev_ctx_xpu->Alloc(&xpu[i], cpu[i].type()));
      }

      phi::backends::xpu::XPUDeviceGuard guard(place_.device);
      int r = xpu_event_record(events_[i].get(), compute_stream_);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "xpu_event_record");
      r = xpu_stream_wait_event(stream_.get(), events_[i].get());
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "xpu_stream_wait_event");

      phi::RecordEvent record_event(
          "BufferedReader:MemoryCopy", phi::TracerEventType::UserDefined, 1);
      for (size_t i = 0; i < cpu.size(); ++i) {
        auto cpu_place = cpu[i].place();
        auto cpu_ptr = cpu[i].data();
        auto xpu_ptr = xpu_ptrs[i];
        auto size = cpu[i].numel() * phi::SizeOf(cpu[i].dtype());
        // TODO(zhanghuan) for now hardware not support xpu_memcpy_async, maybe
        // KL3
        if ((cpu_place.GetType() == phi::AllocationType::XPU)) {
          platform::XPUStreamSync(stream_.get());
          char *tmp = new char[size];
          PADDLE_ENFORCE_XPU_SUCCESS(xpu_memcpy(
              tmp, cpu_ptr, size, XPUMemcpyKind::XPU_DEVICE_TO_HOST));
          PADDLE_ENFORCE_XPU_SUCCESS(xpu_memcpy(
              xpu_ptr, tmp, size, XPUMemcpyKind::XPU_HOST_TO_DEVICE));
          delete[] tmp;
        } else {
          phi::memory_utils::Copy(place_, xpu_ptr, cpu_place, cpu_ptr, size);
        }
        xpu[i].set_lod(cpu[i].lod());
      }
      platform::XPUStreamSync(stream_.get());
    }
#endif

#ifdef PADDLE_WITH_CUSTOM_DEVICE
    if (place_.GetType() == phi::AllocationType::CUSTOM) {
      auto dev_ctx_custom =
          (phi::CustomContext *)(phi::DeviceContextPool::Instance().Get(
              place_));
      phi::DeviceManager::SetDevice(place_);

      TensorVec &custom_device = custom_device_buffer_[i];
      if (custom_device.empty()) {
        custom_device.resize(cpu.size());
      } else {
        PADDLE_ENFORCE_EQ(custom_device.size(),
                          cpu.size(),
                          common::errors::InvalidArgument(
                              "Input tensor number on CustomDevice and CPU "
                              "devices are not matched. "
                              "The number on CustomDevice is %d, on CPU is %d",
                              custom_device.size(),
                              cpu.size()));
      }

      std::vector<void *> custom_device_ptrs;
      custom_device_ptrs.reserve(cpu.size());
      for (size_t i = 0; i < cpu.size(); ++i) {
        custom_device[i].Resize(cpu[i].dims());
        custom_device[i].set_layout(cpu[i].layout());
        custom_device_ptrs.emplace_back(
            dev_ctx_custom->Alloc(&custom_device[i], cpu[i].type()));
      }

      phi::DeviceManager::GetDeviceWithPlace(place_)->RecordEvent(
          custom_device_events_[i].get(), custom_device_compute_stream_.get());
      phi::DeviceManager::GetDeviceWithPlace(place_)->StreamWaitEvent(
          custom_device_stream_.get(), custom_device_events_[i].get());

      phi::RecordEvent record_event(
          "BufferedReader:MemoryCopy", phi::TracerEventType::UserDefined, 1);
      for (size_t i = 0; i < cpu.size(); ++i) {
        auto cpu_place = cpu[i].place();
        auto cpu_ptr = cpu[i].data();
        auto custom_device_ptr = custom_device_ptrs[i];
        auto size = cpu[i].numel() * phi::SizeOf(cpu[i].dtype());
        if ((cpu_place.GetType() == phi::AllocationType::CUSTOM)) {
          phi::memory_utils::Copy(
              place_, custom_device_ptr, cpu_place, cpu_ptr, size);
          custom_device_stream_->Synchronize();
        } else {
          phi::memory_utils::Copy(
              place_, custom_device_ptr, cpu_place, cpu_ptr, size);
        }
        custom_device[i].set_lod(cpu[i].lod());
      }
      custom_device_stream_->Synchronize();
    }
#endif
    return i;
  }));
}

void BufferedReader::ShutdownImpl() {
  VLOG(1) << "ShutdownImpl";
  reader_->Shutdown();
  while (!position_.empty()) {
    position_.pop();
  }
  prev_pos_ = -1UL;
}

void BufferedReader::StartImpl() {
  reader_->Start();
  ReadTillBufferFullAsync();
}

void BufferedReader::ReadNextImpl(phi::TensorArray *out) {
  if (position_.empty()) {
    out->clear();
    return;
  }
  size_t i = position_.front().get();
  position_.pop();

  if (i == -1UL) {
    ReadNextImpl(out);
    return;
  }

  if (place_.GetType() == phi::AllocationType::GPU) {  // NOLINT
    *out = std::move(cuda_buffer_[i]);
  } else if (place_.GetType() == phi::AllocationType::XPU) {
    *out = std::move(xpu_buffer_[i]);
  } else if (place_.GetType() == phi::AllocationType::CUSTOM) {
    *out = std::move(custom_device_buffer_[i]);
  } else {
    *out = std::move(cpu_buffer_[i]);
  }

  // Do not push current position into ReadAsync. Push the previous position
  // Since all computation in fluid are async, change the data of
  // current position may cause data error.
  if (prev_pos_ != -1Ul) {
    ReadAsync(prev_pos_);
  }
  prev_pos_ = i;
}

}  // namespace paddle::operators::reader
