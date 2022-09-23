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

#include "paddle/fluid/operators/reader/buffered_reader.h"

#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/platform/device/device_wrapper.h"
#include "paddle/fluid/platform/profiler.h"
#include "paddle/fluid/platform/profiler/event_tracing.h"

#include "paddle/phi/backends/device_guard.h"
#include "paddle/phi/backends/device_manager.h"

namespace paddle {
namespace operators {
namespace reader {
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
    const platform::Place &place,
    size_t buffer_size,
    bool pin_memory)
    : framework::DecoratedReader(reader),
      thread_pool_(1),
      place_(place),
      buffer_size_(buffer_size),
      pin_memory_(pin_memory) {
  VLOG(1) << "BufferedReader";
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  if (platform::is_gpu_place(place_) && !pin_memory) {
    int dev_idx = place_.device;
    compute_stream_ =
        ((phi::GPUContext *)(platform::DeviceContextPool::Instance().Get(
             place_)))
            ->stream();
    events_.resize(buffer_size);
    for (auto &event : events_) {
      event = platform::CudaEventResourcePool::Instance().New(dev_idx);
    }
    stream_ = platform::CudaStreamResourcePool::Instance().New(dev_idx);
  }
#endif

#ifdef PADDLE_WITH_ASCEND_CL
  if (platform::is_npu_place(place_)) {
    int dev_idx = place_.device;
    compute_stream_ =
        ((platform::NPUDeviceContext *)(platform::DeviceContextPool::Instance()
                                            .Get(place_)))
            ->stream();
    events_.resize(buffer_size);
    for (auto &event : events_) {
      event = platform::NpuEventResourcePool::Instance().New(dev_idx);
    }
    stream_ = platform::NpuStreamResourcePool::Instance().New(dev_idx);
  }
#endif

#ifdef PADDLE_WITH_MLU
  if (platform::is_mlu_place(place_)) {
    int dev_idx = place_.device;
    compute_stream_ =
        ((platform::MLUDeviceContext *)(platform::DeviceContextPool::Instance()
                                            .Get(place_)))
            ->stream();
    events_.resize(buffer_size);
    for (auto &event : events_) {
      event = platform::MluEventResourcePool::Instance().New(dev_idx);
    }
    stream_ = platform::MluStreamResourcePool::Instance().New(dev_idx);
  }
#endif

#ifdef PADDLE_WITH_XPU
  if (platform::is_xpu_place(place_)) {
    int dev_idx = place_.device;
    compute_stream_ =
        ((platform::XPUDeviceContext *)(platform::DeviceContextPool::Instance()
                                            .Get(place_)))
            ->stream();
    events_.resize(buffer_size);
    for (auto &event : events_) {
      event = platform::XpuEventResourcePool::Instance().New(dev_idx);
    }
    stream_ = platform::XpuStreamResourcePool::Instance().New(dev_idx);
  }
#endif

#ifdef PADDLE_WITH_CUSTOM_DEVICE
  if (platform::is_custom_place(place_)) {
    auto stream = ((platform::CustomDeviceContext
                        *)(platform::DeviceContextPool::Instance().Get(place_)))
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
  npu_buffer_.resize(buffer_size);
  mlu_buffer_.resize(buffer_size);
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
    if (platform::is_gpu_place(place_)) {
      TensorVec &cuda = cuda_buffer_[i];
      if (cuda.empty()) {
        cuda.resize(cpu.size());
      } else {
        PADDLE_ENFORCE_EQ(
            cuda.size(),
            cpu.size(),
            platform::errors::InvalidArgument(
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
        platform::CUDAPinnedPlace cuda_pinned_place;
        std::vector<void *> cuda_pinned_ptrs;
        cuda_pinned_ptrs.reserve(cpu.size());
        platform::RecordEvent record_event(
            "BufferedReader:MemoryCopy",
            platform::TracerEventType::UserDefined,
            1);
        // NODE(chenweihang): When we use CUDAPinned Memory, we need call
        // cudaHostAlloc, that is a CUDA API, calling CUDA API need load
        // cuda lib into device, it will cost hundreds of MB of GPU memory.
        // If we don't set Device here, which will use CUDAPlace(0) default.
        platform::SetDeviceId(place_.device);
        for (size_t i = 0; i < cpu.size(); ++i) {
          if (platform::is_cpu_place(cpu[i].place())) {
            cuda[i].Resize(cpu[i].dims());
            cuda[i].set_layout(cpu[i].layout());
            cuda_pinned_ptrs[i] =
                cuda[i].mutable_data(cuda_pinned_place, cpu[i].type());
            auto size = cpu[i].numel() *
                        paddle::framework::DataTypeSize(cpu[i].dtype());

            memory::Copy(cuda_pinned_place,
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
          gpu_ptrs.emplace_back(cuda[i].mutable_data(place_, cpu[i].type()));
        }

        // NOTE(zjl): cudaStreamWaitEvent() must be called after all
        // cuda[i].mutable_data() is called, since some ops release
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

        platform::RecordEvent record_event(
            "BufferedReader:MemoryCopy",
            platform::TracerEventType::UserDefined,
            1);
        for (size_t i = 0; i < cpu.size(); ++i) {
          auto cpu_place = cpu[i].place();
          auto cpu_ptr = cpu[i].data();
          auto gpu_ptr = gpu_ptrs[i];
          auto size =
              cpu[i].numel() * paddle::framework::DataTypeSize(cpu[i].dtype());
          if (platform::is_cuda_pinned_place(cpu_place)) {
            memory::Copy(
                place_, gpu_ptr, cpu_place, cpu_ptr, size, stream_.get());
          } else if ((platform::is_gpu_place(cpu_place))) {
            memory::Copy(
                place_, gpu_ptr, cpu_place, cpu_ptr, size, stream_.get());
          } else {
            platform::CUDAPinnedPlace cuda_pinned_place;
            framework::LoDTensor cuda_pinned_tensor;
            cuda_pinned_tensor.Resize(cpu[i].dims());
            auto cuda_pinned_ptr = cuda_pinned_tensor.mutable_data(
                cuda_pinned_place, cpu[i].type());
            memory::Copy(
                cuda_pinned_place, cuda_pinned_ptr, cpu_place, cpu_ptr, size);
            memory::Copy(place_,
                         gpu_ptr,
                         cuda_pinned_place,
                         cuda_pinned_ptr,
                         size,
                         stream_.get());

            platform::GpuStreamSync(stream_.get());
          }
          cuda[i].set_lod(cpu[i].lod());
        }
        platform::GpuStreamSync(stream_.get());
      }
    }
#endif

#ifdef PADDLE_WITH_ASCEND_CL
    if (platform::is_npu_place(place_)) {
      TensorVec &npu = npu_buffer_[i];
      if (npu.empty()) {
        npu.resize(cpu.size());
      } else {
        PADDLE_ENFORCE_EQ(
            npu.size(),
            cpu.size(),
            platform::errors::InvalidArgument(
                "Input tensor number on NPU and CPU devices are not matched. "
                "The number on NPU is %d, on CPU is %d",
                npu.size(),
                cpu.size()));
      }

      std::vector<void *> npu_ptrs;
      npu_ptrs.reserve(cpu.size());
      for (size_t i = 0; i < cpu.size(); ++i) {
        npu[i].Resize(cpu[i].dims());
        npu[i].set_layout(cpu[i].layout());
        npu_ptrs.emplace_back(npu[i].mutable_data(place_, cpu[i].type()));
      }

      platform::SetNPUDeviceId(place_.device);
      platform::NPUEventRecord(events_[i].get(), compute_stream_);
      platform::NPUStreamWaitEvent(stream_.get(), events_[i].get());

      platform::RecordEvent record_event("BufferedReader:MemoryCopy",
                                         platform::TracerEventType::UserDefined,
                                         1);
      for (size_t i = 0; i < cpu.size(); ++i) {
        auto cpu_place = cpu[i].place();
        auto cpu_ptr = cpu[i].data();
        auto npu_ptr = npu_ptrs[i];
        auto size =
            cpu[i].numel() * paddle::framework::DataTypeSize(cpu[i].dtype());
        if ((platform::is_npu_place(cpu_place))) {
          memory::Copy(
              place_, npu_ptr, cpu_place, cpu_ptr, size, stream_.get());
        } else {
          memory::Copy(
              place_, npu_ptr, cpu_place, cpu_ptr, size, stream_.get());
          platform::NPUStreamSync(stream_.get());
        }
        npu[i].set_lod(cpu[i].lod());
      }
      platform::NPUStreamSync(stream_.get());
    }
#endif

#ifdef PADDLE_WITH_MLU
    if (platform::is_mlu_place(place_)) {
      TensorVec &mlu = mlu_buffer_[i];
      if (mlu.empty()) {
        mlu.resize(cpu.size());
      } else {
        PADDLE_ENFORCE_EQ(
            mlu.size(),
            cpu.size(),
            platform::errors::InvalidArgument(
                "Input tensor number on MLU and CPU devices are not matched. "
                "The number on MLU is %d, on CPU is %d",
                mlu.size(),
                cpu.size()));
      }

      std::vector<void *> mlu_ptrs;
      mlu_ptrs.reserve(cpu.size());
      for (size_t i = 0; i < cpu.size(); ++i) {
        mlu[i].Resize(cpu[i].dims());
        mlu[i].set_layout(cpu[i].layout());
        mlu_ptrs.emplace_back(mlu[i].mutable_data(place_, cpu[i].type()));
      }

      platform::SetMLUDeviceId(place_.device);
      PADDLE_ENFORCE_MLU_SUCCESS(
          cnPlaceNotifier(events_[i].get(), compute_stream_));
      PADDLE_ENFORCE_MLU_SUCCESS(cnWaitNotifier(events_[i].get()));

      platform::RecordEvent record_event("BufferedReader:MemoryCopy",
                                         platform::TracerEventType::UserDefined,
                                         1);
      for (size_t i = 0; i < cpu.size(); ++i) {
        auto cpu_place = cpu[i].place();
        auto cpu_ptr = cpu[i].data();
        auto mlu_ptr = mlu_ptrs[i];
        auto size =
            cpu[i].numel() * paddle::framework::DataTypeSize(cpu[i].dtype());
        if ((platform::is_mlu_place(cpu_place))) {
          memory::Copy(
              place_, mlu_ptr, cpu_place, cpu_ptr, size, stream_.get());
        } else {
          memory::Copy(
              place_, mlu_ptr, cpu_place, cpu_ptr, size, stream_.get());
          platform::MLUStreamSync(stream_.get());
        }
        mlu[i].set_lod(cpu[i].lod());
      }
      platform::MLUStreamSync(stream_.get());
    }
#endif

#ifdef PADDLE_WITH_XPU
    if (platform::is_xpu_place(place_)) {
      TensorVec &xpu = xpu_buffer_[i];
      if (xpu.empty()) {
        xpu.resize(cpu.size());
      } else {
        PADDLE_ENFORCE_EQ(
            xpu.size(),
            cpu.size(),
            platform::errors::InvalidArgument(
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
        xpu_ptrs.emplace_back(xpu[i].mutable_data(place_, cpu[i].type()));
      }

      platform::XPUDeviceGuard gurad(place_.device);
      int r = xpu_event_record(events_[i].get(), compute_stream_);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "xpu_event_record");
      r = xpu_stream_wait_event(stream_.get(), events_[i].get());
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "xpu_stream_wait_event");

      platform::RecordEvent record_event("BufferedReader:MemoryCopy",
                                         platform::TracerEventType::UserDefined,
                                         1);
      for (size_t i = 0; i < cpu.size(); ++i) {
        auto cpu_place = cpu[i].place();
        auto cpu_ptr = cpu[i].data();
        auto xpu_ptr = xpu_ptrs[i];
        auto size =
            cpu[i].numel() * paddle::framework::DataTypeSize(cpu[i].dtype());
        // TODO(zhanghuan) for now hardware not support xpu_memcpy_async, maybe
        // KL3
        if ((platform::is_xpu_place(cpu_place))) {
          memory::Copy(place_, xpu_ptr, cpu_place, cpu_ptr, size);
          platform::XPUStreamSync(stream_.get());
        } else {
          memory::Copy(place_, xpu_ptr, cpu_place, cpu_ptr, size);
        }
        xpu[i].set_lod(cpu[i].lod());
      }
      platform::XPUStreamSync(stream_.get());
    }
#endif

#ifdef PADDLE_WITH_CUSTOM_DEVICE
    if (platform::is_custom_place(place_)) {
      TensorVec &custom_device = custom_device_buffer_[i];
      if (custom_device.empty()) {
        custom_device.resize(cpu.size());
      } else {
        PADDLE_ENFORCE_EQ(custom_device.size(),
                          cpu.size(),
                          platform::errors::InvalidArgument(
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
            custom_device[i].mutable_data(place_, cpu[i].type()));
      }

      phi::DeviceManager::SetDevice(place_);
      phi::DeviceManager::GetDeviceWithPlace(place_)->RecordEvent(
          custom_device_events_[i].get(), custom_device_compute_stream_.get());
      phi::DeviceManager::GetDeviceWithPlace(place_)->StreamWaitEvent(
          custom_device_stream_.get(), custom_device_events_[i].get());

      platform::RecordEvent record_event("BufferedReader:MemoryCopy",
                                         platform::TracerEventType::UserDefined,
                                         1);
      for (size_t i = 0; i < cpu.size(); ++i) {
        auto cpu_place = cpu[i].place();
        auto cpu_ptr = cpu[i].data();
        auto custom_device_ptr = custom_device_ptrs[i];
        auto size =
            cpu[i].numel() * paddle::framework::DataTypeSize(cpu[i].dtype());
        if ((platform::is_custom_place(cpu_place))) {
          memory::Copy(place_, custom_device_ptr, cpu_place, cpu_ptr, size);
          custom_device_stream_->Synchronize();
        } else {
          memory::Copy(place_, custom_device_ptr, cpu_place, cpu_ptr, size);
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

void BufferedReader::ReadNextImpl(paddle::framework::LoDTensorArray *out) {
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

  if (platform::is_gpu_place(place_)) {
    *out = std::move(cuda_buffer_[i]);
  } else if (platform::is_npu_place(place_)) {
    *out = std::move(npu_buffer_[i]);
  } else if (platform::is_mlu_place(place_)) {
    *out = std::move(mlu_buffer_[i]);
  } else if (platform::is_xpu_place(place_)) {
    *out = std::move(xpu_buffer_[i]);
  } else if (platform::is_custom_place(place_)) {
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

}  // namespace reader
}  // namespace operators
}  // namespace paddle
