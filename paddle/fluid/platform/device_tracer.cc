/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#include "paddle/fluid/platform/device_tracer.h"

#include <deque>
#include <fstream>
#include <map>
#include <mutex>  // NOLINT
#include <numeric>
#include <string>
#include <thread>  // NOLINT
#include <vector>

#include "glog/logging.h"
#include "google/protobuf/text_format.h"
#include "paddle/fluid/framework/block_desc.h"
#include "paddle/fluid/string/printf.h"

namespace paddle {
namespace platform {
namespace {
// Current thread's id. Note, we don't distinguish nested threads
// for now.
thread_local int cur_thread_id = 0;
// Tracking the nested block stacks of each thread.
thread_local std::deque<int> block_id_stack;
// Tracking the nested event stacks.
thread_local std::deque<std::string> annotation_stack;

std::once_flag tracer_once_flag;
DeviceTracer *tracer = nullptr;
}  // namespace
#ifdef PADDLE_WITH_CUPTI

namespace {
// TODO(panyx0718): Revisit the buffer size here.
uint64_t kBufSize = 32 * 1024;
uint64_t kAlignSize = 8;

#define ALIGN_BUFFER(buffer, align)                                 \
  (((uintptr_t)(buffer) & ((align)-1))                              \
       ? ((buffer) + (align) - ((uintptr_t)(buffer) & ((align)-1))) \
       : (buffer))

#define CUPTI_CALL(call)                                                   \
  do {                                                                     \
    CUptiResult _status = call;                                            \
    if (_status != CUPTI_SUCCESS) {                                        \
      const char *errstr;                                                  \
      dynload::cuptiGetResultString(_status, &errstr);                     \
      fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n", \
              __FILE__, __LINE__, #call, errstr);                          \
      exit(-1);                                                            \
    }                                                                      \
  } while (0)

std::string MemcpyKind(CUpti_ActivityMemcpyKind kind) {
  switch (kind) {
    case CUPTI_ACTIVITY_MEMCPY_KIND_HTOD:
      return "MEMCPY_HtoD";
    case CUPTI_ACTIVITY_MEMCPY_KIND_DTOH:
      return "MEMCPY_DtoH";
    case CUPTI_ACTIVITY_MEMCPY_KIND_HTOA:
      return "MEMCPY_HtoA";
    case CUPTI_ACTIVITY_MEMCPY_KIND_ATOH:
      return "MEMCPY_AtoH";
    case CUPTI_ACTIVITY_MEMCPY_KIND_ATOA:
      return "MEMCPY_AtoA";
    case CUPTI_ACTIVITY_MEMCPY_KIND_ATOD:
      return "MEMCPY_AtoD";
    case CUPTI_ACTIVITY_MEMCPY_KIND_DTOA:
      return "MEMCPY_DtoA";
    case CUPTI_ACTIVITY_MEMCPY_KIND_DTOD:
      return "MEMCPY_DtoD";
    case CUPTI_ACTIVITY_MEMCPY_KIND_HTOH:
      return "MEMCPY_HtoH";
    case CUPTI_ACTIVITY_MEMCPY_KIND_PTOP:
      return "MEMCPY_PtoP";
    case CUPTI_ACTIVITY_MEMCPY_KIND_FORCE_INT:
      return "MEMCPY_FORCE_INT";
    default:
      break;
  }
  return "MEMCPY";
}

void EnableActivity() {
  // Device activity record is created when CUDA initializes, so we
  // want to enable it before cuInit() or any CUDA runtime call.
  CUPTI_CALL(dynload::cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY));
  CUPTI_CALL(dynload::cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL));
  CUPTI_CALL(dynload::cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DEVICE));
  CUPTI_CALL(dynload::cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMSET));
  CUPTI_CALL(dynload::cuptiActivityEnable(CUPTI_ACTIVITY_KIND_OVERHEAD));
  // We don't track these activities for now.
  // CUPTI_CALL(dynload::cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONTEXT));
  // CUPTI_CALL(dynload::cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DRIVER));
  // CUPTI_CALL(dynload::cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME));
  // CUPTI_CALL(dynload::cuptiActivityEnable(CUPTI_ACTIVITY_KIND_NAME));
  // CUPTI_CALL(dynload::cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MARKER));
}

void DisableActivity() {
  CUPTI_CALL(dynload::cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MEMCPY));
  CUPTI_CALL(dynload::cuptiActivityDisable(CUPTI_ACTIVITY_KIND_KERNEL));
  CUPTI_CALL(dynload::cuptiActivityDisable(CUPTI_ACTIVITY_KIND_DEVICE));
  // Disable all other activity record kinds.
  CUPTI_CALL(dynload::cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONTEXT));
  CUPTI_CALL(dynload::cuptiActivityDisable(CUPTI_ACTIVITY_KIND_DRIVER));
  CUPTI_CALL(dynload::cuptiActivityDisable(CUPTI_ACTIVITY_KIND_RUNTIME));
  CUPTI_CALL(dynload::cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MEMSET));
  CUPTI_CALL(dynload::cuptiActivityDisable(CUPTI_ACTIVITY_KIND_NAME));
  CUPTI_CALL(dynload::cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MARKER));
  CUPTI_CALL(dynload::cuptiActivityDisable(CUPTI_ACTIVITY_KIND_OVERHEAD));
}

void CUPTIAPI bufferRequested(uint8_t **buffer, size_t *size,
                              size_t *maxNumRecords) {
  uint8_t *buf = reinterpret_cast<uint8_t *>(malloc(kBufSize + kAlignSize));
  *size = kBufSize;
  *buffer = ALIGN_BUFFER(buf, kAlignSize);
  *maxNumRecords = 0;
}

void CUPTIAPI bufferCompleted(CUcontext ctx, uint32_t streamId, uint8_t *buffer,
                              size_t size, size_t validSize) {
  CUptiResult status;
  CUpti_Activity *record = NULL;
  if (validSize > 0) {
    do {
      status = dynload::cuptiActivityGetNextRecord(buffer, validSize, &record);
      if (status == CUPTI_SUCCESS) {
        switch (record->kind) {
          case CUPTI_ACTIVITY_KIND_KERNEL:
          case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL: {
            auto *kernel =
                reinterpret_cast<const CUpti_ActivityKernel3 *>(record);
            tracer->AddKernelRecords(kernel->start, kernel->end,
                                     kernel->deviceId, kernel->streamId,
                                     kernel->correlationId);
            break;
          }
          case CUPTI_ACTIVITY_KIND_MEMCPY: {
            auto *memcpy =
                reinterpret_cast<const CUpti_ActivityMemcpy *>(record);
            tracer->AddMemRecords(
                MemcpyKind(
                    static_cast<CUpti_ActivityMemcpyKind>(memcpy->copyKind)),
                memcpy->start, memcpy->end, memcpy->deviceId, memcpy->streamId,
                memcpy->correlationId, memcpy->bytes);
            break;
          }
          case CUPTI_ACTIVITY_KIND_MEMCPY2: {
            auto *memcpy =
                reinterpret_cast<const CUpti_ActivityMemcpy2 *>(record);
            tracer->AddMemRecords(
                MemcpyKind(
                    static_cast<CUpti_ActivityMemcpyKind>(memcpy->copyKind)),
                memcpy->start, memcpy->end, memcpy->deviceId, memcpy->streamId,
                memcpy->correlationId, memcpy->bytes);
            break;
          }
          default: { break; }
        }
      } else if (status == CUPTI_ERROR_MAX_LIMIT_REACHED) {
        // Seems not an error in this case.
        break;
      } else {
        CUPTI_CALL(status);
      }
    } while (1);

    size_t dropped;
    CUPTI_CALL(
        dynload::cuptiActivityGetNumDroppedRecords(ctx, streamId, &dropped));
    if (dropped != 0) {
      fprintf(stderr, "Dropped %u activity records\n", (unsigned int)dropped);
    }
  }
  free(buffer);
}
}  // namespace

class DeviceTracerImpl : public DeviceTracer {
 public:
  DeviceTracerImpl() : enabled_(false) {}

  void AddAnnotation(uint64_t id, const std::string &anno) {
    std::lock_guard<std::mutex> l(trace_mu_);
    correlations_[id] = anno;
  }

  void AddCPURecords(const std::string &anno, uint64_t start_ns,
                     uint64_t end_ns, int64_t device_id, int64_t thread_id) {
    if (anno.empty()) {
      VLOG(1) << "Empty timeline annotation.";
      return;
    }
    std::lock_guard<std::mutex> l(trace_mu_);
    cpu_records_.push_back(
        CPURecord{anno, start_ns, end_ns, device_id, thread_id});
  }

  void AddMemRecords(const std::string &name, uint64_t start_ns,
                     uint64_t end_ns, int64_t device_id, int64_t stream_id,
                     uint32_t correlation_id, uint64_t bytes) {
    // 0 means timestamp information could not be collected for the kernel.
    if (start_ns == 0 || end_ns == 0) {
      VLOG(3) << name << " cannot be traced";
      return;
    }
    std::lock_guard<std::mutex> l(trace_mu_);
    mem_records_.push_back(MemRecord{name, start_ns, end_ns, device_id,
                                     stream_id, correlation_id, bytes});
  }

  void AddKernelRecords(uint64_t start, uint64_t end, int64_t device_id,
                        int64_t stream_id, uint32_t correlation_id) {
    // 0 means timestamp information could not be collected for the kernel.
    if (start == 0 || end == 0) {
      VLOG(3) << correlation_id << " cannot be traced";
      return;
    }
    std::lock_guard<std::mutex> l(trace_mu_);
    kernel_records_.push_back(
        KernelRecord{start, end, device_id, stream_id, correlation_id});
  }

  bool IsEnabled() {
    std::lock_guard<std::mutex> l(trace_mu_);
    return enabled_;
  }

  void Enable() {
    std::lock_guard<std::mutex> l(trace_mu_);
    if (enabled_) {
      return;
    }
    EnableActivity();

    // Register callbacks for buffer requests and completed by CUPTI.
    CUPTI_CALL(dynload::cuptiActivityRegisterCallbacks(bufferRequested,
                                                       bufferCompleted));

    CUptiResult ret;
    ret = dynload::cuptiSubscribe(
        &subscriber_, static_cast<CUpti_CallbackFunc>(ApiCallback), this);
    if (ret == CUPTI_ERROR_MAX_LIMIT_REACHED) {
      fprintf(stderr, "CUPTI subcriber limit reached.\n");
    } else if (ret != CUPTI_SUCCESS) {
      fprintf(stderr, "Failed to create CUPTI subscriber.\n");
    }
    CUPTI_CALL(
        dynload::cuptiEnableCallback(1, subscriber_, CUPTI_CB_DOMAIN_DRIVER_API,
                                     CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel));
    CUPTI_CALL(dynload::cuptiGetTimestamp(&start_ns_));
    enabled_ = true;
  }

  proto::Profile GenProfile(const std::string &profile_path) {
    std::lock_guard<std::mutex> l(trace_mu_);
    proto::Profile profile_pb;
    profile_pb.set_start_ns(start_ns_);
    profile_pb.set_end_ns(end_ns_);
    for (const KernelRecord &r : kernel_records_) {
      if (correlations_.find(r.correlation_id) == correlations_.end()) {
        fprintf(stderr, "cannot relate a kernel activity\n");
        continue;
      }
      auto *event = profile_pb.add_events();
      event->set_type(proto::Event::GPUKernel);
      event->set_name(correlations_.at(r.correlation_id));
      event->set_start_ns(r.start_ns);
      event->set_end_ns(r.end_ns);
      event->set_sub_device_id(r.stream_id);
      event->set_device_id(r.device_id);
    }

    for (const CPURecord &r : cpu_records_) {
      auto *event = profile_pb.add_events();
      event->set_type(proto::Event::CPU);
      event->set_name(r.name);
      event->set_start_ns(r.start_ns);
      event->set_end_ns(r.end_ns);
      event->set_sub_device_id(r.thread_id);
      event->set_device_id(r.device_id);
    }
    for (const MemRecord &r : mem_records_) {
      auto *event = profile_pb.add_events();
      event->set_type(proto::Event::GPUKernel);
      event->set_name(r.name);
      event->set_start_ns(r.start_ns);
      event->set_end_ns(r.end_ns);
      event->set_sub_device_id(r.stream_id);
      event->set_device_id(r.device_id);
      event->mutable_memcopy()->set_bytes(r.bytes);
    }
    std::ofstream profile_f;
    profile_f.open(profile_path, std::ios::out | std::ios::trunc);
    std::string profile_str;
    profile_pb.SerializeToString(&profile_str);
    profile_f << profile_str;
    profile_f.close();
    return profile_pb;
  }

  void Disable() {
    // flush might cause additional calls to DeviceTracker.
    dynload::cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED);
    std::lock_guard<std::mutex> l(trace_mu_);
    DisableActivity();
    dynload::cuptiUnsubscribe(subscriber_);
    CUPTI_CALL(dynload::cuptiGetTimestamp(&end_ns_));
    enabled_ = false;
  }

 private:
  static void CUPTIAPI ApiCallback(void *userdata, CUpti_CallbackDomain domain,
                                   CUpti_CallbackId cbid, const void *cbdata) {
    auto *cbInfo = reinterpret_cast<const CUpti_CallbackData *>(cbdata);
    DeviceTracer *tracer = reinterpret_cast<DeviceTracer *>(userdata);

    if ((domain == CUPTI_CB_DOMAIN_DRIVER_API) &&
        (cbid == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel)) {
      if (cbInfo->callbackSite == CUPTI_API_ENTER) {
        const std::string anno = !annotation_stack.empty()
                                     ? annotation_stack.back()
                                     : cbInfo->symbolName;
        tracer->AddAnnotation(cbInfo->correlationId, anno);
      }
    } else {
      VLOG(1) << "Unhandled API Callback for " << domain << " " << cbid;
    }
  }

  std::mutex trace_mu_;
  bool enabled_;
  uint64_t start_ns_;
  uint64_t end_ns_;
  std::vector<KernelRecord> kernel_records_;
  std::vector<MemRecord> mem_records_;
  std::vector<CPURecord> cpu_records_;
  std::unordered_map<uint32_t, std::string> correlations_;
  CUpti_SubscriberHandle subscriber_;
};

#endif  // PADDLE_WITH_CUPTI

class DeviceTracerDummy : public DeviceTracer {
 public:
  DeviceTracerDummy() {}

  void AddAnnotation(uint64_t id, const std::string &anno) {}

  void AddCPURecords(const std::string &anno, uint64_t start_ns,
                     uint64_t end_ns, int64_t device_id, int64_t thread_id) {}

  void AddMemRecords(const std::string &name, uint64_t start_ns,
                     uint64_t end_ns, int64_t device_id, int64_t stream_id,
                     uint32_t correlation_id, uint64_t bytes) {}

  void AddKernelRecords(uint64_t start, uint64_t end, int64_t device_id,
                        int64_t stream_id, uint32_t correlation_id) {}

  bool IsEnabled() { return false; }

  void Enable() {}

  proto::Profile GenProfile(const std::string &profile_path) {
    return proto::Profile();
  }

  void Disable() {}
};

void CreateTracer(DeviceTracer **t) {
#ifdef PADDLE_WITH_CUPTI
  *t = new DeviceTracerImpl();
#else
  *t = new DeviceTracerDummy();
#endif  // PADDLE_WITH_CUPTI
}

DeviceTracer *GetDeviceTracer() {
  std::call_once(tracer_once_flag, CreateTracer, &tracer);
  return tracer;
}

void SetCurAnnotation(const std::string &anno) {
  annotation_stack.push_back(anno);
}

void ClearCurAnnotation() { annotation_stack.pop_back(); }

std::string CurAnnotation() {
  if (annotation_stack.empty()) return "";
  return annotation_stack.back();
}

void SetCurBlock(int block_id) { block_id_stack.push_back(block_id); }

void ClearCurBlock() { block_id_stack.pop_back(); }

int BlockDepth() { return block_id_stack.size(); }

void SetCurThread(int thread_id) { cur_thread_id = thread_id; }

void ClearCurThread() { cur_thread_id = 0; }

int CurThread() { return cur_thread_id; }

}  // namespace platform
}  // namespace paddle
