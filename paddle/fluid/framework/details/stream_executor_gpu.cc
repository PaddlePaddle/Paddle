#include "paddle/fluid/framework/details/stream_executor_gpu.h"
#include "paddle/fluid/platform/stream/paddle_stream.h"
#include "paddle/fluid/platform/stream/gpu_stream.h"
#include "paddle/fluid/platform/stream/gpu_event.h"
#include "paddle/fluid/platform/stream/gpu_event_impl.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/gpu/gpu_event_manage.h"

namespace paddle {
namespace framework {
namespace details {

GPUStreamExecutor::GPUStreamExecutor() {
    event_manager_ = paddle::platform::gpu::EventManagerFactory::Singleton()->GetEventManager(this);
    main_stream_ = new pfs::BaseStream(this);
    device_to_host_stream_ = new pfs::BaseStream(this);
}

GPUStreamExecutor::~GPUStreamExecutor() {
    if (event_manager_) {
        delete event_manager_;
        event_manager_ = NULL;
    }
    if (main_stream_) {
        delete main_stream_;
        main_stream_ = NULL;
    }
    if (device_to_host_stream_) {
        delete device_to_host_stream_;
        device_to_host_stream_ = NULL;
    }
}

static pfs::GpuEvent *GetGpuEvent(pfs::Event *event) {
  return static_cast<pfs::GpuEvent *>(event->implementation());
}

bool GPUStreamExecutor::AllocateStream(pfs::BaseStream *stream) {
  return pfs::GetGpuStream(stream)->Init();
}

void GPUStreamExecutor::DeleteStream(pfs::BaseStream *stream) {
    pfs::GpuStream* cuda_stream = GetGpuStream(stream);
    if (!cuda_stream->IsIdle()) {
        LOG(ERROR)<<"still has pending work, cannot delete stream";
    }
    cuda_stream->Destroy();
}

void GPUStreamExecutor::DeleteEvent(pfs::Event *event) {
    pfs::GpuEvent* event_g = GetGpuEvent(event);
    event_g->Destroy();
}

bool GPUStreamExecutor::AllocateEvent(pfs::Event *event) {
  return GetGpuEvent(event)->Init();
}

std::unique_ptr<pfs::internal::StreamInterface>
GPUStreamExecutor::CreateStreamImplementation() {
  return std::unique_ptr<pfs::internal::StreamInterface>(new pfs::GpuStream());
}

std::unique_ptr<pfs::internal::EventInterface>
GPUStreamExecutor::CreateEventImplementation() {
  return std::unique_ptr<pfs::internal::EventInterface>(new pfs::GpuEvent());
}

pfs::Event::Status GPUStreamExecutor::PollForStatus(pfs::Event *event) {
  return GetGpuEvent(event)->GetEventStatus();
}

// Make dependent wait other to finish, suppose this hold gpu device
bool GPUStreamExecutor::CreateStreamDependency(pfs::BaseStream *dependent,
                                              pfs::BaseStream *other) {
  cudaEvent_t event_finished = *GetGpuStream(other)->finish_event();
  PADDLE_ENFORCE_CUDA_SUCCESS(
      cudaEventRecord(event_finished, pfs::GetCUDAStream(other)));
  PADDLE_ENFORCE_CUDA_SUCCESS(
      cudaStreamWaitEvent(pfs::GetCUDAStream(dependent), event_finished, 0));
  return true;
}

bool GPUStreamExecutor::InsertEvent(pfs::BaseStream* stream, pfs::Event* event) {
  return GetGpuEvent(event)->InsertEvent(pfs::GetGpuStream(stream));
}

void GPUStreamExecutor::Memcpy(pfs::BaseStream* stream, void* dst, const void* src, int size) {
  auto stream_on = pfs::GetCUDAStream(stream);
  PADDLE_ENFORCE_CUDA_SUCCESS(cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, stream_on));
}

GPUStreamExecutorFactory* GPUStreamExecutorFactory::Singleton() {
    static GPUStreamExecutorFactory* instance = new GPUStreamExecutorFactory();
    return instance;
}
   
StreamExecutor* GPUStreamExecutorFactory::GetStreamExecutor() {
    if (!stream_executor_) {
        stream_executor_ = new GPUStreamExecutor();
    }
    return stream_executor_;
}

}
}
}
