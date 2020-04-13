#pragma once

#include <memory>
#include <mutex>
#include <utility>
#include "paddle/fluid/framework/details/stream_executor_internal.h"

namespace paddle {
namespace platform{
namespace stream {
class BaseStream;
class Event;
}
namespace gpu {
class EventManager;
}
}
namespace framework {
namespace details {

namespace pfg = paddle::platform::gpu;
namespace pfs = paddle::platform::stream;

class GPUStreamExecutor : public StreamExecutor {
    public:
        GPUStreamExecutor();
        virtual ~GPUStreamExecutor();

  bool AllocateStream(pfs::BaseStream *stream);
  void DeleteStream(pfs::BaseStream *stream);

  bool AllocateEvent(pfs::Event *event);
  void DeleteEvent(pfs::Event* event);
  std::unique_ptr<pfs::internal::StreamInterface> CreateStreamImplementation();
  std::unique_ptr<pfs::internal::EventInterface> CreateEventImplementation();
  pfs::Event::Status PollForStatus(pfs::Event *event);
  bool CreateStreamDependency(pfs::BaseStream *dependent,
                              pfs::BaseStream *other);
  void Memcpy(pfs::BaseStream* stream, void* dst, const void* src, int size);

  bool InsertEvent(pfs::BaseStream* stream, pfs::Event* event);
  pfg::EventManager* GetEventManager() { return event_manager_; }
  //void SetMainStream(pfs::BaseStream* stream) { main_stream_ = stream; }

  pfs::BaseStream* GetMainStream() { return main_stream_; }
  pfs::BaseStream* GetD2HStream() { return device_to_host_stream_; }

    private:
      pfg::EventManager* event_manager_;
      pfs::BaseStream* main_stream_;
      pfs::BaseStream* device_to_host_stream_;
};

class GPUStreamExecutorFactory {
 public:
  static GPUStreamExecutorFactory* Singleton();
  StreamExecutor* GetStreamExecutor();

 private:
  std::mutex mu_;

  // Maintain one EventMgr per physical device (StreamExecutor is
  // per-physical-device).
  StreamExecutor* stream_executor_;
};

}
}
}
