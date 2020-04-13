#pragma once

#define PADDLE_DISALLOW_COPY_AND_ASSIGN(type__) \
  type__(const type__ &) = delete;              \
  void operator=(const type__ &) = delete;

#include "paddle/fluid/platform/stream/gpu_event.h"

namespace paddle {
namespace platform{
namespace stream {
class BaseStream;
class Event;
namespace internal {
class StreamInterface;
class EventInterface;
}
}
}
namespace framework {
namespace details {

namespace pfs = paddle::platform::stream;

class StreamExecutor {
  public:
    // Default constructor for the abstract interface.
    StreamExecutor() {}

    // Default destructor for the abstract interface.
    virtual ~StreamExecutor() {}
    virtual bool AllocateStream(pfs::BaseStream *stream) = 0;
    virtual void DeleteStream(pfs::BaseStream *stream) = 0;
    virtual bool AllocateEvent(pfs::Event *event) = 0;
    virtual void DeleteEvent(pfs::Event *event) = 0;
    virtual std::unique_ptr<pfs::internal::StreamInterface> CreateStreamImplementation() = 0;
    virtual std::unique_ptr<pfs::internal::EventInterface> CreateEventImplementation() = 0;
    virtual pfs::Event::Status PollForStatus(pfs::Event *event) = 0;
    virtual bool CreateStreamDependency(pfs::BaseStream *dependent,
                              pfs::BaseStream *other) = 0;
    virtual bool InsertEvent(pfs::BaseStream* stream, pfs::Event* event) = 0;
    virtual void Memcpy(pfs::BaseStream* stream, void* dst, const void* src, int size) = 0;

  private:
    PADDLE_DISALLOW_COPY_AND_ASSIGN(StreamExecutor);
};

}
}
}
