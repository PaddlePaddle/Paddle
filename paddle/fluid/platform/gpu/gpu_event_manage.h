//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include <map>
#include <ThreadPool.h>
#include "paddle/fluid/framework/utils/notification.h"
#include "paddle/fluid/platform/stream/paddle_stream.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {
namespace details {
class StreamExecutor;
}
}

namespace platform{
namespace gpu {

namespace pfs = paddle::platform::stream;
namespace pfd = paddle::framework::details;

//class paddle::framework::ParallelExexutor;

class EventManager {
  
  private:
    friend class EventManagerFactory;
    pfd::StreamExecutor* pe_;
    std::mutex mu_;
    std::condition_variable events_pending_;
    ::ThreadPool threadpool_;
    int polling_active_delay_usecs_ = 10;
    
    struct InUse {
      pfs::Event* event;
      std::function<void()> func;
    };

    typedef std::vector<InUse> FreeVector;

    EventManager(pfd::StreamExecutor* pe_);
    void QueueInUse(pfs::BaseStream* stream, InUse in_use);
    void QueueFunc(pfs::BaseStream* stream, std::function<void()> func);
    void FreeVec(const FreeVector& to_free) {
      for (const auto& iu : to_free) {
        // The function must be called in another thread.
        if (iu.func != nullptr) threadpool_.enqueue(iu.func);
      }
    }

    void PollEvents(bool is_dedicated_poller, FreeVector* to_free);
    void PollLoop();
    void StartPollingLoop();
    void StopPollingLoop();

    std::vector<pfs::Event*> free_events_;
    std::deque<InUse> used_events_;
    bool stop_polling_;
    std::unique_ptr<paddle::framework::Notification> polling_stopped_;

  public:
    virtual ~EventManager();
    // If all stream completed, execute the function
    inline void Execute(pfs::BaseStream* stream, std::function<void()> func) {
      FreeVector to_free;
      VLOG(3)<<"stream:"<<stream<<" execute function";
      {
        std::unique_lock<std::mutex> l(mu_);
        QueueFunc(stream, std::move(func));
        PollEvents(false, &to_free);
      }
      FreeVec(to_free);
    }
};

class EventManagerFactory {
 public:
  static EventManagerFactory* Singleton();

  EventManager* GetEventManager(pfd::StreamExecutor* pe);

 private:
  std::mutex mu_;

  // Maintain one EventMgr per physical device (StreamExecutor is
  // per-physical-device).
  std::map<pfd::StreamExecutor*, EventManager*> event_mgr_map_;
};

} // namespace gpu
} // namespace framework
} // namespace paddle
