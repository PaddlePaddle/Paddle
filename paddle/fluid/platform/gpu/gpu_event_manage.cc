//   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/platform/gpu/gpu_event_manage.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace platform {
namespace gpu {

namespace pf = paddle::framework;

EventManager::EventManager(pfd::StreamExecutor* pe)
    : pe_(pe),
      threadpool_(2) {
  //gpu_event_mgr::InitThreadpoolLabels(&threadpool_);
  StartPollingLoop();
}


EventManager::~EventManager() {
  StopPollingLoop();

  // Events are owned by this object.
  for (auto& e : free_events_) {
    delete e;
  }
  while (!used_events_.empty()) {
    InUse* ue = &used_events_[0];
    delete ue->event;
    if (ue->func != nullptr) threadpool_.enqueue(ue->func);
    used_events_.pop_front();
  }
}

void EventManager::StartPollingLoop() {
  mu_.lock();
  stop_polling_ = false;
  mu_.unlock();
  polling_stopped_.reset(new pf::Notification);
  threadpool_.enqueue([this]() { PollLoop(); });
  LOG(INFO)<<"++++start poolingloop";
}

void EventManager::StopPollingLoop() {
  if (polling_stopped_) {
    mu_.lock();
    stop_polling_ = true;
    events_pending_.notify_all();
    mu_.unlock();
    polling_stopped_->WaitForNotification();
    polling_stopped_.reset(nullptr);
  }
  VLOG(3)<<"StopPollingLoop";
}

void EventManager::PollLoop() {
  FreeVector to_free;
  while (true) {
    LOG(INFO)<<"++++start pool loop";
    bool events_still_pending;
    {
      std::unique_lock<std::mutex> lock(mu_);
      if (stop_polling_) {
        break;
      }
      if (used_events_.empty()) {
        events_pending_.wait(lock);
      }
      PollEvents(true, &to_free);
      events_still_pending = !used_events_.empty();
    }
    FreeVec(to_free);
    //to_free.clear();

    if (events_still_pending) {
      std::this_thread::sleep_for(std::chrono::microseconds(polling_active_delay_usecs_ / 1000));
    }
  }
  polling_stopped_->Notify();
}

void EventManager::QueueFunc(pfs::BaseStream* stream, std::function<void()> func)
{
  VLOG(2)<<"QueueFunc for stream:"<<stream;
  QueueInUse(stream, {nullptr, std::move(func)});
}

void EventManager::QueueInUse(pfs::BaseStream* stream, InUse iu) {
  VLOG(2) << "QueueInUse  free_events_ " << free_events_.size()
          << " used_events_ " << used_events_.size();
  // Events are created on demand, and repeatedly reused.  There is no
  // limit placed here on the number of allocated Events.
  if (free_events_.empty()) {
    free_events_.push_back(new pfs::Event(pe_));
    free_events_.back()->Init();
  }
  pfs::Event* e = free_events_.back();
  free_events_.pop_back();
  stream->InsertEvent(e);
  iu.event = e;
  bool was_empty = used_events_.empty();
  used_events_.push_back(iu);
  // Maybe wake up the polling thread
  if (was_empty) events_pending_.notify_all();
}

void EventManager::PollEvents(bool is_dedicated_poller,
                          std::vector<InUse>* to_free) {
  VLOG(2) << "PollEvents  free_events_ " << free_events_.size()
          << " used_events_ " << used_events_.size();
  // Sweep the remaining events in order.  If this is the dedicated
  // polling thread, check the entire set.  Otherwise, just sweep up to
  // the first non-complete record that is still pending.
  for (auto& iu : used_events_) {
    if (iu.event == nullptr) continue;
    pfs::Event::Status st = iu.event->PollForStatus();
    switch (st) {
      case pfs::Event::Status::kUnknown:
      case pfs::Event::Status::kError:
        // We don't expect to see these.  Someday maybe propagate
        // a Status error, but for now fail hard.
        LOG(FATAL) << "Unexpected Event status: " << static_cast<int>(st);
        break;
      case pfs::Event::Status::kPending:
        if (!is_dedicated_poller) return;  // quit processing queue
        break;
      case pfs::Event::Status::kComplete:
        // Make a copy of the InUse record so we can free it after releasing
        // the lock
        to_free->push_back(iu);
        //to_free->at(to_free->end()) = iu.event;
        // Mark this InUse record as completed.
        iu.event = nullptr;
    }
  }
  // Then clear any completed InUse records from the front of the queue.
  while (!used_events_.empty()) {
    InUse& iu = used_events_.front();
    if (iu.event == nullptr) {
      used_events_.pop_front();
    } else {
      break;
    }
  }
}

EventManagerFactory* EventManagerFactory::Singleton() {
  static EventManagerFactory* instance = new EventManagerFactory;
  return instance;
}

EventManager* EventManagerFactory::GetEventManager(pfd::StreamExecutor* pe) {
  const std::lock_guard<std::mutex> lock(mu_);

  auto itr = event_mgr_map_.find(pe);
  if (itr == event_mgr_map_.end()) {
    auto event_mgr = new EventManager(pe);
    event_mgr_map_[pe] = event_mgr;
    return event_mgr;
  } else {
    return itr->second;
  }
}

}
}
}
