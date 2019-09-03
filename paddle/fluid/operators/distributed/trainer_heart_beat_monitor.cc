// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/operators/distributed/trainer_heart_beat_monitor.h"

namespace paddle {
namespace operators {
namespace distributed {

DEFINE_int32(trainer_update_interval_secs, 900,
             " the longest time interval between the trainer update variables");

void TrainerHeartBeatMonitor::Update(const int trainer_id,
                                     TrainerStatus status) {
  std::lock_guard<std::mutex> guard(mutex_);

  Trainer& trainer = trainer_status_map_.at(trainer_id);
  double timestamp = GetCurrentUS();

  if (status == UNINITED || status == COMPLETED) {
    VLOG(4) << "trainer " << trainer.id << " is << " << status << " at "
            << timestamp;
  } else if (status == RUNNING) {
    VLOG(4) << "update trainer " << trainer_id << "'s timestamp from "
            << trainer.timestamp << " to " << timestamp << " the interval is "
            << timestamp - trainer.timestamp;

    trainer.status = status;
    trainer.timestamp = timestamp;

  } else {
    PADDLE_THROW("trainer %d 's status can not be verified.", trainer_id);
  }
}

void TrainerHeartBeatMonitor::LostTrainerMonitor() {
  VLOG(1) << "trainer heartbeat monitor start at NO.0 parameter server";
  while (running_) {
    for (int id = 0; id < trainers_; ++id) {
      auto& trainer = trainer_status_map_.at(id);

      if (trainer.status == UNINITED) {
        VLOG(4) << "trainer " << trainer.id << " is under UNINITED, skip it";
        continue;
      }
      if (trainer.status == COMPLETED) {
        VLOG(4) << "trainer " << trainer.id << " is under COMPLETED, skip it";
        continue;
      }

      double timestamp = GetCurrentUS();
      if (timestamp - trainer.timestamp >= FLAGS_trainer_update_interval_secs) {
        PADDLE_THROW(
            "the latest update of trainer %d is %f secs ago, we doubt the "
            "the trainer is not alive and this may have a bad effect on the "
            "fitting result, please check",
            trainer.id, timestamp);
      }
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(60 * 1000));
  }
  VLOG(1) << "trainer heartbeat monitor stopped, thread exit";
}

std::once_flag TrainerHeartBeatMonitor::init_flag_;
std::unique_ptr<TrainerHeartBeatMonitor> TrainerHeartBeatMonitor::monitor_(
    nullptr);

}  // namespace distributed
}  // namespace operators
}  // namespace paddle
