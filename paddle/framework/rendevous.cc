/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#include "paddle/framework/rendevous.h"
#include <condition_variable>
#include <utility>

// static
namespace paddle {
namespace framework {

void Rendevous::Send(const PairKey& key,
                     const platform::DeviceContext& src_device,
                     const Variable& t) {
  size_t keyhash = Hash64(key);
  std::lock_guard<std::mutex> lk(mu_);
  BlockingChannel* channel = &table_[keyhash];
  if (channel->empty() || !channel->front()->RecvReady()) {
    channel->emplace_back(
        new Rendevous::Msg{Variable(), &src_device, nullptr, nullptr});
    return;
  }
  auto& msg = *channel->front();
  msg.done_cb(src_device, *msg.dst_device, t);
  channel->pop_front();
}
void Rendevous::RecvAsync(const PairKey& key,
                          const platform::DeviceContext& dst_device,
                          const DoneCallback& cb) {
  size_t keyhash = Hash64(key);
  std::lock_guard<std::mutex> lk(mu_);
  BlockingChannel* channel = &table_[keyhash];
  if (channel->empty() || channel->front()->RecvReady()) {
    channel->emplace_back(
        new Rendevous::Msg{Variable(), nullptr, &dst_device, std::move(cb)});
    return;
  }
  auto& msg = *channel->front();
  cb(*msg.src_device, &dst_device, msg.var);
  channel->pop_front();
}
// a wrapper on RecvAsync
void Rendevous::Recv(const PairKey& key, platform::DeviceContext& dst_device,
                     Variable* t) {
  std::condition_variable cv;
  std::mutex cv_mtx;
  bool is_done = false;
  RecvAsync(key, dst_device,
            [&cv, &cv_mtx, &t, &is_done](
                const platform::DeviceContext& src_device,
                const platform::DeviceContext& dst_device, const Variable& v) {
              {
                std::unique_lock<std::mutex> lck(cv_mtx);
                *t = v;
                is_done = true;
              }
              cv.notify_all();
            });

  cv.wait(lck, [&is_done, &cv, &cv_mtx]() { return is_done == true; });
}

}  // namespace framework
}  // namespace paddle
