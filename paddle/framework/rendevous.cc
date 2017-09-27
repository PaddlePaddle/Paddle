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

using Rendevous::Msg;

void Rendevous::Send(const PairKey& key,
                     const platform::DeviceContext& src_device,
                     const Variable& t) {
  size_t keyhash = Hash64(key);
  std::lock_guard<std::mutex> lk(mu_);
  BlockingChannel* channel = &table_[keyhash];
  if (channel->empty() || !channel->front()->RecvReady()) {
    channel->emplace_back(new Msg{t, &src_device, nullptr, nullptr});
    return;
  }
  auto& msg = *channel->front();
  msg.done_cb(src_device, *msg.dst_device, t);
  channel->pop_front();
}
void Rendevous::RecvAsync(const PairKey& key,
                          platform::DeviceContext& dst_device,
                          const DoneCallback& cb) {
  size_t keyhash = Hash64(key);
  std::lock_guard<std::mutex> lk(mu_);
  BlockingChannel* channel = &table_[keyhash];
  if (channel->empty() || channel->front()->RecvReady()) {
    channel->emplace_back(new Msg{t, nullptr, &dst_device, std::move(cb)});
    return;
  }
  auto& msg = *channel->front();
  cb(*msg.src_device, &dst_device, msg.var);
  channel->pop_front();
}
// a wrapper on RecvAsync
void Rendevous::Recv(const PairKey& key, platform::DeviceContext& dst_device,
                     Variable* t, int timeout_ms) {
  size_t keyhash = Hash64(key);
  std::condition_variable cv;
  std::mutex cv_mtx;
  if () }

}  // namespace framework
}  // namespace paddle
