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

#include "BaseClient.h"
#include <gflags/gflags.h>
#include <string.h>
#include <vector>
#include "paddle/utils/Stat.h"

DECLARE_string(pservers);

namespace paddle {

BaseClient::BaseClient(bool separate, int numPorts)
    : stopping_(false), numPorts_(numPorts), separateSendAndRecv_(separate) {
  CHECK_GT(numPorts, 0);
}

BaseClient::~BaseClient() {}

void BaseClient::recvData() { recvSyncBarrier_->wait(); }

void BaseClient::synchronize(SyncObject syncObjectId) {
  SynchronizeRequest request;
  request.set_sync_object_id(syncObjectId);
  std::vector<SynchronizeResponse> responses;
  multiCall(__func__, request, &responses);
}

void BaseClient::startThreads() {
  if (!separateSendAndRecv_) {
    return;
  }
  recvSyncBarrier_.reset(new ThreadBarrier(threadNum_ + 1));

  sendThreads_.resize(threadNum_);
  recvThreads_.resize(threadNum_);
  sendJobQueue_.resize(threadNum_);
  recvJobQueue_.resize(threadNum_);

  for (int i = 0; i < threadNum_; ++i) {
    sendJobQueue_[i].reset(new SendQueue());
    recvJobQueue_[i].reset(new SendQueue());

    sendThreads_[i].reset(
        new std::thread([this](int id) { this->send(id); }, i));

    recvThreads_[i].reset(
        new std::thread([this](int id) { this->recv(id); }, i));
  }
}

void BaseClient::finishThreads() {
  if (!separateSendAndRecv_) {
    return;
  }
  stopping_ = true;
  for (int i = 0; i < threadNum_; i++) {
    sendJobQueue_[i]->enqueue(nullptr);
  }
  for (auto& thread : sendThreads_) {
    thread->join();
  }
  for (auto& thread : recvThreads_) {
    thread->join();
  }
  stopping_ = false;
}
}  // namespace paddle
