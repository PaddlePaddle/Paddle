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

#include "ProtoServer.h"

namespace paddle {

void ProtoServer::handleRequest(std::unique_ptr<MsgReader> msgReader,
                                ResponseCallback callback) {
  /// 0 for funcName
  /// 1 for proto
  CHECK_GE(msgReader->getNumBlocks(), (size_t)2);

  std::string funcName(msgReader->getNextBlockLength(), 0);
  /// read function name string
  msgReader->readNextBlock(&funcName[0]);
  /// looking up rpc wrapped callback function
  auto it = nameToFuncMap_.find(funcName);
  if (it != nameToFuncMap_.end()) {
#ifndef PADDLE_DISABLE_TIMER
    gettimeofday(&(*(handleRequestBegin_)), nullptr);
#endif
    it->second(std::move(msgReader), callback);
  } else {
    LOG(ERROR) << "Unknown funcName: " << funcName;
    std::vector<iovec> iovs;
    callback(iovs);
  }
}

void ProtoServer::registerServiceFunctionImp(const std::string& funcName,
                                             ServiceFunction func) {
  CHECK(!nameToFuncMap_.count(funcName)) << "Duplicated registration: "
                                         << funcName;
  nameToFuncMap_[funcName] = func;
}

void ProtoClient::send(const char* funcName,
                       const google::protobuf::MessageLite& proto,
                       const std::vector<iovec>& userIovs) {
  std::string protoStr;
  CHECK(proto.SerializeToString(&protoStr));
  std::vector<iovec> iovs;
  iovs.reserve(iovs.size() + 2);
  /// sending function name string, protobuf data and user additional data
  iovs.push_back({(void*)funcName, strlen(funcName)});
  iovs.push_back({&protoStr[0], protoStr.size()});
  iovs.insert(iovs.end(), userIovs.begin(), userIovs.end());
  channel_->writeMessage(iovs);
}

std::unique_ptr<MsgReader> ProtoClient::recv(
    google::protobuf::MessageLite* proto) {
  std::vector<iovec> iovs;
  std::unique_ptr<MsgReader> msgReader = channel_->readMessage();
  CHECK_GE(msgReader->getNumBlocks(), (size_t)1);
  std::string str(msgReader->getNextBlockLength(), 0);
  msgReader->readNextBlock(&str[0]);
  CHECK(proto->ParseFromString(str));
  return msgReader;
}

}  // namespace paddle
