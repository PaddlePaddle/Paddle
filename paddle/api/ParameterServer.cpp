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

#include "PaddleAPI.h"

#include "PaddleAPIPrivate.h"

ParameterServer::ParameterServer() : m(new ParameterServerPrivate()) {}

ParameterServer* ParameterServer::createByConfigProtoPtr(const void* confPtr) {
  auto& conf = *(const paddle::ParameterServerConfig*)(confPtr);
  auto pServer = new ParameterServer();
  pServer->m->pServerUtil.reset(paddle::PServerUtil::create(conf));
  return pServer;
}

ParameterServer* ParameterServer::createByConfigProtoStr(
    const std::string& protoStr) {
  paddle::ParameterServerConfig conf;
  conf.ParseFromString(protoStr);
  if (conf.IsInitialized()) {
    return ParameterServer::createByConfigProtoPtr(&conf);
  } else {
    return nullptr;
  }
}

ParameterServer::~ParameterServer() { delete m; }

void ParameterServer::start() { m->pServerUtil->start(); }

void ParameterServer::join() { m->pServerUtil->join(); }
