/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/parameter/Parameter.h"
#include "PaddleAPI.h"
#include "PaddleAPIPrivate.h"

Parameter::Parameter() : m(new ParameterPrivate()) {}

Parameter::~Parameter() { delete m; }

Parameter* Parameter::createFromRawPtr(void* ptr) {
  auto p = new Parameter();
  p->m->rawPtr = *static_cast<paddle::Parameter**>(ptr);
  return p;
}

Parameter* Parameter::createFromSharedPtr(void* ptr) {
  auto& p = *(paddle::ParameterPtr*)(ptr);
  if (p == nullptr) {
    return nullptr;
  } else {
    auto retParam = new Parameter();
    retParam->m->sharedPtr = p;
    return retParam;
  }
}

std::string Parameter::getName() const { return m->getPtr()->getName(); }

Vector* Parameter::getBuf(ParameterType type) {
  auto buf = m->getPtr()->getBuf(type);
  return Vector::createByPaddleVectorPtr(&buf);
}

ParameterConfig* Parameter::getConfig() {
  if (m->sharedPtr) {
    return ParameterConfig::createParameterConfigFromParameterSharedPtr(
        &m->sharedPtr);
  } else {
    return ParameterConfig::createParameterConfigFromParameterPtr(m->rawPtr);
  }
}

size_t Parameter::getID() const { return m->getPtr()->getID(); }

void Parameter::setValueUpdated() { m->getPtr()->setValueUpdated(); }

bool Parameter::save(const std::string& filename) const {
  return m->getPtr()->save(filename);
}

bool Parameter::load(const std::string& filename) const {
  return m->getPtr()->load(filename);
}

size_t Parameter::getSize() const { return m->getPtr()->getSize(); }
