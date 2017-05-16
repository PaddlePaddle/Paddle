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

#pragma once
#include <unordered_map>
#include "paddle/utils/Any.h"
#include "paddle/utils/Error.h"

namespace paddle {
namespace topology {

class Attribute;
namespace meta {
class AttributeMap;
class FunctionMeta;
}  // namespace meta

namespace details {
class AttributeParser
    : public std::unordered_map<std::string,
                                std::function<Error(const any*, Attribute*)>> {
public:
  template <typename T, typename SubClass>
  void append(const std::string& name, T SubClass::*memPtr) {
    (*this)[name] = [memPtr](const any* attr, Attribute* self) {
      auto& ins = dynamic_cast<SubClass&>(*self);
      try {
        ins.*memPtr = any_cast<T>(*attr);
      } catch (bad_any_cast& ex) {
        return Error(ex.what());
      }
      return Error();
    };
  }

  Error operator()(Attribute* instance, const meta::AttributeMap& attrs) const;
};
typedef std::shared_ptr<AttributeParser> AttributeParserPtr;

extern AttributeParserPtr gCurParser;
extern meta::FunctionMeta* gCurFuncMeta;

class ParserScope {
public:
  ParserScope() { gCurParser.reset(new AttributeParser()); }
  ~ParserScope() { gCurParser.reset(); }

  ParserScope(const ParserScope& o) = delete;
  ParserScope& operator=(const ParserScope& o) = delete;
};

class FunctionMetaScope {
public:
  explicit FunctionMetaScope(meta::FunctionMeta* meta) { gCurFuncMeta = meta; }
  ~FunctionMetaScope();

  FunctionMetaScope(const FunctionMetaScope& o) = delete;
  FunctionMetaScope& operator=(const FunctionMetaScope& o) = delete;

private:
  ParserScope withParser_;
};

}  // namespace details
}  // namespace topology
}  // namespace paddle
