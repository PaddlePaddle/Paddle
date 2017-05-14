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
#include <memory>
#include <string>
#include <unordered_map>
#include "AttributeMap.h"
#include "meta/AttributeMeta.h"
#include "meta/FunctionMeta.h"

namespace paddle {
namespace topology {

class Attribute;
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

  Error operator()(Attribute* instance, const AttributeMap& attrs) const {
    for (auto it = attrs.begin(); it != attrs.end(); ++it) {
      auto& name = it->first;
      auto parserIt = this->find(name);
      if (parserIt == end()) {
        return Error("Cannot found parser %s", name.c_str());
      }
      auto err = parserIt->second(&it->second, instance);
      if (!err.isOK()) return err;
    }
    return Error();
  }
};
typedef std::shared_ptr<AttributeParser> AttributeParserPtr;

class Attribute {
public:
  bool useGPU;

  virtual ~Attribute();

protected:
  class ParserScope {
  public:
    ParserScope() { Attribute::parser.reset(new AttributeParser()); }
    ~ParserScope() { Attribute::parser.reset(); }
  };

  class FunctionMetaScope {
  public:
    explicit FunctionMetaScope(const meta::FunctionMetaPtr& metaPtr) {
      Attribute::funcMeta = metaPtr;
    }
    ~FunctionMetaScope() {
      auto parser = Attribute::parser;
      Attribute::funcMeta->regAttributeParser<Attribute>(
          [parser](const AttributeMap& attrs, Attribute* instance) {
            return (*parser)(instance, attrs);
          });
      Attribute::funcMeta.reset();
    }

  private:
    ParserScope withParser_;
  };

  template <typename T, typename SubClass>
  static meta::Constraints<T>& regAttr(T SubClass::*memPtr,
                                       const std::string& name,
                                       const std::string& description) {
    parser->append(name, memPtr);
    return funcMeta->addAttribute<T>(name, description);
  }

  static void parentRegAttrs() {
    regAttr(&Attribute::useGPU, "useGPU", "Use GPU or not").mustSet();
  }

  static AttributeParserPtr parser;
  static meta::FunctionMetaPtr funcMeta;
};

#define REGISTER_FUNC_ATTRIBUTE()                              \
  static void registerFunctionAttribute(                       \
      const paddle::topology::meta::FunctionMetaPtr metaPtr) { \
    Attribute::FunctionMetaScope __with_func_meta__(metaPtr);  \
    parentRegAttrs();                                          \
    registerFunctionAttribute__impl__();                       \
  }                                                            \
  static void registerFunctionAttribute__impl__()

}  // namespace topology

}  // namespace paddle
