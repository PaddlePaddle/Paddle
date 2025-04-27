// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once
#include "paddle/ap/include/adt/adt.h"
#include "paddle/ap/include/axpr/type.h"
#include "paddle/ap/include/code_module/directory.h"
#include "paddle/ap/include/code_module/file_content.h"
#include "paddle/ap/include/code_module/soft_link.h"

namespace ap::code_module {

template <typename FileT>
using FileImpl = std::variant<FileContent, SoftLink, Directory<FileT>>;

struct File : public FileImpl<File> {
  using FileImpl<File>::FileImpl;
  ADT_DEFINE_VARIANT_METHODS(FileImpl<File>);

  static adt::Result<File> CastFromAxprValue(const axpr::Value& val) {
    if (val.template CastableTo<FileContent>()) {
      ADT_LET_CONST_REF(file_content, val.template CastTo<FileContent>());
      return file_content;
    }
    if (val.template CastableTo<SoftLink>()) {
      ADT_LET_CONST_REF(soft_link, val.template CastTo<SoftLink>());
      return soft_link;
    }
    if (val.template CastableTo<Directory<File>>()) {
      ADT_LET_CONST_REF(directory, val.template CastTo<Directory<File>>());
      return directory;
    }
    return adt::errors::TypeError{"File::CastFromAxprValue() failed."};
  }
};

}  // namespace ap::code_module
