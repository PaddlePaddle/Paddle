// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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


namespace paddle {
namespace ps {

class ValueAccessor {
 public:
  ValueAccessor() {}
  virtual ~ValueAccessor() {}
  virtual int initialize() = 0;
  virtual int32_t GetTableInfo() = 0;  //  获取 dim、size 等
  virtual int32_t Add() = 0;  // 新增 key-value
  virtual int32_t Delete() = 0;  
  virtual int32_t Update() = 0;  
  virtual int32_t Query() = 0;  
  virtual void Release() = 0;  //  销毁

}
}