// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

struct LoadConfig {
    int table;
    int mode;
};
// example
struct RequestContext {
  int table_id;
  int num;
  void* keys;
  void* values;
};

class PSClient {
 public:
  PSClient() {}
  PSClient(PSClient &&) = delete;
  PSClient(const PSClient &) = delete;
  virtual void Initialize() = 0;
  virtual void Start() = 0;  // 启动 push、pull 异步线程等
  // Load model
  virtual void Load(void* context) = 0;
  // save model
  virtual void Save(void* context) = 0;
  // pull sparse/dense
  virtual void Pull(void* context) = 0;

  // push sparse/dense grads
  virtual void Push(void* context) = 0;

  // shrink sparse 
  virtual void Delete(void* context) = 0；
  // stop client
  virtual void Stop() = 0;
}

}
}