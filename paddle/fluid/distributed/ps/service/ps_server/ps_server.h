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

class PSServer {
public:
    PSServer() {}
    virtual ~PSServer() {}
    PSServer(PSServer&&) = delete;
    PSServer(const PSServer&) = delete;
    virtual void Initialize() = 0;  // 1. 创建 Table 及初始化；2. 初始化 PSServer service；3. 和 coordinator 建立连接
    virtual void Start() = 0;  // 1. 启动 server，并向 PSEnvironment 注册
    virtual int32_t Stop() = 0;
};

}
}