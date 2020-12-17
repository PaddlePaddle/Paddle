//   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

REGISTER_OP_VERSION(for_pybind_test__)
    .AddCheckpoint("Note 0", framework::compatible::OpVersionDesc()
                                 .BugfixWithBehaviorChanged(
                                     "BugfixWithBehaviorChanged Remark"))
    .AddCheckpoint("Note 1", framework::compatible::OpVersionDesc()
                                 .ModifyAttr("BOOL", "bool", true)
                                 .ModifyAttr("FLOAT", "float", 1.23f)
                                 .ModifyAttr("INT", "int32", -1)
                                 .ModifyAttr("STRING", "std::string",
                                             std::string{"hello"}))
    .AddCheckpoint("Note 2",
                   framework::compatible::OpVersionDesc()
                       .ModifyAttr("BOOLS", "std::vector<bool>",
                                   std::vector<bool>{true, false})
                       .ModifyAttr("FLOATS", "std::vector<float>",
                                   std::vector<float>{2.56f, 1.28f})
                       .ModifyAttr("INTS", "std::vector<int32>",
                                   std::vector<int32_t>{10, 100})
                       .NewAttr("LONGS", "std::vector<int64>",
                                std::vector<int64_t>{10000001, -10000001}))
    .AddCheckpoint("Note 3", framework::compatible::OpVersionDesc()
                           .NewAttr("STRINGS", "std::vector<std::string>",
                                std::vector<std::string>{"str1", "str2"})
                                .ModifyAttr("LONG", "int64", static_cast<int64_t>(10000001))
                                 .NewInput("NewInput", "NewInput_")
                                 .NewOutput("NewOutput", "NewOutput_")
                                 .BugfixWithBehaviorChanged(
                                     "BugfixWithBehaviorChanged_"));
