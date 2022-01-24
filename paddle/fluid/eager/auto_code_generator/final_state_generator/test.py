# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

name = "A"
B = "B"
C = "C"
D = "D"
E = "E"
x = """
class GradNode%s : public egr::GradNodeBase {{
   public:
    GradNode%s() : egr::GradNodeBase() {{}}
    GradNode%s(size_t bwd_in_slot_num, size_t bwd_out_slot_num) : 
  egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {{}}
    ~GradNode%s() override = default;
  
    virtual std::vector<std::vector<egr::EagerTensor>>
  operator()(const
  std::vector<std::vector<egr::EagerTensor>>& grads)
  override;
  
    // SetX, SetY, ...
  {}
    // SetAttrMap
  {}
  
   private:
     // TensorWrappers
  {}
     // Attribute Map
  {}
}};
"""

print(x.format("A", "B", "C", "D"))
