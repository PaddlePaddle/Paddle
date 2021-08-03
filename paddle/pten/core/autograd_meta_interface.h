/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

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

namespace pt {

class Tensor;

/**
 * AutogradMetaInterface is designed to split compilation unit of Pten with
 * eager. In Pten, all tensor related usage should only care about basic 
 * computation logic instead of eager execution. So, there is no need for
 * Pten to depends on autograd related unit. However, we need tensor to hold
 * autograd info since, tensor is what we only have in both forward and backward
 * computation. 
 * 
 * In this case, we designed a AutogradMetaInterface, which is defined in Pten
 * and has no other members to use. In eager mode, we can cast it as AutogradMeta
 * which is its derived class. Then we use it in eager execution. 
 * 
 * TODO:(jiabin) May be we need some virtual method? It seeems no need now.
 * 
 *  **/


class AutogradMetaInterface {
 public:
  virtual ~AutogradMetaInterface() = 0;
  // TODO(yangjiabin): design other methods
};

}  // namespace pt
