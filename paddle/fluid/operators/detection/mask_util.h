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

#pragma once
#include <vector>

namespace paddle {
namespace operators {

template <class T>
void Poly2Mask(const T* ploy, int k, int h, int w, uint8_t* mask);

template <class T>
void Poly2Boxes(const std::vector<std::vector<std::vector<T>>>& polys,
                T* boxes);

template <class T>
void Polys2MaskWrtBox(const std::vector<std::vector<T>>& polygons, const T* box,
                      int M, uint8_t* mask);
}  // namespace operators
}  // namespace paddle
