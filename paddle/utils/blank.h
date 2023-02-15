/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

// This file copy from boost/blank.hpp, boost version: 1.41.0
// Modified the following points:
// 1. modify namespace from boost to paddle
// 2. remove the depending boost header files
// 3. remove the type traits specializations
// 4. remove streaming support

//-----------------------------------------------------------------------------
// boost blank.hpp header file
// See http://www.boost.org for updates, documentation, and revision history.
//-----------------------------------------------------------------------------
//
// Copyright (c) 2003
// Eric Friedman
//
// Distributed under the Boost Software License, Version 1.0. (See
// accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#pragma once

namespace paddle {

struct blank {};

inline bool operator==(const blank&, const blank&) { return true; }

inline bool operator<=(const blank&, const blank&) { return true; }

inline bool operator>=(const blank&, const blank&) { return true; }

inline bool operator!=(const blank&, const blank&) { return false; }

inline bool operator<(const blank&, const blank&) { return false; }

inline bool operator>(const blank&, const blank&) { return false; }

}  // namespace paddle
