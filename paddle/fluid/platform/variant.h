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

#ifdef __CUDACC__
#ifdef __CUDACC_VER_MAJOR__
// CUDA 9 define `__CUDACC_VER__` as a warning message, manually define
// __CUDACC_VER__ instead.
#undef __CUDACC_VER__

#define __CUDACC_VER__                                         \
  (__CUDACC_VER_MAJOR__ * 10000 + __CUDACC_VER_MINOR__ * 100 + \
   __CUDACC_VER_BUILD__)
#endif

#endif

#include <boost/config.hpp>

#ifdef PADDLE_WITH_CUDA

// Because boost's variadic templates has bug on nvcc, boost will disable
// variadic template support when GPU enabled on nvcc.
// Define BOOST_NO_CXX11_VARIADIC_TEMPLATES on gcc/clang to generate same
// function symbols.
//
// https://github.com/PaddlePaddle/Paddle/issues/3386
#ifndef BOOST_NO_CXX11_VARIADIC_TEMPLATES
#define BOOST_NO_CXX11_VARIADIC_TEMPLATES
#endif
#endif

#include <boost/mpl/comparison.hpp>
#include <boost/mpl/less_equal.hpp>
#include <boost/variant.hpp>
