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

# 1 "CMakeCUDACompilerId.cu"
# 40 "CMakeCUDACompilerId.cu"
extern const char *info_compiler;
# 278 "CMakeCUDACompilerId.cu"
static const char info_version[50];
# 325 "CMakeCUDACompilerId.cu"
extern const char *info_platform;
extern const char *info_arch;

extern const char *info_language_dialect_default;
# 40 "CMakeCUDACompilerId.cu"
const char *info_compiler = ((const char *)"INFO:compiler[NVIDIA]");
# 278 "CMakeCUDACompilerId.cu"
static const char info_version[50] = {
    ((char)73),  ((char)78),  ((char)70),  ((char)79),  ((char)58),
    ((char)99),  ((char)111), ((char)109), ((char)112), ((char)105),
    ((char)108), ((char)101), ((char)114), ((char)95),  ((char)118),
    ((char)101), ((char)114), ((char)115), ((char)105), ((char)111),
    ((char)110), ((char)91),  ((char)48),  ((char)48),  ((char)48),
    ((char)48),  ((char)48),  ((char)48),  ((char)49),  ((char)48),
    ((char)46),  ((char)48),  ((char)48),  ((char)48),  ((char)48),
    ((char)48),  ((char)48),  ((char)48),  ((char)50),  ((char)46),
    ((char)48),  ((char)48),  ((char)48),  ((char)48),  ((char)48),
    ((char)48),  ((char)56),  ((char)57),  ((char)93),  ((char)0)};
# 325 "CMakeCUDACompilerId.cu"
const char *info_platform = ((const char *)"INFO:platform[Linux]");
const char *info_arch = ((const char *)"INFO:arch[]");

const char *info_language_dialect_default =
    ((const char *)"INFO:dialect_default[14]");
