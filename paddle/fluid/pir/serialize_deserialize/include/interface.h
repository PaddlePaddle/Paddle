// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
#pragma once

#include "paddle/pir/include/core/program.h"
namespace pir {
/**
 * @brief Write the given PIR program into a file at the specified file path.
 *
 * @param[in] program      The PIR program to be written.
 * @param[in] file_path    The path to the file to be written.
 * @param[in] pir_version  The version number of PIR, used to identify or verify
 * the written program version
 * @param[in] overwrite    If the file already exists, this flag determines
 * whether to overwrite the existing file.
 * @param[in] readable     (Optional parameter, default to false) If true, the
 * generated file will be has indent structure.
 * @param[in] trainable    (Optional parameter, default to true) If true,
 * operation has opresult_attrs for training like stop_gradient,persistable;
 * Otherwise, it may only has opinfo attrs.
 *
 * @return void。
 *
 * @note readable and trainable Parameters may affect the content and format of
 * the generated file, depending on implementation.
 */
void WriteModule(const pir::Program& program,
                 const std::string& file_path,
                 const uint64_t& pir_version,
                 bool overwrite,
                 bool readable = false,
                 bool trainable = true);

/**
 * @brief Gets a PIR program from the specified file path.
 *
 * @param[in] file_path    The path to the file from which the PIR program
 * should be read.
 * @param[out] program     A pointer to the PIR program object where the
 * deserilize program will be stored.
 * @param[in] pir_version  The current version of the PIR program format.
 *
 * @return Void. The function modifies the 'program' object to contain the data
 * read from the file.
 *
 * @note If 'pir_version' is larger than the version of file, will trigger
 * version compatibility modification rule.
 */
void ReadModule(const std::string& file_path,
                pir::Program* program,
                const uint64_t& pir_version);
}  // namespace pir
