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

#include <string>
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/pir/include/core/dll_decl.h"
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
void IR_API WriteModule(const pir::Program& program,
                        const std::string& file_path,
                        uint64_t pir_version,
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
 * @return bool. The function modifies the 'program' object to contain the data
 * read from the file. return bool indicates whether the program can use to
 * funtune.
 *
 * @note If 'pir_version' is larger than the version of file, will trigger
 * version compatibility modification rule.
 */
bool IR_API ReadModule(const std::string& file_path,
                       pir::Program* program,
                       int64_t pir_version = -1);

/**
 * @brief Save the given tensor into a single file at the specified file path
 * with its name.
 *
 * @param[in] x                 The tensor to be saved.
 * @param[in] name              The name of the tensor
 * @param[in] file_path         The path of the file to be written.
 * @param[in] overwrite         If the file already exists, this flag determines
 *                              whether to overwrite the existing file.
 * @param[in] save_as_fp16      If the flag is true, the tensor will be saved as
 * fp16 type.
 *
 * @return void。
 *
 */
void IR_API SaveFunction(const phi::DenseTensor& x,
                         const std::string& name,
                         const std::string& file_path,
                         bool overwrite,
                         bool save_as_fp16);

/**
 * @brief Save the given tensor list into a combined file at the specified file
 * path with the given name.
 *
 * @param[in] x                 The tensor list to be saved.
 * @param[in] name              The names of the tensors.
 * @param[in] file_path         The path of the file to be written.
 * @param[in] overwrite         If the file already exists, this flag determines
 *                              whether to overwrite the existing file.
 * @param[in] save_as_fp16      If the flag is true, the tensor will be saved as
 * fp16 type.
 *
 * @param[in] save_to_memory    If the flag is true, the tensor will be saved in
 * memory.
 *
 * @return void。
 *
 */
void IR_API SaveCombineFunction(const std::vector<const phi::DenseTensor*>& x,
                                const std::vector<std::string>& names,
                                const std::string& file_path,
                                bool overwrite,
                                bool save_as_fp16,
                                bool save_to_memory);

/**
 * @brief Save the given tensor into a single file at the specified file path
 * with its name.
 *
 * @param[in] file_path         The path of the file to be read.
 * @param[in] seek              The position of the file to be read.
 * @param[in] shape             The shape of the tensor to be loaded.
 * @param[in] load_as_fp16      If the flag is true, the tensor will be loaded
 * as fp16 type.
 * @param[out] out              The tensor to be loaded.
 *
 * @return void。
 *
 */
void IR_API LoadFunction(const std::string& file_path,
                         int64_t seek,
                         const std::vector<int64_t>& shape,
                         bool load_as_fp16,
                         phi::DenseTensor* out,
                         phi::Place place = phi::Place());

/**
 * @brief Save the given tensor into a single file at the specified file path
 * with its name.
 *
 * @param[in] file_path         The path of the file to be read.
 * @param[in] names             The names of the tensors.
 * @param[out] out              The tensor to be loaded.
 * @param[in] load_as_fp16      If the flag is true, the tensor will be loaded
 * as fp16 type.
 *
 * @return void。
 *
 */
void IR_API LoadCombineFunction(const std::string& file_path,
                                const std::vector<std::string>& names,
                                std::vector<phi::DenseTensor*>* out,
                                bool load_as_fp16,
                                phi::Place place = phi::Place());
}  // namespace pir
