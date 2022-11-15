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

#pragma once

#include <string>
#include <vector>

namespace paddle {
namespace inference {

//
// A simple table printer.
//
class TablePrinter {
 public:
  explicit TablePrinter(const std::vector<std::string>& header);

  // Insert a row at the end of the table
  void InsertRow(const std::vector<std::string>& row);

  // Insert a divider.
  void InsetDivider();

  std::string PrintTable();

 private:
  // Update the `shares_` such that all the excess
  // amount of space not used a column is fairly allocated
  // to the other columns
  void CalcLayout();

  // Add a row divider
  void AddRowDivider(std::stringstream& ss);

  // Append a row to `table`. This function handles the cases where a wrapping
  // occurs.
  void AddRow(std::stringstream& ss, size_t row_idx);

 private:
  // Max row width.
  std::vector<float> widths_;

  // Max row height.
  std::vector<float> heights_;

  // Fair share of every column
  std::vector<float> shares_;

  // A vector of vectors of vectors containing data items for every column
  // The record is stored in a vector of string, where each of the vector items
  // contains a single line from the record. For example, ["Item 1", "Item 2",
  // "Item 3 line 1\n Item 3 line 2"] will be stored as [["Item 1"], ["Item 2"],
  // ["Item
  // 3 line 1", "Item 3 line 2"]]
  std::vector<std::vector<std::vector<std::string>>> data_;
};

}  // namespace inference
}  // namespace paddle
