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

#include "paddle/fluid/inference/utils/table_printer.h"

#ifdef WIN32
// suppress the min and max definitions in Windef.h.
#define NOMINMAX
#include <Windows.h>
#else
#include <sys/ioctl.h>
#include <unistd.h>
#endif

#include <algorithm>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

namespace paddle {
namespace inference {

std::string TablePrinter::PrintTable() {
  std::stringstream ss;
  ss << "\n";

  CalcLayout();

  AddRowDivider(ss);
  AddRow(ss, 0);
  AddRowDivider(ss);

  for (size_t i = 1; i < data_.size(); ++i) {
    if (data_[i].empty()) {
      AddRowDivider(ss);
    } else {
      AddRow(ss, i);
    }
  }

  AddRowDivider(ss);

  return ss.str();
}

TablePrinter::TablePrinter(const std::vector<std::string>& header) {
  size_t terminal_witdh = 500;
#ifdef WIN32
  CONSOLE_SCREEN_BUFFER_INFO csbi;
  int ret = GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &csbi);
  if (ret && (csbi.dwSize.X != 0)) {
    terminal_witdh = csbi.dwSize.X;
  }
#else
  struct winsize terminal_size;
  int status = ioctl(STDOUT_FILENO, TIOCGWINSZ, &terminal_size);
  if (status == 0 && terminal_size.ws_col != 0) {
    terminal_witdh = terminal_size.ws_col;
  }
#endif

  size_t num_cols = header.size();
  for (size_t i = 0; i < num_cols; ++i) {
    widths_.emplace_back(0);
  }

  terminal_witdh = terminal_witdh - (2 * num_cols) - (num_cols + 1);
  int avg_width = terminal_witdh / num_cols;

  for (size_t i = 0; i < num_cols; ++i) {
    shares_.emplace_back(avg_width);
  }

  InsertRow(header);
}

void TablePrinter::InsertRow(const std::vector<std::string>& row) {
  std::vector<std::vector<std::string>> table_row;
  size_t max_height = 0;

  for (size_t i = 0; i < row.size(); ++i) {
    table_row.emplace_back(std::vector<std::string>());
    std::stringstream ss(row[i]);
    std::string line;
    size_t max_width = 0;
    while (std::getline(ss, line, '\n')) {
      table_row[i].emplace_back(line);
      if (line.length() > max_width) max_width = line.length();
    }

    if (max_width > widths_[i]) widths_[i] = max_width;

    size_t num_lines = table_row[i].size();
    if (num_lines > max_height) max_height = num_lines;
  }

  heights_.emplace_back(max_height);
  data_.emplace_back(table_row);
}

void TablePrinter::InsetDivider() {
  heights_.emplace_back(1);
  data_.emplace_back(std::vector<std::vector<std::string>>());
}

void TablePrinter::CalcLayout() {
  size_t field_num = widths_.size();
  std::vector<size_t> idx(field_num);
  std::iota(idx.begin(), idx.end(), 0);

  std::stable_sort(idx.begin(), idx.end(), [this](size_t i1, size_t i2) {
    return this->widths_[i1] < this->widths_[i2];
  });

  for (auto it = idx.begin(); it != idx.end(); ++it) {
    // If a column not used all the space allocated to it
    if (widths_[*it] < shares_[*it]) {
      float remain = shares_[*it] - widths_[*it];
      shares_[*it] -= remain;

      if (it == idx.end() - 1) break;

      auto next_it = it + 1;
      float remain_per_column = remain / (idx.end() - next_it);
      for (; next_it != idx.end(); ++next_it) {
        shares_[*next_it] += remain_per_column;
      }
    }
  }

  for (auto it = idx.begin(); it != idx.end(); ++it) {
    shares_[*it] = static_cast<size_t>(shares_[*it]);
  }

  // For each record.
  for (size_t i = 0; i < data_.size(); ++i) {
    // For each field in the record.
    for (size_t j = 0; j < data_[i].size(); ++j) {
      // For each line in the field.
      for (size_t line_index = 0; line_index < data_[i][j].size();
           ++line_index) {
        std::string line = data_[i][j][line_index];
        size_t num_rows = (line.length() + shares_[j] - 1) / shares_[j];

        // If the number of rows required for this record is larger than 1, we
        // will break that line and put it in multiple lines
        if (num_rows > 1) {
          data_[i][j].erase(data_[i][j].begin() + line_index);
          for (size_t k = 0; k < num_rows; ++k) {
            size_t start =
                std::min(static_cast<size_t>(k * shares_[j]), line.length());
            size_t end = std::min(static_cast<size_t>((k + 1) * shares_[j]),
                                  line.length());
            data_[i][j].insert(data_[i][j].begin() + line_index + k,
                               line.substr(start, end - start));
          }

          // update line_index
          line_index += num_rows - 1;
        }

        if (heights_[i] < (num_rows - 1 + data_[i][j].size()))
          heights_[i] += num_rows - 1;
      }
    }
  }
}

void TablePrinter::AddRowDivider(std::stringstream& ss) {
  ss << "+";
  for (auto share : shares_) {
    for (size_t j = 0; j < share + 2; ++j) ss << "-";
    ss << "+";
  }
  ss << "\n";
}

void TablePrinter::AddRow(std::stringstream& ss, size_t row_idx) {
  auto row = data_[row_idx];
  size_t max_height = heights_[row_idx];

  for (size_t h = 0; h < max_height; ++h) {
    ss << "|" << std::left;
    for (size_t i = 0; i < row.size(); ++i) {
      if (h < row[i].size()) {
        ss << " " << std::setw(shares_[i]) << row[i][h] << " |";
      } else {
        ss << " " << std::setw(shares_[i]) << " "
           << " |";
      }
    }
    ss << "\n";
  }
}

}  // namespace inference
}  // namespace paddle
