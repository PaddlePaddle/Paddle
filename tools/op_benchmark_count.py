# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import argparse
import re
from collections import defaultdict

gpu_time_categories = {
    "within_1%": 0,
    "increase_1_to_5%": 0,
    "increase_above_5_to_10%": 0,
    "increase_above_10%": 0,
    "decrease_1_to_5%": 0,
    "decrease_above_5%": 0,
}

total_time_categories = {
    "within_1%": 0,
    "increase_1_to_5%": 0,
    "increase_above_5_to_10%": 0,
    "increase_above_10%": 0,
    "decrease_1_to_5%": 0,
    "decrease_above_5%": 0,
}

parser = argparse.ArgumentParser(
    description="Analyze time changes in log files"
)
parser.add_argument('file_name', type=str, help='The name of the log file')
args = parser.parse_args()

gpu_time_pattern = re.compile(r"GPU time change: ([\d.-]*)")
total_time_pattern = re.compile(r"Total time change: ([\d.-]+)%")
error_pattern = re.compile(r'Check speed result with case "(.*?)"')

gpu_time_lines = 0
error_cases = defaultdict(int)

with open(args.file_name, 'r') as file:
    for line in file:
        if "GPU time change" in line:
            gpu_time_lines += 1
            gpu_time_match = gpu_time_pattern.search(line)
            if gpu_time_match:
                gpu_time_change_str = gpu_time_match.group(1)
                gpu_time_change = (
                    float(gpu_time_change_str) if gpu_time_change_str else 0.0
                )

                if -1 < gpu_time_change < 1:
                    gpu_time_categories["within_1%"] += 1
                elif 1 <= gpu_time_change < 5:
                    gpu_time_categories["increase_1_to_5%"] += 1
                elif 5 <= gpu_time_change < 10:
                    gpu_time_categories["increase_above_5_to_10%"] += 1
                elif gpu_time_change >= 10:
                    gpu_time_categories["increase_above_10%"] += 1
                elif -5 < gpu_time_change <= -1:
                    gpu_time_categories["decrease_1_to_5%"] += 1
                elif gpu_time_change <= -5:
                    gpu_time_categories["decrease_above_5%"] += 1

        elif "Total time change" in line:
            total_time_match = total_time_pattern.search(line)
            if total_time_match:
                total_time_change = float(total_time_match.group(1))

                if -1 < total_time_change < 1:
                    total_time_categories["within_1%"] += 1
                elif 1 <= total_time_change < 5:
                    total_time_categories["increase_1_to_5%"] += 1
                elif 5 <= total_time_change < 10:
                    total_time_categories["increase_above_5_to_10%"] += 1
                elif total_time_change >= 10:
                    total_time_categories["increase_above_10%"] += 1
                elif -5 < total_time_change <= -1:
                    total_time_categories["decrease_1_to_5%"] += 1
                elif total_time_change <= -5:
                    total_time_categories["decrease_above_5%"] += 1

        elif error_pattern.search(line):
            error_match = error_pattern.search(line)
            if error_match:
                case_name = error_match.group(1)
                error_cases[case_name] += 1


def print_categories(categories, title):
    total = sum(categories.values())
    print(f"\n{title} Categories:")
    for category, count in categories.items():
        percentage = (count / total * 100) if total > 0 else 0
        print(f"{category}: {count} ({percentage:.2f}%)")


print_categories(gpu_time_categories, "GPU Time Change")
print_categories(total_time_categories, "Total Time Change")

total_errors = sum(error_cases.values())
error_percentage = (
    (total_errors / gpu_time_lines * 100) if gpu_time_lines > 0 else 0
)
unique_errors = len(error_cases)

print(f"\nError Cases Total: {total_errors}")
print(f"Error Lines Percentage: {error_percentage:.2f}%")
print(f"Unique Error OP: {unique_errors}\n")

for case, count in error_cases.items():
    print(f"OP '{case}': {count} occurrences")
