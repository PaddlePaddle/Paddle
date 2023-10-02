# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import os


def main():
    all_record = []
    all_files = os.listdir('./')
    all_files = sorted(
        filter(
            lambda file: file.startswith("profile_record_tmp_file_for_rank_"),
            all_files,
        )
    )

    for files in all_files:
        with open(files, 'r') as f:
            for line in f:
                all_record.append(line.strip())

    with open('pipeline_profile.json', 'w') as f:
        f.write('[ ')
        for i in range(len(all_record) - 1):
            f.write(all_record[i] + ',\n')
        f.write(all_record[-1])
        f.write(' ]\n')


if __name__ == "__main__":
    main()
