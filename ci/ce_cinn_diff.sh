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

set -x
PATH=/usr/local/bin:${PATH}
echo "PATH=${PATH}" >> ~/.bashrc
ln -sf $(which python3.10) /usr/local/bin/python
ln -sf $(which pip3.10) /usr/local/bin/pip

cp -r /PaddleTest/framework/e2e/PaddleLT_new/support/pr_info.py .
python pr_info.py --pr_id ${PR_ID} --title_keyword CINN
title_num=`grep -o '[0-9]\+' pr_title.log`

git diff --numstat --diff-filter=AMRD $BRANCH | grep paddle/cinn | awk '{print $NF}' | tee pr_filelist1.log
filelist_num1=`cat pr_filelist1.log | wc -l`

git diff --numstat --diff-filter=AMRD $BRANCH | grep paddle/fluid/primitive/ | awk '{print $NF}' | tee pr_filelist2.log
filelist_num2=`cat pr_filelist2.log | wc -l`

git diff --numstat --diff-filter=AMRD $BRANCH | grep paddle/fluid/pir/dialect/operator/interface/infer_symbolic_shape/ | awk '{print $NF}' | tee pr_filelist3.log
filelist_num3=`cat pr_filelist3.log | wc -l`

git diff --numstat --diff-filter=AMRD $BRANCH | grep paddle/fluid/pir/dialect/operator/interface | awk '{print $NF}' | tee pr_filelist4.log
filelist_num4=`cat pr_filelist4.log | wc -l`

sum_num=$((title_num + filelist_num1 + filelist_num2 + filelist_num3 + filelist_num4))
echo "sum_num: ${sum_num}"
