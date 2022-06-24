# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

set +x

# use pre-commit 2.17
if ! [[ $(pre-commit --version) == *"2.17.0"* ]]; then
    pip install pre-commit==2.17.0 1>nul
fi

# Install clang-format before git commit to avoid repeat installation due to
# pre-commit multi-thread running.
readonly VERSION="13.0.0"
version=$(clang-format -version)
if ! [[ $(python -V 2>&1 | awk '{print $2}' | awk -F '.' '{print $1$2}') -ge 36 ]]; then
    echo "clang-format installation by pip need python version great equal 3.6, 
          please change the default python to higher version."
    exit 1
fi

diff_files=$(git diff --numstat ${BRANCH} | awk '{print $NF}')
echo -e "diff files between pr and ${BRANCH}:\n${diff_files}"

echo "Checking code style by pre-commit ..."
pre-commit run --files ${diff_files};check_error=$?

if test ! -z "$(git diff)"; then
    echo -e '\n************************************************************************************'
    echo -e "These files have been formated by code format hook. You should use pre-commit to \
format them before git push."
    echo -e '************************************************************************************\n'
    git diff 2>&1
fi

echo -e '\n***********************************'
if [ ${check_error} != 0 ];then
    echo "Your PR code style check failed."
else
    echo "Your PR code style check passed."
fi
echo -e '***********************************\n'

exit ${check_error}
