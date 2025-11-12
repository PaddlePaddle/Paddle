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


function formers_api() {
  cd /workspace/PaddleFormers && git config --global --add safe.directory $PWD
  echo "Check whether the local model file exists:"
  ls -l ./models
  timeout 30m bash scripts/unit_test/ci_unittest.sh ${paddle_whl} false ${PYTEST_EXECUTE_FLAG_FILE} ${BRANCH}
}

function formers_models() {
  rm -rf /root/.cache/aistudio/
  cd /workspace/PaddleFormers && git config --global --add safe.directory $PWD
  echo "Check whether the local model file exists:"
  ls -l ./models
  timeout 30m bash scripts/regression/ci_model_unittest.sh ${paddle_whl} ${BRANCH}
}

function formers_test() {
  python /workspace/tools/get_pr_title.py skip_distribute_test && CINN_OR_BUAA_PR=1
  if [[ "${CINN_OR_BUAA_PR}" = "1" ]];then
      echo "PR's title with 'CINN' or 'BUAA', skip the run distribute ci test !"
      exit 0
  fi

  echo "::group::Start formers api tests"
  formers_api
  echo "End api tests"
  echo "::endgroup::"

  echo "::group::Start formers models tests"
  formers_models
  echo "End models tests"
  echo "::endgroup::"
}

set -e
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib64/:/usr/local/lib/
PATH=/usr/local/bin:${PATH}
ln -sf $(which python3.10) /usr/local/bin/python
ln -sf $(which pip3.10) /usr/local/bin/pip

echo "Downloading PaddleFormers.tar.gz..."
wget -q https://paddleformers.bj.bcebos.com/wheels/PaddleFormers.tar.gz
tar xf PaddleFormers.tar.gz
echo "Extracting PaddleFormers.tar.gz..."
cd PaddleFormers
cp -r ${CFS_DIR}/models ./models

echo "::group::Install paddle dependencies"
pip config set global.cache-dir "/root/.cache/pip"
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
echo "::endgroup::"
ldconfig

formers_test
