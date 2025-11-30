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

function init_env() {
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib64/:/usr/local/lib/
  export PATH=/usr/local/bin:${PATH}
}

function formers_api() {
  init_env
  cd /workspace/PaddleFormers && git config --global --add safe.directory $PWD
  echo "Check whether the local model file exists:"
  ls -l ./models
  sed -i '/python setup.py bdist_wheel > \/dev\/null/d' scripts/unit_test/ci_unittest.sh
  timeout 30m bash scripts/unit_test/ci_unittest.sh ${paddle_whl} false ${PYTEST_EXECUTE_FLAG_FILE} ${BRANCH}
}

function formers_models() {
  init_env
  rm -rf /root/.cache/aistudio/
  cd /workspace/PaddleFormers && git config --global --add safe.directory $PWD
  echo "Check whether the local model file exists:"
  ls -l ./models
  sed -i '/python setup.py bdist_wheel > \/dev\/null/d' scripts/regression/ci_model_unittest.sh
  timeout 30m bash scripts/regression/ci_model_unittest.sh ${paddle_whl} ${BRANCH}
}

function install_deps() {
  set -e
  init_env
  ln -sf $(which python3.10) /usr/local/bin/python
  ln -sf $(which pip3.10) /usr/local/bin/pip

  echo "Downloading PaddleFormers.tar.gz..."
  wget -q https://paddleformers.bj.bcebos.com/wheels/PaddleFormers.tar.gz
  tar xf PaddleFormers.tar.gz
  echo "Extracting PaddleFormers.tar.gz..."

  if [ -d "PaddleFormers" ]; then
      cd PaddleFormers
      cp -r ${CFS_DIR}/models ./models
  else
      echo "Error: PaddleFormers dir not found after tar xf"
      exit 1
  fi

  echo "::group::Install paddle dependencies"
  pip config set global.cache-dir "/root/.cache/pip"
  pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
  echo "::endgroup::"
  ldconfig
}

case "$1" in
  setup)
    install_deps
    ;;
  api)
    formers_api
    ;;
  models)
    formers_models
    ;;
  *)
    echo "Usage: $0 {setup|api|models}"
    exit 1
    ;;
esac
