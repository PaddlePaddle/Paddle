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

PATH=/usr/local/bin:${PATH}
ln -sf $(which python3.10) /usr/local/bin/python
ln -sf $(which pip3.10) /usr/local/bin/pip
pip config set global.cache-dir "/home/data/cfs/.cache/pip"

function determine_sot_ci_trigger() {
    set +x
    # use "git commit -m 'message, test=sot'" to force ci to run
    COMMIT_RUN_CI=$(git log -10 --pretty=format:"%s" | grep -w "test=sot" || true)
    # check pr title
    TITLE_RUN_CI=$(curl -s https://github.com/PaddlePaddle/Paddle/pull/${GIT_PR_ID} | grep "<title>" | grep -i "sot" || true)
    if [[ ${COMMIT_RUN_CI} || ${TITLE_RUN_CI} ]]; then
        # set -x
        return
    fi

    # git diff
    SOT_FILE_LIST=(
        .github/workflows
        ci
        paddle/pir
        paddle/phi
        paddle/scripts
        paddle/fluid/eager/to_static
        paddle/fluid/pybind/
        python/
        test/sot
    )

    run_sot_ut="OFF"
    for change_file in $(git diff --name-only upstream/develop);
    do
        for sot_file in ${SOT_FILE_LIST[@]};
        do
            if [[ ${change_file} =~ ^"${sot_file}".* ]]; then
                echo "Detect change about SOT: "
                echo "Changes related to the sot code were detected: " ${change_file}
                run_sot_ut="ON"
                break
            fi
        done
        if [[ "ON" == ${run_sot_ut} ]]; then
            break
        fi
    done

    if [[ "OFF" == ${run_sot_ut} ]]; then
        echo "No SOT-related changes were found"
        echo "Skip SOT UT CI"
        exit 1
    fi
    # set -x
}

determine_sot_ci_trigger
