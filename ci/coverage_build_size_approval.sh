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

if [ ${BRANCH} != 'develop' ];then
    exit 0
fi

rm -f coverage_build_size
coverage_build_size_url="https://paddle-github-action.bj.bcebos.com/night/coverage/coverage_build_size"
if ! curl --noproxy '*' -fsSL -o coverage_build_size "${coverage_build_size_url}"; then
    echo "develop coverage build size not found: ${coverage_build_size_url}"
    exit 1
fi

dev_coverage_build_size=`grep -Eo '[0-9]+G' coverage_build_size | head -n1 | sed 's#G##g'`
if [ -z "${dev_coverage_build_size}" ]; then
    echo "invalid develop coverage build size: $(cat coverage_build_size | tr -d '\n')"
    exit 1
fi

pr_coverage_build_size=`echo $buildSize |sed 's#G##g'`

echo "========================================================"
echo "The develop coverage build size is ${dev_coverage_build_size}G"
echo "The develop coverage build size source is ${coverage_build_size_url}"
echo "The pr coverage build size is $buildSize"
echo "========================================================"

diff_coverage_build_size=`echo $(($pr_coverage_build_size - $dev_coverage_build_size))`
set +x
if [ ${diff_coverage_build_size} -gt 3 ]; then
    approval_line=`curl -H "Authorization: token ${GITHUB_TOKEN}" https://api.github.com/repos/PaddlePaddle/Paddle/pulls/${GIT_PR_ID}/reviews?per_page=10000`
    APPROVALS=`echo ${approval_line}|python ${PADDLE_ROOT}/tools/check_pr_approval.py 1 swgu98 luotao1 risemeup1`
    echo "current pr ${GIT_PR_ID} got approvals: ${APPROVALS}"
    if [ "${APPROVALS}" == "FALSE" ]; then
        echo "=========================================================================================="
        echo "This PR make the release paddlepaddle coverage build size growth exceeds 3 G, please explain why your PR exceeds 3G to ext_ppee@baidu.com and in PR description."
        echo "Then you must have one RD (swgu98 (Recommend) or luotao1 or risemeup1) approval for this PR\n"
        echo "=========================================================================================="
        exit 6
    fi
fi
