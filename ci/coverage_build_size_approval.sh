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
    return
fi

rm -f build_size
curl --noproxy '*' -O https://paddle-docker-tar.bj.bcebos.com/paddle_ci_index/build_size
dev_coverage_build_size=`cat build_size|sed 's#G##g'`
pr_coverage_build_size=`echo "$1" |sed 's#G##g'`

echo "========================================================"
echo "The develop coverage build size is $(cat build_size | tr -d '\n')"
echo "The pr coverage build size is $1"
echo "========================================================"

diff_coverage_build_size=`echo $(($pr_coverage_build_size - $dev_coverage_build_size))`
set +x
if [ ${diff_coverage_build_size} -gt 3 ]; then
    approval_line=`curl -H "Authorization: token ${GITHUB_TOKEN}" https://api.github.com/repos/PaddlePaddle/Paddle/pulls/${GIT_PR_ID}/reviews?per_page=10000`
    APPROVALS=`echo ${approval_line}|python ${PADDLE_ROOT}/tools/check_pr_approval.py 1 29832297 6836917 43953930`
    echo "current pr ${GIT_PR_ID} got approvals: ${APPROVALS}"
    if [ "${APPROVALS}" == "FALSE" ]; then
        echo "=========================================================================================="
        echo "This PR make the release paddlepaddle coverage build size growth exceeds 3 G, please explain why your PR exceeds 3G to ext_ppee@baidu.com and in PR description."
        echo "Then you must have one RD (tianshuo78520a (Recommend) or luotao1 or phlrain) approval for this PR\n"
        echo "=========================================================================================="
        exit 6
    fi
fi
