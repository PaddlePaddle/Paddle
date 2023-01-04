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

set -e
set +x
export PADDLE_ROOT="$(cd "$PWD/../" && pwd )"
GITHUB_API_TOKEN=$GITHUB_API_TOKEN
GIT_PR_ID=$AGILE_PULL_ID
BRANCH=$BRANCH
if [ "${GITHUB_API_TOKEN}" == "" ] || [ "${GIT_PR_ID}" == "" ];then
    exit 0 
fi

unittest_spec_diff=$(cat $PADDLE_ROOT/deleted_ut | sed 's/^/ - /g')
if [ "$unittest_spec_diff" != "" ]; then
    approval_line=`curl -H "Authorization: token ${GITHUB_API_TOKEN}" https://api.github.com/repos/PaddlePaddle/Paddle/pulls/${GIT_PR_ID}/reviews?per_page=10000`
    APPROVALS=`echo ${approval_line}|python ${PADDLE_ROOT}/tools/check_pr_approval.py 1 22165420 52485244 32428676 45041955`
    echo "current pr ${GIT_PR_ID} got approvals: ${APPROVALS}"
    if [ "${APPROVALS}" == "FALSE" ]; then
        echo "************************************"
        echo -e "It is forbidden to disable or delete the unit-test.\n"
        echo -e "If you must delete it temporarily, please add it to[https://github.com/PaddlePaddle/Paddle/wiki/Temporarily-disabled-Unit-Test]."
        echo -e "Then you must have one RD (kolinwei(recommended), chalsliu, XieYunshen or zhouwei25) approval for the deletion of unit-test. \n"
        echo -e "If you have any problems about deleting unit-test, please read the specification [https://github.com/PaddlePaddle/Paddle/wiki/Deleting-unit-test-is-forbidden]. \n"
        echo -e "Following unit-tests are deleted in this PR: \n${unittest_spec_diff} \n"
        echo "************************************"
        exit 6
    fi
fi
set -x
