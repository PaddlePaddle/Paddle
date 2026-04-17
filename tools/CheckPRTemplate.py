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

import os
import re
import sys

import httpx
import requests

PR_checkTemplate = ['Paddle']

BRANCH = os.environ['BRANCH']
if BRANCH.startswith("develop"):
    REPO_TEMPLATE = {
        "Paddle": r'''### PR Category(.*[^\s].*)### PR Types(.*[^\s].*)### Description(.*[^\s].*)### 是否引起精度变化(.*[^\s].*)'''
    }
elif BRANCH.startswith("release"):
    REPO_TEMPLATE = {
        "Paddle": r'''### PR Category(.*[^\s].*)### PR Types(.*[^\s].*)### Description(.*?devPR:https://github\.com/PaddlePaddle/Paddle/pull/.*?)(?:\n###|\Z)'''
    }
else:
    REPO_TEMPLATE = {
        "Paddle": r'''### PR Category(.*[^\s].*)### PR Types(.*[^\s].*)### Description(.*[^\s].*)'''
    }


def re_rule(body, CHECK_TEMPLATE):
    PR_RE = re.compile(CHECK_TEMPLATE, re.DOTALL)
    result = PR_RE.search(body)
    return result


def extract_pr_links(description_text):
    pattern = r'(?:https://github\.com/PaddlePaddle/Paddle/pull/|#)(\d+)'
    return re.findall(pattern, description_text)


def check_link_accessible(url):
    try:
        response = requests.head(url, allow_redirects=True, timeout=5)
        return response.status_code == 200
    except requests.RequestException:
        return False


def parameter_accuracy(body):
    PR_dic = {}
    PR_Category = [
        'User Experience',
        'Execute Infrastructure',
        'Operator Mechanism',
        'CINN',
        'Custom Device',
        'Performance Optimization',
        'Distributed Strategy',
        'Parameter Server',
        'Communication Library',
        'Auto Parallel',
        'Inference',
        'Environment Adaptation',
    ]
    PR_Types = [
        'New features',
        'Bug fixes',
        'Improvements',
        'Performance',
        'BC Breaking',
        'Deprecations',
        'Docs',
        'Devs',
        'Not User Facing',
        'Security',
        'Others',
    ]

    Accuracy_Change = [
        '是',
        '否',
    ]
    body = re.sub("\r\n", "", body)
    type_end = body.find('### PR Types')
    changes_end = body.find('### Description')
    PR_dic['PR Category'] = body[len('### PR Category') : type_end]
    PR_dic['PR Types'] = body[type_end + len('### PR Types') : changes_end]
    message = ''
    for key in PR_dic:
        test_list = PR_Category if key == 'PR Category' else PR_Types
        test_list_lower = [l.lower() for l in test_list]
        value = PR_dic[key].strip().split(',')
        single_mess = ''
        if len(value) == 1 and value[0] == '':
            message += f'{key} should be in {test_list}. but now is None.'
        else:
            for i in value:
                i = i.strip().lower()
                if i not in test_list_lower:
                    single_mess += f'{i}.'
            if len(single_mess) != 0:
                message += f'{key} should be in {test_list}. but now is [{single_mess}].'
    if BRANCH.startswith("release"):
        PR_dic['Description'] = body[changes_end + len('### Description') :]
        des_pr_id = extract_pr_links(PR_dic['Description'])
        if len(des_pr_id) == 0 or not check_link_accessible(
            "https://github.com/PaddlePaddle/Paddle/pull/" + str(des_pr_id[0])
        ):
            message += 'The PR link does not exist. To merge into the release branch, you need to merge into the develop branch first and then cherry-pick it to the release branch. Please merge into develop first and fill in the PR link in the Description. Use this format: devPR:https://github.com/PaddlePaddle/Paddle/pull/xxxx'

    if BRANCH.startswith("develop"):
        accuracy_start = body.find('### 是否引起精度变化')
        if accuracy_start != -1:
            # 确保description_start在accuracy_start之后
            PR_dic['Precision Change Impact'] = body[
                accuracy_start + len('### 是否引起精度变化') :
            ]
        else:
            PR_dic['Precision Change Impact'] = ''
        accuracy_value = PR_dic.get('Precision Change Impact', '').strip()
        print(f'Extracted Precision Change Impact: "{accuracy_value}"')
        if not accuracy_value:
            message += '必须填写是否引起精度变化'
        else:
            found_valid = False
            for option in Accuracy_Change:
                if option in accuracy_value:
                    found_valid = True
                    break
            if not found_valid:
                message += f'精度变化必须填写为：{Accuracy_Change}. 现在是 {accuracy_value}.'
    return message


def check_precision_change_approval(body, pr_number, pr_user):
    """Check if PR with precision change has sufficient approval"""
    # Only check for develop branch
    if not BRANCH.startswith("develop"):
        return True, "Not develop branch, skip precision change approval check"

    # Check if involves precision change
    body_without_comments = re.sub(r'<!--.*?-->', '', body, flags=re.DOTALL)
    accuracy_start = body_without_comments.find('### 是否引起精度变化')
    if accuracy_start != -1:
        precision_text = body_without_comments[
            accuracy_start + len('### 是否引起精度变化') :
        ]
    else:
        precision_text = ''
        return False, '未匹配到是否引起精度变化字段，无法判断是否涉及精度变化'
    precision_text = precision_text.strip()

    print(f'Extracted precision text: "{precision_text}"')
    has_precision_change = '是' in precision_text

    if not has_precision_change:
        return True, "不涉及精度变化，无需检查"
    REQUIRED_APPROVERS = [
        'From00',
        'lugimzzz',
        'Jiang-Jia-Jun',
        'wanghuancoder',
    ]
    REQUIRED_APPROVERS_LOWER = [user.lower() for user in REQUIRED_APPROVERS]
    print(
        f"PR {pr_number} involves precision change, checking approvals from required approvers: {REQUIRED_APPROVERS}"
    )

    # If has precision change, check approval status
    reviews_url = f"https://api.github.com/repos/PaddlePaddle/Paddle/pulls/{pr_number}/reviews"
    headers = {
        'Authorization': 'token ' + GITHUB_API_TOKEN,
        'Accept': 'application/vnd.github+json',
    }

    try:
        response = httpx.get(reviews_url, headers=headers, timeout=10)
        if response.status_code != 200:
            return (
                False,
                f"Cannot get review information: {response.status_code}",
            )

        reviews = response.json()

        # Check for approved reviews
        approved_by = {}
        for approver in REQUIRED_APPROVERS:
            approved_by[approver] = False

        if pr_user.lower() in REQUIRED_APPROVERS_LOWER:
            approved_by[pr_user] = True
            print(f"  ✓ Approved by PR author {pr_user}")

        # Check all reviews
        for review in reviews:
            if review['state'] == 'APPROVED':
                reviewer = review['user']['login']
                if reviewer.lower() in REQUIRED_APPROVERS_LOWER:
                    # Find the correct casing for this reviewer
                    for approver in REQUIRED_APPROVERS:
                        if approver.lower() == reviewer.lower():
                            approved_by[approver] = True
                            print(f"  ✓ Approved by {approver}")
                            break
        # Check if all required approvers have approved
        missing_approvers = [
            approver
            for approver, has_approved in approved_by.items()
            if not has_approved
        ]
        if len(missing_approvers) == 0:
            approvers_list = ", ".join(REQUIRED_APPROVERS)
            return (
                True,
                f"✅ All required approvers have approved: {approvers_list}",
            )
        else:
            approved_list = [a for a in REQUIRED_APPROVERS if approved_by[a]]
            missing_list = ", ".join(missing_approvers)

            if len(approved_list) > 0:
                approved_str = ", ".join(approved_list)
                return (
                    False,
                    f"❌ Missing approvals from: {missing_list}. Approved by: {approved_str}",
                )
            else:
                return (
                    False,
                    f"❌ No approvals from required approvers. Missing: {missing_list}",
                )
    except Exception as e:
        return False, f"Error checking approval: {e!s}"


def checkComments(url):
    headers = {
        'Authorization': 'token ' + GITHUB_API_TOKEN,
    }
    response = httpx.get(
        url, headers=headers, timeout=None, follow_redirects=True
    ).json()
    return response


def checkPRTemplate(repo, body, CHECK_TEMPLATE):
    """
    Check if PR's description meet the standard of template
    Args:
        body: PR's Body.
        CHECK_TEMPLATE: check template str.
    Returns:
        res: True or False
    """
    res = False
    comment_pattern = re.compile(r'<!--.*?-->', re.DOTALL)
    if body is None:
        body = ''
    body = comment_pattern.sub('', body)
    result = re_rule(body, CHECK_TEMPLATE)
    message = ''
    if len(CHECK_TEMPLATE) == 0 and len(body) == 0:
        res = False
    elif result is not None:
        message = parameter_accuracy(body)
        res = True if message == '' else False
    elif result is None:
        res = False
        message = parameter_accuracy(body)
        if BRANCH.startswith("release") and len(message) == 0:
            message = 'The PR link does not exist. To merge into the release branch, you need to merge into the develop branch first and then cherry-pick it to the release branch. Please merge into develop first and fill in the PR link in the Description. Use this format: devPR:https://github.com/PaddlePaddle/Paddle/pull/xxxx'
    return res, message


def pull_request_event_template(event, repo, *args, **kwargs):
    pr_effect_repos = PR_checkTemplate
    pr_num = event['number']
    url = event["comments_url"]
    BODY = event["body"]
    sha = event["head"]["sha"]
    title = event["title"]
    pr_user = event["user"]["login"]
    print(f'receive data : pr_num: {pr_num}, title: {title}, user: {pr_user}')
    if repo in pr_effect_repos:
        CHECK_TEMPLATE = REPO_TEMPLATE[repo]
        global check_pr_template
        global check_pr_template_message
        check_pr_template, check_pr_template_message = checkPRTemplate(
            repo, BODY, CHECK_TEMPLATE
        )
        print(f"check_pr_template: {check_pr_template} pr: {pr_num}")
        if check_pr_template is False:
            print("ERROR MESSAGE:", check_pr_template_message)
            sys.exit(7)
        else:
            print("check approve")
            approval_ok, approval_msg = check_precision_change_approval(
                BODY, pr_num, pr_user
            )
            print(
                f"Approval check result: {approval_ok}, message: {approval_msg}"
            )
            if not approval_ok:
                check_pr_template_message = approval_msg
                print("ERROR MESSAGE:", check_pr_template_message)
                sys.exit(8)
            sys.exit(0)


def get_a_pull(pull_id):
    url = "https://api.github.com/repos/PaddlePaddle/Paddle/pulls/ " + str(
        pull_id
    )

    payload = {}
    headers = {
        'Authorization': 'token ' + GITHUB_API_TOKEN,
        'Accept': 'application/vnd.github+json',
    }
    response = httpx.request(
        "GET", url, headers=headers, data=payload, follow_redirects=True
    )
    return response.json()


def main(org, repo, pull_id):
    pull_info = get_a_pull(pull_id)
    pull_request_event_template(pull_info, repo)


if __name__ == "__main__":
    AGILE_PULL_ID = os.getenv("AGILE_PULL_ID")
    GITHUB_API_TOKEN = os.getenv("GITHUB_API_TOKEN")
    main("PaddlePaddle", "Paddle", AGILE_PULL_ID)
