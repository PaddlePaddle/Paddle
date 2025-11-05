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
        "Paddle": r'''### PR Category(.*[^\s].*)### PR Types(.*[^\s].*)### Description(.*[^\s].*)'''
    }
elif BRANCH.startswith("fleety_"):
    REPO_TEMPLATE = {
        "Paddle": r'''### PR Category(.*[^\s].*)### PR Types(.*[^\s].*)### Description(.*?devPR:https://github\.com/PaddlePaddle/Paddle/pull/.*?)(?:\n###|\Z)'''
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
    if BRANCH.startswith("fleety_"):
        PR_dic['Description'] = body[changes_end + len('### Description') :]
        des_pr_id = extract_pr_links(PR_dic['Description'])
        if len(des_pr_id) == 0 or not check_link_accessible(
            "https://github.com/PaddlePaddle/Paddle/pull/" + str(des_pr_id[0])
        ):
            message += 'The PR link does not exist. To merge into the fleety branch, you need to merge into the develop branch first and then cherry-pick it to the fleety branch. Please merge into develop first and fill in the PR link in the Description'
    return message


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
        if BRANCH.startswith("fleety_") and len(message) == 0:
            message = 'The PR link does not exist. To merge into the fleety branch, you need to merge into the develop branch first and then cherry-pick it to the fleety branch. Please merge into develop first and fill in the PR link in the Description'
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
