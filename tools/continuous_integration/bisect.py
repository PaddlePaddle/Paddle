#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

# A script to bisect the mainline commits and find the culprit commit.
# The default 'git bisect' checks feature branches, which is not desired
# because commits in feature branch might not pass tests or compile.
#
# Example:
#   python ../bisect.py --git_dir=$PWD/../Paddle --build_dir=$PWD \
#       --good_commit=3647ed6 --bad_commit=279aa6 \
#       --test_target=test_rnn_encoder_decoder

import argparse
import os
import subprocess
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    '--git_dir', type=str, default='', help='git repo root directory.'
)
parser.add_argument(
    '--build_dir', type=str, default='', help='build directory.'
)
parser.add_argument(
    '--good_commit',
    type=str,
    default='',
    help='The old commit known to be good.',
)
parser.add_argument(
    '--bad_commit', type=str, default='', help='The new commit known to be bad.'
)
parser.add_argument(
    '--test_target', type=str, default='', help='The test target to evaluate.'
)
parser.add_argument(
    '--bisect_branch',
    type=str,
    default='develop',
    help='The mainline branch to bisect (feature branch ignored.',
)
parser.add_argument(
    '--log_file', type=str, default='', help='The file use to log outputs.'
)
parser.add_argument(
    '--test_times',
    type=int,
    default=10,
    help="Number of times to run the test target.",
)
parser.add_argument(
    '--build_parallel', type=int, default=32, help="make parallelism."
)
args = parser.parse_args()

if not args.log_file:
    args.log_file = f'/tmp/{args.good_commit}...{args.bad_commit}.log'


def print_arguments():
    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(vars(args).iteritems()):
        print(f'{arg}: {value}')
    print('------------------------------------------------')


print_arguments()

# List the commits in mainline branch.
os.chdir(args.git_dir)
ret = subprocess.check_output(
    [f'git rev-list --first-parent {args.good_commit}...{args.bad_commit}'],
    shell=True,
)
sys.stdout.write(f'commits found:\n{ret}\n')
commits = ret.strip().split('\n')
os.chdir(args.build_dir)
# Clean up previous logs.
subprocess.check_output([f'echo "" > {args.log_file}'], shell=True)

last_culprit = ''
while True:
    # Get to the mainline branch and clean up
    os.chdir(args.git_dir)
    subprocess.check_output(
        [
            f'git checkout {args.bisect_branch} && git clean -fd && git checkout .'
        ],
        shell=True,
    )

    if not commits:
        sys.stdout.write('no commits to bisect\n')
        sys.exit()
    # checkout the picked branch.
    pick_idx = len(commits) / 2
    pick = commits[pick_idx]
    os.chdir(args.git_dir)
    subprocess.check_output([f'git checkout {pick}'], shell=True)

    # Clean builds and compile.
    # We assume mainline commits should always compile.
    os.chdir(args.build_dir)
    sys.stdout.write(f'eval commit {pick_idx}/{len(commits)}: {pick}\n')
    # Link error can happen without complete clean up.
    cmd = (
        'rm -rf * && '
        f'cmake -DWITH_TESTING=ON {args.git_dir} >> {args.log_file} && make -j{args.build_parallel} >> {args.log_file}'
    )
    sys.stdout.write(f'cmd: {cmd}\n')
    try:
        subprocess.check_output([cmd], shell=True)
    except subprocess.CalledProcessError as e:
        sys.stderr.write(f'failed to build commit: {pick}\n{e}\n')
        sys.exit()
    # test the selected branch.
    passed = True
    try:
        cmd = f'ctest --repeat-until-fail {args.test_times} -R {args.test_target} >> {args.log_file}'
        sys.stdout.write(f'cmd: {cmd}\n')
        subprocess.check_output([cmd], shell=True)
    except subprocess.CalledProcessError as e:
        passed = False
        last_culprit = pick
    sys.stdout.write(f'eval {pick} passed: {passed}\n')
    if passed:
        if pick_idx == 0:
            break
        commits = commits[:pick_idx]
    else:
        if pick_idx + 1 >= len(commits):
            break
        commits = commits[pick_idx + 1 :]

sys.stdout.write(f'Culprit commit: {last_culprit}\n')
