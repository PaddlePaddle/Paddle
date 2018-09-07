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
    '--git_dir', type=str, default='', help='git repo root directory.')
parser.add_argument(
    '--build_dir', type=str, default='', help='build directory.')
parser.add_argument(
    '--good_commit',
    type=str,
    default='',
    help='The old commit known to be good.')
parser.add_argument(
    '--bad_commit',
    type=str,
    default='',
    help='The new commit known to be bad.')
parser.add_argument(
    '--test_target', type=str, default='', help='The test target to evaluate.')
parser.add_argument(
    '--bisect_branch',
    type=str,
    default='develop',
    help='The mainline branch to bisect (feature branch ignored.')
parser.add_argument(
    '--log_file', type=str, default='', help='The file use to log outputs.')
parser.add_argument(
    '--test_times',
    type=int,
    default=10,
    help="Number of times to run the test target.")
parser.add_argument(
    '--build_parallel', type=int, default=32, help="make parallelism.")
args = parser.parse_args()

if not args.log_file:
    args.log_file = '/tmp/%s...%s.log' % (args.good_commit, args.bad_commit)


def print_arguments():
    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(vars(args).iteritems()):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')


print_arguments()

# List the commits in mainline branch.
os.chdir(args.git_dir)
ret = subprocess.check_output(
    [
        'git rev-list --first-parent %s...%s' % (args.good_commit,
                                                 args.bad_commit)
    ],
    shell=True)
sys.stdout.write('commits found:\n%s\n' % ret)
commits = ret.strip().split('\n')
os.chdir(args.build_dir)
# Clean up previous logs.
subprocess.check_output(['echo "" > %s' % args.log_file], shell=True)

last_culprit = ''
while True:
    # Get to the mainline branch and clean up
    os.chdir(args.git_dir)
    subprocess.check_output(
        [
            'git checkout %s && git clean -fd && git checkout .' %
            args.bisect_branch
        ],
        shell=True)

    if not commits:
        sys.stdout.write('no commits to bisect\n')
        exit()
    # checkout the picked branch.
    pick_idx = len(commits) / 2
    pick = commits[pick_idx]
    os.chdir(args.git_dir)
    subprocess.check_output(['git checkout %s' % pick], shell=True)

    # Clean builds and compile.
    # We assume mainline commits should always compile.
    os.chdir(args.build_dir)
    sys.stdout.write('eval commit %d/%d: %s\n' % (pick_idx, len(commits), pick))
    # Link error can happen without complete clean up.
    cmd = ('rm -rf * && '
           'cmake -DWITH_TESTING=ON %s >> %s && make -j%s >> %s' %
           (args.git_dir, args.log_file, args.build_parallel, args.log_file))
    sys.stdout.write('cmd: %s\n' % cmd)
    try:
        subprocess.check_output([cmd], shell=True)
    except subprocess.CalledProcessError as e:
        sys.stderr.write('failed to build commit: %s\n%s\n' % (pick, e))
        exit()
    # test the selected branch.
    passed = True
    try:
        cmd = ('ctest --repeat-until-fail %s -R %s >> %s' %
               (args.test_times, args.test_target, args.log_file))
        sys.stdout.write('cmd: %s\n' % cmd)
        subprocess.check_output([cmd], shell=True)
    except subprocess.CalledProcessError as e:
        passed = False
        last_culprit = pick
    sys.stdout.write('eval %s passed: %s\n' % (pick, passed))
    if passed:
        if pick_idx == 0: break
        commits = commits[:pick_idx]
    else:
        if pick_idx + 1 >= len(commits): break
        commits = commits[pick_idx + 1:]

sys.stdout.write('Culprit commit: %s\n' % last_culprit)
