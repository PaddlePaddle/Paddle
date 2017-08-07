#!/usr/bin/env python2
import argparse
import select
import subprocess
import sys


def main(runnable, i, n, extra_args):
    if runnable[0] != '/' and runnable[0] != '.':
        runnable = './' + runnable

    process = subprocess.Popen(
        (runnable, '--gtest_list_tests'), stdout=subprocess.PIPE)

    cur_section = ""
    all_tests = []
    for line in process.stdout:
        if line[:2] == '  ':
            all_tests.append(cur_section + line.strip())
        else:
            cur_section = line.strip()

    test_filter = ":".join((all_tests[j] for j in xrange(i, len(all_tests), n)))
    if len(test_filter) == 0:
        sys.exit(0)
    process = subprocess.Popen(
        [runnable, '--gtest_filter=' + test_filter] + extra_args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)
    while True:
        reads = [process.stdout.fileno(), process.stderr.fileno()]
        ret = select.select(reads, [], [])
        for fd in ret[0]:
            if fd == process.stdout.fileno():
                read = process.stdout.readline()
                sys.stdout.write(read)
            if fd == process.stderr.fileno():
                read = process.stderr.readline()
                sys.stderr.write(read)
        if process.poll() is not None:
            break

    sys.exit(process.returncode)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('runnable', help='gtest executor path', type=str)
    parser.add_argument('-i', help='start index', type=int, default=0)
    parser.add_argument('-n', help='step', type=int, default=1)
    result, unparsed_args = parser.parse_known_args()
    main(
        runnable=result.runnable,
        i=result.i,
        n=result.n,
        extra_args=unparsed_args)
