#!/usr/bin/env python
import difflib
import sys

with open(sys.argv[1], 'r') as f:
    origin = f.read()
    origin = origin.splitlines()

with open(sys.argv[2], 'r') as f:
    new = f.read()
    new = new.splitlines()

differ = difflib.Differ()
result = differ.compare(origin, new)

error = False
diffs = []
for each_diff in result:
    if each_diff[0] == '-':  # delete unit test is not allowed
        error = True
        diffs.append(each_diff)
'''
If you delete the unit test, such as commenting it out, 
please ask for approval of one RD below for passing CI:

    - kolinwei(recommended) or zhouwei25 or luotao1
'''
if error:
    for each_diff in diffs:
        print(each_diff)
