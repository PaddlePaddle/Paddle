#!/usr/bin/env python
import difflib
import sys

try:
    f1 = open(sys.argv[1], 'r')
    origin = f1.read()
    origin = origin.splitlines()
except:
    sys.exit(0)
else:
    f1.close()

try:
    f2 = open(sys.argv[2], 'r')
    new = f2.read()
    new = new.splitlines()
except:
    sys.exit(0)
else:
    f2.close()

error = False
diffs = []
for i in origin:
    if i not in new:
        error = True
        diffs.append(i)
'''
If you delete the unit test, such as commenting it out, 
please ask for approval of one RD below for passing CI:

    - kolinwei(recommended) or zhouwei25 or luotao1
'''
if error:
    for each_diff in diffs:
        print("- %s" % each_diff)
