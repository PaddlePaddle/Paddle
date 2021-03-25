#!/usr/bin/env python
from __future__ import print_function
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
    if each_diff[0] in ['-', '?']:  # delete or change API is not allowed
        error = True
    elif each_diff[0] == '+':
        error = True

    if each_diff[0] != ' ':
        diffs.append(each_diff)
'''
If you modify/add/delete the API files, including code and comment, 
please follow these steps in order to pass the CI:

  1. cd ${paddle_path}, compile paddle;
  2. pip install build/python/dist/(build whl package);
  3. run "python tools/print_signatures.py paddle.fluid> paddle/fluid/API.spec"
'''
if error:
    print('API Difference is: ')
    for each_diff in diffs:
        print(each_diff)
