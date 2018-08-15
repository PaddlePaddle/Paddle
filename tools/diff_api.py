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
print('API Difference is: ')
for each_diff in result:
    if each_diff[0] in ['-', '?']:  # delete or change API is not allowed
        error = True
    elif each_diff[0] == '+':
        # only new layers is allowed.
        if not each_diff.startswith('+ paddle.fluid.layers.'):
            error = True

    if each_diff[0] != ' ':
        print(each_diff)

if error:
    sys.exit(1)
