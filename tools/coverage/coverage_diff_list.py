#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
usage: coverage_diff_list.py list_file max_rate > coverage-diff-list-90.out
"""

import sys


def filter_by(list_file, max_rate):
    """
    Args:
        list_file (str): File of list.  
        max_rate (float): Max rate.  

    Returns:
        tuple: File and coverage rate.
    """
    with open(list_file) as list_file:
        for line in list_file:
            line = line.strip()

            split = line.split('|')

            # name

            name = split[0].strip()

            if name.startswith('/paddle/'):
                name = name[len('/paddle/'):]

            # rate

            try:
                rate = split[1].split()[0].strip('%')
                rate = float(rate)

                if rate >= max_rate:
                    continue
            except:
                pass

            print name, rate


if __name__ == '__main__':
    if len(sys.argv) < 2:
        exit()

    list_file = sys.argv[1]
    max_rate = float(sys.argv[2])

    filter_by(list_file, max_rate)
