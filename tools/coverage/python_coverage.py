#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
usage: python_coverage.py > python-coverage.info
"""

from os import path
from xml.etree import ElementTree

tree = ElementTree.parse('python-coverage.xml')
root = tree.getroot()

sources = root.findall('sources/source')

if len(sources) > 1:
    exit(1)

source = sources[0].text

for clazz in root.findall('packages/package/classes/class'):
    clazz_filename = clazz.attrib.get('filename')
    clazz_filename = path.join(source, clazz_filename)

    if clazz_filename.startswith('/paddle/build/python/'):
        clazz_filename = '/paddle/python/' + clazz_filename[len(
            '/paddle/build/python/'):]

    if not path.exists(clazz_filename):
        continue

    print 'TN:'
    print 'SF:{}'.format(clazz_filename)

    branch_index = 0

    for line in clazz.findall('lines/line'):
        line_hits = line.attrib.get('hits')
        line_number = line.attrib.get('number')

        line_branch = line.attrib.get('branch')
        line_condition_coverage = line.attrib.get('condition-coverage')
        line_missing_branches = line.attrib.get('missing-branches')

        if line_branch == 'true':
            line_condition_coverage = line_condition_coverage.split()
            line_condition_coverage = line_condition_coverage[1].strip('()')
            line_condition_coverage = line_condition_coverage.split('/')

            taken = line_condition_coverage[0]
            taken = int(taken)

            for _ in range(taken):
                print 'BRDA:{},{},{},{}'.format(line_number, 0, branch_index,
                                                line_hits)
                branch_index += 1

            if line_missing_branches:
                for missing_branch in line_missing_branches.split(','):
                    print 'BRDA:{},{},{},{}'.format(line_number, 0,
                                                    branch_index, 0)
                    branch_index += 1

        print 'DA:{},{}'.format(line_number, line_hits)

    print 'end_of_record'
