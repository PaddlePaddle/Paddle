# -*- coding:utf-8 -*-
import commands
from xml.etree import ElementTree
import re
import time
import queue
import threading
import os
import json
import sys

def analysisPyXml(rootPath, ut):
    xml_path = '%s/build/pytest/%s/python-coverage.xml' %(rootPath, ut)
    ut_map_file = '%s/build/ut_map/%s/%s.txt' %(rootPath, ut, ut)
    tree = ElementTree.parse(xml_path)
    root = tree.getroot()
    error_files = []
    pyCov_file = []
    for clazz in root.findall('packages/package/classes/class'):
        clazz_filename = clazz.attrib.get('filename')
        if not clazz_filename.startswith('/paddle'):
            clazz_filename = '/paddle/%s' %clazz_filename
        for line in clazz.findall('lines/line'):
            line_hits = int(line.attrib.get('hits'))
            if line_hits != 0:
                line_number = int(line.attrib.get('number'))
                command = 'sed -n %sp %s' %(line_number, clazz_filename)
                _code, output = commands.getstatusoutput(command)
                if _code == 0:
                    if output.strip().startswith(('from', 'import', '__all__', 'def', 'class', '"""', '@', '\'\'\'', 'logger', '_logger', 'logging', 'r"""', 'pass', 'try', 'except', 'if __name__ == "__main__"')) == False:
                        #print(line_hits, line_number)
                        pattern = "(.*) = ('*')|(.*) = (\"*\")|(.*) = (\d)|(.*) = (-\d)|(.*) = (None)|(.*) = (True)|(.*) = (False)|(.*) = (URL_PREFIX*)|(.*) = (\[)|(.*) = (\{)|(.*) = (\()"  #a='b'/a="b"/a=0
                        if re.match(pattern, output.strip()) == None:
                            pyCov_file.append(clazz_filename)
                            os.system('echo %s >> %s' %(clazz_filename, ut_map_file))
                            break   
                else:
                    error_files.append(clazz_filename)
                    break
    print("============len(pyCov_file)")
    print(len(pyCov_file))
    print("============error")
    print(error_files)

if __name__ == "__main__":
    rootPath = sys.argv[1]
    ut = sys.argv[2]
    analysisPyXml(rootPath, ut)
