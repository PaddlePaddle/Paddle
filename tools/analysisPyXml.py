# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

<<<<<<< HEAD
import os
import re
import sys
from xml.etree import ElementTree

import commands
=======
import commands
from xml.etree import ElementTree
import re
import time
import queue
import threading
import os
import json
import sys
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81


def analysisPyXml(rootPath, ut):
    xml_path = '%s/build/pytest/%s/python-coverage.xml' % (rootPath, ut)
<<<<<<< HEAD
    related_ut_map_file = '%s/build/ut_map/%s/related_%s.txt' % (
        rootPath,
        ut,
        ut,
    )
    notrelated_ut_map_file = '%s/build/ut_map/%s/notrelated_%s.txt' % (
        rootPath,
        ut,
        ut,
    )
=======
    related_ut_map_file = '%s/build/ut_map/%s/related_%s.txt' % (rootPath, ut,
                                                                 ut)
    notrelated_ut_map_file = '%s/build/ut_map/%s/notrelated_%s.txt' % (rootPath,
                                                                       ut, ut)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    tree = ElementTree.parse(xml_path)
    root = tree.getroot()
    error_files = []
    pyCov_file = []
    for clazz in root.findall('packages/package/classes/class'):
        clazz_filename = clazz.attrib.get('filename')
        if not clazz_filename.startswith('/paddle'):
            clazz_filename = '/paddle/%s' % clazz_filename
        for line in clazz.findall('lines/line'):
            line_hits = int(line.attrib.get('hits'))
            if line_hits != 0:
                line_number = int(line.attrib.get('number'))
                command = 'sed -n %sp %s' % (line_number, clazz_filename)
                _code, output = commands.getstatusoutput(command)
                if _code == 0:
<<<<<<< HEAD
                    if not output.strip().startswith(
                        (
                            'from',
                            'import',
                            '__all__',
                            'def',
                            'class',
                            '"""',
                            '@',
                            '\'\'\'',
                            'logger',
                            '_logger',
                            'logging',
                            'r"""',
                            'pass',
                            'try',
                            'except',
                            'if __name__ == "__main__"',
                        )
                    ):
                        pattern = r"""(.*) = ('*')|(.*) = ("*")|(.*) = (\d)|(.*) = (-\d)|(.*) = (None)|(.*) = (True)|(.*) = (False)|(.*) = (URL_PREFIX*)|(.*) = (\[)|(.*) = (\{)|(.*) = (\()"""  # a='b'/a="b"/a=0
                        if re.match(pattern, output.strip()) is None:
=======
                    if output.strip().startswith(
                        ('from', 'import', '__all__', 'def', 'class', '"""',
                         '@', '\'\'\'', 'logger', '_logger', 'logging', 'r"""',
                         'pass', 'try', 'except',
                         'if __name__ == "__main__"')) == False:
                        pattern = "(.*) = ('*')|(.*) = (\"*\")|(.*) = (\d)|(.*) = (-\d)|(.*) = (None)|(.*) = (True)|(.*) = (False)|(.*) = (URL_PREFIX*)|(.*) = (\[)|(.*) = (\{)|(.*) = (\()"  #a='b'/a="b"/a=0
                        if re.match(pattern, output.strip()) == None:
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                            pyCov_file.append(clazz_filename)
                            coverageMessage = 'RELATED'
                            break
                        else:
<<<<<<< HEAD
                            coverageMessage = 'FILTER'  # hit filter logic
=======
                            coverageMessage = 'FILTER'  #hit filter logic
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                    else:
                        coverageMessage = 'FILTER'
                else:
                    coverageMessage = 'ERROR'
                    error_files.append(clazz_filename)
                    break
            else:
                coverageMessage = 'NOT_RELATED'
        if coverageMessage in ['NOT_RELATED', 'ERROR', 'FILTER']:
<<<<<<< HEAD
            os.system(
                'echo %s >> %s' % (clazz_filename, notrelated_ut_map_file)
            )
=======
            os.system('echo %s >> %s' %
                      (clazz_filename, notrelated_ut_map_file))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        elif coverageMessage == 'RELATED':
            os.system('echo %s >> %s' % (clazz_filename, related_ut_map_file))

    print("============len(pyCov_file)")
    print(len(pyCov_file))
    print("============error")
    print(error_files)


if __name__ == "__main__":
    rootPath = sys.argv[1]
    ut = sys.argv[2]
    analysisPyXml(rootPath, ut)
