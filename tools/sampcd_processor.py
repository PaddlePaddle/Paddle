#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 14:10:36 2019

@author: v_wanghao11
"""

import os
import subprocess


def find_all(srcstr, substr):

    indices = []

    gotone = srcstr.find(substr)

    while (gotone != -1):

        indices.append(gotone)

        gotone = srcstr.find(substr, gotone + 1)

    return indices


def check_indent(cdline):

    indent = 0
    for c in cdline:
        indent += 1
        if c != ' ':
            break
    indent -= 1
    return indent


#srccom: raw comments in the source,including ''' and original indent


def sampcd_extract_and_run(srccom, name, logf):

    sampcd_begins = find_all(srccom, ".. code-block:: python")
    #print str(sampcd_begins)

    if (len(sampcd_begins) == 0):
        print "----example code check--------\n"
        print "No sample code!\n"
        logf.write("----example code check--------\nNo sample code!\n")

    for y in range(1, len(sampcd_begins) + 1):

        sampcd_begin = sampcd_begins[y - 1]
        sampcd = srccom[sampcd_begin + len(".. code-block:: python") + 1:]

        sampcd = sampcd.split("\n")

        #print sampcd

        #remove starting empty lines
        while sampcd[0].replace(' ', '').replace('\t', '') == '':
            sampcd.pop(0)

        min_indent = check_indent(sampcd[0])
        #print min_indent

        #print sampcd
        sampcd_to_write = []
        for i in range(0, len(sampcd)):

            cdline = sampcd[i]

            #handle empty lines or those only with spaces/tabs
            if cdline.strip() == '':
                continue

            this_indent = check_indent(cdline)

            if (this_indent < min_indent):
                break
                #print "removed:   "+str(sampcd)
            else:
                #print ">>>"+cdline
                sampcd_to_write.append(cdline[min_indent:])

        sampcd = '\n'.join(sampcd_to_write)
        sampcd += '\nprint ' + '\"' + name + ' sample code is executed successfully!\"\n'

        print "\n"
        print "Sample code " + str(y) + " extracted for " + name + "   :"
        print "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^"
        print(sampcd)

        logf.write("\nSample code extracted for " + name + "   :\n")
        logf.write("\n" + sampcd + "\n")

        print "----example code check----\n"
        print "executing sample code ....."

        logf.write("\n----example code check----\n")
        logf.write("\nexecuting sample code .....\n")

        if (len(sampcd_begins) > 1):
            tfname = name + "_example_" + str(y) + ".py"
        else:
            tfname = name + "_example" + ".py"
        tempf = open(tfname, 'w')
        tempf.write(sampcd)
        tempf.close()

        cmd = ["python", tfname]

        subprc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output = subprc.communicate()
        print "execution result:"
        logf.write("\nexecution result:\n")
        msg = "\n".join(output)

        if (msg.find("sample code is executed successfully!") == -1):
            print("Error Raised from Sample Code " + name + " :\n")
            logf.write("\nError Raised from Sample Code " + name + " :\n")

        #msg is the returned code execution report
        print msg
        logf.write("\n" + msg + "\n")

        os.remove(tfname)

    #print sampcd


'''
to extract a def function/class comments body
start_from: the line num of "def" header
'''


def single_defcom_extract(start_from, srcls, is_class_begin=False):

    i = start_from

    fcombody = ""  #def comment body

    comstart = -1

    for x in range(i + 1, len(srcls)):

        if is_class_begin:

            if (srcls[x].startswith('    def ')):
                break

        if ((srcls[x].startswith('def ') or srcls[x].startswith('class '))):
            break

        else:

            if (comstart == -1 and srcls[x].replace(" ", '').replace(
                    "\t", '').replace("\n", '').startswith("\"\"\"")):
                comstart = x
                continue
            if (comstart != -1 and srcls[x].replace(" ", '').replace(
                    "\t", '').replace("\n", '').startswith("\"\"\"")):
                break
            if (comstart !=
                    -1):  #when the comments start, begin to add line to fcombody
                fcombody += srcls[x]

    return fcombody


def srccoms_extract(srcfile, logf):

    #srclines=srcfile.readlines()
    print "source file name:" + srcfile.name
    print "---------------------------------------------------"

    logf.write("source file name:" + srcfile.name + "\n")
    logf.write("---------------------------------------------------\n\n")

    srcc = srcfile.read()

    #1. fetch__all__ list
    allidx = srcc.find("__all__")

    if (allidx != -1):
        alllist_b = allidx + len("__all__")

        allstr = srcc[alllist_b + srcc[alllist_b:].find("[") + 1:alllist_b +
                      srcc[alllist_b:].find("]")]
        allstr = allstr.replace("\n", '').replace(" ", '').replace("'", '')
        alllist = allstr.split(',')
        if '' in alllist:
            alllist.remove('')
        print "__all__:" + str(alllist) + "\n"
        logf.write("__all__:" + str(alllist) + "\n\n")

        #2. get defs and classes header line number
        #set file pointer to its beginning
        srcfile.seek(0, 0)
        srcls = srcfile.readlines()  #source lines

        for i in range(0, len(srcls)):

            if srcls[i].startswith('def '):

                #print srcls[i]

                f_header = srcls[i].replace(" ", '')
                fn = f_header[len('def'):f_header.find('(')]  #function name

                #print fn

                if fn in alllist:

                    print "\n"
                    print "def name:" + fn
                    print "-----------------------"

                    logf.write("\ndef name:" + fn + "\n")
                    logf.write("-----------------------\n")

                    fcombody = single_defcom_extract(i, srcls)
                    if (fcombody == ""):
                        print "no comments in function " + fn
                        logf.write("no comments in function " + fn + "\n\n")
                    else:
                        sampcd_extract_and_run(fcombody, fn, logf)

                else:
                    print fn + " not in __all__ list"
                    logf.write(fn + " not in __all__ list\n\n")

            if srcls[i].startswith('class '):

                print srcls[i]

                c_header = srcls[i].replace(" ", '')
                cn = c_header[len('class'):c_header.find('(')]  #function name

                print "\n"
                print "class name:" + cn
                print "-----------------------"

                logf.write("\nclass name:" + cn + "\n")
                logf.write("-----------------------\n")

                if cn in alllist:

                    allcoms = []

                    #class comment
                    classcom = single_defcom_extract(i, srcls, True)
                    allcoms.append(classcom)
                    if (classcom != ""):
                        sampcd_extract_and_run(classcom, cn, logf)
                    else:
                        print "no comments in class itself " + cn + "\n"
                        logf.write("no comments in class itself " + cn +
                                   "\n\n\n")

                    #raw_input("1press any key to continue...")

                    for x in range(
                            i + 1,
                            len(srcls)):  #from the next line of class header 

                        if (srcls[x].startswith('def ') or
                                srcls[x].startswith('class ')):
                            break
                        else:
                            #property def header

                            if (srcls[x].startswith(
                                    '    def ')):  #detect a mehtod header..

                                thisl = srcls[x]
                                indent = len(thisl) - len(thisl.lstrip())
                                mn = thisl[indent + len('def '):thisl.find(
                                    '(')]  #method name
                                print "method name:" + cn + "." + mn + ":"
                                print "- - - - - - - - - - - - - - - - - - - - - -"

                                logf.write("method name:" + cn + "." + mn +
                                           ":\n")
                                logf.write(
                                    "- - - - - - - - - - - - - - - - - - - - - -\n"
                                )

                                thismethod = []
                                thismtdstr = ""
                                thismethod.append(thisl[indent:])
                                thismtdstr += thisl[indent:]

                                for y in range(x + 1, len(srcls)):

                                    if (srcls[y].startswith('def ') or
                                            srcls[y].startswith('class ')):
                                        break
                                    elif (srcls[y].lstrip().startswith('def ')):
                                        break
                                    else:
                                        thismethod.append(srcls[y][indent:])
                                        thismtdstr += srcls[y][indent:]
                                '''
                                print "\n"
                                print thismethod
                                print "\n"
                                print thismtdstr
                                '''

                                thismtdcom = single_defcom_extract(0,
                                                                   thismethod)
                                allcoms.append(thismtdcom)
                                '''
                                print "\nextracted comments:::\n"
                                print thismtdcom
                                '''

                                name = cn + "." + mn
                                if (thismtdcom != ""):
                                    sampcd_extract_and_run(thismtdcom, name,
                                                           logf)
                                else:
                                    print "no comments in method " + name + "\n"
                                    logf.write("no comments in method " + name +
                                               "\n\n\n")

                                #raw_input("1press any key to continue...")
                else:

                    print cn + " is not in __all__ list"
                    logf.write(cn + " is not in __all__ list\n\n")

                #raw_input("press any key to continue...")


filenames = [
    "layers/control_flow.py",
    "layers/io.py",
    #"layers/ops.py",
    "layers/tensor.py",
    "layers/learning_rate_scheduler.py",
    "layers/detection.py",
    "layers/metric_op.py"
]

filenames += [
    "data_feeder.py",
    "dataset.py",
    "clip.py",
    "metrics.py",
    "executor.py",
    "initializer.py",
    "io.py",
    "nets.py",
    "optimizer.py",
    "profiler.py",
    "regularizer.py",
    #"transpiler.py", 
    "recordio_writer.py",
    "backward.py",
    "average.py",
    "profiler.py",
    "unique_name.py"
]

logf = open("log.txt", 'w')
for filename in filenames:
    srcfile = open("python/paddle/fluid/%s" % (filename), 'r')
    srccoms = srccoms_extract(srcfile, logf)
    srcfile.close()
logf.close()
