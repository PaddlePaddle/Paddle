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

taskQueue = queue.Queue()
lock = threading.RLock()

def worker(fun):
    while True:
        temp = taskQueue.get()
        fun(temp)
        taskQueue.task_done()

def threadPool(threadPoolNum):
    threadPool = []
    for i in range(threadPoolNum):
        thread = threading.Thread(target=worker,args={doFun,})
#         设置为守护线程，当主线程退出后，守护线程也会退出。如果不设置成守护线程，线程将因为执行的是死循环而无法退出
#         主线程将阻塞与子线程
        thread.daemon = True
        threadPool.append(thread)
    return threadPool

def getPyCovResult(params):
    rootPath = params[0]
    ut = params[1]
    print("ut: %s" %ut)
    startTime = int(time.time())
    path = '%s/build/pytest/%s' %(rootPath, ut)
    os.system('cd %s && coverage combine `ls python-coverage.data.*`' %path)
    os.system('cd %s && pwd && coverage xml -i -o python-coverage.xml' %path)
    xml_path = '%s/python-coverage.xml' %path
    os.system("python %s/tools/analysisPyXml.py %s %s" %(rootPath, rootPath, ut))
    endTime = int(time.time())
    print('pyCov Time: %s' %(endTime-startTime))

def doFun(params):
    getPyCovResult(params)
    
def main(rootPath):
    """
    1. 获取每个gcda文件的gcov文件
    2. 收集覆盖率不为0的gcov文件
    """
    path = '%s/build/pytest' %rootPath
    dirs = os.listdir(path)
    pool = threadPool(23)
    for i in range(pool.__len__()):
        pool[i].start()
    for ut in dirs:
        params = [rootPath, ut]
        taskQueue.put(params)
    taskQueue.join()  

if __name__ == "__main__":
    rootPath = sys.argv[1]
    main(rootPath)