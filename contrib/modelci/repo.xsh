#!/usr/bin/env xonsh
'''
some utils for clone repo, commit.
'''
from utils import log

def clone(url, dst):
    '''
    url: url of a git repo.
    dst: a abstract path in local file system.
    '''
    log.warn('clone from', url, 'to', dst)
    ![git clone @(url) @(dst)]

def pull(dst):
    cd @(dst)
    git pull
    log.warn(dst, 'updated')

def reset_commit(dst, commitid):
    cd dst
    git reset --hard dst commitid
    log.warn(dst, 'reset to commit', commitid)
