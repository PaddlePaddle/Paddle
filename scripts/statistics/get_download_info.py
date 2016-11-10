# -*- coding: UTF-8 -*-
#!/usr/bin/env python
import time
import os
import stat
import subprocess
import sys
import pycurl
import StringIO
import json
import psycopg2

dbname = "paddle"
host = "127.0.0.1"
user = "user"
password = "password"
port = 5432

github_user = 'user:password'
github_user = 'user:password'

def get_github_clone_info():
    """
    get clone info from github
    :return:
    """
    ret = ''

    b = StringIO.StringIO()
    
    c = pycurl.Curl()
    c.setopt(pycurl.URL, 'https://api.github.com/repos/baidu/Paddle/traffic/clones')
    c.setopt(pycurl.HTTPHEADER, ['Accept: application/vnd.github.spiderman-preview'])
    c.setopt(pycurl.WRITEFUNCTION, b.write) 
    c.setopt(pycurl.USERPWD, github_user)
    c.perform()

    # print b.getvalue()
    ret = json.loads(''.join(b.getvalue()))
    #print c.getinfo(c.HTTP_CODE)
    b.close() 
    c.close() 

    return ret


def get_github_download_info():
    """
    get download info from github
    :return:
    """
    ret = ''

    b = StringIO.StringIO()
    
    c = pycurl.Curl()
    c.setopt(pycurl.URL, 'https://api.github.com/repos/baidu/Paddle/releases')
    c.setopt(pycurl.WRITEFUNCTION, b.write) 
    c.perform()

    ret = json.loads(''.join(b.getvalue()))
    b.close() 
    c.close() 

    return ret


def get_docker_pull_info():
    """
    get pull info from docker
    :return:
    """
    ret = ''

    b = StringIO.StringIO()
    
    c = pycurl.Curl()
    c.setopt(pycurl.URL, 'https://hub.docker.com/v2/repositories/paddledev/paddle/')
    c.setopt(pycurl.WRITEFUNCTION, b.write) 
    c.perform()

    ret = json.loads(''.join(b.getvalue()))
    b.close() 
    c.close() 

    return ret


def exec_sql(sql):
    """
    exec sql to postgressql
    :param sql:
    :return:
    """
    results = ''

    db = None
    cursor = None

    try:
        db = psycopg2.connect(
            host=host,
            user=user,
            password=password,
            database=dbname,
            port=port
        )
        cursor = db.cursor()

        results = cursor.execute(sql)
        if sql.startswith('select'):
            results = cursor.fetchall()
        elif sql.startswith('insert'):
            results = cursor.lastrowid

        db.commit()
    except:
        print sql,sys.exc_info()[0], sys.exc_info()[1]
        if db is not None:
            db.rollback()
        return
    finally:
        if cursor is not None:
            cursor.close()
        if db is not None:
            db.close

    return results


def update_download_info(day, key, value):
    """
    update download info to db
    :param date:
    :param key:
    :param value:
    :return:
    """
    condition = "where date = '%s' and key = '%s'" % (
        day, key)
    sql = "select * from download_info %s" % condition
    if exec_sql(sql):
        sql = "update download_info set value = %s %s" % (value, condition)
    else:
        sql = "insert into download_info(date, key, value) values('%s', '%s', '%s')" % (
            day, key, value)

    exec_sql(sql)

    return True


if __name__ == "__main__":

    clone_info = get_github_clone_info()
    for r in clone_info['clones']:
        count = r['count']
        uniques = r['uniques']
        date = r['timestamp'][:10]

        update_download_info(date, 'github_clone_count', count)
        update_download_info(date, 'github_clone_uniques', uniques)
    print 'get github clone info success!'

    download_info = get_github_download_info()
    download_num = 0
    date = time.strftime("%Y-%m-%d")
    for version in download_info:
        for r in version['assets']:
            download_num += r['download_count']
            update_download_info(date, 'github_download_count_%s' % r['name'], r['download_count'])
    print 'get github download info success!'

    date = time.strftime("%Y-%m-%d")
    pull_info = get_docker_pull_info()
    update_download_info(date, 'docker_download_count', pull_info['pull_count'])
    print 'get docker download info success!'
