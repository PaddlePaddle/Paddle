#encoding=utf-8
import sys
import urllib2


def download_file():
    """Get disabled unit tests"""
    url="https://sys-p0.bj.bcebos.com/prec/{}".format('disable_ut')
    f = urllib2.urlopen(url)
    data = f.read()
    if len(data.strip()) == 0:
        sys.exit(1)
    else:
        lt = data.strip().split('\n')
        lt = '^' + '$^'.join(lt) + '$'
        print(lt)
        sys.exit(0)



if __name__=='__main__':
    try:
        download_file()
    except Exception as e:
        print(e)
        sys.exit(1)
