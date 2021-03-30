# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import ssl
import re
import urllib2
import json
import collections
import sys, getopt
import cuda_error_pb2


def parsing(cuda_errorDesc, version, url):
    All_Messages = cuda_errorDesc.AllMessages.add()
    All_Messages.version = int(version)

    ssl._create_default_https_context = ssl._create_unverified_context
    html = urllib2.urlopen(url).read()
    res_div = r'<div class="section">.*?<p>CUDA error types </p>.*?</div>.*?<div class="enum-members">(.*?)</div>'
    m_div = re.findall(res_div, html, re.S | re.M)

    url_list = url.split('/')
    url_prefix = '/'.join(url_list[0:url_list.index('cuda-runtime-api') + 1])

    dic = collections.OrderedDict()
    dic_message = collections.OrderedDict()
    for line in m_div:
        res_dt = r'<dt>(.*?)</dt>.*?<dd>(.*?)</dd>'
        m_dt = re.findall(res_dt, line, re.S | re.M)
        for error in m_dt:
            res_type = r'<span class="ph ph apiData">(.*?)</span>'
            m_type = re.findall(res_type, error[0], re.S | re.M)[0]
            m_message = error[1]
            m_message = m_message.replace('\n', '')
            res_a = r'(<a class=.*?</a>)'
            res_shape = r'<a class=.*?>(.*?)</a>'
            list_a = re.findall(res_a, m_message, re.S | re.M)
            list_shape = re.findall(res_shape, m_message, re.S | re.M)
            assert len(list_a) == len(list_shape)
            for idx in range(len(list_a)):
                m_message = m_message.replace(list_a[idx], list_shape[idx])

            m_message = m_message.replace(
                '<h6 class=\"deprecated_header\">Deprecated</h6>', '')

            res_span = r'(<span class=.*?</span>)'
            res_span_detail = r'<span class=.*?>(.*?)</span>'
            list_span = re.findall(res_span, m_message, re.S | re.M)
            list_span_detail = re.findall(res_span_detail, m_message, re.S |
                                          re.M)
            assert len(list_span) == len(list_span_detail)
            for idx in range(len(list_span)):
                m_message = m_message.replace(list_span[idx],
                                              list_span_detail[idx])

            res_p = r'(<p>.*?</p>)'
            res_p_detail = r'<p>(.*?)</p>'
            list_p = re.findall(res_p, m_message, re.S | re.M)
            list_p_detail = re.findall(res_p_detail, m_message, re.S | re.M)
            assert len(list_p) == len(list_p_detail)
            for idx in range(len(list_p)):
                m_message = m_message.replace(list_p[idx], list_p_detail[idx])

            m_message = m_message.replace('  ', '')
            _Messages = All_Messages.Messages.add()
            try:
                _Messages.errorCode = int(m_type)
            except ValueError:
                if re.match('0x', m_type):
                    _Messages.errorCode = int(m_type, 16)
                else:
                    raise ValueError
            _Messages.errorMessage = m_message  # save for cudaErrorMessage.pb from python-protobuf interface


def main(argv):
    version = []
    url = []
    try:
        opts, args = getopt.getopt(argv, "hv:u:", ["help", "version=", "url="])
    except getopt.GetoptError:
        print 'python spider.py -v <version1,version2,...,> -u <url1,url2,...,>'
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print 'python spider.py -v <version1,version2,...,> -u <url1,url2,...,>'
            sys.exit()
        elif opt in ("-v", "--version"):
            version = arg
        elif opt in ("-u", "--url"):
            url = arg
    version = version.split(',')
    url = url.split(',')
    assert len(version) == len(url)
    cuda_errorDesc = cuda_error_pb2.cudaerrorDesc()
    for idx in range(len(version)):
        if version[idx] == "-1":
            print("crawling errorMessage for CUDA%s from %s" %
                  ("-latest-version", url[idx]))
        else:
            print("crawling errorMessage for CUDA%s from %s" %
                  (version[idx], url[idx]))
        parsing(cuda_errorDesc, version[idx], url[idx])

    serializeToString = cuda_errorDesc.SerializeToString()
    with open("cudaErrorMessage.pb", "wb") as f:
        f.write(serializeToString
                )  # save for cudaErrorMessage.pb from python-protobuf interface
    print("crawling errorMessage for CUDA has been done!!!")


if __name__ == "__main__":
    main(sys.argv[1:])
