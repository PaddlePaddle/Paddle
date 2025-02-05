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

import getopt
import re
import ssl
import sys
import urllib.request
from html.parser import HTMLParser

import external_error_pb2


def parsing(externalErrorDesc):
    # *********************************************************************************************#
    # *********************************** CUDA Error Message **************************************#
    print("start crawling errorMessage for nvidia CUDA API--->")
    url = 'https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g3f51e3575c2178246db0a94a430e0038'

    allMessageDesc = externalErrorDesc.errors.add()
    allMessageDesc.type = external_error_pb2.CUDA

    ssl._create_default_https_context = ssl._create_unverified_context
    html = urllib.request.urlopen(url).read().decode('utf-8')
    res_div = r'<div class="section">.*?<p>CUDA error types </p>.*?</div>.*?<div class="enum-members">(.*?)</div>'
    m_div = re.findall(res_div, html, re.DOTALL | re.MULTILINE)[0]

    res_dt = r'<dt>(.*?)</dt>.*?<dd>(.*?)</dd>'
    m_dt = re.findall(res_dt, m_div, re.DOTALL | re.MULTILINE)
    for error in m_dt:
        res_type = r'<span class="enum-member-name-def">(.*?) = <span class="ph ph apiData">(.*?)</span></span>'
        m_type = re.findall(res_type, error[0], re.DOTALL | re.MULTILINE)[0]
        m_message = error[1]
        m_message = m_message.replace('\n', '')
        res_a = r'(<a class=.*?</a>)'
        res_shape = r'<a class=.*?>(.*?)</a>'
        list_a = re.findall(res_a, m_message, re.DOTALL | re.MULTILINE)
        list_shape = re.findall(res_shape, m_message, re.DOTALL | re.MULTILINE)
        assert len(list_a) == len(list_shape)
        for idx in range(len(list_a)):
            m_message = m_message.replace(list_a[idx], list_shape[idx])

        m_message = m_message.replace(
            '<h6 class="deprecated_header">Deprecated</h6>', ''
        )

        res_span = r'(<span class=.*?</span>)'
        res_span_detail = r'<span class=.*?>(.*?)</span>'
        list_span = re.findall(res_span, m_message, re.DOTALL | re.MULTILINE)
        list_span_detail = re.findall(
            res_span_detail, m_message, re.DOTALL | re.MULTILINE
        )
        assert len(list_span) == len(list_span_detail)
        for idx in range(len(list_span)):
            m_message = m_message.replace(list_span[idx], list_span_detail[idx])

        res_p = r'(<p>.*?</p>)'
        res_p_detail = r'<p>(.*?)</p>'
        list_p = re.findall(res_p, m_message, re.DOTALL | re.MULTILINE)
        list_p_detail = re.findall(
            res_p_detail, m_message, re.DOTALL | re.MULTILINE
        )
        assert len(list_p) == len(list_p_detail)
        for idx in range(len(list_p)):
            m_message = m_message.replace(list_p[idx], list_p_detail[idx])

        m_message = m_message.replace('  ', '')
        _Messages = allMessageDesc.messages.add()
        try:
            _Messages.code = int(m_type[1])
        except ValueError:
            if re.match('0x', m_type[1]):
                _Messages.code = int(m_type[1], 16)
            else:
                raise ValueError
        _Messages.message = f"'{m_type[0]}'. {m_message}"
    print("End crawling errorMessage for nvidia CUDA API!\n")

    # ***********************************************************************************************#
    # *********************************** CURAND Error Message **************************************#
    print("start crawling errorMessage for nvidia CURAND API--->")
    url = 'https://docs.nvidia.com/cuda/curand/group__HOST.html#group__HOST_1gb94a31d5c165858c96b6c18b70644437'

    allMessageDesc = externalErrorDesc.errors.add()
    allMessageDesc.type = external_error_pb2.CURAND

    html = urllib.request.urlopen(url).read().decode('utf-8')

    res_div = r'<div class="section">.*?<p>CURAND function call status types </p>.*?</div>.*?<div class="enum-members">(.*?)</div>'
    m_div = re.findall(res_div, html, re.DOTALL | re.MULTILINE)[0]

    res_dt = r'<dt>(.*?)</dt>.*?<dd>(.*?)</dd>'
    m_dt = re.findall(res_dt, m_div, re.DOTALL | re.MULTILINE)
    for error in m_dt:
        res_type = r'<span class="enum-member-name-def">(.*?) = <span class="ph ph apiData">(.*?)</span></span>'
        m_type = re.findall(res_type, error[0], re.DOTALL | re.MULTILINE)[0]
        m_message = error[1]

        _Messages = allMessageDesc.messages.add()
        try:
            _Messages.code = int(m_type[1])
        except ValueError:
            if re.match('0x', m_type[1]):
                _Messages.code = int(m_type[1], 16)
            else:
                raise ValueError
        _Messages.message = f"'{m_type[0]}'. {m_message}"
    print("End crawling errorMessage for nvidia CURAND API!\n")

    # **************************************************************************************************#
    # *********************************** CUDNN Error Message ******************************************#
    cudnnStatus_t = {
        "CUDNN_STATUS_SUCCESS": 0,
        "CUDNN_STATUS_NOT_INITIALIZED": 1,
        "CUDNN_STATUS_ALLOC_FAILED": 2,
        "CUDNN_STATUS_BAD_PARAM": 3,
        "CUDNN_STATUS_INTERNAL_ERROR": 4,
        "CUDNN_STATUS_INVALID_VALUE": 5,
        "CUDNN_STATUS_ARCH_MISMATCH": 6,
        "CUDNN_STATUS_MAPPING_ERROR": 7,
        "CUDNN_STATUS_EXECUTION_FAILED": 8,
        "CUDNN_STATUS_NOT_SUPPORTED": 9,
        "CUDNN_STATUS_LICENSE_ERROR": 10,
        "CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING": 11,
        "CUDNN_STATUS_RUNTIME_IN_PROGRESS": 12,
        "CUDNN_STATUS_RUNTIME_FP_OVERFLOW": 13,
    }

    print("start crawling errorMessage for nvidia CUDNN API--->")
    url = 'https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnStatus_t'

    allMessageDesc = externalErrorDesc.errors.add()
    allMessageDesc.type = external_error_pb2.CUDNN

    html = urllib.request.urlopen(url).read().decode('utf-8')
    f = open('1.txt', 'w')
    f.write(html)

    res_div = r'<div class="section" id="cudnnStatus_t__section_lmp_dgr_2jb"><a name="cudnnStatus_t__section_lmp_dgr_2jb" shape="rect">(.*?)</div>'
    m_div = re.findall(res_div, html, re.DOTALL | re.MULTILINE)[0]

    res_dt = r'<dt class="dt dlterm"><samp class="ph codeph">(.*?)</samp></dt>.*?<dd class="dd">(.*?)</dd>'
    m_dt = re.findall(res_dt, m_div, re.DOTALL | re.MULTILINE)
    for error in m_dt:
        m_message = error[1]

        res_class = r'<p class="p">.*?</p>'
        res_class_detail = r'<p class="p">(.*?)</p>'
        list_class = re.findall(res_class, m_message, re.DOTALL | re.MULTILINE)
        list_class_detail = re.findall(
            res_class_detail, m_message, re.DOTALL | re.MULTILINE
        )
        assert len(list_class) == len(list_class_detail)
        for idx in range(len(list_class)):
            m_message = m_message.replace(
                list_class[idx], list_class_detail[idx]
            )

        res_a = r'(<a class="xref".*?</a>)'
        res_shape = r'<a class="xref".*?>(.*?)</a>'
        list_a = re.findall(res_a, m_message, re.DOTALL | re.MULTILINE)
        list_shape = re.findall(res_shape, m_message, re.DOTALL | re.MULTILINE)
        assert len(list_a) == len(list_shape)
        for idx in range(len(list_a)):
            m_message = m_message.replace(list_a[idx], list_shape[idx])

        res_span = r'(<span class="ph">.*?</span>)'
        res_span_detail = r'<span class="ph">(.*?)</span>'
        list_span = re.findall(res_span, m_message, re.DOTALL | re.MULTILINE)
        list_span_detail = re.findall(
            res_span_detail, m_message, re.DOTALL | re.MULTILINE
        )
        assert len(list_span) == len(list_span_detail)
        for idx in range(len(list_span)):
            m_message = m_message.replace(list_span[idx], list_span_detail[idx])

        res_samp = r'(<samp class="ph codeph">.*?</samp>)'
        res_samp_detail = r'<samp class="ph codeph">(.*?)</samp>'
        list_samp = re.findall(res_samp, m_message, re.DOTALL | re.MULTILINE)
        list_samp_detail = re.findall(
            res_samp_detail, m_message, re.DOTALL | re.MULTILINE
        )
        assert len(list_samp) == len(list_samp_detail)
        for idx in range(len(list_samp)):
            m_message = m_message.replace(list_samp[idx], list_samp_detail[idx])

        m_message = re.sub(r'\n +', ' ', m_message)

        _Messages = allMessageDesc.messages.add()
        _Messages.code = int(cudnnStatus_t[error[0]])
        _Messages.message = f"'{error[0]}'. {m_message}"
    print("End crawling errorMessage for nvidia CUDNN API!\n")

    # *************************************************************************************************#
    # *********************************** CUBLAS Error Message ****************************************#
    cublasStatus_t = {
        "CUBLAS_STATUS_SUCCESS": 0,
        "CUBLAS_STATUS_NOT_INITIALIZED": 1,
        "CUBLAS_STATUS_ALLOC_FAILED": 3,
        "CUBLAS_STATUS_INVALID_VALUE": 7,
        "CUBLAS_STATUS_ARCH_MISMATCH": 8,
        "CUBLAS_STATUS_MAPPING_ERROR": 11,
        "CUBLAS_STATUS_EXECUTION_FAILED": 13,
        "CUBLAS_STATUS_INTERNAL_ERROR": 14,
        "CUBLAS_STATUS_NOT_SUPPORTED": 15,
        "CUBLAS_STATUS_LICENSE_ERROR": 16,
    }

    print("start crawling errorMessage for nvidia CUBLAS API--->")
    url = 'https://docs.nvidia.com/cuda/cublas/index.html#cublasstatus_t'

    allMessageDesc = externalErrorDesc.errors.add()
    allMessageDesc.type = external_error_pb2.CUBLAS

    html = urllib.request.urlopen(url).read().decode('utf-8')

    res_div = r'<p class="p">The type is used for function status returns. All cuBLAS library.*?<div class="tablenoborder">(.*?)</div>'
    m_div = re.findall(res_div, html, re.DOTALL | re.MULTILINE)[0]

    res_dt = r'<p class="p"><samp class="ph codeph">(.*?)</samp></p>.*?colspan="1">(.*?)</td>'
    m_dt = re.findall(res_dt, m_div, re.DOTALL | re.MULTILINE)

    for error in m_dt:
        m_message = error[1]
        m_message = re.sub(r'\n +', ' ', m_message)

        res_p = r'<p class="p">.*?</p>'
        res_p_detail = r'<p class="p">(.*?)</p>'
        list_p = re.findall(res_p, m_message, re.DOTALL | re.MULTILINE)
        list_p_detail = re.findall(
            res_p_detail, m_message, re.DOTALL | re.MULTILINE
        )
        assert len(list_p) == len(list_p_detail)
        for idx in range(len(list_p)):
            m_message = m_message.replace(list_p[idx], list_p_detail[idx])

        res_samp = r'<samp class="ph codeph">.*?</samp>'
        res_samp_detail = r'<samp class="ph codeph">(.*?)</samp>'
        list_samp = re.findall(res_samp, m_message, re.DOTALL | re.MULTILINE)
        list_samp_detail = re.findall(
            res_samp_detail, m_message, re.DOTALL | re.MULTILINE
        )
        assert len(list_samp) == len(list_samp_detail)
        for idx in range(len(list_samp)):
            m_message = m_message.replace(list_samp[idx], list_samp_detail[idx])

        _Messages = allMessageDesc.messages.add()
        _Messages.code = int(cublasStatus_t[error[0]])
        _Messages.message = f"'{error[0]}'. {m_message}"
    print("End crawling errorMessage for nvidia CUBLAS API!\n")

    # *************************************************************************************************#
    # *********************************** CUSOLVER Error Message **************************************#
    cusolverStatus_t = {
        "CUSOLVER_STATUS_SUCCESS": 0,
        "CUSOLVER_STATUS_NOT_INITIALIZED": 1,
        "CUSOLVER_STATUS_ALLOC_FAILED": 2,
        "CUSOLVER_STATUS_INVALID_VALUE": 3,
        "CUSOLVER_STATUS_ARCH_MISMATCH": 4,
        "CUSOLVER_STATUS_MAPPING_ERROR": 5,
        "CUSOLVER_STATUS_EXECUTION_FAILED": 6,
        "CUSOLVER_STATUS_INTERNAL_ERROR": 7,
        "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED": 8,
        "CUSOLVER_STATUS_NOT_SUPPORTED": 9,
        "CUSOLVER_STATUS_ZERO_PIVOT": 10,
        "CUSOLVER_STATUS_INVALID_LICENSE": 11,
        "CUSOLVER_STATUS_IRS_PARAMS_NOT_INITIALIZED": 12,
        "CUSOLVER_STATUS_IRS_PARAMS_INVALID": 13,
        "CUSOLVER_STATUS_IRS_INTERNAL_ERROR": 14,
        "CUSOLVER_STATUS_IRS_NOT_SUPPORTED": 15,
        "CUSOLVER_STATUS_IRS_OUT_OF_RANGE": 16,
        "CUSOLVER_STATUS_IRS_NRHS_NOT_SUPPORTED_FOR_REFINE_GMRES": 17,
        "CUSOLVER_STATUS_IRS_INFOS_NOT_INITIALIZED": 18,
    }
    print("start crawling errorMessage for nvidia CUSOLVER API--->")
    url = 'https://docs.nvidia.com/cuda/cusolver/index.html#cuSolverSPstatus'

    allMessageDesc = externalErrorDesc.errors.add()
    allMessageDesc.type = external_error_pb2.CUSOLVER

    html = urllib.request.urlopen(url).read().decode('utf-8')

    res_div = r'This is a status type returned by the library functions and.*?<div class="tablenoborder">(.*?)</div>'
    m_div = re.findall(res_div, html, re.DOTALL | re.MULTILINE)[0]

    res_dt = (
        r'<samp class="ph codeph">(.*?)</samp></td>.*?colspan="1">(.*?)</td>'
    )
    m_dt = re.findall(res_dt, m_div, re.DOTALL | re.MULTILINE)

    for error in m_dt:
        m_message = error[1]
        m_message = re.sub(r'\n +', '', m_message)
        m_message = re.sub(r'<p class="p"></p>', '', m_message)

        res_p = r'<p class="p">.*?</p>'
        res_p_detail = r'<p class="p">(.*?)</p>'
        list_p = re.findall(res_p, m_message, re.DOTALL | re.MULTILINE)
        list_p_detail = re.findall(
            res_p_detail, m_message, re.DOTALL | re.MULTILINE
        )
        assert len(list_p) == len(list_p_detail)
        for idx in range(len(list_p)):
            m_message = m_message.replace(list_p[idx], list_p_detail[idx])

        res_samp = r'<samp class="ph codeph">.*?</samp>'
        res_samp_detail = r'<samp class="ph codeph">(.*?)</samp>'
        list_samp = re.findall(res_samp, m_message, re.DOTALL | re.MULTILINE)
        list_samp_detail = re.findall(
            res_samp_detail, m_message, re.DOTALL | re.MULTILINE
        )
        assert len(list_samp) == len(list_samp_detail)
        for idx in range(len(list_samp)):
            m_message = m_message.replace(list_samp[idx], list_samp_detail[idx])

        res_strong = r'<strong class="ph b">.*?</strong>'
        res_strong_detail = r'<strong class="ph b">(.*?)</strong>'
        list_strong = re.findall(
            res_strong, m_message, re.DOTALL | re.MULTILINE
        )
        list_strong_detail = re.findall(
            res_strong_detail, m_message, re.DOTALL | re.MULTILINE
        )
        assert len(list_strong) == len(list_strong_detail)
        for idx in range(len(list_strong)):
            m_message = m_message.replace(
                list_strong[idx], list_strong_detail[idx]
            )

        _Messages = allMessageDesc.messages.add()
        _Messages.code = int(cusolverStatus_t[error[0]])
        _Messages.message = f"'{error[0]}'. {m_message}"
    print("End crawling errorMessage for nvidia CUSOLVER API!\n")

    # **********************************************************************************************#
    # *************************************** NCCL error *******************************************#
    print("start crawling errorMessage for nvidia NCCL API--->")
    url = 'https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#ncclresult-t'
    allMessageDesc = externalErrorDesc.errors.add()
    allMessageDesc.type = external_error_pb2.NCCL
    html = urllib.request.urlopen(url).read().decode('utf-8')
    res_div = r'<code class="descname">ncclResult_t</code>(.*?)</div>'
    m_div = re.findall(res_div, html, re.DOTALL | re.MULTILINE)[0]

    res_dt = r'<code class="descname">(.*?)</code>.*?<span class="pre">(.*?)</span></code>\)(.*?)</p>\n</dd></dl>'
    m_dt = re.findall(res_dt, m_div, re.DOTALL | re.MULTILINE)
    for error in m_dt:
        m_message = re.sub(r'\n', '', error[2])
        _Messages = allMessageDesc.messages.add()
        _Messages.code = int(error[1])
        _Messages.message = f"'{error[0]}'. {m_message}"
    print("End crawling errorMessage for nvidia NCCL API!\n")

    # *************************************************************************************************#
    # *********************************** CUFFT Error Message **************************************#
    print("start crawling errorMessage for nvidia CUFFT API--->")
    url = 'https://docs.nvidia.com/cuda/cufft/index.html#cufftresult'

    allMessageDesc = externalErrorDesc.errors.add()
    allMessageDesc.type = external_error_pb2.CUFFT

    html = urllib.request.urlopen(url).read().decode('utf-8')

    class CUFFTHTMLParser(HTMLParser):
        '''CUFFTHTML Parser'''

        def handle_data(self, data):
            if 'typedef enum cufftResult_t' in data:
                for line in data.strip().splitlines()[1:-1]:
                    status, code, desc = re.split('=|//', line.strip())
                    _Messages = allMessageDesc.messages.add()
                    _Messages.code = int(code.strip(' ,'))
                    _Messages.message = f"'{status.strip()}'. {desc.strip()}"

    CUFFTHTMLParser().feed(html)


def main(argv):
    try:
        opts, _ = getopt.getopt(argv, "h", ["help"])
    except getopt.GetoptError:
        print('python spider.py')
        sys.exit(2)
    for opt, _ in opts:
        if opt in ("-h", "--help"):
            print('python spider.py')
            sys.exit(2)
    externalErrorDesc = external_error_pb2.ExternalErrorDesc()
    parsing(externalErrorDesc)

    serializedString = externalErrorDesc.SerializeToString()
    with open("externalErrorMsg.pb", "wb") as f:
        # save for externalErrorMsg.pb from Python-protobuf interface
        # load from C++-protobuf interface and get error message
        f.write(serializedString)
    print(
        "Generating data file [externalErrorMsg.pb] for external third_party API error has been done!"
    )


if __name__ == "__main__":
    main(sys.argv[1:])
