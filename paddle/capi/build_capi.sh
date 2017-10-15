#/bin/bash

set -xe

mkdir -p /paddle/build
cd /paddle/build

cmake .. \
      -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE:-Release}  \
      -DWITH_TESTING=${WITH_TESTING:-OFF} \
      -DWITH_TEST=${WITH_TEST:-OFF} \
      -DRUN_TEST=${RUN_TEST:-OFF} \
      -DWITH_GOLANG=${WITH_GOLANG:-OFF} \
      -DWITH_PYTHON=${WITH_PYTHON:-OFF} \
      -DWITH_SWIG_PY=${WITH_SWIG_PY:-OFF} \
      -DWITH_DOC=${WITH_DOC:-OFF} \
      -DWITH_STYLE_CHECK=${WITH_STYLE_CHECK:-OFF} \
      -DWITH_GPU=${WITH_GPU:-OFF} \
      -DWITH_AVX=${WITH_AVX:-ON} \
      -DWITH_C_API=${WITH_C_API:-ON} \
      -DWITH_MKLDNN=${WITH_MKLDNN:-ON} \
      -DWITH_MKLML=${WITH_MKLML:-OFF}

make -j `nproc`
make DESTDIR="./output" install

if [[ ! -z "${WITH_MKLML}" ]]; then
   find ./third_party/install -name 'libiomp5.so' -exec cp {} output/usr/local/lib \;
   find ./third_party/install -name 'libmklml_gnu.so' -exec cp {} output/usr/local/lib \;
   find ./third_party/install -name 'libmklml_intel.so' -exec cp {} output/usr/local/lib \;
fi

if [[ ! -z "${WITH_MKLDNN}" ]]; then
   cp -P ./third_party/install/mkldnn/lib/* output/usr/local/lib/
fi

cd output/usr/local
tar -czvf /paddle/build/paddle.tgz *
