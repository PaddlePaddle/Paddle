#!/usr/bin/env bash

# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

echo "#!/usr/bin/env bash" >> $1
echo "unset GREP_OPTIONS" >> $1
echo "set -e" >> $1
echo -e >> $1 
echo "# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved." >> $1
echo "#" >> $1
echo "# Licensed under the Apache License, Version 2.0 (the \"License\");" >> $1
echo "# you may not use this file except in compliance with the License." >> $1
echo "# You may obtain a copy of the License at" >> $1
echo "#" >> $1
echo "#     http://www.apache.org/licenses/LICENSE-2.0" >> $1
echo "#" >> $1 
echo "# Unless required by applicable law or agreed to in writing, software" >> $1
echo "# distributed under the License is distributed on an \"AS IS\" BASIS," >> $1
echo "# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied." >> $1
echo "# See the License for the specific language governing permissions and" >> $1
echo "# limitations under the License." >> $1
echo -e >> $1
echo -e >> $1
echo "## CUDA_MODULE_LOADING=EAGER,DEFAULT,LAZY" >> $1
echo -e >> $1
echo "# set cicc PATH for Centos" >> $1
echo "export PATH=\$PATH:$2/bin" >> $1
echo "export PATH=\$PATH:$2/nvvm/bin" >> $1
echo -e >> $1
echo "# check nvcc version, if nvcc >= 11.7, just run nvcc itself" >> $1
echo "CUDA_VERSION=\$(nvcc --version | grep -oP '(?<=V)\d*\.\d*')" >> $1
echo "CUDA_VERSION_MAJOR=\${CUDA_VERSION%.*}" >> $1
echo "CUDA_VERSION_MINOR=\${CUDA_VERSION#*.}" >> $1
echo "if (( CUDA_VERSION_MAJOR > 11 || (CUDA_VERSION_MAJOR == 11 && CUDA_VERSION_MINOR >= 7) )); then" >> $1
echo "  nvcc \"\$@\"" >> $1
echo "  exit" >> $1
echo "fi" >> $1
echo -e >> $1
echo "BUILDDIR=\$(mktemp -d  /tmp/nvcc-lazy-build.XXXXXXXX)" >> $1
echo "echo \"\$@\" > \${BUILDDIR}/args" >> $1
echo "BUILDSH=\${BUILDDIR}/build.sh" >> $1
echo "$2/bin/nvcc --dryrun --keep --keep-dir=\${BUILDDIR} \"\$@\" 2>&1 | sed -e 's/#\\$ //;/^rm/d' > \$BUILDSH" >> $1
echo "sed -i -e '/^\s*--/d' \$BUILDSH" >> $1
echo "sed -ne '1,/^cicc.*cudafe1.stub.c/p' \${BUILDSH} > \${BUILDSH}.pre" >> $1
echo "sed -e '1,/^cicc.*cudafe1.stub.c/d' \${BUILDSH} > \${BUILDSH}.post" >> $1
echo -e >> $1
echo "sed -i -e '/LIBRARIES=/{s/\s//g;s/\"\"/ /g}' \${BUILDSH}.pre" >> $1
echo -e >> $1
echo "/usr/bin/env bash \${BUILDSH}.pre" >> $1
echo "STUBF=\$(find \$BUILDDIR -name *.cudafe1.stub.c)" >> $1
echo "CUFILE=\$(basename -s '.cudafe1.stub.c' \$STUBF)" >> $1
echo "sed -i -e '/__sti____cudaRegisterAll.*__attribute__/a static void __try____cudaRegisterAll(int);' \$STUBF" >> $1
echo "sed -i -e 's/__sti____cudaRegisterAll\(.*{\)/__do____cudaRegisterAll\1/' \$STUBF" >> $1
echo "# sed -i -e \"/__do____cudaRegisterAll\(.*{\)/a static void __try____cudaRegisterAll(int l){static int _ls = 0; if (_ls) return; const char* lm = getenv(\\\"CUDA_MODULE_LOADING\\\"); if (lm&&(lm[0]=='L')&&(lm[1]=='A')&&(lm[2]=='Z')&&(lm[3]=='Y')&&(l!=1)) return; _ls = 1; fprintf(stderr,\\\"===> \${CUFILE} lazy-load? %d\\\\\\\\n\\\", l); __do____cudaRegisterAll();}\" \$STUBF" >> $1
echo "sed -i -e \"/__do____cudaRegisterAll\(.*{\)/a static void __try____cudaRegisterAll(int l){static int _ls = 0; if (_ls) return; const char* lm = getenv(\\\"CUDA_MODULE_LOADING\\\"); if (lm&&(lm[0]=='L')&&(lm[1]=='A')&&(lm[2]=='Z')&&(lm[3]=='Y')&&(l!=1)) return; _ls = 1; __do____cudaRegisterAll();}\" \$STUBF" >> $1
echo "sed -i -e '/__try____cudaRegisterAll\(.*{\)/a static void __sti____cudaRegisterAll(void){__try____cudaRegisterAll(0);}' \$STUBF" >> $1
echo "sed -i -e 's/{\(__device_stub__\)/{__try____cudaRegisterAll(1);\1/' \$STUBF" >> $1
echo "/usr/bin/env bash \${BUILDSH}.post" >> $1
echo "rm -rf \$BUILDDIR" >> $1
