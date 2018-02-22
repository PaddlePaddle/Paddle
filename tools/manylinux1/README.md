# buildtools

We release PaddlePaddle and PaddlePaddle Fluid as shared libraries,
which, we hope could be released as wheel packages on PyPI, so we need
to make sure that the build follows the
[manulinux1](https://www.python.org/dev/peps/pep-0513/) standard.

The manylinux standard suggests building Python modules on an old
system, because that a module would anyway depend on some shared
libraries, and Linux's shared library standard states that those built
with newer version compilers cannot work with those with older
versions.  The suggested building environment is as old as CentOS 5.
However, PaddlePaddle relies on CUDA, and the earlies version of
[CentOS works with CUDA is 6](https://hub.docker.com/r/nvidia/cuda/).
So, here we provide a Docker image based on CentOS 6 and CUDA for
building PaddlePaddle and making the release supports "as-manylinux as
possible."  or "sufficiently many Linux" according to [this
discussion](https://mail.python.org/pipermail/wheel-builders/2016-July/000175.html).

The build output of our Docker image includes multiple wheel files --
some contain the CPU-only binary, some others support CUDA; some are
compatible with the cp27m Python ABI, some others with cp27.

To build these wheels, please run the following commands:

```bash
git clone https://github.com/paddlepaddle/paddle
cd paddle/tools/manylinux1
REPO=[yourrepo] ./build_all.sh
```
