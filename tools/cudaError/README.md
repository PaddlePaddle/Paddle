Usage:

Please run:
```
bash start.sh
```

The error message of CUDA9.0 / CUDA10.0 / CUDA-latest-version will be crawled by default.

If you want to crawl a specified version of CUDA, Please run:
```
bash start.sh <version> <URL(optional)>
```
URL can be derived by default, so you don't have to enter a URL.

for example:
```
bash start.sh 11.0
```
will capture error message of CUDA11.0(in future).

Every time when Nvidia upgrade the CUDA major version, you need to run `bash start.sh` in current directory, and upload cudaErrorMessage.tar.gz to https://paddlepaddledeps.bj.bcebos.com/cudaErrorMessage.tar.gz
