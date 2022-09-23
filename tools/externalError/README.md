#### **Introduction for crawling new error message:**



1. add new spider code in spider.py for crawling error message from website.

2. run `bash start.sh` in current  directory to generate new externalErrorMsg_${date}.tar.gz file, for example `externalErrorMsg_20210928.tar.gz`.

3. upload above tar file into bos https://paddlepaddledeps.bj.bcebos.com **paddlepaddledeps** bucket, and copy download link `${download_url}`. ***\*Be careful not to delete original tar file\****.

4. compute md5 value of above tar file `${md5}`, and modify cmake/third_party.cmake file

   ```
   set(URL  "${download_url}" CACHE STRING "" FORCE)
   file_download_and_uncompress(${URL} "externalError" MD5 ${md5})
   ```

   for example:

   ```
   set(URL  "https://paddlepaddledeps.bj.bcebos.com/externalErrorMsg_20210928.tar.gz" CACHE STRING "" FORCE)
   file_download_and_uncompress(${URL} "externalError" MD5 a712a49384e77ca216ad866712f7cafa)
   ```

5. commit your changes, and create pull request.
