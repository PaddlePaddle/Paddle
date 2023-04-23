#### **爬取新错误信息介绍：**



1. 在 spider.py 中添加新的爬虫代码，用于从网站抓取错误消息。

2. 在当前目录下运行 `bash start.sh` 生成新的 externalErrorMsg_${date}.tar.gz 文件，例如 `externalErrorMsg_20210928.tar.gz`。

3. 将上面的 tar 文件上传到 bos https://paddlepaddledeps.bj.bcebos.com **paddlepaddledeps** 中，并复制下载链接 `${download_url}`。 ***\*注意不要删除原始 tar 文件\****。

4. 计算上述tar文件`${md5}`的md5值，并修改cmake/third_party.cmake文件

   ```
   set(URL  "${download_url}" CACHE STRING "" FORCE)
   file_download_and_uncompress(${URL} "externalError" MD5 ${md5})
   ```

   例如：

   ```
   set(URL  "https://paddlepaddledeps.bj.bcebos.com/externalErrorMsg_20210928.tar.gz" CACHE STRING "" FORCE)
   file_download_and_uncompress(${URL} "externalError" MD5 a712a49384e77ca216ad866712f7cafa)
   ```

5. 提交你的更改，并创建 pull request。
