## CTR READER

An multi-thread cpp reader that has the same interface with py_reader. It
uses cpp multi-thread to read file and is much more faster then the Python read
thread in py_reader.

Currently, it support two types of file:
 - gzip
 - plain text file

and two types of data format:
 - cvs data format is :
   * label dense_fea,dense_fea sparse_fea,sparse_fea
 - the svm data format is :
   * label slot1:fea_sign slot2:fea_sign slot1:fea_sign

## Distributed reader

The distributed reader is mainly used by multi-process tasks, it splits the origin batch samples to N sub-batch samples, and the N is equal to the number of processes. The usage is similar to `paddle.batch`.

Cons:
  - It can be operated conveniently so that different processes can read different data.

Pros:
  - Because each process reads the original batch data and then divides the data, the performance may be poor.
