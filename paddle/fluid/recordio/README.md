## Background

The RecordIO file format is a container for records.  This package is a C++ implementation of https://github.com/paddlepaddle/recordio, which originates from https://github.com/wangkuiyi/recordio.

## Fault-tolerant Writing

For the initial design purpose of RecordIO within Google, which was logging, RecordIO groups record into *chunks*, whose header contains an MD5 hash of the chunk.  A process that writes logs is supposed to call the Writer interface to add records.  Once the writer accumulates a handful of them, it groups a chunk, put the MD5 into the chunk header, and appends the chunk to the file.  In the event the process crashes unexpected, the last chunk in the RecordIO file could be incomplete/corrupt. The RecordIO reader is able to recover from these errors when the process restarts by identifying incomplete chucks and skipping over them.

## Reading Ranges

A side-effect of chunks is to make it easy to indexing records while reading, thus allows us to read a range of successive records.  This is good for distributed log process, where each MapReduce task handles only part of records in a big RecordIO file.

The procedure that creates the index starts from reading the header of the first chunk. It indexes the offset (0) and the size of the chunk, and skips to the header of the next chunk by calling the `fseek` API. Please be aware that most distributed filesystems and all POSIX-compatible local filesystem provides `fseek`, and makes sure that `fseek` runs much faster than `fread`.  This procedure generates a map from chunks to their offsets, which allows the readers is to locate and read a range of records.
