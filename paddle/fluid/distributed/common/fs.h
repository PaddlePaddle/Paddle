#pragma once

#include <string>
#include <stdio.h>
#include "glog/logging.h"
#include "ps_string.h"
#include "shell.h"

namespace paddle {
namespace ps {

int fs_select_internal(const std::string& path);

// localfs
extern size_t localfs_buffer_size();

extern void localfs_set_buffer_size(size_t x);

extern std::shared_ptr<FILE> localfs_open_read(std::string path, const std::string& converter);

extern std::shared_ptr<FILE> localfs_open_write(std::string path, const std::string& converter);

extern int64_t localfs_file_size(const std::string& path);

extern void localfs_remove(const std::string& path);

extern std::vector<std::string> localfs_list(const std::string& path);

extern std::string localfs_tail(const std::string& path);

extern bool localfs_exists(const std::string& path);

extern void localfs_mkdir(const std::string& path);

// hdfs
extern size_t hdfs_buffer_size();

extern void hdfs_set_buffer_size(size_t x);

extern const std::string& hdfs_command();

extern void hdfs_set_command(const std::string& x);

extern std::shared_ptr<FILE> hdfs_open_read(std::string path, int* err_no,
	const std::string& converter);

extern std::shared_ptr<FILE> hdfs_open_write(std::string path, int* err_no,
	const std::string& converter);

extern void hdfs_remove(const std::string& path);

extern std::vector<std::string> hdfs_list(const std::string& path);

extern std::string hdfs_tail(const std::string& path);

extern bool hdfs_exists(const std::string& path);

extern void hdfs_mkdir(const std::string& path);

// aut-detect fs
extern std::shared_ptr<FILE> fs_open_read(const std::string& path, int* err_no,
	const std::string& converter);

extern std::shared_ptr<FILE> fs_open_write(const std::string& path, int* err_no,
	const std::string& converter);

extern std::shared_ptr<FILE> fs_open(const std::string& path, const std::string& mode,
	int* err_no, const std::string& converter = "");

extern int64_t fs_file_size(const std::string& path);

extern void fs_remove(const std::string& path);

extern std::vector<std::string> fs_list(const std::string& path);

extern std::string fs_tail(const std::string& path);

extern bool fs_exists(const std::string& path);

extern void fs_mkdir(const std::string& path);

}
}
