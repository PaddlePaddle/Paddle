# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved
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

import os
import sys

__all__ = ['MultiSlotDataset']


class DatasetGenerator(object):
    def __init__(self):
        self._proto_info = None
        self._hadoop_host = None
        self._batch_size = 32
        self._hadoop_ugi = None
        self._hadoop_path = None

    def _set_proto_filename(self, proto_filename):
        if not isinstance(proto_filename, str):
            raise ValueError("proto_filename%s must be in str type" %
                             type(proto_filename))
        if not proto_filename:
            raise ValueError("proto_filename can not be empty")
        self._proto_filename = proto_filename

    def generate_sample(self, line):
        '''
        This function needs to be overridden by the user to process the
        original data row into a list or tuple

        Args:
            line(str): the original data row

        Returns:
            Returns the data processed by the user.
              The data format is list or tuple:
            [(name, [feasign, ...]), ...]
              or ((name, [feasign, ...]), ...)
            
            For example:
            [("words", [1926, 08, 17])], ("label", [1])]
              or (("words", [1926, 08, 17]), ("label", [1]))

        Note:
            The type of feasigns must be in int or float. Once the float
            element appears in the feasign, the type of that slot will be
            processed into a float.
        '''
        raise NotImplementedError(
            "please rewrite this function to return a list" +
            "[(name, [int, int ...]), ...]")

    def set_batch(self, batch):
        self.batch = batch

    def generate_batch(self, samples):
        '''
        This function can be overridden by the user to process batch
        data, a user can define how to generate batch with this function
        
        Args:
            samples(list of results from generate_samples)
        
        Returns:
            Returns the processed batch by the user
            [[(name, [int, ...]), ...],
             [(name, [int, ...]), ...],
             [(name, [int, ...])]]

        Default:
            Do nothing about current batch
        '''

        def batch_iter():
            for sample in samples:
                yield sample

        return batch_iter

    def _gen_str(self, line):
        raise NotImplementedError(
            "Please inherit this class and implement _gen_str")

    def _upload_proto_file(self):
        if self.proto_output_path == None:
            raise ValueError("If you are running data generation on hadoop, "
                             "please set proto output path first")

        if self._hadoop_host == None or self._hadoop_ugi == None or \
           self._hadoop_path == None:
            raise ValueError(
                "If you are running data generation on hadoop, "
                "please set hadoop_host, hadoop_path, hadoop_ugi first")
        cmd = "$HADOOP_HOME/bin/hadoop fs" \
              + " -Dhadoop.job.ugi=" + self.hadoop_ugi \
              + " -Dfs.default.name=" + self.hadoop_host \
              + " -put " + self._proto_filename + " " + self._proto_output_path
        os.system(cmd)

    def set_hadoop_config(self,
                          hadoop_host=None,
                          hadoop_ugi=None,
                          proto_path=None):
        '''
        This function set hadoop configuration for map-reduce based data
        generation. 
        
        Args:
            hadoop_host(str): The host name of the hadoop. It should be
                              in this format: "hdfs://${HOST}:${PORT}".
            hadoop_ugi(str): The ugi of the hadoop. It should be in this
                             format: "${USERNAME},${PASSWORD}".
            proto_path(str): The hadoop path you want to upload the
                             protofile to.
        '''
        self.hadoop_host = hadoop_host
        self.hadoop_ugi = hadoop_ugi
        self.proto_output_path = proto_path

    def run_from_memory(self, is_local=True, proto_filename='data_feed.proto'):
        '''
        This function generates data from memory, user needs to
        define how to generate samples by define generate_sample
        and generate_batch
        '''
        self._set_proto_filename(proto_filename)
        batch_data = []
        line_iter = self.generate_sample(None)
        for user_parsed_line in line_iter():
            if user_parsed_line == None:
                continue
            batch_data.append(user_parsed_line)
            if len(batch_data) == self._batch_size:
                batched_iter = self.generate_batch(batch_data)
                for batched_line in batched_iter():
                    sys.stdout.write(self._gen_str(batched_line))
                batch_data = []
        if len(batch_data) > 0:
            batched_iter = self.generate_batch(batch_data)
            for batched_line in batched_iter():
                sys.stdout.write(self._gen_str(batched_line))
        if self.proto_info is not None:
            with open(self._proto_filename, "w") as f:
                f.write(self._get_proto_desc(self._proto_info))
            if is_local == False:
                self._upload_proto_file()

    def run_from_stdin(self, is_local=True, proto_filename='data_feed.proto'):
        '''
        This function reads the data row from stdin, parses it with the
        process function, and further parses the return value of the
        process function with the _gen_str function. The parsed data will
        be wrote to stdout and the corresponding protofile will be
        generated. If local is set to False, the protofile will be
        uploaded to hadoop.
        
        Args:
            is_local(bool): Whether user wants to run this function from local
            proto_filename(str): The name of protofile. The default value
                                 is "data_feed.proto". It is not
                                 recommended to modify it.
        '''
        self._set_proto_filename(proto_filename)
        batch_data = []
        for line in sys.stdin:
            line_iter = self.generate_sample(line)
            for user_parsed_line in line_iter():
                if user_parsed_line == None:
                    continue
                batch_data.append(user_parsed_line)
                if len(batch_data) == self._batch_size:
                    batched_iter = self.generate_batch(batch_data)
                    for batched_line in batched_iter():
                        sys.stdout.write(self._gen_str(batched_line))
                    batch_data = []
        if len(batch_data) > 0:
            batched_iter = self.generate_batch(batch_data)
            for batched_line in batched_iter():
                sys.stdout.write(self._gen_str(batched_line))

        if self._proto_info is not None:
            with open(self._proto_filename, "w") as f:
                f.write(self._get_proto_desc(self._proto_info))
            if is_local == False:
                self._upload_proto_file()


class MultiSlotDataset(DatasetGenerator):
    def _get_proto_desc(self, proto_info):
        proto_str = "name: \"MultiSlotDataFeed\"\n" \
                    + "batch_size: 32\nmulti_slot_desc {\n"
        for elem in proto_info:
            proto_str += "  slots {\n" \
                         + "    name: \"%s\"\n" % elem[0]\
                         + "    type: \"%s\"\n" % elem[1]\
                         + "    is_dense: false\n" \
                         + "    is_used: false\n" \
                         + "  }\n"
        proto_str += "}"
        return proto_str

    def generate_batch(self, samples):
        super(MultiSlotDataset, self).generate_batch(samples)

        def batch_iter():
            for sample in samples:
                yield sample

        return batch_iter

    def _gen_str(self, line):
        if not isinstance(line, list) and not isinstance(line, tuple):
            raise ValueError(
                "the output of process() must be in list or tuple type")
        output = ""

        if self._proto_info is None:
            self._proto_info = []
            for item in line:
                name, elements = item
                if not isinstance(name, str):
                    raise ValueError("name%s must be in str type" % type(name))
                if not isinstance(elements, list):
                    raise ValueError("elements%s must be in list type" %
                                     type(elements))
                if not elements:
                    raise ValueError(
                        "the elements of each field can not be empty, you need padding it in process()."
                    )
                self._proto_info.append((name, "uint64"))
                if output:
                    output += " "
                output += str(len(elements))
                for elem in elements:
                    if isinstance(elem, float):
                        self._proto_info[-1] = (name, "float")
                    elif not isinstance(elem, int) and not isinstance(elem,
                                                                      long):
                        raise ValueError(
                            "the type of element%s must be in int or float" %
                            type(elem))
                    output += " " + str(elem)
        else:
            if len(line) != len(self._proto_info):
                raise ValueError(
                    "the complete field set of two given line are inconsistent.")
            for index, item in enumerate(line):
                name, elements = item
                if not isinstance(name, str):
                    raise ValueError("name%s must be in str type" % type(name))
                if not isinstance(elements, list):
                    raise ValueError("elements%s must be in list type" %
                                     type(elements))
                if not elements:
                    raise ValueError(
                        "the elements of each field can not be empty, you need padding it in process()."
                    )
                if name != self._proto_info[index][0]:
                    raise ValueError(
                        "the field name of two given line are not match: require<%s>, get<%d>."
                        % (self._proto_info[index][0], name))
                if output:
                    output += " "
                output += str(len(elements))
                for elem in elements:
                    if self._proto_info[index][1] != "float":
                        if isinstance(elem, float):
                            self._proto_info[index] = (name, "float")
                        elif not isinstance(elem, int) and not isinstance(elem,
                                                                          long):
                            raise ValueError(
                                "the type of element%s must be in int or float"
                                % type(elem))
                    output += " " + str(elem)
        return output + "\n"
