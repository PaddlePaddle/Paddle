#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import unittest

import numpy as np
from op_test import OpTest


class Segment:
    def __init__(self, chunk_type, start_idx, end_idx):
        self.chunk_type = chunk_type
        self.start_idx = start_idx
        self.end_idx = end_idx

    def __str__(self):
        return f'(Segment: {self.chunk_type}, {self.start_idx}, {self.end_idx})'

    __repr__ = __str__


class TestChunkEvalOp(OpTest):
    num_sequences = 5
    batch_size = 50

    def parse_scheme(self):
        if self.scheme in ['IOB', 'IOE']:
            self.num_tag_types = 2

    def fill_with_chunks(self, data, chunks):
        for chunk in chunks:
            if self.scheme == 'IOB':
                data[chunk.start_idx] = chunk.chunk_type * self.num_tag_types
                data[chunk.start_idx + 1 : chunk.end_idx] = (
                    chunk.chunk_type * self.num_tag_types
                    + (self.num_tag_types - 1)
                )
                data[chunk.end_idx] = (
                    chunk.chunk_type * self.num_tag_types
                    + (self.num_tag_types - 1)
                    if chunk.start_idx < chunk.end_idx
                    else data[chunk.start_idx]
                )
            elif self.scheme == 'IOE':
                data[chunk.start_idx : chunk.end_idx] = (
                    chunk.chunk_type * self.num_tag_types
                )
                data[chunk.end_idx] = chunk.chunk_type * self.num_tag_types + (
                    self.num_tag_types - 1
                )

    def rand_chunks(self, starts, num_chunks):
        if num_chunks < 0:
            num_chunks = np.random.randint(starts[-1])
        chunks = []
        # generate chunk beginnings
        chunk_begins = sorted(
            np.random.choice(list(range(starts[-1])), num_chunks, replace=False)
        )
        seq_chunk_begins = []
        begin_idx = 0
        # divide chunks into sequences
        for i in range(len(starts) - 1):
            tmp_chunk_begins = []
            while (
                begin_idx < len(chunk_begins)
                and chunk_begins[begin_idx] < starts[i + 1]
            ):
                tmp_chunk_begins.append(chunk_begins[begin_idx])
                begin_idx += 1
            seq_chunk_begins.append(tmp_chunk_begins)
        # generate chunk ends
        chunk_ends = []
        for i in range(len(seq_chunk_begins)):
            for j in range(len(seq_chunk_begins[i])):
                low = seq_chunk_begins[i][j]
                high = (
                    seq_chunk_begins[i][j + 1]
                    if j < len(seq_chunk_begins[i]) - 1
                    else starts[i + 1]
                )
                chunk_ends.append(np.random.randint(low, high))
        # generate chunks
        for chunk_pos in zip(chunk_begins, chunk_ends):
            chunk_type = np.random.randint(self.num_chunk_types)
            chunks.append(Segment(chunk_type, *chunk_pos))
        return chunks

    def gen_chunks(self, infer, label, starts):
        chunks = self.rand_chunks(
            starts,
            self.num_infer_chunks
            + self.num_label_chunks
            - self.num_correct_chunks,
        )
        correct_chunks = np.random.choice(
            list(range(len(chunks))), self.num_correct_chunks, replace=False
        )
        infer_chunks = np.random.choice(
            [x for x in range(len(chunks)) if x not in correct_chunks],
            self.num_infer_chunks - self.num_correct_chunks,
            replace=False,
        )
        infer_chunks = sorted(correct_chunks.tolist() + infer_chunks.tolist())
        label_chunks = np.random.choice(
            [x for x in range(len(chunks)) if x not in infer_chunks],
            self.num_label_chunks - self.num_correct_chunks,
            replace=False,
        )
        label_chunks = sorted(correct_chunks.tolist() + label_chunks.tolist())
        self.fill_with_chunks(infer, [chunks[idx] for idx in infer_chunks])
        self.fill_with_chunks(label, [chunks[idx] for idx in label_chunks])
        # exclude types in excluded_chunk_types
        if len(self.excluded_chunk_types) > 0:
            for idx in correct_chunks:
                if chunks[idx].chunk_type in self.excluded_chunk_types:
                    self.num_correct_chunks -= 1
            for idx in infer_chunks:
                if chunks[idx].chunk_type in self.excluded_chunk_types:
                    self.num_infer_chunks -= 1
            for idx in label_chunks:
                if chunks[idx].chunk_type in self.excluded_chunk_types:
                    self.num_label_chunks -= 1
        return (
            self.num_correct_chunks,
            self.num_infer_chunks,
            self.num_label_chunks,
        )

    def set_confs(self):
        # Use the IOB scheme and labels with 2 chunk types
        self.scheme = 'IOB'
        self.num_chunk_types = 2
        self.excluded_chunk_types = []
        self.other_chunk_type = self.num_chunk_types
        self.attrs = {
            'num_chunk_types': self.num_chunk_types,
            'chunk_scheme': self.scheme,
            'excluded_chunk_types': self.excluded_chunk_types,
        }
        self.parse_scheme()
        (
            self.num_correct_chunks,
            self.num_infer_chunks,
            self.num_label_chunks,
        ) = (4, 5, 9)

    def set_data(self):
        infer = np.zeros((self.batch_size,)).astype('int64')
        infer.fill(self.num_chunk_types * self.num_tag_types)
        label = np.copy(infer)
        starts = np.random.choice(
            list(range(1, self.batch_size)),
            self.num_sequences - 1,
            replace=False,
        ).tolist()
        starts.extend([0, self.batch_size])
        starts = sorted(starts)
        (
            self.num_correct_chunks,
            self.num_infer_chunks,
            self.num_label_chunks,
        ) = self.gen_chunks(infer, label, starts)
        lod = []
        for i in range(len(starts) - 1):
            lod.append(starts[i + 1] - starts[i])
        self.set_input(infer, label, lod)
        precision = (
            float(self.num_correct_chunks) / self.num_infer_chunks
            if self.num_infer_chunks
            else 0
        )
        recall = (
            float(self.num_correct_chunks) / self.num_label_chunks
            if self.num_label_chunks
            else 0
        )
        f1 = (
            float(2 * precision * recall) / (precision + recall)
            if self.num_correct_chunks
            else 0
        )
        self.outputs = {
            'Precision': np.asarray([precision], dtype='float32'),
            'Recall': np.asarray([recall], dtype='float32'),
            'F1-Score': np.asarray([f1], dtype='float32'),
            'NumInferChunks': np.asarray(
                [self.num_infer_chunks], dtype='int64'
            ),
            'NumLabelChunks': np.asarray(
                [self.num_label_chunks], dtype='int64'
            ),
            'NumCorrectChunks': np.asarray(
                [self.num_correct_chunks], dtype='int64'
            ),
        }

    def set_input(self, infer, label, lod):
        self.inputs = {'Inference': (infer, [lod]), 'Label': (label, [lod])}

    def setUp(self):
        self.op_type = 'chunk_eval'
        self.set_confs()
        self.set_data()

    def test_check_output(self):
        # NODE(yjjiang11): This op will be deprecated.
        self.check_output(check_dygraph=False)


class TestChunkEvalOpWithExclude(TestChunkEvalOp):
    def set_confs(self):
        # Use the IOE scheme and labels with 3 chunk types
        self.scheme = 'IOE'
        self.num_chunk_types = 3
        self.excluded_chunk_types = [1]
        self.other_chunk_type = self.num_chunk_types
        self.attrs = {
            'num_chunk_types': self.num_chunk_types,
            'chunk_scheme': self.scheme,
            'excluded_chunk_types': self.excluded_chunk_types,
        }
        self.parse_scheme()
        (
            self.num_correct_chunks,
            self.num_infer_chunks,
            self.num_label_chunks,
        ) = (15, 18, 20)


class TestChunkEvalOpWithTensorInput(TestChunkEvalOp):
    def set_input(self, infer, label, lod):
        max_len = np.max(lod)
        pad_infer = []
        pad_label = []
        start = 0
        for i in range(len(lod)):
            end = lod[i] + start
            pad_infer.append(
                np.pad(
                    infer[start:end],
                    (0, max_len - lod[i]),
                    'constant',
                    constant_values=(-1,),
                )
            )
            pad_label.append(
                np.pad(
                    label[start:end],
                    (0, max_len - lod[i]),
                    'constant',
                    constant_values=(-1,),
                )
            )
            start = end

        pad_infer = np.expand_dims(np.array(pad_infer, dtype='int64'), 2)
        pad_label = np.expand_dims(np.array(pad_label, dtype='int64'), 2)
        lod = np.array(lod, dtype='int64')
        self.inputs = {
            'Inference': pad_infer,
            'Label': pad_label,
            'SeqLength': lod,
        }


if __name__ == '__main__':
    unittest.main()
