# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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


from .metadata import LocalTensorIndex, LocalTensorMetadata, Metadata

TensorLocation = tuple[str, str]


class MetadataManager:
    def __init__(self):
        self._metadata_list: list[Metadata] = []
        self.local_tensor_metadata: dict[
            TensorLocation, LocalTensorMetadata
        ] = {}
        self.has_flattened_tensors: bool = False

    def set_metadata_list(self, metadata_list: list[Metadata]):
        assert len(metadata_list) == 1, "Only support single metadata list"

        self.local_tensor_metadata = {}
        self.has_flattened_tensors = False

        self._metadata_list = metadata_list
        self._extract_local_tensor_metadata()

    def get_metadata_list(self) -> list[Metadata]:
        return self._metadata_list

    def is_metadata_list_empty(self) -> bool:
        return not self._metadata_list

    def get_flat_mapping(self) -> dict:
        if self.is_metadata_list_empty():
            raise ValueError(
                "Cannot get flat mapping because metadata list is empty."
            )
        return self._metadata_list[0].flat_mapping

    def _extract_local_tensor_metadata(self):
        if self.is_metadata_list_empty():
            return

        metadata = self._metadata_list[0]
        state_dict_metadata = metadata.state_dict_metadata
        storage_metadata = metadata.storage_metadata

        for k, local_tensor_meta_list in state_dict_metadata.items():
            for local_tensor_meta in local_tensor_meta_list:
                local_tensor_index = LocalTensorIndex(
                    k,
                    local_tensor_meta.global_offset,
                    local_tensor_meta.is_flattened,
                    local_tensor_meta.flattened_range,
                )

                if local_tensor_index not in storage_metadata:
                    continue

                file_name = storage_metadata[local_tensor_index]
                location_key: TensorLocation = (k, file_name)

                self.local_tensor_metadata[location_key] = local_tensor_meta

                if local_tensor_meta.is_flattened:
                    self.has_flattened_tensors = True

    def clear(self):
        self._metadata_list = []
        self.local_tensor_metadata = {}
        self.has_flattened_tensors = False
