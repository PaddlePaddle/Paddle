#!/usr/bin/env python3
# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
from unittest.mock import patch

from paddle.distributed.flex_checkpoint.dcp.load_state_dict import (
    get_rank_to_read_files,
)


class TestGetRankToReadFiles(unittest.TestCase):
    """Unit tests for get_rank_to_read_files function."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_rank = 0
        self.patcher = patch(
            'paddle.distributed.get_rank', return_value=self.mock_rank
        )
        self.mock_get_rank = self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def test_local_files_assignment(self):
        """Test assignment when rank has all required files locally."""
        # Arrange
        rank_to_required = {
            0: ['file1.distcp', 'file2.distcp'],
            1: ['file3.distcp', 'file4.distcp'],
        }

        rank_to_available_files = {
            0: ['file1.distcp', 'file2.distcp'],
            1: ['file3.distcp', 'file4.distcp'],
        }

        # Act
        result = get_rank_to_read_files(
            rank_to_required, rank_to_available_files
        )

        # Assert
        expected_files = ['file1.distcp', 'file2.distcp']
        self.assertEqual(sorted(result), sorted(expected_files))

    def test_cross_node_files_assignment(self):
        """Test assignment when rank needs files from other nodes."""
        # Arrange
        rank_to_required = {
            0: ['file1.distcp', 'file2.distcp'],
            1: ['file1.distcp', 'file3.distcp'],
        }

        rank_to_available_files = {
            0: ['file1.distcp'],
            1: ['file2.distcp', 'file3.distcp'],
        }

        # Act
        result = get_rank_to_read_files(
            rank_to_required, rank_to_available_files
        )

        # Assert
        # Rank 0 should get file1 locally and file2 might be assigned from rank 1
        self.assertIn('file1.distcp', result)
        self.assertEqual(len(result), 1)  # Should balance workload

    def test_empty_rank_assignment(self):
        """Test when current rank has no files to read."""
        # Arrange
        rank_to_required = {
            1: ['file1.distcp', 'file2.distcp'],
            2: ['file3.distcp', 'file4.distcp'],
        }

        rank_to_available_files = {
            1: ['file1.distcp', 'file2.distcp'],
            2: ['file3.distcp', 'file4.distcp'],
        }

        self.mock_rank = 0  # Current rank has nothing to do

        # Act
        result = get_rank_to_read_files(
            rank_to_required, rank_to_available_files
        )

        # Assert
        self.assertEqual(result, [])

    def test_load_balancing_multiple_candidates(self):
        """Test load balancing when multiple ranks can read the same file."""
        # Arrange
        rank_to_required = {
            0: ['shared_file.distcp', 'file2.distcp'],
            1: ['shared_file.distcp', 'file3.distcp'],
            2: ['file4.distcp', 'file5.distcp'],
        }

        rank_to_available_files = {
            0: ['shared_file.distcp', 'file2.distcp'],
            1: ['shared_file.distcp', 'file3.distcp'],
            2: ['shared_file.distcp', 'file4.distcp', 'file5.distcp'],
        }

        # Act
        result = get_rank_to_read_files(
            rank_to_required, rank_to_available_files
        )

        # Assert
        # Should include shared_file and file2, but workload should be balanced
        self.assertIn('file2.distcp', result)
        self.assertEqual(len(result), 2)

    def test_missing_file_warning(self):
        """Test behavior when required file is not available on any rank."""
        # Arrange
        rank_to_required = {
            0: ['missing_file.distcp', 'existing_file.distcp'],
            1: ['file2.distcp'],
        }

        rank_to_available_files = {
            0: ['existing_file.distcp'],
            1: ['file2.distcp'],
        }
        # missing_file.distcp is not in any rank_to_available_files

        # Act & Assert - should not raise exception but handle gracefully
        result = get_rank_to_read_files(
            rank_to_required, rank_to_available_files
        )

        # Should still return files that are available
        self.assertIn('existing_file.distcp', result)

    def test_single_rank_scenario(self):
        """Test single rank scenario (non-distributed mode)."""
        # Arrange
        rank_to_required = {0: ['file1.distcp', 'file2.distcp', 'file3.distcp']}
        rank_to_available_files = {
            0: ['file1.distcp', 'file2.distcp', 'file3.distcp']
        }

        # Act
        result = get_rank_to_read_files(
            rank_to_required, rank_to_available_files
        )

        # Assert
        expected_files = ['file1.distcp', 'file2.distcp', 'file3.distcp']
        self.assertEqual(sorted(result), sorted(expected_files))

    def test_empty_inputs(self):
        """Test with empty input dictionaries."""
        # Arrange
        rank_to_required = {}
        rank_to_available_files = {}

        # Act
        result = get_rank_to_read_files(
            rank_to_required, rank_to_available_files
        )

        # Assert
        self.assertEqual(result, [])

    def test_rank_with_no_local_files(self):
        """Test when a rank has logical files but no local files available."""
        # Arrange
        rank_to_required = {
            0: ['file1.distcp', 'file2.distcp'],
            1: ['file3.distcp'],
        }

        rank_to_available_files = {
            1: ['file1.distcp', 'file2.distcp', 'file3.distcp']
        }
        # Rank 0 has no local files but needs file1 and file2

        # Act
        result = get_rank_to_read_files(
            rank_to_required, rank_to_available_files
        )

        # Assert
        # Rank 0 should get files assigned from rank 1
        self.assertEqual(len(result), 0)

    def test_rank_not_in_mappings(self):
        """Test when current rank is not present in input mappings."""
        # Arrange
        rank_to_required = {1: ['file1.distcp'], 2: ['file2.distcp']}

        rank_to_available_files = {1: ['file1.distcp'], 2: ['file2.distcp']}

        self.mock_rank = 0  # Current rank not in mappings

        # Act
        result = get_rank_to_read_files(
            rank_to_required, rank_to_available_files
        )

        # Assert
        self.assertEqual(result, [])

    def test_duplicate_files_across_ranks(self):
        """Test handling of duplicate files across different ranks."""
        # Arrange
        rank_to_required = {
            0: ['file1.distcp', 'file2.distcp'],
            1: ['file1.distcp', 'file3.distcp'],  # file1 duplicated
            2: ['file4.distcp'],
        }

        rank_to_available_files = {
            0: ['file1.distcp', 'file2.distcp'],
            1: ['file1.distcp', 'file3.distcp'],
            2: ['file4.distcp'],
        }

        # Act
        result = get_rank_to_read_files(
            rank_to_required, rank_to_available_files
        )

        # Assert
        # file1 should be assigned to only one rank (load balanced)
        self.assertIn('file1.distcp', result)
        # Should have exactly 2 files (file1 plus one more for balance)
        self.assertEqual(len(result), 2)


if __name__ == '__main__':
    unittest.main()
