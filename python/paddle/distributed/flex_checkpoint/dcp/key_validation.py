# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import paddle
from paddle.distributed.fleet.utils.log_util import logger

if TYPE_CHECKING:
    from paddle.distributed.communication.group import Group

    from ..aoa.aoa_engine import AOAEngine
    from .metadata import Metadata

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_MAX_TOTAL_LINES = 500
_MAX_KEYS_SHOWN = 50
_MAX_SHAPE_MISMATCHES = 20
_MAX_PATTERNS_SHOWN = 30
_SRC_FOLD_THRESHOLD = 5
_MAX_SLICE_DETAIL_KEYS = 5


def _get_rank() -> int:
    return paddle.distributed.get_rank()


# ---------------------------------------------------------------------------
# Color support (disabled by default)
# ---------------------------------------------------------------------------


class _C:
    """No-op color helpers. Colors are disabled by default."""

    @staticmethod
    def green(t):
        return t

    @staticmethod
    def yellow(t):
        return t

    @staticmethod
    def red(t):
        return t

    @staticmethod
    def cyan(t):
        return t


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ShapeMismatchInfo:
    key: str
    src_global_shape: tuple[int, ...]
    dst_global_shape: tuple[int, ...]
    src_dtype: str | None = None
    dst_dtype: str | None = None


@dataclass
class KeyValidationResult:
    missing_keys: set[str] = field(default_factory=set)
    unexpected_keys: set[str] = field(default_factory=set)
    shape_mismatches: list[ShapeMismatchInfo] = field(default_factory=list)
    randomly_initialized_keys: set[str] = field(default_factory=set)


@dataclass
class AOASliceMapping:
    src_key: str
    src_slice: tuple[slice, ...]
    dst_slice: tuple[slice, ...]
    postprocess: list[str] | None = None


@dataclass
class AOAMappingEntry:
    dst_key: str
    dst_global_shape: tuple[int, ...]
    slice_mappings: list[AOASliceMapping] = field(default_factory=list)
    is_identity: bool = False


# ---------------------------------------------------------------------------
# Public API: Standard (non-AOA) validation
# ---------------------------------------------------------------------------


def validate_and_report_keys_standard(
    metadata_list: list[Metadata],
    state_dict_param_names: set[str],
    process_group: Group | None,
    use_dist: bool,
    checkpoint_path: str,
    state_dict: dict,
) -> KeyValidationResult:
    """Validate keys for the standard (non-AOA) loading path.

    Gathers global dst keys across all ranks, compares with global src keys,
    checks shape mismatches. Prints report on rank 0 only.
    """
    # 1. Gather global dst keys
    if use_dist:
        global_dst_key_list = []
        paddle.distributed.all_gather_object(
            global_dst_key_list, list(state_dict_param_names), process_group
        )
        global_dst_keys = {
            k for sublist in global_dst_key_list for k in sublist
        }
    else:
        global_dst_keys = state_dict_param_names

    # 2. Collect global src keys from metadata
    global_src_keys = set()
    for metadata in metadata_list:
        for local_tensor_index in metadata.storage_metadata:
            if (
                local_tensor_index.replica_id is not None
                and local_tensor_index.replica_id != 0
            ):
                continue
            global_src_keys.add(local_tensor_index.tensor_key)

    # 3. Compute missing / unexpected
    missing_keys = global_dst_keys - global_src_keys
    unexpected_keys = global_src_keys - global_dst_keys

    # 4. Check shape mismatches for matching keys
    shape_mismatches = []
    assert state_dict is not None, "state_dict must not be None"
    # Gather dst global shapes: {key: global_shape}
    local_dst_shapes = {}
    for key, val in state_dict.items():
        k = key if isinstance(key, str) else key[0]
        if hasattr(val, "global_shape"):
            local_dst_shapes[k] = tuple(val.global_shape)
        else:
            local_dst_shapes[k] = tuple(val.shape)

    if use_dist:
        all_dst_shapes_list = []
        paddle.distributed.all_gather_object(
            all_dst_shapes_list, local_dst_shapes, process_group
        )
        global_dst_shapes = {}
        for d in all_dst_shapes_list:
            global_dst_shapes.update(d)
    else:
        global_dst_shapes = local_dst_shapes

    # Build src global shapes from metadata
    src_global_shapes: dict[str, tuple[int, ...]] = {}
    for metadata in metadata_list:
        if not metadata.state_dict_metadata:
            continue
        for key, src_metas in metadata.state_dict_metadata.items():
            if not src_metas or src_metas[0].global_shape is None:
                continue
            src_global_shapes[key] = tuple(src_metas[0].global_shape)

    matching_keys = global_dst_keys & global_src_keys
    for key in sorted(matching_keys):
        src_shape = src_global_shapes.get(key)
        dst_shape = global_dst_shapes.get(key)
        if src_shape is None or dst_shape is None:
            continue
        if src_shape != dst_shape:
            shape_mismatches.append(
                ShapeMismatchInfo(
                    key=key,
                    src_global_shape=src_shape,
                    dst_global_shape=dst_shape,
                )
            )

    result = KeyValidationResult(
        missing_keys=missing_keys,
        unexpected_keys=unexpected_keys,
        shape_mismatches=shape_mismatches,
        randomly_initialized_keys=set(),
    )

    # 5. Print on rank 0 (or always when not using dist)
    if not use_dist or _get_rank() == 0:
        _print_standard_report(result, checkpoint_path, len(global_dst_keys))

    return result


# ---------------------------------------------------------------------------
# Public API: AOA validation
# ---------------------------------------------------------------------------


def validate_and_report_keys_aoa(
    aoa_engine: AOAEngine,
    metadata: Metadata,
    checkpoint_path: str,
    use_dist: bool = True,
) -> KeyValidationResult:
    """Validate keys for the AOA loading path.

    Called AFTER AOAEngine is initialized. Uses output_vars/input_vars to
    compute truly missing/unexpected keys and builds the mapping table.
    """
    # 1. Covered dst keys
    aoa_covered_dst_keys = {
        k for k, v in aoa_engine.output_vars.items() if v is not None
    }
    randomly_initialized_keys = set(aoa_engine.need_add_output_vars)

    # 2. Consumed src keys
    consumed_src_keys = set()
    for tensor_desc in aoa_engine.output_vars.values():
        if tensor_desc is None:
            continue
        for src_key, _, _, _ in tensor_desc.slices:
            consumed_src_keys.add(src_key)

    # 3. Explicitly removed / all src keys
    explicitly_removed = set(aoa_engine.need_remove_input_vars)
    all_src_keys = set(aoa_engine.input_vars.keys())

    # 4. Compute truly missing / unexpected
    dst_state_keys = aoa_engine.context.get_all_dst_state_keys()
    truly_missing = (
        dst_state_keys - aoa_covered_dst_keys - randomly_initialized_keys
    )
    truly_unexpected = all_src_keys - consumed_src_keys - explicitly_removed

    # 5. Build AOA mapping entries
    aoa_mappings = _build_aoa_mappings(aoa_engine)

    result = KeyValidationResult(
        missing_keys=truly_missing,
        unexpected_keys=truly_unexpected,
        shape_mismatches=[],
        randomly_initialized_keys=randomly_initialized_keys,
    )

    # 6. Print on rank 0 (or always when not using dist)
    if not use_dist or _get_rank() == 0:
        _print_aoa_report(
            result, aoa_mappings, explicitly_removed, checkpoint_path
        )

    return result


def _build_aoa_mappings(aoa_engine: AOAEngine) -> list[AOAMappingEntry]:
    """Extract mapping entries from AOA engine's output_vars."""
    entries = []
    for dst_key, tensor_desc in sorted(aoa_engine.output_vars.items()):
        if tensor_desc is None:
            continue
        shape = tuple(tensor_desc.shape)
        slice_mappings = []
        for src_key, src_sl, dst_sl, pp_list in tensor_desc.slices:
            slice_mappings.append(
                AOASliceMapping(
                    src_key=src_key,
                    src_slice=src_sl,
                    dst_slice=dst_sl,
                    postprocess=pp_list,
                )
            )
        # Determine if identity
        is_identity = (
            len(slice_mappings) == 1
            and slice_mappings[0].src_key == dst_key
            and slice_mappings[0].postprocess is None
            and _slice_covers_full(slice_mappings[0].dst_slice, shape)
        )
        entries.append(
            AOAMappingEntry(
                dst_key=dst_key,
                dst_global_shape=shape,
                slice_mappings=slice_mappings,
                is_identity=is_identity,
            )
        )
    return entries


def _slice_covers_full(sl: tuple[slice, ...], shape: tuple[int, ...]) -> bool:
    """Check if a slice tuple covers the full tensor."""
    if len(sl) != len(shape):
        return False
    for s, dim in zip(sl, shape):
        if s.start != 0 or s.stop != dim:
            return False
    return True


# ---------------------------------------------------------------------------
# Printing: Standard report
# ---------------------------------------------------------------------------

_SEP = "=" * 70
_THIN_SEP = "-" * 70


def _print_standard_report(
    result: KeyValidationResult, path: str, total_keys: int
) -> None:
    lines = [_SEP, f"FlexCheckpoint Load Report (Checkpoint: {path})", _SEP]

    if (
        not result.missing_keys
        and not result.unexpected_keys
        and not result.shape_mismatches
    ):
        lines.append(
            _C.green(
                f"[OK] All {total_keys} keys matched successfully. "
                f"(missing: 0, unexpected: 0, shape_mismatch: 0)"
            )
        )
    else:
        matched = total_keys - len(result.missing_keys)
        lines.append(
            f"Matched: {matched}/{total_keys} keys | "
            f"Missing: {len(result.missing_keys)} | "
            f"Unexpected: {len(result.unexpected_keys)} | "
            f"Shape mismatch: {len(result.shape_mismatches)}"
        )
        if result.missing_keys:
            lines.append("")
            lines.append(
                _C.yellow(
                    f"[WARNING] Missing keys ({len(result.missing_keys)} total) "
                    f"- model expects but not in checkpoint:"
                )
            )
            lines.extend(_format_key_list(result.missing_keys))
        if result.unexpected_keys:
            lines.append("")
            lines.append(
                _C.yellow(
                    f"[WARNING] Unexpected keys ({len(result.unexpected_keys)} total) "
                    f"- in checkpoint but not used:"
                )
            )
            lines.extend(_format_key_list(result.unexpected_keys))
        if result.shape_mismatches:
            lines.append("")
            lines.append(
                _C.yellow(
                    f"[WARNING] Shape mismatches ({len(result.shape_mismatches)} total):"
                )
            )
            for m in result.shape_mismatches[:_MAX_SHAPE_MISMATCHES]:
                lines.append(
                    f"    {m.key}: ckpt={list(m.src_global_shape)} vs model={list(m.dst_global_shape)}"
                )
            remaining = len(result.shape_mismatches) - _MAX_SHAPE_MISMATCHES
            if remaining > 0:
                lines.append(f"    ... and {remaining} more")

    lines.append(_SEP)
    _emit(lines)


# ---------------------------------------------------------------------------
# Printing: AOA report
# ---------------------------------------------------------------------------


def _print_aoa_report(
    result: KeyValidationResult,
    aoa_mappings: list[AOAMappingEntry],
    explicitly_removed: set[str],
    path: str,
) -> None:
    lines = [
        _SEP,
        f"FlexCheckpoint Load Report (Checkpoint: {path}, AOA enabled)",
        _SEP,
    ]

    # Status
    total_dst = (
        len(aoa_mappings)
        + len(result.missing_keys)
        + len(result.randomly_initialized_keys)
    )
    if not result.missing_keys and not result.unexpected_keys:
        lines.append(
            _C.green(
                f"[OK] All {total_dst} keys resolved via AOA mapping. "
                f"(missing: 0, unexpected: 0)"
            )
        )
    else:
        matched = total_dst - len(result.missing_keys)
        lines.append(
            f"Matched: {matched}/{total_dst} keys | "
            f"Missing: {len(result.missing_keys)} | "
            f"Unexpected: {len(result.unexpected_keys)}"
        )
        if result.missing_keys:
            lines.append("")
            lines.append(
                _C.yellow(
                    f"[WARNING] Missing keys ({len(result.missing_keys)} total) "
                    f"- no AOA source mapping:"
                )
            )
            lines.extend(_format_key_list(result.missing_keys))
        if result.unexpected_keys:
            lines.append("")
            lines.append(
                _C.yellow(
                    f"[WARNING] Unexpected keys ({len(result.unexpected_keys)} total) "
                    f"- in checkpoint but not consumed by any AOA mapping:"
                )
            )
            lines.extend(_format_key_list(result.unexpected_keys))

    # AOA mapping table
    lines.append("")
    lines.append(_C.cyan(_THIN_SEP))

    # Classify mappings
    non_identity = [m for m in aoa_mappings if not m.is_identity]
    rename_only, with_transform, structural = _classify_mappings(non_identity)

    total_dst = len(aoa_mappings)
    total_src = len(
        {sm.src_key for m in aoa_mappings for sm in m.slice_mappings}
    )
    lines.append(
        _C.cyan(f"AOA Key Mapping ({total_dst} dst keys, {total_src} src keys)")
    )
    lines.append(_C.cyan(_THIN_SEP))

    # Summary
    lines.append("Summary:")
    lines.append(
        f"  1-to-1 rename (same shape, no transform):  {len(rename_only)} keys (not shown)"
    )
    lines.append(
        f"  1-to-1 with transform:                     {len(with_transform)} keys "
        f"({min(len(_group_by_signature(with_transform)), _MAX_PATTERNS_SHOWN)} pattern(s) below)"
    )
    lines.append(
        f"  Structural (N-to-1 / 1-to-N / reshape):    {len(structural)} keys "
        f"({min(len(_group_by_signature(structural)), _MAX_PATTERNS_SHOWN)} pattern(s) below)"
    )

    # Print transform patterns
    next_index = 1
    if with_transform:
        lines.append("")
        result_lines, next_index = _format_pattern_groups(
            _group_by_signature(with_transform), "1-to-1 transform", next_index
        )
        lines.extend(result_lines)

    # Print structural patterns
    if structural:
        lines.append("")
        result_lines, next_index = _format_pattern_groups(
            _group_by_signature(structural), "structural", next_index
        )
        lines.extend(result_lines)

    # Removed / Initialized
    lines.append("")
    removed_str = ", ".join(sorted(explicitly_removed)[:5])
    if len(explicitly_removed) > 5:
        removed_str += f" ... +{len(explicitly_removed) - 5} more"
    lines.append(f"Removed ({len(explicitly_removed)}): {removed_str or '-'}")
    init_keys = result.randomly_initialized_keys
    init_str = ", ".join(sorted(init_keys)[:5])
    if len(init_keys) > 5:
        init_str += f" ... +{len(init_keys) - 5} more"
    lines.append(f"Initialized ({len(init_keys)}): {init_str or '-'}")

    lines.append("")
    lines.append(_THIN_SEP)
    lines.append(_SEP)
    _emit(lines)


# ---------------------------------------------------------------------------
# Helpers: Classification & Pattern Merging
# ---------------------------------------------------------------------------


def _classify_mappings(
    non_identity: list[AOAMappingEntry],
) -> tuple[list[AOAMappingEntry], list[AOAMappingEntry], list[AOAMappingEntry]]:
    """Classify non-identity mappings into rename_only, with_transform, structural."""
    rename_only = []
    with_transform = []
    structural = []
    for entry in non_identity:
        if len(entry.slice_mappings) != 1:
            structural.append(entry)
            continue
        sm = entry.slice_mappings[0]
        src_norm = re.sub(r"\d+", "{N}", sm.src_key)
        dst_norm = re.sub(r"\d+", "{N}", entry.dst_key)
        if src_norm != dst_norm:
            structural.append(entry)
        elif sm.postprocess is None:
            rename_only.append(entry)
        else:
            with_transform.append(entry)
    return rename_only, with_transform, structural


def _get_signature(entry: AOAMappingEntry) -> str:
    """Compute a structure signature for pattern grouping."""
    dst_norm = re.sub(r"\d+", "{N}", entry.dst_key)
    parts = [dst_norm, str(len(entry.slice_mappings))]
    for sm in entry.slice_mappings:
        src_norm = re.sub(r"\d+", "{N}", sm.src_key)
        pp = "|".join(sm.postprocess) if sm.postprocess else ""
        parts.append(f"{src_norm}:{pp}")
    return "@@".join(parts)


def _group_by_signature(
    entries: list[AOAMappingEntry],
) -> dict[str, list[AOAMappingEntry]]:
    """Group entries by structure signature."""
    groups: dict[str, list[AOAMappingEntry]] = defaultdict(list)
    for entry in entries:
        groups[_get_signature(entry)].append(entry)
    return groups


def _format_pattern_groups(
    groups: dict[str, list[AOAMappingEntry]], label: str, start_index: int = 1
) -> tuple[list[str], int]:
    """Format grouped patterns with box-drawing style. Returns (lines, next_index)."""
    lines = []
    shown = 0
    idx = start_index
    for _sig, entries in sorted(groups.items(), key=lambda x: -len(x[1])):
        if shown >= _MAX_PATTERNS_SHOWN:
            remaining = len(groups) - shown
            lines.append(f"  ... and {remaining} more {label} pattern(s)")
            break
        shown += 1
        representative = entries[0]
        count = len(entries)

        # Build pattern title
        dst_pattern = re.sub(r"\d+", "*", representative.dst_key)
        lines.append(f"[Pattern #{idx}] {dst_pattern}  ({count} keys, {label})")
        lines.append("\u250c" + "\u2500" * 69)
        # DST line
        shape_str = list(representative.dst_global_shape)
        lines.append(f"\u2502 DST: {representative.dst_key}  {shape_str}")
        # SRC lines (with folding)
        _append_src_lines(lines, representative.slice_mappings)
        # OP line
        ops = _describe_ops(representative)
        if ops:
            lines.append(f"\u2502 OP:  {ops}")
        lines.append("\u2514" + "\u2500" * 69)
        lines.append("")
        idx += 1
    return lines, idx


def _append_src_lines(
    lines: list[str], slice_mappings: list[AOASliceMapping]
) -> None:
    """Append SRC lines, folding consecutive numeric patterns."""
    if len(slice_mappings) <= _SRC_FOLD_THRESHOLD:
        for i, sm in enumerate(slice_mappings):
            prefix = "\u2502 SRC:" if i == 0 else "\u2502    +"
            slice_info = _format_slice_range(sm.src_slice, sm.dst_slice)
            lines.append(f"{prefix} {sm.src_key}{slice_info}")
        return

    # Try to fold: find common pattern
    src_keys = [sm.src_key for sm in slice_mappings]
    folded = _try_fold_src_keys(src_keys)
    if folded:
        lines.append(f"\u2502 SRC: {folded}  (\u00d7{len(slice_mappings)})")
    else:
        # Show first 2 and last 1
        lines.append(f"\u2502 SRC: {src_keys[0]}")
        lines.append(f"\u2502    + {src_keys[1]}")
        lines.append(f"\u2502    + ... ({len(src_keys) - 3} more)")
        lines.append(f"\u2502    + {src_keys[-1]}")


def _format_slice_range(
    src_slice: tuple[slice, ...], dst_slice: tuple[slice, ...]
) -> str:
    """Format slice info when same src_key appears multiple times."""
    src_str = ",".join(f"{s.start}:{s.stop}" for s in src_slice)
    dst_str = ",".join(f"{s.start}:{s.stop}" for s in dst_slice)
    return f"  [{src_str}] -> dst[{dst_str}]"


def _try_fold_src_keys(keys: list[str]) -> str | None:
    """Try to fold src keys like experts.0, experts.1, ..., experts.255 into a pattern."""
    if len(keys) < 2:
        return None
    # Find varying digit segments
    pattern = re.sub(r"\d+", "{}", keys[0])
    for k in keys[1:]:
        if re.sub(r"\d+", "{}", k) != pattern:
            return None
    # Extract the varying numbers
    nums_per_key = [re.findall(r"\d+", k) for k in keys]
    num_positions = len(nums_per_key[0])
    # Find which position varies
    varying_pos = []
    for pos in range(num_positions):
        vals = [int(n[pos]) for n in nums_per_key]
        if len(set(vals)) > 1:
            varying_pos.append(pos)
    if len(varying_pos) != 1:
        return None
    vpos = varying_pos[0]
    vals = [int(n[vpos]) for n in nums_per_key]
    lo, hi = min(vals), max(vals)
    # Reconstruct pattern with {lo..hi}
    segments = re.split(r"\d+", keys[0])
    digits = re.findall(r"\d+", keys[0])
    result_parts = []
    for i, seg in enumerate(segments):
        result_parts.append(seg)
        if i < len(digits):
            if i == vpos:
                result_parts.append(f"{{{lo}..{hi}}}")
            else:
                result_parts.append(digits[i])
    return "".join(result_parts)


def _describe_ops(entry: AOAMappingEntry) -> str:
    """Describe the operations for a mapping entry."""
    ops = []
    if len(entry.slice_mappings) > 1:
        ops.append("concat")
    # Collect postprocess from first slice (representative)
    if entry.slice_mappings:
        pp = entry.slice_mappings[0].postprocess
        if pp:
            for p in pp:
                if p.startswith("["):
                    ops.append(f"permute({p})")
                else:
                    ops.append(f"cast({p})")
    return " + ".join(ops)


# ---------------------------------------------------------------------------
# Helpers: Key list formatting
# ---------------------------------------------------------------------------


def _format_key_list(keys: set[str]) -> list[str]:
    """Format a set of keys with prefix grouping and truncation."""
    if not keys:
        return []
    sorted_keys = sorted(keys)
    if len(sorted_keys) <= _MAX_KEYS_SHOWN:
        return [f"    {k}" for k in sorted_keys]

    # Adaptive grouping: find the prefix depth that gives reasonable group sizes
    groups = _group_keys_adaptive(sorted_keys)

    lines = []
    groups_shown = 0
    for prefix, group_keys in sorted(groups.items(), key=lambda x: -len(x[1])):
        if groups_shown >= _MAX_KEYS_SHOWN:
            remaining_groups = len(groups) - groups_shown
            remaining_keys = sum(
                len(v)
                for i, v in enumerate(
                    sorted(groups.values(), key=len, reverse=True)
                )
                if i >= groups_shown
            )
            lines.append(
                f"    ... and {remaining_groups} more groups ({remaining_keys} keys)"
            )
            break
        groups_shown += 1
        if len(group_keys) > 3:
            lines.append(f"    [{prefix}] ({len(group_keys)} keys):")
            for k in group_keys[:3]:
                lines.append(f"      {k}")
            lines.append(f"      ... +{len(group_keys) - 3} more")
        else:
            for k in group_keys:
                lines.append(f"    {k}")
    return lines


def _group_keys_adaptive(keys: list[str]) -> dict[str, list[str]]:
    """Group keys by normalized pattern (digits replaced with *)."""
    groups: dict[str, list[str]] = defaultdict(list)
    for k in keys:
        # Replace all digit segments with * to get the pattern
        pattern = re.sub(r"(?<=\.)\d+(?=\.)|(?<=\.)\d+$", "*", k)
        groups[pattern].append(k)
    return dict(groups)


# ---------------------------------------------------------------------------
# Helpers: Output
# ---------------------------------------------------------------------------


def _emit(lines: list[str]) -> None:
    """Output lines via logger, respecting total line limit."""
    for i, line in enumerate(lines):
        if i >= _MAX_TOTAL_LINES:
            logger.info(
                f"... output truncated ({len(lines) - i} lines omitted)"
            )
            break
        logger.info(line)
