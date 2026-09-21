# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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
"""Generation-side support for modular AOA statement generation.

The AOA engine in this package consumes ``source -> target`` statements; this
module backs the side that produces them, driven by the
``Layer.gen_aoa_statements`` / ``gen_inv_aoa_statements`` recursion. It holds
``AOAContext`` -- the frozen, behavior-free container of model config and naming
protocols, built once at the whole-model entry and forwarded unchanged to every
component override -- plus the stateless naming helpers used to read it.

Carrying data only means a component reading ``ctx`` for the forward and inverse
directions does not couple the two; forwarding a single ``ctx`` also turns a
missed constant map into an immediate ``TypeError`` instead of a silent bug.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = []


# Placeholder segments captured by name templates. They only ever appear
# in templates / final AOA macros, never as a ``pp_to_single_mapping`` query key.
ID_PLACEHOLDERS = frozenset({"$LAYER_ID", "$EXPERT_ID"})


@dataclass(frozen=True)
class AOAContext:
    """Frozen, read-only recursion context for AOA statement generation.

    Every field is constant across one whole-model generation pass. The
    whole-model entry builds one context (per tower for multi-tower models) and
    forwards it unchanged down the module tree. Per-position path info
    (``structured_name_prefix`` and the optional
    ``checkpoint_lookup_drop_segment``) travels alongside ``ctx`` rather than
    inside it. Maps are ready on construction: an empty mapping is an empty
    ``dict``, never ``None``.
    """

    config: object
    """Config of the sub-structure being generated: the whole-model config for a
    single-tower model, the current tower's sub-config for a multi-tower one."""

    checkpoint_name_prefix: str
    """Checkpoint root prefix shared by every layer in this generation pass. The
    identity fallback prepends it; a mapped name does not go through it."""

    checkpoint_name_mapping: Mapping[str, str]
    """Absolute model name template -> checkpoint name template. The key carries
    the model root; the value is the final checkpoint name as written, so it may
    sit outside the shared checkpoint root (as the output head does)."""

    pp_to_single_mapping: Mapping[str, str]
    """Structured name -> single name mapping."""

    model_name_prefix: str
    """Single-name model root prefix. Every ``checkpoint_name_mapping`` key
    carries it; the identity fallback strips it. Declared by the model, so a
    multi-tower model can root a tower elsewhere (e.g.
    ``"model.language_model"``)."""


def join_name(prefix: str, name: str) -> str:
    """Joins a prefix and a name into a dotted path."""
    if not prefix:
        return name
    if not name:
        return prefix
    return f"{prefix}.{name}"


def _match_template(template: str, name: str) -> dict[str, str] | None:
    """Compares a template against a name segment by segment.

    Placeholder segments (``$LAYER_ID`` / ``$EXPERT_ID``) match a decimal-only
    segment and capture its value; a repeated placeholder must capture the same
    value.
    """
    t_parts = template.split(".")
    n_parts = name.split(".")
    if len(t_parts) != len(n_parts):
        return None
    captures: dict[str, str] = {}
    for t_part, n_part in zip(t_parts, n_parts):
        if t_part in ID_PLACEHOLDERS:
            if (
                not n_part.isdigit()
                or captures.setdefault(t_part, n_part) != n_part
            ):
                return None
        elif t_part != n_part:
            return None
    return captures


def _render_template(template: str, captures: Mapping[str, str]) -> str:
    """Fills captured placeholders in a value template, segment by segment."""
    return ".".join(captures.get(p, p) for p in template.split("."))


def _strip_name_prefix(name: str, prefix: str) -> str:
    """Removes a leading prefix from a dotted name."""
    if name == prefix:
        return ""
    if name.startswith(prefix + "."):
        return name[len(prefix) + 1 :]
    raise ValueError(f"{name!r} is not under prefix {prefix!r}")


def strip_name_suffix(name: str, suffix: str) -> str:
    """Removes a trailing suffix from a dotted name."""
    if name == suffix:
        return ""
    if name.endswith("." + suffix):
        return name[: -len(suffix) - 1]
    raise ValueError(f"{name!r} does not end with suffix {suffix!r}")


def _map_checkpoint_name(
    single_name: str, checkpoint_name_mapping: Mapping[str, str]
) -> str | None:
    """Maps a single-space model name to its checkpoint name by template.

    Both sides of the mapping are absolute: a key starts at the model root and a
    value is a complete checkpoint name, so a hit is already the final
    checkpoint name and needs no further prefixing -- including when it sits
    outside the shared checkpoint root.
    """
    hits = []
    for key_template, value_template in checkpoint_name_mapping.items():
        captures = _match_template(key_template, single_name)
        if captures is not None:
            hits.append(_render_template(value_template, captures))
    if len(hits) > 1:
        raise ValueError(
            f"ambiguous checkpoint name mapping for {single_name!r}: {hits}"
        )
    return hits[0] if hits else None


def _resolve_checkpoint_name(
    single_name: str,
    checkpoint_name_prefix: str,
    checkpoint_name_mapping: Mapping[str, str],
    model_name_prefix: str,
) -> str:
    """Maps a single name to a full checkpoint name.

    A mapping hit is the full checkpoint name already. A miss falls back to the
    identity name, which still has to swap the model root for the checkpoint
    root -- the only place the two root prefixes are needed.
    """
    mapped_name = _map_checkpoint_name(single_name, checkpoint_name_mapping)
    if mapped_name is not None:
        return mapped_name
    return join_name(
        checkpoint_name_prefix,
        _strip_name_prefix(single_name, model_name_prefix),
    )


def _drop_checkpoint_lookup_segment(
    single_name: str, segment: str | None
) -> str:
    """Drops one live path segment before the checkpoint-name lookup.

    A subtree may be nested one module deeper in the live tree than in the
    checkpoint (an MTP block holding its transformer layer as a child, where
    the checkpoint keeps that layer's tensors directly under the layer).
    Removing the extra segment makes the subtree's names shaped like an
    ordinary layer's, so the same ``checkpoint_name_mapping`` templates and the
    same identity fallback apply. Only the lookup input changes; the model-side
    name keeps the segment.

    Absent segments are a no-op so an owner can pass the same value down to
    children that do not carry it.
    """
    if segment is None:
        return single_name
    parts = single_name.split(".")
    count = parts.count(segment)
    if count == 0:
        return single_name
    if count > 1:
        raise ValueError(
            f"checkpoint lookup segment {segment!r} appears {count} times in "
            f"{single_name!r}; which one to drop is ambiguous"
        )
    parts.remove(segment)
    return ".".join(parts)


def resolve_single_name(
    local_name: str,
    structured_name_prefix: str,
    pp_to_single_mapping: Mapping[str, str],
    model_name_prefix: str,
) -> str:
    """Resolves a real model tensor's structured name to its single name."""
    structured_name = structured_name_prefix + local_name
    if pp_to_single_mapping:
        try:
            return pp_to_single_mapping[structured_name]
        except KeyError:
            raise KeyError(
                f"structured name {structured_name!r} missing from "
                f"pp_to_single_mapping (local_name={local_name!r}, "
                f"structured_name_prefix={structured_name_prefix!r})"
            ) from None
    if structured_name == model_name_prefix or structured_name.startswith(
        model_name_prefix + "."
    ):
        return structured_name
    raise KeyError(
        f"empty pp_to_single_mapping only allows "
        f"{model_name_prefix}/{model_name_prefix}.* identity, "
        f"got {structured_name!r}"
    )


def resolve_names(
    local_name: str,
    checkpoint_name_prefix: str,
    structured_name_prefix: str,
    pp_to_single_mapping: Mapping[str, str],
    checkpoint_name_mapping: Mapping[str, str],
    *,
    model_name_prefix: str,
    checkpoint_lookup_drop_segment: str | None = None,
) -> tuple[str, str]:
    """Resolves a real model tensor to its ``(checkpoint_name, single_name)``
    pair."""
    # Pre-mapping live structured name -> canonical model name.
    single_name = resolve_single_name(
        local_name,
        structured_name_prefix,
        pp_to_single_mapping,
        model_name_prefix,
    )

    # single name -> checkpoint name
    checkpoint_name = _resolve_checkpoint_name(
        _drop_checkpoint_lookup_segment(
            single_name, checkpoint_lookup_drop_segment
        ),
        checkpoint_name_prefix,
        checkpoint_name_mapping,
        model_name_prefix,
    )

    return checkpoint_name, single_name


def resolve_checkpoint_name_from_anchor(
    anchor_single_name: str,
    anchor_local_name: str,
    checkpoint_local_name: str,
    checkpoint_name_prefix: str,
    checkpoint_name_mapping: Mapping[str, str],
    *,
    model_name_prefix: str,
    checkpoint_lookup_drop_segment: str | None = None,
) -> str:
    """Builds a checkpoint-only name (Q/K/V, gate/up, fused alpha) from an anchor.

    Strips ``anchor_local_name`` off the resolved model target to get the
    enclosing single-name scope, appends the checkpoint-only local name, then
    maps to the checkpoint side. Checkpoint-only names are never sent through
    ``pp_to_single_mapping``.
    """
    scope_single = strip_name_suffix(anchor_single_name, anchor_local_name)
    synthetic_single = join_name(scope_single, checkpoint_local_name)
    return _resolve_checkpoint_name(
        _drop_checkpoint_lookup_segment(
            synthetic_single, checkpoint_lookup_drop_segment
        ),
        checkpoint_name_prefix,
        checkpoint_name_mapping,
        model_name_prefix,
    )


def validate_checkpoint_name_mapping(
    checkpoint_name_mapping: Mapping[str, str],
    *,
    model_name_prefix: str,
) -> None:
    """Validates ``checkpoint_name_mapping`` once, from the generation entry.

    Called by the generation side while it builds an :class:`AOAContext` -- once
    per context, so a multi-tower model validates each tower's mapping. Running
    it there turns an invalid mapping into an error at the entry instead of a
    later ambiguous match or an unrendered placeholder leaking into a
    checkpoint name.

    Enforces four rules:

    - neither the key nor the value template may be an empty string;
    - a key template must sit under ``model_name_prefix``. A key is matched
      against a single name, which always carries that root, so a rootless key
      could never match and would silently degrade to the identity fallback
      instead of failing. The value side carries no such rule: a mapped value is
      the final checkpoint name, used exactly as written, so one that sits
      outside the shared checkpoint root is honoured rather than mis-keyed --
      which is how the ``ForCausalLM`` layout keeps its output head a top-level
      sibling of the backbone;
    - a ``$``-bearing segment must be a whole segment and a known placeholder,
      otherwise it would survive rendering into a bogus checkpoint name;
    - every placeholder used in a value template is captured by its key template
      (otherwise the value cannot be rendered).
    """
    for key_template, value_template in checkpoint_name_mapping.items():
        if not key_template or not value_template:
            raise ValueError(
                f"checkpoint name mapping contains an empty-string "
                f"key or value: {key_template!r} -> {value_template!r}"
            )
        if model_name_prefix and not (
            key_template == model_name_prefix
            or key_template.startswith(model_name_prefix + ".")
        ):
            raise ValueError(
                f"checkpoint name mapping {key_template!r} -> "
                f"{value_template!r} has key template {key_template!r} "
                f"outside its root prefix {model_name_prefix!r}; a key is an "
                f"absolute model name and must carry the model root"
            )
        for role, template in (
            ("key", key_template),
            ("value", value_template),
        ):
            for segment in template.split("."):
                if "$" in segment and segment not in ID_PLACEHOLDERS:
                    raise ValueError(
                        f"checkpoint name mapping {key_template!r} -> "
                        f"{value_template!r} has {role} segment {segment!r} "
                        f"containing '$'; a placeholder must be a whole dotted "
                        f"segment and one of {sorted(ID_PLACEHOLDERS)}"
                    )
        key_placeholders = {
            p for p in key_template.split(".") if p in ID_PLACEHOLDERS
        }
        value_placeholders = {
            p for p in value_template.split(".") if p in ID_PLACEHOLDERS
        }
        uncaptured = value_placeholders - key_placeholders
        if uncaptured:
            raise ValueError(
                f"checkpoint name mapping {key_template!r} -> {value_template!r} "
                f"uses placeholder(s) {sorted(uncaptured)} not captured by the "
                f"key template"
            )
