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
``Layer.gen_aoa_statements`` / ``gen_inv_aoa_statements`` recursion. It holds the
two frozen, behavior-free containers threaded through that recursion, plus the
stateless naming helpers used to read them:

- ``AOAContext``: model config and naming protocols, built once at the
  whole-model entry and forwarded unchanged to every component override.
- ``AOANameScope``: path information for a subtree whose checkpoint side is
  re-rooted (MTP subtrees, the output head).

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
    (``structured_name_prefix`` and the optional ``AOANameScope``) travels
    alongside ``ctx`` rather than inside it. Maps are ready on construction: an
    empty mapping is an empty ``dict``, never ``None``.
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


@dataclass(frozen=True)
class AOANameScope:
    """Path scope for a subtree whose checkpoint side is re-rooted.

    Built at the whole-model entry for one subtree call, then forwarded as-is
    during that subtree's recursion. Used wherever the checkpoint layout does
    not follow the live module path: MTP subtrees (checkpoint keeps them under
    their own root) and the output head. It is not a new model protocol and
    holds no leaf mapping.
    """

    checkpoint_prefix: str
    """Checkpoint subtree prefix, relative to the shared checkpoint name prefix
    (unless ``is_checkpoint_prefix_absolute`` is set, in which case it is the
    full prefix and the shared checkpoint name prefix is not prepended)."""

    logical_model_prefix: str
    """Logical model prefix used only to match ``checkpoint_name_mapping``."""

    actual_model_prefix: str
    """Real model single subtree prefix (after PP mapping resolution)."""

    is_checkpoint_prefix_absolute: bool = False
    """When True, ``checkpoint_prefix`` is already a full checkpoint prefix and
    ``_scoped_checkpoint_name`` does not prepend ``checkpoint_name_prefix``.
    Needed whenever the subtree's checkpoint names sit at the checkpoint root
    rather than under the shared prefix: an MTP boundary that keeps its own
    params at the root, or the output head, which the ``ForCausalLM`` layout
    keeps a top-level sibling of the backbone (bare ``lm_head.weight``)."""


# --------------------------------------------------------------------------- #
# Naming & prefix helpers (stateless functions)
#
# Public to the component generators: ``join_name``, ``resolve_single_name``,
# ``resolve_names``, ``resolve_checkpoint_name_from_anchor`` and
# ``strip_name_suffix``.
#
# ``validate_checkpoint_name_mapping`` is public as well but does not belong to
# the component recursion: it is an entry-side static check the generation entry
# runs once per context, before any component reads the mapping.
#
# Anything underscore-prefixed is internal detail. The only non-trivial logic is
# absolute-name template matching (``_match_template`` / ``_render_template``).
# A re-rooted subtree passes an ``AOANameScope`` so its checkpoint side is
# routed through the logical normal-layer root (``_resolve_scoped_names``).
# --------------------------------------------------------------------------- #


def join_name(prefix: str, name: str) -> str:
    """Joins a prefix and a name into a dotted path.

    Args:
        prefix: Leading path segment; may be empty.
        name: Trailing path segment; may be empty.

    Returns:
        ``"{prefix}.{name}"``, or whichever operand is non-empty when the other
        is empty.
    """
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

    Args:
        template: Dotted template, possibly containing placeholder segments.
        name: Dotted name to test against ``template``.

    Returns:
        The capture dict on a match (possibly empty) or ``None`` on a miss.
        Callers must test ``is not None`` since an empty dict is falsy.
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
    """Fills captured placeholders in a value template, segment by segment.

    Args:
        template: Dotted value template, possibly containing placeholders.
        captures: Placeholder -> captured value mapping from ``_match_template``.

    Returns:
        The template with every captured placeholder replaced by its value.
    """
    return ".".join(captures.get(p, p) for p in template.split("."))


def _strip_name_prefix(name: str, prefix: str) -> str:
    """Removes a leading prefix from a dotted name.

    Args:
        name: Dotted name expected to be at or under ``prefix``.
        prefix: Prefix path to strip.

    Returns:
        ``name`` with ``prefix`` removed (empty string when equal).

    Raises:
        ValueError: If ``name`` is not at or under ``prefix``.
    """
    if name == prefix:
        return ""
    if name.startswith(prefix + "."):
        return name[len(prefix) + 1 :]
    raise ValueError(f"{name!r} is not under prefix {prefix!r}")


def strip_name_suffix(name: str, suffix: str) -> str:
    """Removes a trailing suffix from a dotted name.

    Args:
        name: Dotted name expected to end with ``suffix``.
        suffix: Suffix path to strip.

    Returns:
        ``name`` with the trailing ``suffix`` removed (empty string when equal).

    Raises:
        ValueError: If ``name`` does not end with ``suffix``.
    """
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

    Args:
        single_name: Single-space model name, starting at the model root prefix.
        checkpoint_name_mapping: Absolute model name template -> absolute
            checkpoint name template.

    Returns:
        The rendered checkpoint name on exactly one hit, or ``None`` when no
        template matches. A miss is not an error: the caller falls back to the
        identity name.

    Raises:
        ValueError: If more than one template matches (ambiguous config).
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

    Args:
        single_name: Single-space model name, starting at the model root prefix.
        checkpoint_name_prefix: Checkpoint root prefix, prepended by the
            identity fallback only.
        checkpoint_name_mapping: Absolute model name template -> absolute
            checkpoint name template.
        model_name_prefix: Model root prefix, stripped by the identity fallback
            only.

    Returns:
        The full checkpoint name.
    """
    mapped_name = _map_checkpoint_name(single_name, checkpoint_name_mapping)
    if mapped_name is not None:
        return mapped_name
    return join_name(
        checkpoint_name_prefix,
        _strip_name_prefix(single_name, model_name_prefix),
    )


def _scoped_checkpoint_name(
    single_space_name: str,
    checkpoint_name_prefix: str,
    checkpoint_name_mapping: Mapping[str, str],
    aoa_name_scope: AOANameScope,
    model_name_prefix: str,
) -> str:
    """Maps a re-rooted subtree's single-space name to its checkpoint name.

    Extracts the subtree-relative path off the real model subtree root, replaces
    it under the logical normal-layer root so the same ``checkpoint_name_mapping``
    leaf rules apply, then re-anchors the mapped leaf name under the checkpoint
    subtree root. Relies on leaf templates preserving their root prefix in the
    value (``model.layers.$LAYER_ID.x`` ->
    ``<checkpoint root>.layers.$LAYER_ID.y``), which is what makes the logical
    root strippable from the mapped result.

    Args:
        single_space_name: Single-space model name under the real subtree root.
        checkpoint_name_prefix: Checkpoint root prefix to prepend.
        checkpoint_name_mapping: Absolute model name template -> absolute
            checkpoint name template.
        aoa_name_scope: Subtree path scope (real / logical / checkpoint roots).
        model_name_prefix: Model root prefix.

    Returns:
        The full checkpoint name anchored under the checkpoint subtree root.

    Raises:
        ValueError: If a leaf mapping does not preserve the logical root.
    """
    actual_relative = _strip_name_prefix(
        single_space_name, aoa_name_scope.actual_model_prefix
    )
    logical_name = join_name(
        aoa_name_scope.logical_model_prefix, actual_relative
    )
    mapped_name = _resolve_checkpoint_name(
        logical_name,
        checkpoint_name_prefix,
        checkpoint_name_mapping,
        model_name_prefix,
    )
    # The logical root as it appears in checkpoint space.
    # checkpoint_name_mapping only contains leaf-level templates and never
    # rewrites the root prefix (e.g. "layers.$LAYER_ID" stays as-is in values,
    # under the checkpoint root), so the mapped result always starts with it.
    logical_checkpoint_root = join_name(
        checkpoint_name_prefix,
        _strip_name_prefix(
            aoa_name_scope.logical_model_prefix, model_name_prefix
        ),
    )
    try:
        checkpoint_local = _strip_name_prefix(
            mapped_name, logical_checkpoint_root
        )
    except ValueError:
        raise ValueError(
            f"checkpoint_name_mapping does not preserve the logical root: "
            f"mapped_name={mapped_name!r} does not start with "
            f"logical_checkpoint_root={logical_checkpoint_root!r}. "
            f"Scoped resolution requires leaf mapping values to keep both the "
            f"checkpoint root prefix and the logical_model_prefix structure "
            f"(single_space_name={single_space_name!r}, "
            f"logical_model_prefix={aoa_name_scope.logical_model_prefix!r})."
        ) from None
    if aoa_name_scope.is_checkpoint_prefix_absolute:
        # The scope prefix is already a full checkpoint prefix; the shared
        # checkpoint prefix is deliberately not prepended (an MTP boundary
        # whose own params, or an output head, sit at the checkpoint root).
        return join_name(aoa_name_scope.checkpoint_prefix, checkpoint_local)
    return join_name(
        checkpoint_name_prefix,
        join_name(aoa_name_scope.checkpoint_prefix, checkpoint_local),
    )


def _resolve_scoped_names(
    local_name: str,
    checkpoint_name_prefix: str,
    structured_name_prefix: str,
    pp_to_single_mapping: Mapping[str, str],
    checkpoint_name_mapping: Mapping[str, str],
    aoa_name_scope: AOANameScope,
    model_name_prefix: str,
) -> tuple[str, str]:
    """Re-rooted-subtree variant of ``resolve_names``.

    The real model key still resolves through the live structured prefix and
    ``pp_to_single_mapping``; only the checkpoint side is routed through the
    logical normal-layer root.

    Args:
        local_name: Tensor name local to the current layer.
        checkpoint_name_prefix: Checkpoint root prefix.
        structured_name_prefix: Pre-mapping live module path accumulated from
            ancestors, ending in ``.`` when non-empty.
        pp_to_single_mapping: Structured name -> single name mapping.
        checkpoint_name_mapping: Absolute model name template -> absolute
            checkpoint name template.
        aoa_name_scope: Subtree path scope.
        model_name_prefix: Model root prefix.

    Returns:
        A ``(checkpoint_name, single_name)`` pair.
    """
    single_name = resolve_single_name(
        local_name,
        structured_name_prefix,
        pp_to_single_mapping,
        model_name_prefix,
    )
    checkpoint_name = _scoped_checkpoint_name(
        single_name,
        checkpoint_name_prefix,
        checkpoint_name_mapping,
        aoa_name_scope,
        model_name_prefix,
    )
    return checkpoint_name, single_name


def resolve_single_name(
    local_name: str,
    structured_name_prefix: str,
    pp_to_single_mapping: Mapping[str, str],
    model_name_prefix: str,
) -> str:
    """Resolves a real model tensor's structured name to its single name.

    Args:
        local_name: Tensor name local to the current layer.
        structured_name_prefix: Live module-tree prefix accumulated from
            ancestors. Like ``Layer.state_dict`` and ``Layer.sharded_state_dict``
            prefixes, a non-empty value ends in ``.``.
        pp_to_single_mapping: Structured name -> single name mapping. When non-empty the
            pre-mapping structured name must hit exactly (no fallback). When empty, only
            names starting with ``model_name_prefix`` pass through as identity.
        model_name_prefix: Model root prefix for the identity fallback.

    Returns:
        The single name for the tensor.

    Raises:
        KeyError: If ``pp_to_single_mapping`` is non-empty and the structured name misses, or
            ``pp_to_single_mapping`` is empty and the name doesn't start with model_name_prefix.
    """
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
    aoa_name_scope: AOANameScope | None = None,
) -> tuple[str, str]:
    """Resolves a real model tensor to its ``(checkpoint_name, single_name)`` pair.

    Args:
        local_name: Tensor name local to the current layer.
        checkpoint_name_prefix: Checkpoint root prefix shared by the generation
            pass.
        structured_name_prefix: Pre-mapping live module path accumulated from
            ancestors, ending in ``.`` when non-empty.
        pp_to_single_mapping: Structured name -> single name mapping.
        checkpoint_name_mapping: Absolute model name template -> absolute
            checkpoint name template.
        model_name_prefix: Model root prefix.
        aoa_name_scope: Optional re-rooted subtree scope; when set the
            checkpoint side is routed through the logical normal-layer root.

    Returns:
        A stable ``(checkpoint_name, single_name)`` pair. Both directions
        independently resolve this pair and choose their emission order.
    """
    if aoa_name_scope is not None:
        return _resolve_scoped_names(
            local_name,
            checkpoint_name_prefix,
            structured_name_prefix,
            pp_to_single_mapping,
            checkpoint_name_mapping,
            aoa_name_scope,
            model_name_prefix,
        )

    # Pre-mapping live structured name -> canonical model name.
    single_name = resolve_single_name(
        local_name,
        structured_name_prefix,
        pp_to_single_mapping,
        model_name_prefix,
    )

    # single name -> checkpoint name
    checkpoint_name = _resolve_checkpoint_name(
        single_name,
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
    aoa_name_scope: AOANameScope | None = None,
) -> str:
    """Builds a checkpoint-only name (Q/K/V, gate/up, fused alpha) from an anchor.

    Strips ``anchor_local_name`` off the resolved model target to get the
    enclosing single-name scope, appends the checkpoint-only local name, then
    maps to the checkpoint side. Checkpoint-only names are never sent through
    ``pp_to_single_mapping``.

    Args:
        anchor_single_name: Resolved single name of a real anchor tensor.
        anchor_local_name: Local name of that anchor, stripped to reach its
            scope.
        checkpoint_local_name: Checkpoint-only local name to place inside the
            anchor's scope.
        checkpoint_name_prefix: Checkpoint root prefix.
        checkpoint_name_mapping: Absolute model name template -> absolute
            checkpoint name template.
        model_name_prefix: Model root prefix.
        aoa_name_scope: Optional re-rooted subtree scope.

    Returns:
        The full checkpoint name for the synthetic checkpoint-only tensor.
    """
    scope_single = strip_name_suffix(anchor_single_name, anchor_local_name)
    synthetic_single = join_name(scope_single, checkpoint_local_name)
    if aoa_name_scope is not None:
        return _scoped_checkpoint_name(
            synthetic_single,
            checkpoint_name_prefix,
            checkpoint_name_mapping,
            aoa_name_scope,
            model_name_prefix,
        )
    return _resolve_checkpoint_name(
        synthetic_single,
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

    Args:
        checkpoint_name_mapping: Absolute model name template -> checkpoint name
            template mapping to validate.
        model_name_prefix: Model root prefix every key must sit under. For a
            multi-tower model this is the current tower's root, since the
            generation side builds one context per tower.

    Raises:
        ValueError: On the first violation of any rule.
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
