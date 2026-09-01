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
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = []


# Placeholder segments captured by root-relative templates. They only ever appear
# in templates / final AOA macros, never as a ``pp_to_single_mapping`` query key.
ID_PLACEHOLDERS = frozenset({"$LAYER_ID", "$EXPERT_ID"})


class DtypeCastRule(TypedDict):
    """One model-declared dtype cast: the checkpoint stores the tensor as
    ``checkpoint_dtype`` while the live model holds ``model_dtype``."""

    checkpoint_dtype: str
    model_dtype: str


_DTYPE_CAST_RULE_KEYS = frozenset(DtypeCastRule.__annotations__)


@dataclass(frozen=True)
class AOAContext:
    """Frozen, read-only recursion context for AOA statement generation.

    Every field is constant across one whole-model generation pass. The
    whole-model entry builds one context (per tower for multi-tower models) and
    forwards it unchanged down the module tree. Per-position path info
    (``structured_name_prefix`` and the optional ``AOANameScope``) travels
    alongside ``ctx`` rather than inside it. Maps are ready on construction:
    empty rules are an empty ``dict``, never ``None``.
    """

    config: object
    """Config of the sub-structure being generated: the whole-model config for a
    single-tower model, the current tower's sub-config for a multi-tower one."""

    checkpoint_name_prefix: str
    """Checkpoint-side prefix shared by every layer in this generation pass."""

    checkpoint_name_mapping: Mapping[str, str]
    """Model root-relative template -> checkpoint root-relative mapping."""

    dtype_cast_rules: Mapping[str, DtypeCastRule]
    """Model root-relative template -> dtype cast rule."""

    pp_to_single_mapping: Mapping[str, str]
    """Structured name -> single name mapping."""

    model_name_prefix: str
    """Single-name model root prefix, stripped by the checkpoint resolvers
    before template matching. Declared by the model, so a multi-tower model can
    root a tower elsewhere (e.g. ``"model.language_model"``)."""

    excluded_names: frozenset[str] = frozenset()
    """Structured names the module-tree recursion must not emit.

    Tied / shared tensors registered under several structured names would
    otherwise produce duplicate or conflicting statements, so the whole-model
    entry resolves them before recursion, keeps one producer and lists the rest
    here, re-emitting them itself (fan-out in the checkpoint->model direction,
    ``-> _`` deletion in the inverse one). Honoring this set is part of the
    recursion contract: a component override must skip an own-parameter whose
    ``structured_name_prefix + name`` is in here, exactly as the default
    ``Layer`` implementation does."""


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
# ``resolve_names``, ``resolve_checkpoint_name_from_anchor``,
# ``resolve_dtype_cast_rule``, ``format_dtype_cast_attr``,
# ``format_inv_dtype_cast_attr``, ``should_skip``, ``strip_name_suffix`` and
# ``validate_checkpoint_name_mapping``; anything underscore-prefixed is internal
# detail. The only non-trivial logic is root-relative template matching
# (``_match_template`` / ``_render_template``). A re-rooted subtree passes an
# ``AOANameScope`` so its checkpoint side is routed through the logical
# normal-layer root (``_resolve_scoped_names``).
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


def _match_checkpoint_relative(
    model_relative_name: str, checkpoint_name_mapping: Mapping[str, str]
) -> str:
    """Maps a root-stripped model relative name to its checkpoint relative name.

    Args:
        model_relative_name: Model name with the ``model`` root already stripped.
        checkpoint_name_mapping: Model root-relative template -> checkpoint root-relative
            template.

    Returns:
        The rendered checkpoint relative name on exactly one hit, or
        ``model_relative_name`` unchanged (identity) when no template matches.

    Raises:
        ValueError: If more than one template matches (ambiguous config).
    """
    hits = []
    for key_template, value_template in checkpoint_name_mapping.items():
        captures = _match_template(key_template, model_relative_name)
        if captures is not None:
            hits.append(_render_template(value_template, captures))
    if len(hits) > 1:
        raise ValueError(
            f"ambiguous checkpoint name mapping for {model_relative_name!r}: {hits}"
        )
    return hits[0] if hits else model_relative_name


def _resolve_checkpoint_relative(
    name: str,
    checkpoint_name_mapping: Mapping[str, str],
    model_name_prefix: str,
) -> str:
    """Strips the model root, then template-maps to a checkpoint relative name.

    Args:
        name: Single-space model name (starting at the model root prefix).
        checkpoint_name_mapping: Model root-relative template -> checkpoint root-relative
            template.
        model_name_prefix: Model root prefix to strip before template matching.

    Returns:
        The checkpoint relative name (root-relative, without the checkpoint prefix).
    """
    model_relative_name = _strip_name_prefix(name, model_name_prefix)
    return _match_checkpoint_relative(
        model_relative_name, checkpoint_name_mapping
    )


def _resolve_checkpoint_name(
    single_name: str,
    checkpoint_name_prefix: str,
    checkpoint_name_mapping: Mapping[str, str],
    model_name_prefix: str,
) -> str:
    """Maps a single name to a full checkpoint name.

    Args:
        single_name: Single-space model name (starting at the model root prefix).
        checkpoint_name_prefix: Checkpoint-side prefix to prepend to the mapped relative
            name.
        checkpoint_name_mapping: Model root-relative template -> checkpoint root-relative
            template.
        model_name_prefix: Model root prefix to strip before template matching.

    Returns:
        The full checkpoint name (``checkpoint_name_prefix`` joined with the mapped
        relative name).
    """
    return join_name(
        checkpoint_name_prefix,
        _resolve_checkpoint_relative(
            single_name, checkpoint_name_mapping, model_name_prefix
        ),
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
    value (``layers.$LAYER_ID.x -> layers.$LAYER_ID.y``), which is what makes the
    logical root strippable from the mapped result.

    Args:
        single_space_name: Single-space model name under the real subtree root.
        checkpoint_name_prefix: Checkpoint-side prefix to prepend.
        checkpoint_name_mapping: Model root-relative template -> checkpoint
            root-relative template.
        aoa_name_scope: Subtree path scope (real / logical / checkpoint roots).
        model_name_prefix: Model root prefix to strip before template matching.

    Returns:
        The full checkpoint name anchored under the checkpoint subtree root.

    Raises:
        ValueError: If a leaf mapping rewrites the logical root prefix.
    """
    actual_relative = _strip_name_prefix(
        single_space_name, aoa_name_scope.actual_model_prefix
    )
    logical_name = join_name(
        aoa_name_scope.logical_model_prefix, actual_relative
    )
    mapped_name = _resolve_checkpoint_relative(
        logical_name, checkpoint_name_mapping, model_name_prefix
    )
    # Strip the logical_model_prefix's model-relative form directly.
    # checkpoint_name_mapping only contains leaf-level templates and never
    # rewrites the root prefix (e.g. "layers.$LAYER_ID" stays as-is in
    # values), so the mapped result always starts with logical_model_relative.
    logical_model_relative = _strip_name_prefix(
        aoa_name_scope.logical_model_prefix, model_name_prefix
    )
    try:
        checkpoint_local = _strip_name_prefix(
            mapped_name, logical_model_relative
        )
    except ValueError:
        raise ValueError(
            f"checkpoint_name_mapping rewrites the logical root prefix: "
            f"mapped_name={mapped_name!r} does not start with "
            f"logical_model_relative={logical_model_relative!r}. "
            f"Scoped resolution requires that leaf mappings preserve "
            f"the logical_model_prefix prefix structure in their values "
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
        checkpoint_name_prefix: Checkpoint-side prefix.
        structured_name_prefix: Pre-mapping live module path accumulated from
            ancestors, ending in ``.`` when non-empty.
        pp_to_single_mapping: Structured name -> single name mapping.
        checkpoint_name_mapping: Model root-relative template -> checkpoint
            root-relative template.
        aoa_name_scope: Subtree path scope.
        model_name_prefix: Model root prefix to strip before template matching.

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
        checkpoint_name_prefix: Fixed checkpoint-side prefix for the generation
            pass.
        structured_name_prefix: Pre-mapping live module path accumulated from
            ancestors, ending in ``.`` when non-empty.
        pp_to_single_mapping: Structured name -> single name mapping.
        checkpoint_name_mapping: Model root-relative template -> checkpoint
            root-relative template.
        model_name_prefix: Model root prefix to strip before template matching.
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
        checkpoint_name_prefix: Checkpoint-side prefix.
        checkpoint_name_mapping: Model root-relative template -> checkpoint
            root-relative template.
        model_name_prefix: Model root prefix to strip before template matching.
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


def resolve_dtype_cast_rule(
    single_name: str,
    dtype_rules: Mapping[str, DtypeCastRule],
    model_name_prefix: str,
) -> DtypeCastRule | None:
    """Looks up a dtype cast rule by the single name's root-relative template.

    Args:
        single_name: Single-space model name (starting at the model root prefix).
        dtype_rules: Model root-relative template -> dtype cast rule.
        model_name_prefix: Model root prefix to strip before template matching.

    Returns:
        The matching rule on a single hit, or ``None`` on a miss (including an
        empty ``dtype_rules``).

    Raises:
        ValueError: If more than one template matches, or if the matched rule
            does not declare both dtype endpoints.
    """
    if not dtype_rules:
        return None
    model_relative_name = _strip_name_prefix(single_name, model_name_prefix)
    hits = [
        (key_template, rule)
        for key_template, rule in dtype_rules.items()
        if _match_template(key_template, model_relative_name) is not None
    ]
    if len(hits) > 1:
        raise ValueError(
            f"ambiguous dtype cast rule for {model_relative_name!r}: matched "
            f"templates {[key for key, _ in hits]}, rules {[r for _, r in hits]}"
        )
    if not hits:
        return None
    key_template, rule = hits[0]
    missing = _DTYPE_CAST_RULE_KEYS - rule.keys()
    if missing:
        raise ValueError(
            f"dtype cast rule {key_template!r} (matched by "
            f"{model_relative_name!r}) is missing {sorted(missing)}; a rule "
            f"must declare {sorted(_DTYPE_CAST_RULE_KEYS)}, got {sorted(rule)}"
        )
    return rule


def format_dtype_cast_attr(rule: DtypeCastRule | None) -> str:
    """Formats the checkpoint->model dtype-cast attribute suffix.

    Args:
        rule: A rule from :func:`resolve_dtype_cast_rule`, or ``None`` on a miss.

    Returns:
        The ``", src_dtype=..., dst_dtype=..."`` suffix to append to a
        single-input single-output statement, casting ``checkpoint_dtype ->
        model_dtype``; ``""`` when no cast is needed (no rule, or both
        endpoints equal).
    """
    if rule is None or rule["checkpoint_dtype"] == rule["model_dtype"]:
        return ""
    return (
        f", src_dtype='{rule['checkpoint_dtype']}'"
        f", dst_dtype='{rule['model_dtype']}'"
    )


def format_inv_dtype_cast_attr(rule: DtypeCastRule | None) -> str:
    """Formats the inverse (model -> checkpoint) dtype-cast attribute suffix.

    Mirror of :func:`format_dtype_cast_attr` with the endpoints swapped: casts
    ``model_dtype -> checkpoint_dtype``.

    Args:
        rule: A rule from :func:`resolve_dtype_cast_rule`, or ``None`` on a miss.

    Returns:
        The attribute suffix string, or ``""`` when no cast applies.
    """
    if rule is None or rule["checkpoint_dtype"] == rule["model_dtype"]:
        return ""
    return (
        f", src_dtype='{rule['model_dtype']}'"
        f", dst_dtype='{rule['checkpoint_dtype']}'"
    )


def should_skip(source_name: str, target_name: str, cast: str) -> bool:
    """Reports whether a component may omit the AOA statement for a param.

    ``AOAEngine`` fills any destination key that no statement produced from the
    same-named source, so a ``foo -> foo`` line with no cast is redundant and
    generators rely on that passthrough instead. Any real transform (rename,
    ``^T``, fusion / split, dtype cast) makes the two names differ or sets a
    non-empty ``cast``, so it is never skipped.

    Args:
        source_name: The resolved source-side name.
        target_name: The resolved target-side name.
        cast: The dtype-cast attribute suffix (``""`` when no cast applies).

    Returns:
        ``True`` when the statement is redundant and can be omitted.
    """
    return source_name == target_name and not cast


def validate_checkpoint_name_mapping(
    checkpoint_name_mapping: Mapping[str, str],
) -> None:
    """Validates ``checkpoint_name_mapping`` once, run from model ``__init__``.

    Enforces three rules:

    - a ``$``-bearing segment must be a whole segment and a known placeholder,
      otherwise it would survive rendering into a bogus checkpoint name;
    - every placeholder used in a value template is captured by its key template
      (otherwise the value cannot be rendered);
    - no two keys map to the same checkpoint target (value template). Tied /
      shared aliases must be expressed by the whole-model alias handler, never
      implied by a duplicate value here.

    Args:
        checkpoint_name_mapping: Model root-relative template -> checkpoint
            root-relative template mapping to validate.

    Raises:
        ValueError: On the first violation of any rule.
    """
    seen_targets: dict[str, str] = {}
    for key_template, value_template in checkpoint_name_mapping.items():
        if not key_template or not value_template:
            raise ValueError(
                f"checkpoint name mapping contains an empty-string "
                f"key or value: {key_template!r} -> {value_template!r}"
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
        if value_template in seen_targets:
            raise ValueError(
                f"duplicate checkpoint target {value_template!r} from keys "
                f"{seen_targets[value_template]!r} and {key_template!r}; "
                f"tied/shared aliases must go through the whole-model alias "
                f"handler, not a duplicate mapping value"
            )
        seen_targets[value_template] = key_template

    # Detect simple overlaps: a concrete key (no placeholders) that is also
    # matchable by another key with placeholders would cause ambiguous hits at
    # runtime. This is a best-effort static check covering the most common
    # mistake (a specific layer override coexisting with its general template).
    concrete_keys = [
        k
        for k in checkpoint_name_mapping
        if not any(seg in ID_PLACEHOLDERS for seg in k.split("."))
    ]
    templated_keys = [
        k
        for k in checkpoint_name_mapping
        if any(seg in ID_PLACEHOLDERS for seg in k.split("."))
    ]
    for concrete in concrete_keys:
        for tmpl in templated_keys:
            if _match_template(tmpl, concrete) is not None:
                raise ValueError(
                    f"overlapping checkpoint name mapping: concrete key "
                    f"{concrete!r} is also matchable by templated key "
                    f"{tmpl!r}; this would cause ambiguous hits at runtime"
                )
