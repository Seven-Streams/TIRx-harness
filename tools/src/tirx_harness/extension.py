"""Author, lower and verify IR extensions for TIRx-lite kernels.

tvm-tirx treats compiler features as plugins: a package may register its own IR
types, operators and region operators under its own namespace, and they are
written in a kernel exactly like built-in ones. Nothing downstream knows them,
however: the code generator, NumSim, Synccheck and Racecheck accept only the
core IR (``ir.*``, ``prim.*`` and ``tirx.*``). An extension therefore comes
with a lowering pass that rewrites its constructs into the TIRx-lite subset.

This module is the harness-side workflow around that facility:

1. :class:`Namespace` registers the extension's operators, region operators and
   types once per process and publishes their Python builders, which are
   called inside ``@txl.kernel`` bodies like ``txl`` builders.
2. The author writes the lowering as passes (``Function -> Function``), for
   example with ``tvm_ffi.structural_map``. :func:`emit` macro-expands a
   replacement statement sequence with the ordinary TIRx-lite builders.
3. :func:`lower` applies the passes to a traced kernel, rejects any remaining
   extension construct (:class:`ExtensionResidueError`) and re-runs the
   TIRx-lite low-level IR contract on the result. The returned ``txl.Kernel`` is
   the one the checkers (``.func``), ``.compile()`` and the benchmark consume.

The unlowered kernel is neither checkable nor compilable; lower first.
"""

from __future__ import annotations

import keyword
import sys
import types
import warnings
from collections.abc import Callable, Iterable, Sequence
from contextlib import contextmanager
from typing import Any, NamedTuple

import tvm_ffi

import tvm_tirx
from tvm_tirx import ir, tirx

#: Operator-name and type-key roots of the core IR. Everything else is an extension.
CORE_PREFIXES: tuple[str, ...] = ("ir", "prim", "tirx")

#: The result type of a statement-like operator. A call of this type is emitted as a
#: statement by the builders; ``ir.NoneType()`` is not void and would drop the call.
VOID = ir.PrimType("")

_NAMESPACE_MODULE_ROOT = f"{__name__}._ns"
_IDENTIFIER_OK = str.isidentifier


class Residue(NamedTuple):
    """One extension construct found in IR that should contain only core IR."""

    kind: str
    """``"call"``, ``"region"`` or ``"type"``."""
    name: str
    """The operator name or the registered type key."""
    loc: str | None
    """``file:line`` of the construct when the IR carries a source location."""

    def __str__(self) -> str:
        where = f" at {self.loc}" if self.loc else ""
        return f"{self.kind} {self.name}{where}"


class ExtensionResidueError(ValueError):
    """Raised when extension constructs remain after lowering."""

    def __init__(self, residues: Sequence[Residue], subject: str = "the kernel"):
        self.residues = tuple(residues)
        lines = [
            f"{len(self.residues)} extension construct(s) remain in {subject}:",
            *(f"  - {residue}" for residue in self.residues),
            "Lower them with the extension's pass (tirx_harness.extension.lower) before "
            "synccheck, racecheck, NumSim, compile or the benchmark consume the kernel.",
        ]
        super().__init__("\n".join(lines))


def is_core_name(name: str) -> bool:
    """Whether an operator name or type key belongs to the core IR."""
    return name.partition(".")[0] in CORE_PREFIXES


def type_key(obj: Any) -> str | None:
    """The registered type key of an FFI object, or None for unregistered classes."""
    info = tvm_ffi.core._type_cls_to_type_info(type(obj))
    return None if info is None else info.type_key


def _source_loc(loc: Any) -> str | None:
    """``file:line`` for a SourceLoc, following call sites; None for unknown locations."""
    if loc is None or isinstance(loc, ir.UnknownLoc):
        return None
    if isinstance(loc, ir.CallSiteLoc):
        return _source_loc(loc.callee) or _source_loc(loc.caller)
    source_name = getattr(loc, "source_name", None)
    source = getattr(source_name, "name", None)
    line = getattr(loc, "start_line", None)
    if source is not None and line is not None:
        return f"{source}:{line}"
    return str(loc)


def _function_of(value: Any) -> tirx.Function:
    func = getattr(value, "func", value)
    if not isinstance(func, tirx.Function):
        raise TypeError(
            f"expected a txl.Kernel or a tvm_tirx.tirx.Function, got {type(value).__name__}"
        )
    return func


def residue(value: Any) -> list[Residue]:
    """List the extension calls, regions and types in a kernel, function or statement.

    ``value`` may be a ``txl.Kernel``, a ``Function``, an ``IRModule`` or any IR node.
    """
    root = value.func if hasattr(value, "func") else value
    found: list[Residue] = []

    def visit_call(node, visitor):
        if isinstance(node.op, ir.Op) and not is_core_name(node.op.name):
            found.append(Residue("call", node.op.name, _source_loc(node.loc)))
        visitor.default_visit(node)

    def visit_region(node, visitor):
        if not is_core_name(node.op.name):
            found.append(Residue("region", node.op.name, _source_loc(node.loc)))
        visitor.default_visit(node)

    def visit_type(node, visitor):
        key = type_key(node)
        if key is not None and not is_core_name(key):
            found.append(Residue("type", key, _source_loc(getattr(node, "loc", None))))
        visitor.default_visit(node)

    tvm_ffi.structural_visit(
        root,
        [(ir.Call, visit_call), (ir.RegionStmt, visit_region), (ir.Type, visit_type)],
    )
    # Report each construct once, in traversal order.
    unique: dict[Residue, None] = dict.fromkeys(found)
    return list(unique)


def verify_lowered(value: Any, *, allowed_func_calls: Iterable[str] = ()):
    """Require core-only IR that satisfies the TIRx-lite low-level contract.

    Raises :class:`ExtensionResidueError` when an extension construct remains and
    ``LowLevelIRContractError`` when the lowered function breaks the contract.
    Returns the contract report otherwise.
    """
    func = _function_of(value)
    residues = residue(func)
    if residues:
        name = getattr(value, "name", None) or func.attrs.get("global_symbol", "the kernel")
        raise ExtensionResidueError(residues, subject=f"kernel {name!r}")
    from tirx_kernels.tirx_lite.low_level_ir import check_low_level_ir

    return check_low_level_ir(func, allowed_func_calls=tuple(allowed_func_calls))


Pass = Callable[[tirx.Function], tirx.Function]


def _apply_pass(pass_, func: tirx.Function) -> tirx.Function:
    if isinstance(pass_, ir.transform.Pass):
        module = pass_(tvm_tirx.IRModule({"main": func}))
        return module["main"]
    if not callable(pass_):
        raise TypeError(f"a lowering pass must be callable, got {type(pass_).__name__}")
    result = pass_(func)
    if not isinstance(result, tirx.Function):
        raise TypeError(
            f"lowering pass {getattr(pass_, '__name__', pass_)!r} must return the rewritten "
            f"tvm_tirx.tirx.Function, got {type(result).__name__}"
        )
    return result


@contextmanager
def _active_session(session):
    """Make ``session`` the current ``txl`` kernel session, restoring the previous one."""
    from tirx_kernels.tirx_lite import entry

    previous = getattr(entry._TLS, "session", None)
    entry._TLS.session = session
    try:
        yield
    finally:
        entry._TLS.session = previous


def _flatten(passes) -> list:
    flat = []
    for item in passes:
        if isinstance(item, list | tuple):
            flat.extend(_flatten(item))
        else:
            flat.append(item)
    return flat


def lower(kernel, *passes, allowed_func_calls: Iterable[str] = ()):
    """Lower a traced kernel to core TIRx-lite IR and verify the result.

    ``kernel`` is the ``txl.Kernel`` returned by ``@txl.kernel`` (or a bare
    ``Function``). Each pass is a callable ``Function -> Function`` or a
    ``tvm_tirx`` pass; they run in order with the kernel's ``txl`` session active,
    so :func:`emit` and the ``txl`` coordinate helpers work inside them. The
    result is a new ``txl.Kernel`` sharing the launch metadata (or a ``Function``
    for a bare function). It must contain no extension construct and must satisfy
    the TIRx-lite low-level IR contract; ``allowed_func_calls`` is forwarded to
    that contract check exactly as ``@txl.kernel`` accepts it.
    """
    func = _function_of(kernel)
    session = getattr(kernel, "session", None)
    with _active_session(session):
        for pass_ in _flatten(passes):
            func = _apply_pass(pass_, func)
    verify_lowered(func, allowed_func_calls=allowed_func_calls)
    if session is None:
        return func
    from tirx_kernels.tirx_lite.entry import Kernel

    return Kernel(func, session)


class Emitted:
    """The statements produced by one :func:`emit` block, available after the block exits."""

    def __init__(self) -> None:
        self.stmt: ir.Stmt | None = None


@contextmanager
def emit(kernel=None, *, loc: ir.Location | None = None):
    """Build replacement statements with TIRx-lite builders outside a kernel body.

    Use it inside a lowering pass::

        with extension.emit(loc=stmt.loc) as block:
            tmp = txl.local_scalar("float32", init=value * scale)
            txl.ptx.st.global_.f32(pointer, tmp)
        return block.stmt

    The block runs under a fresh IRBuilder, so ``txl.ptx``/``txl.cuda``
    instructions, ``txl.local_scalar``, ``txl.serial`` and ``txl.if_`` emit into
    it; ``block.stmt`` holds the result once the block exits. The ``txl``
    coordinate helpers (``txl.lane_id()`` and friends) resolve against the kernel
    whose session is active: the kernel being lowered by :func:`lower`, or the
    ``kernel`` argument. Pass the location of the construct being lowered as
    ``loc``; the emitted statements then carry it, so checker findings point at
    the kernel line that used the abstraction. Shared-memory allocation is not
    available here; pass pointers to the extension's operators as operands.
    """
    from contextlib import nullcontext

    from tirx_kernels.tirx_lite import entry
    from tvm_tirx.script.ir_builder import IRBuilder
    from tvm_tirx.tirx.script import ir_builder as builder

    session = kernel.session if kernel is not None else entry.current(required=False)
    block = Emitted()
    with _active_session(session):
        with IRBuilder() as ib:
            located = (
                nullcontext() if loc is None or isinstance(loc, ir.UnknownLoc) else ib.with_loc(loc)
            )
            with builder.function_(), located:
                builder.func_name_("_tirx_harness_extension_emit")
                yield block
        block.stmt = ib.get().body


def _normalize_args(args) -> list[tuple[str, str]]:
    normalized = []
    for item in args:
        if isinstance(item, str):
            normalized.append((item, f"The {item} operand."))
        else:
            name, doc = item
            normalized.append((str(name), str(doc)))
    return normalized


def _effect_value(effect) -> int:
    if isinstance(effect, str):
        effect = ir.CallEffectKind[effect]
    return int(effect)


def _is_void(ty) -> bool:
    return isinstance(ty, ir.PrimType) and str(ty.dtype) == ""


class Namespace:
    """The operators, region operators and types of one extension namespace.

    ``Namespace("my_ext")`` publishes its builders as attributes: after
    ``ns.op("fma3", ...)``, ``ns.fma3(a, b, c)`` constructs the call inside a
    ``@txl.kernel`` body (a void operator is emitted as a statement, a valued one
    is an expression); after ``ns.region_op("repeat", ...)``,
    ``with ns.repeat(n) as i:`` opens the region. Registrations are
    process-global and happen once; defining a name again replaces its
    signature and attributes with a warning.
    """

    def __init__(self, prefix: str):
        if not _IDENTIFIER_OK(prefix) or keyword.iskeyword(prefix):
            raise ValueError(f"extension namespace must be a Python identifier, got {prefix!r}")
        if prefix in CORE_PREFIXES:
            raise ValueError(f"{prefix!r} is a core IR namespace; choose your own")
        self.prefix = prefix
        module_name = f"{_NAMESPACE_MODULE_ROOT}.{prefix}"
        module = sys.modules.get(module_name)
        if module is None:
            module = types.ModuleType(module_name, f"Builders of the {prefix!r} IR extension.")
            module.__all__ = []
            sys.modules[module_name] = module
        self.module = module
        from tvm_tirx.ir._expose_api import _init_exposed_op_api, _init_exposed_type_api
        from tvm_tirx.script.ir_builder.base import auto_emit_void_call

        _init_exposed_op_api(prefix, module_name, wrapper=auto_emit_void_call)
        _init_exposed_type_api(prefix, module_name)

    def __repr__(self) -> str:
        return f"<extension.Namespace {self.prefix!r} ops={sorted(self.ops)}>"

    def __getattr__(self, name: str):
        # Only called when normal lookup fails: forward to the published builders.
        if name.startswith("__"):
            raise AttributeError(name)
        try:
            return getattr(self.module, name)
        except AttributeError:
            raise AttributeError(
                f"{self.prefix!r} has no builder {name!r}; define it with op(), "
                "region_op() or type()"
            ) from None

    @property
    def ops(self) -> dict[str, ir.Op]:
        """Registered operators of this namespace, keyed by their name without the prefix."""
        head = self.prefix + "."
        return {
            name[len(head) :]: ir.Op.get(name)
            for name in ir.Op.list_op_names()
            if name.startswith(head)
        }

    def _qualify(self, name: str) -> str:
        parts = name.split(".")
        if not all(_IDENTIFIER_OK(part) and not keyword.iskeyword(part) for part in parts):
            raise ValueError(f"operator name must be dotted Python identifiers, got {name!r}")
        return f"{self.prefix}.{name}"

    def _register(self, full_name: str, doc: str) -> tuple[ir.Op, bool]:
        existing = full_name in ir.Op.list_op_names()
        if existing:
            warnings.warn(
                f"{full_name} is already registered; its signature and attributes are replaced",
                stacklevel=3,
            )
        else:
            tvm_ffi.get_global_func("ir.RegisterOp")(full_name, doc)
        return ir.Op.get(full_name), existing

    def op(
        self,
        name: str,
        doc: str,
        args: Sequence[str | tuple[str, str]] = (),
        *,
        ret_type: ir.Type | str | None = None,
        infer_type: Callable[[ir.Call], ir.Type] | None = None,
        effect: ir.CallEffectKind | str | None = None,
    ) -> ir.Op:
        """Register ``<prefix>.<name>`` and publish its builder.

        ``args`` names the operands (optionally with a doc each); the builder binds
        them positionally or by keyword. Give the result type either as a fixed
        ``ret_type`` (a ``Type``, a dtype string, or ``"void"``/:data:`VOID` for a
        statement) or as ``infer_type(call) -> Type`` computed from the operand
        types. ``effect`` defaults to ``UpdateState`` for a void operator and
        ``Pure`` otherwise; pass ``ir.CallEffectKind.ReadState`` for a load-like
        operator so it is not reordered across stores.
        """
        if (ret_type is None) == (infer_type is None):
            raise TypeError("give exactly one of ret_type or infer_type")
        if isinstance(ret_type, str):
            ret_type = VOID if ret_type == "void" else ir.PrimType(ret_type)
        full_name = self._qualify(name)
        op, existing = self._register(full_name, doc)
        op.set_signature(args=_normalize_args(args))
        if infer_type is not None:
            op.set_attr("FInferType", infer_type, override=existing)
        else:
            op.set_attr("TFixedReturnType", ret_type, override=existing)
        if effect is None:
            effect = (
                ir.CallEffectKind.UpdateState
                if ret_type is not None and _is_void(ret_type)
                else ir.CallEffectKind.Pure
            )
        op.set_attr("TCallEffectKind", _effect_value(effect), override=existing)
        op.expose()
        return op

    def region_op(
        self,
        name: str,
        doc: str,
        args: Sequence[str | tuple[str, str]] = (),
        *,
        body_params: Sequence[tuple[str, str]] | Callable[[ir.Call], Sequence[ir.Var]] = (),
        validate: Callable[[ir.RegionStmt], None] | None = None,
    ) -> ir.Op:
        """Register a region operator: a statement with operands and one body.

        ``body_params`` declares the variables the body receives, as
        ``[(name, dtype), ...]`` (fresh variables per use) or as a hook
        ``(call) -> [Var, ...]`` reading the operands. In a kernel the region is
        written ``with ns.<name>(operands...) as params:``; ``params`` is the single
        variable or the sequence of variables. The lowering replaces the
        ``RegionStmt`` with core statements that bind ``region.body_params`` and
        contain ``region.body``. ``validate(region)`` may reject a complete region.
        """
        full_name = self._qualify(name)
        op, existing = self._register(full_name, doc)
        op.set_signature(args=_normalize_args(args))
        if callable(body_params):
            hook = body_params
        else:
            declared = [(str(var_name), dtype) for var_name, dtype in body_params]

            def hook(call):
                return [ir.Var(var_name, dtype) for var_name, dtype in declared]

        op.set_attr("FRegionGetBodyParams", hook, override=existing)
        if validate is not None:
            op.set_attr("FRegionValidate", validate, override=existing)
        op.expose()
        return op

    def type(self, cls=None, *, name: str | None = None):
        """Register a Python-defined IR type under this namespace (decorator).

        The class derives from ``tvm_tirx.ir.Type`` (or another type) and declares
        its fields as annotations; it gets a constructor taking them in order and
        takes part in printing, structural equality and serialization. The type is
        published under the class name without a ``Type`` suffix, so
        ``PairType`` is built as ``ns.Pair(dtype)``. Register each type once per
        process.
        """
        from tvm_ffi.dataclasses import py_class

        def register(klass):
            key = f"{self.prefix}.{name or klass.__name__}"
            registered = py_class(key)(klass)
            tvm_ffi.get_global_func("ir.expose_type")(key)
            return registered

        return register if cls is None else register(cls)


__all__ = [
    "CORE_PREFIXES",
    "VOID",
    "Emitted",
    "ExtensionResidueError",
    "Namespace",
    "Pass",
    "Residue",
    "emit",
    "is_core_name",
    "lower",
    "residue",
    "type_key",
    "verify_lowered",
]
