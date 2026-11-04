# IR extensions for TIRx-lite kernels

TIRx is built so that compiler features are plugins: a package may register its
own IR **operators**, **region operators** (statements with a body) and
**types** under its own namespace, and a kernel writes them exactly like the
built-in ones. The core compiler and the harness tools know nothing about them,
so every extension ships with a **lowering pass** that rewrites its constructs
into the TIRx-lite subset. `tirx_harness.extension` wraps this facility for
kernel authoring on the installed `apache-tvm-tirx`.

Why do this instead of a Python helper function? A helper expands at trace
time and leaves no trace in the IR. An extension construct *is* IR: the pass
sees it as one node with typed operands, can rewrite it differently per
candidate or per shape, can reorder or fuse several of them before expanding,
and the printed kernel (`kernel.func.script()`) stays readable at the level of
the abstraction. Use it when that pays off; plain TIRx-lite remains valid.

## Workflow

1. **Define** the namespace, its operators/types, and their result types.
2. **Author** kernels with `@txl.kernel`, calling the published builders.
3. **Lower**: `lowered = extension.lower(kernel, *passes)`. The passes rewrite
   `Function -> Function`; `lower` rejects any remaining extension construct
   (`ExtensionResidueError`) and re-runs the TIRx-lite low-level IR contract.
4. **Consume only the lowered kernel**: `synccheck(lowered.func, inputs)`,
   `racecheck(lowered.func, inputs)`, NumSim on `lowered.func`,
   `lowered.compile()`, and the compiled callable in `setup()`. An unlowered
   kernel fails in codegen (`Unsupported region op ...`) and makes the checkers
   report `incomplete` (`unregistered raw call`).

## API (`from tirx_harness import extension`)

| Name | Role |
| --- | --- |
| `Namespace(prefix)` | The extension's namespace. `prefix` is a Python identifier other than `ir`, `prim`, `tirx`. Builders are published as attributes: `ns.<op>`, `ns.<Type without "Type" suffix>`. |
| `ns.op(name, doc, args, *, ret_type=None, infer_type=None, effect=None)` | Register `<prefix>.<name>`. `args` lists operand names (or `(name, doc)` pairs). Give **either** `ret_type` (a `Type`, a dtype string, or `"void"`/`extension.VOID` for a statement) **or** `infer_type(call) -> Type` computed from `call.args[i].ty`. `effect` defaults to `UpdateState` for void ops and `Pure` otherwise (`ir.CallEffectKind.ReadState` for load-like ops). Returns the `ir.Op`. |
| `ns.region_op(name, doc, args, *, body_params=(), validate=None)` | Register a statement with a body. `body_params` is `[(name, dtype), ...]` (fresh `Var`s per use) or a hook `(call) -> [Var, ...]`. In a kernel: `with ns.<name>(operands...) as params:`. |
| `@ns.type` | Register a Python-defined IR type: a class deriving from `ir.Type` whose annotated fields are its constructor arguments. Published without the `Type` suffix. |
| `ns.ops` | `{name: ir.Op}` of the registered operators. |
| `extension.VOID` | The void result type (`T.void`). `ir.NoneType()` is **not** void: a call of that type is silently dropped at trace time. |
| `extension.emit(kernel=None, *, loc=None)` | Context manager for lowering: runs TIRx-lite builders (`txl.ptx.*`, `txl.cuda.*`, `txl.local_scalar`, `txl.serial`, `txl.if_`, ...) into a fresh builder; `block.stmt` holds the statements after the block. `txl.lane_id()` and friends resolve against the kernel being lowered. Pass `loc=stmt.loc` so diagnostics on the emitted code point at the kernel line. |
| `extension.lower(kernel, *passes, allowed_func_calls=())` | Apply the passes (callables `Function -> Function`, or `tvm_tirx` passes) and verify. Returns a `txl.Kernel` sharing the launch metadata (or a `Function` for a bare function). |
| `extension.verify_lowered(kernel_or_func)` | The verification alone: residue check, then the low-level IR contract. |
| `extension.residue(kernel_or_func)` | List the extension calls, regions and types with `file:line` where known. |

Rewrite IR with `tvm_ffi.structural_map(func, [(ir.Call, f), (ir.Stmt, g)])`:
each callback returns its node unchanged or a replacement; children are
visited before parents (post-order), so operands are lowered before the call
that uses them. Compare operators by identity: `call.op.same_as(MY_OP)`.

## Complete example

Everything below ran against the installed packages (synccheck `clean`, GPU
output matching NumPy). Keep the extension in the candidate directory,
in `solution.py` or a sibling module imported relatively.

```python
import numpy as np
import tvm_ffi
from tvm_tirx import ir

import tirx_kernels.tirx_lite as txl
from tirx_harness import extension, synccheck

ns = extension.Namespace("kda_ext")


@ns.type
class PairType(ir.Type):
    """Two scalars sharing one dtype; exists only in the IR."""

    dtype: ir.PrimType


PAIR = ns.op("pair", "Pack two scalars.", ["a", "b"], infer_type=lambda c: PairType(c.args[0].ty))
FIRST = ns.op("first", "First element.", ["p"], infer_type=lambda c: c.args[0].ty.dtype)
SECOND = ns.op("second", "Second element.", ["p"], infer_type=lambda c: c.args[0].ty.dtype)
FMA3 = ns.op("fma3", "a * b + c.", ["a", "b", "c"], infer_type=lambda c: c.args[0].ty)
ST_SCALED = ns.op("st_scaled", "ptr <- value * scale.", ["ptr", "value", "scale"], ret_type="void")
REPEAT = ns.region_op("repeat", "Run the body n times.", ["n"], body_params=[("i", "int32")])


@txl.kernel()
def k(x: txl.Tensor((txl.int64(),), "float32"), y: txl.Tensor((txl.int64(),), "float32")):
    txl.device_entry(launch=txl.cuda.LaunchConfig(grid=1, block=32))
    lane = txl.lane_id()
    v = txl.local_scalar("float32")
    txl.ptx.ld.global_.f32(v, x.ptr_to([lane]))
    acc = txl.local_scalar("float32", init=txl.float32(0.0))
    with ns.repeat(txl.int32(3)) as i:                       # region op: a statement with a body
        p = ns.pair(v, txl.cast(i, "float32") + txl.float32(1.0))   # valued op with an extension type
        txl.assign(acc, ns.fma3(ns.first(p), ns.second(p), acc))
    ns.st_scaled(y.ptr_to([lane]), acc, txl.float32(0.5))    # void op: emitted as a statement


def lower_exprs(call):
    if call.op.same_as(FMA3):
        a, b, c = call.args
        return a * b + c
    if call.op.same_as(FIRST) or call.op.same_as(SECOND):
        inner = call.args[0]
        if isinstance(inner, ir.Call) and inner.op.same_as(PAIR):
            return inner.args[0 if call.op.same_as(FIRST) else 1]
    return call


def lower_stmts(stmt):
    if isinstance(stmt, ir.Evaluate) and isinstance(stmt.value, ir.Call):
        if stmt.value.op.same_as(ST_SCALED):
            ptr, value, scale = stmt.value.args
            with extension.emit(loc=stmt.loc) as block:      # macro-expand with TIRx-lite builders
                tmp = txl.local_scalar("float32", init=value * scale)
                txl.ptx.st.global_.f32(ptr, tmp)
            return block.stmt
    if isinstance(stmt, ir.RegionStmt) and stmt.op.same_as(REPEAT):
        (i,) = stmt.body_params
        zero, one = ir.IntImm("int32", 0), ir.IntImm("int32", 1)
        return ir.For(i, zero, stmt.args[0], one, ir.ForKind.DEFAULT, stmt.body, loc=stmt.loc)
    return stmt


def lowering(func):
    return tvm_ffi.structural_map(func, [(ir.Call, lower_exprs), (ir.Stmt, lower_stmts)])


lowered = extension.lower(k, lowering)          # txl.Kernel; raises if anything remains
print(lowered.func.script())                     # core TIRx-lite only
x = np.arange(32, dtype=np.float32)
synccheck(lowered.func, {"x": x, "y": np.zeros(32, dtype=np.float32)}).require_clean()
executable = lowered.compile()                   # the callable the benchmark times
```

Before lowering, `k.func.script()` shows the abstraction level:

```python
with kda_ext.repeat(3) as i:
    v_3[0] = kda_ext.fma3(kda_ext.first(kda_ext.pair(v_2[0], ...)), kda_ext.second(...), v_3[0])
kda_ext.st_scaled(T.address_of(y[lane_id]), v_3[0], T.float32(0.5))
```

and `extension.verify_lowered(k)` on the unlowered kernel reports every
construct with its source line:

```text
ExtensionResidueError: 7 extension construct(s) remain in kernel 'k':
  - region kda_ext.repeat at solution.py:37
  - call kda_ext.fma3
  ...
  - type kda_ext.PairType
  - call kda_ext.st_scaled at solution.py:40
Lower them with the extension's pass (tirx_harness.extension.lower) before synccheck, ...
```

## Rules of the road

- **Result types.** A statement-like operator must have `ret_type="void"`
  (`extension.VOID`). Valued operators need a result type the kernel can use:
  a `PrimType` for scalars, or one of your own types that the lowering removes.
- **Valued calls are plain Python values** in TIRx-lite: `p = ns.pair(...)`
  re-emits the call at every use (ptxas CSEs it). Anything that must run once
  goes through `txl.local_scalar(dtype, init=...)`, which needs a `PrimType`
  result. Extension-typed values therefore live between instructions only;
  lower them away (`first(pair(a, b)) -> a`).
- **Effect kinds.** A void operator is a statement; keep the default
  `UpdateState` whether it writes memory or a local (`ReadState` on a void op
  is harmless but means nothing extra). For a *valued* operator the kind says
  what the compiler may do with the expression: `Pure` may be re-emitted,
  merged or dropped when unused; `ReadState` reads memory or mutable state and
  must not move across stores (a load-like op); `UpdateState` has side effects.
- **Checker inputs for bf16.** NumPy has no bf16; bind bf16 buffers as
  `ml_dtypes.bfloat16` arrays (from a torch tensor:
  `t.cpu().view(torch.int16).numpy().view(ml_dtypes.bfloat16)`), which
  `synccheck`/`racecheck` accept directly.
- **Explicit types bypass inference.** `ir.Call(op, args, ty=...)` is not
  validated against `infer_type`; the Python route has no custom validator.
  Check operands inside `infer_type` (raise `TypeError`) so mistakes fail at
  trace time.
- **Lowering must stay in the TIRx-lite subset.** `lower` re-runs the same
  contract as `@txl.kernel(check_ir=True)`: no `TensorLoad`/`TensorStore` on
  global or shared memory (use `txl.ptx.ld/st`), no tile ops, no undeclared
  `func_call`. Keep `check_ir=True` on the kernel.
- **No shared-memory allocation inside `emit`.** The pool is committed when
  the kernel body closes. Allocate in the kernel and pass pointers as operands.
- **Locations.** Pass `loc=stmt.loc` to `emit` and to the `ir.For`/`ir.SeqStmt`
  you construct, so synccheck/racecheck findings point at the kernel line.
- **Registration is process-global and once.** A second definition of the
  same operator name replaces its signature and attributes with a warning;
  a second `@ns.type` with the same class name raises. The benchmark loads one
  candidate per process; your own driver scripts should do the same, or use
  one namespace per candidate.
- **Nested regions and `For`.** A region body is a `SeqStmt`; it may contain
  other regions, which the same pass lowers (post-order visits children
  first). Replace a `RegionStmt` with a statement that binds its
  `body_params` (an `ir.For` over the trip variable, or `ir.Bind`s followed by
  the body in a `SeqStmt`).
- **Debugging a pass.** An error inside an `emit` block is re-raised as
  `ExtensionLoweringError` whose message names the original error and the
  kernel line being lowered, and `lower` notes which pass failed at the end
  of the (long) `structural_map` traceback. Print `func.script()` before and
  after; use `extension.residue(func)` to see what remains; `ir.Op.list_op_names()`
  lists every registered operator, and `ir.Op.get("tirx.ptx.st")` shows how a
  core instruction is spelled when `emit` is not convenient.
