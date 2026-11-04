# IR extensions

`tirx_harness.extension` is the authoring workflow for tvm-tirx IR
extensions on TIRx-lite kernels: register your own operators, region
operators and types under a namespace, write the kernel against them, then
lower them to the core IR before any tool consumes the kernel.

The code generator, NumSim, `synccheck` and `racecheck` accept only the core
IR (`ir.*`, `prim.*`, `tirx.*`). The lowering is yours; `lower` runs it,
rejects any construct that remains and re-applies the TIRx-lite low-level IR
contract to the result.

```python
from tirx_harness import extension

ns = extension.Namespace("my_ext")
FMA3 = ns.op("fma3", "a * b + c.", ["a", "b", "c"], infer_type=lambda c: c.args[0].ty)
# ... author @txl.kernel bodies calling ns.fma3(...) ...
lowered = extension.lower(kernel, my_lowering_pass)   # a txl.Kernel with core IR only
synccheck(lowered.func, inputs)
executable = lowered.compile()
```

The tirx-wiki skill page `references/tirx-ir-extension.md` walks through a
complete extension with a region operator, an extension type, the lowering
pass and the checks.

## Definitions

```{eval-rst}
.. autoapiclass:: tirx_harness.extension.Namespace
   :members: op, region_op, type, ops
   :undoc-members:
```

```{eval-rst}
.. autoapidata:: tirx_harness.extension.VOID
```

## Lowering

```{eval-rst}
.. autoapifunction:: tirx_harness.extension.lower
```

```{eval-rst}
.. autoapifunction:: tirx_harness.extension.emit
```

## Verification

```{eval-rst}
.. autoapifunction:: tirx_harness.extension.verify_lowered
```

```{eval-rst}
.. autoapifunction:: tirx_harness.extension.residue
```

```{eval-rst}
.. autoapiclass:: tirx_harness.extension.ExtensionResidueError
```

```{eval-rst}
.. autoapiclass:: tirx_harness.extension.Residue
   :members: kind, name, loc
```
