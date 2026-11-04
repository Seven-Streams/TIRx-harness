# NumSim - Deterministic CPU Execution

NumSim executes supported TIRx kernels deterministically on CPU, without a GPU.
It transpiles a specialized kernel, runs concrete inputs, and returns outputs
and diagnostics. Use it to reproduce numerical behavior, compare outputs with
an independent reference, and locate mismatches by output name, logical index,
actual and expected values, and static writer/source sites.

NumSim is not a reference oracle and does not detect races or synchronization
errors. Use Racecheck and Synccheck for those properties.

## Run it

```python
import numpy as np
from tirx_harness import numsim

case = numsim.NumSimCase(
    kernel=get_kernel().func,                 # the TIRx Function (lowered, see below)
    args={"x": x, "y": np.zeros_like(y_ref), "s": 2.0},   # every parameter by name
    outputs=("y",),                           # buffers to compare; None = all bound outputs
    reference=lambda: {"y": y_ref},           # independent reference, same names
    comparisons={"y": numsim.ComparisonSpec(rtol=1e-2, atol=1e-2)},
)
report = numsim.run_case(case)
report.require_ok()                           # raises AssertionError with the mismatches
```

Without a reference, run and read the outputs directly:

```python
module = numsim.transpile(kernel.func)
result = numsim.Engine().run(module, inputs, outputs=("y",))
result.outputs["y"], result.diagnostics, result.verdict
```

## Inputs

- `args`/`inputs` bind every kernel parameter by name: buffers as CPU NumPy
  arrays of the parameter's dtype (bf16 as `ml_dtypes.bfloat16`; from torch:
  `t.cpu().view(torch.int16).numpy().view(ml_dtypes.bfloat16)`), scalars as
  Python numbers, tensor maps through `numsim.TensorMap(...).numpy()`.
  Outputs are bound like inputs and overwritten.
- Every launch extent must be static. A grid such as `grid=N * Hv` with `N`
  a runtime scalar fails at transpile with
  `UnsupportedTIRxError: NumSim launch extent is not statically known: N`,
  even when `N` is bound in `args`. Trace a NumSim-only variant of the kernel
  with the concrete grid (shape specialization), or specialize the production
  kernel on the shape it is launched with; Synccheck and Racecheck accept the
  dynamic form.
- A kernel that uses IR extensions is simulated after its own lowering: pass
  the `.func` of the `tirx_harness.extension.lower(...)` result (or of the
  `extension.attach(...)` kernel). An unlowered kernel raises
  `ExtensionResidueError` naming the constructs.
- Start with the smallest representative shape; build and simulation cost
  grow with the simulated work. The first transpile in a fresh cache also
  builds the native engine (tens of seconds); later kernels reuse it.

## Read the result

| Attribute or method | Meaning |
|---|---|
| `report.ok` | Every compared output is within its `ComparisonSpec` |
| `report.mismatches` | `Mismatch(output, index, actual, expected)`; `mismatch.render()` adds the writer/source sites |
| `report.verdict` | `clean` or `review` (simulator advisories, read `report.diagnostics`) |
| `result.outputs` | Selected arrays in the parameter's dtype (bf16 arrays come back as `ml_dtypes.bfloat16`) |
| `ComparisonSpec(rtol, atol, equal_nan, actual_encoding, regions)` | Per-output tolerance; `actual_encoding="bfloat16"` decodes a `uint16` bit-pattern output; `regions` restricts the comparison |

Compare only buffers the kernel writes: an untouched output holds its initial
bytes on both sides. fp32 results agree with the GPU to a few ULP (FMA
contraction and `ex2`-style approximations differ); bf16 results typically
agree exactly after the same rounding.

## API

```python
numsim.transpile(func, *, cache_dir=None) -> CompiledModule
numsim.Engine(max_workers="auto").run(module, inputs, *, outputs=None) -> NumSimResult
numsim.compare(result, expected, *, tolerances=None) -> NumSimReport
numsim.run_case(case, *, engine=None) -> NumSimReport
numsim.dump_rust(module) -> str
```

The full reference is the harness documentation page `docs/api/numsim.md`
(inputs: `docs/api/inputs.md`). Set `NUMSIM_CACHE_DIR` to place the build
cache outside the home directory when disk space is tight.
