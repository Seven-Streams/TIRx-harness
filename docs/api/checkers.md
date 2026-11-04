# Checkers and reports

Import the synchronization checker (`synccheck`) and memory-race checker
(`racecheck`) from `tirx_harness`. Both execute on the CPU for a concrete kernel
invocation. Their guarantees and limitations are described in
[Compiler analysis](../components/tools.md).

## Entry points

Pass a specialized TIRx function as `kernel`; for a TIRx-lite kernel, use
`kernel.func`. Bind all runtime parameters by name in `inputs`, including
output buffers. Buffer bindings are CPU NumPy arrays (bf16 buffers as
`ml_dtypes.bfloat16`); scalar parameters take concrete scalar values. Only
parameterless kernels may omit the bindings. Tensor-map parameters use
{py:meth}`tirx_harness.numsim.TensorMap.numpy`. A kernel that uses
[IR extensions](extension.md) is checked after its lowering, through the
`.func` of the `lower` (or `attach`) result. An unlowered kernel returns
`incomplete` with one `extension_residue` finding that lists the constructs and
their source lines.

```{eval-rst}
.. autoapifunction:: tirx_harness.synccheck
```

Returns a {py:class}`~tirx_harness.numsim.checker_report.SyncCheckReport`.

```{eval-rst}
.. autoapifunction:: tirx_harness.racecheck
```

Returns a {py:class}`~tirx_harness.numsim.checker_report.RaceReport`.

## Reports

Inspect `report.verdict` and `report.findings`, save `report.to_dict()`, or call
`report.print()` for readable evidence. `report.require_clean()` raises a
`RuntimeError` subclass unless the verdict is `clean`.

| Verdict | Meaning |
| --- | --- |
| `clean` | The invocation passed within the checker's supported model and coverage. |
| `review` | An advisory needs investigation. |
| `incomplete` | Missing bindings, unsupported behavior, or a coverage limit prevented a complete check. |
| `error` | The checker detected a violation. |

Each finding has `id`, `status`, `kind`, `message`, and `details` fields, plus
`to_dict()` for serialization. Details carry available source and witness
evidence. A clean report applies to the supplied specialization, launch, and
inputs; see the [checker scope](../components/tools.md#checker-api).

The following classes are return types; obtain them from the entry points above.

```{eval-rst}
.. autoapiclass:: tirx_harness.numsim.checker_report.SyncCheckReport
   :members: verdict, findings, to_dict, format, print, require_clean
   :undoc-members:
```

```{eval-rst}
.. autoapiclass:: tirx_harness.numsim.checker_report.RaceReport
   :members: verdict, findings, to_dict, format, print, require_clean
   :undoc-members:
```
