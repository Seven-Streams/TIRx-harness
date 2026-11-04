# Synccheck - Barrier Verification

Run a TIRx kernel with concrete inputs and check its synchronization protocol
before compiling or running it on GPU.

## Run it

```python
from tirx_harness import synccheck

report = synccheck(
    get_kernel(),
    inputs=inputs,
)
report.print()
report.require_clean()
```

The public API is:

```python
synccheck(kernel, inputs=None)
```

`kernel` is the TIRx `PrimFunc`. `inputs` is the concrete scalar and buffer
binding dictionary for one invocation. A parameterless kernel may omit
`inputs`; otherwise provide every runtime argument needed by the executed
control flow.

Buffer bindings are CPU NumPy arrays of the parameter's dtype; for bf16 use
`ml_dtypes.bfloat16` (from a torch tensor:
`t.cpu().view(torch.int16).numpy().view(ml_dtypes.bfloat16)`). A kernel that
uses IR extensions is checked through the `tirx_harness.extension.lower(...)`
result's `.func`.

Missing bindings and unsupported executed operations produce `incomplete`.
Synccheck never treats an execution it could not certify as clean.

Verdicts are configuration-specific. Choose inputs that exercise the relevant
boundaries, pipeline fill, steady state, drain, slot reuse, and work-item
lifetimes. A clean result applies only to the exact specialization, launch
topology, and inputs checked.

## Read the result

| Finding | Meaning |
|---|---|
| mbarrier protocol error | An executed barrier lifecycle or generation ordering is illegal. |
| deadlock | The modeled launch cannot make further synchronization progress. |
| barrier participant/count error | A named, CTA, or cluster barrier has an invalid participant set or arrival count. |
| `setmaxnreg` protocol error | Register-pool requests or releases cannot be satisfied legally. |
| out-of-bounds access | An executed concrete memory access is outside its view or backing allocation. |
| incomplete | Missing evidence or a resource limit prevented certification. |

`report.print()` identifies the kernel operation responsible for a finding and,
when relevant, related or witness operations. The output includes source text
and warp/CTA context when available. Use `finding.details` or
`report.to_dict()` for structured evidence.

The report API is:

| Attribute or method | Meaning |
|---|---|
| `report.verdict` | `clean`, `review`, `incomplete`, or `error` |
| `report.findings` | Structured findings with kind, message, status, and evidence |
| `report.print()` | Print findings and their source evidence |
| `report.to_dict()` | Return a JSON-safe report snapshot |
| `report.require_clean()` | Raise unless the verdict is `clean` |

Verdict precedence is `error > incomplete > review > clean`. Treat
`incomplete` as not OK: it means Synccheck could not certify the configuration,
even if it did not report an error.

## Coverage

For the supplied inputs, Synccheck executes concrete control flow and checks
the modeled synchronization interleavings of the resulting synchronization
program.

It does not enumerate alternate ordinary-memory values, alternate atomic/CAS
return orders, or control-flow paths not selected by that concrete execution.
Run additional configurations when those values can change synchronization.
Data races and scoped memory ordering belong to [racecheck](racecheck.md).

Internal analysis limits fail closed as `incomplete`.

Current limitations:

- Opaque CUDA bodies and unsupported executed TIRx effects are incomplete.
- Exit-sensitive cluster-barrier participation that cannot be modeled is
  `cluster_barrier_warp_exit_unmodeled` and incomplete.
- An execution stopped by an internal limit cannot certify the full launch.
- Barrier names may use physical pool offsets instead of source-level names.
