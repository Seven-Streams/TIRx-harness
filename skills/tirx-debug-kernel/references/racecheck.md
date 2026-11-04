# Racecheck - Data Race Detection

Run a TIRx kernel with concrete inputs and check its global-memory,
shared-memory, and TMEM access ordering before running it on GPU.

## Run it

```python
from tirx_harness import racecheck

report = racecheck(
    get_kernel(),
    inputs,
)
report.print()
report.require_clean()
```

The public API is:

```python
racecheck(kernel, inputs=None)
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

Missing bindings and unsupported executed effects produce `incomplete`.
Racecheck never treats an execution it could not certify as clean.

## When to use it

- After changing a kernel's global-memory, shared-memory, or TMEM access pattern.
- When incorrect output may come from cross-lane, cross-warp, or asynchronous
  memory overlap.
- To verify that barrier and async-drain protocols order every conflicting
  access.

## Detection model

Racecheck executes the concrete kernel control flow once and records exact
lane-resolved physical accesses. It uses vector clocks to combine program
order, barrier release/acquire edges, and asynchronous issue/completion edges.
Two overlapping global-memory, shared-memory, or TMEM accesses form a race
when at least one is a write and neither access happens before the other.

The checker also reports:

- exact view or backing-allocation out-of-bounds accesses;
- asynchronous TMEM lifetime conflicts;
- synchronization errors encountered by the same execution;
- typed coverage gaps when an executed operation cannot be modeled.

Global conflicts are checked across the full launch. Racecheck recognizes
compatible release/acquire publication, fences, barriers, and modeled proxy or
async handoffs as happens-before. Atomic modification order alone does not
establish happens-before.

## Inputs and coverage

A verdict applies only to the supplied invocation. Choose a small set of input
configurations that covers the relevant boundaries, including pipeline fill,
steady state, drain, slot reuse, and multi-work-item lifetimes.

Racecheck does not enumerate alternate ordinary-memory values, atomic return
orders, or control-flow paths that were not selected by the supplied inputs.
Run additional inputs when those values can change synchronization or memory
accesses.

## Read the result

| Finding | Meaning |
|---|---|
| `write_read` / `read_write` / `write_write` | Two unordered, physically overlapping global-memory, shared-memory, or TMEM accesses conflict. |
| asynchronous lifetime error | A TMEM or async-copy footprint is reused before its operation is complete. |
| out-of-bounds access | An executed lane accesses bytes outside its view or backing allocation. |
| synchronization error | The execution violates a barrier or synchronization protocol. |
| `incomplete` | Missing evidence or an unsupported executed effect prevented certification. |
| `review` | A non-fatal advisory needs an explicit disposition. |

`report.print()` includes the responsible operation, source text, CTA/warp/lane
context, the conflicting or witness operation when relevant, and the exact
overlap.

The structured report API is:

| Attribute or method | Meaning |
|---|---|
| `report.verdict` | `clean`, `review`, `incomplete`, or `error` |
| `report.findings` | Structured findings with kind, status, message, and evidence |
| `report.print()` | Print findings and their source evidence |
| `report.to_dict()` | Return a JSON-safe report snapshot |
| `report.require_clean()` | Raise unless the verdict is `clean` |

Verdict precedence is `error > incomplete > review > clean`.

## Physical overlap

Racecheck compares physical bytes rather than buffer names:

- shared-memory aliases from the same allocation pool overlap when their byte
  ranges overlap;
- global-memory aliases are compared by physical byte range across the full
  launch;
- TMEM views with different element dtypes overlap when they resolve to the
  same physical columns and bytes;
- lane layouts, swizzles, and exact active-lane predicates are resolved before
  conflict checking;
- shared memory is CTA-scoped, while cluster-visible storage keeps its encoded
  ownership and target CTA.

## Current limitations

- Unsupported or ambiguous global-memory observations are `incomplete`, not
  clean.
- Opaque CUDA bodies and unsupported executed TIRx effects are `incomplete`.
- A resource-limited or otherwise partial execution cannot certify the full
  invocation.
- The checker observes one concrete atomic return order and one concrete
  control-flow path per input configuration.
