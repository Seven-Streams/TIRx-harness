### IR extensions: design your own abstractions

TIRx is extensible. A kernel may use operators, region operators (statements
with a body) and value types that you define yourself under your own
namespace, provided you also define the pass that lowers them to the TIRx-lite
subset. Use this when a reusable abstraction makes the kernel easier to write,
restructure or optimize than spelling it out inline: a fused instruction
sequence with its own operands, a scheduling construct with a body, a value
carried between instructions, a staging/pipeline protocol. Prefer an
abstraction with a lowering over copying an instruction sequence between
kernels or candidates; the pass is also where a schedule-level rewrite lives.
Plain TIRx-lite remains valid.

- Read `references/tirx-ir-extension.md` in `tirx-wiki` before the first use.
  It documents `tirx_harness.extension` (`Namespace`, `.op`, `.region_op`,
  `.type`, `emit`, `lower`) for the installed tvm-tirx and walks through a
  complete kernel, lowering and check.
- Define the extension inside the candidate directory (in `solution.py` or a
  sibling module imported relatively); a frontier member stays
  self-contained. Registrations are process-global: register each name once
  per process, which the one-candidate-per-benchmark-process model guarantees.
- Author kernels with `@txl.kernel` as usual, calling your builders in the
  body. Write the lowering as passes (`Function -> Function`, for example with
  `tvm_ffi.structural_map`) and emit replacement statements with
  `extension.emit()` and the TIRx-lite builders. The pass is also where
  schedule-level rewrites live: fusing adjacent constructs, choosing a lowering
  per shape, moving a block; not only macro expansion.
- Lower before anything else consumes a kernel:
  `lowered = tirx_harness.extension.lower(kernel, *passes)`, or bind the passes
  once with `kernel = extension.attach(kernel, *passes)` so that one object
  serves everything. Only the lowered `txl.Kernel` goes to
  synccheck/racecheck/NumSim (`.func`), to `.compile()`, and into the callable
  the benchmark times. Both reject any remaining extension construct and
  re-run the TIRx-lite low-level IR contract on the result; an unlowered
  kernel is neither checkable nor compilable (the checkers report it as an
  `extension_residue` finding, not a tool defect).
- `check_ir=True` stays on. The lowered kernel still originates from
  `@txl.kernel` and is bound by the same implementation contract.
- Record in `frontier/index.json` which abstractions a member defines and how
  they lower, so the mechanism stays reviewable.
