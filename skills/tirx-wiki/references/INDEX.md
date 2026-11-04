# External reference corpus

Use the installed TIRx source, fetched `repos/tirx-kernels/` checkout, and PTX ISA manual as
anchors when they match the question. Use this corpus for architecture
background, external implementation patterns, profiler semantics, and hardware
context as appropriate.

## Contents

- [IR extensions](tirx-ir-extension.md): defining operators, region operators
  and types of your own for TIRx-lite kernels, and lowering them with
  `tirx_harness.extension` before the checkers, NumSim and the compiler see
  the kernel. Versioned with this skill.
- [NVIDIA manuals](manuals/INDEX.md): CUDA programming, PTX ISA, and Nsight
  Compute CLI/profiling references fetched from NVIDIA by the skill fetcher and
  converted to RST for local reading and keyword search.
- [GPU engineering blogs](blogs/INDEX.md): Colfax/CUTLASS and Hazy Research /
  ThunderKittens article snapshots.
- [External repositories](repos/INDEX.md): shallow clones of the upstream
  default branches made by the skill fetcher.

## Reading order

1. For implementation patterns and the root README, inspect `repos/tirx-kernels/` and
   installed TIRx source.
2. For instruction, ordering, or memory semantics, verify against
   the locally materialized `manuals/ptx_isa.rst`.
3. For architecture context, optimization patterns, or external mechanics, use
   the relevant blog, manual, or live repository checkout.

Search the offline corpus without loading it wholesale:

```bash
rg -n '<term>' references/blogs references/manuals
```

The blog files remain snapshots. Manuals and external repositories are not
versioned with this skill: their URLs are owned by [sources.json](sources.json),
and [fetch_references.py](../scripts/fetch_references.py) obtains the versions
served by those upstreams at that time. Eval setup invokes the same fetcher.
These remain third-party materials; preserve their provenance and consult the
upstream publisher for redistribution terms.
