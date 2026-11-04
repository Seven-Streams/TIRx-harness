"""An unlowered IR-extension kernel is reported as such, not as a frontend gap."""

import uuid

import numpy as np
import pytest
import tvm_ffi

import tirx_kernels.tirx_lite as txl
from tirx_harness import extension, numsim, racecheck, synccheck
from tvm_tirx import ir


def _kernel_and_lowering():
    ns = extension.Namespace(f"res_{uuid.uuid4().hex[:8]}")
    twice = ns.op("twice", "Double.", ["a"], infer_type=lambda c: c.args[0].ty)

    @txl.kernel()
    def k(x: txl.Tensor((txl.int64(),), "float32")):
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=1, block=32))
        lane = txl.lane_id()
        v = txl.local_scalar("float32")
        txl.ptx.ld.global_.f32(v, x.ptr_to([lane]))
        txl.ptx.st.global_.f32(x.ptr_to([lane]), ns.twice(v))

    def lowering(func):
        def calls(call):
            if call.op.same_as(twice):
                return call.args[0] * ir.FloatImm("float32", 2.0)
            return call

        return tvm_ffi.structural_map(func, [(ir.Call, calls)])

    return ns, k, lowering


@pytest.mark.parametrize("check", [synccheck, racecheck], ids=["synccheck", "racecheck"])
def test_unlowered_kernel_reports_extension_residue(check):
    ns, k, _ = _kernel_and_lowering()
    report = check(k.func, {"x": np.zeros(32, dtype=np.float32)})
    assert report.verdict == "incomplete"
    (finding,) = report.findings
    assert (finding.status, finding.kind) == ("incomplete", "extension_residue")
    assert f"call {ns.prefix}.twice" in finding.message
    assert finding.details["constructs"] == [
        {"kind": "call", "name": f"{ns.prefix}.twice", "loc": None}
    ]
    assert "tirx_harness.extension.lower" in finding.details["hint"]
    text = report.format()
    assert "extension_residue" in text and "Hint:" in text
    assert report.to_dict()["native"]["coverage"]["termination"]["kind"] == "extension_residue"


def test_lowered_kernel_is_checked_normally():
    _, k, lowering = _kernel_and_lowering()
    lowered = extension.lower(k, lowering)
    inputs = {"x": np.zeros(32, dtype=np.float32)}
    assert synccheck(lowered.func, inputs).verdict == "clean"
    assert racecheck(lowered.func, inputs).verdict == "clean"


def test_numsim_transpile_rejects_an_unlowered_kernel():
    ns, k, _ = _kernel_and_lowering()
    with pytest.raises(extension.ExtensionResidueError, match=f"{ns.prefix}.twice"):
        numsim.transpile(k.func)
