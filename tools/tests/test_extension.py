"""IR extensions: define them, use them in a TIRx-lite kernel, lower, verify."""

import uuid
import warnings

import numpy as np
import pytest
import tvm_ffi

import tirx_kernels.tirx_lite as txl
from tirx_harness import extension, synccheck
from tirx_kernels.tirx_lite.low_level_ir import LowLevelIRContractError
from tvm_tirx import ir


def _fresh_namespace() -> extension.Namespace:
    # Registrations are process-global; every test defines its own namespace.
    return extension.Namespace(f"ext_{uuid.uuid4().hex[:8]}")


def _collect(root, node_type) -> list:
    """Every node of ``node_type`` in ``root``, including nested ones."""
    found = []

    def visit(node, visitor):
        found.append(node)
        visitor.default_visit(node)

    tvm_ffi.structural_visit(root, [(node_type, visit)])
    return found


class _Complex:
    """A small extension: a value type, pure ops, a statement op and a region op."""

    def __init__(self):
        ns = self.ns = _fresh_namespace()

        @ns.type
        class PairType(ir.Type):
            dtype: ir.PrimType

        self.PairType = PairType
        self.pair = ns.op("pair", "Pack.", ["a", "b"], infer_type=lambda c: PairType(c.args[0].ty))
        self.first = ns.op("first", "First.", ["p"], infer_type=lambda c: c.args[0].ty.dtype)
        self.second = ns.op("second", "Second.", ["p"], infer_type=lambda c: c.args[0].ty.dtype)
        self.fma3 = ns.op("fma3", "a*b+c.", ["a", "b", "c"], infer_type=lambda c: c.args[0].ty)
        self.st_scaled = ns.op(
            "st_scaled", "ptr <- v*s.", ["ptr", "value", "scale"], ret_type="void"
        )
        self.repeat = ns.region_op("repeat", "Loop.", ["n"], body_params=[("i", "int32")])

    def kernel(self):
        ns = self.ns

        @txl.kernel()
        def k(x: txl.Tensor((txl.int64(),), "float32"), y: txl.Tensor((txl.int64(),), "float32")):
            txl.device_entry(launch=txl.cuda.LaunchConfig(grid=1, block=32))
            lane = txl.lane_id()
            v = txl.local_scalar("float32")
            txl.ptx.ld.global_.f32(v, x.ptr_to([lane]))
            acc = txl.local_scalar("float32", init=txl.float32(0.0))
            with ns.repeat(txl.int32(3)) as i:
                p = ns.pair(v, txl.cast(i, "float32") + txl.float32(1.0))
                txl.assign(acc, ns.fma3(ns.first(p), ns.second(p), acc))
            ns.st_scaled(y.ptr_to([lane]), acc, txl.float32(0.5))

        return k

    def lowering(self, func):
        def exprs(call):
            if call.op.same_as(self.fma3):
                a, b, c = call.args
                return a * b + c
            if call.op.same_as(self.first) or call.op.same_as(self.second):
                inner = call.args[0]
                if isinstance(inner, ir.Call) and inner.op.same_as(self.pair):
                    return inner.args[0 if call.op.same_as(self.first) else 1]
            return call

        def stmts(stmt):
            if isinstance(stmt, ir.Evaluate) and isinstance(stmt.value, ir.Call):
                if stmt.value.op.same_as(self.st_scaled):
                    ptr, value, scale = stmt.value.args
                    with extension.emit(loc=stmt.loc) as block:
                        tmp = txl.local_scalar("float32", init=value * scale)
                        txl.ptx.st.global_.f32(ptr, tmp)
                    return block.stmt
            if isinstance(stmt, ir.RegionStmt) and stmt.op.same_as(self.repeat):
                (i,) = stmt.body_params
                zero, one = ir.IntImm("int32", 0), ir.IntImm("int32", 1)
                return ir.For(
                    i, zero, stmt.args[0], one, ir.ForKind.DEFAULT, stmt.body, loc=stmt.loc
                )
            return stmt

        return tvm_ffi.structural_map(func, [(ir.Call, exprs), (ir.Stmt, stmts)])

    @staticmethod
    def expected(x: np.ndarray) -> np.ndarray:
        acc = np.zeros_like(x)
        for i in range(3):
            acc = x * (i + 1.0) + acc
        return acc * 0.5


def test_namespace_publishes_builders_and_registers_ops():
    ext = _Complex()
    ns = ext.ns
    assert set(ns.ops) == {"pair", "first", "second", "fma3", "st_scaled", "repeat"}
    assert ns.ops["fma3"].same_as(ext.fma3)
    assert ns.Pair is ext.PairType  # the "Type" suffix is dropped in the published alias
    assert callable(ns.fma3) and callable(ns.repeat)
    with pytest.raises(AttributeError, match="no builder 'missing'"):
        ns.missing
    with pytest.raises(ValueError, match="core IR namespace"):
        extension.Namespace("tirx")
    with pytest.raises(ValueError, match="Python identifier"):
        extension.Namespace("my-ext")


def test_void_ops_emit_statements_and_fixed_return_types_apply():
    ns = _fresh_namespace()
    ns.op("touch", "Statement.", ["ptr"], ret_type="void")
    sq = ns.op("sq", "Square.", ["a"], ret_type="float32")
    assert ns.ops["touch"].get_attr("TCallEffectKind") == int(ir.CallEffectKind.UpdateState)
    assert sq.get_attr("TCallEffectKind") == int(ir.CallEffectKind.Pure)
    assert str(ir.Call(sq, [ir.FloatImm("float32", 2.0)]).ty) == "float32"

    @txl.kernel()
    def k(y: txl.Tensor((txl.int64(),), "float32")):
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=1, block=32))
        ns.touch(y.ptr_to([txl.lane_id()]))

    values = [node.value for node in _collect(k.func, ir.Evaluate)]
    assert any(isinstance(v, ir.Call) and v.op.same_as(ns.ops["touch"]) for v in values)


def test_residue_lists_every_extension_construct_with_kernel_locations():
    ext = _Complex()
    k = ext.kernel()
    found = extension.residue(k)
    kinds = {(r.kind, r.name.rpartition(".")[2]) for r in found}
    assert kinds == {
        ("region", "repeat"),
        ("call", "pair"),
        ("call", "first"),
        ("call", "second"),
        ("call", "fma3"),
        ("call", "st_scaled"),
        ("type", "PairType"),
    }
    located = {r.name.rpartition(".")[2]: r.loc for r in found if r.loc}
    # Statement-level constructs point at the kernel source, not at the harness.
    assert located["repeat"].startswith(__file__) and located["st_scaled"].startswith(__file__)
    with pytest.raises(extension.ExtensionResidueError, match="7 extension construct") as info:
        extension.verify_lowered(k)
    assert "tirx_harness.extension.lower" in str(info.value)
    assert len(info.value.residues) == 7


def test_lower_returns_a_core_only_kernel_that_checks_clean():
    ext = _Complex()
    k = ext.kernel()
    lowered = extension.lower(k, ext.lowering)
    assert isinstance(lowered, txl.Kernel)
    assert lowered.name == k.name and lowered.warps == k.warps and lowered.session is k.session
    assert extension.residue(lowered) == []
    assert lowered.func.attrs["global_symbol"] == "k"
    # The lowered statements carry the location of the kernel line they replace.
    locs = {extension._source_loc(node.loc) for node in _collect(lowered.func, ir.TensorStore)}
    assert any(loc and loc.startswith(__file__) for loc in locs)
    x = np.arange(32, dtype=np.float32)
    report = synccheck(lowered.func, {"x": x, "y": np.zeros(32, dtype=np.float32)})
    assert report.verdict == "clean", report.to_dict()


def test_lower_accepts_bare_functions_and_tvm_passes():
    from tvm_tirx import tirx

    ext = _Complex()
    k = ext.kernel()
    marker = tirx.transform.function_pass(
        lambda func, mod, ctx: func.with_attr("marker", 1), opt_level=0, name="marker"
    )
    lowered = extension.lower(k.func, [ext.lowering], marker)
    assert isinstance(lowered, tirx.Function)
    assert lowered.attrs["marker"] == 1
    assert extension.residue(lowered) == []


def test_lower_rejects_bad_passes_and_contract_violations():
    ext = _Complex()
    k = ext.kernel()
    with pytest.raises(TypeError, match="must return the rewritten"):
        extension.lower(k, lambda func: None)
    with pytest.raises(TypeError, match="must be callable"):
        extension.lower(k, object())
    with pytest.raises(extension.ExtensionResidueError):
        extension.lower(k, lambda func: func)

    def to_buffer_store(func):
        # A lowering that writes global memory through TensorStore breaks the
        # TIRx-lite contract; lower must re-run that check on its result.
        def stmts(stmt):
            if isinstance(stmt, ir.Evaluate) and isinstance(stmt.value, ir.Call):
                if stmt.value.op.same_as(ext.st_scaled):
                    ptr, value, scale = stmt.value.args
                    load = ptr.args[0]  # the TensorLoad under tirx.address_of
                    return ir.TensorStore(load.source, load.indices, value * scale, loc=stmt.loc)
            return stmt

        return ext.lowering(tvm_ffi.structural_map(func, [(ir.Stmt, stmts)]))

    with pytest.raises(LowLevelIRContractError, match="buffer_store"):
        extension.lower(k, to_buffer_store)


def test_emit_uses_the_active_kernel_session():
    ext = _Complex()
    k = ext.kernel()
    lane_vars = []

    def record(func):
        with extension.emit() as block:
            lane_vars.append(txl.lane_id())
            txl.ptx.st.global_.f32(func.params[1].ptr_to([lane_vars[-1]]), txl.float32(1.0))
        assert isinstance(block.stmt, ir.Stmt)
        return func

    extension.lower(k, record, ext.lowering)
    bound = {str(v.name) for v in lane_vars}
    assert bound == {"lane_id"}
    with pytest.raises(RuntimeError, match="active txl.device_entry"):
        with extension.emit():
            txl.lane_id()


def test_redefinition_replaces_the_definition_with_a_warning():
    ns = _fresh_namespace()
    ns.op("f", "First definition.", ["a"], infer_type=lambda c: c.args[0].ty)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        op = ns.op("f", "Second definition.", ["a", "b"], ret_type="int32")
    assert any("already registered" in str(w.message) for w in caught)
    assert [info.name for info in op.args_info] == ["a", "b"]
    assert str(ir.Call(op, [ir.IntImm("int32", 1), ir.IntImm("int32", 2)]).ty) == "int32"


def test_region_body_params_hook_and_validation():
    ns = _fresh_namespace()
    seen = []

    def params(call):
        return [ir.Var("row", "int32"), ir.Var("col", "int32")]

    def validate(region):
        seen.append(len(region.body_params))

    ns.region_op("tile", "2-D tile loop.", ["rows", "cols"], body_params=params, validate=validate)

    @txl.kernel()
    def k(y: txl.Tensor((txl.int64(),), "float32")):
        txl.device_entry(launch=txl.cuda.LaunchConfig(grid=1, block=32))
        with ns.tile(txl.int32(2), txl.int32(4)) as (row, col):
            txl.ptx.st.global_.f32(y.ptr_to([row * 4 + col]), txl.float32(1.0))

    tiles = [r for r in _collect(k.func, ir.RegionStmt) if r.op.same_as(ns.ops["tile"])]
    assert len(tiles) == 1 and [str(v.name) for v in tiles[0].body_params] == ["row", "col"]
    assert seen == [2]


@pytest.mark.numsim_gpu
def test_lowered_kernel_compiles_and_runs_on_the_gpu():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    ext = _Complex()
    lowered = extension.lower(ext.kernel(), ext.lowering)
    executable = lowered.compile()
    x = torch.arange(32, dtype=torch.float32, device="cuda")
    y = torch.zeros(32, dtype=torch.float32, device="cuda")
    executable(x, y)
    torch.cuda.synchronize()
    np.testing.assert_allclose(y.cpu().numpy(), ext.expected(x.cpu().numpy()))
