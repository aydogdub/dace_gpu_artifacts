import dace
from dace.sdfg import nodes
import numpy as np
import cupy as cp
import pytest
import copy

"""
These tests specifically check whether the 'MoveArrayOutOfKernel' pass is correctly applied during
GPU transformations, whether the resulting SDFG compiles, and whether correctness is preserved
after the transformation.

It is difficult to verify that the pass behaves exactly as intended — for example, whether memlets
reflect the precise subset of data moved or just a superset. Since the pass supports a rare and
discouraged use case, some edge cases may remain untested, and exhaustive validation is not
considered worthwhile.

Still, compilation, correctness, and coarse-grained SDFG-level changes are covered here, which
provides strong practical confidence in the pass.
"""
@pytest.mark.gpu
def test_flat_transient():
    """
    Kernel has no nested SDFGs inside it and one local, 
    GPU_Global array which needs to be moved out.
    """
    N = dace.symbol('N')

    @dace.program
    def flat_transient(A: dace.float64[N, N, 10, 64]):
        for x, y, z in dace.map[0:N, 0:N, 0:10:2]:
            # Create local array with the same name as an outer array (A becomes gpu_A as well)
            gpu_A = dace.define_local([64], np.float64, storage=dace.dtypes.StorageType.GPU_Global)
            gpu_A[:] = 1
            A[x, y, z, :] = gpu_A
            

    sdfg = flat_transient.to_sdfg()


    # Capture the MoveArrayOutOfKernel warning
    with pytest.warns(UserWarning, match="Transient array 'gpu_A' with storage type GPU_Global"):
        sdfg.apply_gpu_transformations()

    # Verify that gpu_A has been moved out of the Kernel (which is done via the mapExit).
    # So get MapExit first
    mapExit = None
    mapExit_parent_state = None
    for node, parent_state in sdfg.all_nodes_recursive():
        if isinstance(node, nodes.MapExit) and node.map.schedule == dace.dtypes.ScheduleType.GPU_Device:
            mapExit = node
            mapExit_parent_state = parent_state
            break

    # We expect two outgoing edges from the MapExit:
    # 1. One to the previously local `gpu_A`
    # 2. One to the GPU_Global version of `A` (created during GPU transformation), likely named `gpu_A_0`
    count = 0
    for edge in mapExit_parent_state.out_edges(mapExit):
        dst = edge.dst
        assert isinstance(dst, nodes.AccessNode), \
            f"Expected destination node to be an AccessNode, but got {type(dst).__name__}"

        assert dst.data in {"gpu_A", "gpu_A_0"}, \
            f"Unexpected access node name: {dst.data}. Expected 'gpu_A' or 'gpu_A_0'."

        count += 1

    assert count == 2, \
        f"Expected exactly 2 outgoing AccessNodes from MapExit (gpu_A and gpu_A_0), but found {count}."
    
    # Run with input and verify correctness, indicating nothing else went wrong

    # Define dimensions
    N_val = 4  # or any small testable number
    A = np.zeros((N_val, N_val, 10, 64), dtype=np.float64)
    sdfg(A=A, N=N_val)

    # For z in 0, 2, 4, 6, 8 slices along axis=2,
    # the corresponding slice along last dim should be all ones
    for z in range(0, 10, 2):
        # Check all x,y slices for this z index
        assert np.all(A[:, :, z, :] == 1), f"Slice A[:,:,{z},:] not correctly set to 1"

    # For other z indices (1,3,5,7,9) the array remains zeros
    for z in range(1, 10, 2):
        assert np.all(A[:, :, z, :] == 0), f"Slice A[:,:,{z},:] changed unexpectedly"

@pytest.mark.gpu
def test_simple_nested_transient():
    """
    Kernel has one nested SDFGs inside it and one local, 
    GPU_Global array which needs to be moved out.
    """
    N = dace.symbol('N')

    @dace.program
    def simple_nested_transient(A: dace.float64[N, N, N, 64]):
        for x, y, z in dace.map[0:N, 0:N, 0:N]:
            # Create local array with the same name as an outer array, again
            gpu_A = dace.define_local([64], cp.float64, storage=dace.dtypes.StorageType.GPU_Global)
            gpu_A[:] = 0
            gpu_A[:] = 1
            A[x, y, z, :] = gpu_A


        # Get sdfg and apply the GPU transformations
        sdfg = simple_nested_transient.to_sdfg()

        # Capture the MoveArrayOutOfKernel warning
        with pytest.warns(UserWarning, match="Transient array 'gpu_A' with storage type GPU_Global"):
            sdfg.apply_gpu_transformations()


        
        # Verify that gpu_A has been moved out of the Kernel (which is done via the mapExit).
        # So get MapExit first
        mapExit = None
        mapExit_parent_state = None
        for node, parent_state in sdfg.all_nodes_recursive():
            if isinstance(node, nodes.MapExit) and node.map.schedule == dace.dtypes.ScheduleType.GPU_Device:
                mapExit = node
                mapExit_parent_state = parent_state
                break


        # We expect two outgoing edges from the MapExit:
        # 1. One to the GPU_Global version of `A` changed to `gpu_A`
        # 2. One to the previously local `gpu_A`, now changed to f"local_0_gpu_A" since
        #    it had to be renamed after moving out to the outer sdfg (renamed via the MoveOutOfKernel pass)

        # what new name will look like
        old_name = "gpu_A"
        new_name = f"local_0_{old_name}"

        count = 0
        for edge in mapExit_parent_state.out_edges(mapExit):
            dst = edge.dst
            assert isinstance(dst, nodes.AccessNode), \
                f"Expected destination node to be an AccessNode, but got {type(dst).__name__}"

            assert dst.data in {old_name , new_name}, \
                f"Unexpected access node name: {dst.data}. Expected '{old_name}' or '{new_name}'."

            count += 1

        assert count == 2, \
            f"Expected exactly 2 outgoing AccessNodes from MapExit ({old_name} and {new_name}), but found {count}."
        

        # Run with an example veriying that the sdfg compiles and returns expected result
        N_val = 4
        A = np.zeros((N_val, N_val, N_val, 64), dtype=np.float64)
        sdfg(A=A, N=N_val)

        # Assert all values in A were set to 1 by the kernel
        assert np.allclose(A, 1.0), "Expected all elements in A to be 1.0, but some are not."

@pytest.mark.gpu
def test_difficult_nested_transient():
    """
    Nested twice, same name is used at 3 different layers.
    """

    # Define the SDFG using the SDFG API

    N = dace.symbol('N')

    # SDFG and the main state
    outer_sdfg = dace.SDFG("transient5")
    state = outer_sdfg.add_state("main")

    kernel_entry, kernel_exit = state.add_map("KernelMap", dict(bx="0:N:32", by="0:N:32"), schedule=dace.dtypes.ScheduleType.GPU_Device)

    # handle first nested sdfg
    inner_sdfg1 = dace.SDFG("inner_sdfg1")
    nsdfg1 = state.add_nested_sdfg(inner_sdfg1, inputs=dict(), outputs={"tmp_middle"})
    inner_state1 = inner_sdfg1.add_state("innerState1")
    tb_entry, tb_exit = inner_state1.add_map("ThreadBlockMap", dict(x="0:32:1", y="0:32:1"), schedule=dace.dtypes.ScheduleType.GPU_ThreadBlock)


    # handle second nested sdfg
    inner_sdfg2 = dace.SDFG("inner_sdfg2")
    nsdfg2 = inner_state1.add_nested_sdfg(inner_sdfg2, inputs=dict(), outputs={"tmp_innermost"})
    inner_state2 = inner_sdfg2.add_state("innerState2")

    # innermost local gpu_a
    inner_sdfg2.add_transient("gpu_A", (1,), dace.uint32, storage=dace.dtypes.StorageType.GPU_Global)
    inner_acc2 = inner_state2.add_access("gpu_A")

    # actual stuff
    assign_tasklet = inner_state2.add_tasklet(
        "assign", inputs={}, outputs={"__out"},
        code="__out = 2;",
        language=dace.dtypes.Language.CPP
    )

    # Assign 2 to gpu_A[0]
    inner_state2.add_edge(assign_tasklet, "__out", inner_acc2, None, dace.Memlet("gpu_A[0]"))


    #tmp between innermost and middle sdfg and connect
    tmp_innermost_name, tmp_innermost_arr = inner_sdfg1.add_transient("tmp_innermost", (1,), dace.uint32, storage=dace.dtypes.StorageType.GPU_Global)
    copy_desc = copy.deepcopy(tmp_innermost_arr)
    copy_desc.transient = False
    inner_sdfg2.add_datadesc("tmp_innermost", copy_desc)
    nsdfg2.add_out_connector("tmp_innermost")
    tmp_innermost = inner_state2.add_access("tmp_innermost")
    inner_state2.add_edge(inner_acc2, None, tmp_innermost, None, dace.Memlet("tmp_innermost[0]"))

    # middle local gpu_A
    inner_sdfg1.add_transient("gpu_A", (32,32,1,), dace.uint32, storage=dace.dtypes.StorageType.GPU_Global)
    inner_acc1 = inner_state1.add_access("gpu_A")

    #tmp between middle and outer sdfg and connect
    tmp_middle_name, tmp_middle_arr = outer_sdfg.add_transient("tmp_middle", (32,32,1), dace.uint32, storage=dace.dtypes.StorageType.GPU_Global)
    copy_desc = copy.deepcopy(tmp_middle_arr)
    copy_desc.transient = False
    inner_sdfg1.add_datadesc(tmp_middle_name, copy.deepcopy(copy_desc))
    nsdfg1.add_out_connector("tmp_middle")
    tmp_middle = inner_state1.add_access("tmp_middle")
    inner_state1.add_edge(inner_acc1, None, tmp_middle, None, dace.Memlet("[0:32,0:32,0] -> tmp_middle[0:32,0:32,0]"))

    # outermost gpu_A definition, non-transient 
    outer_sdfg.add_array("gpu_A", (N,N,1), dace.uint32, storage=dace.dtypes.StorageType.GPU_Global)
    global_A_access = state.add_access("gpu_A")

    # Add edges

    # outer state
    state.add_edge(kernel_entry, None, nsdfg1, None, dace.Memlet())
    state.add_edge(nsdfg1, "tmp_middle", kernel_exit, "IN_gpu_A", dace.Memlet("gpu_A[bx:bx+32,by:by+32,0]"))
    kernel_exit.add_in_connector("IN_gpu_A")
    state.add_edge(kernel_exit, "OUT_gpu_A", global_A_access, None,  dace.Memlet("gpu_A[0:N,0:N,0]"))
    kernel_exit.add_out_connector("OUT_gpu_A")

    # inner state (1) edges
    inner_state1.add_edge(tb_entry, None, nsdfg2, None, dace.Memlet())
    inner_state1.add_edge(nsdfg2, "tmp_innermost", tb_exit, "IN_gpu_A", dace.Memlet("gpu_A[x,y,0]"))
    tb_exit.add_in_connector("IN_gpu_A")
    inner_state1.add_edge(tb_exit, "OUT_gpu_A", inner_acc1, None, dace.Memlet("gpu_A[0:32,0:32,0]"))
    tb_exit.add_out_connector("OUT_gpu_A")

    # sdfg = outer_sdfg
    sdfg = outer_sdfg

    with pytest.warns(UserWarning) as record:
        # Do not simplify
        sdfg.apply_gpu_transformations(simplify=False)


    # Verify that the two local gpu_A have been moved out of the Kernel twice (which is done via the mapExit).
    mapExit = None
    mapExit_parent_state = None
    for node, parent_state in sdfg.all_nodes_recursive():
        if isinstance(node, nodes.MapExit) and node.map.schedule == dace.dtypes.ScheduleType.GPU_Device:
            mapExit = node
            mapExit_parent_state = parent_state
            break


    # We expect two outgoing edges from the MapExit:
    # 1. One to the GPU_Global version of `A` changed to `gpu_A`
    # 2. One to the previously local `gpu_A`, now changed to f"local_0_gpu_A" since
    #    it had to be renamed after moving out to the outer sdfg (renamed via the MoveOutOfKernel pass)

    # what new name will look like
    old_name = "gpu_A"
    new_name1 = f"local_0_{old_name}"
    new_name2 = f"local_1_{old_name}"

    count = 0
    for edge in mapExit_parent_state.out_edges(mapExit):
        dst = edge.dst
        assert isinstance(dst, nodes.AccessNode), \
            f"Expected destination node to be an AccessNode, but got {type(dst).__name__}"

        assert dst.data in {old_name , new_name1, new_name2}, \
            f"Unexpected access node name: {dst.data}. Expected '{old_name}','{new_name1}' or '{new_name1}'."

        count += 1

    assert count == 3, \
        f"Expected exactly 3 outgoing AccessNodes from MapExit ({old_name}, {new_name1} and {new_name1}), but found {count}."
    

    # Run with an example veriying that the sdfg compiles and returns expected result
    N_val = 32
    A = cp.zeros((N_val, N_val, 1), dtype=cp.uint32)

    outer_sdfg(gpu_A = A, N = N_val)
    assert cp.all(A == 2), "Not all elements in A are 2"

@pytest.mark.gpu
def test_stencil_with_transient():
    """
    A kernel that performs a stencil computation using two kernel maps.
    Using the SDFG API, a randomly chosen transient array is converted into GPU_Global storage.
    The DaCe frontend automatically assigns the same name to independent transient arrays 
    in both kernels (e.g., '__tmp{counter}'), because both kernels generate the same number 
    of such temporaries due to their similar structure. As a result, the transformation 
    applies correctly across both kernels. This handles avoiding naming conflicts appropriately even when 
    the transients are not nested within each other, as long as they reside at the kernel-map level, 
    ensuring correct renaming and definition tracking. Furthermore, it demonstrates that even in a more practical
    program, applying the 'MoveArrayOutOfKernel' transformation preserves correct compilation and produces valid results.
    """

    N = dace.symbol("N", dtype=dace.int64)

    # The stencil program for GPUs
    @dace.program
    def gpu_stencil(
        TSTEPS: dace.int64,
        vals_A: dace.float64[N, N, N] @ dace.dtypes.StorageType.GPU_Global,
        vals_B: dace.float64[N, N, N] @ dace.dtypes.StorageType.GPU_Global,
        neighbors: dace.int64[N, N, 8] @ dace.dtypes.StorageType.GPU_Global,
    ):

        for _ in range(1, TSTEPS):
            for i, j, k in dace.map[0 : N - 2, 0 : N - 2, 0 : N - 2] @ dace.dtypes.ScheduleType.GPU_Device:
                vals_B[i + 1, j + 1, k + 1] = 0.2 * (
                    vals_A[i + 1, j + 1, k + 1]
                    + vals_A[i + 1, j , k + 1]
                    + vals_A[i + 1, j + 2, k + 1]
                    + vals_A[neighbors[i+1, k+1, 0], j + 1, neighbors[i+1, k+1, 4]]
                    + vals_A[neighbors[i+1, k+1, 1], j + 1, neighbors[i+1, k+1, 5]]
                    + vals_A[neighbors[i+1, k+1, 2], j + 1, neighbors[i+1, k+1, 6]]
                    + vals_A[neighbors[i+1, k+1, 3], j + 1, neighbors[i+1, k+1, 7]]
                )
            for i, j, k in dace.map[0 : N - 2, 0 : N - 2, 0 : N - 2] @ dace.dtypes.ScheduleType.GPU_Device:
                vals_A[i + 1, j + 1, k + 1] = 0.2 * (
                    vals_B[i + 1, j + 1, k + 1]
                    + vals_B[i + 1, j , k + 1]
                    + vals_B[i + 1, j + 2, k + 1]
                    + vals_B[neighbors[i+1, k+1, 0], j + 1, neighbors[i+1, k+1, 4]]
                    + vals_B[neighbors[i+1, k+1, 1], j + 1, neighbors[i+1, k+1, 5]]
                    + vals_B[neighbors[i+1, k+1, 2], j + 1, neighbors[i+1, k+1, 6]]
                    + vals_B[neighbors[i+1, k+1, 3], j + 1, neighbors[i+1, k+1, 7]]
                )

    # The stencil program for CPUs, used for verification
    @dace.program
    def cpu_stencil(
        TSTEPS: dace.int64,
        vals_A: dace.float64[N, N, N],
        vals_B: dace.float64[N, N, N],
        neighbors: dace.int64[N, N, 8],
    ):

        for _ in range(1, TSTEPS):
            for i, j, k in dace.map[0 : N - 2, 0 : N - 2, 0 : N - 2]:
                vals_B[i + 1, j + 1, k + 1] = 0.2 * (
                    vals_A[i + 1, j + 1, k + 1]
                    + vals_A[i + 1, j , k + 1]
                    + vals_A[i + 1, j + 2, k + 1]
                    + vals_A[neighbors[i+1, k+1, 0], j + 1, neighbors[i+1, k+1, 4]]
                    + vals_A[neighbors[i+1, k+1, 1], j + 1, neighbors[i+1, k+1, 5]]
                    + vals_A[neighbors[i+1, k+1, 2], j + 1, neighbors[i+1, k+1, 6]]
                    + vals_A[neighbors[i+1, k+1, 3], j + 1, neighbors[i+1, k+1, 7]]
                )
            for i, j, k in dace.map[0 : N - 2, 0 : N - 2, 0 : N - 2]:
                vals_A[i + 1, j + 1, k + 1] = 0.2 * (
                    vals_B[i + 1, j + 1, k + 1]
                    + vals_B[i + 1, j , k + 1]
                    + vals_B[i + 1, j + 2, k + 1]
                    + vals_B[neighbors[i+1, k+1, 0], j + 1, neighbors[i+1, k+1, 4]]
                    + vals_B[neighbors[i+1, k+1, 1], j + 1, neighbors[i+1, k+1, 5]]
                    + vals_B[neighbors[i+1, k+1, 2], j + 1, neighbors[i+1, k+1, 6]]
                    + vals_B[neighbors[i+1, k+1, 3], j + 1, neighbors[i+1, k+1, 7]]
                )   


    # This code identifies a transient array that appears in both kernels and changes its
    # storage to GPU_Global. This enables the 'MoveArrayOutOfKernel' transformation to apply
    # correctly. At the time of writing, the target array was named '__tmp4'.
    # Making this detection fully robust would require significantly more code, 
    # which is intentionally avoided for simplicity.
    gpu_sdfg = gpu_stencil.to_sdfg()
    array_name = None
    for node, parent in gpu_sdfg.all_nodes_recursive():
        if isinstance(node, nodes.AccessNode):
            desc = node.desc(parent)
            if desc.transient and type(desc) == dace.data.Array:
                array_name = node.data
                break

    for node, parent in gpu_sdfg.all_nodes_recursive():
        if isinstance(node, nodes.AccessNode):
            desc = node.desc(parent)
            if desc.transient and node.data == array_name:
                desc.storage = dace.dtypes.StorageType.GPU_Global


    # Check that the warning is issued for each kernel (i.e. twice, same array name)
    with pytest.warns(UserWarning) as record:
        gpu_sdfg.apply_gpu_transformations()

    # Expected warning message
    expected_msg = f"Transient array '{array_name}' with storage type GPU_Global"

    # Filter matching warnings
    matches = [w for w in record if expected_msg in str(w.message)]

    # Assert the warning appears exactly twice
    assert len(matches) == 2, (
        f"Expected the 'MoveArrayOutOfKernel' warning for array '{array_name}' to be issued twice, "
        f"but observed it {len(matches)} time(s)."
    )

    # For simplicity we verify that for each kernel exit the new names are moved out (i.e. exit -> previous transient)
    matches = 0
    old_name = array_name
    new_name = f"local_0_{old_name}"
    old_name_match_found = False
    new_name_match_found = False
    for node, parent_state in gpu_sdfg.all_nodes_recursive():
        if isinstance(node, nodes.MapExit) and node.map.schedule == dace.dtypes.ScheduleType.GPU_Device:
            
            for edge in parent_state.out_edges(node):
                dst = edge.dst
                if not isinstance(dst, nodes.AccessNode):
                    continue

                if dst.data == old_name:
                    old_name_match_found = True
                elif dst.data == new_name:
                    new_name_match_found = True


    assert old_name_match_found and new_name_match_found, (
        "Both MapExit nodes should be connected to their respective transient arrays: "
        f"one with the original name ('{old_name}') and one with the renamed version ('{new_name}')."
    )

    # initialize and run both stencils with some input -> compilation and execution works fine
    N_val = 10
    tsteps = 3
    seed = 42 

    np.random.seed(seed)

    cpu_vals_A = np.fromfunction(lambda i, j, k: i * k * (j + 2) / N_val, (N_val, N_val, N_val), dtype=np.float64)
    cpu_vals_B = np.fromfunction(lambda i, j, k: i * k * (j + 3) / N_val, (N_val, N_val, N_val), dtype=np.float64)
    cpu_neighbors = np.random.randint(1, N_val, size=(N_val, N_val, 8), dtype=np.int64)

    gpu_vals_A = cp.asarray(cpu_vals_A)
    gpu_vals_B = cp.asarray(cpu_vals_B)
    gpu_neighbors = cp.asarray(cpu_neighbors)

    gpu_sdfg(TSTEPS= tsteps, N=N_val, vals_A=gpu_vals_A, vals_B=gpu_vals_B, neighbors=gpu_neighbors)

    cpu_sdfg = cpu_stencil.to_sdfg()
    cpu_sdfg(TSTEPS= tsteps, N=N_val, vals_A=cpu_vals_A, vals_B=cpu_vals_B, neighbors=cpu_neighbors)

    # verify that gpu version matches the cpu one, demonstrating correctness
    assert np.allclose(cp.asnumpy(gpu_vals_A), cpu_vals_A), "Mismatch gpu and cpu do not compute same result"
    assert np.allclose(cp.asnumpy(gpu_vals_B), cpu_vals_B), "Mismatch gpu and cpu do not compute same result"
    assert np.allclose(cp.asnumpy(gpu_neighbors), cpu_neighbors), "Mismatch gpu and cpu do not compute same result"
