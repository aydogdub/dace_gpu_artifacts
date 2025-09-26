import dace
import os
import re
import pytest

#-------------------- Helper functions for finding function arguments in signature ---------------------------

def parse_functions(code):
    pattern = re.compile(
        r'(?:[\w:\*\s]+)\s+(\w+)\s*\(([^)]*)\)\s*(?:;|{)', re.MULTILINE)
    return pattern.findall(code)

def parse_args(arg_str):
    args = [arg.strip() for arg in arg_str.split(',') if arg.strip()]
    parsed_args = []
    for arg in args:
        tokens = arg.split()
        if not tokens:
            continue
        name = tokens[-1]
        type_tokens = tokens[:-1]

        if '*' in name:
            star_parts = name.split('*')
            if star_parts[0] == '':
                name = star_parts[1]
                type_tokens.append('*')

        parsed_args.append((' '.join(type_tokens), name))
    return parsed_args

def is_const(type_str):
    return 'const' in type_str.split()

def check_const_qualifiers_in_code(code, expected):
    funcs = parse_functions(code)
    for func_name, args_str in funcs:
        # We don't care about the init and exit function, more about the kernel and its
        # helper functions
        if func_name.startswith('__dace_init') or func_name.startswith('__dace_exit'):
            continue

        args = parse_args(args_str)
        for arg_type, arg_name in args:
            if arg_name in expected:
                actual = is_const(arg_type)
                expected_const = expected[arg_name]
                if actual != expected_const:
                    msg = (f"Const mismatch in function '{func_name}': argument '{arg_name}' expected const={expected_const}, got const={actual}")
                    return False, msg
    return True, ''

@pytest.mark.gpu
def test_const_args_and_symbols():
    """
    This test requires that both the input arrays and the symbols, that do not get changed in the kernel scope,
    to be marked as constant.
    
    Note: The legacy CUDACodeGen correctly applies const qualifiers to constant arrays, 
    but it does not handle constant symbols properly. The experimental implementation gets it right.
    """

    # Used symbols, both will be const
    N = dace.symbol('N')
    K = dace.symbol('K')

    @dace.program
    def const_args_and_symbols(
        A: dace.float64[N] @ dace.dtypes.StorageType.GPU_Global,
        B: dace.float64[N] @ dace.dtypes.StorageType.GPU_Global,
        C: dace.float64[N] @ dace.dtypes.StorageType.GPU_Global,
    ):
        for i in dace.map[0:N:256 * K] @ dace.dtypes.ScheduleType.GPU_Device:
            for k in dace.map[0:K] @ dace.dtypes.ScheduleType.Sequential:
                for j in dace.map[0:256] @ dace.dtypes.ScheduleType.GPU_ThreadBlock:
                    C[i + j + k * 256] = A[i + j + k * 256] + B[i + j + k * 256]


    sdfg = const_args_and_symbols.to_sdfg()

    # Which Arrays/Symbols are constant
    expected_qualifiers = {
        'A': True,  # A should be const
        'B': True,   # B should be const
        'C': False,  # C should NOT be const
        'N': True,    # N should be const,
        'K': True    # N should be const
    }

    code = sdfg.generate_code()[1].clean_code

    check, msg = check_const_qualifiers_in_code(code, expected_qualifiers)

    assert check, f"FAILURE: {msg}"

@pytest.mark.gpu
def test_const_qualifier_handling_in_codegen():
    """
    This test checks a pointless programs constant checks.
    Althought pointless, the SDFG is still valid.

    Note: Legacy CUDACodeGen does not only get the constants wrong,
    it also does not compile since it marks non constants as constants
    and uses them in a copy function template which expects a non constant.
    """

    current_dir = os.path.dirname(os.path.abspath(__file__))
    sdfg_path = os.path.join(current_dir, "../../scratch/yakups_examples/smem_related/weird_global_to_global.sdfg")
    sdfg = dace.SDFG.from_file(sdfg_path)

    # Which Arrays/Symbols are constant
    expected_qualifiers = {
        'A': False,  # A should NOT be const
        'B': False,   # B should NOT be const
        'C': False,  # C should NOT be const
        'N': True    # N should be const
    }

    code = sdfg.generate_code()[1].clean_code

    check, msg = check_const_qualifiers_in_code(code, expected_qualifiers)

    assert check, f"FAILURE: {msg}"