from backend.systolic import systolic_os_verify_numpy_equivalence as vnp
from backend.systolic import systolic_os_verify_linked_to_sim as vlnk

def _case(M,K,L,tm,tk,tl,dtype):
    assert vnp(M=M,K=K,L=L, m_block=tm,k_block=tk,l_block=tl, dtype=dtype, verbose=False)
    assert vlnk(M=M,K=K,L=L, m_block=tm,k_block=tk,l_block=tl, dtype=dtype,
                yaml_path="config/upmem.yaml", run_timing_sim=False, verbose=False)

def test_small_int():
    _case(8,8,8, 4,4,4, "int16")

def test_ragged_int():
    _case(33,17,29, 8,5,7, "int16")

def test_float():
    _case(32,48,24, 7,5,6, "float32")
