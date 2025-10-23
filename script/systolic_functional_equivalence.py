#!/usr/bin/env python3
# scripts/function_equivelance.py
import argparse, numpy as np

def systolic_os_matmul(A: np.ndarray, B: np.ndarray, tile_m: int, tile_k: int, tile_l: int):
    M,K = A.shape; K2,L = B.shape; assert K == K2
    acc_dtype = np.int64 if np.issubdtype(A.dtype,np.integer) and np.issubdtype(B.dtype,np.integer) else np.float64
    C = np.zeros((M, L), dtype=acc_dtype)
    for l0 in range(0, L, tile_l):
        tl = min(tile_l, L - l0)
        for m0 in range(0, M, tile_m):
            tm = min(tile_m, M - m0)
            C_tile = np.zeros((tm, tl), dtype=acc_dtype)
            for k0 in range(0, K, tile_k):
                tk = min(tile_k, K - k0)
                A_sub = A[m0:m0+tm, k0:k0+tk].astype(acc_dtype, copy=False)
                B_sub = B[k0:k0+tk, l0:l0+tl].astype(acc_dtype, copy=False)
                C_tile += A_sub @ B_sub
            C[m0:m0+tm, l0:l0+tl] = C_tile
    return C

def gen(rng, shape, dtype):
    return (rng.integers(-8,8,shape,dtype=dtype)
            if np.issubdtype(dtype,np.integer)
            else rng.standard_normal(shape).astype(dtype))

def check(A,B,tm,tk,tl):
    C_os = systolic_os_matmul(A,B,tm,tk,tl)
    if np.issubdtype(A.dtype,np.integer) and np.issubdtype(B.dtype,np.integer):
        C_ref = A.astype(np.int64) @ B.astype(np.int64)
        return np.array_equal(C_os, C_ref)
    C_ref = A.astype(np.float64) @ B.astype(np.float64)
    return np.allclose(C_os, C_ref, rtol=1e-5, atol=1e-8)

def main():
    rng = np.random.default_rng(0xC0FFEE)
    cases = [(8,8,8),(17,13,31),(3,29,7),(64,3,5),(5,64,9),(33,33,33),(1,1,1)]
    dtypes = [np.float32,np.float64,np.int8,np.int16]
    tile_sets = [(4,4,4),(8,8,8),(16,8,4),(7,5,3)]
    total=0; ok=0
    for (M,K,L) in cases:
        for da in dtypes:
            for db in dtypes:
                if (np.issubdtype(da,np.integer) != np.issubdtype(db,np.integer)): continue
                A = gen(rng,(M,K),da); B = gen(rng,(K,L),db)
                for (tm,tk,tl) in tile_sets:
                    total += 1
                    ok += check(A,B,tm,tk,tl)
    if ok == total:
        print(f"[OK] Functional OS-systolic equivalence: {ok}/{total} tests passed.")
    else:
        print(f"[FAIL] {ok}/{total} passed")
        raise SystemExit(1)

if __name__ == "__main__":
    main()
