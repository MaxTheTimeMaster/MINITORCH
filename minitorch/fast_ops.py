from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numba import njit, prange

from .tensor_data import (
    MAX_DIMS,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

if TYPE_CHECKING:
    from typing import Callable, Optional

    from .tensor import Tensor
    from .tensor_data import Index, Shape, Storage, Strides

# TIP: Use `NUMBA_DISABLE_JIT=1 pytest tests/ -m task3_1` to run these tests without JIT.

# This code will JIT compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.
to_index = njit(inline="always")(to_index)
index_to_position = njit(inline="always")(index_to_position)
broadcast_index = njit(inline="always")(broadcast_index)


class FastOps(TensorOps):
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        "See `tensor_ops.py`"

        # This line JIT compiles your tensor_map
        f = tensor_map(njit()(fn))

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)
            f(*out.tuple(), *a.tuple())
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        "See `tensor_ops.py`"

        f = tensor_zip(njit()(fn))

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            f(*out.tuple(), *a.tuple(), *b.tuple())
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        "See `tensor_ops.py`"
        f = tensor_reduce(njit()(fn))

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = 1

            # Other values when not sum.
            out = a.zeros(tuple(out_shape))
            out._tensor._storage[:] = start

            f(*out.tuple(), *a.tuple(), dim)
            return out

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """
        Batched tensor matrix multiply ::

            for n:
              for i:
                for j:
                  for k:
                    out[n, i, j] += a[n, i, k] * b[n, k, j]

        Where n indicates an optional broadcasted batched dimension.

        Should work for tensor shapes of 3 dims ::

            assert a.shape[-1] == b.shape[-2]

        Args:
            a : tensor data a
            b : tensor data b

        Returns:
            New tensor data
        """
        # print("YES")
        # Make these always be a 3 dimensional multiply
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        tensor_matrix_multiply(*out.tuple(), *a.tuple(), *b.tuple())

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implementations


def tensor_map(
    fn: Callable[[float], float]
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """
    NUMBA low_level tensor_map function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * When `out` and `in` are stride-aligned, avoid indexing

    Args:
        fn: function mappings floats-to-floats to apply.

    Returns:
        Tensor map function.
    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        # TODO: Implement for Task 3.1.
  

        out_size = 1
        for d in out_shape:
            out_size *= d

        for l in prange(out_size):
            out_index = np.empty(len(out_shape), dtype=np.int64)
            in_index = np.empty(len(in_shape), dtype=np.int64)
            to_index(l, out_shape, out_index)
            broadcast_index(out_index, out_shape, in_shape, in_index)
            out[index_to_position(out_index, out_strides)] = fn(in_storage[index_to_position(in_index, in_strides)])

    return njit(parallel=True)(_map)  # type: ignore


def tensor_zip(
    fn: Callable[[float, float], float]
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """
    NUMBA higher-order tensor zip function. See `tensor_ops.py` for description.


    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * When `out`, `a`, `b` are stride-aligned, avoid indexing

    Args:
        fn: function maps two floats to float to apply.

    Returns:
        Tensor zip function.
    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        # TODO: Implement for Task 3.1.
        
   

        out_size = 1
        for d in out_shape:
            out_size *= d

        #print(out_size)

        for l in prange(out_size):
            out_index = np.empty(len(out_shape), dtype=np.int64)
            in_index_a = np.empty(len(a_shape), dtype=np.int64)
            in_index_b = np.empty(len(b_shape), dtype=np.int64) 
            to_index(l, out_shape, out_index)
            # print(out_index)
            broadcast_index(out_index, out_shape, a_shape, in_index_a)
            broadcast_index(out_index, out_shape, b_shape, in_index_b)

            #print(out_index)
            #print(in_index_a, "1")
            #print(in_index_b, "2")
            #print(a_shape)
            #print(b_shape)

            out[index_to_position(out_index, out_strides)] = fn(a_storage[index_to_position(in_index_a, a_strides)], b_storage[index_to_position(in_index_b, b_strides)])


    return njit(parallel=True)(_zip)  # type: ignore


def tensor_reduce(
    fn: Callable[[float, float], float]
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """
    NUMBA higher-order tensor reduce function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * Inner-loop should not call any functions or write non-local variables

    Args:
        fn: reduction function mapping two floats to float.

    Returns:
        Tensor reduce function
    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
    ) -> None:
        # TODO: Implement for Task 3.1.
        
        out_size = 1
        for d in out_shape:
            out_size *= d

        for l in prange(out_size):
            out_index = np.empty(len(out_shape), dtype=np.int64)
            to_index(l, out_shape, out_index)
            a_index = out_index.copy()


            result = a_storage[index_to_position(a_index, a_strides)]
            for i in range(1, a_shape[reduce_dim]):
                a_index[reduce_dim] = i
                result = fn(result, a_storage[index_to_position(a_index, a_strides)])
            
            out[index_to_position(out_index, out_strides)] = result
        

    return njit(parallel=True)(_reduce)  # type: ignore


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """
    NUMBA tensor matrix multiply function.

    Should work for any tensor shapes that broadcast as long as

    ```
    assert a_shape[-1] == b_shape[-2]
    ```

    Optimizations:

    * Outer loop in parallel
    * No index buffers or function calls
    * Inner loop should have no global writes, 1 multiply.


    Args:
        out (Storage): storage for `out` tensor
        out_shape (Shape): shape for `out` tensor
        out_strides (Strides): strides for `out` tensor
        a_storage (Storage): storage for `a` tensor
        a_shape (Shape): shape for `a` tensor
        a_strides (Strides): strides for `a` tensor
        b_storage (Storage): storage for `b` tensor
        b_shape (Shape): shape for `b` tensor
        b_strides (Strides): strides for `b` tensor

    Returns:
        None : Fills in `out`
    """
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0

    assert len(a_shape) == len(b_shape), "MISMATCHED DIMENSIONS"

    out_size = 1
    for d in out_shape:
        out_size *= d


    # print(out_size)

    ss = len(a_shape)

    for l in prange(out_size):
        """out_index = np.empty(len(out_shape))
        out_index = to_index(l, out_shape, out_index)


        in_index_left = out_index.copy()
        in_index_left = in_index_left[:len(in_index_left) - 1]

        in_index_right = np.empty(len(out_index) - 1)
        in_index_right[:l - 2] = out_index[:len(out_index) - 2].copy()
        in_index_right[l - 2] = out_index[l - 2]
        in_index_right[l - 1] = out_index[l - 1]"""

        result = 0

        C = out_shape[-1]
        # B = a_shape[-1]
        # A = out_shape[-2]

        # print(A, B, C, "SHAPES")
        # print(a_strides, b_strides, "STRIDES")
        # print(out_shape, "OUT_SHAPE")
        # print(a_shape, "A_SHAPE", b_shape, "B_SHAPE")

        # last_out_index = l % C
        # pred_last_out_index = ((l - last_out_index) // C) % A
        
        # right_index = ((l - last_out_index - pred_last_out_index * C) // A) * B + last_out_index
        # left_index = ((l - last_out_index - pred_last_out_index * C) // C) * B + pred_last_out_index * B
        right_index = 0
        left_index = 0

        last_index_value = l % C
        left_sum = l

        for i in range(len(a_shape)):


            if i == 0:
                if b_shape[ss - i - 1] == 1:
                    right_index += 0 # BROADCAST
                else:
                    right_index += b_strides[ss - i - 1] * last_index_value
                
                next_pointer = ((left_sum - last_index_value) // out_shape[ss - i - 1]) % out_shape[ss - i - 2]
                left_sum = (left_sum - last_index_value) // out_shape[ss - i - 1]
                last_index_value = next_pointer
                continue
            
            if i == 1:
                if a_shape[ss - i - 1] == 1:
                    left_index += 0 # BROADCAST
                else:
                    left_index += a_strides[ss - i - 1] * last_index_value
                
                next_pointer = ((left_sum - last_index_value) // out_shape[ss - i - 1]) % out_shape[ss - i - 2]
                left_sum = (left_sum - last_index_value) // out_shape[ss - i - 1]
                last_index_value = next_pointer
                continue
            

            # print(a_shape[ss - i - 1], b_shape[ss - i - 1], "SHAPES")
            
            if a_shape[ss - i - 1] == 1 and b_shape[ss - i - 1] == 1:
                continue

            if a_shape[ss - i - 1] == 1 and b_shape[ss - i - 1] != 1:
                left_index += 0
                right_index += b_strides[ss - i - 1] * last_index_value

            if a_shape[ss - i - 1] != 1 and b_shape[ss - i - 1] == 1:
                right_index += 0
                left_index += a_strides[ss - i - 1] * last_index_value

            next_pointer = ((left_sum - last_index_value) // out_shape[ss - i - 1]) % out_shape[ss - i - 2]
            left_sum = (left_sum - last_index_value) // out_shape[ss - i - 1]
            last_index_value = next_pointer


            #curr_stride_left *= a_shape[ss - i - 1]
            #curr_stride_right *= b_shape[ss - i - 1]

            


        # print(left_index, right_index)
            

        for _ in range(a_shape[-1]):
            # print(left_index, right_index)
            result += a_storage[int(left_index)] * b_storage[int(right_index)]
            left_index += a_strides[-1]
            right_index += b_strides[-2]
        

        '''rl = l * B
        li = rl % C 
        pli = ((rl - li) // C) % B
        p2li = (((rl - li) // C) - pli * B) % A

        left_index_old = (rl - li - pli ) // C
        right_index_old = ((rl - li - pli - p2li) // A) + li

        for _ in range(a_shape[-1]):
            # print(left_index, right_index)
            result += a_storage[left_index_old] * b_storage[right_index_old]
            left_index_old += a_strides[-1]
            right_index_old += b_strides[-2]'''

        out[l] = result


tensor_matrix_multiply = njit(parallel=True, fastmath=True)(_tensor_matrix_multiply)
