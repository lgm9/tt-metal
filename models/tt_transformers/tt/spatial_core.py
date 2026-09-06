"""
spatial_core.py
---------------
Thread-local spatial partitioning for Tenstorrent inference.

Patches ttnn.CoreCoord, ttnn.CoreRange, and ttnn.CoreGrid at import time so
that every core-coordinate construction automatically applies the calling
thread's y_offset.  No changes required in model_config, attention, mlp, etc.

Usage
-----
    from spatial_core import set_spatial_partition, install_patches

    install_patches()   # call once before loading any model

    # In thread 0 (top half, rows 0-4):
    set_spatial_partition(y_offset=0, max_y=5)
    model0 = create_tt_model(...)

    # In thread 1 (bottom half, rows 5-9):
    set_spatial_partition(y_offset=5, max_y=5)
    model1 = create_tt_model(...)

How it works
------------
The patches intercept EVERY call to the three constructors:

  ttnn.CoreCoord(x, y)         → CoreCoord(x, y + y_offset)
  ttnn.CoreRange(start, end)   → CoreRange shifted (start/end already shifted
                                  because they were built via CoreCoord)
  ttnn.CoreGrid(x=X, y=Y)     → CoreGrid(x=X, y=min(Y, max_y))
  (X, Y) tuple grids           → (X, min(Y, max_y))   [compute_with_storage…]

The CoreRange patch is a no-op because its arguments are already-shifted
CoreCoord objects; it is included only to intercept any (x0,y0,x1,y1) form.

Thread safety
-------------
All state is in threading.local(), so top-half and bottom-half threads never
interfere with each other. Full-chip code (y_offset=0, max_y=10) is the
default and is a zero-cost passthrough.
"""

from __future__ import annotations

import ctypes
import threading
from typing import Any

import ttnn

# ── C++ thread-local compute-grid override ─────────────────────────────────────
# tt_set_spatial_compute_grid / tt_clear_spatial_compute_grid are exported from
# libtt_metal.so via extern "C" in mesh_device.cpp.  Calling them makes every
# C++ op that calls device->compute_with_storage_grid_size() in the current
# thread see the spatially-partitioned grid size instead of the full hardware
# grid.  We load libtt_metal.so with RTLD_GLOBAL so its symbols join the global
# namespace (the extension loads it RTLD_LOCAL by default).
def _find_libtt_metal() -> str:
    """Return the absolute path to libtt_metal.so loaded by _ttnn.so."""
    import subprocess
    import os
    try:
        # _ttnn.so depends on libtt_metal.so; ldd tells us the resolved path.
        ttnn_so = ttnn._ttnn.__file__
        out = subprocess.check_output(['ldd', ttnn_so], stderr=subprocess.DEVNULL).decode()
        for line in out.splitlines():
            if 'libtt_metal.so' in line and '=>' in line:
                parts = line.split('=>')
                if len(parts) == 2:
                    path = parts[1].strip().split()[0]
                    if os.path.isfile(path):
                        return path
    except Exception:
        pass
    return ''


def _init_cpp_grid_fns():
    try:
        import os
        lib_path = _find_libtt_metal()
        if not lib_path:
            return None, None
        # Loading with RTLD_GLOBAL promotes the already-loaded library's symbols
        # into the process-wide namespace; no double-loading occurs.
        _lib = ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
        set_fn = _lib.tt_set_spatial_compute_grid
        set_fn.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
        set_fn.restype = None
        clr_fn = _lib.tt_clear_spatial_compute_grid
        clr_fn.argtypes = []
        clr_fn.restype = None
        # v2: also carries y_start so DRAM reader assignment is offset-aware.
        try:
            set_fn_v2 = _lib.tt_set_spatial_compute_grid_v2
            set_fn_v2.argtypes = [ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32]
            set_fn_v2.restype = None
        except AttributeError:
            set_fn_v2 = None
        return set_fn, clr_fn, set_fn_v2
    except AttributeError:
        return None, None, None

_cpp_set_grid, _cpp_clear_grid, _cpp_set_grid_v2 = None, None, None
_cpp_grid_fns_tried = False

def _ensure_cpp_grid_fns():
    global _cpp_set_grid, _cpp_clear_grid, _cpp_set_grid_v2, _cpp_grid_fns_tried
    if not _cpp_grid_fns_tried:
        _cpp_grid_fns_tried = True
        _cpp_set_grid, _cpp_clear_grid, _cpp_set_grid_v2 = _init_cpp_grid_fns()

# ── Thread-local state ─────────────────────────────────────────────────────────

_state = threading.local()


def set_spatial_partition(y_offset: int = 0, max_y: int = 10,
                          x_offset: int = 0, max_x: int = 13) -> None:
    """
    Call from a worker thread before building or running a model instance.
    y_offset / max_y : row-based partitioning (original)
    x_offset / max_x : column-based partitioning (new for 10×4 side-by-side)
    """
    _state.y_offset = y_offset
    _state.max_y    = max_y
    _state.x_offset = x_offset
    _state.max_x    = max_x
    # Also override the C++ thread-local so that program factories calling
    # device->compute_with_storage_grid_size() respect the spatial partition.
    _ensure_cpp_grid_fns()
    # Physical compute_with_storage_grid_size() on Blackhole returns (13,10)
    # (full chip including dispatch columns 8-12). Calling _cpp_clear_grid()
    # would expose those dispatch columns to program compilation, causing
    # "Kernel group cores do not match sub device cores" when sub-devices only
    # cover columns 0-7. Always explicitly restrict to (min(max_x,8), max_y).
    if _cpp_set_grid_v2 is not None:
        _cpp_set_grid_v2(min(max_x, 8), max_y, y_offset)
    elif _cpp_set_grid is not None:
        _cpp_set_grid(min(max_x, 8), max_y)


def clear_spatial_partition() -> None:
    """Reset to full-chip defaults (no-op passthrough)."""
    _state.y_offset = 0
    _state.max_y    = 10
    _state.x_offset = 0
    _state.max_x    = 13
    _ensure_cpp_grid_fns()
    if _cpp_clear_grid is not None:
        _cpp_clear_grid()


def _y_offset() -> int:
    return getattr(_state, 'y_offset', 0)


def _max_y() -> int:
    return getattr(_state, 'max_y', 10)


def _x_offset() -> int:
    return getattr(_state, 'x_offset', 0)


def _max_x() -> int:
    return getattr(_state, 'max_x', 13)


# ── Patch installation ─────────────────────────────────────────────────────────

_installed = False


def install_patches() -> None:
    """
    Monkey-patch ttnn.CoreCoord and ttnn.CoreGrid to be spatial-partition-aware.
    Idempotent — safe to call multiple times.
    """
    global _installed
    if _installed:
        return
    _installed = True

    # ── CoreCoord ──────────────────────────────────────────────────────────────
    _orig_CoreCoord = ttnn.CoreCoord

    class _SpatialCoreCoord:
        """
        Wraps ttnn.CoreCoord to shift both x and y by the calling thread's offsets.
        Behaves like the original CoreCoord (passes isinstance checks via
        delegation, exposes .x / .y).
        """
        __slots__ = ('_inner',)

        def __init__(self, x: int, y: int):
            self._inner = _orig_CoreCoord(x + _x_offset(), y + _y_offset())

        def __getattr__(self, name: str) -> Any:
            return getattr(self._inner, name)

        def __repr__(self) -> str:
            return repr(self._inner)

        @property
        def x(self) -> int:
            return self._inner.x

        @property
        def y(self) -> int:
            return self._inner.y

    ttnn.CoreCoord = _SpatialCoreCoord

    # ── CoreGrid ───────────────────────────────────────────────────────────────
    _orig_CoreGrid = ttnn.CoreGrid

    class _SpatialCoreGridMeta(type):
        # ttnn internals do isinstance(x, ttnn.CoreGrid) — must still work
        # after we replace ttnn.CoreGrid with this class.
        def __instancecheck__(cls, instance):
            return isinstance(instance, _orig_CoreGrid)

    class _SpatialCoreGrid(metaclass=_SpatialCoreGridMeta):
        """
        Grid patch for spatial partitioning.

        Row-partitioned (default, max_x=13):
            Clamp y to max_y; x is unchanged.

        Column-partitioned (max_x < 13, e.g. max_x=4):
            When x > max_x, transpose to keep total cores constant:
              CoreGrid(y=4, x=8) → CoreGrid(y=8, x=4)  for max_x=4
            When x <= max_x, y-clamp only (same as row-partitioned).
        """
        def __new__(cls, *, x: int, y: int) -> Any:
            mx = _max_x()
            if mx < 13 and x > mx:
                # column-partitioned: transpose to maintain total core count
                new_x = mx
                new_y = y * (x // mx)
            else:
                new_x = x
                new_y = min(y, _max_y())
            return _orig_CoreGrid(x=new_x, y=new_y)

    ttnn.CoreGrid = _SpatialCoreGrid

    # ── CoreRange ─────────────────────────────────────────────────────────────
    # CoreRange takes two CoreCoord objects; because those are already shifted
    # by the patch above, CoreRange itself needs no changes.  We patch only
    # the (start_x, start_y, end_x, end_y) 4-tuple shorthand if it exists.
    if hasattr(ttnn, 'CoreRange'):
        _orig_CoreRange = ttnn.CoreRange
        class _SpatialCoreRange:
            __slots__ = ('_inner',)

            def __new__(cls, *args, **kwargs):
                # Pass-through: args are already-patched CoreCoord objects
                # OR raw (x0,y0,x1,y1) integers (rare, shift them too)
                if len(args) == 4 and all(isinstance(a, int) for a in args):
                    xoff, yoff = _x_offset(), _y_offset()
                    x0, y0, x1, y1 = args
                    inner = _orig_CoreRange(
                        _orig_CoreCoord(x0 + xoff, y0 + yoff),
                        _orig_CoreCoord(x1 + xoff, y1 + yoff),
                    )
                else:
                    # Unwrap _SpatialCoreCoord → real CoreCoord: new pybind11
                    # binding requires exact ttnn.CoreCoord, not duck-typed wrappers
                    unwrapped = tuple(
                        a._inner if isinstance(a, _SpatialCoreCoord) else a
                        for a in args
                    )
                    inner = _orig_CoreRange(*unwrapped, **kwargs)
                obj = object.__new__(cls)
                obj._inner = inner
                return obj

            def __getattr__(self, name: str) -> Any:
                return getattr(self._inner, name)

            def __repr__(self) -> str:
                return repr(self._inner)

            def __hash__(self):
                return hash(self._inner)

            def __eq__(self, other):
                inner = other._inner if isinstance(other, _SpatialCoreRange) else other
                return self._inner == inner

        ttnn.CoreRange = _SpatialCoreRange

        # CoreRangeSet patch: unwrap _SpatialCoreRange → _inner before handing
        # the set to pybind11, which requires actual ttnn.CoreRange elements.
        if hasattr(ttnn, 'CoreRangeSet'):
            _orig_CoreRangeSet = ttnn.CoreRangeSet

            class _SpatialCoreRangeSetMeta(type):
                def __instancecheck__(cls, instance):
                    return isinstance(instance, _orig_CoreRangeSet)

            class _SpatialCoreRangeSet(metaclass=_SpatialCoreRangeSetMeta):
                def __new__(cls, core_ranges, *args, **kwargs):
                    # Unwrap _SpatialCoreRange wrappers for any iterable
                    if isinstance(core_ranges, (set, frozenset, list, tuple)):
                        unwrap = lambda r: r._inner if isinstance(r, _SpatialCoreRange) else r
                        if isinstance(core_ranges, (set, frozenset)):
                            unwrapped = {unwrap(r) for r in core_ranges}
                        else:
                            unwrapped = [unwrap(r) for r in core_ranges]
                    else:
                        unwrapped = core_ranges
                    return _orig_CoreRangeSet(unwrapped, *args, **kwargs)

            ttnn.CoreRangeSet = _SpatialCoreRangeSet

    # ── MinimalMatmulConfig (experimental; prefill QKV and large-seq matmuls) ──
    # Its nanobind __init__ requires a real ttnn.CoreCoord for
    # compute_with_storage_grid_size and rejects the _SpatialCoreCoord wrapper
    # ("TypeError: incompatible function arguments"), which made every prefill
    # with seq_len > 128 fail under install_patches().  model_config also asks
    # for the full 8x10 Blackhole grid here, so clamp the height to the partition.
    # NOTE: minimal_matmul_program_factory.cpp does not read
    # tt_get_spatial_compute_grid_y_start(), so the op always lands on rows
    # 0..y-1.  That is correct only for the top partition (y_offset == 0);
    # making it partition-aware is a C++ change (rebuild).
    if hasattr(ttnn, "MinimalMatmulConfig"):
        _orig_MinimalMatmulConfig = ttnn.MinimalMatmulConfig

        def _spatial_minimal_matmul_config(*args, **kwargs):
            g = kwargs.get("compute_with_storage_grid_size")
            if g is not None:
                if isinstance(g, (tuple, list)):
                    gx, gy = int(g[0]), int(g[1])
                elif isinstance(g, _SpatialCoreCoord):
                    gx, gy = int(g.x) - _x_offset(), int(g.y) - _y_offset()   # undo the wrapper's shift
                else:
                    gx, gy = int(g.x), int(g.y)
                gx = max(1, min(gx, min(_max_x(), 8)))
                gy = max(1, min(gy, _max_y()))
                kwargs["compute_with_storage_grid_size"] = _orig_CoreCoord(gx, gy)
            return _orig_MinimalMatmulConfig(*args, **kwargs)

        ttnn.MinimalMatmulConfig = _spatial_minimal_matmul_config

    # ── compute_with_storage_grid_size tuple form ──────────────────────────────
    # Some ops receive compute_with_storage_grid_size as a plain (x, y) tuple.
    # We can't patch that generically here, but SpatialDevice already returns
    # the clamped value from .compute_with_storage_grid_size(), so model_config
    # picks up the right y automatically.  The remaining hardcoded (8,8) tuples
    # in attention/mlp will use _clamp_grid_tuple() below.


def clamp_grid_tuple(x: int, y: int):
    """
    Use this instead of a bare (8, 8) literal wherever a
    compute_with_storage_grid_size=(x, y) tuple is hardcoded.
    Returns (cx, p2) where:
      cx = min(x, max_x)   (clamp columns to partition width)
      p2 = largest power-of-2 ≤ min(y, max_y)  (clamp rows, power-of-2 required
           by rms_norm/SDPA all-reduce communication patterns)

    Replace:
        compute_with_storage_grid_size=(8, 8)
    With:
        compute_with_storage_grid_size=clamp_grid_tuple(8, 8)
    """
    cx = min(x, _max_x())
    clamped = min(y, _max_y())
    p2 = 1
    while p2 * 2 <= clamped:
        p2 *= 2
    return (cx, p2)
