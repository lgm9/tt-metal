# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os

from models.common.utility_functions import is_blackhole


def is_blackhole_p100(device):
    # 69: this identifies the BOARD by counting cores -- it compares the product against 130, which
    # is the full 13x10 P150 grid.  Under a sub-device band compute_with_storage_grid_size() reports
    # the BAND, the product is never 130, and the model silently reconfigures itself onto the P100
    # path.  69 measured the consequence: every band fails, and the L1 clash at `first conv` is
    # byte-identical at 64 cores (8x8) and 65 cores (13x5) -- so it is the code path, not the
    # geometry.  There is no error and no log line; the tenant just becomes a different model.
    #
    # With SPATIAL_RESNET_TRUE_BOARD=1 the check reads the UNPARTITIONED grid instead, so a band no
    # longer changes which model is built.  Default off: the shipped path is byte-for-byte what 69
    # recorded, and the finding stands as written.
    if os.environ.get("SPATIAL_RESNET_TRUE_BOARD") == "1":
        try:
            from models.tt_transformers.tt.spatial_core import (
                clear_spatial_partition,
                set_spatial_partition,
                _y_offset,
                _max_y,
                _x_offset,
                _max_x,
            )

            y0, my, x0, mx = _y_offset(), _max_y(), _x_offset(), _max_x()
            clear_spatial_partition()  # drop the override entirely; read the real hardware grid
            grid = device.compute_with_storage_grid_size()
            set_spatial_partition(y0, my, x0, mx)  # restore the band
            return is_blackhole() and grid.x * grid.y != 130
        except Exception:
            pass  # fall through to the stock check rather than guess

    is_p100 = (
        is_blackhole() and device.compute_with_storage_grid_size().x * device.compute_with_storage_grid_size().y != 130
    )
    return is_p100
