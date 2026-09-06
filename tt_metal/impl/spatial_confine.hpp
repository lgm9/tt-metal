// SPDX-FileCopyrightText: © 2026 SwiftNPU
// SPDX-License-Identifier: Apache-2.0
#pragma once

// Central spatial confinement (SwiftNPU mechanism, 09_dual_run_rebuild step 1).
//
// When a spatial partition with y_start > 0 is active, a core spec expressed in partition-relative
// rows [0, y) is shifted by y_start so the program lands inside the partition.  Specs already inside
// [y_start, y_start + y) are left alone (factories the mechanism made partition-aware shift
// themselves).  The two cases are only distinguishable when the intervals do not overlap, i.e.
// y_start >= y (a bottom partition no taller than the top: 5+5, 6+4, 7+3); otherwise nothing is
// shifted.  A CoreRangeSet is shifted only when every range is relative.  Used by CreateKernel /
// CreateKernelFromString / CreateCircularBuffer / CreateSemaphore (tt_metal.cpp) and by the
// ProgramDescriptor constructor (program.cpp), which adds CBs and semaphores without those calls.
// Ops whose kernels carry other cores' NoC coordinates in runtime args (in-kernel multicast) must
// place their grid absolutely in the factory instead (minimal_matmul, the matmul factories).

#include <cstdint>
#include <vector>

#include <tt-metalium/core_coord.hpp>

extern "C" uint32_t tt_get_spatial_compute_grid_y();
extern "C" uint32_t tt_get_spatial_compute_grid_y_start();

namespace tt::tt_metal {

inline bool spatial_shift_active(uint32_t& y0, uint32_t& y) {
    y = tt_get_spatial_compute_grid_y();
    if (y == 0) {
        return false;
    }
    y0 = tt_get_spatial_compute_grid_y_start();
    return y0 >= y;
}

inline CoreRangeSet spatial_confine(const CoreRangeSet& crs) {
    uint32_t y0 = 0, y = 0;
    if (!spatial_shift_active(y0, y)) {
        return crs;
    }
    for (const auto& r : crs.ranges()) {
        if (r.end_coord.y >= y) {
            return crs;
        }
    }
    std::vector<CoreRange> shifted;
    shifted.reserve(crs.ranges().size());
    for (const auto& r : crs.ranges()) {
        shifted.emplace_back(
            CoreCoord(r.start_coord.x, r.start_coord.y + y0), CoreCoord(r.end_coord.x, r.end_coord.y + y0));
    }
    return CoreRangeSet(std::move(shifted));
}

}  // namespace tt::tt_metal
