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
#include <cstdlib>
#include <vector>

#include <tt-metalium/core_coord.hpp>

extern "C" uint32_t tt_get_spatial_compute_grid_y();
extern "C" uint32_t tt_get_spatial_compute_grid_y_start();

namespace tt::tt_metal {

inline bool spatial_shift_active(uint32_t& y0, uint32_t& y, bool& forced) {
    forced = false;
    y = tt_get_spatial_compute_grid_y();
    if (y == 0) {
        return false;
    }
    y0 = tt_get_spatial_compute_grid_y_start();
    if (y0 >= y) {
        return true;
    }
    // 59: y0 >= y is an AMBIGUITY guard, not a capacity limit, and it is the whole row-axis wall.  A bottom
    // partition taller than its own offset (prefill on rows 2-9, decode on 0-1) cannot be told apart from an
    // absolute spec by interval arithmetic alone, so nothing was shifted and the program stayed at rows
    // [0, y) -- outside the partition.  Measured on thirteen single-sub-device prefills: every band with
    // y0 >= y builds, every band with 0 < y0 < y fails num_intersections == num_cores at
    // nlp_create_qkv_heads, and the five co-located splits follow the same rule.
    //
    // The per-range test below is the real safeguard: a factory that already placed its grid absolutely
    // produces ranges reaching past the partition height, and those are never shifted.  Only an absolute
    // spec lying wholly inside [y0, y) is misread, so forcing the shift is a bounded risk rather than an
    // unknown one -- and a wrong shift shows up as a wrong token, which every run checks.  Env-gated,
    // default off, so the shipped behaviour is byte-for-byte what 58 measured.
    static const bool force = [] {
        const char* e = std::getenv("SPATIAL_CONFINE_FORCE");
        return e != nullptr && e[0] == '1';
    }();
    if (!force || y0 == 0) {
        return false;
    }
    forced = true;
    return true;
}

inline CoreRangeSet spatial_confine(const CoreRangeSet& crs) {
    uint32_t y0 = 0, y = 0;
    bool forced = false;
    if (!spatial_shift_active(y0, y, forced)) {
        return crs;
    }
    for (const auto& r : crs.ranges()) {
        if (r.end_coord.y >= y) {
            return crs;
        }
    }
    // 59: in the overlapping case height alone cannot separate the two kinds of spec, so use the one
    // structural difference between them.  split_work_to_cores hands out cores from (0,0) upward, so a
    // relative set always contains row 0; a factory that already placed itself in the partition starts at
    // y0, which is non-zero here.  Shifting an absolute grid that happens to fit under y put two data
    // movement kernels on core (0,4) with the same NoC -- the failure this test removes.
    if (forced) {
        bool touches_row_zero = false;
        for (const auto& r : crs.ranges()) {
            if (r.start_coord.y == 0) {
                touches_row_zero = true;
                break;
            }
        }
        if (!touches_row_zero) {
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
