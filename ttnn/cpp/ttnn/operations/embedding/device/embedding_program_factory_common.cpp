// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "embedding_program_factory_common.hpp"

extern "C" uint32_t tt_get_spatial_compute_grid_y_start();

namespace ttnn::prim {

CoreSplitResult split_work_to_cores_aligned(
    const CoreCoord grid_size, const uint32_t units_to_divide, const uint32_t alignment) {
    ZoneScoped;

    uint32_t num_cores_x = grid_size.x, num_cores_y = grid_size.y;
    uint32_t total_cores = num_cores_x * num_cores_y;

    // Initialize units_per_core and required_cores
    uint32_t units_per_core = alignment;
    uint32_t required_cores = (units_to_divide + units_per_core - 1) / units_per_core;

    // find units per core and required cores
    if (required_cores > total_cores) {
        units_per_core = ((units_to_divide + total_cores - 1) / total_cores + alignment - 1) / alignment * alignment;
        required_cores = (units_to_divide + units_per_core - 1) / units_per_core;
    }

    // Core set for all active cores
    CoreRangeSet all_cores = tt::tt_metal::num_cores_to_corerangeset(required_cores, grid_size, false);

    // Calculate remaining units for the last core
    uint32_t evenly_distributed_units = (required_cores - 1) * units_per_core;
    uint32_t remaining_units = units_to_divide - evenly_distributed_units;

    // Create core groups
    CoreRangeSet core_group_1 = all_cores;
    CoreRangeSet core_group_2;

    // Handle the last core if remaining units are less than units_per_core
    if (remaining_units > 0 && remaining_units < units_per_core) {
        uint32_t last_core_x = (required_cores - 1) % num_cores_x;
        uint32_t last_core_y = (required_cores - 1) / num_cores_x;

        core_group_2 =
            CoreRangeSet(CoreRange(CoreCoord(last_core_x, last_core_y), CoreCoord(last_core_x, last_core_y)));
        core_group_1 = tt::tt_metal::num_cores_to_corerangeset(required_cores - 1, grid_size, false);
    }

    // Adjust the units per core for each group
    uint32_t units_per_core_group_1 = units_per_core;
    uint32_t units_per_core_group_2 = remaining_units < units_per_core ? remaining_units : 0;

    uint32_t y_start = tt_get_spatial_compute_grid_y_start();
    if (y_start > 0) {
        auto shift_crs = [y_start](const CoreRangeSet& crs) {
            std::vector<CoreRange> shifted;
            shifted.reserve(crs.ranges().size());
            for (const auto& r : crs.ranges()) {
                shifted.emplace_back(
                    CoreCoord(r.start_coord.x, r.start_coord.y + y_start),
                    CoreCoord(r.end_coord.x, r.end_coord.y + y_start));
            }
            return CoreRangeSet(std::move(shifted));
        };
        all_cores = shift_crs(all_cores);
        core_group_1 = shift_crs(core_group_1);
        if (!core_group_2.ranges().empty()) {
            core_group_2 = shift_crs(core_group_2);
        }
    }

    return CoreSplitResult{
        required_cores, all_cores, core_group_1, core_group_2, units_per_core_group_1, units_per_core_group_2};
}
}  // namespace ttnn::prim
