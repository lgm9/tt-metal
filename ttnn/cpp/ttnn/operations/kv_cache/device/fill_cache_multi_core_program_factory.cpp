// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "fill_cache_multi_core_program_factory.hpp"

// Thread-local spatial partition helpers (set by _cpp_set_grid_v2).
// Returns 0 when no spatial partition is active.
extern "C" uint32_t tt_get_spatial_compute_grid_y();
extern "C" uint32_t tt_get_spatial_compute_grid_y_start();

using namespace tt::tt_metal;

namespace ttnn::prim {

using namespace tt::constants;

FillCacheMultiCoreProgramFactory::cached_program_t FillCacheMultiCoreProgramFactory::create(
    const KvCacheParams& operation_attributes, const KvCacheInputs& tensor_args, Tensor& /*output_tensor*/) {
    const auto& cache_tensor = tensor_args.cache;
    const auto& input_tensor = tensor_args.input;
    const auto batch_idx = operation_attributes.batch_idx;
    const auto update_idx = operation_attributes.update_idx;
    Program program{};

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    uint32_t single_tile_size = tt::tile_size(cb_data_format);

    // TODO: For interleaved and kv_heads > 1, we assert that each core only gets 1 tile along seq_len
    // For sharded, each core gets shard_shape[0] number of tiles along seq_len.
    // For either case, assume that work doesn't spill over to next head, so we just increment by Wt within
    // reader/writer
    uint32_t num_blocks_of_work = input_tensor.padded_shape()[1] * input_tensor.padded_shape()[-2] / TILE_HEIGHT;

    uint32_t Wt = cache_tensor.padded_shape()[-1] / TILE_WIDTH;
    uint32_t input_Ht = input_tensor.padded_shape()[-2] / TILE_HEIGHT;  // seq_len
    uint32_t cache_HtWt = cache_tensor.padded_shape()[-2] * Wt / TILE_HEIGHT;
    uint32_t cache_CHtWt = cache_tensor.padded_shape()[1] * cache_HtWt;
    uint32_t update_idxt = update_idx / TILE_HEIGHT;
    uint32_t start_idx = (batch_idx * cache_CHtWt) + (update_idxt * Wt);
    tt::tt_metal::IDevice* device = input_tensor.device();

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    // Cap x at 8: columns 0-7 are compute cores; columns 8+ are dispatch/ethernet.
    uint32_t num_cores_x = (uint32_t)std::min(compute_with_storage_grid_size.x, (size_t)8);
    uint32_t num_cores_y = compute_with_storage_grid_size.y;

    // Respect the thread-local spatial partition so we don't dispatch onto cores
    // that belong to another partition (or outside any partition entirely).
    uint32_t sp_y  = tt_get_spatial_compute_grid_y();
    uint32_t sp_y0 = sp_y > 0 ? tt_get_spatial_compute_grid_y_start() : 0u;
    uint32_t effective_y = sp_y > 0 ? sp_y : num_cores_y;
    bool row_major;
    uint32_t num_cores, num_blocks_per_core_group_1, num_blocks_per_core_group_2;

    CoreRangeSet all_cores, core_group_1, core_group_2;

    const std::optional<ShardSpec>& shard_spec = input_tensor.shard_spec();

    uint32_t num_input_tiles;
    if (shard_spec.has_value()) {
        row_major = shard_spec.value().orientation == ShardOrientation::ROW_MAJOR;
        all_cores = shard_spec.value().grid;
        num_cores = all_cores.num_cores();
        core_group_1 = all_cores;
        core_group_2 = CoreRangeSet();
        num_blocks_per_core_group_1 = shard_spec.value().shape[0] / TILE_HEIGHT;
        num_blocks_per_core_group_2 = 0;
        num_input_tiles = shard_spec.value().shape[0] * shard_spec.value().shape[1] / TILE_HW;
        auto bbox = all_cores.bounding_box();
        num_cores_x = bbox.end_coord.x + 1;
        num_cores_y = bbox.end_coord.y + 1;
    } else {
        row_major = true;
        std::tie(
            num_cores,
            all_cores,
            core_group_1,
            core_group_2,
            num_blocks_per_core_group_1,
            num_blocks_per_core_group_2) =
            tt::tt_metal::split_work_to_cores(
                // Limit to 1 core to prevent NOC routing deadlock on Blackhole:
                // 32-core fills generate concurrent DRAM traffic that deadlocks
                // at specific DRAM address ranges (user_id=9, layer=18 in LLaMA-3.1-8B).
                CoreCoord{1, 1}, num_blocks_of_work, row_major);

        // Shift all core ranges up by the partition y-offset when sp_y0 > 0.
        if (sp_y0 > 0) {
            auto shift_crs = [sp_y0](const CoreRangeSet& crs) {
                std::vector<CoreRange> shifted;
                for (const auto& cr : crs.ranges()) {
                    shifted.push_back(
                        {{cr.start_coord.x, cr.start_coord.y + sp_y0},
                         {cr.end_coord.x,   cr.end_coord.y + sp_y0}});
                }
                return CoreRangeSet(shifted);
            };
            all_cores    = shift_crs(all_cores);
            core_group_1 = shift_crs(core_group_1);
            core_group_2 = shift_crs(core_group_2);
        }
        num_input_tiles = 2;  // double buffered
    }

    uint32_t src0_cb_index = 0;
    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, cb_data_format}})
            .set_page_size(src0_cb_index, single_tile_size);
    if (shard_spec.has_value()) {
        cb_src0_config = cb_src0_config.set_globally_allocated_address(*input_tensor.buffer());
    }
    auto cb_src0 = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

    uint32_t output_cb_index = src0_cb_index;

    auto* src_buffer = input_tensor.buffer();
    auto* dst_buffer = cache_tensor.buffer();

    std::vector<uint32_t> reader_compile_time_args;
    tt::tt_metal::TensorAccessorArgs(*src_buffer).append_to(reader_compile_time_args);

    std::vector<uint32_t> writer_compile_time_args = {(std::uint32_t)output_cb_index};
    tt::tt_metal::TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);

    std::map<std::string, std::string> reader_kernel_defines;
    if (shard_spec.has_value()) {
        reader_kernel_defines["INPUT_SHARDED"] = "1";
    }

    tt::tt_metal::KernelHandle unary_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/kv_cache/device/kernels/dataflow/reader_fill_cache_interleaved_start_id.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args, reader_kernel_defines));

    tt::tt_metal::KernelHandle unary_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
        all_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    uint32_t g1_numcores = core_group_1.num_cores();

    // Build the per-core work list in partition-row order, then apply the y-offset.
    auto cores_base = grid_to_cores(num_cores, num_cores_x, effective_y, row_major);
    std::vector<CoreCoord> cores;
    cores.reserve(cores_base.size());
    for (auto c : cores_base) {
        c.y += sp_y0;
        cores.push_back(c);
    }

    for (uint32_t i = 0, num_blocks_written = 0; i < num_cores; i++) {
        const CoreCoord& core = cores.at(i);
        uint32_t num_blocks_per_core = 0;
        if (i < g1_numcores) {
            num_blocks_per_core = num_blocks_per_core_group_1;
        } else {
            num_blocks_per_core = num_blocks_per_core_group_2;
        }

        tt::tt_metal::SetRuntimeArgs(
            program,
            unary_reader_kernel_id,
            core,
            {
                src_buffer->address(),
                num_blocks_per_core * Wt,
                num_blocks_written * Wt,
            });

        const uint32_t cache_start_id = start_idx                                       // user batch start
                                        + (num_blocks_written / input_Ht * cache_HtWt)  // cache head offset
                                        + ((num_blocks_written % input_Ht) * Wt);       // seq_len offset

        tt::tt_metal::SetRuntimeArgs(
            program,
            unary_writer_kernel_id,
            core,
            {
                dst_buffer->address(),
                num_blocks_per_core * Wt,
                cache_start_id,
            });
        num_blocks_written += num_blocks_per_core;
    }

    return cached_program_t{
        std::move(program),
        shared_variables_t{
            .unary_reader_kernel_id = unary_reader_kernel_id,
            .unary_writer_kernel_id = unary_writer_kernel_id,
            .cb_src0 = cb_src0,
            .cores = cores,
            .g1_numcores = g1_numcores,
            .core_group_1 = core_group_1,
            .num_blocks_per_core_group_1 = num_blocks_per_core_group_1,
            .core_group_2 = core_group_2,
            .num_blocks_per_core_group_2 = num_blocks_per_core_group_2,
            .Wt = Wt,
            .input_Ht = input_Ht,
            .cache_HtWt = cache_HtWt,
            .cache_CHtWt = cache_CHtWt,
        }};
}

void FillCacheMultiCoreProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const KvCacheParams& operation_attributes,
    const KvCacheInputs& tensor_args,
    Tensor& /*output_tensor*/) {
    auto& program = cached_program.program;
    const auto& unary_reader_kernel_id = cached_program.shared_variables.unary_reader_kernel_id;
    const auto& unary_writer_kernel_id = cached_program.shared_variables.unary_writer_kernel_id;
    const auto& cb_src0 = cached_program.shared_variables.cb_src0;
    const auto& cores = cached_program.shared_variables.cores;
    const auto g1_numcores = cached_program.shared_variables.g1_numcores;
    const auto num_blocks_per_core_group_1 = cached_program.shared_variables.num_blocks_per_core_group_1;
    const auto num_blocks_per_core_group_2 = cached_program.shared_variables.num_blocks_per_core_group_2;
    const auto Wt = cached_program.shared_variables.Wt;
    const auto input_Ht = cached_program.shared_variables.input_Ht;
    const auto cache_HtWt = cached_program.shared_variables.cache_HtWt;
    const auto cache_CHtWt = cached_program.shared_variables.cache_CHtWt;

    const auto batch_idx = operation_attributes.batch_idx;
    const auto update_idx = operation_attributes.update_idx;

    uint32_t update_idxt = update_idx / TILE_HEIGHT;
    uint32_t start_idx = (batch_idx * cache_CHtWt) + (update_idxt * Wt);

    auto* src_buffer = tensor_args.input.buffer();

    auto* dst_buffer = tensor_args.cache.buffer();

    if (tensor_args.input.is_sharded()) {
        UpdateDynamicCircularBufferAddress(program, cb_src0, *src_buffer);
    }

    for (uint32_t i = 0, num_blocks_written = 0; i < cores.size(); i++) {
        const CoreCoord& core = cores.at(i);
        uint32_t num_blocks_per_core = 0;
        if (i < g1_numcores) {
            num_blocks_per_core = num_blocks_per_core_group_1;
        } else {
            num_blocks_per_core = num_blocks_per_core_group_2;
        }

        {
            auto& runtime_args = GetRuntimeArgs(program, unary_reader_kernel_id, core);
            runtime_args[0] = src_buffer->address();
        }

        {
            const uint32_t cache_start_id = start_idx                                       // user batch start
                                            + (num_blocks_written / input_Ht * cache_HtWt)  // cache head offset
                                            + ((num_blocks_written % input_Ht) * Wt);       // seq_len offset

            auto& runtime_args = GetRuntimeArgs(program, unary_writer_kernel_id, core);
            runtime_args[0] = dst_buffer->address();
            runtime_args[2] = cache_start_id;
        }
        num_blocks_written += num_blocks_per_core;
    }
}

}  // namespace ttnn::prim
