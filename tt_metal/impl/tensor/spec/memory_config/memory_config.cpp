// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include <tt_stl/reflection.hpp>

#include <tt-metalium/experimental/tensor/tensor_types.hpp>
#include <tt-metalium/experimental/tensor/spec/memory_config/memory_config.hpp>

#include "spatial_confine.hpp"

namespace tt::tt_metal {

namespace {
// Spatial confinement of L1 shard grids (SwiftNPU mechanism, 09 step 1): a shard grid expressed in
// partition-relative rows [0, y) moves to the active partition, the same rule spatial_confine.hpp applies
// to kernels, circular buffers and semaphores.  Ops that build their output memory config from the
// origin in C++ (matmul, nlp_*_heads_decode, rotary, binary_ng, ...) otherwise leave the bottom tenant's
// sharded tensors in the L1 of the other tenant's rows while their kernels run inside the partition
// (bench_09_decode_ab: first wrong op was sharded_to_interleaved of the QKV matmul output).  Grids that
// are already absolute (Python-built configs) are unchanged.  DRAM shard grids index banks, not worker
// rows, and are never touched.
std::optional<ShardSpec> confine_shard_spec(std::optional<ShardSpec> spec, BufferType buffer_type) {
    if (spec.has_value() && buffer_type != BufferType::DRAM) {
        spec->grid = spatial_confine(spec->grid);
    }
    return spec;
}
std::optional<NdShardSpec> confine_nd_shard_spec(std::optional<NdShardSpec> spec, BufferType buffer_type) {
    if (spec.has_value() && buffer_type != BufferType::DRAM) {
        spec->grid = spatial_confine(spec->grid);
    }
    return spec;
}
}  // namespace

MemoryConfig::MemoryConfig(
    TensorMemoryLayout memory_layout, BufferType buffer_type, std::optional<ShardSpec> shard_spec) :
    memory_layout_(memory_layout),
    buffer_type_(buffer_type),
    shard_spec_(confine_shard_spec(std::move(shard_spec), buffer_type)) {}

MemoryConfig::MemoryConfig(BufferType buffer_type, std::optional<NdShardSpec> nd_shard_spec) :
    memory_layout_(nd_shard_spec.has_value() ? TensorMemoryLayout::ND_SHARDED : TensorMemoryLayout::INTERLEAVED),
    buffer_type_(buffer_type),
    nd_shard_spec_(confine_nd_shard_spec(std::move(nd_shard_spec), buffer_type)),
    created_with_nd_shard_spec_(nd_shard_spec_.has_value()) {}

MemoryConfig::MemoryConfig(
    TensorMemoryLayout memory_layout,
    BufferType buffer_type,
    std::optional<ShardSpec> shard_spec,
    std::optional<NdShardSpec> nd_shard_spec,
    bool created_with_nd_shard_spec) :
    memory_layout_(memory_layout),
    buffer_type_(buffer_type),
    shard_spec_(confine_shard_spec(std::move(shard_spec), buffer_type)),
    nd_shard_spec_(confine_nd_shard_spec(std::move(nd_shard_spec), buffer_type)),
    created_with_nd_shard_spec_(created_with_nd_shard_spec) {}

MemoryConfig MemoryConfig::create_with_prepopulated_shard_specs(
    TensorMemoryLayout memory_layout,
    BufferType buffer_type,
    std::optional<ShardSpec> shard_spec,
    std::optional<NdShardSpec> nd_shard_spec,
    bool created_with_nd_shard_spec) {
    return MemoryConfig(
        memory_layout, buffer_type, std::move(shard_spec), std::move(nd_shard_spec), created_with_nd_shard_spec);
}

bool MemoryConfig::is_sharded() const {
    switch (this->memory_layout_) {
        case TensorMemoryLayout::HEIGHT_SHARDED:
        case TensorMemoryLayout::WIDTH_SHARDED:
        case TensorMemoryLayout::BLOCK_SHARDED:
        case TensorMemoryLayout::ND_SHARDED: return true;
        default: return false;
    }
}

bool MemoryConfig::is_l1() const { return buffer_type_ == BufferType::L1 or buffer_type_ == BufferType::L1_SMALL; }

bool MemoryConfig::is_dram() const { return buffer_type_ == BufferType::DRAM; }

bool operator==(const MemoryConfig& config_a, const MemoryConfig& config_b) {
    return config_a.buffer_type() == config_b.buffer_type() && config_a.memory_layout() == config_b.memory_layout() &&
           config_a.shard_spec() == config_b.shard_spec();
}

bool operator!=(const MemoryConfig& config_a, const MemoryConfig& config_b) { return not(config_a == config_b); }

std::ostream& operator<<(std::ostream& os, const MemoryConfig& config) {
    tt::stl::reflection::operator<<(os, config);
    return os;
}

}  // namespace tt::tt_metal

nlohmann::json ttsl::json::to_json_t<tt::tt_metal::MemoryConfig>::operator()(
    const tt::tt_metal::MemoryConfig& config) const {
    nlohmann::json json_object;
    json_object["memory_layout"] = config.memory_layout();
    json_object["buffer_type"] = config.buffer_type();
    json_object["created_with_nd_shard_spec"] = config.created_with_nd_shard_spec();
    if (config.created_with_nd_shard_spec()) {
        if (config.nd_shard_spec().has_value()) {
            json_object["nd_shard_spec"] = ttsl::json::to_json(config.nd_shard_spec().value());
        }
    } else {
        if (config.shard_spec().has_value()) {
            json_object["shard_spec"] = ttsl::json::to_json(config.shard_spec().value());
        }
    }
    return json_object;
}

tt::tt_metal::MemoryConfig ttsl::json::from_json_t<tt::tt_metal::MemoryConfig>::operator()(
    const nlohmann::json& json_object) const {
    auto memory_layout = json_object["memory_layout"].get<tt::tt_metal::TensorMemoryLayout>();
    auto buffer_type = json_object["buffer_type"].get<tt::tt_metal::BufferType>();
    auto created_with_nd_shard_spec = json_object["created_with_nd_shard_spec"].get<bool>();
    if (created_with_nd_shard_spec) {
        auto nd_shard_spec = ttsl::json::from_json<tt::tt_metal::NdShardSpec>(json_object["nd_shard_spec"]);
        return tt::tt_metal::MemoryConfig(buffer_type, std::move(nd_shard_spec));
    }
    std::optional<tt::tt_metal::ShardSpec> shard_spec;
    if (json_object.contains("shard_spec")) {
        shard_spec = ttsl::json::from_json<tt::tt_metal::ShardSpec>(json_object["shard_spec"]);
    }
    return tt::tt_metal::MemoryConfig(memory_layout, buffer_type, std::move(shard_spec));
}
