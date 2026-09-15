// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdlib>

#include <optional>
#include <cstdint>
#include <map>
#include <string_view>
#include <unordered_map>
#include <string>
#include <vector>

#include <tt-metalium/base_types.hpp>
#include <tt-metalium/hal_types.hpp>
#include <umd/device/types/arch.hpp>

namespace tt::tt_metal {

enum class DataMovementProcessor {
    RISCV_0 = 0,  // BRISC; Core DM0 on Quasar
    RISCV_1 = 1,  // NCRISC; Core DM1 on Quasar
    RISCV_2 = 2,  // Core DM2 on Quasar
    RISCV_3 = 3,  // Core DM3 on Quasar
    RISCV_4 = 4,  // Core DM4 on Quasar
    RISCV_5 = 5,  // Core DM5 on Quasar
    RISCV_6 = 6,  // Core DM6 on Quasar
    RISCV_7 = 7,  // Core DM7 on Quasar
};

enum NOC : uint8_t {
    RISCV_0_default = 0,
    RISCV_1_default = 1,
    NOC_0 = 0,
    NOC_1 = 1,
};

enum NOC_MODE : uint8_t {
    DM_DEDICATED_NOC = 0,
    DM_DYNAMIC_NOC = 1,
};

// 341 = (4096/(3 * sizeof(uint32_t)), where
// - 4096 - packet size in dispatch
// - 3 - number of kernels per tensix
constexpr uint32_t max_runtime_args = 341;

using KernelHandle = std::uint32_t;

// Option that controls build optimization level
enum class KernelBuildOptLevel : uint8_t {
    O1,     // Turns on level 1 optimization. Same as O.
    O2,     // Turns on level 2 optimization and also all flags specified by O1.
    O3,     // Turns on level 3 optimization and also all flags specified by O2.
    O0,     // Reduce compilation time and make debugging produce the expected results.
    Os,     // Optimize for size and also O2 optimizations except for those that increase binary size.
    Ofast,  // Turns on level O3 and also non standard optimizations.
    Oz,     // Aggressively optimize for size rather than speed.
};

struct DataMovementConfig {
    DataMovementProcessor processor = DataMovementProcessor::RISCV_0;  // For data transfer kernels: NCRISC & BRISC
    NOC noc = NOC::RISCV_0_default;
    NOC_MODE noc_mode = NOC_MODE::DM_DEDICATED_NOC;
    std::vector<uint32_t> compile_args;
    // Will cause CompileProgram to emit a file hlk_defines_generated.h
    // Each unique combination of defines will produce a unique compiled instantiation
    // This file is then automatically included in the generated compiled kernel files
    std::map<std::string, std::string> defines;
    // Both compile_args and named_compile_args contain compile time arguments
    // The former is accessed by index, the latter by name
    // Can be used in new/existing kernels by explicitly defining them in the config
    // Ex. std::vector<uint32_t> compile_args = {5, 7};
    //     std::unordered_map<std::string, uint32_t> named_compile_args = {{"arg1", 5}, {"arg2", 7}};
    //     CreateKernel(program, "kernel.cpp", core, DataMovementConfig{.compile_args = compile_args,
    //     .named_compile_args = named_compile_args})
    std::unordered_map<std::string, uint32_t> named_compile_args;
    // Set the compiler and linker optimization level
    KernelBuildOptLevel opt_level = KernelBuildOptLevel::O2;
};

struct ReaderDataMovementConfig : public DataMovementConfig {
    ReaderDataMovementConfig(
        std::vector<uint32_t> compile_args = {},
        std::map<std::string, std::string> defines = {},
        std::unordered_map<std::string, uint32_t> named_compile_args = {},
        KernelBuildOptLevel opt_level = KernelBuildOptLevel::O2);
};

struct WriterDataMovementConfig : public DataMovementConfig {
    WriterDataMovementConfig(
        std::vector<uint32_t> compile_args = {},
        std::map<std::string, std::string> defines = {},
        std::unordered_map<std::string, uint32_t> named_compile_args = {},
        KernelBuildOptLevel opt_level = KernelBuildOptLevel::O2);
};

struct ComputeConfig {
    MathFidelity math_fidelity = MathFidelity::HiFi4;
    bool fp32_dest_acc_en = false;
    bool dst_full_sync_en = false;
    std::vector<UnpackToDestMode> unpack_to_dest_mode;
    bool bfp8_pack_precise = false;
    bool math_approx_mode = false;
    std::vector<uint32_t> compile_args;
    // Will cause CompileProgram to emit a file hlk_defines_generated.h
    // Each unique combination of defines will produce a unique compiled instantiation
    // This file is then automatically included in the generated compiled kernel files
    std::map<std::string, std::string> defines;
    // Both compile_args and named_compile_args contain compile time arguments
    // The former is accessed by index, the latter by name
    // Can be used in new/existing kernels by explicitly defining them in the config
    // Ex. std::vector<uint32_t> compile_args = {5, 7};
    //     std::unordered_map<std::string, uint32_t> named_compile_args = {{"arg1", 5}, {"arg2", 7}};
    //     CreateKernel(program, "kernel.cpp", core, ComputeConfig{.compile_args = compile_args, .named_compile_args =
    //     named_compile_args})
    std::unordered_map<std::string, uint32_t> named_compile_args;
    // Set the compiler and linker optimization level
    KernelBuildOptLevel opt_level = KernelBuildOptLevel::O3;
};

// These are only used in op_profiler, are unstable and have not been designed for general use.
namespace detail {

inline NOC preferred_noc_for_dram_read(ARCH arch) {
    // 61: swept so the null is recorded rather than left for a reviewer.  60 measured the interleaved
    // route's ceiling at ~24 GB/s per reader against the sharded route's 43.4, and located the cause in
    // outstanding reads (one block in flight against three behind four transaction ids) rather than in the
    // fabric.  One stream on either NoC is still one stream, so moving the read to NOC_1 is expected to
    // change nothing; SPATIAL_DRAM_READ_NOC=1 makes that testable instead of assumed.  Default unchanged.
    static const int forced = [] {
        const char* e = std::getenv("SPATIAL_DRAM_READ_NOC");
        return (e && (e[0] == '0' || e[0] == '1')) ? (e[0] - '0') : -1;
    }();
    if (forced == 0) {
        return NOC::NOC_0;
    }
    if (forced == 1) {
        return NOC::NOC_1;
    }
    switch (arch) {
        case ARCH::WORMHOLE_B0:
        default: return NOC::NOC_0;
    }
}

inline NOC preferred_noc_for_dram_write(ARCH arch) {
    // 61: the companion to SPATIAL_DRAM_READ_NOC.  The two NoCs are ALREADY both in use -- reads default to
    // NOC_0 and writes to NOC_1 -- so moving the read alone puts two data movement kernels on one core
    // sharing a NoC, which deadlocks (measured: a silent hang, no assertion).  The meaningful test is the
    // SWAP, read on NOC_1 and write on NOC_0, which stays legal and is identical if the fabric is symmetric.
    static const int forced = [] {
        const char* e = std::getenv("SPATIAL_DRAM_WRITE_NOC");
        return (e && (e[0] == '0' || e[0] == '1')) ? (e[0] - '0') : -1;
    }();
    if (forced == 0) {
        return NOC::NOC_0;
    }
    if (forced == 1) {
        return NOC::NOC_1;
    }
    switch (arch) {
        case ARCH::WORMHOLE_B0:
        default: return NOC::NOC_1;
    }
}

struct KernelBinaryMeta {
    // This maps to Kernel::get_kernel_processor_type
    using ProcessorType = uint32_t;
    ProcessorType processor_type;
    std::size_t packed_size;
};

struct KernelMeta {
    // Kernel identifiers:
    // Owner for name and source is the Program class,
    // (direct owner is the Kernel class internally)
    std::string_view name, source;

    // Core identifiers:
    HalProcessorClassType processor_class;
    HalProgrammableCoreType programmable_core_type;

    // Core configuration:
    // This optinonal only contains a MathFidelity if the kernel is a compute kernel
    std::optional<MathFidelity> math_fidelity;

    // Binary metadata:
    std::vector<KernelBinaryMeta> binary_meta;
};

}  // namespace detail

}  // namespace tt::tt_metal
