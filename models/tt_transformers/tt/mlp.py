# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import torch

import torch
import ttnn
from models.common.lightweightmodule import LightweightModule
from models.tt_transformers.tt.attention import _attn_sync
from models.tt_transformers.tt.ccl import tt_all_reduce
from models.tt_transformers.tt.common import Mode, pad_to_size
from models.tt_transformers.tt.model_config import OpGroup, TensorGroup


class MLP(LightweightModule):
    def __init__(
        self,
        mesh_device,
        tt_ccl,
        args,
        state_dict,
        weight_cache_path,
        layer_num,
        dtype,
        model_config,
        state_dict_prefix=None,
        prefetcher=None,
    ):
        super().__init__()

        self.mesh_device = mesh_device
        self.tt_ccl = tt_ccl
        self.args = args
        self.dim = args.dim
        self.model_config = model_config
        self.layer_num = layer_num

        # Define the prefetcher object
        self.prefetcher = prefetcher

        state_dict_prefix = state_dict_prefix or args.get_state_dict_prefix(self.__class__.__name__, layer_num)
        torch_weight = lambda name: torch.transpose(state_dict[f"{state_dict_prefix}.{name}.weight"], -2, -1)
        pad_hidden_dim = lambda tensor, dim: pad_to_size(tensor, dim=dim, size=args.hidden_dim)
        # If padding was applied (e.g. via env var), add the unpadded hidden dim to the cache name to avoid loading incorrect weights
        hidden_dim_string = f".hidden_dim_{args.hidden_dim}" if args.hidden_dim != args.unpadded_hidden_dim else ""

        # 24: as_tensor restores a cached tensor with the memory config it was SAVED under, so a cache hit
        # silently defeats the interleave flag -- the weight came back 8-bank width-sharded with the flag set.
        # Bypass the cache for these three when the flag is on.
        if args.dummy_weights or os.environ.get("SPATIAL_FF_WEIGHTS_INTERLEAVED") == "1":
            cache_name = lambda _: None
        else:
            cache_name = lambda name: weight_cache_path / f"{state_dict_prefix}.{name}{hidden_dim_string}"

        w1_w3_mem_config = args.create_dram_sharded_mem_config(args.dim, args.hidden_dim // args.num_devices)
        w2_mem_config = args.create_dram_sharded_mem_config(args.hidden_dim // args.num_devices, args.dim)
        # 24: the 2D-mcast prefill matmul accepts a grid whose column count does not divide the weight's N-tile
        # count ONLY when the weight is DRAM-interleaved; with the weight width-sharded over the 8 DRAM banks
        # every column count except 8 produces nan, including counts that divide exactly.  Interleaving costs
        # 0.75% on FF1/FF3 and nothing on FF2/QKV (measured in isolation) and unlocks 13 columns.  It is gated
        # because the DRAM-sharded DECODE matmul needs the shards: only a prefill-only instance may set it.
        if os.environ.get("SPATIAL_FF_WEIGHTS_INTERLEAVED") == "1":
            w1_w3_mem_config = ttnn.DRAM_MEMORY_CONFIG
            w2_mem_config = ttnn.DRAM_MEMORY_CONFIG

        # TODO Clean up this code. With sharding, we load the normal weights and then shard them
        # Note: unsqueeze(0).unsqueeze(0) makes weights 4D [1, 1, H, W] to match attention weights
        # This is required for the dram_prefetcher to correctly interpret all weights
        def as_sharded_tensor(name, type, dims):
            # First get the raw weight and transpose it
            raw_weight = torch_weight(name[:2])  # This is 2D: [H, W]
            # Pad if needed
            padded_weight = pad_hidden_dim(raw_weight, dims[0] if args.is_galaxy else dims[-1])
            # Make 4D: [1, 1, H, W] - CRITICAL for prefetcher to work correctly
            torch_tensor = padded_weight.unsqueeze(0).unsqueeze(0)

            # 48 step 1: zeroing one half of w1/w3's columns and the matching half of w2's rows makes the
            # DEFAULT path compute exactly what one split part computes, which is the reference the
            # half-against-full comparison lacked.
            _zh = os.environ.get("SPATIAL_MLP_ZERO_HALF")
            # 49 run 2: masking w1's columns already zeroes half of h, because SiLU(0) * w3 = 0, so the w2 row
            # mask is redundant and the two masks must agree.  w2 is width-sharded on its 4096 COLUMNS while
            # the mask needs its 14336 ROWS, which is the easiest axis to get backwards.
            if _zh is not None and "w2" in name and os.environ.get("SPATIAL_MLP_ZERO_W2") == "0":
                _zh = None
            if _zh is not None and name[:2] in ("w1", "w2", "w3"):
                torch_tensor = torch_tensor.clone()
                _ax = -2 if "w2" in name else -1
                _h = torch_tensor.shape[_ax] // 2
                _z = int(_zh)
                if _ax == -2:
                    torch_tensor[..., _z * _h:(_z + 1) * _h, :] = 0
                else:
                    torch_tensor[..., _z * _h:(_z + 1) * _h] = 0

            result = ttnn.as_tensor(
                torch_tensor,
                dtype=type,
                device=self.mesh_device,
                mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, dims=dims, mesh_shape=args.cluster_shape),
                layout=ttnn.TILE_LAYOUT,
                memory_config=(
                    ttnn.DRAM_MEMORY_CONFIG if args.is_galaxy else w2_mem_config if "w2" in name else w1_w3_mem_config
                ),
                cache_file_name=None if _zh is not None else cache_name(name),
            )
            if os.environ.get("SPATIAL_MLP_SPLIT_TRACE") == "1" and max(layer_num, 0) == 0:
                print(f"[WEIGHT] {name:<12} host {tuple(torch_tensor.shape)} dev {tuple(result.shape)} "
                      f"dtype {result.dtype}  {result.memory_config()}", flush=True)
            return result

        # Sharded weights
        w1_dims = (-1, -2) if args.is_galaxy else (-2, -1)
        w2_dims = (-2, -1) if args.is_galaxy else (-1, -2)

        layer_num = max(layer_num, 0)  # cross_block uses the configuration of the first decoder

        # When prefetcher is enabled, use consistent dtypes across all layers to avoid
        # race conditions caused by different block sizes
        use_prefetcher = prefetcher is not None

        self.decoders_optimizations = self.args.decoders_optimizations

        ff1_3_dtype = self.decoders_optimizations.get_tensor_dtype(
            decoder_id=layer_num, tensor=TensorGroup.FF1_FF3, prefetcher=use_prefetcher
        )
        # 49 run 3: force FF1/FF3's weight dtype on BOTH paths, so a bfp4-against-bfp8 comparison is not
        # confounded by quantisation.  The model's default here is BFLOAT4_B; the op-level proof used bfp8.
        _ff13_override = os.environ.get("SPATIAL_MLP_FF13_DTYPE")
        ff2_dtype = self.decoders_optimizations.get_tensor_dtype(
            decoder_id=layer_num, tensor=TensorGroup.FF2, prefetcher=use_prefetcher
        )

        if os.environ.get("SPATIAL_MLP_DTYPE_TRACE") == "1":
            print(f"[FFDTYPE] layer {max(layer_num,0):>2}  ff1_3 {ff1_3_dtype}  ff2 {ff2_dtype}", flush=True)
        if _ff13_override:
            ff1_3_dtype = {"bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b, "bf16": ttnn.bfloat16}[_ff13_override]
        self.w1 = as_sharded_tensor(
            "w1_sharded", ff1_3_dtype, dims=w1_dims
        )  # bfp4 normally ok here but sub .99 pcc for llama 3.1 weights
        self.w2 = as_sharded_tensor("w2_sharded", ff2_dtype, dims=w2_dims)
        self.w3 = as_sharded_tensor("w3_sharded", ff1_3_dtype, dims=w1_dims)

        # 47: below four rows the decode MLP's circular buffers overflow L1 -- 1,720,832 B against 1,572,864 --
        # because FF1 is issued as ONE matmul of n=hidden_dim, giving per_core_N 28 on sixteen cores and a
        # 731 kB triple-buffered in1.  Two column halves give per_core_N 14, which is exactly what the
        # four-row configuration already uses and fits.  The halves write DISJOINT output columns, so this is
        # exact rather than approximate (measured PCC 0.999839 for FF1, 0.999476 for the whole MLP).  FF2's K
        # is split to match, so the halves are summed instead of being rejoined on device -- ttnn.concat on
        # width-sharded tensors fails with "bad optional access" anyway.
        # STATUS (47): NOT YET CORRECT IN THE MODEL, DO NOT USE FOR MEASUREMENT.  The decomposition is proven
        # at the op level (bench_46_split_mlp.py, PCC 0.999476) and the plumbing here is proven with
        # SPATIAL_MLP_N_SPLIT=1, which returns PCC 1.000000 against the default.  With 2 it returns 0.684290,
        # deterministically, and the cause is not yet found: the slices are distinct and correctly ranged, the
        # program configs match what the op-level test used, an explicit output shard spec changes nothing,
        # the accumulate is not aliasing, and each half contributes.
        # Off unless SPATIAL_MLP_N_SPLIT is set: the default path is the one every section 4 number stands on,
        # and it is verified bit-identical with the variable unset.
        _n_split = int(os.environ.get("SPATIAL_MLP_N_SPLIT", "1"))
        # n_split=1 still takes the split branch when the variable is set, which isolates "the new code path"
        # from "the slicing" when one of them is wrong.
        self._split_path = os.environ.get("SPATIAL_MLP_N_SPLIT") is not None and not args.is_galaxy and prefetcher is None
        self.n_split = max(1, _n_split) if self._split_path else 1
        self.w1_parts = self.w3_parts = self.w2_parts = None
        if self._split_path:
            _hp = (args.hidden_dim // args.num_devices) // self.n_split

            def as_split_tensor(name, type, dims, axis):
                """Same weight as as_sharded_tensor, cut into n_split pieces along `axis` (-1 for w1/w3's N,
                -2 for w2's K).  Never cached: 24 found as_tensor restoring a cached tensor with the memory
                config it was SAVED under, which would hand back an unsplit weight under a split config."""
                raw_weight = torch_weight(name[:2])
                padded_weight = pad_hidden_dim(raw_weight, dims[0] if args.is_galaxy else dims[-1])
                torch_tensor = padded_weight.unsqueeze(0).unsqueeze(0)
                extent = torch_tensor.shape[axis] // self.n_split
                # 48 step 2: halving the weight halves the per-bank DRAM shard (1792 -> 896 columns) while
                # the cores-per-bank stays 4, so a logical column block of w1 and a logical row block of w2
                # need not name the same intermediate channels.  Sharding each half over half the banks keeps
                # the shard width at the unsplit 1792 and tests exactly that.
                _sb = int(os.environ.get("SPATIAL_MLP_SPLIT_BANKS", "0"))
                # ttnn.CoreCoord is patched by spatial_core to shift by the partition's y offset, which is
                # wrong for a DRAM bank grid; take the unpatched types off the existing grid.
                _grid = None
                if _sb:
                    _CC = ttnn._ttnn.tensor.CoreCoord      # the unpatched type; ttnn.CoreCoord shifts by y0
                    _CR = ttnn._ttnn.tensor.CoreRange
                    _grid = ttnn.CoreRangeSet({_CR(_CC(0, 0), _CC(_sb - 1, 0))})
                mem = (args.create_dram_sharded_mem_config(_hp, args.dim, dram_grid=_grid) if "w2" in name
                       else args.create_dram_sharded_mem_config(args.dim, _hp, dram_grid=_grid))
                parts = []
                for _p in range(self.n_split):
                    lo, hi = _p * extent, (_p + 1) * extent
                    piece = (torch_tensor[..., lo:hi] if axis == -1 else torch_tensor[..., lo:hi, :]).contiguous()
                    if os.environ.get("SPATIAL_MLP_SPLIT_TRACE") == "1" and self.layer_num == 0:
                        print(f"[SPLIT] {name} part {_p} axis {axis} [{lo}:{hi}] shape {tuple(piece.shape)} "
                              f"sum {float(piece.float().abs().sum()):.3f}", flush=True)
                    parts.append(
                        ttnn.as_tensor(
                            piece,
                            dtype=type,
                            device=self.mesh_device,
                            mesh_mapper=ttnn.ShardTensor2dMesh(
                                self.mesh_device, dims=dims, mesh_shape=args.cluster_shape
                            ),
                            layout=ttnn.TILE_LAYOUT,
                            memory_config=mem,
                            cache_file_name=None,
                        )
                    )
                return parts

            self.w1_parts = as_split_tensor("w1_sharded", ff1_3_dtype, dims=w1_dims, axis=-1)
            self.w3_parts = as_split_tensor("w3_sharded", ff1_3_dtype, dims=w1_dims, axis=-1)
            self.w2_parts = as_split_tensor("w2_sharded", ff2_dtype, dims=w2_dims, axis=-2)

            # 49 run 1: read every sliced weight back and compare against the SAME slice of the device's full
            # weight, so both sides carry identical dequantisation.  This is the one place where slice, dtype
            # conversion and shard assignment all land; it separates "the tensor is wrong" from "the tensor is
            # right and the op is fed wrong".
            if os.environ.get("SPATIAL_MLP_SLICE_CHECK") == "1" and max(layer_num, 0) == 0:
                for _nm, _full, _prts, _ax in (("w1", self.w1, self.w1_parts, -1),
                                               ("w3", self.w3, self.w3_parts, -1),
                                               ("w2", self.w2, self.w2_parts, -2)):
                    _F = ttnn.to_torch(_full).float()
                    _ext = _F.shape[_ax] // len(_prts)
                    for _pi, _pt in enumerate(_prts):
                        _P = ttnn.to_torch(_pt).float()
                        _ref = (_F[..., _pi * _ext:(_pi + 1) * _ext] if _ax == -1
                                else _F[..., _pi * _ext:(_pi + 1) * _ext, :])
                        _same = bool(torch.equal(_P, _ref))
                        _diff = float((_P - _ref).abs().max())
                        _pcc = float(torch.corrcoef(torch.stack([_P.flatten(), _ref.flatten()]))[0, 1])
                        print(f"[SLICE] {_nm} part {_pi}  shape {tuple(_P.shape)}  identical={_same}  "
                              f"max|diff|={_diff:.6f}  PCC={_pcc:.8f}", flush=True)
                        del _P, _ref
                    del _F

        # Default activation is SILU
        self.activation_type = (
            args.mlp_activation_type if hasattr(args, "mlp_activation_type") else ttnn.UnaryOpType.SILU
        )

        # Insert the tensors into the prefetcher if it is used
        if self.prefetcher is not None:

            def register_weights():
                self.prefetcher.insert_tensor(self.w1)
                self.prefetcher.insert_tensor(self.w3)
                self.prefetcher.insert_tensor(self.w2)

            self.prefetcher.register_callback(register_weights)

    def forward(self, x: ttnn.Tensor, mode: Mode) -> ttnn.Tensor:
        """
        w1 -> gate_proj
        w2 -> down_proj
        w3 -> up_proj
        HF reference: self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        """
        seq_len = x.shape[-2]
        TG = self.args.is_galaxy
        layer_num = max(self.layer_num, 0)  # cross_block uses the configuration of the first decoder
        activation_dtype = self.decoders_optimizations.get_tensor_dtype(
            decoder_id=layer_num, tensor=TensorGroup.ACTIVATION
        )
        li_ff1_3_compute_kernel_cfg = self.decoders_optimizations.get_math_fidelity(
            decoder_id=layer_num, op=OpGroup.LI_FF1_FF3, configuration=self.args
        )

        if mode == Mode.PREFILL and seq_len >= self.args.prefill_len_cutoff:  # 512 if Blackhole, 1024 if Wormhole
            # Reshape input to to fit on device and parallelize computation
            x = ttnn.reshape(x, [1, seq_len // self.args.prefill_len_cutoff, self.args.prefill_len_cutoff, -1])

        # In decode mode (seqlen <= 32) do DRAM sharded matmuls
        # These use HiFi2; this drops 1 bit of the activations but would be FLOP-bound on 12 cores with HiFi4
        pc_1 = self.args.get_mlp_ff1_3_prg_config(mode, seq_len, self.prefetcher)
        pc_2 = self.args.get_mlp_ff2_prg_config(mode, seq_len, self.prefetcher)
        pc_3 = self.args.get_mlp_ff1_3_prg_config(mode, seq_len, self.prefetcher)
        # 56 step 1: w1 and w3 have the same shape, so `n` cannot separate them.  Rebuild the config for
        # whichever one should keep the DEFAULT block width, with the probe env cleared around the call, so
        # the override lands on exactly one of the pair.  If one alone breaks it, the defect has a tensor; if
        # only the two together break it, it is what consumes the pair -- the multiply.
        _which = os.environ.get("SPATIAL_PROBE_IBW_WHICH", "")
        if _which in ("w1", "w3") and mode == Mode.DECODE:
            _sv = os.environ.pop("SPATIAL_PROBE_IN0_BLOCK_W", None)
            _plain = self.args.dram_matmul_config(
                m=self.args.tile_padded_batch_rows, k=self.args.dim,
                n=self.args.hidden_dim // self.args.cluster_shape[1],
                num_cores=self.args.mlp_core_grid.num_cores)
            if _sv is not None:
                os.environ["SPATIAL_PROBE_IN0_BLOCK_W"] = _sv
            if _which == "w1":
                pc_3 = _plain          # override stays on w1 only
            else:
                pc_1 = _plain          # override stays on w3 only

        # 23: FF2 runs ttnn.experimental.minimal_matmul in prefill (seq_len > 128) and degrades 1.12x under a
        # decode neighbour where FF1/FF3, on ttnn.linear, degrade 2.1-2.3x.  Neither fuses an activation in
        # prefill -- SILU is applied in the silu_mul that follows -- so the same op is semantically available
        # to them.  This routes them through it to test whether the implementation carries the spread.
        _ff13_minimal = (os.environ.get("SPATIAL_FF13_MINIMAL") == "1" and mode == Mode.PREFILL
                         and seq_len > 128 and not TG and self.prefetcher is None)
        if mode == Mode.DECODE and getattr(self, "_split_path", False):
            # 47: FF1/FF3 over N in n_split pieces, the SiLU-multiply per piece, FF2's K split to match, and
            # the partial products summed.  Exact bar accumulation order.
            _hp = (self.args.hidden_dim // self.args.num_devices) // self.n_split
            _pc1 = self.args.dram_matmul_config(
                self.args.tile_padded_batch_rows, self.args.dim, _hp, self.args.mlp_core_grid.num_cores
            )
            _pc2 = self.args.dram_matmul_config(
                self.args.tile_padded_batch_rows, _hp, self.args.dim, self.args.mlp2_core_grid.num_cores
            )
            # The generic L1_WIDTH_SHARDED config the default path uses lets ttnn infer the output shard, and
            # that inference is right at full width and wrong at half: with it the split returns PCC 0.684,
            # with an explicit shard spec it returns what the op-level decomposition already showed.  State the
            # shard rather than let it be guessed.
            _mem13 = ttnn.create_sharded_memory_config(
                shape=(self.args.tile_padded_batch_rows, _hp // self.args.mlp_core_grid.num_cores),
                core_grid=self.args.mlp_core_grid,
                strategy=ttnn.ShardStrategy.WIDTH,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            _mem2 = ttnn.create_sharded_memory_config(
                shape=(self.args.tile_padded_batch_rows, self.args.dim // self.args.mlp2_core_grid.num_cores),
                core_grid=self.args.mlp2_core_grid,
                strategy=ttnn.ShardStrategy.WIDTH,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            _ff2_cfg = self.decoders_optimizations.get_math_fidelity(
                decoder_id=layer_num, op=OpGroup.LI_FF2, configuration=self.args
            )
            _acc = None
            _only = os.environ.get("SPATIAL_MLP_SPLIT_ONLY")
            for _p in range(self.n_split):
                if _only is not None and _p != int(_only):
                    continue
                _o1 = ttnn.linear(
                    x, self.w1_parts[_p], dtype=activation_dtype or ttnn.bfloat16, core_grid=None,
                    compute_kernel_config=li_ff1_3_compute_kernel_cfg, program_config=_pc1, memory_config=_mem13,
                )
                _o3 = ttnn.linear(
                    x, self.w3_parts[_p], dtype=activation_dtype or ttnn.bfloat16, core_grid=None,
                    compute_kernel_config=li_ff1_3_compute_kernel_cfg, program_config=_pc1, memory_config=_mem13,
                )
                _prod = ttnn.mul(
                    _o1, _o3, input_tensor_a_activations=[self.activation_type],
                    dtype=activation_dtype or ttnn.bfloat8_b, memory_config=_o1.memory_config(),
                )
                ttnn.deallocate(_o1)
                ttnn.deallocate(_o3)
                _part = ttnn.linear(
                    _prod, self.w2_parts[_p], dtype=ttnn.bfloat16, core_grid=None,
                    compute_kernel_config=_ff2_cfg, program_config=_pc2, memory_config=_mem2,
                )
                ttnn.deallocate(_prod)
                if _acc is None:
                    _acc = _part
                else:
                    # do NOT deallocate the operands: if add aliases one of them the result is freed with it,
                    # which would return a plausible wrong answer rather than an error.
                    _acc = ttnn.add(_acc, _part, memory_config=_acc.memory_config())
            _attn_sync("after w1_w3")
            ttnn.deallocate(x)
            return _acc

        w1_out = ttnn.experimental.minimal_matmul(
            x, self.w1, compute_kernel_config=li_ff1_3_compute_kernel_cfg, config=pc_1,
        ) if _ff13_minimal else ttnn.linear(
            x,
            self.w1,
            dtype=ttnn.bfloat8_b if TG else activation_dtype or ttnn.bfloat16,
            core_grid=None,  # FIXME: validate on TG ttnn.CoreGrid(y=8, x=8) if not pc_1 else None,
            compute_kernel_config=li_ff1_3_compute_kernel_cfg,
            program_config=pc_1,
            memory_config=self.args.get_mlp_ff1_3_mem_config(mode, self.prefetcher),
            global_cb=self.prefetcher.global_cb if self.prefetcher is not None and mode == Mode.DECODE else None,
            sub_device_id=self.prefetcher.worker_sub_device_id
            if self.prefetcher is not None and mode == Mode.DECODE
            else None,
        )
        w3_out = ttnn.experimental.minimal_matmul(
            x, self.w3, compute_kernel_config=li_ff1_3_compute_kernel_cfg, config=pc_3,
        ) if _ff13_minimal else ttnn.linear(
            x,
            self.w3,
            dtype=ttnn.bfloat8_b if TG else activation_dtype or ttnn.bfloat16,
            core_grid=None,  # FIXME: validate on TG ttnn.CoreGrid(y=8, x=8) if not pc_3 else None,
            compute_kernel_config=li_ff1_3_compute_kernel_cfg,
            program_config=pc_3,
            memory_config=self.args.get_mlp_ff1_3_mem_config(mode, self.prefetcher),
            global_cb=self.prefetcher.global_cb if self.prefetcher is not None and mode == Mode.DECODE else None,
            sub_device_id=self.prefetcher.worker_sub_device_id
            if self.prefetcher is not None and mode == Mode.DECODE
            else None,
        )
        # 55 rungs 2-3: the operator is correct standalone and the model is not, and the trace is ruled out.
        # This checks FF1's output INSIDE the model against a torch reference built from the model's OWN
        # inputs, which separates "the op is wrong with these inputs" from "the op is right and something
        # downstream is wrong".
        if os.environ.get("SPATIAL_MLP_FF1_CHECK") == "1" and mode == Mode.DECODE and max(self.layer_num, 0) == int(os.environ.get("SPATIAL_MLP_CHECK_LAYER", "0")):
            _xh = ttnn.to_torch(x).float()
            _w1h = ttnn.to_torch(self.w1).float()
            _got = ttnn.to_torch(w1_out).float()
            _ref = (_xh.reshape(-1, _xh.shape[-1]) @ _w1h.reshape(_w1h.shape[-2], -1))
            _g = _got.reshape(-1, _got.shape[-1])[: _ref.shape[0]]
            _p = float(torch.corrcoef(torch.stack([_g.flatten(), _ref.flatten()]))[0, 1])
            print(f"[FF1CHECK] layer 0 w1_out vs torch(model's own x, w1): PCC {_p:.6f}", flush=True)
            self._w1_first = _got.clone()          # 56 step 4: keep the first read to compare after the mul
        if os.environ.get("SPATIAL_MLP_SPEC_DUMP") == "1" and mode == Mode.DECODE and max(self.layer_num, 0) == 0:
            # 56 step 2: blocking the K loop must not touch the OUTPUT's shape or sharding.  If it does, that
            # is the defect stated outright; if only the address moves, that is the other way a 0.597 happens.
            for _nm, _t in (("w1_out", w1_out), ("w3_out", w3_out)):
                _mc = _t.memory_config()
                _ss = _mc.shard_spec
                print(f"[SPEC] {_nm} shape {tuple(_t.shape)} dtype {_t.dtype} layout {_t.layout} "
                      f"layout={_mc.memory_layout} buffer={_mc.buffer_type} "
                      f"shard={None if _ss is None else (tuple(_ss.shape), str(_ss.orientation))} "
                      f"grid={None if _ss is None else _ss.grid} addr={_t.buffer_address()}", flush=True)
        if os.environ.get("SPATIAL_MLP_SPEC_DUMP") == "1" and mode == Mode.DECODE and max(self.layer_num, 0) == 0:
            # 56 step 2: blocking the K loop must not touch the OUTPUT's shape or sharding.  If it does, that
            # is the defect stated outright; if only the address moves, that is the other way a 0.597 happens.
            for _nm, _t in (("w1_out", w1_out), ("w3_out", w3_out)):
                _mc = _t.memory_config()
                _ss = _mc.shard_spec
                print(f"[SPEC] {_nm} shape {tuple(_t.shape)} dtype {_t.dtype} "
                      f"layout={_mc.memory_layout} buffer={_mc.buffer_type} "
                      f"shard={None if _ss is None else (tuple(_ss.shape), str(_ss.orientation))} "
                      f"addr={_t.buffer_address()}", flush=True)
        if os.environ.get("SPATIAL_MLP_SPEC_DUMP") == "1" and mode == Mode.DECODE and max(self.layer_num, 0) == 0:
            # 56 step 2: blocking the K loop must not touch the OUTPUT's shape or sharding.  If it does, that
            # is the defect stated outright; if only the address moves, that is the other way a 0.597 happens.
            for _nm, _t in (("w1_out", w1_out), ("w3_out", w3_out)):
                _mc = _t.memory_config()
                _ss = _mc.shard_spec
                print(f"[SPEC] {_nm} shape {tuple(_t.shape)} dtype {_t.dtype} "
                      f"layout={_mc.memory_layout} shard="
                      f"{None if _ss is None else (tuple(_ss.shape), str(_ss.orientation))} "
                      f"addr={_t.buffer_address()}", flush=True)
        _attn_sync("after w1_w3")
        ttnn.deallocate(x)

        if TG:
            # if mode == "decode" and self.dim!=8192:
            #     w1_out = ttnn.to_memory_config(w1_out, ttnn.DRAM_MEMORY_CONFIG)
            #     w3_out = ttnn.to_memory_config(w3_out, ttnn.DRAM_MEMORY_CONFIG)
            if self.dim == 8192 or mode == Mode.PREFILL:
                input_mem_cfg = w1_out.memory_config()

                cluster_axis = 1
                w1_out = ttnn.experimental.reduce_scatter_minimal_async(
                    w1_out,
                    persistent_output_buffers=None,
                    dim=3,
                    multi_device_global_semaphore=self.tt_ccl.get_and_cycle_rs_semaphore_handles(cluster_axis),
                    barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis),
                    num_links=self.tt_ccl.get_num_links(cluster_axis),
                    cluster_axis=cluster_axis,
                    memory_config=self.model_config["FF1_OUT_REDUCE_SCATTER_MEMCFG"] if mode == Mode.DECODE else None,
                    intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    topology=ttnn.Topology.Linear,
                    chunks_per_sync=10,
                    num_workers_per_link=2,
                    num_buffers_per_channel=2,
                )

                w3_out = ttnn.experimental.reduce_scatter_minimal_async(
                    w3_out,
                    persistent_output_buffers=None,
                    dim=3,
                    multi_device_global_semaphore=self.tt_ccl.get_and_cycle_rs_semaphore_handles(cluster_axis),
                    barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis),
                    num_links=1,
                    cluster_axis=cluster_axis,
                    memory_config=self.model_config["FF1_OUT_REDUCE_SCATTER_MEMCFG"] if mode == Mode.DECODE else None,
                    intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    topology=ttnn.Topology.Linear,
                    chunks_per_sync=10,
                    num_workers_per_link=2,
                    num_buffers_per_channel=2,
                )
            else:
                # NOTE: In MLP All-reduce hard codes to 2 links, so we do not get the dynamic link count from the CCL class
                # to avoid any performance regressions.
                w1_out = tt_all_reduce(
                    w1_out,
                    self.mesh_device,
                    self.tt_ccl,
                    cluster_axis=1,
                    num_all_gather_links=2,
                    sharded=True if mode == Mode.DECODE else False,
                    topology=self.args.ccl_topology(),
                    memory_config=self.model_config["FF1_OUT_GATHERED_MEMCFG"] if mode == Mode.DECODE else None,
                )
                w3_out = tt_all_reduce(
                    w3_out,
                    self.mesh_device,
                    self.tt_ccl,
                    cluster_axis=1,
                    num_all_gather_links=2,
                    sharded=True if mode == Mode.DECODE else False,
                    topology=self.args.ccl_topology(),
                    memory_config=self.model_config["FF1_OUT_GATHERED_MEMCFG"] if mode == Mode.DECODE else None,
                )

        w2_in = ttnn.mul(
            w1_out,
            w3_out,
            input_tensor_a_activations=[self.activation_type],
            dtype=activation_dtype or ttnn.bfloat8_b,
            memory_config=w1_out.memory_config(),
        )

        if os.environ.get("SPATIAL_MLP_FF1_CHECK") == "1" and mode == Mode.DECODE and max(self.layer_num, 0) == 0 \
                and getattr(self, "_w1_first", None) is not None:
            # 56 step 4: the tensor is correct when read and its spec and address are unchanged, so the only
            # way left for the model to be wrong is that something writes over it between the two.
            _second = ttnn.to_torch(w1_out).float()
            _same = bool(torch.equal(self._w1_first, _second))
            _pp = float(torch.corrcoef(torch.stack([self._w1_first.flatten(), _second.flatten()]))[0, 1])
            print(f"[CLOBBER] w1_out read after FF1 vs read after the multiply: identical={_same} PCC {_pp:.6f}",
                  flush=True)

        if mode == Mode.DECODE and not TG and self.prefetcher is None:
            # w2 may use a different core grid, this is a no-op if they already match
            w2_in = ttnn.to_memory_config(w2_in, self.args.get_mlp_binary_mult_mem_config(mode))

        ttnn.deallocate(w3_out)
        ttnn.deallocate(w1_out)

        if TG and (self.dim == 8192 or mode == Mode.PREFILL):
            cluster_axis = 1
            w2_in = ttnn.experimental.all_gather_async(
                w2_in,
                persistent_output_buffer=None,
                dim=3,
                multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis),
                num_links=2,
                cluster_axis=1,
                topology=ttnn.Topology.Linear,
                memory_config=input_mem_cfg,
                barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis),
                chunks_per_sync=10,
                num_workers_per_link=2,
                num_buffers_per_channel=2,
            )

            if mode == Mode.DECODE:
                w2_in = ttnn.to_memory_config(w2_in, ttnn.L1_MEMORY_CONFIG)

        li_ff2_compute_kernel_cfg = self.decoders_optimizations.get_math_fidelity(
            decoder_id=layer_num, op=OpGroup.LI_FF2, configuration=self.args
        )

        if seq_len > 128 and mode != Mode.DECODE:
            w2_out = ttnn.experimental.minimal_matmul(
                w2_in,
                self.w2,
                compute_kernel_config=li_ff2_compute_kernel_cfg,
                config=pc_2,
            )
        else:
            w2_out = ttnn.linear(
                w2_in,
                self.w2,
                compute_kernel_config=li_ff2_compute_kernel_cfg,
                dtype=self.args.ccl_dtype if TG else activation_dtype or ttnn.bfloat16,
                program_config=pc_2,
                memory_config=self.args.get_mlp_ff2_mem_config(mode, self.prefetcher),
                core_grid=None,  # FIXME: validate on TG ttnn.CoreGrid(y=8, x=8) if not pc_2 else None,
                global_cb=self.prefetcher.global_cb if self.prefetcher is not None and mode == Mode.DECODE else None,
                sub_device_id=self.prefetcher.worker_sub_device_id
                if self.prefetcher is not None and mode == Mode.DECODE
                else None,
            )
        _attn_sync("after w2")
        ttnn.deallocate(w2_in)

        w2_out_reduced = tt_all_reduce(
            w2_out,
            self.mesh_device,
            self.tt_ccl,
            cluster_axis=0,
            dim=0 if (TG and self.dim < 8192) else 3,
            sharded=(mode == Mode.DECODE),
            memory_config=self.args.get_mlp_ff2_all_reduce_mem_config(mode, w2_out),
            rs_memory_config=self.model_config["MLP_RS_CONFIG"]["rs_memory_config"]
            if mode == Mode.DECODE
            else ttnn.DRAM_MEMORY_CONFIG,
            dtype=self.args.ccl_dtype,
            use_composite=True if self.dim == 8192 else False,
            topology=self.args.ccl_topology(),
            chunks_per_sync=self.model_config["MLP_RS_CONFIG"]["chunks_per_sync"] if mode == Mode.DECODE else 10,
            num_workers_per_link=self.model_config["MLP_RS_CONFIG"]["num_workers_per_link"]
            if mode == Mode.DECODE
            else 2,
            subdevice_id=self.prefetcher.worker_sub_device_id
            if mode == Mode.DECODE and self.prefetcher is not None
            else None,
        )
        # Ensure dim 0 and 1 are 1
        original_shape = w2_out_reduced.shape
        w2_out_reduced = ttnn.reshape(
            w2_out_reduced, (1, 1, original_shape[-4] * original_shape[-3] * original_shape[-2], original_shape[-1])
        )

        if mode == Mode.DECODE:
            w2_out_reduced = ttnn.to_memory_config(
                w2_out_reduced,
                self.args.get_mlp_output_mem_config(mode, self.prefetcher),
            )

        return w2_out_reduced
