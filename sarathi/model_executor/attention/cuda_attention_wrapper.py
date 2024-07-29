from typing import List, Optional, Iterable, Union, Tuple

import torch
from flashinfer import (
    BatchPrefillWithPagedKVCacheWrapper,
    append_paged_kv_cache,
)

from sarathi.config import ModelConfig, ParallelConfig
from sarathi.core.datatypes.sequence import SequenceMetadata
from sarathi.metrics.constants import OperationMetrics
from sarathi.model_executor.attention.base_attention_wrapper import BaseAttentionWrapper

MB = 1024 * 1024


class PrefillSharedBuffer:
    def __init__(
        self,
        max_batch_size: int,
        total_num_pages: int,
        block_size: int,
        num_kv_heads: int,
        num_qo_heads: int,
        head_dim: int,  # [64, 128, 256] according to flashinfer
        device,
        kv_data: torch.Tensor,
    ):
        self.max_batch_size = max_batch_size
        self.total_num_pages = total_num_pages
        self.device = device

        # Forward parameters
        # - prefill.begin_forward(..., num_qo_heads, num_kv_heads, head_dim, page_size)
        self.num_qo_heads = num_qo_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.page_size = self.block_size

        # [shared] CUDA Graph capturable,
        # or may be replaced by BaseAttentionWrapper.get_cache_block().
        self.kv_data = kv_data
        assert self.kv_data.shape == (
            self.total_num_pages,
            2,
            self.block_size,
            self.num_kv_heads,
            self.head_dim,
        )

        # Buffer
        self.workspace_buf = torch.empty(128 * MB, dtype=torch.uint8, device=device)
        self.qo_indptr_buf = torch.empty(max_batch_size + 1, dtype=torch.int32, device=device)
        self.paged_kv_indptr_buf = torch.empty(max_batch_size + 1, dtype=torch.int32, device=device)
        self.paged_kv_indices_buf = torch.empty(total_num_pages, dtype=torch.int32, device=device)
        self.paged_kv_last_page_len_buf = torch.empty(max_batch_size, dtype=torch.int32, device=device)
        self.qk_indptr_buf = torch.empty(max_batch_size + 1, dtype=torch.int32, device=device)

        # [prefill] CUDA Graph capturable
        self.q = torch.randn(max_batch_size, num_qo_heads, head_dim, device=device, dtype=torch.half)
        self.out = torch.empty_like(self.q)
        pass

    def get_wrapper_init_buffers(self, batch_size):
        # class BatchPrefillWithPagedKVCacheWrapper:
        #     def __init__(
        #         self,
        #         workspace_buffer: torch.Tensor,
        #         kv_layout: str = "NHD",
        #         use_cuda_graph: bool = False,
        #         qo_indptr_buf: Optional[torch.Tensor] = None,
        #         paged_kv_indptr_buf: Optional[torch.Tensor] = None,
        #         paged_kv_indices_buf: Optional[torch.Tensor] = None,
        #         paged_kv_last_page_len_buf: Optional[torch.Tensor] = None,
        #         custom_mask_buf: Optional[torch.Tensor] = None,
        #         qk_indptr_buf: Optional[torch.Tensor] = None,
        #     ): ...

        return dict(
            qo_indptr_buf=self.qo_indptr_buf[: batch_size + 1],
            paged_kv_indptr_buf=self.paged_kv_indptr_buf[: batch_size + 1],
            paged_kv_indices_buf=self.paged_kv_indices_buf,
            paged_kv_last_page_len_buf=self.paged_kv_last_page_len_buf[: batch_size],
            qk_indptr_buf=self.qk_indptr_buf[: batch_size + 1],
            # qk_indptr_buf=self.qk_indptr_buf[: batch_size + 1],
        )

    def get_q(self, batch_size):
        return self.q[:batch_size]

    def get_out(self, batch_size):
        return self.out[:batch_size]


class DecodeSharedBuffer:
    def __init__(
        self,
        max_batch_size: int,
        total_num_pages: int,
        block_size: int,
        num_kv_heads: int,
        num_qo_heads: int,
        head_dim: int,  # [64, 128, 256] according to flashinfer
        device,
        kv_data: torch.Tensor,
    ):
        self.max_batch_size = max_batch_size
        self.total_num_pages = total_num_pages
        self.device = device

        # Forward parameters
        # - prefill.begin_forward(..., num_qo_heads, num_kv_heads, head_dim, page_size)
        self.num_qo_heads = num_qo_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.page_size = self.block_size

        # [shared] CUDA Graph capturable,
        # or may be replaced by BaseAttentionWrapper.get_cache_block().
        self.kv_data = kv_data
        assert self.kv_data.shape == (
            self.total_num_pages,
            2,
            self.block_size,
            self.num_kv_heads,
            self.head_dim,
        )

        # Buffer
        self.workspace_buf = torch.empty(128 * MB, dtype=torch.uint8, device=device)
        self.paged_kv_indptr_buffer = torch.empty(max_batch_size + 1, dtype=torch.int32, device=device)
        self.paged_kv_indices_buffer = torch.empty(total_num_pages, dtype=torch.int32, device=device)
        self.paged_kv_last_page_len_buffer = torch.empty(max_batch_size, dtype=torch.int32, device=device)

        # [prefill] CUDA Graph capturable
        self.q = torch.randn(max_batch_size, num_qo_heads, head_dim, device=device, dtype=torch.half)
        self.out = torch.empty_like(self.q)
        pass

    def get_wrapper_init_buffers(self, batch_size):
        # class BatchDecodeWithPagedKVCacheWrapper:
        #     def __init__(
        #         self,
        #         workspace_buffer: torch.Tensor,
        #         kv_layout: str = "NHD",
        #         use_cuda_graph: bool = False,
        #         use_tensor_cores: bool = False,
        #         paged_kv_indptr_buffer: Optional[torch.Tensor] = None,
        #         paged_kv_indices_buffer: Optional[torch.Tensor] = None,
        #         paged_kv_last_page_len_buffer: Optional[torch.Tensor] = None,
        #     ):
        return dict(
            paged_kv_indptr_buffer=self.paged_kv_indptr_buffer[: batch_size + 1],
            paged_kv_indices_buffer=self.paged_kv_indices_buffer,
            paged_kv_last_page_len_buffer=self.paged_kv_last_page_len_buffer[: batch_size],
        )

    pass

class CUDAGraphAttentionWrapper(BaseAttentionWrapper):
    _inst = None

    def init(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        block_size: int,
        device: torch.device,
        max_batch_sizes: Iterable[int] = (128, 256, 512),
    ):
        super().init(model_config, parallel_config, block_size, device)

        self.shared_buffer = SharedBuffer(
            max_batch_size=max(max_batch_sizes),
            page_size=block_size,
            device=device,
            total_num_pages=128,  # TODO: Get this from profiled kv cache
            num_kv_heads=self.num_kv_heads,
            num_qo_heads=self.num_q_heads,
            head_dim=self.head_dim,
        )

        self.prefill_wrappers = {0: None}
        self.prefill_graph = {}
        for batch_size in range(1, max(max_batch_sizes) + 1):
            self.shared_buffer.get_prefill_buffers(batch_size)
            wrapper = BatchPrefillWithPagedKVCacheWrapper(
                self.shared_buffer.prefill_workspace_buf, "NHD",
                use_cuda_graph=True,
                **self.shared_buffer.get_prefill_buffers(batch_size),
            )
            self.prefill_wrappers[batch_size] = wrapper

        self.decode_wrappers = {0: None}
        self.decode_graph = {}
        for batch_size in range(1, max(max_batch_sizes) + 1):
            self.shared_buffer.get_decode_buffers(batch_size)
            wrapper = BatchPrefillWithPagedKVCacheWrapper(
                self.shared_buffer.decode_workspace_buf, "NHD",
                use_cuda_graph=True,
                **self.shared_buffer.get_decode_buffers(batch_size),
            )
            self.decode_wrappers[batch_size] = wrapper

        self.is_metadata_initialized = False
        self.is_profiling_iteration = False
        self.contains_prefill = False
        self.contains_decode = False
        self.num_prefill_tokens = 0
        self.num_total_tokens = 0

        self.append_qo_indptr_tensor = None
        self.append_kv_page_indices_tensor = None
        self.append_kv_page_indptr_tensor = None
        self.append_kv_last_page_len_tensor = None

    def to_int_tensor(self, data: List[int]) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.int32, device="cuda")

    def get_cache_block(self, num_blocks: int, **kwargs) -> torch.Tensor:
        return torch.randn(
            num_blocks,
            2,
            self.block_size,
            self.num_kv_heads,
            self.head_dim,
            **kwargs,
        )

    def get_wrappers(self, prefill_batch_size, decode_batch_size):
        # TODO: batch size may not be the right fit
        return self.prefill_wrappers[prefill_batch_size], self.decode_wrappers[decode_batch_size]

    def begin_forward(
        self,
        seq_metadata_list: List[SequenceMetadata],
    ) -> None:
        # The indptr tensor captures the location query tokens in the input tensor.
        # |<---------------------- num_valid_tokens ----------------------------------------------------->|
        # |<--------------- num_prompt_tokens -------------->||<------- num_generation_tokens (M) ------->|
        # |<--prompt_0-->|<--prompt_1-->|...|<--prompt_N-1-->||<--generation_0-->|...|<--generation_M-1-->|<--padding-->|
        #
        # Flashinfer calls this layout as a raggedtensor. The indptr tensor captures the start of each
        # sequence in the ragged tensor. The length of the indptr tensor is the number of sequences + 1.
        # We perform both prefill and decode attention in a single call to batched prefill kernel.
        # prefill_qo_indptr: [0, prompt_0, prompt_0 + prompt_1, ..., prompt_0 + ... + prompt_N-1, generation_0, generation_0 + 1, ..., generation_0 + ... + M]
        prefill_qo_indptr: List[int] = [0]
        decode_qo_indptr: List[int] = [0]
        # The kv_page_indices tensor captures the pages of the key-value cache that
        # are assigned to each token in the input tensor. Since there is a variable number
        # of pages assigned to each sequence, a ragged tensor to represent this.
        prefill_kv_page_indices: List[int] = []
        decode_kv_page_indices: List[int] = []
        # the last page might not be full, so we need to keep track of the length of the last page
        prefill_kv_last_page_len: List[int] = []
        decode_kv_last_page_len: List[int] = []
        # Since the prefill_kv_page_indices tensor is a ragged tensor, we also need to keep track of the
        # indptr tensor for the prefill_kv_page_indices tensor. This tensor captures the start of each sequence
        # in the ragged tensor.
        prefill_kv_page_indptr: List[int] = [0]
        decode_kv_page_indptr: List[int] = [0]

        self.is_profiling_iteration = False
        self.is_metadata_initialized = True

        self.contains_prefill = False
        self.contains_decode = False

        for seq_metadata in seq_metadata_list:
            if not seq_metadata.is_prompt:
                continue

            # ONLY used for profiling
            if seq_metadata.block_table is None:
                self.is_profiling_iteration = True
                # During memory profiling, the block tables are not initialized yet.
                #  We will just skip the attention computation for now.
                return

            self.contains_prefill = True

            prompt_chunk_len = seq_metadata.prompt_chunk_len
            processed_prompt_len = seq_metadata.seq.get_num_prompt_tokens_processed()
            current_total_len = processed_prompt_len + prompt_chunk_len

            # indptr for the prompt tokens in q/o tensor
            prefill_qo_indptr.append(prefill_qo_indptr[-1] + prompt_chunk_len)
            # Compute the kv page indices for the prompt tokens.
            num_blocks_in_use = (
                                    current_total_len + self.block_size - 1
                                ) // self.block_size
            prefill_kv_page_indices.extend(seq_metadata.block_table[:num_blocks_in_use])
            prefill_kv_page_indptr.append(
                prefill_kv_page_indptr[-1] + num_blocks_in_use
            )
            prefill_kv_last_page_len.append(
                current_total_len % self.block_size or self.block_size
            )

        for seq_metadata in seq_metadata_list:
            if seq_metadata.is_prompt:
                continue

            if seq_metadata.block_table is None:
                self.is_profiling_iteration = True
                return

            self.contains_decode = True

            context_len = seq_metadata.seq.get_len()
            # indptr for the prompt tokens in q/o tensor
            decode_qo_indptr.append(decode_qo_indptr[-1] + 1)
            # Compute the kv page indices for the prompt tokens.
            num_blocks_in_use = (context_len + self.block_size - 1) // self.block_size
            decode_kv_page_indices.extend(seq_metadata.block_table[:num_blocks_in_use])
            decode_kv_page_indptr.append(decode_kv_page_indptr[-1] + num_blocks_in_use)
            decode_kv_last_page_len.append(
                context_len % self.block_size or self.block_size
            )

        prefill_batch_size = len(prefill_qo_indptr) - 1
        decode_batch_size = len(decode_qo_indptr) - 1
        prefill_wrapper, decode_wrapper = self.get_wrappers(prefill_batch_size, decode_batch_size)
        if self.contains_prefill:
            prefill_wrapper.begin_forward(
                self.to_int_tensor(prefill_qo_indptr),
                self.to_int_tensor(prefill_kv_page_indptr),
                self.to_int_tensor(prefill_kv_page_indices),
                self.to_int_tensor(prefill_kv_last_page_len),
                self.num_q_heads,
                self.num_kv_heads,
                self.head_dim,
                self.block_size,
            )

        if self.contains_decode:
            decode_wrapper.begin_forward(
                self.to_int_tensor(decode_qo_indptr),
                self.to_int_tensor(decode_kv_page_indptr),
                self.to_int_tensor(decode_kv_page_indices),
                self.to_int_tensor(decode_kv_last_page_len),
                self.num_q_heads,
                self.num_kv_heads,
                self.head_dim,
                self.block_size,
            )

        self.num_prefill_tokens = prefill_qo_indptr[-1]
        self.num_total_tokens = self.num_prefill_tokens + len(decode_qo_indptr) - 1

        self.append_qo_indptr_tensor = self.to_int_tensor(
            prefill_qo_indptr[:-1]
            + [x + prefill_qo_indptr[-1] for x in decode_qo_indptr]
        )
        self.append_kv_page_indices_tensor = self.to_int_tensor(
            prefill_kv_page_indices + decode_kv_page_indices
        )
        self.append_kv_page_indptr_tensor = self.to_int_tensor(
            prefill_kv_page_indptr[:-1]
            + [x + prefill_kv_page_indptr[-1] for x in decode_kv_page_indptr]
        )
        self.append_kv_last_page_len_tensor = self.to_int_tensor(
            prefill_kv_last_page_len + decode_kv_last_page_len
        )

    def end_forward(self):
        if self.contains_prefill:
            self.prefill_wrapper.end_forward()

        if self.contains_decode:
            self.decode_wrapper.end_forward()

        self.is_metadata_initialized = False

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        softmax_scale: float = 1.0,
        layer_id: Optional[int] = None,
    ) -> torch.Tensor:
        assert self.is_metadata_initialized, "Metadata is not initialized."

        if self.is_profiling_iteration:
            # there is no need to call attention in profiling mode
            return torch.zeros_like(query)
