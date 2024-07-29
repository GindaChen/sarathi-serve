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


class PrefillStageCUDAWrapper(BaseAttentionWrapper):

    def __init__(
        self, model_config: ModelConfig, parallel_config: ParallelConfig,
        is_sub_stage: bool = False,  # has a higher stage that contains this wrapper
    ):
        super().__init__(model_config, parallel_config)

        self.max_batch_size = None
        self.is_sub_stage = is_sub_stage
        self.shared_buffer = None
        self.wrappers = None
        self.graphs = None

        self.active_wrapper = None
        self.active_graph = None
        self.is_metadata_initialized = False
        self.is_profiling_iteration = False
        self.contains_prefill = False
        self.num_prefill_tokens = 0
        # self.contains_decode = False
        # self.num_total_tokens = 0

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

    def init_wrappers(self, max_batch_size, kv_data, total_num_pages, block_size):
        self.max_batch_size = max_batch_size
        self.shared_buffer = PrefillSharedBuffer(
            max_batch_size=max_batch_size,
            total_num_pages=total_num_pages,
            block_size=block_size,
            num_kv_heads=self.num_kv_heads,
            num_qo_heads=self.num_q_heads,
            head_dim=self.head_dim,
            device=self.device,
            kv_data=kv_data,
        )

        self.wrappers = {0: None}

        # TODO: Think about what are the possible combinations we want
        for batch_size in range(1, max_batch_size + 1):
            self.wrappers[batch_size] = BatchPrefillWithPagedKVCacheWrapper(
                self.shared_buffer.workspace_buf, "NHD",
                use_cuda_graph=True,
                **self.shared_buffer.get_wrapper_init_buffers(batch_size),
            )
            pass

        return

    def get_wrapper(self, batch_size):
        return self.wrappers[batch_size]

    def _begin_forward__init_cuda_graph(self, wrapper: BatchPrefillWithPagedKVCacheWrapper, batch_size: int, ):
        total_num_pages = batch_size
        block_size = self.block_size

        prefill_qo_indptr = torch.arange(0, batch_size + 1).int()
        prefill_kv_page_indptr = torch.arange(0, batch_size + 1).int()
        prefill_kv_page_indices = torch.arange(0, total_num_pages).int()
        prefill_kv_last_page_len = torch.full((batch_size,), block_size, dtype=torch.int32)
        self._begin_forward(
            wrapper,
            batch_size,
            prefill_qo_indptr.tolist(),
            prefill_kv_page_indices.tolist(),
            prefill_kv_page_indptr.tolist(),
            prefill_kv_last_page_len.tolist(),
        )
        pass

    def init_cuda_graphs(self, pos_encoding_mode="NONE", softmax_scale=1.0):
        # CUDA Graphs for prefill.
        self.graphs = {0: None}

        max_batch_size = self.max_batch_size
        for batch_size in range(1, max_batch_size + 1):
            wrapper = self.wrappers[batch_size]
            self._begin_forward__init_cuda_graph(wrapper, batch_size)
            g = torch.cuda.CUDAGraph()
            o = self.shared_buffer.get_out(batch_size),
            with g.capture_state():
                o[:] = wrapper.forward(
                    self.shared_buffer.get_q(batch_size),
                    self.shared_buffer.kv_data,
                    pos_encoding_mode=pos_encoding_mode,  # TODO: Dangling control vars
                    sm_scale=softmax_scale,
                )
            self.end_forward()
            pass
        pass

    def end_forward(self):
        self.append_qo_indptr_tensor = None
        self.append_kv_page_indices_tensor = None
        self.append_kv_page_indptr_tensor = None
        self.append_kv_last_page_len_tensor = None

        self.is_metadata_initialized = False
        self.num_prefill_tokens = 0
        self.contains_prefill = False
        if not self.active_wrapper:
            return
        self.active_wrapper.end_forward()
        self.active_wrapper = None
        return

    def begin_forward(
        self,
        seq_metadata_list: List[SequenceMetadata],
    ) -> None:
        assert self.active_wrapper is None

        # Find the prefill parts of the requests.
        prefill_qo_indptr: List[int] = [0]  # (batch_size + 1)
        prefill_kv_page_indices: List[int] = []  # (total_num_pages)
        prefill_kv_last_page_len: List[int] = []  # (batch_size)
        prefill_kv_page_indptr: List[int] = [0]  # (batch_size + 1)

        for seq_metadata in seq_metadata_list:
            if not seq_metadata.is_prompt:
                continue

            # ONLY used for profiling
            if seq_metadata.block_table is None:
                self.is_profiling_iteration = True
                # During memory profiling, the block tables are not initialized yet.
                #  We will just skip the attention computation for now.
                return

            # Compute the number of query tokens to use.
            prompt_chunk_len = seq_metadata.prompt_chunk_len
            processed_prompt_len = seq_metadata.seq.get_num_prompt_tokens_processed()
            current_total_len = processed_prompt_len + prompt_chunk_len
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

        # Get the wrapper with the appropriate batch size
        # TODO: (FIX) batch size should get to the appropriate quantile
        batch_size = prefill_qo_indptr[-1]
        wrapper = self.wrappers[batch_size]

        self._begin_forward(
            wrapper,
            batch_size,
            prefill_qo_indptr,
            prefill_kv_page_indices,
            prefill_kv_page_indptr,
            prefill_kv_last_page_len,
        )
        pass

    def _begin_forward(self, wrapper, batch_size, prefill_qo_indptr, prefill_kv_page_indices, prefill_kv_page_indptr,
                       prefill_kv_last_page_len):

        # Set stateful states.
        self.is_metadata_initialized = True
        self.num_prefill_tokens = prefill_qo_indptr[-1]
        self.contains_prefill = batch_size > 0
        self.active_wrapper = wrapper

        if self.contains_prefill:
            wrapper.begin_forward(
                self.to_int_tensor(prefill_qo_indptr),
                self.to_int_tensor(prefill_kv_page_indptr),
                self.to_int_tensor(prefill_kv_page_indices),
                self.to_int_tensor(prefill_kv_last_page_len),
                self.num_q_heads,
                self.num_kv_heads,
                self.head_dim,
                self.block_size,
            )

        if not self.is_sub_stage:
            self.append_qo_indptr_tensor = self.to_int_tensor(
                prefill_qo_indptr[:-1]
            )
            self.append_kv_page_indices_tensor = self.to_int_tensor(
                prefill_kv_page_indices
            )
            self.append_kv_page_indptr_tensor = self.to_int_tensor(
                prefill_kv_page_indptr[:-1]
            )
            self.append_kv_last_page_len_tensor = self.to_int_tensor(
                prefill_kv_last_page_len
            )
        return

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        softmax_scale: float = 1.0,
        layer_id: Optional[int] = None,
    ) -> torch.Tensor:
        assert self.is_metadata_initialized, "Metadata is not initialized."
        if self.is_profiling_iteration:
            # there is no need to call attention in profiling mode
            return torch.zeros_like(query)

        # Called as an entry point

        with self.get_timer(OperationMetrics.ATTN_KV_CACHE_SAVE, layer_id):
            append_paged_kv_cache(
                key,
                value,
                self.append_qo_indptr_tensor,
                kv_cache,
                self.append_kv_page_indices_tensor,
                self.append_kv_page_indptr_tensor,
                self.append_kv_last_page_len_tensor,
                kv_layout="NHD",
            )
        pass

    def forward_cuda(self, query, key, value, kv_cache, softmax_scale, layer_id):
        with self.get_timer(OperationMetrics.ATTN_PREFILL, layer_id):
            graph = self.active_graph
            batch_size = self.num_prefill_tokens

            q = self.shared_buffer.get_q(batch_size)
            q[:] = query[:batch_size]

            graph.replay()

            o = self.shared_buffer.get_out(batch_size)
        return o

    def forward_normal(self, query, key, value, kv_cache, softmax_scale, layer_id):
        raise NotImplementedError("easy")
