import dataclasses
from typing import List, Optional

import torch
from flashinfer import BatchPrefillWithPagedKVCacheWrapper, append_paged_kv_cache

from sarathi.config import ModelConfig, ParallelConfig
from sarathi.core.datatypes.sequence import SequenceMetadata
from sarathi.metrics.constants import OperationMetrics
from sarathi.model_executor.attention.base_attention_wrapper import BaseAttentionWrapper


def copy_tensor_portion(dst, src):
    """
    Copies the portion of the target tensor to the source tensor based on the shape of the target tensor.

    Args:
        dst (torch.Tensor): The tensor to copy to.
        src (torch.Tensor): The tensor to copy from.
    """
    slices = tuple(slice(0, min(s, t)) for s, t in zip(dst.shape, src.shape))
    dst[slices] = src[slices]


@dataclasses.dataclass()
class SharedBuffer:
    # Prefill / Decode Batch Size
    prefill_batch_size: int
    decode_batch_size: int
    device: torch.device = torch.device("cuda")

    # GPU cache
    total_page_count: int = 1024  # TODO: get this number after profiling...
    gpu_cache: List[torch.Tensor] = None

    # Prelude to `ModelRunner.run()`
    # used in `ModelRunner._prepare_inputs()`
    input_tokens: torch.Tensor = None
    input_positions: torch.Tensor = None

    # begin_forward() buffer
    prefill_qo_indptr: torch.Tensor = None
    prefill_kv_page_indptr: torch.Tensor = None
    prefill_kv_page_indices: torch.Tensor = None
    prefill_kv_last_page_len: torch.Tensor = None

    decode_qo_indptr: torch.Tensor = None
    decode_kv_page_indptr: torch.Tensor = None
    decode_kv_page_indices: torch.Tensor = None
    decode_kv_last_page_len: torch.Tensor = None

    # forward() buffer
    # - variables set during begin forward
    append_qo_indptr_tensor: torch.Tensor = None
    append_kv_page_indptr_tensor: torch.Tensor = None
    append_kv_page_indices_tensor: torch.Tensor = None
    append_kv_last_page_len_tensor: torch.Tensor = None

    @property
    def batch_size(self):
        return self.prefill_batch_size + self.decode_batch_size

    def __post_init__(self):
        assert self.prefill_batch_size > 0 or self.decode_batch_size > 0

        device = self.device
        prefill_batch_size = self.prefill_batch_size
        decode_batch_size = self.decode_batch_size
        batch_size = self.batch_size

        alloc = lambda n: torch.empty(n, dtype=torch.int32, device=device)

        if self.input_tokens is None:
            self.input_tokens = torch.empty(batch_size, dtype=torch.long, device=device)
        if self.input_positions is None:
            self.input_positions = torch.empty(batch_size, dtype=torch.long, device=device)

        if self.prefill_qo_indptr is None:
            self.prefill_qo_indptr = alloc(prefill_batch_size + 1)
        if self.prefill_kv_page_indices is None:
            self.prefill_kv_page_indices = alloc(self.total_page_count)
        if self.prefill_kv_last_page_len is None:
            self.prefill_kv_last_page_len = alloc(prefill_batch_size)
        if self.prefill_kv_page_indptr is None:
            self.prefill_kv_page_indptr = alloc(prefill_batch_size + 1)

        if self.decode_qo_indptr is None:
            self.decode_qo_indptr = alloc(decode_batch_size + 1)
        if self.decode_kv_page_indices is None:
            self.decode_kv_page_indices = alloc(self.total_page_count)
        if self.decode_kv_last_page_len is None:
            self.decode_kv_last_page_len = alloc(decode_batch_size)
        if self.decode_kv_page_indptr is None:
            self.decode_kv_page_indptr = alloc(decode_batch_size + 1)

        if self.append_qo_indptr_tensor is None:
            self.append_qo_indptr_tensor = alloc(batch_size + 1)
        if self.append_kv_page_indices_tensor is None:
            self.append_kv_page_indices_tensor = alloc(self.total_page_count)
        if self.append_kv_page_indptr_tensor is None:
            self.append_kv_page_indptr_tensor = alloc(batch_size + 1)
        if self.append_kv_last_page_len_tensor is None:
            self.append_kv_last_page_len_tensor = alloc(batch_size)

    def prepare_sliced_buffer(self, prefill_batch_size: int, decode_batch_size: int):
        """Produce another SharedBuffer object with a smaller batch size, which slices most of the tensors."""
        batch_size = prefill_batch_size + decode_batch_size

        return SharedBuffer(
            prefill_batch_size=prefill_batch_size,
            decode_batch_size=decode_batch_size,
            device=self.device,

            total_page_count=self.total_page_count,
            input_tokens=self.input_tokens[:batch_size],
            input_positions=self.input_positions[:batch_size],
            gpu_cache=self.gpu_cache,

            prefill_qo_indptr=self.prefill_qo_indptr[:prefill_batch_size + 1],
            prefill_kv_page_indptr=self.prefill_kv_page_indptr[:prefill_batch_size + 1],
            prefill_kv_page_indices=self.prefill_kv_page_indices,
            prefill_kv_last_page_len=self.prefill_kv_last_page_len[:prefill_batch_size],

            decode_qo_indptr=self.decode_qo_indptr[:decode_batch_size + 1],
            decode_kv_page_indptr=self.decode_kv_page_indptr[:decode_batch_size + 1],
            decode_kv_page_indices=self.decode_kv_page_indices,
            decode_kv_last_page_len=self.decode_kv_last_page_len[:decode_batch_size],

            append_qo_indptr_tensor=self.append_qo_indptr_tensor[:batch_size + 1],
            append_kv_page_indices_tensor=self.append_kv_page_indices_tensor,
            append_kv_page_indptr_tensor=self.append_kv_page_indptr_tensor[:batch_size + 1],
            append_kv_last_page_len_tensor=self.append_kv_last_page_len_tensor[:batch_size],
        )

    def prepare_buffer_bulk(self, d):
        for key, value in d.items():
            oldvalue = getattr(self, key)
            copy_tensor_portion(oldvalue, value)
        return

    def get_prefill_buf(self):
        return dict(
            qo_indptr_buf=self.prefill_qo_indptr,
            paged_kv_indptr_buf=self.prefill_kv_page_indptr,
            paged_kv_indices_buf=self.prefill_kv_page_indices,
            paged_kv_last_page_len_buf=self.prefill_kv_last_page_len
        )

    def get_decode_buf(self):
        return dict(
            qo_indptr_buf=self.decode_qo_indptr,
            paged_kv_indptr_buf=self.decode_kv_page_indptr,
            paged_kv_indices_buf=self.decode_kv_page_indices,
            paged_kv_last_page_len_buf=self.decode_kv_last_page_len
        )


class FlashinferCUDAAttentionWrapper(BaseAttentionWrapper):
    _inst = None

    def init(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        block_size: int,
        device: torch.device,
    ):
        super().init(model_config, parallel_config, block_size, device)

        self.is_metadata_initialized = False
        self.is_profiling_iteration = False
        self.contains_prefill = False
        self.contains_decode = False
        self.num_prefill_tokens = 0
        self.num_decode_tokens = 0
        self.num_total_tokens = 0

        workspace_buffer = torch.empty(
            128 * 1024 * 1024, dtype=torch.uint8, device=device
        )
        self.wrapper = self.default_wrapper = BatchPrefillWithPagedKVCacheWrapper(
            workspace_buffer, "NHD"
        )

        self.buffer = SharedBuffer(
            prefill_batch_size=1024,  # max prefill batch size
            decode_batch_size=1024,  # max decode batch size
            device=device,
            total_page_count=65535  # max page count we can get. OK to set it bigger.
        )
        self.append_qo_indptr_tensor = self.buffer.append_qo_indptr_tensor
        self.append_kv_page_indices_tensor = self.buffer.append_kv_page_indices_tensor
        self.append_kv_page_indptr_tensor = self.buffer.append_kv_page_indptr_tensor
        self.append_kv_last_page_len_tensor = self.buffer.append_kv_last_page_len_tensor

        # TODO: Tweak the appropriate batch size..
        batch_size_to_sample = list(range(1, 1024 + 1))

        self._wrappers = {0: None}
        for batch_size in batch_size_to_sample:
            buffer = self.buffer.prepare_sliced_buffer(batch_size, 0)
            wrapper = BatchPrefillWithPagedKVCacheWrapper(
                workspace_buffer, "NHD", use_cuda_graph=True,
                **buffer.get_prefill_buf()
            )
            self._wrappers[batch_size] = wrapper
            pass

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

    def set_wrapper(self, batch_size):
        self.wrapper = self._wrappers[batch_size]
        return self.wrapper

    def begin_forward(
        self,
        seq_metadata_list: List[SequenceMetadata],
    ) -> None:
        qo_indptr: List[int] = [0]
        kv_page_indices: List[int] = []
        kv_last_page_len: List[int] = []
        kv_page_indptr: List[int] = [0]

        self.is_profiling_iteration = False
        self.is_metadata_initialized = True

        self.contains_prefill = False
        self.contains_decode = False

        self.is_profiling_iteration = any(
            seq_metadata.block_table is None
            for seq_metadata in seq_metadata_list
        )
        if self.is_profiling_iteration:
            return

        num_prefill_tokens = 0
        num_decode_tokens = 0
        batch_size = len(seq_metadata_list)

        # Handle the prompt logic
        for seq_metadata in seq_metadata_list:
            if not seq_metadata.is_prompt:
                continue
            prompt_chunk_len = seq_metadata.prompt_chunk_len
            processed_prompt_len = seq_metadata.seq.get_num_prompt_tokens_processed()
            current_total_len = processed_prompt_len + prompt_chunk_len

            # Compute the kv page indices for the prompt tokens.
            num_blocks_in_use = (
                                    current_total_len + self.block_size - 1
                                ) // self.block_size
            num_prefill_tokens += prompt_chunk_len

            qo_indptr.append(qo_indptr[-1] + prompt_chunk_len)
            kv_page_indices.extend(seq_metadata.block_table[:num_blocks_in_use])
            kv_page_indptr.append(
                kv_page_indptr[-1] + num_blocks_in_use
            )
            kv_last_page_len.append(
                current_total_len % self.block_size or self.block_size
            )

            pass

        # Handle the decoding logic
        for seq_metadata in seq_metadata_list:
            if seq_metadata.is_prompt:
                continue

            context_len = seq_metadata.seq.get_len()
            qo_indptr.append(qo_indptr[-1] + 1)
            num_blocks_in_use = (context_len + self.block_size - 1) // self.block_size
            kv_page_indices.extend(seq_metadata.block_table[:num_blocks_in_use])
            kv_page_indptr.append(kv_page_indptr[-1] + num_blocks_in_use)
            kv_last_page_len.append(
                context_len % self.block_size or self.block_size
            )

            num_decode_tokens += 1
            pass

        assert not self.is_profiling_iteration, f"wrapper is not set during profiling."

        self.num_prefill_tokens = num_prefill_tokens
        self.num_decode_tokens = num_decode_tokens
        self.num_total_tokens = num_prefill_tokens + num_decode_tokens

        wrapper = self.set_wrapper(batch_size)
        if batch_size > 0:
            wrapper.begin_forward(
                self.to_int_tensor(qo_indptr),
                self.to_int_tensor(kv_page_indptr),
                self.to_int_tensor(kv_page_indices),
                self.to_int_tensor(kv_last_page_len),
                self.num_q_heads,
                self.num_kv_heads,
                self.head_dim,
                self.block_size,
            )

        self.append_qo_indptr_tensor[:len(qo_indptr)] = self.to_int_tensor(qo_indptr)
        self.append_kv_page_indices_tensor[:len(kv_page_indices)] = self.to_int_tensor(kv_page_indices)
        self.append_kv_page_indptr_tensor[:len(kv_page_indptr)] = self.to_int_tensor(kv_page_indptr)
        self.append_kv_last_page_len_tensor[:len(kv_last_page_len)] = self.to_int_tensor(kv_last_page_len)
        return

    def end_forward(self):
        self.is_metadata_initialized = False
        if self.is_profiling_iteration:
            return

        if self.wrapper:
            self.wrapper.end_forward()
        self.wrapper = None
        return

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

        with self.get_timer(OperationMetrics.ATTN_INPUT_RESHAPE, layer_id):
            query = query.contiguous().reshape(-1, self.num_q_heads, self.head_dim)
            key = key.contiguous().reshape(-1, self.num_kv_heads, self.head_dim)
            value = value.contiguous().reshape(-1, self.num_kv_heads, self.head_dim)

        output = torch.empty_like(query)

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

        with self.get_timer(OperationMetrics.ATTN_PREFILL, layer_id):
            num_tokens = self.num_prefill_tokens + self.num_decode_tokens
            wrapper = self.wrapper

            if num_tokens > 0:
                output[: num_tokens] = wrapper.forward(
                    query[: num_tokens],
                    kv_cache,
                    pos_encoding_mode="NONE",
                    sm_scale=softmax_scale,
                )

        with self.get_timer(OperationMetrics.ATTN_OUTPUT_RESHAPE, layer_id):
            output = output.reshape(-1, self.num_q_heads * self.head_dim)

        return output
