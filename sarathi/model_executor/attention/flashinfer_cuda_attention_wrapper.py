import dataclasses
from typing import List, Optional

import torch
from flashinfer import BatchPrefillWithPagedKVCacheWrapper, append_paged_kv_cache

from sarathi.config import ModelConfig, ParallelConfig
from sarathi.core.datatypes.sequence import SequenceMetadata
from sarathi.metrics.constants import OperationMetrics
from sarathi.model_executor.attention.base_attention_wrapper import BaseAttentionWrapper


def copy_tensor_portion(source, target):
    """
    Copies the portion of the target tensor to the source tensor based on the shape of the target tensor.

    Args:
        source (torch.Tensor): The tensor to copy to.
        target (torch.Tensor): The tensor to copy from.
    """
    slices = tuple(slice(0, min(s, t)) for s, t in zip(source.shape, target.shape))
    source[slices] = target[slices]


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

        prefill_workspace_buffer = torch.empty(
            128 * 1024 * 1024, dtype=torch.uint8, device=device
        )
        decode_workspace_buffer = torch.empty(
            128 * 1024 * 1024, dtype=torch.uint8, device=device
        )
        self.prefill_wrapper = self.default_prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(
            prefill_workspace_buffer, "NHD"
        )
        self.decode_wrapper = self.default_decode_wrapper = BatchPrefillWithPagedKVCacheWrapper(
            decode_workspace_buffer, "NHD"
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

        self._prefill_wrappers = {0: None}
        for batch_size in batch_size_to_sample:
            buffer = self.buffer.prepare_sliced_buffer(batch_size, 0)
            wrapper = BatchPrefillWithPagedKVCacheWrapper(
                prefill_workspace_buffer, "NHD",
                **buffer.get_prefill_buf()
            )
            self._prefill_wrappers[batch_size] = wrapper
            pass

        self._decode_wrappers = {0: None}
        for batch_size in batch_size_to_sample:
            buffer = self.buffer.prepare_sliced_buffer(0, batch_size)
            wrapper = BatchPrefillWithPagedKVCacheWrapper(
                decode_workspace_buffer, "NHD",
                **buffer.get_decode_buf()
            )
            self._decode_wrappers[batch_size] = wrapper
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

        self.num_prefill_tokens = prefill_qo_indptr[-1]
        self.num_total_tokens = self.num_prefill_tokens + len(decode_qo_indptr) - 1
        self.num_decode_tokens = self.num_total_tokens - self.num_prefill_tokens

        self.prefill_wrapper = self._prefill_wrappers[self.num_prefill_tokens]
        self.decode_wrapper = self._decode_wrappers[self.num_decode_tokens]

        if self.contains_prefill:
            self.prefill_wrapper.begin_forward(
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
            self.decode_wrapper.begin_forward(
                self.to_int_tensor(decode_qo_indptr),
                self.to_int_tensor(decode_kv_page_indptr),
                self.to_int_tensor(decode_kv_page_indices),
                self.to_int_tensor(decode_kv_last_page_len),
                self.num_q_heads,
                self.num_kv_heads,
                self.head_dim,
                self.block_size,
            )

        append_qo_indptr_tensor = prefill_qo_indptr[:-1] + [x + prefill_qo_indptr[-1] for x in decode_qo_indptr]
        append_kv_page_indices_tensor = prefill_kv_page_indices + decode_kv_page_indices
        append_kv_page_indptr_tensor = prefill_kv_page_indptr[:-1] + [x + prefill_kv_page_indptr[-1] for x in
                                                                      decode_kv_page_indptr]
        append_kv_last_page_len_tensor = prefill_kv_last_page_len + decode_kv_last_page_len

        # Assign the value to captured tensor.
        self.append_qo_indptr_tensor[:len(append_qo_indptr_tensor)] = self.to_int_tensor(
            append_qo_indptr_tensor
        )
        self.append_kv_page_indices_tensor[:len(append_kv_page_indices_tensor)] = self.to_int_tensor(
            append_kv_page_indices_tensor
        )
        self.append_kv_page_indptr_tensor[:len(append_kv_page_indptr_tensor)] = self.to_int_tensor(
            append_kv_page_indptr_tensor
        )
        self.append_kv_last_page_len_tensor[:len(append_kv_last_page_len_tensor)] = self.to_int_tensor(
            append_kv_last_page_len_tensor
        )


    def end_forward(self):
        self.is_metadata_initialized = False
        if self.is_profiling_iteration:
            return

        if self.contains_prefill:
            self.prefill_wrapper.end_forward()

        if self.contains_decode:
            self.decode_wrapper.end_forward()

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
            if self.contains_prefill:
                output[: self.num_prefill_tokens] = self.prefill_wrapper.forward(
                    query[: self.num_prefill_tokens],
                    kv_cache,
                    pos_encoding_mode="NONE",
                    sm_scale=softmax_scale,
                )

        with self.get_timer(OperationMetrics.ATTN_DECODE, layer_id):
            if self.contains_decode:
                output[self.num_prefill_tokens: self.num_total_tokens] = (
                    self.decode_wrapper.forward(
                        query[self.num_prefill_tokens: self.num_total_tokens],
                        kv_cache,
                        pos_encoding_mode="NONE",
                        sm_scale=softmax_scale,
                    )
                )

        with self.get_timer(OperationMetrics.ATTN_OUTPUT_RESHAPE, layer_id):
            output = output.reshape(-1, self.num_q_heads * self.head_dim)

        return output
