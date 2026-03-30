from functools import partial
import torch

from torch.nn.attention.flex_attention import _mask_mod_signature
from torch.nn.attention.flex_attention import flex_attention, create_block_mask


def generate_sliding_window(window_size: int) -> _mask_mod_signature:

    def sliding_window(b, h, q_idx, kv_idx):
        del b, h # not used
        return torch.abs(q_idx - kv_idx) <= window_size // 2

    sliding_window_mask = sliding_window
    sliding_window_mask.__name__ = f"sliding_window_{window_size}"
    return sliding_window_mask


if __name__ == "__main__":
    B, H, SEQ_LEN, HEAD_DIM = 1, 16, 40320, 32
    WINDOW_SIZE = 512

    def make_tensor():
        return torch.ones(B, H, SEQ_LEN, HEAD_DIM, device="cuda", dtype=torch.float16)

    q, k, v = make_tensor(), make_tensor(), make_tensor()
    print(q.size())
    sliding_window_mask = generate_sliding_window(window_size=WINDOW_SIZE)

    block_mask = create_block_mask(
        sliding_window_mask, B=None, H=None, Q_LEN=SEQ_LEN, KV_LEN=SEQ_LEN, _compile=True
    )
    opt_flex_attention = torch.compile(partial(flex_attention, block_mask=block_mask))
    out = opt_flex_attention(q, k, v, block_mask=block_mask)
    print(f"Shape of output tensor: {list(out.shape)}")