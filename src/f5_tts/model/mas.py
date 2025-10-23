"""
Monotonic Alignment Search (MAS) wrapper using Cython implementation.

This module provides a high-level interface to the Cython MAS algorithm
for computing optimal monotonic alignments between text and mel frames.
"""

import torch

from f5_tts.utils.monotonic_align import maximum_path


def monotonic_alignment_search(
    similarity: torch.Tensor,  # [b, nt, n]
    text_lens: torch.Tensor,  # [b]
    mel_lens: torch.Tensor,  # [b]
) -> torch.Tensor:
    """
    Compute optimal monotonic alignment between text and mel frames using MAS.

    Args:
        similarity: [b, nt, n] - similarity scores between text tokens and mel frames
                   Higher scores indicate better alignment
        text_lens: [b] - actual text lengths (number of valid text tokens)
        mel_lens: [b] - actual mel lengths (number of valid mel frames)

    Returns:
        attn: [b, nt, n] - binary alignment matrix where attn[b, i, j] = 1
              indicates text token i aligns to mel frame j

    Note:
        The Cython implementation expects input shape [b, t_x, t_y] where:
        - t_x is text length (source)
        - t_y is mel length (target)
        The algorithm finds a monotonic path from (0,0) to (t_x, t_y).
    """
    batch_size, max_text_len, max_mel_len = similarity.shape
    device = similarity.device
    dtype = similarity.dtype

    # Create mask for valid regions [b, nt, n]
    text_mask = torch.arange(max_text_len, device=device)[None, :, None] < text_lens[:, None, None]
    mel_mask = torch.arange(max_mel_len, device=device)[None, None, :] < mel_lens[:, None, None]
    mask = text_mask & mel_mask  # [b, nt, n]

    # Transpose similarity to [b, t_y, t_x] format expected by Cython MAS
    # The algorithm treats first dim as target (mel), second as source (text)
    # Use .contiguous() to ensure C-contiguous memory layout required by Cython
    similarity_transposed = similarity.transpose(1, 2).contiguous()  # [b, n, nt]
    mask_transposed = mask.transpose(1, 2).contiguous()  # [b, n, nt]

    # Run Cython MAS: returns path [b, n, nt]
    path = maximum_path(similarity_transposed, mask_transposed)

    # Transpose back to [b, nt, n] format
    attn = path.transpose(1, 2).to(dtype=dtype)

    return attn


def compute_duration_from_alignment(
    attn: torch.Tensor,  # [b, nt, n]
    text_lens: torch.Tensor,  # [b]
) -> torch.Tensor:
    """
    Extract duration (number of mel frames) for each text token from alignment.

    Args:
        attn: [b, nt, n] - binary alignment matrix
        text_lens: [b] - actual text lengths

    Returns:
        durations: [b, nt] - number of mel frames aligned to each text token
    """
    # Sum over mel dimension to get duration per text token
    durations = attn.sum(dim=2)  # [b, nt]

    # Mask out padding tokens
    text_mask = torch.arange(attn.shape[1], device=attn.device)[None, :] < text_lens[:, None]
    durations = durations * text_mask.float()

    return durations


def convert_pad_shape(pad_shape):
    """
    Convert padding shape format for torch.nn.functional.pad.

    Args:
        pad_shape: List of [left, right] pairs for each dimension

    Returns:
        Flattened list in reverse order for F.pad format
    """
    inverted_shape = pad_shape[::-1]
    pad_shape = [item for sublist in inverted_shape for item in sublist]
    return pad_shape


def sequence_mask(length, max_length=None):
    """
    Create boolean mask from lengths.

    Args:
        length: [N] - lengths for each sequence
        max_length: Maximum length (defaults to max(length))

    Returns:
        mask: [N, max_length] - boolean mask where mask[i, j] = (j < length[i])
    """
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


def generate_path(duration, mask):
    """
    Generate monotonic alignment path from duration predictions.

    Converts per-token duration predictions into a binary alignment matrix,
    creating a monotonic path that assigns consecutive mel frames to text tokens
    according to their predicted durations.

    Args:
        duration: [b, nt] - predicted durations (number of mel frames per text token)
        mask: [b, nt, n] - valid region mask (text_mask & mel_mask)

    Returns:
        path: [b, nt, n] - binary alignment matrix where path[b, i, j] = 1
              indicates text token i is aligned to mel frame j

    Example:
        duration = [[2, 3, 1]]  # 3 text tokens with durations 2, 3, 1
        result   = [[1, 1, 0, 0, 0, 0],   # token 0 → frames 0-1
                    [0, 0, 1, 1, 1, 0],   # token 1 → frames 2-4
                    [0, 0, 0, 0, 0, 1]]   # token 2 → frame 5
    """
    device = duration.device
    b, t_x, t_y = mask.shape

    # Compute cumulative durations to find frame boundaries
    # cum_duration[b, i] = sum of durations for tokens 0..i
    cum_duration = torch.cumsum(duration, 1)  # [b, nt]

    # Create sequence masks for each cumulative duration
    # This creates a mask where mask[i, j] = (j < cum_duration[i])
    cum_duration_flat = cum_duration.view(b * t_x)  # [b*nt]
    path = sequence_mask(cum_duration_flat, t_y).to(mask.dtype)  # [b*nt, n]
    path = path.view(b, t_x, t_y)  # [b, nt, n]

    # Subtract previous cumulative mask to get per-token assignment
    # path[i] - path[i-1] gives frames belonging only to token i
    path = path - torch.nn.functional.pad(path, convert_pad_shape([[0, 0], [1, 0], [0, 0]]))[:, :-1]

    # Apply mask to ensure validity (respect actual text/mel lengths)
    path = path * mask

    return path
