"""Distinct BF16 expert MLPs with masked, grouped and registered-output paths."""
import math
import torch


def create_weights(world, hidden, intermediate=128):
    generator = torch.Generator().manual_seed(313)
    first = torch.randn(world*4, hidden, intermediate, generator=generator).bfloat16() / math.sqrt(hidden)
    second = torch.randn(world*4, intermediate, hidden, generator=generator).bfloat16() / math.sqrt(intermediate)
    bias = torch.randn(world*4, hidden, generator=generator).bfloat16() * .05
    return first, second, bias


def counts_for(batch):
    counts = getattr(batch.context, 'counts', None) or getattr(batch.context, 'expert_counts', None)
    if counts is None:
        raise ValueError('Grouped computation requires validated host expert counts')
    return counts


@torch.no_grad()
def evaluate_experts(batch, weights, rank, *, contiguous=False):
    first, second, bias = weights
    output = torch.empty_like(batch.tokens)
    counts = counts_for(batch) if contiguous else None
    offset = 0
    for local in range(4):
        if contiguous:
            selected = slice(offset, offset+counts[local])
            offset += counts[local]
            if not counts[local]:
                continue
        else:
            selected = batch.local_expert_ids == local
        expert = rank*4+local
        output[selected] = torch.relu(batch.tokens[selected] @ first[expert]) @ second[expert] + bias[expert]
    return output


@torch.no_grad()
def evaluate_experts_into(batch, weights, rank, registered):
    """Write valid rows directly to a caller-provided padded expert-major buffer."""
    first, second, bias = weights
    counts = counts_for(batch)
    offset = 0
    for local, count in enumerate(counts):
        if count:
            expert = rank*4+local
            source = batch.packed_tokens[local,:count] if hasattr(batch,'packed_tokens') else batch.tokens[offset:offset+count]
            hidden = torch.relu(source @ first[expert])
            destination = registered[local, :count]
            torch.mm(hidden, second[expert], out=destination)
            destination.add_(bias[expert])
        offset += count
