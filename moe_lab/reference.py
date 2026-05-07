"""Independent direct expert evaluation without permutation or communication."""
import torch


@torch.no_grad()
def dense_reference(x, ids, weights, matrices, biases):
    output = torch.zeros((x.shape[0], matrices.shape[-1]), dtype=torch.float64, device=x.device)
    for token in range(x.shape[0]):
        for slot in range(ids.shape[1]):
            expert = int(ids[token, slot])
            if expert >= 0:
                value = x[token].double() @ matrices[expert].double() + biases[expert].double()
                output[token] += value * weights[token, slot].double()
    return output


@torch.no_grad()
def run_experts(batch, matrices, biases, rank, local_experts):
    out = batch.tokens.new_empty((batch.tokens.shape[0], matrices.shape[-1]))
    for local in range(local_experts):
        mask = batch.local_expert_ids == local
        global_id = rank * local_experts + local
        out[mask] = batch.tokens[mask] @ matrices[global_id] + biases[global_id]
    return out
