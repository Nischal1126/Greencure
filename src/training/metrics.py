import torch

def get_accuracy(
    outputs : torch.Tensor,
    labels  : torch.Tensor
) -> float:

    _, predicted = outputs.max(1)
    correct      = predicted.eq(labels).sum().item()
    total        = labels.size(0)
    return 100.0 * correct / total
    
