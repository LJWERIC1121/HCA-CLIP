"""
Supervised Contrastive Learning Loss
Based on "Supervised Contrastive Learning" (Khosla et al., NeurIPS 2020)
https://arxiv.org/abs/2004.11362
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Learning Loss

    For few-shot learning with rotation augmentation:
    - Same image with different rotations are positive pairs
    - Different images with same class are positive pairs
    - Different classes are negative pairs
    """

    def __init__(self, temperature=0.07, base_temperature=0.07):
        """
        Args:
            temperature: temperature parameter for contrastive loss
            base_temperature: base temperature for normalization
        """
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, labels, rotation_multiplier=2):
        """
        Compute supervised contrastive loss

        Args:
            features: [B*k, D] features where k is rotation_multiplier
                     B is original batch size, k is number of rotations
            labels: [B*k] labels (repeated for each rotation)
            rotation_multiplier: number of rotations per image (default: 2)

        Returns:
            loss: scalar supervised contrastive loss
        """
        device = features.device
        batch_size = features.shape[0] // rotation_multiplier  # Original batch size B

        # Normalize features
        features = F.normalize(features, dim=1)  # [B*k, D]

        # Compute similarity matrix: [B*k, B*k]
        similarity_matrix = torch.matmul(features, features.T) / self.temperature

        # Create mask for positive pairs
        # Reshape labels to [B, k] to identify augmented views
        labels_reshaped = labels.view(batch_size, rotation_multiplier)  # [B, k]

        # Create masks
        mask = torch.zeros((batch_size * rotation_multiplier, batch_size * rotation_multiplier),
                          dtype=torch.bool, device=device)

        for i in range(batch_size):
            # Indices for all rotations of image i
            start_i = i * rotation_multiplier
            end_i = (i + 1) * rotation_multiplier

            for j in range(batch_size):
                start_j = j * rotation_multiplier
                end_j = (j + 1) * rotation_multiplier

                # Positive pairs: same class (including same image with different rotations)
                if labels_reshaped[i, 0] == labels_reshaped[j, 0]:
                    mask[start_i:end_i, start_j:end_j] = True

        # Remove diagonal (self-similarity)
        mask.fill_diagonal_(False)

        # Create negative mask (for numerical stability)
        neg_mask = ~mask
        neg_mask.fill_diagonal_(False)

        # For numerical stability, subtract max
        logits_max, _ = similarity_matrix.max(dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()

        # Compute log_prob
        exp_logits = torch.exp(logits)

        # Sum of exp for negative samples (and all samples except self)
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) - torch.diag(exp_logits).unsqueeze(1) + 1e-8)

        # Compute mean of log-likelihood over positive pairs
        mask_sum = mask.sum(dim=1)
        # Only compute loss for samples that have positive pairs
        valid_samples = mask_sum > 0

        if valid_samples.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        mean_log_prob_pos = (log_prob * mask).sum(dim=1)[valid_samples] / mask_sum[valid_samples]

        # Loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss


class SupConLoss(nn.Module):
    """
    Simplified Supervised Contrastive Loss
    More stable implementation for few-shot learning
    """
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        Args:
            features: [N, D] normalized feature vectors
            labels: [N] class labels
        Returns:
            loss: scalar contrastive loss
        """
        device = features.device
        batch_size = features.shape[0]

        # Normalize features
        features = F.normalize(features, p=2, dim=1)

        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features.T) / self.temperature  # [N, N]

        # Create label mask: same class = positive pair
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)  # [N, N]

        # Remove diagonal (self-pairs)
        mask.fill_diagonal_(0)

        # For numerical stability
        logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()

        # Compute exp
        exp_logits = torch.exp(logits)

        # Remove diagonal from exp_logits
        exp_logits = exp_logits * (1 - torch.eye(batch_size, device=device))

        # Log sum exp of negative samples
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)

        # Compute mean of log-likelihood over positive
        mask_sum = mask.sum(1)

        # Only consider samples with at least one positive pair
        valid_mask = mask_sum > 0
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        mean_log_prob_pos = (mask * log_prob).sum(1)[valid_mask] / mask_sum[valid_mask]

        # Loss
        loss = -mean_log_prob_pos.mean()

        return loss
