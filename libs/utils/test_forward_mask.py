import torch
import torch.nn as nn
import torch.nn.functional as F

# Modified drop_path function
def drop_path(x, mask=None, scale_by_keep: bool = True):
    if mask is None:
        return x
    if scale_by_keep:
        keep_prob = mask.float().mean().item()
        scale_factor = 1 / keep_prob if keep_prob > 0. else 0.
    else:
        scale_factor = 1.0
    output = torch.zeros_like(x)
    output[mask] = x[mask] * scale_factor
    return output

# Assuming Block is similar to the provided Bottleneck class with a slight modification for testing
class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(3)
        self.attn = nn.Identity()  # Assuming some attention mechanism
        self.norm2 = nn.BatchNorm2d(3)
        self.mlp = nn.Identity()  # Assuming some MLP block
        self.drop_path1 = lambda x, mask: drop_path(x, mask=mask, scale_by_keep=True)
        self.drop_path2 = lambda x, mask: drop_path(x, mask=mask, scale_by_keep=True)

    def forward(self, x, mask=None):
        # Apply first drop path with mask
        residual = self.attn(self.norm1(x))
        x = x + self.drop_path1(residual, mask=mask)
        
        # Apply second drop path with mask
        residual = self.mlp(self.norm2(x))
        x = x + self.drop_path2(residual, mask=mask)

        return x

# Testing the Block
block = Block()

# Creating a dummy input tensor
batch_size, channels, height, width = 4, 3, 2, 2
dummy_input = torch.randn(batch_size, channels, height, width)

# Mask with two samples dropped and two kept
mask = torch.tensor([True, False, True, False])

# Forwarding the dummy input through the block without a mask
output_without_mask = block(dummy_input)
print("Output without mask:", output_without_mask)

# Forwarding the dummy input through the block with a mask
output_with_mask = block(dummy_input, mask=mask)
print("Output with mask:", output_with_mask)

# Verify that the masked outputs are as expected
masked_elements = output_with_mask[~mask]
expected_elements = torch.zeros_like(masked_elements)
assert torch.allclose(masked_elements, expected_elements, atol=1e-6), "Masked outputs should be zero"
print("Test passed!")
