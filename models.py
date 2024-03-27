import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., num_classes=2):
        super().__init__()

        self.num_classes = num_classes
        self.num_heads   = num_heads
        head_dim         = dim // num_heads
        self.scale       = qk_scale or head_dim**-.5

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=0.4, qkv_bias=False, qk_scale=None, act_layer_fn=nn.GELU, num_classes=2):
        super().__init__()
    
class VisionTransformer(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, embed_size):
        super().__init__()


class ImageToPatches(nn.Module):
    def __init__(self, image_size, patch_size):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        assert len(x.size()) == 4
        
        b, c, h, w = x.shape

        y = self.unfold(x)
        y = y.permute(0, 2, 1)
        # y = y.view(b, c, self.patch_size, self.patch_size, -1).permute(0, 4, 1, 2, 3)
        return y