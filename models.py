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
    

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, embed_size):
        super().__init__()
        self.in_channels = in_channels
        self.embed_size = embed_size
        # A single Layer is used to map all input patches to the output embedding dimension.
        # i.e. each image patch will share the weights of this embedding layer.
        self.embed_layer = nn.Linear(in_features=in_channels, out_features=embed_size)
    
    def forward(self, x):
        assert len(x.size()) == 3
        B, T, C = x.size()
        x = self.embed_layer(x)
        return x
    
class VisionTransformer(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, embedding_size):
        super().__init__()
        self.i2p = ImageToPatches(image_size, patch_size)
        self.pe  = PatchEmbedding(patch_size*patch_size*in_channels, embedding_size)
        num_patches = (image_size // patch_size)**2
        self.position_embedding = nn.Parameter(torch.randn(num_patches, embedding_size))

    def forward(self, x):
        x   = self.i2p(x)
        x   = self.pe(x)
        
        return x + self.position_embedding
        
    
class MLP(nn.Module):
    def __init__(self, embed_size, dropout):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(embed_size, embed_size*4),
            nn.GELU(),
            nn.Linear(embed_size*4, embed_size),
            nn.Dropout(p=dropout),
        )

    def forward(self, x):
        return self.layers(x)
    
class SelfAttention(nn.Module):
    def __init__(self, embed_size, num_heads, dropout) -> None:
        super().__init__()
        self.embed_size = embed_size
        self.layer_norm = nn.LayerNorm(embed_size)
        self.mult_head_att = nn.MultiheadAttention(embed_size, num_heads, dropout, batch_first=True)
        self.layer_norm2 = nn.LayerNorm(embed_size)
        self.mlp = MLP(embed_size, dropout)

    def forward(self, x):
        y = self.layer_norm(x)
        x = x + self.mult_head_att(y, y, y, need_weights=False)[0]
        x = x + self.mlp(self.layer_norm2(x))

        return x
    
class OutputProjection(nn.Module):
    def __init__(self, image_size, patch_size, embd_size, out_dim):
        super().__init__()
        self.patch = patch_size
        self.out_dim = out_dim
        self.proj = nn.Linear(embd_size, patch_size*patch_size*out_dim)
        self.fold = nn.Fold(output_size=(image_size, image_size), kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, T, C = x.shape
        x = self.proj(x) # B x T x Patch_size**2 x OutDIm

        x = x.permute(0, 2, 1)
        x = self.fold(x)

        return x