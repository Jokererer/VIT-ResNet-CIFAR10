# https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit_pytorch.py

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

MIN_NUM_PATCHES = 16

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dropout = 0.):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask = None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask

        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads, dropout = dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))
    def forward(self, x, mask = None):
        for attn, ff in self.layers:
            x = attn(x, mask = mask)
            x = ff(x)
        return x

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels = 3, dropout = 0., emb_dropout = 0.):
        super().__init__()
        assert image_size % patch_size == 0, 'image dimensions must be divisible by the patch size'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        assert num_patches > MIN_NUM_PATCHES, f'your number of patches ({num_patches}) is way too small for attention to be effective. try decreasing your patch size'

        self.patch_size = patch_size

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)

        self.to_cls_token = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, num_classes)
        )

    def forward(self, img, mask = None):
        p = self.patch_size

        x = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = p, p2 = p)
        x = self.patch_to_embedding(x)
        b, n, _ = x.shape

        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x, mask)

        x = self.to_cls_token(x[:, 0])
        return self.mlp_head(x)

# #辅助函数，生成元组
#
# def pair(t):
#   return t if isinstance(t,tuple) else (t,t)
#
# #规范化层的类封装
# class PreNorm(nn.Module):
#   def __init__(self,dim,fn):
#     super().__init__()
#     self.norm = nn.LayerNorm(dim)   #正则化
#     self.fn = fn            #具体的操作
#   def forward(self,x,**kwargs):
#     return self.fn(self.norm(x),**kwargs)
#
#
# #FFN  前向传播
# class FeedForward(nn.Module):
#   def __init__(self,dim,hidden_dim,dropout=0.):
#     super().__init__()
#     #前向传播
#     self.net = nn.Sequential(
#         nn.Linear(dim,hidden_dim),
#         nn.GELU(),
#         nn.Dropout(dropout),
#         nn.Linear(hidden_dim,dim),
#         nn.Dropout(dropout)
#     )
#   def forward(self,x):
#     return self.net(x)
#
#
# # Attention
#
# class Attention(nn.Module):
#     def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
#         super().__init__()
#         inner_dim = dim_head * heads  # 计算最终进行全连接操作时输入神经元的个数
#         project_out = not (heads == 1 and dim_head == dim)  # 多头注意力并且输入和输出维度相同时为True
#
#         self.heads = heads  # 多头注意力中 头的个数
#         self.scale = dim_head ** -0.5  # 缩放操作，论文 Attention is all you need 中有介绍
#
#         self.attend = nn.Softmax(dim=-1)  # 初始化一个Softmax操作
#         self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)  # 对 Q，K，V三组向量进行线性操作
#
#         # 线性全连接，如果不是多头或者输入输出维度不相等，进行空操作
#         self.to_out = nn.Sequential(
#             nn.Linear(inner_dim, dim),
#             nn.Dropout(dropout)
#         ) if project_out else nn.Identity()
#
#     def forward(self, x):
#         b, n, _, h = *x.shape, self.heads  # 获得输入x的维度和多头注意力的"头"数
#         qkv = self.to_qkv(x).chunk(3, dim=-1)  # 先对Q，K，V进行线性操作，然后chunk乘3份
#         q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)  # 整理维度，获得 Q，K，V
#
#         dots = einsum('b h i d,b h j d -> b h i j', q, k) * self.scale  # Q,K向量先做点乘，计算相关性，然后除以缩放因子
#
#         attn = self.attend(dots)  # 做Softmax运算
#
#         out = einsum('b h i j,b h j d -> b h i d', attn, v)  # Softmax运算结果与Value向量相乘，得到最终结果
#         out = rearrange(out, 'b h n d -> b n (h d)')  # 重新整理维度
#         return self.to_out(out)  # 做线性的全连接操作或者空操作（空操作直接输出out)
#
#
# class Transformer(nn.Module):
#     def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
#         super().__init__()
#         self.layers = nn.ModuleList([])  # Transformer 包含多个编码器的叠加
#         for _ in range(depth):
#             # Transformer包含两大块：自注意力模块和前向传播模块
#             self.layers.append(nn.ModuleList([
#                 PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),  # 多头自注意力模块
#                 PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))  # 前向传播模块
#
#             ]))
#
#     def forward(self, x):
#         for attn, ff in self.layers:
#             # 自注意力模块和前向传播模块都使用了残差的模式
#             x = attn(x) + x
#             x = ff(x) + x
#         return x
#
#
# class ViT(nn.Module):
#     def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool='cls', channels=3,
#                  dim_head=64, dropout=0., emb_dropout=0.):
#         super().__init__()
#         image_height, image_width = pair(image_size)  # 原图大小 比如说 256  图块大小 32
#         patch_height, patch_width = pair(patch_size)  # 图块大小
#         assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'  # 保证一定能够完整切块
#         # patch数量
#         num_patches = (image_height // patch_height) * (
#                     image_width // patch_width)  # 获取图像切块的个数 # (256/32)*(256/32)也就是64块
#
#         # patch维度
#         patch_dim = channels * patch_height * patch_width  # 线性变换时的输入大小，即每一个图像宽，高和通道的乘积 图块拉成 3 * 32 * 32 变成一维的长度
#         assert pool in {'cls',
#                         'mean'}, 'pool type must be either cls(cls token) or mean(mean pooling)'  # 池化方法必须为cls或者mean
#
#         # 定义块嵌入,将高维向量转化为低维向量
#         self.to_patch_embedding = nn.Sequential(
#
#             # 展平，是将 3 维图像 reshape 为2维之后进行切分
#             Rearrange('b c (h p1)(w p2) -> b (h w)(p1 p2 c)', p1=patch_height, p2=patch_width),
#             # 将批量为b通道为c高为h*p1宽为w*p2的图像转化为批量为b个数为h*w维度为p1*p2*c的图像块
#             # 即，把b张c通道的图像分割成b*(h*w)张大小为p1*p2*c的图像块
#             # 线性变换，即全连接层，降维后维度为D，通过线性函数 把32*32*3 -> 1024                                                 # 例如：patch_size为16  (8, 3, 48, 48)->(8, 9, 768)
#             nn.Linear(patch_dim, dim),  # 对分割好的图像块进行线性处理（全连接），输入维度为每一个小块的所有像素个数，输出为dim（函数传入的参数）
#
#         )
#
#         self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))  # 位置编码，获取一组正太分布的数据用于训练
#         # 定义类别向量
#         self.cls_token = nn.Parameter(torch.randn(1, 1, dim))  # 分类令牌，可训练
#         self.dropout = nn.Dropout(emb_dropout)
#
#         self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)  # Transformer模块
#
#         self.pool = pool
#         self.to_latent = nn.Identity()  # 占位操作
#
#         self.mlp_head = nn.Sequential(
#             nn.LayerNorm(dim),  # 正则化
#             nn.Linear(dim, num_classes)  # 线性输出
#         )
#
#     def forward(self, img):
#         # 块嵌入
#         x = self.to_patch_embedding(img)  # 切块操作，shape(b,n,dim),b为批量，n为切块数目，dim为最终线性操作时输入的神经元个数
#         b, n, _ = x.shape  # shape(b,n,1024)
#
#         # 追加类别向量,可学习的嵌入向量,最后取该向量作为类别预测结果
#         cls_tokens = repeat(self.cls_token, '() n d ->b n d',
#                             b=b)  # 分类令牌，将self.cls_token（形状为1, 1, dim）赋值为shape (b, 1, dim)
#         x = torch.cat((cls_tokens, x), dim=1)  # 将分类令牌拼接到输入中，x的shape(b.n+1,1024)
#
#         # 追加位置编码，ViT的位置编码没有使用更新的2D位置嵌入方法，而是直接用的一维可学习的位置嵌入变量，原先是论文作者发现实际使用时2D并没有展现出比1D更好的效果
#         x += self.pos_embedding[:, :(n + 1)]  # 进行位置编码，shape (b, n+1, 1024)
#
#         # dropout
#         x = self.dropout(x)
#
#         # 输入到Transformer中
#         x = self.transformer(x)  # transformer操作
#
#         x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
#
#         x = self.to_latent(x)
#
#         # MLP
#         return self.mlp_head(x)  # 线性输出