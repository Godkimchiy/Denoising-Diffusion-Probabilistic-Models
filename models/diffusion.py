import math
import torch
import torch.nn as nn



# Sinusoidal positional embeddings for timesteps
class get_timestep_embedding(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.emb_dim = embedding_dim
    
    def forward(self, timesteps):
        assert len(timestepsl.shape)==1
        half_dim = self.emb_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
        emb = emb.to(device=timesteps.device)
        emb = timesteps.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:  # zero pad
            emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
        
        return emb



class LayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels = in_channels,
                                        out_channels = in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        """ 아래처럼 표현할 수 도 있음
        nn.Sequential(
            nn.Upsample(scale_factor = 2, mode = 'nearest'),
            nn.Conv2d(in_channels, in_channels, 3, 1, 1)
            )
        """
        x = torch.nn.functional.interpolate(
            x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    아래처럼 padding 후 Conv를 하는 것이랑
    그냥 nn.Conv2d(in_channels, in_channels, 4, 2, 1) 을 사용하는거나 dim은 동일하게 유지됨
    """
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


# Basic block module
class Block(nn.Module):
    def __init__(self, dim_in, dim_out, n_groups = 8):
        super().__init__()
        self.proj = nn.Conv2d(dim_in, dim_out, 3, padding = 1)
        self.norm = nn.GroupNorm(num_groups=n_groups, num_channels=dim_out, eps=1e-6, affine=True)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


# ResNet Block
class ResnetBlock(nn.Module):
    """
    DDIM 코드와 layer 순서가 다르고 dropout이 없다는 것이 차이점이다.
    """
    def __init__(self, dim_in, dim_out, *, time_emb_dim = None, n_groups = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim_in, dim_out, n_groups = n_groups)
        self.block2 = Block(dim_out, dim_out, n_groups = n_groups)
        self.res_conv = nn.Conv2d(dim_in, dim_out, 1) if dim_in != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)



class LinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            LayerNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return self.to_out(out)



class Attention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32, scale = 16):
        super().__init__()
        self.scale = scale
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q, k = map(l2norm, (q, k))

        sim = einsum('b h d i, b h d j -> b h i j', q, k) * self.scale
        attn = sim.softmax(dim = -1)

        out = einsum('b h i j, b h d j -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(out)



class Model(nn.Module):
    # def __init__(self, config):
    #     super().__init__()
    #     self.config = config
    #     ch, out_ch, ch_mult = config.model.ch, config.model.out_ch, tuple(config.model.ch_mult)
    #     num_res_blocks = config.model.num_res_blocks
    #     attn_resolutions = config.model.attn_resolutions
    #     dropout = config.model.dropout
    #     in_channels = config.model.in_channels
    #     resolution = config.data.image_size
    #     resamp_with_conv = config.model.resamp_with_conv
    #     num_timesteps = config.diffusion.num_diffusion_timesteps
    def __init__(self,
        dim, # ?
        init_dim = None, # 첫 컨볼루션에서의 out dimension
        out_dim = None,
        dim_mults=(1, 2, 4, 8), # dimension이 변하는 비율
        channels = 3, # input의 channel length
        resnet_block_groups = 8, # ?
        learned_variance = False,
        # learned_sinusoidal_cond = False,
        # learned_sinusoidal_dim = 16
    ):
        super().__init__()

        """ 1. determine dimensions for each block"""
        self.channels = channels # of input
        # init_dim = default(init_dim, dim) # 첫 conv의 out dim을 따로 설정 안해주면 dim을 init_dim으로 사용함
        self.init_conv = nn.Conv2d(channels, init_dim, 7, padding = 3) # down sampling 이전에 적용되는 첫 conv.
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)] # [init_dim, init_dim, 2init_dim, 4init_dim, 8init_dim]
        # dim_mults 개수만큼 enc, dec에 block을 두고, 각 block의 in/out dim은 아래와 같다
        # ([init_dim, init_dim], [init_dim, 2init_dim], [2init_dim, 4init_dim], [4init_dim, 8init_dim])
        in_out = list(zip(dims[:-1], dims[1:])) 
        
        block_klass = partial(ResnetBlock, groups = resnet_block_groups)


        """ 2. set time embeddings"""
        time_dim = dim * 4
        pos_emb = get_timestep_embedding(dim)
        fourier_dim = dim

        self.time_mlp = nn.Sequential(
            pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )


        """ 3. set layers"""
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        # DownSampling
        for idx, (dim_in, dim_out) in enumerate(in_out):
            is_last = idx>=(num_resolutions-1) # 마지막 block이면 True

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1)
            ]))


        # Middle
        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)


        # UpSampling
        for idx, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = idx>=(num_resolutions-1)

            self.up.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_out))),
                Downsample(dim_out, dim_in) if not is_last else nn.Conv2d(dim_out, dim_in, 3, padding = 1)
            ]))


        # End
        ## dimensions
        default_out_dim = channels * (1 if not learned_variance else 2) # output channel. variance를 상수로 두면 input channel과 동일, 학습하도록 하면 2배
        self.out_dim = default(out_dim, default_out_dim) # out_dim(output channel)을 지정하지 않으면, default_out_dim을 사용

        ## blocks
        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim = time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)


        def forward(self, x, time_step):
            x = self.init_conv(x) # input에 conv 적용함으로써 시작
            r = x.clone() # 아마 residual 로 쓰이는 듯한데 아직 잘 모르겠음

            t_emb = self.time_mlp(time_step) # timestep의 embedding

            h = [] # ??

            # down sampling
            for block1, block2, attn, downsample in self.downs:
                x = block1(x, time_step)
                h.append(x)

                x = block2(x, time_step)
                x = attn(x)
                h.append(x)

                x = downsample(x)

            # middle
            x = self.mid_block1(x, time_step)
            x = self.mid_attn(x)
            x = self.mid_block2(x, time_step)

            # up sampling
            for block1, block2, attn, upsample in self.ups:
                x = torch.cat((x, h.pop()), dim = 1)
                x = block1(x, time_step)

                x = torch.cat((x, h.pop()), dim = 1)
                x = block2(x, time_step)
                x = attn(x)

                x = upsample(x)

            # residual block. down/up sampling을 거친 x와 그 이전 x인 r을 더해줌
            x = torch.cat((x, r), dim = 1)

            # last of UNet
            x = self.final_res_block(x, t)
            x = self.final_conv(x)
            return x