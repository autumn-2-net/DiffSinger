import math
from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from utils.hparams import hparams


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv1d(hidden_dim, dim, 1),
                                    nn.GroupNorm(32, dim))

    def forward(self, x):
        # b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) t -> b h c t", h=self.heads), qkv
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c t -> b (h c) t", h=self.heads,)
        return self.to_out(out)

class LinearAttentionC(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32,cdim=256):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_q = nn.Conv1d(dim, hidden_dim , 1, bias=False)
        self.to_kv = nn.Conv1d(cdim, hidden_dim * 2, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv1d(hidden_dim, dim, 1),
                                    nn.GroupNorm(32, dim))

    def forward(self, x,cs):
        # b, c, h, w = x.shape
        q = self.to_q(x)
        k,v = self.to_kv(cs).chunk(2, dim=1)



        q, k, v = map(
            lambda t: rearrange(t, "b (h c) t -> b h c t", h=self.heads),(q,k,v)
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c t -> b (h c) t", h=self.heads, )
        return self.to_out(out)




class attLIN(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32,cdim=512):
        super().__init__()
        self.satt=LinearAttention(dim, heads=heads, dim_head=dim_head)
        self.catt=LinearAttentionC(dim, heads=heads, dim_head=dim_head,cdim=cdim)
        self.mlp=nn.Sequential(nn.Conv1d(dim,int(dim*2.5),kernel_size=1),nn.SiLU(),nn.Conv1d(int(dim*2.5),dim,kernel_size=1))
        self.nm=nn.GroupNorm(32, dim)
        self.nm1= nn.GroupNorm(32, dim)
        self.nm2 = nn.GroupNorm(32, dim)
        self.nm3= nn.GroupNorm(32, cdim)

    def forward(self, x,c):
        x = self.satt(self.nm(x))+x
        x = self.catt(self.nm1(x),self.nm3(c)) + x
        x = self.mlp(self.nm2(x)) + x
        return x








class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = nn.Conv1d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock(nn.Module):


    def __init__(self, dim, dim_out,  time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2))
            if time_emb_dim is not None
            else None
        )

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if self.mlp is not None and time_emb is not None:
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, "b c -> b c  1")
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)

class GLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        out, gate = x.chunk(2, dim=self.dim)

        return out * gate.sigmoid()
        # return torch.tanh(out) * gate.sigmoid()
class downblock(nn.Module):
    def __init__(self, down,indim,outdim):
        super().__init__()
        self.c=nn.Conv1d(indim,outdim*2,kernel_size=down*2,stride=down,padding=down//2)
        self.act=GLU(1)
        self.out = nn.Conv1d(outdim,outdim,kernel_size=3,padding=1)
        self.act1=nn.GELU()

    def forward(self, x):
        return self.act1(self.out (self.act(self.c(x))))


class upblock(nn.Module):
    def __init__(self, ups, indim, outdim):
        super().__init__()
        self.c = nn.ConvTranspose1d(indim, outdim * 2, kernel_size=ups * 2, stride=ups, padding=ups // 2)
        self.act = GLU(1)
        self.out = nn.Conv1d(outdim, outdim, kernel_size=3, padding=1)
        self.act1 = nn.GELU()

    def forward(self, x):
        return self.act1(self.out(self.act(self.c(x))))


class unetb(nn.Module):
    def __init__(self,downs=[2,2,2],dim=[256,384,512],latentdim=512,indim=128,cdim=256):
        super().__init__()
        self.incon=nn.Conv1d(indim,dim[0],1,padding=0)
        self.outcon = nn.Conv1d(dim[0], indim, 1, padding=0)
        self.outcon1= nn.Conv1d(dim[0], dim[0], 1, padding=0)
        self.dlres = nn.ModuleList()
        self.dlatt = nn.ModuleList()
        self.dldd = nn.ModuleList()
        dims=dim.copy()
        dims.append(latentdim)
        for idx,i in enumerate(downs):
            self.dlres.append(nn.ModuleList([ResnetBlock(dim[idx],dim[idx],time_emb_dim=512,groups=32) ,ResnetBlock(dim[idx],dim[idx],time_emb_dim=512,groups=32)]))
            h=dim[idx]//32
            self.dlatt.append(nn.ModuleList([attLIN(dim[idx],heads=h,cdim=cdim),attLIN(dim[idx],heads=h,cdim=cdim)]))
            self.dldd.append(downblock(i,dims[idx],dims[idx+1]))

        self.upres = nn.ModuleList()
        self.upatt = nn.ModuleList()
        self.updd = nn.ModuleList()


        ups = downs.copy()
        ups.reverse()
        upsd=dim.copy()
        upsd.reverse()

        upsds=dims.copy()
        upsds.reverse()



        for idx,i in enumerate(ups):
            self.upres.append(nn.ModuleList([ResnetBlock(upsd[idx]*2,upsd[idx],time_emb_dim=512,groups=32) ,ResnetBlock(upsd[idx]*2,upsd[idx],time_emb_dim=512,groups=32)]))
            h = dim[idx] // 32
            self.upatt.append(nn.ModuleList([attLIN(upsd[idx],heads=h,cdim=cdim),attLIN(upsd[idx],heads=h,cdim=cdim)]))
            self.updd.append(upblock(i,upsds[idx],upsds[idx+1]))

        self.mres1=ResnetBlock(latentdim,latentdim,time_emb_dim=512,groups=32)
        # self.matt = PreNorm(latentdim,LinearAttention(latentdim,heads=32))
        self.matt = attLIN(latentdim,heads=16,cdim=cdim)
        self.mres2 = ResnetBlock(latentdim, latentdim, time_emb_dim=512,groups=32)

        self.outres=nn.ModuleList([ResnetBlock(dim[0]*2,dim[0],time_emb_dim=512,groups=32),])
        self.outatt=attLIN(dim[0],heads=dim[0]//32,cdim=cdim)

    def forward(self, x,time_emb,cs):
        fff=[]

        x=self.incon(x)
        x = F.relu(x)
        res11=x.clone()


        for res ,att,don in zip(self.dlres,self.dlatt,self.dldd  ):
            cccc=[]
            for idx,ii in enumerate(res):
                x=ii(x,time_emb)
                x = att[idx](x, cs)
                cccc.append(x)
            # x=att[1](x,cs)
            # cccc[1]=x
            fff.append(cccc)
            x=don(x)
        x=self.mres2(self.matt (self.mres1(x,time_emb), cs), time_emb)
        fff.reverse()


        for res ,att,up ,fautres in zip(self.upres ,self.upatt,self.updd,fff ):
            x=up(x)
            # x=fautres+x

            for idx,ii in enumerate(res):
                x = torch.cat([x, fautres[idx]], dim=1)
                x=ii(x,time_emb)
                x = att[idx](x, cs)

        x = torch.cat([x, res11], dim=1)
        for ii in self.outres:
            x = ii(x, time_emb)
        x=self.outatt(x, cs)
        x = F.relu(self.outcon1(x))
        return self.outcon(x)

# class unetb(nn.Module):
#     def __init__(self,downs=[2,2,2,2],dim=[192,192,384,512],latentdim=1024,indim=128,cdim=256):
#         super().__init__()
#         self.incon=nn.Conv1d(indim,dim[0],1,padding=0)
#         self.outcon = nn.Conv1d(dim[0], indim, 1, padding=0)
#         self.dlres = nn.ModuleList()
#         self.dlatt = nn.ModuleList()
#         self.dldd = nn.ModuleList()
#         dims=dim.copy()
#         dims.append(latentdim)
#         for idx,i in enumerate(downs):
#             self.dlres.append(nn.ModuleList([ResnetBlock(dim[idx],dim[idx],time_emb_dim=512,groups=32) ,ResnetBlock(dim[idx],dim[idx],time_emb_dim=512,groups=32)]))
#             h=dim[idx]//32
#             self.dlatt.append(nn.ModuleList([attLIN(dim[idx],heads=h,cdim=cdim),attLIN(dim[idx],heads=h,cdim=cdim)]))
#             self.dldd.append(downblock(i,dims[idx],dims[idx+1]))
#
#         self.upres = nn.ModuleList()
#         self.upatt = nn.ModuleList()
#         self.updd = nn.ModuleList()
#
#
#         ups = downs.copy()
#         ups.reverse()
#         upsd=dim.copy()
#         upsd.reverse()
#
#         upsds=dims.copy()
#         upsds.reverse()
#
#
#
#         for idx,i in enumerate(ups):
#             self.upres.append(nn.ModuleList([ResnetBlock(upsd[idx]*2,upsd[idx],time_emb_dim=512,groups=32) ,ResnetBlock(upsd[idx]*2,upsd[idx],time_emb_dim=512,groups=32)]))
#             h = dim[idx] // 32
#             self.upatt.append(nn.ModuleList([attLIN(upsd[idx],heads=h,cdim=cdim),attLIN(upsd[idx],heads=h,cdim=cdim)]))
#             self.updd.append(upblock(i,upsds[idx],upsds[idx+1]))
#
#         self.mres1=ResnetBlock(latentdim,latentdim,time_emb_dim=512,groups=32)
#         # self.matt = PreNorm(latentdim,LinearAttention(latentdim,heads=32))
#         self.matt = attLIN(latentdim,heads=32,cdim=cdim)
#         self.mres2 = ResnetBlock(latentdim, latentdim, time_emb_dim=512,groups=32)
#
#         self.outres=nn.ModuleList([ResnetBlock(dim[0]*2,dim[0],time_emb_dim=512,groups=32),])
#         self.outatt=attLIN(dim[0],heads=dim[0]//32,cdim=cdim)
#
#     def forward(self, x,time_emb,cs):
#         fff=[]
#
#         x=self.incon(x)
#         res11=x.clone()
#
#
#         for res ,att,don in zip(self.dlres,self.dlatt,self.dldd  ):
#             cccc=[]
#             for idx,ii in enumerate(res):
#                 x=ii(x,time_emb)
#                 x = att[idx](x, cs)
#                 cccc.append(x)
#             # x=att[1](x,cs)
#             # cccc[1]=x
#             fff.append(cccc)
#             x=don(x)
#         x=self.mres2(self.matt (self.mres1(x,time_emb), cs), time_emb)
#         fff.reverse()
#
#
#         for res ,att,up ,fautres in zip(self.upres ,self.upatt,self.updd,fff ):
#             x=up(x)
#             # x=fautres+x
#
#             for idx,ii in enumerate(res):
#                 x = torch.cat([x, fautres[idx]], dim=1)
#                 x=ii(x,time_emb)
#                 x = att[idx](x, cs)
#
#         x = torch.cat([x, res11], dim=1)
#         for ii in self.outres:
#             x = ii(x, time_emb)
#         x=self.outatt(x, cs)
#         return self.outcon(x)





class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb





class Unet(nn.Module):
    def __init__(self, in_dims, n_feats, *, n_layers=20, n_chans=256, n_dilates=4):
        super().__init__()
        self.in_dims = in_dims
        self.n_feats = n_feats
        self.unet=unetb()

        self.diffusion_embedding = SinusoidalPosEmb(n_chans)
        self.mlp = nn.Sequential(
            nn.Linear(n_chans, n_chans * 4),
            nn.Mish(),
            nn.Linear(n_chans * 4, n_chans)
        )



    def forward(self, spec, diffusion_step, cond):
        """
        :param spec: [B, F, M, T]
        :param diffusion_step: [B, 1]
        :param cond: [B, H, T]
        :return:
        """
        if self.n_feats == 1:
            x = spec.squeeze(1)  # [B, M, T]
        else:
            x = spec.flatten(start_dim=1, end_dim=2)  # [B, F x M, T]
        # x = self.input_projection(x)  # [B, C, T]


        diffusion_step = self.diffusion_embedding(diffusion_step)
        diffusion_step = self.mlp(diffusion_step)
        b1,c1,t1=x.shape
        pad=t1%8
        if pad!=0:
            pad=8-pad
        x=F.pad(x, (0,pad), "constant", 0)

        x=self.unet(x, diffusion_step, cond)
        if pad!=0:
            x=x[:,:,:-pad]

        # for layer in self.residual_layers:
        #     x, skip_connection = layer(x, cond, diffusion_step)


                                                    # [B, M, T]
        if self.n_feats == 1:
            x = x[:, None, :, :]
        else:
            # This is the temporary solution since PyTorch 1.13
            # does not support exporting aten::unflatten to ONNX
            # x = x.unflatten(dim=1, sizes=(self.n_feats, self.in_dims))
            x = x.reshape(-1, self.n_feats, self.in_dims, x.shape[2])
        return x
