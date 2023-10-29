import torch
import torch.nn as nn
import torchvision.models as models

from .blocks import get_sinusoid_encoding, TransformerEncoderBlock, LayerNorm
from .resnet import resnet8_cifar, resnet10_imagenet


class BranchEncoder(nn.Module):
    """
    A backbone with a stack of transformer encoder layers.
    
    learned knob embeddings
    -> [embedding projection] 
    -> [self-attn transformer x L]
    -> branch embeddings
    """
    def __init__(
        self, 
        embd_dim,           # embedding dimension
        out_dim,            # output dimension
        seq_len,            # input sequence length
        n_heads,            # number of attention heads
        n_layers=5,         # number of transformer encoder layers
        attn_pdrop=0.1,     # dropout rate for attention map
        proj_pdrop=0.1,     # dropout rate for projection
        path_pdrop=0.1,     # dropout rate for residual paths
        eos=True,           # if True, add end-of-sequence token.
        embd_type=0,        # knob embedding type (0, 1)
        pe_type=0,          # position encoding type (-1, 0, 1)
                                # -1: none
                                #  0: transformer-style
                                #  1: perceiver-style
    ):
        super(BranchEncoder, self).__init__()

        self.seq_len = seq_len

        # learned knob encodings
        ## NOTE: include a [EOS] token to encode hidden state
        assert embd_type in (0, 1)
        self.embd_type = embd_type
        self.eos = eos
        if embd_type == 0:
            self.ke = nn.Embedding(seq_len + eos, embd_dim)
            self.be = nn.Embedding(2, embd_dim)
        else:
            self.ke = nn.Embedding(seq_len * 2 + eos, embd_dim)

        # position encodings (t+1, c)
        assert pe_type in (-1, 0, 1)
        self.pe_type = pe_type
        if pe_type in (0, 1):
            pe = get_sinusoid_encoding(seq_len + eos, embd_dim // 2, pe_type)
            pe /= embd_dim ** 0.5
            self.register_buffer('pe', pe, persistent=False)

        # self-attention transformers
        self.transformer = nn.ModuleList()
        for _ in range(n_layers):
            self.transformer.append(
                TransformerEncoderBlock(
                    embd_dim, 
                    n_heads=n_heads, 
                    attn_pdrop=attn_pdrop, 
                    proj_pdrop=proj_pdrop, 
                    path_pdrop=path_pdrop,
                )
            )

        # hidden state projection
        self.fc = nn.Linear(embd_dim, out_dim)
            
        self.apply(self.__init_weights__)

    def __init_weights__(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.01)

    def forward(self, masks):
        """
        Args:
            masks (bool tensor, (n, kb)): branch masks.

        Returns:
            x (float tensor, (n, c)): branch embeddings.
        """
        bs, t = masks.size()
        assert t == self.seq_len

        # add knob, binary and position encodings
        if self.embd_type == 0:
            x = self.ke.weight.transpose(0, 1)              # (c, t+1)
            if self.pe_type in (0, 1):
                x = x + self.pe                             # (c, t+1)
            be = self.be(masks.int()).transpose(1, 2)       # (n, c, t)
            x = x.repeat(bs, 1, 1)                          # (n, c, t+1)
            if self.eos:
                x[..., :-1] = x[..., :-1] + be
            else:
                x = x + be
        else:
            ke_idx = torch.arange(
                self.seq_len, device=masks.device
            ) * 2 + masks                                   # (n, kb)
            if self.eos:
                ke_idx = torch.cat(
                    [ke_idx, self.ke.num_embeddings * ke_idx.new_ones(bs, 1)], 
                    dim=1,
                )
            x = self.ke(ke_idx).transpose(1, 2)             # (n, c, t)
            if self.pe_type in (0, 1):
                x = x + self.pe

        # self-attention transformers
        for idx in range(len(self.transformer)):
            x = self.transformer[idx](x)

        # hidden state projection
        if self.eos:
            x = x[..., -1]
        else:
            x = x.mean(dim=-1)
        x = self.fc(x)                                  # (n, c)
        return x


class ContentEncoder(nn.Module):

    def __init__(self, out_dim, arch='mobilenet_v2', pretrained=False):
        super(ContentEncoder, self).__init__()

        ## NOTE: all GFLOPs assume 224x224 input
        if arch == 'mobilenet_v2':          # 0.3 GFLOPs
            model = models.mobilenet_v2(pretrained).features
            embd_dim = 1280
        elif arch == 'mobilenet_v3_large':  # 0.219 GFLOPs
            model = models.mobilenet_v3_large(pretrained).features
            embd_dim = 960
        elif arch == 'mobilenet_v3_small':  # 0.056 GFLOPs
            model = models.mobilenet_v3_small(pretrained).features
            embd_dim = 576
        elif arch == 'efficientnet_b0':     # 0.39 GFLOPs
            model = models.efficientnet_b0(pretrained).features
            embd_dim = 1280
        elif arch == 'efficientnet_b1':     # 0.70 GFLOPs
            model = models.efficientnet_b1(pretrained).features
            embd_dim = 1280
        elif arch == 'efficientnet_b2':     # 1.0 GFLOPs
            model = models.efficientnet_b2(pretrained).features
            embd_dim = 1408
        elif arch == 'efficientnet_b3':     # 1.8 GFLOPs
            model = models.efficientnet_b3(pretrained).features
            embd_dim = 1536
        elif arch == 'resnet8_cifar':
            model = resnet8_cifar()
            embd_dim = 64
        elif arch == 'resnet10_imagenet':
            model = resnet10_imagenet()
            embd_dim = 512
        else:
            raise NotImplementedError(
                'invalid content encoder architecture: {:s}'.format(arch)
            )

        self.model = model
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(embd_dim, out_dim)

    def forward(self, x):
        x = self.model(x)
        x = self.avgpool(x).flatten(1)
        x = self.fc(x)
        return x


class BranchVAE(nn.Module):
    """
    A variational autoencoder for branch sampling at inference time.
    """
    def __init__(self, in_dim, hid_dim, n_layers, latent_dim):
        super(BranchVAE, self).__init__()

        self.in_dim = in_dim
        self.latent_dim = latent_dim

        # encoder
        ## NOTE: output is Gaussian mean and log variance
        enc = []
        for _ in range(n_layers - 1):
            enc.append(
                nn.Sequential(
                    nn.Linear(in_dim, hid_dim), 
                    nn.ReLU(inplace=True),
                )
            )
            in_dim = hid_dim
        enc.append(nn.Linear(in_dim, latent_dim * 2))
        self.enc = nn.Sequential(*enc)

        # decoder
        ## NOTE: output is Bernoulli mean
        dec = []
        for _ in range(n_layers - 1):
            dec.append(
                nn.Sequential(
                    nn.Linear(latent_dim, hid_dim), 
                    nn.ReLU(inplace=True),
                )
            )
            latent_dim = hid_dim
        dec.append(nn.Linear(latent_dim, self.in_dim))
        self.dec = nn.Sequential(*dec)

    def _encode(self, x):
        x = self.enc(x)
        mu, log_var = x.split(self.latent_dim, dim=1)
        return mu, log_var

    def _sample(self, mu, log_var):
        sigma = torch.exp(0.5 * log_var)
        z = torch.randn_like(sigma) * sigma + mu
        return z

    def _decode(self, z):
        x = self.dec(z)
        return x

    def forward(self, x=None, z=None):
        # train
        if x is not None and z is None:
            mu, log_var = self._encode(x.float())
            z = self._sample(mu, log_var)
            logits = self._decode(z)
            return logits, mu, log_var
        # test
        elif x is None and z is not None:
            p = torch.sigmoid(self._decode(z))
            return p
        else:
            raise ValueError('one of x and z must be given')