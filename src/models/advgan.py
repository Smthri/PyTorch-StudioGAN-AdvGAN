import torch
from torch import nn

import utils.ops as ops
import utils.misc as misc

class GenEncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, subsample):
        super(GenEncoderBlock, self).__init__()
        self.in_ch = in_channels
        self.out_ch = out_channels
        self.subsample = subsample
        if subsample == True:
            self.block = nn.Sequential(
                nn.Conv2d(self.in_ch, self.out_ch, 2, 2),
                nn.BatchNorm2d(self.out_ch),
                nn.LeakyReLU(inplace=True)
            )
        else:
            self.block = nn.Sequential(
                nn.Conv2d(self.in_ch, self.out_ch, 3, 1, 1),
                nn.BatchNorm2d(self.out_ch),
                nn.LeakyReLU(inplace=True)
            )

    def forward(self, x):
        x = self.block(x)
        return x


class GenDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, upsample):
        super(GenDecoderBlock, self).__init__()
        self.in_ch = in_channels
        self.out_ch = out_channels
        self.upsample = upsample
        if upsample == True:
            self.block = nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_channels, out_channels, 3, 1, 1),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(inplace=True)
            )
        else:
            self.block = nn.Sequential(
                nn.Conv2d(2 * self.in_ch, self.out_ch, 3, 1, 1),
                nn.BatchNorm2d(self.out_ch),
                nn.LeakyReLU(inplace=True)
            )

    def forward(self, x):
        x = self.block(x)
        return x


class Generator(nn.Module):
    def __init__(self, img_size, **kwargs):
        super(Generator, self).__init__()
        self.img_size = img_size
        #assert (img_size == 256), 'Only 256 size is supported at the moment'
        channels = [3, 32, 32, 128, 128, 256, 256, 512, 512]
        self.in_dims = channels
        self.out_dims = channels[::-1]
        self.mixed_precision = kwargs['mixed_precision']
        
        self.subsample = [False, True, False, True, False, True, False, True]
        self.enc_blocks = []
        for i in range(len(channels) - 1):
            self.enc_blocks.append(GenEncoderBlock(channels[i], channels[i + 1], self.subsample[i]))
        self.encoder = nn.Sequential(*self.enc_blocks)
        dec_channels = channels[::-1]
        self.upsample = self.subsample[::-1]
        self.dec_blocks = []
        for i in range(len(dec_channels) - 1):
            self.dec_blocks.append(GenDecoderBlock(dec_channels[i], dec_channels[i+1], self.upsample[i]))
        self.decoder = nn.Sequential(*self.dec_blocks)

    def forward(self, x, label, eval=False):
        with torch.cuda.amp.autocast() if self.mixed_precision and not eval else misc.dummy_context_mgr() as mp:
            cat_list = []
            for i, block in enumerate(self.encoder):
                x = block(x)
                if not self.subsample[i]:
                    cat_list.append(x)
            for i, block in enumerate(self.decoder):
                x = block(x)
                if self.upsample[i]:
                    x = torch.cat((x, cat_list.pop()), dim=1)
            #x = self.encoder(x)
            #x = self.decoder(x)
        return x


class DiscBlock(nn.Module):
    def __init__(self, in_channels, out_channels, apply_d_sn, MODULES):
        super(DiscBlock, self).__init__()
        self.apply_d_sn = apply_d_sn

        self.conv0 = MODULES.d_conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.conv1 = MODULES.d_conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1)

        if not apply_d_sn:
            self.bn0 = MODULES.d_bn(in_features=out_channels)
            self.bn1 = MODULES.d_bn(in_features=out_channels)

        self.activation = MODULES.d_act_fn

    def forward(self, x):
        x = self.conv0(x)
        if not self.apply_d_sn:
            x = self.bn0(x)
        x = self.activation(x)

        x = self.conv1(x)
        if not self.apply_d_sn:
            x = self.bn1(x)
        out = self.activation(x)
        return out


class Discriminator(nn.Module):
    def __init__(self, img_size, d_conv_dim, apply_d_sn, apply_attn, attn_d_loc, d_cond_mtd, aux_cls_type, d_embed_dim, normalize_d_embed,
                 num_classes, d_init, d_depth, mixed_precision, MODULES):
        super(Discriminator, self).__init__()
        self.in_dims = [3] + [64, 128]
        self.out_dims = [64, 128, 256]

        self.apply_d_sn = apply_d_sn
        self.d_cond_mtd = d_cond_mtd
        self.aux_cls_type = aux_cls_type
        self.normalize_d_embed = normalize_d_embed
        self.num_classes = num_classes
        self.mixed_precision = mixed_precision

        self.blocks = []
        for index in range(len(self.in_dims)):
            self.blocks += [[
                DiscBlock(in_channels=self.in_dims[index], out_channels=self.out_dims[index], apply_d_sn=self.apply_d_sn, MODULES=MODULES)
            ]]

            if index + 1 in attn_d_loc and apply_attn:
                self.blocks += [[ops.SelfAttention(self.out_dims[index], is_generator=False, MODULES=MODULES)]]

        self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])

        self.activation = MODULES.d_act_fn
        self.conv1 = MODULES.d_conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)

        if not self.apply_d_sn:
            self.bn1 = MODULES.d_bn(in_features=512)

        # linear layer for adversarial training
        if self.d_cond_mtd == "MH":
            self.linear1 = MODULES.d_linear(in_features=512, out_features=1 + num_classes, bias=True)
        elif self.d_cond_mtd == "MD":
            self.linear1 = MODULES.d_linear(in_features=512, out_features=num_classes, bias=True)
        else:
            self.linear1 = MODULES.d_linear(in_features=512, out_features=1, bias=True)

        # double num_classes for Auxiliary Discriminative Classifier
        if self.aux_cls_type == "ADC":
            num_classes = num_classes * 2

        # linear and embedding layers for discriminator conditioning
        if self.d_cond_mtd == "AC":
            self.linear2 = MODULES.d_linear(in_features=self.out_dims[-1], out_features=num_classes, bias=False)
        elif self.d_cond_mtd == "PD":
            self.embedding = MODULES.d_embedding(num_classes, self.out_dims[-1])
        elif self.d_cond_mtd in ["2C", "D2DCE"]:
            self.linear2 = MODULES.d_linear(in_features=self.out_dims[-1], out_features=d_embed_dim, bias=True)
            self.embedding = MODULES.d_embedding(num_classes, d_embed_dim)
        else:
            pass

        # linear and embedding layers for evolved classifier-based GAN
        if self.aux_cls_type == "TAC":
            if self.d_cond_mtd == "AC":
                self.linear_mi = MODULES.d_linear(in_features=self.out_dims[-1], out_features=num_classes, bias=False)
            elif self.d_cond_mtd in ["2C", "D2DCE"]:
                self.linear_mi = MODULES.d_linear(in_features=self.out_dims[-1], out_features=d_embed_dim, bias=True)
                self.embedding_mi = MODULES.d_embedding(num_classes, d_embed_dim)
            else:
                raise NotImplementedError

        if d_init:
            ops.init_weights(self.modules, d_init)

    def forward(self, x, label, eval=False, adc_fake=False):
        with torch.cuda.amp.autocast() if self.mixed_precision and not eval else misc.dummy_context_mgr() as mp:
            embed, proxy, cls_output = None, None, None
            mi_embed, mi_proxy, mi_cls_output = None, None, None
            h = x
            for index, blocklist in enumerate(self.blocks):
                for block in blocklist:
                    h = block(h)
            h = self.conv1(h)
            if not self.apply_d_sn:
                h = self.bn1(h)
            h = self.activation(h)
            h = torch.sum(h, dim=[2, 3])

            # adversarial training
            adv_output = torch.squeeze(self.linear1(h))

            # make class labels odd (for fake) or even (for real) for ADC
            if self.aux_cls_type == "ADC":
                if adc_fake:
                    label = label*2 + 1
                else:
                    label = label*2

            # class conditioning
            if self.d_cond_mtd == "AC":
                if self.normalize_d_embed:
                    for W in self.linear2.parameters():
                        W = F.normalize(W, dim=1)
                    h = F.normalize(h, dim=1)
                cls_output = self.linear2(h)
            elif self.d_cond_mtd == "PD":
                adv_output = adv_output + torch.sum(torch.mul(self.embedding(label), h), 1)
            elif self.d_cond_mtd in ["2C", "D2DCE"]:
                embed = self.linear2(h)
                proxy = self.embedding(label)
                if self.normalize_d_embed:
                    embed = F.normalize(embed, dim=1)
                    proxy = F.normalize(proxy, dim=1)
            elif self.d_cond_mtd == "MD":
                idx = torch.LongTensor(range(label.size(0))).to(label.device)
                adv_output = adv_output[idx, label]
            elif self.d_cond_mtd in ["W/O", "MH"]:
                pass
            else:
                raise NotImplementedError

            # extra conditioning for TACGAN and ADCGAN
            if self.aux_cls_type == "TAC":
                if self.d_cond_mtd == "AC":
                    if self.normalize_d_embed:
                        for W in self.linear_mi.parameters():
                            W = F.normalize(W, dim=1)
                    mi_cls_output = self.linear_mi(h)
                elif self.d_cond_mtd in ["2C", "D2DCE"]:
                    mi_embed = self.linear_mi(h)
                    mi_proxy = self.embedding_mi(label)
                    if self.normalize_d_embed:
                        mi_embed = F.normalize(mi_embed, dim=1)
                        mi_proxy = F.normalize(mi_proxy, dim=1)
        return {
            "h": h,
            "adv_output": adv_output,
            "embed": embed,
            "proxy": proxy,
            "cls_output": cls_output,
            "label": label,
            "mi_embed": mi_embed,
            "mi_proxy": mi_proxy,
            "mi_cls_output": mi_cls_output
        }
