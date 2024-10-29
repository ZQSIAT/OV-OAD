import os.path as osp
from collections import OrderedDict
import math
import numpy as np
import torch
import torch.nn as nn
from itertools import repeat
import collections.abc
from ..clip import clip
# from ..clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

from .oadtr import OadTransformer
import diffdist.functional as diff_dist
import torch.distributed as dist
import torch.nn.functional as F
from einops import rearrange, repeat

from IPython import embed


def dist_collect(x):
    """ collect all tensor from all GPUs
    args:
        x: shape (mini_batch, ...)
    returns:
        shape (mini_batch * num_gpu, ...)
    """
    x = x.contiguous()
    out_list = [torch.zeros_like(x, device=x.device, dtype=x.dtype).contiguous() for _ in range(dist.get_world_size())]
    out_list = diff_dist.all_gather(out_list, x)
    return torch.cat(out_list, dim=0).contiguous()


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.token_embedding = clip_model.token_embedding
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND

        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)   # self.ln_final(x).type(self.dtype) todo dtype

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection  # [20, 512] @ [512, 512]
        return x


class ZsOadCLIP(nn.Module):
    def __init__(self, 
                 cfg, 
                 classnames, 
                 clip_model,  ## pretrained model include img enc and txt enc.
                 logger,
                 ):
        super().__init__()

        ## text enc
        self.zero_shot = cfg.zero_shot
        self.read_from = cfg.read_from
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.loss_weight = cfg.criterion.loss_weight

        if self.read_from == "png":
            ## image enc
            self.img_encoder = clip_model.visual
        else:
            self.img_encoder = nn.Identity()

        # online action detection transformers
        self.oad_encoder_decoder = OadTransformer(
                                                cfg=cfg,
                                                num_class=len(classnames),
                                                num_tokens=cfg.enc_layers,
                                                encoder_layers=cfg.encoder_layers,  # 3
                                                decoder_layers=cfg.decoder_layers,  # 5
                                                decoder_query_frames=cfg.dec_query)
        self.cross_entropy = nn.CrossEntropyLoss()

        # embed()
        # exit()

    def encode_image(self, image):
        if self.read_from == "png":
            # Lets encode the video into required format
            b, t, c, h, w = image.size()  # (B, T, C, H, W)
            # Remove the batch dimensions
            image = image.reshape(-1, c, h, w)  # [B*T, C, 224, 224]
            # with torch.autocast("cuda"):
            image_features = self.img_encoder(image)
            image_features = image_features.view(b, t, -1)  # [B, T, 512] = [1, 32, 512]
            # Then, start of online motion detection
            inputs = (image_features, None)
        else:
            inputs = (image, None)
        image_features = self.oad_encoder_decoder(inputs)   # [BS, cur_1+query_frames, dim] [1, 9, 512]
        return image_features
    
    def encode_text(self, text):
        # ## tokenized text [B, 9, 77]
        # text = rearrange(text, 'b l c -> (b l) c')
        # [B, Dim]
        # with torch.autocast("cuda"):
        text_x = self.text_encoder(text)
        return text_x
    
    ## take from groupvit
    def loss(self, image_x, text_x):

        batch_size = image_x.shape[0]
        # get label globally
        labels = torch.arange(batch_size, dtype=torch.long, device=image_x.device) + batch_size * dist.get_rank()

        # [B, C]
        image_x = F.normalize(image_x, dim=-1)
        text_x = F.normalize(text_x, dim=-1)

        logits_per_img = image_x @ dist_collect(text_x).t()
        logits_per_text = text_x @ dist_collect(image_x).t()

        logit_scale = torch.clamp(self.logit_scale.exp(), max=100)
        loss_img = self.cross_entropy(logits_per_img * logit_scale, labels)
        loss_text = self.cross_entropy(logits_per_text * logit_scale, labels)

        loss = 0.5 * (loss_img + loss_text)

        return loss
    
    def forward_train(self, image, text):

        image_outs = self.encode_image(image)

        ## encoder image feature [B, C]
        image_enc = image_outs[:, 0, :]

        ## decoder image feature [Bx8, C]
        image_dec = rearrange(image_outs[:, 1:, :], 'b l c -> (b l) c') 

        ## encoder text [B, 77]
        text_enc = text[:, 0, :]
        text_enc = self.encode_text(text_enc)

        ## encoder text [Bx8, 77]     
        text_dec = rearrange(text[:, 1:, :], 'b l c -> (b l) c')   
        text_dec = self.encode_text(text_dec)

        # embed()
        # exit()

        losses_enc = self.loss(image_enc, text_enc) * self.loss_weight.enc_vtc
        losses_dec = self.loss(image_dec, text_dec) * self.loss_weight.dec_vtc
        # print("#140, loss: ", losses)

        losses_dict = dict(loss_enc_vtc=losses_enc,
                           loss_dec_vtc=losses_dec,)
        return losses_dict

    def forward_test(self, image, text):
        return self.zero_shot_pred(image, text)
    
    def forward(self, image, text):
        if self.training:
            return self.forward_train(image, text)
        else:
            return self.forward_test(image, text)
        
    @torch.no_grad()
    def zero_shot_pred(self, image, text_weights):  # todo take from groupvit, have bugs
        # [B, Dim]
        image_features = self.encode_image(image)
        image_features = F.normalize(image_features, dim=-1)
        # text_weights: (Dim, num_cls) = [512, 11]
        # cosine similarity as logits
        logits_per_image = self.logit_scale.exp() * image_features @ text_weights

        return logits_per_image
    

def returnCLIP(config, logger=None, class_names=None):
    logger.info(f"Loading CLIP (backbone: {config.models_name})")
    clip_model, preprocess = clip.load(config.models_name)

    logger.info("Building ZsOad-CLIP Models")
    model = ZsOadCLIP(config, class_names, clip_model, logger)

    # Now need to control freezing of CLIP for fine-tuning
    train_complete_clip = config.USE

    if train_complete_clip == "none":
        logger.info("Turning off gradients in both the image and the text encoder, Only tuning the oad_encoder_decoder")
        for name, param in model.named_parameters():
            if "oad_encoder_decoder" in name:
                 param.requires_grad_(True)
                 logger.info(f"requires_grad: {name}")
            else:
                param.requires_grad_(False)
    elif train_complete_clip == "both":
        logger.info("Turning on gradients for COMPLETE ViFi-CLIP model")
        for name, param in model.named_parameters():
            param.requires_grad_(True)
            logger.info(f"requires_grad: {name}")
    elif train_complete_clip == "image":
        logger.info("Turning on gradients for image side the ViFi-CLIP model")
        for name, param in model.named_parameters():
            if "img_encoder" in name:  # replace by "text_encoder" incase you want to freeze text
                param.requires_grad_(True)
                logger.info(f"requires_grad: {name}")
            else:
                param.requires_grad_(False)
    elif train_complete_clip == "text":
        logger.info("Turning on gradients for TEXT side the ViFi-CLIP model")
        for name, param in model.named_parameters():
            if "text_encoder" in name:  # replace by "text_encoder" incase you want to freeze text
                param.requires_grad_(True)
                logger.info(f"requires_grad: {name}")
            else:
                param.requires_grad_(False)
    else:
        raise NotImplementedError
    
    # Double check
    enabled = set()
    for name, param in model.named_parameters():
        if param.requires_grad:
            enabled.add(name)
    logger.info(f"Parameters to be updated: {enabled}")
    logger.info(f"Total learnable items: {len(enabled)}")
    model.float()
    return model, preprocess


if __name__ == "__main__":
    model = ZsOadCLIP()
    x_train = torch.randn(1, 32, 3, 224, 224).cuda()  # BxTxCxHxW
    out = model(x_train)
    print(out.shape)
    pass



