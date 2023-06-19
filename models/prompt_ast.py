import torch 
import torch.nn as nn
from transformers import Autoprocessor, ASTModel
import math
from operator import mul
from functools import reduce
from torch.nn import Dropout


class PromptAST(nn.Module):

    def __init__(self, prompt_config):

        # PROMPT CONFIG
        self.prompt_config = prompt_config

        # LOADING PRETRAINED MODEL
        base_model = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        model_config = base_model.config

        # Getting Patch Embedder and Transformer Encder From Pretrained Model
        self.embeddings = base_model._modules['embeddings']
        self.encoder = base_model._modules['encoder']

        # Prompt Config
        self.num_tokens = self.prompt_config.NUM_TOKENS
        self.prompt_dropout = Dropout(self.prompt_config.DROPOUT)

        # Prompt PROJECT (No projection for now)
        prompt_dim = model_config.hidden_size
        self.prompt_proj = nn.Identity()

        # INITIATE PROMPT
        if self.prompt_config.INITIATION == "random":

            val =  math.sqrt(6. / float(3 * reduce(mul, self.model_config.patch_size, 1) + prompt_dim)) # patch_size taken from AST CONFIG
            self.prompt_embeddings = nn.Parameter(torch.zeros(
                1, self.num_tokens, prompt_dim
            ))

            # xavier_uniform initialization
            nn.init(self.prompt_embeddings.data, -val, +val)
            if self.prompt_config.DEEP:
                raise NotImplementedError("DEEP Prompting not implemented for now.")
        else:
            raise ValueError("Other initiation scheme is not supported")
    
    def incorporate_rpompt(self, x):
        # combine prompt embeddings with image-patch embeddings
        B = x.shape[0]
        # after CLS token, all before image patches
        x = torch.cat((
            x[:, :1, :],
            self.prompt_dropout(self.prompt_proj(self.prompt_embeddings).expand(B, -1, -1)),
            x[:, 1:, :]
        ), dim=1)

        return x
    
    def forward(self, x):
        embedding_output = self.incorporate_prompt(x)

        if self.prompt_config.DEEP:
            raise NotImplementedError("DEEP Prompting not implemented for now.")
        else:
            encoded, attn_weights = self.encoder(embedding_output)

        return encoded, attn_weights