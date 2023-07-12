import torch 
import torch.nn as nn
from transformers import ASTModel
import math
from operator import mul
from functools import reduce
from torch.nn import Dropout
from torch.nn.modules.utils import _pair


class PromptAST(nn.Module):

    def __init__(self, prompt_config, max_length, num_classes, model_ckpt="MIT/ast-finetuned-audioset-10-10-0.4593"):

        super().__init__()
        # PROMPT CONFIG
        self.prompt_config = prompt_config

        # LOADING PRETRAINED MODEL
        base_model = ASTModel.from_pretrained(model_ckpt, max_length=max_length, ignore_mismatched_sizes=True)
        self.model_config = base_model.config
        print("BASE_MODEL:\n", base_model)
        # Getting Patch Embedder and Transformer Encder From Pretrained Model
        self.embeddings = base_model._modules['embeddings']
        self.encoder = base_model._modules['encoder']

        self.patch_size = _pair(self.model_config.patch_size)
        # Prompt Config
        self.num_tokens = self.prompt_config.NUM_TOKENS
        self.prompt_dropout = Dropout(self.prompt_config.DROPOUT)

        # Prompt PROJECT (No projection for now)
        prompt_dim = self.model_config.hidden_size
        self.prompt_proj = nn.Identity()

        self.classification_head = nn.Linear(self.model_config.hidden_size, num_classes)

        # INITIATE PROMPT
        if self.prompt_config.INITIATION == "random":

            val =  math.sqrt(6. / float(3 * reduce(mul, self.patch_size, 1) + prompt_dim)) # patch_size taken from AST CONFIG
            self.prompt_embeddings = nn.Parameter(torch.zeros(
                1, self.num_tokens, prompt_dim
            ))

            # xavier_uniform initialization
            nn.init.uniform_(self.prompt_embeddings.data, -val, +val)

            # DEEP PROMPT INITIALIZATION
            if self.prompt_config.DEEP:
                total_d_layer = self.model_config.num_hidden_layers-1
                self.deep_prompt_embeddings = nn.Parameter(torch.zeros(
                    total_d_layer, self.num_tokens, prompt_dim))
                # xavier_uniform initialization
                nn.init.uniform_(self.deep_prompt_embeddings.data, -val, val)
        else:
            raise ValueError("Other initiation scheme is not supported")
    
    def train(self, mode=True):
        # set train status for this class: disable all but the prompt-related modules
        if mode:
            # training:
            self.encoder.eval()
            self.embeddings.eval()
            self.prompt_proj.train()
            self.prompt_dropout.train()
            self.classification_head.train()
        else:
            # eval:
            for module in self.children():
                module.train(mode)
    
    def incorporate_prompt(self, x):
        # combine prompt embeddings with image-patch embeddings
        B = x.shape[0]
        # after CLS token, all before image patches
        #print(self.prompt_dropout(self.prompt_proj(self.prompt_embeddings).expand(B, -1, -1)).device)
        # print(x.shape)
        # print(self.prompt_dropout(self.prompt_proj(self.prompt_embeddings).expand(B, -1, -1)).shape)
        x = self.embeddings(x)  # (batch_size, 1 + n_patches, hidden_dim)
        # print(self.prompt_embeddings)
        x = torch.cat((
            x[:, :1, :],
            self.prompt_dropout(self.prompt_proj(self.prompt_embeddings).expand(B, -1, -1)),
            x[:, 1:, :]
        ), dim=1)

        return x
    
    # DEEP PROMPT FORWARDING
    def forward_deep_prompt(self, embedding_output):
        attn_weights = []
        hidden_states = None
        weights = None
        B = embedding_output.shape[0]
        num_layers = self.model_config.num_hidden_layers

        for i in range(num_layers):
            # print(i)
            if i == 0:
                
                hidden_states = self.encoder.layer[i](embedding_output)[0]
                # print(hidden_states[0])
                # print(type(hidden_states[0]))
            else:
                if i <= self.deep_prompt_embeddings.shape[0]:
                    
                    deep_prompt_emb = self.prompt_dropout(self.prompt_proj(
                        self.deep_prompt_embeddings[i-1]).expand(B, -1, -1))
                    
                    hidden_states = torch.cat((
                        hidden_states[:, :1, :],
                        deep_prompt_emb,
                        hidden_states[:, (1+self.num_tokens):, :]
                    ), dim=1)


                hidden_states = self.encoder.layer[i](hidden_states)[0]

            # if self.encoder.vis:
            #     attn_weights.append(weights)

        # encoded = self.encoder[layernorm_after](hidden_states)
        # print(hidden_states.shape)
        return hidden_states
    
    def forward(self, x):
        embedding_output = self.incorporate_prompt(x)

        if self.prompt_config.DEEP:
            x = self.forward_deep_prompt(
                embedding_output)
            out = self.classification_head(torch.mean(x, 1))
        else:
            x = self.encoder(embedding_output)
            out = self.classification_head(torch.mean(x.last_hidden_state, 1))
        
        
        return out