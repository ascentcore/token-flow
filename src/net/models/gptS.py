import math
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F

from .utils import CfgNode as CN


class StimulusSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, stimulus):
        x_lst = x.tolist()
        stimulus_lst = stimulus.tolist()

        b_s = len(x)
        products = []
        for i in range(0, b_s):
            products.append([np.dot(word_embed, s).tolist() for (word_embed, s) in zip(x_lst[i], stimulus_lst[i])])

        res = torch.tensor(products)
        # res = []        
        # for p in products:
        #     res.append(np.array(p).sum(axis=0).tolist())
        # res = torch.tensor(res)

        return res


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size

        pretrained_wte = nn.Embedding(config.vocab_size, config.n_embd)
        pretrained_wte.load_state_dict({'weight': config.pretrained_embeddings})
        pretrained_wte.weight.requires_grad = False

        self.layers = nn.ModuleDict(dict(
            wte=pretrained_wte,
            attn_stimulus=StimulusSelfAttention(),
            dropout=nn.Dropout(config.embd_pdrop),
            ln=nn.LayerNorm(config.n_embd),
            fc=nn.Linear(config.n_embd, 4 * config.n_embd),
            fc2=nn.Linear(4 * config.n_embd, 4 * config.n_embd),
            proj=nn.Linear(4 * config.n_embd, config.n_embd),
            act=nn.ReLU(),
            lm_head = nn.Linear(config.block_size * config.n_embd, config.vocab_size, bias=False)
        ))

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        n_params = sum(p.numel() for p in self.layers.parameters())
        print('n_params', n_params)
        print("number of parameters: %.2fM" % (n_params/1e6,))


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)


    def configure_optimizers(self, train_config):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name
                if pn.endswith('bias'):
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(
            inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
            % (str(param_dict.keys() - union_params), )

        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(
                list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn]
                        for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(
            optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer


    def forward(self, idx, stimulus, targets=None):
        x = self.layers.wte(idx)
        x = self.layers.attn_stimulus(x, stimulus)
        x = self.layers.ln(x)
        x = self.layers.fc(x)
        x = self.layers.dropout(x)
        x = self.layers.fc2(x)
        x = self.layers.dropout(x)
        x = self.layers.fc2(x)
        x = self.layers.dropout(x)
        x = self.layers.proj(x)
        x = self.layers.ln(x)
        x = x.reshape([x.size(dim=0), x.size(dim=1) * x.size(dim=2)])

        logits = self.layers.lm_head(x)

        loss = None
        final_train_acc = None
        if targets is not None:
            loss = F.cross_entropy(logits, targets)
        
            logits_array = logits.detach().numpy()
            targets_array = targets.numpy()
            logits_argmax = np.argmax(logits_array, axis=1)
            targets_argmax = np.argmax(targets_array, axis=1)

            # print('')
            # print('logits_argmax', logits_argmax)
            # print('targets_argmax', targets_argmax)

            train_acc = torch.sum(torch.tensor(logits_argmax) == torch.tensor(targets_argmax))
            final_train_acc = train_acc / x.size(dim=0)

        return logits, loss, final_train_acc

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(
                1) <= self.block_size else idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                _, idx_next = torch.topk(probs, k=1, dim=-1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
