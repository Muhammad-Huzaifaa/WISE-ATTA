import os
import torch
import torch.nn as nn
import torch.nn.functional as F
# import logging
import math
import copy
import random
from methods.base import TTAMethod
from utils.registry import ADAPTATION_REGISTRY
from utils.losses import Entropy
from utils.misc import set_deterministic
from models.model import split_up_model
from torch.utils.data import Dataset, DataLoader


@ADAPTATION_REGISTRY.register()
class EATTA(TTAMethod):
    """EATTA with optional 50% label usage:
       odd batches: entropy + CE (full EATTA),
       even batches: entropy only (no supervised loss).
    """
    def __init__(self, cfg, model, num_classes):
        super().__init__(cfg, model, num_classes)
        # hyper-parameters (the same with EATA and SAR)
        set_deterministic(0)
        self.e_margin = cfg.EATA.MARGIN_E0 * math.log(num_classes)   
        # hyper-parameters for effortless active labeling
        self.cls_num_count = [0 for _ in range(num_classes)]
        self.cls_diff = [0 for _ in range(num_classes)]
        self.oracle_num = cfg.MODEL.ORACLE_NUM
        self.select_ = select_sample(self.oracle_num, self.device)
        self.noise_std = 0.01
        # hyper-parameters for gradient norm-based denoising
        self.w1_ema = 0
        self.w2_ema = 0
        self.mo = 0.8
        # loss function for unannotated samples
        self.softmax_entropy = Entropy()
        # use a buffer
        self.use_buffer = cfg.MODEL.BUFFER
        self.samplebuffer = SampleBuffer()
        self.buffer_bs = 32
        # set annotator ('human' or 'large_model')
        self.annotator = cfg.MODEL.HUMAN_OR_LARGE_MODEL
        # split model into feature extractor and classifier
        arch_name = cfg.MODEL.EDGE_ARCH
        self.featurizer, self.classifier = split_up_model(self.model[1], arch_name, self.dataset_name) 
        for param in self.model[0].parameters():
            param.detach_()
        #
        self.cfg = cfg

        # --------- NEW: simple 50% label schedule ----------
        # Always use entropy; use CE only on some batches.
        # Default: CE every other batch (alternate batches).
        self.ce_period = getattr(cfg.MODEL, "CE_PERIOD", 2)         # unused in the simple 1-on / 1-off schedule
        self.ce_use_prob = getattr(cfg.MODEL, "CE_USE_PROB", None)  # optional stochastic mode
        self.global_step = 0
        self.sorted_idx = None
        # ------------------------------------------------------
        # ---------------- anchor / teacher model (EMA of student) ----------------
        # anchor is a slow-moving copy of the edge model (student)
        self.anchor_featurizer = copy.deepcopy(self.featurizer).to(self.device)
        self.anchor_classifier = copy.deepcopy(self.classifier).to(self.device)
        for p in self.anchor_featurizer.parameters():
            p.requires_grad_(False)
        for p in self.anchor_classifier.parameters():
            p.requires_grad_(False)
        # EMA momentum for teacher update
        self.anchor_momentum = getattr(cfg.MODEL, "ANCHOR_MOMENTUM", 0.9)
        # ------------------------------------------------------------------------
        
    def loss_calculation(self, x, y, apply_ce=True):
        # forward
        imgs_test = x[0]
        features = self.featurizer(imgs_test)
        outputs = self.classifier(features)
        py, y_prime = F.softmax(outputs, dim=-1).max(1)

        entropys = self.softmax_entropy(outputs)
        ids1 = torch.where(entropys < self.e_margin)[0]

        # effortless active labeling (only when CE is on)
        # if apply_ce:
        #     noise = torch.randn(features.size()).to(self.device) * self.noise_std # std=0.01
        #     fea = features.clone().detach() + noise
        #     out = self.classifier(fea)
        #     py2 = F.softmax(out, dim=-1)[:, y_prime]
        #     py2 = torch.diag(py2)
        #     diff = torch.abs(py - py2) # Eq.(3)
        #     sorted_indices = torch.argsort(diff, descending=True)
        #     self.sorted_idx, self.cls_num_count, self.cls_diff =  self.select_(y_prime, sorted_indices, diff, self.cls_num_count, self.cls_diff)

        # ---------------- forgetting-aware active labeling ----------------
        # use prediction drift between student and EMA anchor as sample score
        if apply_ce:
            with torch.no_grad():
                anchor_features = self.anchor_featurizer(imgs_test)
                anchor_outputs = self.anchor_classifier(anchor_features)  # same 200-dim head
                p_anchor = F.softmax(anchor_outputs, dim=-1)

            p_student = F.softmax(outputs, dim=-1)
            # L2 distance in probability space per sample
            diff = torch.norm(p_student - p_anchor, p=2, dim=1)  # [B]

            # sort by drift (largest first)
            sorted_indices = torch.argsort(diff, descending=True)
            self.sorted_idx, self.cls_num_count, self.cls_diff = self.select_(
                y_prime, sorted_indices, diff, self.cls_num_count, self.cls_diff
            )
        # ------------------------------------------------------------------
        
        
        
        # remove labeled indices from entropy set
        for i in self.sorted_idx:
            for j in ids1:
                if i==j:
                    mask = ids1 != i
                    ids1 = ids1[mask]

        # unsupervised entropy loss
        if ids1.numel() > 0:
            loss_ent = entropys[ids1].mean(0)
        else:
            loss_ent = entropys.mean(0)

        # ============================================
        # CASE 1: entropy-only batch (no labels used)
        # ============================================
        if not apply_ce:
            # Pure unsupervised update on this batch
            loss = loss_ent
            return outputs, loss

        # ============================================
        # CASE 2: full EATTA batch (entropy + CE)
        # ============================================
        # supervised CE (human or large model)
        if self.annotator == 'HUMAN':
            loss_ce = F.cross_entropy(outputs[self.sorted_idx], y[self.sorted_idx])
        elif self.annotator == 'LARGE_MODEL':
            with torch.no_grad():
                imgs_cloud = imgs_test[self.sorted_idx]
                cloud_outputs = self.cloud_model(imgs_cloud)
                py_c, y_prime_c = F.softmax(cloud_outputs, dim=-1).max(1)
            loss_ce = F.cross_entropy(outputs[self.sorted_idx], y_prime_c.detach())
        else:
            raise NotImplementedError
        
        # IF BUFFER (optional)
        if self.use_buffer:
            oracle_labels = y_prime_c if self.annotator == 'LARGE_MODEL' else y[self.sorted_idx]
            samples = copy.deepcopy(imgs_test)[self.sorted_idx]
            self.samplebuffer.add(samples, oracle_labels)
            buffer_dataset = Buffer(self.samplebuffer.buffer)
            buffer_loader = DataLoader(buffer_dataset, batch_size=self.buffer_bs, shuffle=True)
            
            for imgs, labels in buffer_loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                features_o = self.featurizer(imgs).view(imgs.size(0), -1)
                outputs_o = self.classifier(features_o)
                loss_buffer = F.cross_entropy(outputs_o, labels)
            loss_ce = loss_ce + loss_buffer
        
        # gradient norm-based debiasing (only when CE is on)
        grad1 = torch.autograd.grad(loss_ent, list(param for param in self.model[1].parameters() if param.requires_grad), retain_graph=True)
        grad1_norm = torch.norm(torch.stack([g.norm() for g in grad1]))
        grad2 = torch.autograd.grad(loss_ce, list(param for param in self.model[1].parameters() if param.requires_grad), retain_graph=True)
        grad2_norm = torch.norm(torch.stack([g.norm() for g in grad2]))
        
        w1 = 2 * grad2_norm / (grad2_norm + grad1_norm)
        w2 = 2 * grad1_norm / (grad2_norm + grad1_norm)
        w1 = w1.detach().item()
        w2 = w2.detach().item()
        
        self.w1_ema, self.w2_ema = update_w1_w2(w1, w2, self.w1_ema, self.w2_ema, self.mo)
        

        # use the loss for the two connsecutive time and then skip the next two
        # if self.global_step % 4 < 2:

        loss = self.w1_ema * loss_ent + self.w2_ema * loss_ce 

        return outputs, loss
        
    @torch.enable_grad()
    def forward_and_adapt(self, x, y):
        """Forward and adapt model on batch of data.
        Alternates between:
          - full EATTA (entropy + CE),
          - entropy-only batches (no labels used).
        """
        # --- Decide if we use CE on this batch ---
        if self.ce_use_prob is not None:
            # stochastic usage if desired
            apply_ce = (random.random() < float(self.ce_use_prob))
        else:
            # simple 50% usage:
            # global_step starts at 0
            # step 0 -> (0+1)%2=1 -> True  (use CE)
            # step 1 -> (1+1)%2=0 -> False (entropy only)
            # step 2 -> True, step 3 -> False, etc.
            if (self.global_step + 1) % 2 == 0:
                apply_ce = False
            else:
                apply_ce = True

        outputs, loss = self.loss_calculation(x, y, apply_ce=apply_ce)
       
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        self.global_step += 1

        # ---------------- EMA update of anchor teacher ----------------
        # if apply_ce:
        with torch.no_grad():
            # featurizer EMA
            for p_anchor, p_student in zip(
                self.anchor_featurizer.parameters(),
                self.featurizer.parameters()
            ):
                p_anchor.data.mul_(self.anchor_momentum).add_(
                    p_student.data * (1.0 - self.anchor_momentum)
                )

            # classifier EMA
            for p_anchor, p_student in zip(
                self.anchor_classifier.parameters(),
                self.classifier.parameters()
            ):
                p_anchor.data.mul_(self.anchor_momentum).add_(
                    p_student.data * (1.0 - self.anchor_momentum)
                )
        # ----------------------------------------------------------------
        
        return outputs

    def collect_params(self):
        """Collect the affine scale + shift parameters from batch norms."""
        params = []
        names = []
        if self.cfg.CORRUPTION.DATASET in [
            'imagenet_c', 'imagenet_r', 'imagenet_a', 'imagenet_k', 'pacs'
        ]:
            for nm, m in self.model[1].named_modules():
                if 'layer4' in nm:
                    continue
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                    for np, p in m.named_parameters():
                        if np in ['weight', 'bias'] and p.requires_grad:  # weight is scale, bias is shift
                            params.append(p)
                            names.append(f"{nm}.{np}")
        return params, names

    def configure_model(self):
        """Configure model for use with EATTA."""
        self.model[1].eval()  # eval mode to avoid stochastic depth in swin. test-time normalization is still applied
        self.model[0].eval()
        # disable grad, to (re-)enable only what EATTA updates
        self.model[1].requires_grad_(False)
        self.model[0].requires_grad_(False)
        # configure norm for EATTA updates: enable grad + force batch statistics
        for m in self.model[1].modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(True)
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
            elif isinstance(m, nn.BatchNorm1d):
                m.requires_grad_(True)
            elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
                m.requires_grad_(True)

    def reset(self):
        if self.model_states is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        self.load_model_and_optimizer()
        self.w1_ema, self.w2_ema = 0, 0
        self.cls_num_count = [0 for _ in range(self.num_classes)]
        self.cls_diff = [0 for _ in range(self.num_classes)]
        self.global_step = 0
  
class select_sample(nn.Module):
    def __init__(self, oracle_num, device):
        self.last_cls = []
        self.oracle_num = oracle_num
        self.device = device
        
    def __call__(self, y_prime, sorted_indices, c_diff, cls_num_count, cls_diff):
        self.mask = torch.zeros(sorted_indices.size(0), dtype=torch.bool).to(self.device)
        if len(self.last_cls) == 0:
            can = y_prime[sorted_indices][:self.oracle_num]
            diff = c_diff[sorted_indices][:self.oracle_num]
            for idx, i in enumerate(can):
                cls_num_count[i] += 1
                cls_diff[i] = diff[idx]
                self.mask[idx] = True
            sorted_idx = sorted_indices[self.mask]
            for i in y_prime[sorted_idx]:
                self.last_cls.append(i.item())
        else:
            can = y_prime[sorted_indices][:self.oracle_num]
            diff = c_diff[sorted_indices][:self.oracle_num]
            
            for idx, i in enumerate(can):
                if i.item() not in self.last_cls:
                    cls_num_count[i.item()] += 1
                    cls_diff[i.item()] = diff[idx]
                    self.mask[idx] = True
                else:
                    if diff[idx] < cls_diff[i]:
                        can2 = y_prime[sorted_indices][self.oracle_num:]
                        diff2 = c_diff[sorted_indices][self.oracle_num:]
                        for idx_, j in enumerate(can2):
                            if j not in self.last_cls:
                                cls_num_count[j] += 1
                                cls_diff[j] = diff2[idx_]
                                self.mask[self.oracle_num + idx_] = True
                                break
                            else:
                                continue
                    else:
                        cls_num_count[i] += 1
                        cls_diff[i] = diff[idx]
                        self.mask[idx] = True
                
            sorted_idx = sorted_indices[self.mask]
            
            if len(sorted_idx) >= 1:
                for i in y_prime[sorted_idx]:
                    self.last_cls.append(i.item())
            else:
                sorted_idx = sorted_indices[:self.oracle_num]
            
            if len(self.last_cls) > 2:
                self.last_cls = self.last_cls[len(self.last_cls)-1:]
        
        return sorted_idx, cls_num_count, cls_diff
        
class SampleBuffer:
    def __init__(self, capacity=300):
        self.capacity = capacity
        self.buffer = []

    def add(self, samples, labels):
        new_entries = list(zip(samples, labels))
        if len(self.buffer) + len(new_entries) > self.capacity:
            # Remove the oldest samples if buffer is full
            excess = len(self.buffer) + len(new_entries) - self.capacity
            self.buffer = self.buffer[excess:]
        self.buffer.extend(new_entries)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

class Buffer(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample, label = self.data[idx]
        return sample, label



@torch.no_grad()
def update_w1_w2(w1, w2, w1_ema, w2_ema, momentum=0.8):
    if w1_ema == 0 and w2_ema == 0:
        w1_ema = w1
        w2_ema = w2
        return w1_ema, w2_ema
    else:
        w1_ema = momentum * w1_ema + (1-momentum) * w1
        w2_ema = momentum * w2_ema + (1-momentum) * w2
        return w1_ema, w2_ema
