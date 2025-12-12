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
    """EATTA with budget-aware, dynamic labeling:
       - per-sample utility: diff(x_i)
       - per-batch informativeness via EMA of top-k diffs
       - global label budget ratio (e.g., 50% of baseline labels)
    """
    def __init__(self, cfg, model, num_classes):
        super().__init__(cfg, model, num_classes)
        # hyper-parameters (the same with EATA and SAR)
        set_deterministic(0)
        self.e_margin = cfg.EATA.MARGIN_E0 * math.log(num_classes)   
        # hyper-parameters for effortless active labeling
        self.cls_num_count = [0 for _ in range(num_classes)]
        self.cls_diff = [0 for _ in range(num_classes)]
        self.oracle_num = cfg.MODEL.ORACLE_NUM          # max labels per batch
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
        self.cfg = cfg

        # ===================== NEW: Budget-aware labeling =====================
        # target fraction of labels compared to vanilla EATTA
        # e.g., LABEL_RATIO = 0.5 ⇒ 50% of labels
        self.label_ratio = getattr(cfg.MODEL, "LABEL_RATIO", 1.0)

        # exponent for how strongly batch informativeness scales m_t
        self.budget_beta = getattr(cfg.MODEL, "BUDGET_BETA", 1.0)

        # EMA of batch informativeness score S_t (top-k diff)
        self.batch_score_ema = 0.0
        self.batch_score_momentum = getattr(cfg.MODEL, "BATCH_SCORE_MO", 1.0)

        # global accounting of labels
        self.total_labels_used = 0              # sum_t m_t
        self.total_possible_labels = 0          # oracle_num * #batches (baseline)
        self.k = 1
        # fractional budget residual (for rounding)
        self.budget_residual = 0.0
        # count processed batches
        self.global_step = 0
        self.sorted_idx = None
        # ======================================================================

        
    def loss_calculation(self, x, y):
        # forward
        imgs_test = x[0]
        features = self.featurizer(imgs_test)
        outputs = self.classifier(features)
        py, y_prime = F.softmax(outputs, dim=-1).max(1)

        batch_size = features.size(0)

        # entropy + reliable set
        entropys = self.softmax_entropy(outputs)
        ids1 = torch.where(entropys < self.e_margin)[0]

        # effortless active labeling (boundary-based index selection)
        noise = torch.randn(features.size()).to(self.device) * self.noise_std  # std=0.01
        fea = features.clone().detach() + noise
        out = self.classifier(fea)
        py2 = F.softmax(out, dim=-1)[:, y_prime]
        py2 = torch.diag(py2)
        diff = torch.abs(py - py2)  # Eq.(3)

        # ---------------------------------------------------------------------
        # 1) Batch informativeness S_t from top-k diffs
        # ---------------------------------------------------------------------
        sorted_indices_all = torch.argsort(diff, descending=True)

        k = min(self.k, batch_size)
        if k > 0:
            topk_diffs = diff[sorted_indices_all[:k]]
            S_t = topk_diffs.mean()
        else:
            S_t = torch.tensor(0.0, device=self.device)

        # update EMA of batch score
        if self.batch_score_ema == 0.0:
            self.batch_score_ema = S_t.item()
        else:
            self.batch_score_ema = (
                self.batch_score_momentum * self.batch_score_ema
                + (1.0 - self.batch_score_momentum) * S_t.item()
            )

        # ---------------------------------------------------------------------
        # 2) Global budget update & compute dynamic m_t (with residual)
        # ---------------------------------------------------------------------
        # this batch contributes oracle_num possible labels in the baseline
        self.total_possible_labels += self.oracle_num

        # target total labels so far (upper bound if we matched label_ratio)
        target_total_labels = self.label_ratio * self.total_possible_labels
        max_allowed_now = int(math.floor(target_total_labels - self.total_labels_used + 1e-6))
        if max_allowed_now < 0:
            max_allowed_now = 0

        # batch informativeness factor r_t
        eps = 1e-6
        r_t = S_t.item() / (self.batch_score_ema + eps) if self.batch_score_ema > 0 else 1.0

        # ideal fractional increment this batch (before clipping)
        ideal_increment = self.label_ratio * self.oracle_num * (r_t ** self.budget_beta)

        # accumulate fractional budget
        self.budget_residual += ideal_increment

        # initial integer suggestion from residual
        m_t = int(self.budget_residual)

        # clip by [0, oracle_num], global max_allowed_now, and batch size
        m_t = max(0, min(m_t, self.oracle_num, max_allowed_now, batch_size))

        # consume from residual what we actually spend
        self.budget_residual -= m_t

        # ---------------------------------------------------------------------
        # 3) Class-balanced sample selection (up to m_t samples)
        # ---------------------------------------------------------------------
        if m_t == 0:
            # we don't label anything this batch, pure entropy update below
            selected_idx = torch.tensor([], dtype=torch.long, device=self.device)
        else:
            # select at most m_t indices (class-balanced) from the
            # sorted list, using the existing select_sample logic
            self.sorted_idx, self.cls_num_count, self.cls_diff = self.select_(
                y_prime,
                sorted_indices_all,
                diff,
                self.cls_num_count,
                self.cls_diff,
                max_return=m_t     # NEW argument
            )
            selected_idx = self.sorted_idx

        # enforce that we never exceed the requested m_t
        if selected_idx.numel() > m_t:
            selected_idx = selected_idx[:m_t]

        # ---------------------------------------------------------------------
        # 4) Unsupervised entropy loss (remove labeled idx if any)
        # ---------------------------------------------------------------------
        if selected_idx.numel() > 0:
            # remove labeled indices from ids1, as in original EATTA
            mask = torch.ones_like(ids1, dtype=torch.bool)
            for si in selected_idx:
                mask = mask & (ids1 != si)
            ids1 = ids1[mask]

        # unsupervised entropy loss
        if ids1.numel() > 0:
            loss_ent = entropys[ids1].mean(0)
        else:
            loss_ent = entropys.mean(0)

        # ---------------------------------------------------------------------
        # 5) If no labels this batch: entropy-only update
        # ---------------------------------------------------------------------
        if selected_idx.numel() == 0:
            loss = 1.7 * loss_ent
            return outputs, loss

        # ---------------------------------------------------------------------
        # 6) Supervised CE loss on selected indices
        # ---------------------------------------------------------------------
        # we are using m_t labels this batch → update global usage
        self.total_labels_used += selected_idx.numel()

        if self.annotator == 'HUMAN':
            # ground-truth labels from y
            loss_ce = F.cross_entropy(outputs[selected_idx], y[selected_idx])
            oracle_labels = y[selected_idx]
        elif self.annotator == 'LARGE_MODEL':
            # labels from cloud model
            with torch.no_grad():
                imgs_cloud = imgs_test[selected_idx]
                cloud_outputs = self.cloud_model(imgs_cloud)
                py_c, y_prime_c = F.softmax(cloud_outputs, dim=-1).max(1)
            loss_ce = F.cross_entropy(outputs[selected_idx], y_prime_c.detach())
            oracle_labels = y_prime_c
        else:
            raise NotImplementedError
        
        # IF BUFFER (optional)
        if self.use_buffer:
            samples = copy.deepcopy(imgs_test)[selected_idx]
            self.samplebuffer.add(samples, oracle_labels)
            buffer_dataset = Buffer(self.samplebuffer.buffer)
            buffer_loader = DataLoader(buffer_dataset, batch_size=self.buffer_bs, shuffle=True)
            
            for imgs_buf, labels_buf in buffer_loader:
                imgs_buf, labels_buf = imgs_buf.to(self.device), labels_buf.to(self.device)
                features_o = self.featurizer(imgs_buf).view(imgs_buf.size(0), -1)
                outputs_o = self.classifier(features_o)
                loss_buffer = F.cross_entropy(outputs_o, labels_buf)
            loss_ce = loss_ce + loss_buffer
        
        # ---------------------------------------------------------------------
        # 7) Gradient norm-based debiasing (same as EATTA)
        # ---------------------------------------------------------------------
        grad1 = torch.autograd.grad(
            loss_ent,
            [param for param in self.model[1].parameters() if param.requires_grad],
            retain_graph=True
        )
        grad1_norm = torch.norm(torch.stack([g.norm() for g in grad1]))

        grad2 = torch.autograd.grad(
            loss_ce,
            [param for param in self.model[1].parameters() if param.requires_grad],
            retain_graph=True
        )
        grad2_norm = torch.norm(torch.stack([g.norm() for g in grad2]))
        
        w1 = 2 * grad2_norm / (grad2_norm + grad1_norm)
        w2 = 2 * grad1_norm / (grad2_norm + grad1_norm)
        w1 = w1.detach().item()
        w2 = w2.detach().item()
        
        self.w1_ema, self.w2_ema = update_w1_w2(
            w1, w2, self.w1_ema, self.w2_ema, self.mo
        )

        loss = self.w1_ema * loss_ent + self.w2_ema * loss_ce 

        return outputs, loss

        
    @torch.enable_grad()
    def forward_and_adapt(self, x, y):
        """Forward and adapt model on batch of data.
        Budget-aware:
          - decides inside loss_calculation how many labels to use (m_t),
            possibly 0 (entropy-only batch).
        """
        outputs, loss = self.loss_calculation(x, y)
       
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        self.global_step += 1
        
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

        # reset budget-aware state
        self.batch_score_ema = 0.0
        self.total_labels_used = 0
        self.total_possible_labels = 0
        self.budget_residual = 0.0
  
class select_sample(nn.Module):
    def __init__(self, oracle_num, device):
        self.last_cls = []
        self.oracle_num = oracle_num
        self.device = device
        
    def __call__(self, y_prime, sorted_indices, c_diff, cls_num_count, cls_diff, max_return=None):
        """
        y_prime: pseudo-labels
        sorted_indices: indices sorted by diff (descending)
        c_diff: diff scores
        max_return: maximum number of samples to actually select this batch.
                    If None, defaults to self.oracle_num.
        """
        if max_return is None:
            max_n = self.oracle_num
        else:
            max_n = int(max_return)

        # if we decide to use 0 labels this batch, return empty
        if max_n <= 0:
            empty_idx = torch.tensor([], dtype=torch.long, device=self.device)
            return empty_idx, cls_num_count, cls_diff

        self.mask = torch.zeros(sorted_indices.size(0), dtype=torch.bool).to(self.device)

        if len(self.last_cls) == 0:
            can = y_prime[sorted_indices][:max_n]
            diff = c_diff[sorted_indices][:max_n]
            for idx, i in enumerate(can):
                cls_num_count[i] += 1
                cls_diff[i] = diff[idx]
                self.mask[idx] = True
            sorted_idx = sorted_indices[self.mask]
            for i in y_prime[sorted_idx]:
                self.last_cls.append(i.item())
        else:
            can = y_prime[sorted_indices][:max_n]
            diff = c_diff[sorted_indices][:max_n]
            
            for idx, i in enumerate(can):
                if i.item() not in self.last_cls:
                    cls_num_count[i.item()] += 1
                    cls_diff[i.item()] = diff[idx]
                    self.mask[idx] = True
                else:
                    if diff[idx] < cls_diff[i]:
                        can2 = y_prime[sorted_indices][max_n:]
                        diff2 = c_diff[sorted_indices][max_n:]
                        for idx_, j in enumerate(can2):
                            if j not in self.last_cls:
                                cls_num_count[j] += 1
                                cls_diff[j] = diff2[idx_]
                                self.mask[max_n + idx_] = True
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
                # fallback: still respect max_n
                sorted_idx = sorted_indices[:max_n]
            
            if len(self.last_cls) > 2:
                self.last_cls = self.last_cls[len(self.last_cls)-1:]

            # safety: ensure we don't return more than max_n indices
        if sorted_idx.numel() > max_n:
            sorted_idx = sorted_idx[:max_n]
        
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
