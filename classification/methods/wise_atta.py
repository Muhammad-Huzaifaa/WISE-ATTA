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
from models.model import split_up_model
from utils.misc import set_deterministic
from torch.utils.data import Dataset, DataLoader
from collections import deque


@ADAPTATION_REGISTRY.register()
class WISE(TTAMethod):
    """Tent adapts a model by entropy minimization during testing.
    Once tented, a model adapts itself by updating on every forward.
    """
    def __init__(self, cfg, model, num_classes):
        super().__init__(cfg, model, num_classes)
        self.e_margin = cfg.EATA.MARGIN_E0 * math.log(num_classes)
        set_deterministic(cfg.SEED)  
        # hyper-parameters for effortless active labeling
        self.cls_num_count = [0 for _ in range(num_classes)]
        self.cls_diff = [0 for _ in range(num_classes)]
        self.oracle_num = cfg.MODEL.ORACLE_NUM
        self.select_ = select_sample(self.oracle_num, self.device)
        self.noise_std = 0.01
        # hyper-parameters for gradient norm-based denoising
        self.mo = 0.8
        # loss function for unannotated samples
        self.softmax_entropy = Entropy()
        #
        self.cfg = cfg
        # ------------------------------------------------------------------------
        self.lr_step1 = getattr(cfg.OPTIM, "LR_STEP1", cfg.OPTIM.LR)
        self.lr_step2 = getattr(cfg.OPTIM, "LR_STEP2", cfg.OPTIM.LR)
        # ------------------------------------------------------
        
        
    def loss_calculation(self, x, y):

        if self.current_inner_step == 0:
            self.batch_count += 1

            # Compute outputs for utility (no labels needed)
            with torch.no_grad():
                feats_s = self.featurizer(x[0])
                out_s = self.classifier(feats_s)
                ent = self.softmax_entropy(out_s)  # [B]
                # py, y_prime = F.softmax(out_s, dim=-1).max(1)
                # p_s = F.softmax(out_s, dim=-1)
                # feats_a = self.anchor_featurizer(x[0])
                # out_a = self.anchor_classifier(feats_a)
                # p_a = F.softmax(out_a, dim=-1)
                # div = torch.norm(p_s - p_a, p=2, dim=1)  # [B]
                # u_div = (div.topk(k=1, largest=True).values).mean().item()
                # count entropy idx that is less than margin
                ids = torch.where(ent < self.e_margin)[0]
                utility = len(ids) #+ u_div

            self.use_labels = self.batch_selector.decide(utility)
            if self.use_labels:
                self.count += 1

        # ------------------------------------------------------
        if not self.use_labels and self.current_inner_step == 0:
            return None, None
        elif not self.use_labels and self.current_inner_step == 1:
            features = self.featurizer(x[0])
            outputs = self.classifier(features)
            entropys = self.softmax_entropy(outputs)
            ids1 = torch.where(entropys < self.e_margin)[0]
            entropys_loss = entropys[ids1].mean(0)
            return outputs, 0.1 * entropys_loss
        else:
        # forward
            imgs_test = x[0]
            features = self.featurizer(imgs_test)
            outputs = self.classifier(features)
            py, y_prime = F.softmax(outputs, dim=-1).max(1)

            entropys = self.softmax_entropy(outputs)
            ids1 = torch.where(entropys < self.e_margin)[0]
        
            py, y_prime = F.softmax(outputs, dim=-1).max(1)

            if self.current_inner_step == 0:
                self.cache_output = outputs
                # effortless active labeling
                # ----------------
                if self.batch_count <= 4:
                    noise = torch.randn(features.size()).to(self.device) * self.noise_std 
                    fea = features.clone().detach() + noise
                    out = self.classifier(fea)
                    py2 = F.softmax(out, dim=-1)[:, y_prime]
                    py2 = torch.diag(py2)
                    diff = torch.abs(py - py2) # Eq.(3)
                # forgetting-aware active labeling 
                else:
                    with torch.no_grad():
                        anchor_features = self.anchor_featurizer(imgs_test)
                        anchor_outputs = self.anchor_classifier(anchor_features)  # same 200-dim head
                        p_anchor = F.softmax(anchor_outputs, dim=-1)
                    p_student = F.softmax(outputs, dim=-1)
                    diff = torch.norm(p_student - p_anchor, p=2, dim=1)  # [B]
                # ------------------------------------------------------------------
                sorted_indices = torch.argsort(diff, descending=True)
                self.sorted_idx, self.cls_num_count, self.cls_diff = self.select_(y_prime, sorted_indices, diff, self.cls_num_count, self.cls_diff)
                for i in self.sorted_idx:
                    for j in ids1:
                        if i==j:
                            mask = ids1 != i
                            ids1 = ids1[mask]
                loss_ent = entropys[ids1]
                loss_ent = loss_ent.mean(0) 
                # ------------------------------------------------------------------

            loss_ce = F.cross_entropy(outputs[self.sorted_idx], y[self.sorted_idx])
                    
            if self.current_inner_step == 0:
                loss = 1.7 * loss_ent + 0.15 * loss_ce 
            else:
                loss = 0.15 * loss_ce 

            return self.cache_output, loss
        
        

    @torch.enable_grad()
    def forward_and_adapt(self, x, y):
        """Forward and adapt model on batch of data.
        Measure entropy of the model prediction, take gradients, and update params.
        """       
        # -----------------------------------------------------
        apply_grad = True if self.current_inner_step == 1 else False
        if hasattr(self.model[1], "model"):
            backbone = self.model[1].model

            # ----- CASE 1: ResNet classifier -----
            if hasattr(backbone, "fc"):
                # ResNet classifier
                for name, p in backbone.fc.named_parameters():
                    if name == "bias":
                        p.requires_grad_(apply_grad)   # adapt only bias
                    else:
                        p.requires_grad_(False)        # freeze weight

            elif hasattr(backbone, "heads"):
                heads = backbone.heads
                if hasattr(heads, "head"):
                    # timm ViT: heads.head
                    for name, p in heads.head.named_parameters():
                        if name == "bias":
                            p.requires_grad_(apply_grad)
                        else:
                            p.requires_grad_(False)
                else:
                    for p in heads.parameters():
                        p.requires_grad_(False)

        # set learning rate
        if self.current_inner_step == 0:
            lr = self.lr_step1
        else:
            lr = self.lr_step2

        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

        outputs, loss = self.loss_calculation(x, y)

        if self.use_labels:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        else:
            if self.current_inner_step == 1:
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()


        # ---------------- EMA update of anchor teacher ----------------
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
        # print total number of labeled samples used
        # if self.batch_count % 100 == 0 or self.batch_count == 468:
        #     print(f"Total labeled samples used so far: {self.count}")
        #     print(f"Current batch count: {self.batch_count:.2f}")
        return outputs

    def collect_params(self):
        """Collect the affine scale + shift parameters from batch norms.

        Walk the model's modules and collect all batch normalization parameters.
        Return the parameters and their names.

        Note: other choices of parameterization are possible!
        """
        params = []
        names = []

        ds = self.cfg.CORRUPTION.DATASET
        is_imagenet_c = (ds == "imagenet_c")
        is_other = (ds in {"imagenet_r", "imagenet_a", "imagenet_k"})

        # If dataset isn't one of the expected ones, keep old behavior (return empty)
        if not (is_imagenet_c or is_other):
            return params, names

        for nm, m in self.model[1].named_modules():

            # imagenet_c only: skip layer4
            if is_imagenet_c and ("layer4" in nm):
                continue

            # collect norm params (only if currently requires_grad=True)
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                for np, p in m.named_parameters():
                    if np in ["weight", "bias"] and p.requires_grad:
                        params.append(p)
                        names.append(f"{nm}.{np}")

            # imagenet_c only: force layer1 params to be trainable + include them
            if is_imagenet_c and ("layer1" in nm):
                for np, p in m.named_parameters():
                    if np in ["weight", "bias"]:
                        p.requires_grad_(True)
                        params.append(p)
                        names.append(f"{nm}.{np}")

        # --- include classifier head (ResNet or ViT) ---
        if hasattr(self.model[1], "model"):
            backbone = self.model[1].model

            head_module = None
            # ResNet-style
            if hasattr(backbone, "fc"):
                head_module = backbone.fc
                head_name_prefix = "fc"
            # ViT-style (timm): heads.head
            elif hasattr(backbone, "heads"):
                if hasattr(backbone.heads, "head"):
                    head_module = backbone.heads.head
                    head_name_prefix = "heads.head"
                else:
                    head_module = backbone.heads
                    head_name_prefix = "heads"

            if head_module is not None:
                for np, p in head_module.named_parameters():
                    # train only bias
                    if np == "bias":
                        p.requires_grad_(True)
                        params.append(p)
                        names.append(f"{head_name_prefix}.{np}")
                    else:
                        # freeze weight
                        p.requires_grad_(False)

        return params, names


  

    def configure_model(self):
      
        # self.edge_model = self.model[1]
        # self.cloud_model = self.model[0]
        """Configure model for use with tent."""
        # train mode, because tent optimizes the model to minimize entropy
        self.model[1].eval()  # eval mode to avoid stochastic depth in swin. test-time normalization is still applied
        self.model[0].eval()
        # disable grad, to (re-)enable only what tent updates
        self.model[1].requires_grad_(False)
        self.model[0].requires_grad_(False)
        # configure norm for tent updates: enable grad + force batch statisics
        for m in self.model[1].modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(True)
                # force use of batch stats in train and eval modes
                # m.track_running_stats = True
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
        self.cls_num_count = [0 for _ in range(self.num_classes)]
        self.cls_diff = [0 for _ in range(self.num_classes)]
    
    def reset_config(self, cfg):
        self.label_ratio = getattr(cfg.MODEL, "LABEL_RATIO", 0.5)  # e.g. 0.1
        self.use_labels = False
        self.label_budget = 0.0
        self.count = 0
        self.total_batches = getattr(cfg, "TOTAL_TEST_BATCHES", None)  # or pass explicitly
        self.total_labels = int(round(self.label_ratio * self.total_batches))
        self.batch_selector = PacedBatchSelector(
            total_batches=self.total_batches,
            total_labels=self.total_labels,
            window=250,
            min_hist=1,
            slack=1.0,   
            device=self.device
        )
        arch_name = cfg.MODEL.EDGE_ARCH
        self.featurizer, self.classifier = split_up_model(self.model[1], arch_name, self.dataset_name) 
        for param in self.model[0].parameters():
            param.detach_()
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
        self.batch_count = 0
        # ------------------------------------------------------------------------
        self.sorted_idx = None
        self.current_inner_step = 0  # will be set by the outer loop
        self.lr_step1 = getattr(cfg.OPTIM, "LR_STEP1", cfg.OPTIM.LR)
        self.lr_step2 = getattr(cfg.OPTIM, "LR_STEP2", cfg.OPTIM.LR)
        self.cache_output = None
        self.use_labels = False
        self.label_budget = 0.0
        self.count = 0
        # ------------------------------------------------------


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
            
            if len(sorted_idx)>=1:
                for i in y_prime[sorted_idx]:
                    self.last_cls.append(i.item())
            else:
                
                sorted_idx = sorted_indices[:self.oracle_num]
                
            # self.last_cls.append(y_prime[sorted_idx])
            if len(self.last_cls)>2:
                self.last_cls = self.last_cls[len(self.last_cls)-1:]
        
        return sorted_idx, cls_num_count, cls_diff
    

class PacedBatchSelector:
    def __init__(self, total_batches: int, total_labels: int,
                 window: int = 200, min_hist: int = 30,
                 slack: float = 1.0, device="cuda"):
        self.total_batches = int(total_batches)
        self.total_labels = int(total_labels)

        self.device = device
        self.window = int(window)
        self.min_hist = int(min_hist)
        self.hist = deque(maxlen=self.window)

        self.used_labels = 0
        self.seen_batches = 0
        self.slack = float(slack)  # allow small deviations from target

    @property
    def label_ratio(self):
        return self.total_labels / max(1, self.total_batches)

    def decide(self, utility_raw: float) -> bool:
        """
        Force spend if behind schedule; else select by utility threshold,
        while still guaranteeing total budget used by end.
        """
        with torch.no_grad():
            self.seen_batches += 1
            remaining_batches = self.total_batches - self.seen_batches + 1
            remaining_labels = self.total_labels - self.used_labels

            # no budget left
            if remaining_labels <= 0:
                self.hist.append(utility_raw)
                return False

            # hard guarantee: if labels >= batches left, label everything remaining
            if remaining_labels >= remaining_batches:
                self.used_labels += 1
                self.hist.append(utility_raw)
                return True

            # pacing target
            target_spent = self.label_ratio * self.seen_batches

            # if behind schedule (by more than slack), force label now
            if self.used_labels + self.slack < target_spent:
                self.used_labels += 1
                self.hist.append(utility_raw)
                return True

            # otherwise: "important batch" selection
            p = remaining_labels / max(1, remaining_batches)  # desired rate from now on

            if len(self.hist) < self.min_hist:
                # mild exploration early to avoid back-loading
                do_label = (torch.rand(1).item() < p)
            else:
                vals = torch.tensor(list(self.hist), device=self.device, dtype=torch.float32)
                q = torch.quantile(vals, 1.0 - float(p))
                do_label = (utility_raw >= q.item())

            if do_label:
                self.used_labels += 1

            self.hist.append(utility_raw)
            return do_label
    