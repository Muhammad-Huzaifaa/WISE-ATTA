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
from utils.misc import SelectionStatsLogger
from torch.utils.data import Dataset, DataLoader


@ADAPTATION_REGISTRY.register()
class EATTA(TTAMethod):
    """Tent adapts a model by entropy minimization during testing.
    Once tented, a model adapts itself by updating on every forward.
    """
    def __init__(self, cfg, model, num_classes):
        super().__init__(cfg, model, num_classes)
        # hyper-parameters (the same with EATA and SAR)
        self.e_margin = cfg.EATA.MARGIN_E0 * math.log(num_classes)
        set_deterministic(cfg.SEED)   # 0, 40, 100
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
        # loss function for annotated samples -> cross entropy loss: F.cross_entropy(X,X)
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
        self.anchor_momentum = getattr(cfg.MODEL, "ANCHOR_MOMENTUM", 0.90)
        self.count = 0
        # ------------------------------------------------------------------------
        self.log_selection = getattr(cfg, "LOG_SELECTION", True)
        self.query_strategy = getattr(cfg.MODEL, "QUERY_STRATEGY", "pds_drift")  # default

        self.log_path = os.path.join(
            "/mnt/SAS_A/huzaifa/research/RobustAdaptation/plots",
            f"selection_stats_{cfg.CORRUPTION.DATASET}_{self.query_strategy}_seed{cfg.SEED}.csv"
        )
        self.stats_logger = SelectionStatsLogger(self.log_path) if self.log_selection else None

        # shadow selectors for fair comparison (independent state from training selector)
        self.sel_entropy = select_sample(self.oracle_num, self.device)
        self.sel_eatta   = select_sample(self.oracle_num, self.device)
        self.sel_pds     = select_sample(self.oracle_num, self.device)

        self.cls_cnt_entropy = [0 for _ in range(num_classes)]
        self.cls_cnt_eatta   = [0 for _ in range(num_classes)]
        self.cls_cnt_pds     = [0 for _ in range(num_classes)]

        self.cls_diff_entropy = [0 for _ in range(num_classes)]
        self.cls_diff_eatta   = [0 for _ in range(num_classes)]
        self.cls_diff_pds     = [0 for _ in range(num_classes)]

        self.batch_id = 0  # increments each forward
        
    def loss_calculation(self, x, y):
        # forward
        imgs_test = x[0]
        features = self.featurizer(imgs_test)
        outputs = self.classifier(features)
        py, y_prime = F.softmax(outputs, dim=-1).max(1)

        entropys = self.softmax_entropy(outputs)


        with torch.no_grad():
            # ---- PDS drift scores for ALL samples ----
            anchor_features = self.anchor_featurizer(imgs_test)
            anchor_outputs = self.anchor_classifier(anchor_features)
            p_anchor = F.softmax(anchor_outputs, dim=-1)
            p_student = F.softmax(outputs, dim=-1)
            pds_drift_vec = torch.norm(p_student - p_anchor, p=2, dim=1)  # [B]

            # ---- EATTA noise-sensitivity scores for ALL samples ----
            noise = torch.randn_like(features) * self.noise_std
            out_noise = self.classifier(features + noise)
            # prob of the original pseudo-label after perturbation
            py2 = F.softmax(out_noise, dim=-1).gather(1, y_prime.view(-1, 1)).squeeze(1)
            eatta_diff_vec = torch.abs(py - py2)  # [B]

        # --- Max-Entropy criterion (largest entropy first) ---
        sorted_ent = torch.argsort(entropys.detach(), descending=True)
        idx_ent, self.cls_cnt_entropy, self.cls_diff_entropy = self.sel_entropy(
            y_prime, sorted_ent, entropys.detach(), self.cls_cnt_entropy, self.cls_diff_entropy
        )

        # --- EATTA criterion (largest noise diff first) ---
        sorted_eatta = torch.argsort(eatta_diff_vec, descending=True)
        idx_eatta, self.cls_cnt_eatta, self.cls_diff_eatta = self.sel_eatta(
            y_prime, sorted_eatta, eatta_diff_vec, self.cls_cnt_eatta, self.cls_diff_eatta
        )

        # --- PDS criterion (largest drift first) ---
        sorted_pds = torch.argsort(pds_drift_vec, descending=True)
        idx_pds, self.cls_cnt_pds, self.cls_diff_pds = self.sel_pds(
            y_prime, sorted_pds, pds_drift_vec, self.cls_cnt_pds, self.cls_diff_pds
        )

        # Log ONLY the first selected index (works for oracle_num=1 too)
        if self.stats_logger is not None:
            def _log(criterion: str, idx_tensor: torch.Tensor):
                j = int(idx_tensor[0].item())
                self.stats_logger.log(
                    step=int(self.count),
                    batch_id=int(self.batch_id),
                    run_strategy=self.query_strategy,
                    criterion=criterion,
                    used_for_update=int(criterion == self.query_strategy),
                    sel_idx=j,
                    entropy_sel=float(entropys[j].detach().item()),
                    drift_sel=float(pds_drift_vec[j].detach().item()),
                )
            _log("max_entropy", idx_ent)
            _log("eatta_noise", idx_eatta)
            _log("pds_drift", idx_pds)

            if self.query_strategy == "max_entropy":
                sorted_idx = idx_ent
            elif self.query_strategy == "eatta_noise":
                sorted_idx = idx_eatta
            elif self.query_strategy == "pds_drift":
                sorted_idx = idx_pds
            else:
                raise ValueError(f"Unknown QUERY_STRATEGY={self.query_strategy}")



        ids1 = torch.where(entropys < self.e_margin)[0]
       
        py, y_prime = F.softmax(outputs, dim=-1).max(1)

        # effortless active labeling
        if self.count == 0:
            noise = torch.randn(features.size()).to(self.device) * self.noise_std # std=0.01
            fea = features.clone().detach() + noise
            out = self.classifier(fea)
            py2 = F.softmax(out, dim=-1)[:, y_prime]
            py2 = torch.diag(py2)
            diff = torch.abs(py - py2) # Eq.(3)
            sorted_indices = torch.argsort(diff, descending=True)
            sorted_idx, self.cls_num_count, self.cls_diff =  self.select_(y_prime, sorted_indices, diff, self.cls_num_count, self.cls_diff)

        # # ---------------- forgetting-aware active labeling ----------------
        # # use prediction drift between student and EMA anchor as sample score
        # else:
        #     with torch.no_grad():
        #         anchor_features = self.anchor_featurizer(imgs_test)
        #         anchor_outputs = self.anchor_classifier(anchor_features)  # same 200-dim head
        #         p_anchor = F.softmax(anchor_outputs, dim=-1)

        #     p_student = F.softmax(outputs, dim=-1)
        #     # L2 distance in probability space per sample
        #     diff = torch.norm(p_student - p_anchor, p=2, dim=1)  # [B]

        #     # sort by drift (largest first)
        #     sorted_indices = torch.argsort(diff, descending=True)
        #     sorted_idx, self.cls_num_count, self.cls_diff = self.select_(
        #         y_prime, sorted_indices, diff, self.cls_num_count, self.cls_diff
        #     )
        # ------------------------------------------------------------------
        
        for i in sorted_idx:
            for j in ids1:
                if i==j:
                    mask = ids1 != i
                    ids1 = ids1[mask]
              
        

        loss_ent = entropys[ids1]
        loss_ent = loss_ent.mean(0) 
                    
        
        
        if self.annotator == 'HUMAN':
            loss_ce = F.cross_entropy(outputs[sorted_idx], y[sorted_idx])
            
        elif self.annotator == 'LARGE_MODEL':
            with torch.no_grad():
                imgs_cloud = imgs_test[sorted_idx]
                cloud_outputs = self.cloud_model(imgs_cloud)
                py_c, y_prime_c = F.softmax(cloud_outputs, dim=-1).max(1)
            loss_ce = F.cross_entropy(outputs[sorted_idx], y_prime_c.detach())
        else:
            raise NotImplementedError
        
         
        # # IF BUFFER
        if self.use_buffer:
            oracle_labels = y_prime_c if self.annotator == 'large_model' else y[sorted_idx]
            samples = copy.deepcopy(imgs_test)[sorted_idx]
            self.samplebuffer.add(samples,oracle_labels)
            buffer_dataset = Buffer(self.samplebuffer.buffer)
            buffer_loader = DataLoader(buffer_dataset, batch_size=self.buffer_bs, shuffle=True)
            
            for imgs, labels in buffer_loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                features_o = self.featurizer(imgs).view(imgs.size(0),-1)
                outputs_o = self.classifier(features_o)
                loss_buffer = F.cross_entropy(outputs_o, labels)
            loss_ce = loss_ce + loss_buffer
        
        # # gradient norm-based debiasing
        # Calculate gradients of loss1
        grad1 = torch.autograd.grad(loss_ent, list(param for param in self.model[1].parameters() if param.requires_grad), retain_graph=True)
        grad1_norm = torch.norm(torch.stack([g.norm() for g in grad1]))
        # Calculate gradients of loss2
        grad2 = torch.autograd.grad(loss_ce, list(param for param in self.model[1].parameters() if param.requires_grad), retain_graph=True)
        grad2_norm = torch.norm(torch.stack([g.norm() for g in grad2]))
        
        
        # 动态调整权重
        w1 = 2 * grad2_norm / (grad2_norm + grad1_norm)
        w2 = 2 * grad1_norm / (grad1_norm + grad2_norm)
        w1 = w1.detach().item()
        w2 = w2.detach().item()
        
        self.w1_ema, self.w2_ema = update_w1_w2(w1, w2, self.w1_ema, self.w2_ema, self.mo)
        
        
        loss = self.w1_ema * loss_ent + self.w2_ema * loss_ce 
        
        
        return outputs, loss
        
        

    @torch.enable_grad()
    def forward_and_adapt(self, x, y):
        """Forward and adapt model on batch of data.
        Measure entropy of the model prediction, take gradients, and update params.
        """
      
        outputs, loss = self.loss_calculation(x, y)
       
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
        
        self.count += 1
        self.batch_id += 1
        return outputs

    def collect_params(self):
        """Collect the affine scale + shift parameters from batch norms.

        Walk the model's modules and collect all batch normalization parameters.
        Return the parameters and their names.

        Note: other choices of parameterization are possible!
        """
        params = []
        names = []
        if self.cfg.CORRUPTION.DATASET == 'imagenet_c' or self.cfg.CORRUPTION.DATASET == 'imagenet_r' or self.cfg.CORRUPTION.DATASET == 'imagenet_a' or self.cfg.CORRUPTION.DATASET == 'imagenet_k':
            for nm, m in self.model[1].named_modules():
                if 'layer4' in nm:
                    continue

                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                    for np, p in m.named_parameters():
                        if np in ['weight', 'bias'] and p.requires_grad:  # weight is scale, bias is shift
                            params.append(p)
                            names.append(f"{nm}.{np}")

                if 'layer1' in nm:
                    for np, p in m.named_parameters():
                        if np in ['weight', 'bias']:  # weight is scale, bias is shift
                            p.requires_grad_(True)
                            params.append(p)
                            names.append(f"{nm}.{np}")

                        
        return params, names

  

    def configure_model(self):
      
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
    
    def reset_config(self, cfg):
        """Reset configuration according to cfg (e.g., after a domain shift)."""
        self.cfg = cfg
  
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
                
            if len(self.last_cls)>2:
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
    