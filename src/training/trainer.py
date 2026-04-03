"""
Improved training loop with:
- Learning-rate scheduling (cosine annealing or step decay)
- Gradient clipping
- Per-epoch metric logging
- Clean early-stopping that restores the best checkpoint in-memory

The Trainer is model-agnostic: pass any nn.Module that accepts
(x, edge_index, data) and returns log-softmax logits.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import average_precision_score, roc_auc_score


class Trainer:
    """
    Wraps the training loop with improved hyper-parameter handling.

    Parameters
    ----------
    model : nn.Module
    args : Namespace
        Must contain: lr, weight_decay, epochs, patience, dropout.
        Optional: lr_scheduler ('cosine' | 'step' | None).
    idx_train, idx_val, idx_test : LongTensor (meta-node indices)
    meta_y : LongTensor  shape (n_meta_nodes,)
    number_of_input_nodes : int
    device : str
    """

    def __init__(
        self,
        model: torch.nn.Module,
        args,
        idx_train: torch.Tensor,
        idx_val: torch.Tensor,
        idx_test: torch.Tensor,
        meta_y: torch.Tensor,
        number_of_input_nodes: int,
        device: str = 'cuda',
    ):
        self.model = model
        self.args = args
        self.idx_train = idx_train
        self.idx_val = idx_val
        self.idx_test = idx_test
        self.meta_y = meta_y
        self.number_of_input_nodes = number_of_input_nodes
        self.device = device

        # Optimiser
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        # LR scheduler (optional)
        sched = getattr(args, 'lr_scheduler', None)
        if sched == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=args.epochs, eta_min=1e-5
            )
        elif sched == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=500, gamma=0.5
            )
        else:
            self.scheduler = None

        # Metric history
        self.history: Dict[str, List[float]] = {
            'train_loss': [], 'val_loss': [],
            'train_aupr': [], 'val_aupr': [],
            'train_auroc': [], 'val_auroc': [],
        }

    # ──────────────────────────────────────────────────────────────────────────

    def _forward(self, loader) -> torch.Tensor:
        """Run a single forward pass; return meta-node log-probs."""
        data = next(iter(loader)).to(self.device)
        out = self.model(data.x.float(), data.edge_index, data).squeeze()
        return out[self.number_of_input_nodes:]

    def _metrics(self, output: torch.Tensor, idx: torch.Tensor) -> Dict[str, float]:
        loss = F.nll_loss(output[idx], self.meta_y[idx]).item()
        probs  = torch.exp(output[idx, 1]).cpu().detach().numpy()
        labels = self.meta_y[idx].cpu().detach().numpy()
        auroc = roc_auc_score(labels, probs)
        aupr  = average_precision_score(labels, probs)
        return {'loss': loss, 'auroc': auroc, 'aupr': aupr}

    # ──────────────────────────────────────────────────────────────────────────

    def fit(self, loader, epochs: Optional[int] = None,
            patience: Optional[int] = None) -> Dict[str, List[float]]:
        """
        Train model, apply early stopping, restore best weights.

        Returns the metric history dict.
        """
        epochs  = epochs  or self.args.epochs
        patience = patience or self.args.patience

        best_val_loss = float('inf')
        best_epoch = 0
        bad_counter = 0
        best_state: Optional[dict] = None

        for epoch in range(epochs):
            t0 = time.time()

            # ── Training step ──────────────────────────────────────────────
            self.model.train()
            data = next(iter(loader)).to(self.device)
            self.optimizer.zero_grad()
            out = self.model(data.x.float(), data.edge_index, data).squeeze()
            out = out[self.number_of_input_nodes:]

            # Use model's own loss if it has one (supports label smoothing)
            if hasattr(self.model, 'loss'):
                train_loss = self.model.loss(out[self.idx_train],
                                              self.meta_y[self.idx_train])
            else:
                train_loss = F.nll_loss(out[self.idx_train],
                                        self.meta_y[self.idx_train])

            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()

            # ── Evaluation ────────────────────────────────────────────────
            self.model.eval()
            with torch.no_grad():
                out_eval = self._forward(loader)

            tr = self._metrics(out_eval, self.idx_train)
            vl = self._metrics(out_eval, self.idx_val)

            for k, v in tr.items():
                self.history[f'train_{k}'].append(v)
            for k, v in vl.items():
                self.history[f'val_{k}'].append(v)

            # ── Logging ───────────────────────────────────────────────────
            if epoch % 50 == 0:
                lr = self.optimizer.param_groups[0]['lr']
                print(
                    f"Epoch {epoch:04d} | "
                    f"Train Loss: {train_loss.item():.4f} | "
                    f"Val Loss: {vl['loss']:.4f} | "
                    f"Val AUPR: {vl['aupr']:.4f} | "
                    f"LR: {lr:.2e} | "
                    f"Time: {time.time()-t0:.1f}s"
                )

            # ── Early stopping ────────────────────────────────────────────
            if vl['loss'] < best_val_loss:
                best_val_loss = vl['loss']
                best_epoch = epoch
                bad_counter = 0
                best_state = {k: v.clone() for k, v in
                              self.model.state_dict().items()}
            else:
                bad_counter += 1

            if bad_counter >= patience:
                print(f"  Early stopping at epoch {epoch} "
                      f"(best epoch: {best_epoch})")
                break

        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)
            print(f"  Restored weights from epoch {best_epoch}")

        return self.history

    # ──────────────────────────────────────────────────────────────────────────

    def evaluate(self, loader) -> Dict[str, float]:
        """Compute test-set metrics using the current (best) model."""
        self.model.eval()
        with torch.no_grad():
            out = self._forward(loader)
        return self._metrics(out, self.idx_test)

    def predict(self, loader) -> torch.Tensor:
        """Return cancer-gene probabilities for all meta-nodes."""
        self.model.eval()
        with torch.no_grad():
            out = self._forward(loader)
        return torch.exp(out)
