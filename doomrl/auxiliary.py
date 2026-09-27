"""Auxiliary enemy-visibility head, after Lample & Chaplot's Arnold.

The policy's only learning signal in `defend_center` is a sparse +1 per kill,
which is a thin supervisor for "what does an enemy look like". ViZDoom's labels
buffer gives that away for free, so we train a second head off the shared CNN to
predict whether an enemy is on screen, with binary cross-entropy against the
engine's ground truth. Its gradients flow back into the convolutional trunk and
push the features to be enemy-aware.

The label rides along inside the observation dict under "enemy" only because
that is how SB3's rollout buffer carries per-step data to the update. The
feature extractor reads "screen" alone, so nothing privileged reaches the policy
at inference -- it is a training target, not an input.
"""

from __future__ import annotations

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, NatureCNN
from stable_baselines3.common.utils import explained_variance
from torch.nn import functional as F

# weight on the auxiliary loss relative to policy + value
AUX_COEF = 0.1


def algo_for(scenario: str):
    """The PPO class a scenario trains and loads with.

    Saving with AuxPPO and loading with plain PPO fails: the aux head adds an
    optimiser param group, so the saved optimiser state has one more group than
    a stock PPO builds.
    """
    from doomrl.scenarios import SCENARIOS

    return AuxPPO if SCENARIOS[scenario].aux_enemy else PPO


class ScreenOnlyExtractor(BaseFeaturesExtractor):
    """NatureCNN over observations["screen"]; ignores every other key."""

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        self.cnn = NatureCNN(
            observation_space["screen"], features_dim=features_dim
        )

    def forward(self, observations: dict[str, th.Tensor]) -> th.Tensor:
        return self.cnn(observations["screen"])


class AuxPPO(PPO):
    """PPO with an extra BCE head predicting enemy visibility.

    Overrides only the loss: everything else -- clipping, GAE, schedules -- is
    stock PPO, so results stay comparable with the other scenarios.
    """

    def __init__(self, *args, aux_coef: float = AUX_COEF, **kwargs):
        self.aux_coef = aux_coef
        super().__init__(*args, **kwargs)

    def _setup_model(self) -> None:
        # built here rather than in __init__ so it also exists on the load path,
        # where SB3 constructs with _init_setup_model=False and self.policy does
        # not exist yet. The extra optimiser param group must be present before
        # set_parameters() restores the saved optimiser state, or the group
        # counts disagree and loading fails.
        super()._setup_model()
        self.aux_head = th.nn.Linear(
            self.policy.features_extractor.features_dim, 1
        ).to(self.device)
        self.policy.optimizer.add_param_group(
            {"params": list(self.aux_head.parameters())}
        )

    def _excluded_save_params(self) -> list[str]:
        # recreated by _setup_model; pickling a live module would pin a device
        return super()._excluded_save_params() + ["aux_head"]

    def _aux_loss(self, observations) -> th.Tensor:
        features = self.policy.extract_features(
            observations, self.policy.features_extractor
        )
        logits = self.aux_head(features).squeeze(-1)
        # frame stacking gives one flag per stacked frame; the last is current
        target = observations["enemy"][:, -1].float()
        return F.binary_cross_entropy_with_logits(logits, target)

    def train(self) -> None:
        """Stock PPO update with one extra term in the loss.

        Mirrors stable_baselines3 2.9 PPO.train(). The only changes are the
        `aux_loss` term added to `loss`, including the aux head in the gradient
        clip, and logging `train/aux_loss`. If SB3's PPO.train() changes
        materially, re-sync this.
        """
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses, aux_losses = [], []
        pg_losses, value_losses = [], []
        clip_fractions = []
        continue_training = True

        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations, actions
                )
                values = values.flatten()
                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (
                        advantages.std() + 1e-8
                    )

                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(
                    ratio, 1 - clip_range, 1 + clip_range
                )
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                pg_losses.append(policy_loss.item())
                clip_fractions.append(
                    th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                )

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                if entropy is None:
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)
                entropy_losses.append(entropy_loss.item())

                # --- the auxiliary term ---------------------------------
                aux_loss = self._aux_loss(rollout_data.observations)
                aux_losses.append(aux_loss.item())

                loss = (
                    policy_loss
                    + self.ent_coef * entropy_loss
                    + self.vf_coef * value_loss
                    + self.aux_coef * aux_loss
                )

                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = (
                        th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    )
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl")
                    break

                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(
                    list(self.policy.parameters()) + list(self.aux_head.parameters()),
                    self.max_grad_norm,
                )
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten()
        )
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/aux_loss", np.mean(aux_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
