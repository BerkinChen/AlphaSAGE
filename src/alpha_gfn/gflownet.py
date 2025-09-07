import torch
from typing import Tuple
from gfn.containers import Trajectories
from gfn.env import Env
from gfn.gflownet.trajectory_balance import TBGFlowNet
from gfn.modules import DiscretePolicyEstimator
from torchtyping import TensorType as TT


class EntropyTBGFlowNet(TBGFlowNet):
    def __init__(
        self,
        pf: DiscretePolicyEstimator,
        pb: DiscretePolicyEstimator,
        entropy_coef: float = 0.0,
        entropy_temperature: float = 1.0,
        **kwargs,
    ):
        super().__init__(pf, pb, **kwargs)
        self.entropy_coef = entropy_coef
        self.entropy_temperature = entropy_temperature

    def get_trajectories_scores(
        self, trajectories: Trajectories, recalculate_all_logprobs: bool = False
    ) -> Tuple[
        TT["n_trajectories", torch.float],
        TT["n_trajectories", torch.float],
        TT["n_trajectories", torch.float],
        TT["n_trajectories", torch.float]
    ]:
        if self.entropy_coef <= 0:
            return (*super().get_trajectories_scores(trajectories, recalculate_all_logprobs), torch.zeros(trajectories.n_trajectories, device=trajectories.states.device))

        log_pf, log_pb, scores = super().get_trajectories_scores(
            trajectories, recalculate_all_logprobs
        )

        # Calculate entropy term
        entropy_term = torch.zeros(trajectories.n_trajectories, device=trajectories.states.device)
        if trajectories.estimator_outputs is not None:
            logits = trajectories.estimator_outputs  # (max_len, batch_size, n_actions)
            temp_logits = logits / self.entropy_temperature
            probs = torch.softmax(temp_logits, dim=-1)
            entropy_per_step = -torch.sum(
                probs * torch.log(probs.clamp(min=1e-9)), dim=-1
            )  # (max_len, batch_size)

            # Mask out invalid steps for each trajectory
            max_len = logits.shape[0]
            step_indices = torch.arange(max_len, device=logits.device).unsqueeze(1)
            valid_steps_mask = step_indices < trajectories.when_is_done.unsqueeze(0)

            masked_entropy = entropy_per_step * valid_steps_mask

            # Sum entropy over trajectory steps for each trajectory in the batch
            entropy_sum_per_trajectory = masked_entropy.sum(dim=0)  # (batch_size,)

            entropy_term = entropy_sum_per_trajectory * self.entropy_coef
        elif self.entropy_coef > 0:
            # Fallback for older gfn versions or if outputs are not saved
            non_terminal_mask = ~trajectories.is_done
            if torch.any(non_terminal_mask):
                # This part is not batched correctly for n_trajectories > 1
                # but kept for compatibility with the old logic.
                non_terminal_states = trajectories.states[non_terminal_mask]

                preprocessed_states = self.pf.preprocessor(non_terminal_states)
                logits = self.pf.module(preprocessed_states)

                temp_logits = logits / self.entropy_temperature

                probs = torch.softmax(temp_logits, dim=-1)
                entropy = -torch.sum(
                    probs * torch.log(probs.clamp(min=1e-9)), dim=-1
                )

                entropy_term = entropy.sum() * self.entropy_coef

        return log_pf, log_pb, scores, entropy_term
    
    def loss(
        self,
        env: Env,
        trajectories: Trajectories,
        recalculate_all_logprobs: bool = False,
    ) -> TT[0, float]:
        """Trajectory balance loss.

        The trajectory balance loss is described in 2.3 of
        [Trajectory balance: Improved credit assignment in GFlowNets](https://arxiv.org/abs/2201.13259))

        Raises:
            ValueError: if the loss is NaN.
        """
        del env  # unused
        _, _, scores, entropy_term = self.get_trajectories_scores(
            trajectories, recalculate_all_logprobs=recalculate_all_logprobs
        )
        loss = (scores + self.logZ).pow(2).mean() + entropy_term.mean()
        if torch.isnan(loss):
            # set inf
            loss = torch.tensor(float('inf'))

        return loss

