import torch
from torch import Tensor

def detailed_balance_loss(
    pf_logits: Tensor,
    pb_logits: Tensor,
    rewards: Tensor,
    term_flow: float = 1.0
):
    """
    Computes the detailed balance loss for a GFlowNet.
    
    Args:
        pf_logits: Logits of the forward policy P_F(s'|s)
        pb_logits: Logits of the backward policy P_B(s|s')
        rewards: Rewards R(s') of the terminal states
        term_flow: The flow of the terminal state F(s_f)
    """
    
    pf_probs = torch.softmax(pf_logits, dim=-1)
    pb_probs = torch.softmax(pb_logits, dim=-1)
    
    # Select the probabilities of the actions taken
    forward_flow = pf_probs.gather(-1, ...).squeeze(-1)
    backward_flow = pb_probs.gather(-1, ...).squeeze(-1)
    
    loss = (torch.log(term_flow) + torch.log(forward_flow) - torch.log(rewards) - torch.log(backward_flow)) ** 2
    return loss.mean()

def trajectory_balance_loss(logZ: Tensor, total_log_P_F: Tensor, total_log_P_B: Tensor, log_reward: Tensor):
    """
    Computes the trajectory balance loss for a GFlowNet.
    """
    return (logZ + total_log_P_F - total_log_P_B - log_reward).pow(2)
