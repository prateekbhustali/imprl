Remarks:

- Discount factor: 

We distinguish between the discount factors in agents and environments.

    (A) Environments with an inherent notion of discounting
        MDPs defined with a discount factor are evaluated with discounting, 
        since the optimization objective is to maximize the expected *discounted* return.

        For evaluation, the return is computed as the sum of discounted rewards:
            G = sum_{t=0}^{T} gamma^t * r_t

        The agent uses the same discount factor during training 
        to compute the target value for the value function.

    (B) Environments without an inherent notion of discounting
        In this case, the optimization objective is to maximize the expected return,
        which is the sum of rewards. 
        
        (B1) If a discount factor is required to impose bounded sums, it
             becomes part of the solution method (algorithm).
            The agent may use any discount factor during training to compute the target value.

        (B2) If the environment has a finite time horizon (and rewards are bounded),
            additional discounting is not required to impose bounded sums. 
            The agent uses a discount factor of 1.0 to compute the target value.

        For (B1) and (B2), the optimization objective is to maximize 
        the expected return, which is the sum of rewards:
            G = sum_{t=0}^{T} r_t


Example: <S, A, R, T, gamma> with a finite time horizon T=5

The return is computed as the sum of discounted rewards:
    G = r_0 + gamma * r_1 + gamma^2 * r_2 + gamma^3 * r_3 + gamma^4 * r_4

    The target value for the value function is computed as:
    td_target = r_0 + gamma * V(s_1)

    or we can add the discount factor to the reward function since t is part of the state:
    R(s, a) = R'((t, s'), a)

    r_t' = gamma^t * r_t

    G = r_0' + r_1' + r_2' + r_3' + r_4'

    The target value for the value function is computed as:
    td_target = r_0' + V(s_1)

References:
- Examining average and discounted reward optimality criteria in reinforcement learning (https://arxiv.org/abs/2107.01348)
- Empirical Design in Reinforcement Learning (https://arxiv.org/abs/2304.01315)