"""RSL-RL's PPO update (returns, advantages, all epochs and minibatches) as one CUDA graph.

Reuses an existing `rsl_rl.algorithms.PPO`: its actor, critic, rollout storage
and hyperparameters. Only the control flow changes, so that every replay runs
the same kernels on the same memory:

- returns and advantages are written into the storage's buffers in place;
- the adaptive learning rate is a GPU tensor updated with torch.where, and
  Adam runs in capturable mode reading it;
- gradients are allocated inside the graph (zero_grad(set_to_none=True)),
  so every replay reuses the same addresses from the graph's memory pool;
- losses are accumulated on the GPU and read once after replay.

Supported: feedforward actor and critic (MLPModel), adaptive or fixed
learning rate, Adam or AdamW, one GPU. Not yet: recurrent policies, RND,
symmetry augmentation, multi-GPU.
"""

from contextlib import contextmanager
import types

import torch
from tensordict import TensorDict
from torch import nn


def capture_safe_sampling(model):
    """Sample Gaussian actions as mean + std * noise.

    torch.normal(mean, std) checks std >= 0 on the host, which a graph cannot
    do. The distribution is the same; the random sequence differs from stock.
    """
    distribution = getattr(model, "distribution", None)
    if distribution is None or not hasattr(distribution, "_distribution"):
        return

    def sample(self):
        normal = self._distribution
        return normal.loc + normal.scale * torch.randn_like(normal.loc)
    distribution.sample = types.MethodType(sample, distribution)


class CapturedUpdate:
    def __init__(self, alg, last_observations):
        self.alg = alg
        if alg.actor.is_recurrent or alg.critic.is_recurrent:
            raise ValueError("recurrent policies are not supported yet")
        if alg.rnd or alg.symmetry or alg.is_multi_gpu:
            raise ValueError("RND, symmetry and multi-GPU are not supported yet")
        device = alg.device
        capture_safe_sampling(alg.actor)
        self.storage = alg.storage
        # Observations after the last rollout step, for bootstrapping the returns.
        self.last_observations = last_observations
        self.lr = torch.tensor(float(alg.learning_rate), device=device)
        self._capturable_optimizer()
        steps, worlds = self.storage.num_transitions_per_env, self.storage.num_envs
        self.batch = steps * worlds
        self.minibatch = self.batch // alg.num_mini_batches
        self.indices = torch.arange(alg.num_mini_batches * self.minibatch, device=device)
        self.losses = torch.zeros(3, device=device)  # value, surrogate, entropy
        # The stock code replaces st.advantages each iteration; keep one buffer instead.
        self.storage.advantages = torch.zeros_like(self.storage.returns)
        self.graph = None

    def _capturable_optimizer(self):
        old = self.alg.optimizer
        group = old.param_groups[0]
        options = {k: group[k] for k in ("betas", "eps", "weight_decay", "amsgrad") if k in group}
        params = [p for g in old.param_groups for p in g["params"]]
        self.alg.optimizer = type(old)(params, lr=self.lr, capturable=True, foreach=True, **options)
        if old.state:
            # Carry Adam moments and step counts over (step counts become GPU tensors).
            state = old.state_dict()
            for group in state["param_groups"]:
                # Saved groups would otherwise restore capturable=False and a float lr.
                group.update(lr=self.lr, capturable=True, foreach=True)
            self.alg.optimizer.load_state_dict(state)
            for value in self.alg.optimizer.state.values():
                value["step"] = value["step"].to(self.lr.device, torch.float32)

    def _returns(self):
        alg, st = self.alg, self.storage
        last_values = alg.critic(self.last_observations).detach()
        advantage = torch.zeros_like(last_values)
        for step in reversed(range(st.num_transitions_per_env)):
            next_values = last_values if step == st.num_transitions_per_env - 1 else st.values[step + 1]
            not_terminal = 1.0 - st.dones[step].float()
            delta = st.rewards[step] + not_terminal * alg.gamma * next_values - st.values[step]
            advantage = delta + not_terminal * alg.gamma * alg.lam * advantage
            st.returns[step].copy_(advantage + st.values[step])
        st.advantages.copy_(st.returns - st.values)
        if not alg.normalize_advantage_per_mini_batch:
            st.advantages.copy_((st.advantages - st.advantages.mean()) / (st.advantages.std() + 1e-8))

    def _adapt_learning_rate(self, kl):
        alg = self.alg
        lower = torch.clamp(self.lr / 1.5, min=1e-5)
        higher = torch.clamp(self.lr * 1.5, max=1e-2)
        lr = torch.where(kl > alg.desired_kl * 2.0, lower,
                         torch.where((kl < alg.desired_kl / 2.0) & (kl > 0.0), higher, self.lr))
        self.lr.copy_(lr)

    def _update(self):
        alg, st = self.alg, self.storage
        self.losses.zero_()
        self._returns()
        self.indices.copy_(torch.randperm(self.indices.numel(), device=self.indices.device))
        observations = st.observations.flatten(0, 1)
        actions, values = st.actions.flatten(0, 1), st.values.flatten(0, 1)
        returns, advantages = st.returns.flatten(0, 1), st.advantages.flatten(0, 1)
        old_log_prob = st.actions_log_prob.flatten(0, 1)
        old_params = tuple(p.flatten(0, 1) for p in st.distribution_params)
        for _ in range(alg.num_learning_epochs):
            for i in range(alg.num_mini_batches):
                idx = self.indices[i * self.minibatch:(i + 1) * self.minibatch]
                batch_advantages = advantages[idx]
                if alg.normalize_advantage_per_mini_batch:
                    batch_advantages = (batch_advantages - batch_advantages.mean()) / (batch_advantages.std() + 1e-8)
                # TensorDict's own tensor indexing checks on the host; index each tensor instead.
                batch_obs = TensorDict({k: v[idx] for k, v in observations.items()}, batch_size=[self.minibatch])
                alg.actor(batch_obs, stochastic_output=True)
                log_prob = alg.actor.get_output_log_prob(actions[idx])
                new_values = alg.critic(batch_obs)
                params = tuple(alg.actor.output_distribution_params)
                entropy = alg.actor.output_entropy
                if alg.desired_kl is not None and alg.schedule == "adaptive":
                    with torch.no_grad():
                        kl = alg.actor.get_kl_divergence(tuple(p[idx] for p in old_params), params).mean()
                        self._adapt_learning_rate(kl)
                ratio = torch.exp(log_prob - torch.squeeze(old_log_prob[idx]))
                surrogate = -torch.squeeze(batch_advantages) * ratio
                clipped = -torch.squeeze(batch_advantages) * torch.clamp(ratio, 1.0 - alg.clip_param, 1.0 + alg.clip_param)
                surrogate_loss = torch.max(surrogate, clipped).mean()
                old_values = values[idx]
                if alg.use_clipped_value_loss:
                    value_clipped = old_values + (new_values - old_values).clamp(-alg.clip_param, alg.clip_param)
                    value_loss = torch.max((new_values - returns[idx]).pow(2),
                                           (value_clipped - returns[idx]).pow(2)).mean()
                else:
                    value_loss = (returns[idx] - new_values).pow(2).mean()
                loss = surrogate_loss + alg.value_loss_coef * value_loss - alg.entropy_coef * entropy.mean()
                alg.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(alg.actor.parameters(), alg.max_grad_norm, foreach=True)
                nn.utils.clip_grad_norm_(alg.critic.parameters(), alg.max_grad_norm, foreach=True)
                alg.optimizer.step()
                with torch.no_grad():
                    self.losses += torch.stack((value_loss, surrogate_loss, entropy.mean()))

    @contextmanager
    def _preserved(self):
        """Run warmup updates without changing the weights, optimizer, or learning rate."""
        alg = self.alg
        saved = ({k: v.clone() for k, v in alg.actor.state_dict().items()},
                 {k: v.clone() for k, v in alg.critic.state_dict().items()},
                 {p: {k: v.clone() if torch.is_tensor(v) else v for k, v in s.items()}
                  for p, s in alg.optimizer.state.items()},
                 self.lr.clone(), torch.cuda.get_rng_state())
        yield
        actor, critic, optimizer, lr, rng = saved
        with torch.no_grad():
            for name, value in alg.actor.state_dict().items():
                value.copy_(actor[name])
            for name, value in alg.critic.state_dict().items():
                value.copy_(critic[name])
            for p, state in alg.optimizer.state.items():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        # State created during warmup goes back to fresh: zero moments, step 0.
                        v.copy_(optimizer[p][k]) if p in optimizer else v.zero_()
            self.lr.copy_(lr)
        torch.cuda.set_rng_state(rng)

    def _release_autograd_graphs(self):
        # The actor caches its last output distribution, which keeps that forward's
        # autograd graph alive. Its gradient-accumulation nodes stay bound to the
        # stream they ran on, and capture cannot depend on another stream.
        for model in (self.alg.actor, self.alg.critic):
            distribution = getattr(model, "distribution", None)
            if distribution is not None and hasattr(distribution, "_distribution"):
                distribution._distribution = None

    def capture(self):
        self._release_autograd_graphs()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with self._preserved(), torch.cuda.stream(stream):
            for _ in range(2):
                self._update()  # Allocates gradients and optimizer state outside the graph.
        torch.cuda.current_stream().wait_stream(stream)
        # Gradients must be allocated by the graph itself, not left over from warmup.
        self.alg.optimizer.zero_grad(set_to_none=True)
        self._release_autograd_graphs()
        self.graph = torch.cuda.CUDAGraph()
        with self._preserved(), torch.cuda.graph(self.graph):
            self._update()

    def __call__(self):
        """Run one PPO update; returns the same loss dictionary as rsl_rl's PPO.update."""
        if self.graph is None:
            self.capture()
        self.graph.replay()
        value, surrogate, entropy = (self.losses / (self.alg.num_learning_epochs * self.alg.num_mini_batches)).tolist()
        self.alg.learning_rate = self.lr.item()
        self.alg.storage.clear()
        return {"value": value, "surrogate": surrogate, "entropy": entropy}
