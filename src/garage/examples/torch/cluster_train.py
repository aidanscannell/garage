#!/usr/bin/env python3
"""Example script to run RL2 in HalfCheetah."""
# pylint: disable=no-value-for-parameter
from dataclasses import dataclass, field
from typing import Any, List, Optional

import hydra
import wandb

from garage import wrap_experiment


@dataclass
class MainConfig:
    # _target_: str = "__main__.main"
    _target_: str = "cluster_train.main"
    env_name: str = "HalfCheetah"  # HopperV2/HalfCheetahV2/Walker2DV2/HalfCheetahVelEnv
    seed: int = 1

    num_epochs: int = 500
    num_train_tasks: int = 100
    num_test_tasks: int = 100
    encoder_hidden_size: int = 200
    net_size: int = 300
    num_steps_per_epoch: int = 2000
    num_initial_steps: int = 2000
    num_steps_prior: int = 400
    num_extra_rl_steps_posterior: int = 600
    batch_size: int = 256
    embedding_batch_size: int = 100
    embedding_mini_batch_size: int = 100
    max_episode_length: int = 1000


@dataclass
class TrainConfig:
    wandb_run_name: str
    main_config: MainConfig

    defaults: List[Any] = field(
        default_factory=lambda: [{"main_config": "half_cheetah_config"}]
    )

    _target_: str = "__main__.main"
    use_wandb: bool = True
    wandb_project_name: str = "adaptive-context-rl"
    wandb_group: str = "PEARL"
    wandb_tags: List[str] = field(default_factory=lambda: ["TrMRL"])


from hydra.core.config_store import ConfigStore

cs = ConfigStore.instance()
cs.store(group="main_config", name="half_cheetah_config", node=MainConfig)
cs.store(name="train_config", node=TrainConfig)


@wrap_experiment
def main(
    ctxt=None,
    env_name="HalfCheetahV2",
    seed=1,
    num_epochs=500,
    num_train_tasks=25,
    num_test_tasks=25,
    latent_size=5,
    encoder_hidden_size=200,
    net_size=300,
    meta_batch_size=16,
    num_steps_per_epoch=2000,
    num_initial_steps=2000,
    num_tasks_sample=5,
    num_steps_prior=400,
    num_extra_rl_steps_posterior=600,
    batch_size=256,
    embedding_batch_size=100,
    embedding_mini_batch_size=100,
    max_episode_length=1000,
    reward_scale=5.0,
    use_gpu=True,
):
    """Train PEARL with HalfCheetahVel environment.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by Trainer to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.
        num_epochs (int): Number of training epochs.
        num_train_tasks (int): Number of tasks for training.
        num_test_tasks (int): Number of tasks to use for testing.
        latent_size (int): Size of latent context vector.
        encoder_hidden_size (int): Output dimension of dense layer of the
            context encoder.
        net_size (int): Output dimension of a dense layer of Q-function and
            value function.
        meta_batch_size (int): Meta batch size.
        num_steps_per_epoch (int): Number of iterations per epoch.
        num_initial_steps (int): Number of transitions obtained per task before
            training.
        num_tasks_sample (int): Number of random tasks to obtain data for each
            iteration.
        num_steps_prior (int): Number of transitions to obtain per task with
            z ~ prior.
        num_extra_rl_steps_posterior (int): Number of additional transitions
            to obtain per task with z ~ posterior that are only used to train
            the policy and NOT the encoder.
        batch_size (int): Number of transitions in RL batch.
        embedding_batch_size (int): Number of transitions in context batch.
        embedding_mini_batch_size (int): Number of transitions in mini context
            batch; should be same as embedding_batch_size for non-recurrent
            encoder.
        max_episode_length (int): Maximum episode length.
        reward_scale (int): Reward scale.
        use_gpu (bool): Whether or not to use GPU for training.

    """
    import gym
    import d4rl
    import numpy as np

    from garage.envs import GymEnv, normalize
    from garage.envs.mujoco import HalfCheetahVelEnv
    from garage.experiment.deterministic import set_seed
    from garage.experiment.task_sampler import SetTaskSampler
    from garage.sampler import LocalSampler
    from garage.torch.algos import PEARL
    from garage.torch.algos.pearl import PEARLWorker
    from garage.torch.embeddings import MLPEncoder
    from garage.torch.policies import ContextConditionedPolicy, TanhGaussianMLPPolicy
    from garage.torch.q_functions import ContinuousMLPQFunction
    from garage.torch import set_gpu_mode
    from garage.trainer import Trainer

    set_seed(seed)

    class MassDampingENV(gym.Env):
        def __init__(self, env, task_idx: int = 0):
            self._env = env
            self.action_space = env.action_space
            self.observation_space = env.observation_space
            self.mass_ratios = (0.75, 0.85, 1, 1.15, 1.25)
            self.damping_ratios = (0.75, 0.85, 1, 1.15, 1.25)
            self.original_body_mass = env.env.wrapped_env.model.body_mass.copy()
            self.original_damping = env.env.wrapped_env.model.dof_damping.copy()

            self.num_tasks = 25
            self.task_idxs = np.arange(self.num_tasks)
            self.task_idx = task_idx
            self._reset(ind=self.task_idx)

        # ind is from 0 to 24
        def reset(self, sample_task: bool = False):
            if sample_task:
                self.task_idx = np.random.choice(self.task_idxs, 1)

            return self._reset(ind=self.task_idx)

        def _reset(self, ind: int):
            if isinstance(ind, np.ndarray):
                ind = ind.item()
            model = self._env.env.wrapped_env.model
            n_link = model.body_mass.shape[0]
            ind_mass = ind // 5
            ind_damp = ind % 5
            for i in range(n_link):
                model.body_mass[i] = (
                    self.original_body_mass[i] * self.mass_ratios[ind_mass]
                )
                model.dof_damping[i] = (
                    self.original_damping[i] * self.damping_ratios[ind_damp]
                )
            return self._env.reset()

        def step(self, action):
            return self._env.step(action)

        def get_normalized_score(self, score):
            return self._env.get_normalized_score(score)

        def sample_tasks(self, num_tasks):
            """Sample a list of `num_tasks` tasks.

            Args:
                num_tasks (int): Number of tasks to sample.

            Returns:
                list[dict[str, float]]: A list of "tasks," where each task is a
                    dictionary containing a single key, "direction", mapping to -1
                    or 1.

            """
            tasks = np.random.choice(self.task_idxs, num_tasks)
            return tasks

        def set_task(self, task):
            """Reset with a task.

            Args:
                task (dict[str, float]): A task (a dictionary containing a single
                    key, "direction", mapping to -1 or 1).

            """
            self.task_idx = task
            self._reset(ind=task)

    class HopperV2(MassDampingENV):
        def __init__(self, task=0):
            env = gym.make("hopper-medium-v2")
            super().__init__(env=env, task_idx=task)

    class HalfCheetahV2(MassDampingENV):
        def __init__(self, task=0):
            env = gym.make("halfcheetah-medium-v2")
            super().__init__(env=env, task_idx=task)

    class Walker2DV2(MassDampingENV):
        def __init__(self, task=0):
            env = gym.make("walker2d-medium-v2")
            super().__init__(env=env, task_idx=task)

    if env_name in "HopperV2":
        env_class = HopperV2
    elif env_name in "HalfCheetahV2":
        env_class = HalfCheetahV2
    elif env_name in "Walker2DV2":
        env_class = Walker2DV2
    elif env_name in "HalfCheetahVelEnv":
        env_class = HalfCheetahVelEnv
    else:
        raise NotImplementedError("Only HopperV2/HalfCheetahV2/Walker2DV2 accepted")
    print(f"Using env {env_class}")

    encoder_hidden_sizes = (
        encoder_hidden_size,
        encoder_hidden_size,
        encoder_hidden_size,
    )
    # create multi-task environment and sample tasks
    env_sampler = SetTaskSampler(
        # env_class,
        HalfCheetahVelEnv,
        wrapper=lambda env, _: normalize(
            GymEnv(env, max_episode_length=max_episode_length)
        ),
    )
    env = env_sampler.sample(num_train_tasks)
    test_env_sampler = SetTaskSampler(
        HalfCheetahVelEnv,
        wrapper=lambda env, _: normalize(
            GymEnv(env, max_episode_length=max_episode_length)
        ),
    )

    trainer = Trainer(ctxt)

    # instantiate networks
    augmented_env = PEARL.augment_env_spec(env[0](), latent_size)
    qf = ContinuousMLPQFunction(
        env_spec=augmented_env, hidden_sizes=[net_size, net_size, net_size]
    )

    vf_env = PEARL.get_env_spec(env[0](), latent_size, "vf")
    vf = ContinuousMLPQFunction(
        env_spec=vf_env, hidden_sizes=[net_size, net_size, net_size]
    )

    inner_policy = TanhGaussianMLPPolicy(
        env_spec=augmented_env, hidden_sizes=[net_size, net_size, net_size]
    )

    sampler = LocalSampler(
        agents=None,
        envs=env[0](),
        max_episode_length=env[0]().spec.max_episode_length,
        n_workers=1,
        worker_class=PEARLWorker,
    )

    pearl = PEARL(
        env=env,
        policy_class=ContextConditionedPolicy,
        encoder_class=MLPEncoder,
        inner_policy=inner_policy,
        qf=qf,
        vf=vf,
        sampler=sampler,
        num_train_tasks=num_train_tasks,
        num_test_tasks=num_test_tasks,
        latent_dim=latent_size,
        encoder_hidden_sizes=encoder_hidden_sizes,
        test_env_sampler=test_env_sampler,
        meta_batch_size=meta_batch_size,
        num_steps_per_epoch=num_steps_per_epoch,
        num_initial_steps=num_initial_steps,
        num_tasks_sample=num_tasks_sample,
        num_steps_prior=num_steps_prior,
        num_extra_rl_steps_posterior=num_extra_rl_steps_posterior,
        batch_size=batch_size,
        embedding_batch_size=embedding_batch_size,
        embedding_mini_batch_size=embedding_mini_batch_size,
        reward_scale=reward_scale,
    )

    set_gpu_mode(use_gpu, gpu_id=0)
    if use_gpu:
        pearl.to()

    trainer.setup(algo=pearl, env=env[0]())

    trainer.train(n_epochs=num_epochs, batch_size=batch_size)


@hydra.main(version_base="1.3", config_path="./cfgs", config_name="half_cheetah")
def hydra_wrapper(cfg: TrainConfig):
    import pprint

    from hydra.utils import get_original_cwd
    import omegaconf

    cfg_dict = omegaconf.OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )
    pprint.pprint(cfg_dict)

    if cfg.use_wandb:  # Initialise WandB
        import wandb

        run = wandb.init(
            project=cfg.wandb_project_name,
            group=cfg.wandb_group,
            tags=cfg.wandb_tags,
            config=cfg_dict,
            name=cfg.wandb_run_name,
            # monitor_gym=cfg.monitor_gym,
            save_code=True,
            dir=get_original_cwd(),  # don't nest wandb inside hydra dir
        )

    return hydra.utils.call(cfg.main_config)


if __name__ == "__main__":
    hydra_wrapper()
    # transformer_ppo_halfcheetah()
