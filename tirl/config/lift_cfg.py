from omni.isaac.lab.utils import configclass

from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)

from tirl.config import RewardCfg, IrlAlgorithmCfg, IrlRunnerCfg

@configclass
class LiftCubePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 50
    experiment_name = "franka_lift"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.006,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.98,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class LiftCubeIrlRunnerCfg(IrlRunnerCfg):
    num_steps_per_env_rl = 24
    max_iterations = 1500
    save_interval = 50
    experiment_name = "franka_lift"
    empirical_normalization = False
    logger = "wandb"
    actor_critic = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    reward = RewardCfg(
        reward_hidden_dims=[256, 128, 64],
        activation="elu",
        reward_is_linear=False,
        reward_features=None,
        num_reward_features=None,
    )
    rl_algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.006,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.98,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
    irl_algorithm = IrlAlgorithmCfg(
        max_rollout_length=128,
        batch_size=256,
        num_learning_epochs=5,
        weight_decay=1e-6,
        max_grad_norm=1.0,
        reward_loss_coef=1.0,
    )
