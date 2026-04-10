"""PPO training configuration for Rydberg Bell state preparation."""

PPO_CONFIG = {
    "learning_rate": 3e-4,
    "n_steps": 4096,
    "batch_size": 256,
    "n_epochs": 10,
    "gamma": 1.0,
    "clip_range": 0.2,
    "ent_coef": 0.005,
    "total_timesteps": 1_000_000,
    "n_seeds": 3,
    "scenario": "B",
    "env_n_steps": 30,
    "reward_shaping_alpha": 0.1,
    "policy_kwargs": {"net_arch": [256, 256]},
}
