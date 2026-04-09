"""PPO training configuration for Rydberg Bell state preparation."""

PPO_CONFIG = {
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 1.0,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "total_timesteps": 50_000,
    "n_seeds": 3,
    "scenario": "B",
    "env_n_steps": 30,
}
