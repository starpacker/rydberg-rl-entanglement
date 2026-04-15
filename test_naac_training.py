"""Quick test of NAAC training (small scale for debugging)."""

import sys
sys.path.insert(0, '.')

from train_naac import train_naac

if __name__ == "__main__":
    print("Running quick NAAC training test...")

    stats = train_naac(
        scenario="C",
        n_steps=30,  # Shorter for testing
        k_calib=5,
        n_envs=8,
        n_episodes=1000,  # Just 1000 episodes
        lr=3e-4,
        lambda_estimator=1.0,
        noise_scale_range=(1.0, 3.0),  # Narrower range
        save_dir="models/naac_test",
        log_dir="logs/naac_test",
        device="cuda",
        seed=42,
    )

    print("\nTest training completed!")
    print(f"Final mean fidelity: {stats['mean_fidelity'][-1]:.4f}")
