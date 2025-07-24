import matplotlib.pyplot as plt
import re

# Define input and output
log_file = "reward_log_2.txt"  # Replace this with your actual file path if needed
output_image = "plot_2.png"

# Lists to store extracted data
timesteps = []
rewards = []

# Read and parse the file
with open(log_file, "r") as file:
    for line in file:
        match = re.search(r"Eval num_timesteps=(\d+), episode_reward=([-\d\.]+)", line)
        if match:
            timestep = int(match.group(1))
            reward = float(match.group(2))
            timesteps.append(timestep)
            rewards.append(reward/20000)
            
def cummax(arr):
    m = -1e10
    out = []
    for item in arr:
        if item > m:
            m = item
        out.append(m)
    return out

# Plotting
plt.figure()
plt.plot(timesteps, rewards, label="reward")
plt.plot(timesteps, cummax(rewards), label="cumulative max")
plt.xlabel("Timesteps")
plt.ylabel("Mean Episode Reward (RMSE)")
plt.title("Episode Reward vs Timesteps")
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.savefig(output_image)

print(f"Plot saved to {output_image}")
