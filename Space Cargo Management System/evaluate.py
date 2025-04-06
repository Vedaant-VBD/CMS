import pickle
import numpy as np
from train import env, discretize_state  


with open("q_table.pkl", "rb") as f:
    Q_table = pickle.load(f)

# Run RL Model Evaluation
num_eval_episodes = 100
total_rewards = []

for episode in range(num_eval_episodes):
    state = env.reset()
    state_key = discretize_state(state)
    done = False
    episode_reward = 0

    while not done:
        action = np.argmax(Q_table[state_key]) if state_key in Q_table else env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        next_state_key = discretize_state(next_state)
        episode_reward += reward
        state = next_state
        state_key = next_state_key

    total_rewards.append(episode_reward)
    print(f"Test Episode {episode}: Total Reward = {episode_reward}")

# Display Evaluation Metrics
average_reward = np.mean(total_rewards)
success_rate = np.sum(np.array(total_rewards) > 0) / num_eval_episodes

print("\nEvaluation Results:")
print(f"✅ Average Reward: {average_reward:.2f}")
print(f"✅ Success Rate: {success_rate * 100:.2f}%")
