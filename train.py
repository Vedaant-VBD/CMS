import pandas as pd
import numpy as np
import gym
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import random

#Load & Preprocess Datasets
items_df = pd.read_csv("generated_items.csv")
logs_df = pd.read_csv("generated_logs.csv")
containers_df = pd.read_csv("generated_containers.csv")

# Convert Expiry Date to Days Remaining
items_df["Days to Expiry"] = (pd.to_datetime(items_df["Expiry Date"]) - pd.Timestamp.today()).dt.days
items_df["Days to Expiry"] = items_df["Days to Expiry"].fillna(9999)

if "Item ID" in logs_df.columns:
    # Count how many times each item has been used
    item_usage_counts = logs_df["Item ID"].value_counts().reset_index()
    item_usage_counts.columns = ["Item ID", "Usage Count"]
    
    # Merge this information back to the items dataframe
    items_df = items_df.merge(item_usage_counts, on="Item ID", how="left")
    items_df["Usage Count"] = items_df["Usage Count"].fillna(0)
else:
    # Simulate usage data (random values between 0 and the usage limit)
    items_df["Usage Count"] = items_df["Usage Limit"].apply(lambda x: random.randint(0, int(x*1.2)))

# Define waste based on both conditions
items_df["Is Waste"] = (items_df["Days to Expiry"] <= 0) | (items_df["Usage Count"] >= items_df["Usage Limit"])

# Print waste statistics to verify we have both classes
waste_count = items_df["Is Waste"].sum()
total_count = len(items_df)
print(f"Waste items: {waste_count} ({waste_count/total_count:.2%})")
print(f"Non-waste items: {total_count - waste_count} ({(total_count - waste_count)/total_count:.2%})")


# Select Features & Labels
features = ["Width (cm)", "Depth (cm)", "Height (cm)", "Mass (kg)", "Priority (1-100)", "Days to Expiry"]
labels = items_df["Is Waste"]

print(items_df.columns)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(items_df[features], labels, test_size=0.2, random_state=42)

# Standardize Data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 2: Train Decision Tree & Random Forest
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train_scaled, y_train)
dt_preds = dt_model.predict(X_test_scaled)
print("\nDecision Tree Accuracy:", accuracy_score(y_test, dt_preds))

rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(X_train_scaled, y_train)
rf_preds = rf_model.predict(X_test_scaled)
print("\nRandom Forest Accuracy:", accuracy_score(y_test, rf_preds))

# Step 3: Train Logistic Regression
log_model = LogisticRegression()
log_model.fit(X_train_scaled, y_train)
log_preds = log_model.predict(X_test_scaled)
print("\nLogistic Regression Accuracy:", accuracy_score(y_test, log_preds))


# A more practical RL approach for cargo retrieval
class SimplifiedCargoRetrievalEnv(gym.Env):
    def __init__(self, items_df):
        super(SimplifiedCargoRetrievalEnv, self).__init__()
        self.items_df = items_df.copy()
        # We'll use top 20 items for simplicity
        self.num_items = min(20, len(items_df))
        self.action_space = gym.spaces.Discrete(self.num_items)
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(self.num_items, 2), dtype=np.float32
        )  # Track [retrieved_status, priority_normalized]
        self.reset()
    
    def reset(self):
        # Select a subset of items for this episode
        self.current_items = self.items_df.sample(self.num_items).reset_index(drop=True)
        
        # Initialize state: [retrieved_status, normalized_priority]
        self.state = np.zeros((self.num_items, 2))
        self.state[:, 1] = self.current_items["Priority (1-100)"].values / 100.0
        
        return self.state.flatten()
    
    def step(self, action):
        reward = 0
        done = False
        
        # If item already retrieved, negative reward
        if self.state[action, 0] == 1:
            reward = -1
        else:
            # Mark as retrieved
            self.state[action, 0] = 1
            
            # Reward based on priority
            priority = self.current_items.iloc[action]["Priority (1-100)"]
            reward = priority / 10.0  # Scale reward
            
            # Additional reward for retrieving waste items that should be removed
            if self.current_items.iloc[action]["Is Waste"]:
                reward += 5
        
        # Episode ends when all items are retrieved
        if np.all(self.state[:, 0] == 1):
            done = True
            
        return self.state.flatten(), reward, done, {}

# Initialize the simplified environment
env = SimplifiedCargoRetrievalEnv(items_df)

# Create a smaller Q-table that can fit in memory
state_dim = env.observation_space.shape[0]  # flattened state
action_dim = env.action_space.n

# We'll use a discretized state space
num_bins = 10  # discretize each dimension into 10 bins
max_states = num_bins ** 2  # conservative estimate for 2D state space per item

# Use a dictionary for Q-table instead of a large array
Q_table = {}

# Train RL Agent
learning_rate = 0.1
discount_factor = 0.9
epsilon = 0.4  # Initial epsilon value


def discretize_state(state):
    # Convert continuous state to discrete bins for lookup
    state_key = tuple((state * num_bins).astype(int).tolist())
    return state_key

reward_list = []  # Initialize an empty list to store rewards for each episode

for episode in range(1000):
    learning_rate = max(0.01, learning_rate * 0.99) # Decay learning rate over episodes
    epsilon = max(0.1, epsilon * 1.01)  # Decay epsilon over episodes
    state = env.reset()
    state_key = discretize_state(state)
    done = False
    total_reward = 0
    
    while not done:
        # Epsilon-greedy action selection
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            # Get Q-values for this state, defaulting to 0 if not seen before
            if state_key not in Q_table:
                Q_table[state_key] = np.zeros(action_dim)
            action = np.argmax(Q_table[state_key])
        
        # Take action
        next_state, reward, done, _ = env.step(action)
        next_state_key = discretize_state(next_state)
        total_reward += reward
        
        # Update Q-table
        if state_key not in Q_table:
            Q_table[state_key] = np.zeros(action_dim)
        if next_state_key not in Q_table:
            Q_table[next_state_key] = np.zeros(action_dim)
            
        Q_table[state_key][action] = (1 - learning_rate) * Q_table[state_key][action] + \
                                    learning_rate * (reward + discount_factor * np.max(Q_table[next_state_key]))
        
        state = next_state
        state_key = next_state_key

    reward_list.append(total_reward)  # Append the total reward for this episode to the list
    
    if episode % 10 == 0:
        print(f"Episode {episode}, Total Reward: {total_reward}")

print("\nReinforcement Learning Model Training Complete!")

# import matplotlib.pyplot as plt
# plt.plot(reward_list)
# plt.xlabel('Episodes')
# plt.ylabel('Total Reward')
# plt.title('Training Progress')
# plt.show()



import pickle
with open("q_table.pkl", "wb") as f:
    pickle.dump(Q_table, f)

