import numpy as np
import pandas as pd
from faker import Faker
import random
from datetime import datetime, timedelta

fake = Faker()

# Function to generate random expiry dates (some items don't expire)
def generate_expiry():
    if random.random() > 0.2:  # 80% of items have expiry
        days = random.randint(30, 730)  # Between 1 month and 2 years
        return (datetime.today() + timedelta(days=days)).strftime('%Y-%m-%d')
    return "N/A"

# Function to generate cargo items
def generate_items(num_items=1000):
    categories = ["Food", "Medical", "Oxygen", "Water", "Tools", "Equipment", "Scientific"]
    preferred_zones = ["Crew Quarters", "Airlock", "Medical Bay", "Laboratory", "Storage Module"]
    
    data = []
    
    for i in range(num_items):
        item_id = f"{i+1:04d}"
        category = random.choice(categories)
        name = f"{category} {fake.word().capitalize()}"
        width, depth, height = np.random.randint(5, 50, size=3)  # Dimensions in cm
        mass = round(np.random.uniform(1, 50), 2)  # Mass in kg
        priority = np.random.randint(1, 101)  # 1 (low) to 100 (high)
        expiry_date = generate_expiry()
        usage_limit = np.random.randint(1, 100) if expiry_date != "N/A" else 1000
        preferred_zone = random.choice(preferred_zones)

        data.append([item_id, name, width, depth, height, mass, priority, expiry_date, usage_limit, preferred_zone])
    
    columns = ["Item ID", "Name", "Width (cm)", "Depth (cm)", "Height (cm)", "Mass (kg)", 
               "Priority (1-100)", "Expiry Date", "Usage Limit", "Preferred Zone"]
    
    df = pd.DataFrame(data, columns=columns)
    return df

# Function to generate storage containers
def generate_containers(num_containers=100):
    zones = ["Crew Quarters", "Airlock", "Medical Bay", "Laboratory", "Storage Module"]
    
    data = []
    for i in range(num_containers):
        container_id = f"cont{i+1:03d}"
        zone = random.choice(zones)
        width, depth, height = np.random.randint(50, 300, size=3)
        
        data.append([container_id, zone, width, depth, height])
    
    columns = ["Container ID", "Zone", "Width (cm)", "Depth (cm)", "Height (cm)"]
    df = pd.DataFrame(data, columns=columns)
    return df

# Function to generate retrieval logs
def generate_retrieval_logs(num_logs=5000):
    astronauts = ["Alice", "Bob", "Charlie", "David", "Eve"]
    actions = ["Retrieved", "Placed", "Moved", "Waste Disposal"]
    
    data = []
    
    for i in range(num_logs):
        log_id = f"{i+1:05d}"
        astronaut = random.choice(astronauts)
        action = random.choice(actions)
        item_id = f"{random.randint(1, 1000):04d}"  # Refers to an item
        container_id = f"cont{random.randint(1, 100):03d}"
        timestamp = fake.date_time_between(start_date="-1y", end_date="now").strftime('%Y-%m-%d %H:%M:%S')
        
        data.append([log_id, astronaut, action, item_id, container_id, timestamp])
    
    columns = ["Log ID", "Astronaut", "Action", "Item ID", "Container ID", "Date-Time"]
    df = pd.DataFrame(data, columns=columns)
    return df

# Generate datasets
items_df = generate_items(1000)
containers_df = generate_containers(100)
logs_df = generate_retrieval_logs(5000)

# Save to CSV
items_df.to_csv("generated_items.csv", index=False)
containers_df.to_csv("generated_containers.csv", index=False)
logs_df.to_csv("generated_logs.csv", index=False)

print("✅ Synthetic dataset generated successfully!")
