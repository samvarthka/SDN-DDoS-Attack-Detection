from train_model import create_demo_dataset, DATA_PATH
import os

os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
create_demo_dataset().to_csv(DATA_PATH, index=False)
print(f"Demo dataset saved to: {DATA_PATH}")
