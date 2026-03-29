import pandas as pd
from sklearn.model_selection import train_test_split
from huggingface_hub import hf_hub_download
import os

repo_id = "basavarajat/tourism-package-prediction"

# Download dataset from Hugging Face
file_path = hf_hub_download(
    repo_id=repo_id,
    filename="tourism.csv",
    repo_type="dataset"
)

# Load dataset
df = pd.read_csv(file_path)

print("Dataset loaded from Hugging Face!")
print(df.shape)
