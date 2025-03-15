# 修改batch_size为1024 还是jina->1024->0.8+0.2->PCA(128)
from transformers import AutoModel
import pandas as pd
import torch
import os
import numpy as np
from tqdm import tqdm  # Import tqdm for progress bar
from sklearn.decomposition import PCA  # Import PCA for dimensionality reduction
from sklearn.metrics.pairwise import cosine_similarity

CUDA="cuda:0"

# Load the item features dataframe
df_item_feature = pd.read_parquet("item_feature.parquet", engine="pyarrow")
print(df_item_feature)

# Initialize the CLIP model
model = AutoModel.from_pretrained('jinaai/jina-clip-v2', trust_remote_code=True)

# Get the path to the images folder
image_folder = 'item_images'

# Initialize an empty list to store image paths and their corresponding titles
image_paths = []
titles = []

# Iterate through the image files in the folder
for image_file in os.listdir(image_folder):
    if image_file.endswith('.jpg'):
        # Extract item_id from the filename (remove .jpg extension)
        item_id = int(image_file.split('.')[0])
        
        # Find the corresponding item_title in the dataframe
        item_title = df_item_feature[df_item_feature['item_id'] == item_id]['item_title'].values[0]
        
        # Add the image path and title to the lists
        image_paths.append(os.path.join(image_folder, image_file))
        titles.append(item_title)

# Ensure that you are using GPU 2 (set device to GPU 2)
device = torch.device(CUDA if torch.cuda.is_available() else "cpu")
model.to(device)

# Define batch size
batch_size = 1024

# Function to process images and texts in batches
def process_batch(image_paths_batch, titles_batch):
    image_embeddings_batch = []
    text_embeddings_batch = []
    
    # Encode images in batch
    with torch.no_grad():
        try:
            image_embeddings_batch = model.encode_image(image_paths_batch)  # Directly using the file paths
            image_embeddings_batch = torch.tensor(image_embeddings_batch).to(device)  # Convert to tensor
        except Exception as e:
            print(f"Error encoding images in batch: {e}")
        
    # Encode texts in batch
    text_embeddings_batch = model.encode_text(titles_batch)  # Directly get the numpy array
    
    return image_embeddings_batch.cpu().numpy(), text_embeddings_batch

# Initialize lists to store all embeddings
all_image_embeddings = []
all_text_embeddings = []

# Process images and texts in batches
for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing Batches"):
    image_paths_batch = image_paths[i:i + batch_size]
    titles_batch = titles[i:i + batch_size]
    
    image_embeddings_batch, text_embeddings_batch = process_batch(image_paths_batch, titles_batch)
    
    all_image_embeddings.append(image_embeddings_batch)
    all_text_embeddings.append(text_embeddings_batch)

# Concatenate all batches
all_image_embeddings = np.concatenate(all_image_embeddings, axis=0)
all_text_embeddings = np.concatenate(all_text_embeddings, axis=0)

# Average the image and text embeddings
final_embeddings = all_image_embeddings*0.8 + all_text_embeddings*0.2

#Apply PCA to reduce dimensionality to 128
pca = PCA(n_components=128)
final_embeddings_pca = pca.fit_transform(final_embeddings)

# 确保嵌入向量为float64类型（与原parquet文件中的数据类型对齐）
final_embeddings_pca = final_embeddings_pca.astype(np.float64)  # 关键修复：改为float64

# Print the shape of the final embeddings
print("Final Embeddings Shape after PCA:", final_embeddings_pca.shape)

# Example of how to compare similarity between an image and a title:
# Compute cosine similarity between image embedding and text embedding
similarities = cosine_similarity(all_image_embeddings, all_text_embeddings)

# Calculate the average similarity across all items
average_similarity = similarities.diagonal().mean()  # 计算对角线的平均值
print(f"Average similarity across all items: {average_similarity:.4f}")

# Load the item_info.parquet file
df_item_info = pd.read_parquet("data/Jina_CLIP_v8/item_info.parquet", engine="pyarrow")

# 检查原始列的数据类型（调试用）
print("Original item_emb_d128 dtype:", df_item_info['item_emb_d128'].iloc[1].dtype)  # 查看第二行的类型

# Ensure that the embeddings are aligned with the item_info dataframe
# The first row (item 0) is reserved, so we start from item 1
for idx, item_id in enumerate(range(1, len(final_embeddings_pca) + 1)):
    # 直接赋值numpy数组（确保类型为float64）
    df_item_info.at[item_id, 'item_emb_d128'] = final_embeddings_pca[idx]

# 检查更新后的列数据类型（调试用）
print("Updated item_emb_d128 dtype:", df_item_info['item_emb_d128'].iloc[1].dtype)

# Save the updated item_info.parquet file
df_item_info.to_parquet(
    "data/Jina_CLIP_v8/item_info_updated.parquet",
    engine="pyarrow",
    index=False
)