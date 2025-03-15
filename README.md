# MMCTR
## Installation
```
git clone https://github.com/ZiaoGao/MMCTR.git
pip install -r requirements
```

## Multimodal Embedding Extraction

We utilize the **[Jina-CLIP-v2](https://huggingface.co/jinaai/jina-clip-v2)** model to extract multimodal embeddings from text and images.

### Usage

Execute the following command to generate embeddings:

```bash
python Jina_text_img_v8.py
```

## Run CTR Prediction

Execute the following command to run the CTR prediction task:

```bash
python run_param_tuner.py --config config/din_dcn_config_tuner.yaml --gpu 0
```

## Checkpoints

- **Multimodal Embeddings:** `data/item_info_updated.parquet`
- **Final Model Checkpoint:** `checkpoints/MicroLens_1M_x1/DIN_DCN_MicroLens_1M_x1_013_1e9c0132.model`

