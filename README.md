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
After running this command, you can get a parquet file at `data/item_info_updated.parquet`, or you can download through **[this link](https://drive.google.com/file/d/10Skum6JAnyvFteqYUZA3ydXlG2XhDuZs/view?usp=sharing)**.

## Run CTR Prediction

Execute the following command to train our model for this CTR prediction task:

```bash
python run_param_tuner.py --config config/din_dcn_config_tuner.yaml --gpu 0
```
And you can use our checkpoints to get the prediction result.
```bash
python prediction.py --config config/din_dcn_config_tuner --expid DIN_DCN_MicroLens_1M_x1_013_1e9c0132  --gpu 0
```

## Checkpoints

- **Multimodal Embeddings:** `data/item_info_updated.parquet`
- **Final Model Checkpoint:** `checkpoints/MicroLens_1M_x1/DIN_DCN_MicroLens_1M_x1_013_1e9c0132.model`

