# Getting More Juice🍹Out of Your Data: Hard Pair Refinement Enhances Visual-Language Models Without Extra Data
**NAACL 2025 Main Conference Accepted! 🎉**

This repository provides code and instructions to enhance visual-language models (e.g., CLIP) by leveraging hard pair mining. Follow the steps below to download datasets, generate hard pairs, and integrate HELIP into your training pipeline.

---

## Step 1: Downloading and Organizing the Dataset

### Example: Conceptual Captions 3M (CC3M)

1. **Download the Dataset:**
   - Visit the [Conceptual Captions Download Page](https://ai.google.com/research/ConceptualCaptions/download).
   - Download the 500MB `.tsv` file.

2. **Prepare the Dataset File:**
   - Add column names to the `.tsv` file by running:
     ```bash
     sed -i '1s/^/caption\turl\n/' cc3m.tsv
     ```

3. **Download Images Using `img2dataset`:**
   - Create a directory for the CC3M dataset and navigate into it:
     ```bash
     mkdir /YOUR_DOWNLOAD_PATH/cc3m
     cd /YOUR_DOWNLOAD_PATH/cc3m
     ```
   - Use `img2dataset` to download images using the URLs and captions in the `.tsv` file:
     ```bash
     img2dataset --url_list cc3m.tsv --input_format "tsv" \
       --url_col "url" --caption_col "caption" --output_format webdataset \
       --output_folder cc3m --processes_count 16 --thread_count 64 --image_size 256 \
       --enable_wandb True
     ```
   - **Note:** Replace `/YOUR_DOWNLOAD_PATH` with your desired download location.

_For instructions on downloading other datasets, please refer to the [img2dataset repository](https://github.com/rom1504/img2dataset/tree/main/dataset_examples)._

---

## Step 2: Cooking Hard Pairs

Generate hard pairs for the dataset by running the `hard_pair_mining.py` script:

```bash
OMP_NUM_THREADS=64 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python hard_pair_mining.py --dataset cc3m --save_path /YOUR_SAVE_PATH --topk YOUR_K
```

- **Outputs:**
  - **URL Index Mapping:** Saved in `/YOUR_SAVE_PATH/url_index`
  - **Hard Pairs CSV:** Saved as `/YOUR_SAVE_PATH/cc3m_hard_sample.csv`
  - **Hard Pair Dictionary:** Saved as `/YOUR_SAVE_PATH/cc3m_hard_sample_dict.json`

_In `cc3m_hard_sample.csv`, each line starts with the target pair's index followed by the indices of the top `YOUR_K` hard pairs._

**Remember:** Replace `/YOUR_SAVE_PATH` with your desired storage path and `YOUR_K` with the number of hard pairs you wish to generate.

---

## Step 3: Boosting Existing CLIP with HELIP

Integrate HELIP into your existing CLIP training pipeline. We provide an example using [OpenCLIP](https://github.com/mlfoundations/open_clip).

### Integration Options:

- **Option 1:** Clone the OpenCLIP repository and replace `src/training/data.py` and `src/training/params.py` with our modified versions.
- **Option 2:** Directly use the adapted code available in the provided `src` folder.

### Training with Hard Samples

1. **Convert WebDataset to CSV:**  
   Use the `wbs_to_csv.py` script to convert your webdataset into a CSV format.

2. **Launch Training:**
   ```bash
   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 -m training.main \
   --train-data '/YOUR_DOWNLOAD_PATH/cc3m/cc3m.csv' \
   --train-num-samples 3000000 \
   --dataset-type csv \
   --batch-size 420 \
   --precision amp \
   --workers 4 \
   --csv-img-key filepath \
   --csv-separator , \
   --imagenet-val your/path/to/imagenet/imagenet/val \
   --use-hard \
   --hard-dict-dir '/YOUR_SAVE_PATH/cc3m_hard_sample.csv' \
   --zeroshot-frequency 1 \
   --pretrained YOUR_PRETRAINED_MODEL
   ```

**Make sure to update the placeholders** `/YOUR_DOWNLOAD_PATH`, `/YOUR_SAVE_PATH`, `YOUR_K`, and `YOUR_PRETRAINED_MODEL` with actual paths and parameters specific to your setup.

