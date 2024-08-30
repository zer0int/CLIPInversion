### Changes 30/AUG/2024

- Added Gradient Ascent (GA): Uses an input image instead of a text prompt.
- Optimizes text embeddings for cosine similarity with image embeddings
- Prints CLIP's 'opinion' about image to console
- Uses text embeddings for inversion image generation
- ⚠️ Same as without GA, innocent images (prompts) can lead to nefarious and NSFW inversions.
- Refer to the paper by the original authors for details (see below).
- ✅ Usage example (only use this code for `--use_image`; use `invert.py` for a text `--prompt`):
```bash
python invert-ga.py --num_iters 3400 --use_image "in/catshoe.jpg" --img_size 64 --tv 0.0005 --batch_size 13 --bri 0.4 --con 0.4 --sat 0.4 --save_every 10 --print_every 10 --model_name ViT-L/14
```
- ✅ Added support for `ViT-L/14@336` (to all code), usage example:
```bash
python invert.py --num_iters 3400 --prompt "an ai robot" --img_size 64 --tv 0.005 --batch_size 13 --bri 0.4 --con 0.4 --sat 0.4 --save_every 10 --print_every 10 --model_name ViT-L/14@336px
```
---
- GA + Inversion examples (generated with [my improved ViT-L/14 fine-tune](https://huggingface.co/zer0int/CLIP-GmP-ViT-L-14/tree/main)):

![gradient-ascent-final-example](https://github.com/user-attachments/assets/a9443a7d-a002-4f89-992a-ef9b3f3ec01a)
-----
Original README.MD by the authors:
-----

[//]: # (# CLIPInversion)
# What do we learn from inverting CLIP models?
**<span style="color:white">Warning</span>: This paper contains sexually explicit images and
language, offensive visuals and terminology, discussions on
pornography, gender bias, and other potentially unsettling,
distressing, and/or offensive content for certain readers.**

[Paper](https://arxiv.org/abs/2403.02580)

![Inverted Images](figures/main.png)

**Installing requirements:**


```bash
pip install requirements.txt
```
**How to run:**


```bash
python invert.py \
    --num_iters 3400 \  # Number of iterations during the inversion process.
    --prompt "The map of the African continent" \  # The text prompt to invert.
    --img_size 64 \  # Size of the image at iteration 0.
    --tv 0.005 \  # Total Variation weight.
    --batch_size 13 \  # How many augmentations to use at each iteration.
    --bri 0.4 \  # ColorJitter Augmentation brightness degree.
    --con 0.4 \  # ColorJitter Augmentation contrast degree.
    --sat 0.4 \  # ColorJitter Augmentation saturation degree.
    --save_every 100 \  # Frequency at which to save intermediate results.
    --print_every 100 \  # Frequency at which to print intermediate information.
    --model_name ViT-B/16 # ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'ViT-B/32', 'ViT-B/16']
```
