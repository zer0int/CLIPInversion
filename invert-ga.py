import warnings
warnings.filterwarnings('ignore')
#warnings.simplefilter(action='ignore', category=FutureWarning)

import argparse
import os
import clip
import kornia.augmentation as kaugs
import kornia
import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
from helpers.augmentations import ColorJitter, RepeatBatch, Jitter, TotalVariation
from helpers.utils import Normalization, Scale, freeze_module
from torch.nn.utils import clip_grad_norm_
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from colorama import Fore, Style

from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

# !!!
# This code is for using CLIP's own 'opinion' (gradient ascent text embeddings) for inversion.
# NOTE: WORK IN PROGRESS. Using "--prompt" will NOT currently work. Use 'invert.py' for "--prompt".


# Argument Parsing
def parse_arguments():
    parser = argparse.ArgumentParser(description='inverting clip!')
    parser.add_argument('--num_iters', default=3400, type=int)
    parser.add_argument('--save_every', default=100, type=int)
    parser.add_argument('--print_every', default=50, type=int)
    parser.add_argument('--batch_size', default=13, type=int)
    parser.add_argument('-p', '--prompt', action='append', type=str, default=[])
    parser.add_argument('-e', '--extra_prompts', action='append', type=str, default=[])
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--tv', default=0.005, type=float)
    parser.add_argument('--jitter', action='store_true')
    parser.add_argument('--color', action='store_true')
    parser.add_argument('--img_size', default=64, type=int)
    parser.add_argument('--eps', default=2 / 255)
    parser.add_argument('--optimizer', default='adam')
    parser.add_argument('--bri', type=float, default=0.4)
    parser.add_argument('--con', type=float, default=0.4)
    parser.add_argument('--sat', type=float, default=0.4)
    parser.add_argument('--l1', type=float, default=0.)
    parser.add_argument('--trial', type=int, default=1)
    parser.add_argument('--cg_std', type=float, default=0.)
    parser.add_argument('--cg_mean', type=float, default=0.)
    parser.add_argument('--model_name', default='ViT-B/16')
    parser.add_argument('--prompt_id', type=int, default=0)
    parser.add_argument('--use_image', type=str, default=None, help="Path to image file; uses text embedding of 'CLIP opinion' instead of prompt")
    return parser.parse_args()

# CLIP Model Loader
def load_clip_model(model_name, device):
    model, preprocess = clip.load(model_name, device)
    return model.eval().float(), preprocess

# Image Loader
def load_image(img_path, sideX, sideY):
    im = torch.tensor(np.array(Image.open(img_path).convert("RGB"))).cuda().unsqueeze(0).permute(0, 3, 1, 2) / 255
    im = F.interpolate(im, (sideX, sideY))
    return im

# Augmentation Pipeline
def augment(into, augs):
    return augs(into)

# Gradient Ascent Functions
def clip_encode_text(model, text, many_tokens, prompt):
    x = torch.matmul(text, model.token_embedding.weight)
    x = x + model.positional_embedding
    x = x.permute(1, 0, 2)
    x = model.transformer(x)
    x = x.permute(1, 0, 2)
    x = model.ln_final(x)
    x = x[torch.arange(x.shape[0]), many_tokens + len(prompt) + 2] @ model.text_projection
    return x

# Entertain user by printing CLIP's 'opinion' rants about image to console
def checkin(loss, tx, lll, tok, bests, imagename):
    unique_tokens = set()

    these = [tok.decode(torch.argmax(lll, 2)[kj].clone().detach().cpu().numpy().tolist()).replace('', '').replace('', '') for kj in range(lll.shape[0])]
    
    for kj in range(lll.shape[0]):
        if loss[kj] < sorted(list(bests.keys()))[-1]:
            cleaned_text = ''.join([c if c.isprintable() else ' ' for c in these[kj]])
            bests[loss[kj]] = cleaned_text
            bests.pop(sorted(list(bests.keys()))[-1], None)
            try:
                decoded_tokens = tok.decode(torch.argmax(lll, 2)[kj].clone().detach().cpu().numpy().tolist())
                decoded_tokens = decoded_tokens.replace('<|startoftext|>', '').replace('<|endoftext|>', '')
                decoded_tokens = ''.join(c for c in decoded_tokens if c.isprintable())
                print(Fore.WHITE + f"Sample {kj} Tokens: ")
                print(Fore.BLUE + Style.BRIGHT + f"{decoded_tokens}" + Fore.RESET)
            except Exception as e:
                print(f"Error decoding tokens for sample {kj}: {e}")
                continue

    for j, k in zip(list(bests.values())[:5], list(bests.keys())[:5]):
        j = j.replace('<|startoftext|>', '')
        j = j.replace('<|endoftext|>', '')
        j = j.replace('\ufffd', '')
        j = j.replace('.', '')
        j = j.replace(';', '')
        j = j.replace('?', '')
        j = j.replace('!', '')
        j = j.replace('_', '')
        j = j.replace('-', '')
        j = j.replace('\\', '')
        j = j.replace('\'', '')
        j = j.replace('"', '')
        j = j.replace('^', '')
        j = j.replace('&', '')
        j = j.replace('#', '')
        j = j.replace(')', '')
        j = j.replace('(', '')
        j = j.replace('*', '')
        j = j.replace(',', '')
        tokens = j.split()
        unique_tokens.update(tokens)

    with open(f"TOK/tokens_{imagename}.txt", "w", encoding='utf-8') as f:
        f.write(" ".join(unique_tokens))

# Softmax
class Pars(torch.nn.Module):
    def __init__(self, batch_size, many_tokens, prompt):
        super(Pars, self).__init__()
        st = torch.zeros(batch_size, many_tokens, 49408).normal_()
        self.normu = torch.nn.Parameter(st.cuda())
        self.much_hard = 1000

        self.start = torch.zeros(batch_size, 1, 49408).cuda()
        self.start[:, :, 49406] = 1

        self.prompt = torch.zeros(batch_size, len(prompt), 49408).cuda()
        for jk, pt in enumerate(prompt):
            self.prompt[:, jk, pt] = 1 
        
        self.pad = torch.zeros(batch_size, 77 - (many_tokens + len(prompt) + 1), 49408).cuda()
        self.pad[:, :, 49407] = 1

    def forward(self):
        self.soft = F.gumbel_softmax(self.normu, tau=self.much_hard, dim=-1, hard=True)
        fin = torch.cat([self.start, self.prompt, self.soft, self.pad], 1)
        return fin

# Gradient Ascent
def ascend_txt(image, model, lats, many_tokens, prompt, nom, augment):
    iii = nom(augment(image[:,:3,:,:].expand(lats.normu.shape[0], -1, -1, -1)))
    iii = model.encode_image(iii).detach()
    lll = lats()
    tx = clip_encode_text(model, lll, many_tokens, prompt)
    return -100 * torch.cosine_similarity(tx.unsqueeze(0), iii.unsqueeze(1), -1).view(-1, lats.normu.shape[0]).T.mean(1), tx, lll

# Loop with AMP
def train(image, model, lats, many_tokens, prompt, optimizer, nom, augment):
    with autocast():
        loss1, tx, lll = ascend_txt(image, model, lats, many_tokens, prompt, nom, augment)
    loss = loss1.mean()
    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    return loss1, tx, lll

# CLIP's opinion to use for inversion
def generate_target_text_embeddings(img_path, model, lats, optimizer, training_iterations, checkin_step, many_tokens, prompt, nom, augment, tok, bests):
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    img = load_image(img_path, model.visual.input_resolution, model.visual.input_resolution)
    print(Fore.YELLOW + Style.BRIGHT + f"\nRunning gradient ascent for {img_name}...\n" + Fore.RESET)
    for j in range(training_iterations):
        loss, tx, lll = train(img, model, lats, many_tokens, prompt, optimizer, nom, augment)
        if j % checkin_step == 0:
            print(Fore.GREEN + f"Iteration {j}: Average Loss: {loss.mean().item()}" + Fore.RESET)
            checkin(loss, tx, lll, tok, bests, img_name)
   
    target_text_embedding = tx.detach()
    torch.save(target_text_embedding, f"txtembeds/{img_name}_text_embedding.pt")
    print(Fore.MAGENTA + Style.BRIGHT + "\nText embedding saved to 'txtembeds'.\nTokens (CLIP 'opinion') saved to 'TOK'.\n" + Fore.RESET)
    return img, target_text_embedding, img_path

# Inversion Functions
def get_optimizer(image, lr, optimizer_type):
    if optimizer_type == 'adam':
        optimizer = torch.optim.Adam([image], lr=lr)
    else:
        optimizer = torch.optim.LBFGS([image], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2000)
    return optimizer, scheduler

def forward(image, model, normalizer, color_jitter, text_features_map, tv_module, args, pre_aug, aug):
    image_input = pre_aug(image)
    image_input = aug(image_input)
    scale = Scale(model.visual.input_resolution)
    image_input = scale(image_input)
    image_input = color_jitter(image_input)
    image_input = normalizer(image_input)
    image_features = model.encode_image(image_input)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    l2_loss = torch.norm(image_features - text_features_map[model], dim=1)
    loss = torch.mean(l2_loss)
    return loss, l2_loss


def run_inversion(args, models, text_features_map, tv_module, normalizer, color_jitter, pre_aug, aug):
    for model in models:
        freeze_module(model)
    
    image = torch.rand((1, 3, args.img_size, args.img_size)).cuda()
    image.requires_grad_()
    
    optimizer, scheduler = get_optimizer(image, args.lr, args.optimizer)
    softmax = nn.Softmax(dim=1)
    change_scale_schedule = [900, 1800]
    print(Fore.YELLOW + Style.BRIGHT + f"Running Inversion...\n" + Fore.RESET)
    # Define save_path based on the args
    if args.use_image:
        save_path = f'images/{os.path.splitext(os.path.basename(args.use_image))[0]}/{args.trial}/{args.lr}_{args.tv}_{args.cg_std}_{args.cg_mean}'
    else:
        save_path = f'images/{args.prompt}/{args.trial}/{args.lr}_{args.tv}_{args.cg_std}_{args.cg_mean}'

    os.makedirs(save_path, exist_ok=True)
    
    for i in range(args.num_iters):
        max_grad_norm = 1.
        if i in change_scale_schedule:
            new_res = image.shape[2] * 2
            if args.jitter:
                jitter.lim = jitter.lim * 2
            if new_res >= model.visual.input_resolution:
                new_res = model.visual.input_resolution
            up_sample = Scale(new_res)
            image = up_sample(image.detach())
            image.requires_grad_(True)
            optimizer, scheduler = get_optimizer(image, args.lr, args.optimizer)
    
        def closure():
            optimizer.zero_grad()
            other_loss = tv_module(image)
            loss = args.tv * other_loss
            image_input = image
            l1_loss = torch.norm(image_input, p=1)
            loss = loss + args.l1 * l1_loss
            for model in models:
                xent_loss, scores = forward(image_input, model, normalizer, color_jitter, text_features_map, tv_module, args, pre_aug, aug)
                loss = loss + xent_loss * (1 / len(models))
            loss.backward()
            clip_grad_norm_([image], max_grad_norm)
            image.data = torch.clip(image.data, 0, 1)
            if i % args.print_every == 0:
                print(f'{i:04d}: loss is {loss:.4f}, xent: {xent_loss:.4f}, tv: {other_loss:.4f}, l1: {l1_loss:.4f}')
            if i % args.save_every == 0:
                path = os.path.join(save_path, f'{i}.png')
                torchvision.utils.save_image(image, path, normalize=True, scale_each=True)
            return loss
    
        optimizer.step(closure)
        if i >= 3400:
            scheduler.step()
    
    path = os.path.join(save_path, 'final.png')
    torchvision.utils.save_image(image, path, normalize=True, scale_each=True)
    print(Fore.MAGENTA + Style.BRIGHT + "\nInversion results saved to 'images'.\n" + Fore.RESET)



# Main loop
def main():
    args = parse_arguments()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    normalizer = Normalization([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]).cuda()
    
    model, preprocess = load_clip_model(args.model_name, device)
    models = [model]
    
    tok = clip.simple_tokenizer.SimpleTokenizer()
    bests = {1000: 'None', 1001: 'None', 1002: 'None', 1003: 'None', 1004: 'None'}
    prompt = clip.tokenize('''''').numpy().tolist()[0]
    prompt = [i for i in prompt if i != 0 and i != 49406 and i != 49407]
    
    lats = Pars(args.batch_size, 4, prompt).cuda()
    optimizer = torch.optim.Adam([{'params': [lats.normu], 'lr': 5}])
    
    augs = torch.nn.Sequential(
        kornia.augmentation.RandomAffine(degrees=10, translate=.1, p=.8).cuda(),
    ).cuda()
    
    seq = []
    if args.jitter:
        jitter = Jitter(lim=32, modeldims=model.visual.input_resolution)
        seq.append(jitter)
    seq.append(RepeatBatch(args.batch_size))
    pre_aug = nn.Sequential(*seq)
    
    if args.use_image:
        img, target_text_embedding, img_path = generate_target_text_embeddings(args.use_image, model, lats, optimizer, 300, 10, 4, prompt, normalizer, augs, tok, bests)
        text_inputs = target_text_embedding
        args.prompt = f"Image-based embedding from {args.use_image}"
    else:
        args.prompt = ' '.join(args.prompt)
        print(f'Using prompt: <{args.prompt}>')
        text_inputs = torch.cat([clip.tokenize(f"{c}") for c in args.prompt]).to(device)
    
    text_features_map = {}
    for model in models:
        if args.use_image:
            text_feature = text_inputs
        else:
            text_feature = model.encode_text(text_inputs)
            text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)
        text_features_map[model] = text_feature
    
    color_jitter = ColorJitter(args.batch_size, True, mean=args.cg_mean, std=args.cg_std)
    tv_module = TotalVariation()
    
    run_inversion(args, models, text_features_map, tv_module, normalizer, color_jitter, pre_aug, augs)

if __name__ == "__main__":
    main()