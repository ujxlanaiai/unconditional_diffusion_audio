from audio_diffusion_pytorch import DiffusionModel, UNetV0, VDiffusion, VSampler

import torch
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

if device == "cuda":
    print("GPU on progress")

model = DiffusionModel(
    # ... same as unconditional model
    net_t=UNetV0, # The model type used for diffusion (U-Net V0 in this case)
    in_channels=2, # U-Net: number of input/output (audio) channels
    channels=[8, 32, 64, 128, 256, 512, 512, 1024, 1024], # U-Net: channels at each layer
    factors=[1, 4, 4, 4, 2, 2, 2, 2, 2], # U-Net: downsampling and upsampling factors at each layer
    items=[1, 2, 2, 2, 2, 2, 2, 4, 4], # U-Net: number of repeating items at each layer
    attentions=[0, 0, 0, 0, 0, 1, 1, 1, 1], # U-Net: attention enabled/disabled at each layer
    attention_heads=8, # U-Net: number of attention heads per attention item
    attention_features=64, # U-Net: number of attention features per attention item
    diffusion_t=VDiffusion, # The diffusion method used
    sampler_t=VSampler, # The diffusion sampler used
    use_text_conditioning=True, # U-Net: enables text conditioning (default T5-base)
    use_embedding_cfg=True, # U-Net: enables classifier free guidance
    embedding_max_length=64, # U-Net: text embedding maximum length (default for T5-base)
    embedding_features=768, # U-Net: text mbedding features (default for T5-base)
    cross_attentions=[0, 0, 0, 1, 1, 1, 1, 1, 1], # U-Net: cross-attention enabled/disabled at each layer
)

model = model.to(device)

# Train model with audio waveforms

print('Creating audio_wave')
audio_wave = torch.randn(1, 2, 2**18).to(device) # [batch, in_channels, length]

print('Creating loss')
loss = model(
    audio_wave,
    text=['Very Soft Violin'], # Text conditioning, one element per batch
    embedding_mask_proba=0.1 # Probability of masking text with learned embedding (Classifier-Free Guidance Mask)
)

print('Processing Backward()')
loss.backward()

# Turn noise into new audio sample with diffusion
print('Create noise')
noise = torch.randn(1, 2, 2**18).to(device)

print('Create sample')
sample = model.sample(
    noise,
    text=['Very soft Violin'],
    embedding_scale=15.0, # Higher for more text importance, suggested range: 1-15 (Classifier-Free Guidance Scale)
    num_steps=100 # Higher for better quality, suggested num_steps: 10-100
)