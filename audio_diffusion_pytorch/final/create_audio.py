import torch
import os
from audio_diffusion_pytorch import DiffusionModel, UNetV0, VDiffusion, VSampler
import librosa
import soundfile as sf

# Define the function to load the model from a checkpoint
def load_model_from_checkpoint(checkpoint_path, device):
    model = DiffusionModel(
        net_t=UNetV0,
        in_channels=1,
        channels=[8, 32, 64, 128, 256, 512, 512, 1024, 1024],
        factors=[1, 4, 4, 4, 2, 2, 2, 2, 2],
        items=[1, 2, 2, 2, 2, 2, 2, 4, 4],
        attentions=[0, 0, 0, 0, 0, 1, 1, 1, 1],
        attention_heads=8,
        attention_features=64,
        diffusion_t=VDiffusion,
        sampler_t=VSampler
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model

# Define the function to generate audio using the model
def generate_audio(model, device, output_length=2 ** 18, num_samples=8, num_steps=77):
    noise = torch.randn(num_samples, 1, output_length).to(device)
    with torch.no_grad():
        samples = model.sample(noise, num_steps=num_steps)
    return samples

# Main function
def main(checkpoint_path, output_dir, device='cuda'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = load_model_from_checkpoint(checkpoint_path, device)
    print(f"Model loaded from {checkpoint_path}")

    samples = generate_audio(model, device)
    print("Audio samples generated!")

    for i, sample in enumerate(samples):
        output_path = os.path.join(output_dir, f'sample_{i + 1}.wav')
        sample = sample.squeeze().cpu().numpy()

        sf.write(output_path, sample, 44100)
        print(f"Sample {i + 1} saved to {output_path}")

if __name__ == '__main__':
    # Specify paths and device
    checkpoint_path = '/home/aix23606/jungmin/audio-diffusion-pytorch/audio_diffusion_pytorch/checkpoints/checkpoint_epoch_30.pt'
    output_dir = '/home/aix23606/jungmin/audio-diffusion-pytorch/audio_diffusion_pytorch/final/created_audio'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Set CUDA visible device
    os.environ["CUDA_VISIBLE_DEVICES"] = "6"  # Set appropriate GPU index here

    # Run main function
    main(checkpoint_path, output_dir, device)
