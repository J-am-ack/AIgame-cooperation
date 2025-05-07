import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler


def load_lora_model(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 使用中文基础模型
    pipe = StableDiffusionPipeline.from_pretrained(
        "IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-v0.1",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        cache_dir="./model_cache",
        local_files_only=False,
        safety_checker=None
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    # 加载适配中文模型的LoRA权重
    
    # edited
    pipe.load_lora_into_unet(state_dict=model_path,
                             network_alphas= 16, 
                             unet= pipe.unet,
                             adapter_name= "lora_style")
    pipe = pipe.to(device)
    return pipe


def generate_image(pipe, prompt, negative_prompt=None, num_images_per_prompt=1,
                   guidance_scale=7.5, num_inference_steps=50, seed=None):
    generator = torch.manual_seed(seed) if seed else None

    # style_prompt = "卡通风格，可爱"
    style_prompt = "科技感，海报，信科"
    full_prompt = f"{prompt},{style_prompt}"

    # 中文默认负面提示
    default_negative_prompt = "低质量，模糊，肢体畸形，手部变形，裁剪不当"
    negative_prompt = negative_prompt or default_negative_prompt

    images = pipe(
        full_prompt,
        negative_prompt=negative_prompt,
        num_images_per_prompt=num_images_per_prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=generator
    ).images
    return images


def main():
    try:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        folder_time = f"data_{timestamp}"
        # pipe = load_lora_model(f"./checkpoints/{folder_time}.{3}/pytorch_lora_weights.safetensors")
        # pipe = load_lora_model()
        # pipe = load_lora_model('pytorch_lora_weights_emo.safetensors')
        pipe = load_lora_model('pytorch_lora_weights_poster.safetensors')

        # 测试中文提示词
        images = generate_image(pipe,
                                prompt="信科学术科创季",
                                
                                num_images_per_prompt=3,
                                seed=64)

        for i, image in enumerate(images):
            image.save(f'./test_generate/{folder_time}.中文生成图_{i}.png')
    except Exception as e:
        print(f"错误: {e}")


if __name__ == '__main__':
    main()