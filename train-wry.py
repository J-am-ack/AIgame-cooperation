import os  # 确保在文件顶部导入os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import logging
from peft import LoraConfig, PeftConfig
import glob

# 识别图片后缀
IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".PNG", ".JPG", ".JPEG", ".WEBP", ".BMP"]

class Text2ImageDataset(Dataset):
    """
    用于构建文本到图像模型的微调数据集
    """
    def __init__(self, images_folder, captions_folder, transform, tokenizer=None):
        """
        参数:
            - images_folder: str, 图像文件夹路径
            - captions_folder: str, 标注文件夹路径
            - transform: function, 将原始图像转换为 torch.Tensor
            - tokenizer: CLIPTokenizer, 将文本标注转为 word ids (可选)
        """
        # 初始化图像路径列表，并根据指定的扩展名找到所有图像文件
        self.image_paths = []
        for ext in IMAGE_EXTENSIONS:
            self.image_paths.extend(glob.glob(os.path.join(images_folder, f"*{ext}")))
        self.image_paths = sorted(self.image_paths)

        # 加载对应的文本标注，根据图片名找到对应的txt文件
        self.captions = []
        for img_path in self.image_paths:
            # 获取不带扩展名的文件名
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            txt_path = os.path.join(captions_folder, f"{base_name}.txt")
            
            try:
                with open(txt_path, "r", encoding="utf-8") as f:
                    caption = f.readline().strip()
                    self.captions.append(caption)
            except Exception as e:
                print(f"⚠️ 无法加载标注文件: {txt_path}, 错误: {e}")
                self.captions.append("")  # 使用空字符串作为默认标注

        # 确保图像和文本标注数量一致
        print(len(self.captions),len(self.image_paths))
        if len(self.captions) != len(self.image_paths):
            raise ValueError("图像数量与文本标注数量不一致，请检查数据集。")

        self.transform = transform
        self.tokenizer = tokenizer

        # 如果有tokenizer，预处理文本
        if tokenizer is not None:
            inputs = tokenizer(
                self.captions, 
                max_length=tokenizer.model_max_length, 
                padding="max_length", 
                truncation=True, 
                return_tensors="pt"
            )
            self.input_ids = inputs.input_ids
        else:
            self.input_ids = None

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        caption = self.captions[idx]
        
        try:
            # 加载图像并将其转换为 RGB 模式，然后应用数据增强
            image = Image.open(img_path).convert("RGB")
            tensor = self.transform(image)
        except Exception as e:
            print(f"⚠️ 无法加载图像路径: {img_path}, 错误: {e}")
            # 返回一个全零的张量以避免崩溃
            tensor = torch.zeros((3, 512, 512))  # 假设分辨率是512
            
        if self.tokenizer is not None:
            input_id = self.input_ids[idx]
            return tensor, input_id  # 返回处理后的图像和tokenized文本
        else:
            return tensor, caption  # 或者返回原始文本

    def __len__(self):
        return len(self.image_paths)

def train_lora():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载基础模型
    new_model_id = "IDEA-CCNL/Taiyi-Stable-Diffusion-1B-chinese-v0.1"
    # 移除函数内部的os.environ设置，使用文件顶部导入的os模块
    os.environ["HF_HOME"] = "./hf_cache"
    os.environ["PYTORCH_HUB_CACHE"] = "./torch_cache"
    os.environ["HF_HUB_OFFLINE"] = "0"
    
    # 添加重试机制
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            pipe = StableDiffusionPipeline.from_pretrained(
                new_model_id,
                torch_dtype=torch.float32,
                cache_dir="./model_cache",
                local_files_only=False,
                safety_checker=None
            )
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
            pipe = pipe.to(device)
            break
        except Exception as e:
            retry_count += 1
            print(f"尝试 {retry_count}/{max_retries} 失败: {str(e)}")
            if retry_count == max_retries:
                raise Exception(f"在 {max_retries} 次尝试后仍无法加载模型: {str(e)}")
            import time
            time.sleep(5)

    # 配置LoRA
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        lora_dropout=0.1,
        bias="none",
        init_lora_weights="gaussian",
    )
    pipe.unet.add_adapter(adapter_config=lora_config ,adapter_name = "lora_style")
    

    # 准备数据集
    resolution = 512
    train_transform = transforms.Compose([
        transforms.Resize((resolution, resolution)), 
        transforms.CenterCrop(resolution),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    # 使用新的数据集类
    dataset = Text2ImageDataset(
        images_folder='emojiset/images',
        captions_folder='emojiset/captions',
        transform=train_transform,
        tokenizer= pipe.tokenizer  # 传入tokenizer以自动处理文本
    )
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # 训练配置
    lora_params = list(filter(lambda p: p.requires_grad, pipe.unet.parameters()))
    num_epochs = 5
    total_steps = num_epochs * len(dataloader)

    # 初始化wandb监控
    import wandb
    wandb.init(project="lora_finetuning", config={
        "learning_rate": 1e-7,
        "architecture": "Stable Diffusion",
        "dataset": "Text2Image"  # modified
    })

    # 优化器与调度器
    optimizer = torch.optim.AdamW(
        lora_params,
        lr=1e-7,
        weight_decay=0.01,
        betas=(0.9, 0.999)
    )

    from diffusers.optimization import get_cosine_schedule_with_warmup
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps,
        num_cycles=0.5
    )
    
    plateau_scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=3
    )

    # 训练循环
    pipe.unet.train()

    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = len(dataloader)
        
        for batch_idx, (images, input_ids) in enumerate(dataloader):
            images = images.to(device)
            input_ids = input_ids.to(device)
            
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                # 潜在空间处理
                latents = pipe.vae.encode(images).latent_dist.sample()
                latents = latents * pipe.vae.config.scaling_factor
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                
                # 时间步采样
                timesteps = torch.randint(0, pipe.scheduler.config.num_train_timesteps, 
                                        (bsz,), device=latents.device).long()
                
                # 加噪处理
                noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)
                
                # 使用数据集中提供的文本编码
                encoder_hidden_states = pipe.text_encoder(input_ids)[0]
                
                # 噪声预测
                noise_pred = pipe.unet(noisy_latents, timesteps, encoder_hidden_states).sample
                
                # 损失计算
                loss = torch.nn.functional.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

            total_loss += loss.item()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pipe.unet.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            plateau_scheduler.step(loss)

            # 日志记录
            if batch_idx % 10 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.9f}')
        
        # 计算epoch平均损失
        epoch_loss = total_loss / num_batches
        wandb.log({
            "train_epochloss": epoch_loss,
            "learning_rate": optimizer.param_groups[0]['lr']
        })
        print(f'Epoch: {epoch}, Epoch Loss: {epoch_loss:.9f}')
        
        # 保存模型
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        folder_time = f"data_{timestamp}"
        pipe.unet.save_lora_adapter(
            save_directory=f"./checkpoints/{folder_time}.{epoch}",
            adapter_name="lora_style",
            safe_serialization=True
        )
        # try:
        #     pipe.save_lora_weights(
        #             save_directory="./checkpoints/w/{folder_time}'.'{epoch}",
        #             # adapter_name="lora_style",  # 必须与add_adapter时一致
        #             unet_lora_layers=pipe.unet.lora_layers,          # 获取已注入的UNet LoRA层
        #             # 显示指定
        #             text_encoder_lora_layers=pipe.text_encoder.lora_layers if hasattr(pipe.text_encoder, "lora_layers") else None,
        #             # unet_lora_layers=,
        #             # text_encoder_lora_layers=,
        #             safe_serialization=True
        #         )
        # except Exception as e:
        #     print("save weights failed,","报错：",e)
        
    
    # # 保存最终模型
    # pipe.save_lora_weights(
    #     save_directory="./checkpoints/final",
    #     safe_serialization=True
    # )

if __name__ == '__main__':
    train_lora()