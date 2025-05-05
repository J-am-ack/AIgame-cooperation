import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import logging
from peft import LoraConfig, PeftConfig  # 从peft库导入LoRA配置

class CustomDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_files = []
        # 遍历主视觉设计和表情包两个子文件夹
        for subdir in ['活动的主视觉设计', '表情包']:
            full_path = os.path.join(image_dir, subdir)
            if os.path.exists(full_path):
                self.image_files.extend(
                    [os.path.join(subdir, f) for f in os.listdir(full_path)
                     if f.endswith(('.png', '.jpg', '.jpeg'))]
                )
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

def train_lora():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载基础模型
    # model_id = "CompVis/stable-diffusion-v1-4"  # 使用v1-4版本，它具有更好的兼容性
    # local_model_path = "./models/sd-v1-5"  # 模型应包含以下文件：
                                                     # - model_index.json
                                                     # - unet/config.json
                                                     # - vae/config.json
                                                     # 以及对应的bin文件
    new_model_id = "IDEA-CCNL/Taiyi-Stable-Diffusion-1B-chinese-v0.1"
    # 设置代理和本地缓存
    import os
    os.environ["HF_HOME"] = "./hf_cache"  # 设置本地缓存目录
    os.environ["PYTORCH_HUB_CACHE"] = "./torch_cache"  # 设置PyTorch缓存目录
    # os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 关键修复：使用国内镜像源
    os.environ["HF_HUB_OFFLINE"] = "0"  # 允许在线下载但优先本地缓存
    
    # 添加重试机制
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            pipe = StableDiffusionPipeline.from_pretrained(
                new_model_id,
                # local_model_path,
                torch_dtype=torch.float32,  # 使用float32以确保兼容性
                cache_dir="./model_cache",
                local_files_only=False,  # 允许在线下载
                # variant="fp16",  # 使用fp16变体
                safety_checker=None  # 禁用安全检查器以减少内存使用
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
            time.sleep(5)  # 等待5秒后重试

    # 使用 add_adapter 添加 LoRA
    # 移除旧的 LoRA 配置代码
    # lora_attn_procs = {}
    # for name in pipe.unet.attn_processors.keys():
    #     # 根据处理器类型确定cross_attention_dim参数
    #     cross_attention_dim = None
    #     if not name.endswith('attn1.processor'):
    #         cross_attention_dim = pipe.unet.config.cross_attention_dim
    #         
    #     # 使用简化的LoRAAttnProcessor初始化，只传入必要参数
    #     # 在最新版本的diffusers中，LoRAAttnProcessor只需要rank和cross_attention_dim参数
    #     lora_attn_procs[name] = LoRAAttnProcessor(
    #         rank=4,
    #         cross_attention_dim=cross_attention_dim
    #     )
    # pipe.unet.set_attn_processor(lora_attn_procs)

    # 使用 add_adapter 添加 LoRA 层
    lora_config = LoraConfig(
        r=4,  # LoRA隐藏维度
        lora_alpha=4,  # LoRA缩放因子
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],  # 需要应用LoRA的模块
        lora_dropout=0.0,  # Dropout概率
        bias="none",  # 是否包含偏置项
        init_lora_weights="gaussian",  # 初始化方法
    )
    # 使用配置对象初始化LoRA适配器
    pipe.unet.add_adapter(adapter_name="lora_style",adapter_config = lora_config)  # 直接传递配置对象

    # 准备数据集
    # 定义数据增强
    resolution = 512
    train_transform = transforms.Compose([
        # transforms.Resize((512, 512)),
        # transforms.ToTensor(),
        # transforms.Normalize([0.5], [0.5])
        transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(resolution),
        # transforms.Resize(512),
        transforms.RandomCrop(512),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    dataset = CustomDataset('Data_pic', transform=train_transform)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    # ds say 3

    # 训练配置
    # 获取 LoRA 参数进行优化
    lora_params = list(filter(lambda p: p.requires_grad, pipe.unet.parameters()))
    # optimizer = torch.optim.AdamW(lora_params, lr=1e-4)
    num_epochs = 10
    total_steps = num_epochs * len(dataloader)

    

    from torch.optim.lr_scheduler import CosineAnnealingLR
    import wandb
    from diffusers import DiffusionPipeline

    # 初始化wandb监控
    wandb.init(project="lora_finetuning", config={
        "learning_rate": 5e-5,
        "architecture": "Stable Diffusion",
        "dataset": "Custom"
    })

    # # 混合精度训练初始化
    # scaler = torch.cuda.amp.GradScaler()

    # 优化器与调度器初始化
    # optimizer = torch.optim.AdamW(
    #     pipe.unet.parameters(), 
    #     lr=1e-5,
    #     # max_lr = 1e-3,
    #     weight_decay=0.01  # 添加权重衰减
    # )
    # scheduler = CosineAnnealingLR(optimizer, T_max= 3)
    from diffusers.optimization import get_cosine_schedule_with_warmup
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    optimizer = torch.optim.AdamW(
        # pipe.unet.parameters(), 
        lora_params,
        lr=5e-5,
        weight_decay=0.01,
        betas=(0.9, 0.999)
    )

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        # num_warmup_steps=100,
        # num_training_steps=1000,
        num_warmup_steps=int(total_steps * 0.1),  # MODIFIED: 按总步数10%预热
        num_training_steps=total_steps,
        num_cycles=0.5  # 半周期余弦退火
    )
    
    # 新增自适应学习率调度器
    plateau_scheduler = ReduceLROnPlateau(  # MODIFIED: 添加损失监控
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=3,
        # verbose=True
    )
    # best_loss 初始化
    # best_loss = float('inf')

    # 训练循环
    pipe.unet.train()
    # running_avg_loss = 0.0  # 用于动态批次跳过的移动平均

    for epoch in range(num_epochs):
        # 在epoch循环内添加
        total_loss = 0.0
        num_batches = len(dataloader)
        for batch_idx, images in enumerate(dataloader):
            images = images.to(device)
            
            # # 动态学习率预热（前500步）
            # if batch_idx < 500:
            #     lr_scale = min(1., float(batch_idx + 1) / 500.)
            #     for pg in optimizer.param_groups:
            #         pg['lr'] = lr_scale * 1e-4

            # 混合精度上下文
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
                
                with torch.no_grad():  # MODIFIED: 固定文本编码器
                    prompt = ["设计一张2025年学术科创季的主视觉海报，包含创新、科技元素"] * images.shape[0]  # 复制prompt匹配batch_size
                    text_input = pipe.tokenizer(
                        prompt,  # 现在输入是列表，长度等于batch_size
                        padding="max_length",
                        max_length=pipe.tokenizer.model_max_length,
                        truncation=True,
                        return_tensors="pt"
                    )
                    encoder_hidden_states = pipe.text_encoder(text_input.input_ids.to(device))[0]  # shape [batch_size, seq_len, hidden_dim]
                # 噪声预测
                noise_pred = pipe.unet(noisy_latents, timesteps, encoder_hidden_states).sample
                
                # 损失计算
                loss = torch.nn.functional.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

            # 动态批次跳过机制
            # if batch_idx == 0:
            #     running_avg_loss = loss.item()
            # else:
            #     running_avg_loss = 0.9 * running_avg_loss + 0.1 * loss.item()
            
            total_loss += loss.item()
            
            # if loss.item() > 3 * running_avg_loss:
            #     optimizer.zero_grad()
            #     print(f"跳过异常批次 {batch_idx}, Loss: {loss.item():.4f}")
            #     continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pipe.unet.parameters(), max_norm=1.0)  # 梯度裁剪
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()  # 学习率调度
            plateau_scheduler.step(loss)  # MODIFIED: 添加自适应调度

            # 日志记录
            if batch_idx % 5 == 0:
                # wandb.log({
                #     "train_loss": loss.item(),
                #     "learning_rate": optimizer.param_groups[0]['lr']
                # })
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.9f}')
        
        # 计算epoch平均损失
        epoch_loss = total_loss / num_batches
        wandb.log({
                    "train_epochloss": epoch_loss,
                    "learning_rate": optimizer.param_groups[0]['lr']
                })
        print(f'Epoch: {epoch},epoch-Loss: {epoch_loss:.9f}')
        # 每个epoch保存最佳模型
        # if epoch_loss < best_loss:
            # print("updated_loss, on", epoch ,"epoch")
            
            # best_loss = epoch_loss
            # 新版参数保存方法
        import os
        from datetime import datetime
        # 生成时间戳（替换Windows非法字符）
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # 2024-05-20_14-30-45
        folder_time = f"data_{timestamp}"
        pipe.unet.save_lora_adapter(
            save_directory=f"./checkpoints/{folder_time}'.'{epoch}",
            adapter_name="lora_style",
            safe_serialization=True
            )
            # best_loss = loss
        
    
    # 这玩意永远报错（但是好像不太影响？）
    pipe.save_lora_weights(
                save_directory="./checkpoints/final{epoch}",
                # adapter_name="lora_style",  # 必须与add_adapter时一致
                safe_serialization=True
            )


if __name__ == '__main__':
    train_lora()












