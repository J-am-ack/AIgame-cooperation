c:\Users\lenovo\.conda\envs\mysd\lib\site-packages\diffusers\loaders\unet.py:484: FutureWarning: `save_attn_procs` is deprecated and will be removed in version 0.40.0. Using the `save_attn_procs()` method has been deprecated and will be removed in a future version. Please use `save_lora_adapter()`.
  deprecate("save_attn_procs", "0.40.0", deprecation_message)
PS C:\Users\lenovo\Desktop\大一下\ai-self\co-w\Lora_exper2-retry\Lora_exper2> ^C
PS C:\Users\lenovo\Desktop\大一下\ai-self\co-w\Lora_exper2-retry\Lora_exper2>
PS C:\Users\lenovo\Desktop\大一下\ai-self\co-w\Lora_exper2-retry\Lora_exper2>  c:; cd 'c:\Users\lenovo\Desktop\大一下\ai-self\co-w\Lora_exper2-retry\Lora_exper2'; & 'c:\Users\lenovo\.conda\envs\mysd\python.exe' 'c:\Users\lenovo\.vscode\extensions\ms-python.debugpy-2025.6.0-win32-x64\bundled\libs\debugpy\launcher' '50578' '--' 'C:\Users\lenovo\Desktop\大一下\ai-self\co-w\Lora_exper2-retry\Lora_exper2\edited_train.py' 
Couldn't connect to the Hub: (ProtocolError('Connection aborted.', ConnectionResetError(10054, '远程主机强迫关闭了一个现有的连接。', None, 10054, None)), '(Request ID: 5a390a48-d4b2-4635-aebd-f964f9c0b33c)').
Will try to load from local cache.
Loading pipeline components...: 100%|██████████████████████████| 6/6 [00:02<00:00,  2.90it/s]
You have disabled the safety checker for <class 'diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline'> by passing `safety_checker=None`. Ensure that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered results in services or applications open to the public. Both the diffusers team and Hugging Face strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling it only for use-cases that involve analyzing network behavior or auditing its results. For more information, please have a look at https://github.com/huggingface/diffusers/pull/254 .       
wandb: (1) Create a W&B account
wandb: (2) Use an existing W&B account
wandb: (3) Don't visualize my results
wandb: Enter your choice: 1
wandb: You chose 'Create a W&B account'
wandb: Create an account here: https://wandb.ai/authorize?signup=true&ref=models
wandb: Paste an API key from your profile and hit enter, or press ctrl+c to quit:
PS C:\Users\lenovo\Desktop\大一下\ai-self\co-w\Lora_exper2-retry\Lora_exper2> ^C
PS C:\Users\lenovo\Desktop\大一下\ai-self\co-w\Lora_exper2-retry\Lora_exper2>
PS C:\Users\lenovo\Desktop\大一下\ai-self\co-w\Lora_exper2-retry\Lora_exper2>  c:; cd 'c:\Users\lenovo\Desktop\大一下\ai-self\co-w\Lora_exper2-retry\Lora_exper2'; & 'c:\Users\lenovo\.conda\envs\mysd\python.exe' 'c:\Users\lenovo\.vscode\extensions\ms-python.debugpy-2025.6.0-win32-x64\bundled\libs\debugpy\launcher' '51645' '--' 'C:\Users\lenovo\Desktop\大一下\ai-self\co-w\Lora_exper2-retry\Lora_exper2\edited_train.py'
Couldn't connect to the Hub: (ProtocolError('Connection aborted.', ConnectionResetError(10054, '远程主机强迫关闭了一个现有的连接。', None, 10054, None)), '(Request ID: 04807739-b437-4db0-a403-b1c0260cc18c)').
Will try to load from local cache.
Loading pipeline components...: 100%|██████████████████████████| 6/6 [00:01<00:00,  4.58it/s]
You have disabled the safety checker for <class 'diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline'> by passing `safety_checker=None`. Ensure that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered results in services or applications open to the public. Both the diffusers team and Hugging Face strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling it only for use-cases that involve analyzing network behavior or auditing its results. For more information, please have a look at https://github.com/huggingface/diffusers/pull/254 .       
wandb: (1) Create a W&B account
wandb: (2) Use an existing W&B account
wandb: (3) Don't visualize my results
wandb: Enter your choice: 2
wandb: You chose 'Use an existing W&B account'
wandb: Logging into wandb.ai. (Learn how to deploy a W&B server locally: https://wandb.me/wandb-server)
wandb: You can find your API key in your browser here: https://wandb.ai/authorize?ref=models  
wandb: Paste an API key from your profile and hit enter, or press ctrl+c to quit:
wandb: No netrc file found, creating one.
wandb: Appending key for api.wandb.ai to your netrc file: C:\Users\lenovo\_netrc
wandb: Currently logged in as: 3922909893 (3922909893-peking-university) to https://api.wandb.ai. Use `wandb login --relogin` to force relogin
wandb: Tracking run with wandb version 0.19.10
wandb: Run data is saved locally in C:\Users\lenovo\Desktop\大一下\ai-self\co-w\Lora_exper2-retry\Lora_exper2\wandb\run-20250504_153034-yzwja762
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run carbonite-commander-1
wandb:  View project at https://wandb.ai/3922909893-peking-university/lora_finetuning
wandb:  View run at https://wandb.ai/3922909893-peking-university/lora_finetuning/runs/yzwja762
C:\Users\lenovo\Desktop\大一下\ai-self\co-w\Lora_exper2-retry\Lora_exper2\edited_train.py:188: FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.
  scaler = torch.cuda.amp.GradScaler()
PS C:\Users\lenovo\Desktop\大一下\ai-self\co-w\Lora_exper2-retry\Lora_exper2> ^C
PS C:\Users\lenovo\Desktop\大一下\ai-self\co-w\Lora_exper2-retry\Lora_exper2>
PS C:\Users\lenovo\Desktop\大一下\ai-self\co-w\Lora_exper2-retry\Lora_exper2>  c:; cd 'c:\Users\lenovo\Desktop\大一下\ai-self\co-w\Lora_exper2-retry\Lora_exper2'; & 'c:\Users\lenovo\.conda\envs\mysd\python.exe' 'c:\Users\lenovo\.vscode\extensions\ms-python.debugpy-2025.6.0-win32-x64\bundled\libs\debugpy\launcher' '51752' '--' 'C:\Users\lenovo\Desktop\大一下\ai-self\co-w\Lora_exper2-retry\Lora_exper2\edited_train.py' 
Couldn't connect to the Hub: (ProtocolError('Connection aborted.', ConnectionResetError(10054, '远程主机强迫关闭了一个现有的连接。', None, 10054, None)), '(Request ID: 3c18bbbf-a9ae-4e3b-b426-ad712d7f5cc5)').
Will try to load from local cache.
Loading pipeline components...: 100%|██████████████████████████| 6/6 [00:02<00:00,  2.39it/s]
You have disabled the safety checker for <class 'diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline'> by passing `safety_checker=None`. Ensure that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered results in services or applications open to the public. Both the diffusers team and Hugging Face strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling it only for use-cases that involve analyzing network behavior or auditing its results. For more information, please have a look at https://github.com/huggingface/diffusers/pull/254 .       
wandb: Currently logged in as: 3922909893 (3922909893-peking-university) to https://api.wandb.ai. Use `wandb login --relogin` to force relogin
wandb: Tracking run with wandb version 0.19.10
wandb: Run data is saved locally in C:\Users\lenovo\Desktop\大一下\ai-self\co-w\Lora_exper2-retry\Lora_exper2\wandb\run-20250504_153224-p6jexin9
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run ancient-womprat-2
wandb:  View project at https://wandb.ai/3922909893-peking-university/lora_finetuning
wandb:  View run at https://wandb.ai/3922909893-peking-university/lora_finetuning/runs/p6jexin9
Epoch: 0, Batch: 0, Loss: 0.2977
跳过异常批次 9, Loss: 1.3230
Epoch: 0, Batch: 10, Loss: 0.3322
Epoch: 0, Batch: 20, Loss: 0.0766
Epoch: 0, Batch: 30, Loss: 0.3078
PS C:\Users\lenovo\Desktop\大一下\ai-self\co-w\Lora_exper2-retry\Lora_exper2> torch.nn.utils.clip_grad_norm_(pipe.unet.parameters(), max_norm=1.0)  # 梯度裁剪^C
PS C:\Users\lenovo\Desktop\大一下\ai-self\co-w\Lora_exper2-retry\Lora_exper2>
PS C:\Users\lenovo\Desktop\大一下\ai-self\co-w\Lora_exper2-retry\Lora_exper2>  c:; cd 'c:\Users\lenovo\Desktop\大一下\ai-self\co-w\Lora_exper2-retry\Lora_exper2'; & 'c:\Users\lenovo\.conda\envs\mysd\python.exe' 'c:\Users\lenovo\.vscode\extensions\ms-python.debugpy-2025.6.0-win32-x64\bundled\libs\debugpy\launcher' '53109' '--' 'C:\Users\lenovo\Desktop\大一下\ai-self\co-w\Lora_exper2-retry\Lora_exper2\edited_train.py'
Couldn't connect to the Hub: (ProtocolError('Connection aborted.', ConnectionResetError(10054, '远程主机强迫关闭了一个现有的连接。', None, 10054, None)), '(Request ID: 95d352c3-8580-4602-9568-f2a96c77907a)').
Will try to load from local cache.
Loading pipeline components...: 100%|██████████████████████████| 6/6 [00:04<00:00,  1.31it/s]
You have disabled the safety checker for <class 'diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline'> by passing `safety_checker=None`. Ensure that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered results in services or applications open to the public. Both the diffusers team and Hugging Face strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling it only for use-cases that involve analyzing network behavior or auditing its results. For more information, please have a look at https://github.com/huggingface/diffusers/pull/254 .       
wandb: Currently logged in as: 3922909893 (3922909893-peking-university) to https://api.wandb.ai. Use `wandb login --relogin` to force relogin
wandb: Tracking run with wandb version 0.19.10
wandb: Run data is saved locally in C:\Users\lenovo\Desktop\大一下\ai-self\co-w\Lora_exper2-retry\Lora_exper2\wandb\run-20250504_160019-9ps4979o
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run grievous-bantha-3
wandb:  View project at https://wandb.ai/3922909893-peking-university/lora_finetuning
wandb:  View run at https://wandb.ai/3922909893-peking-university/lora_finetuning/runs/9ps4979o
Epoch: 0, Batch: 0, Loss: 0.8004
Epoch: 0, Batch: 10, Loss: 0.7169
Epoch: 0, Batch: 20, Loss: 0.0679
Epoch: 0, Batch: 30, Loss: 0.2458
跳过异常批次 34, Loss: 1.2475
PS C:\Users\lenovo\Desktop\大一下\ai-self\co-w\Lora_exper2-retry\Lora_exper2> ^C
PS C:\Users\lenovo\Desktop\大一下\ai-self\co-w\Lora_exper2-retry\Lora_exper2>
PS C:\Users\lenovo\Desktop\大一下\ai-self\co-w\Lora_exper2-retry\Lora_exper2>  c:; cd 'c:\Users\lenovo\Desktop\大一下\ai-self\co-w\Lora_exper2-retry\Lora_exper2'; & 'c:\Users\lenovo\.conda\envs\mysd\python.exe' 'c:\Users\lenovo\.vscode\extensions\ms-python.debugpy-2025.6.0-win32-x64\bundled\libs\debugpy\launcher' '53473' '--' 'C:\Users\lenovo\Desktop\大一下\ai-self\co-w\Lora_exper2-retry\Lora_exper2\edited_train.py' 
Couldn't connect to the Hub: (ProtocolError('Connection aborted.', ConnectionResetError(10054, '远程主机强迫关闭了一个现有的连接。', None, 10054, None)), '(Request ID: 5997aba2-ed2b-4c27-b804-d3e643b2df65)').
Will try to load from local cache.
Loading pipeline components...: 100%|██████████████████████████| 6/6 [00:03<00:00,  1.88it/s]
You have disabled the safety checker for <class 'diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline'> by passing `safety_checker=None`. Ensure that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered results in services or applications open to the public. Both the diffusers team and Hugging Face strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling it only for use-cases that involve analyzing network behavior or auditing its results. For more information, please have a look at https://github.com/huggingface/diffusers/pull/254 .       
wandb: Currently logged in as: 3922909893 (3922909893-peking-university) to https://api.wandb.ai. Use `wandb login --relogin` to force relogin
wandb: Tracking run with wandb version 0.19.10
wandb: Run data is saved locally in C:\Users\lenovo\Desktop\大一下\ai-self\co-w\Lora_exper2-retry\Lora_exper2\wandb\run-20250504_160716-qgmllj27
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run star-fighter-4
wandb:  View project at https://wandb.ai/3922909893-peking-university/lora_finetuning
wandb:  View run at https://wandb.ai/3922909893-peking-university/lora_finetuning/runs/qgmllj27
Epoch: 0, Batch: 0, Loss: 0.4253
Epoch: 0, Batch: 10, Loss: 0.0537
Epoch: 0, Batch: 20, Loss: 0.8834
Epoch: 0, Batch: 30, Loss: 0.1199
PS C:\Users\lenovo\Desktop\大一下\ai-self\co-w\Lora_exper2-retry\Lora_exper2> ^C
PS C:\Users\lenovo\Desktop\大一下\ai-self\co-w\Lora_exper2-retry\Lora_exper2>
PS C:\Users\lenovo\Desktop\大一下\ai-self\co-w\Lora_exper2-retry\Lora_exper2>  c:; cd 'c:\Users\lenovo\Desktop\大一下\ai-self\co-w\Lora_exper2-retry\Lora_exper2'; & 'c:\Users\lenovo\.conda\envs\mysd\python.exe' 'c:\Users\lenovo\.vscode\extensions\ms-python.debugpy-2025.6.0-win32-x64\bundled\libs\debugpy\launcher' '54020' '--' 'C:\Users\lenovo\Desktop\大一下\ai-self\co-w\Lora_exper2-retry\Lora_exper2\edited_train.py' 
Loading pipeline components...: 100%|██████████████████████████| 6/6 [00:03<00:00,  1.72it/s]
You have disabled the safety checker for <class 'diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline'> by passing `safety_checker=None`. Ensure that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered results in services or applications open to the public. Both the diffusers team and Hugging Face strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling it only for use-cases that involve analyzing network behavior or auditing its results. For more information, please have a look at https://github.com/huggingface/diffusers/pull/254 .       
wandb: Currently logged in as: 3922909893 (3922909893-peking-university) to https://api.wandb.ai. Use `wandb login --relogin` to force relogin
wandb: Tracking run with wandb version 0.19.10
wandb: Run data is saved locally in C:\Users\lenovo\Desktop\大一下\ai-self\co-w\Lora_exper2-retry\Lora_exper2\wandb\run-20250504_162416-bnc2vcfq
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run forgotten-womprat-5
wandb:  View project at https://wandb.ai/3922909893-peking-university/lora_finetuning
wandb:  View run at https://wandb.ai/3922909893-peking-university/lora_finetuning/runs/bnc2vcfq
Epoch: 0, Batch: 0, Loss: 0.1795
跳过异常批次 1, Loss: 0.7942
跳过异常批次 8, Loss: 1.3302
Epoch: 0, Batch: 10, Loss: 0.2572
Epoch: 0, Batch: 20, Loss: 0.0651
跳过异常批次 24, Loss: 1.2525
Epoch: 0, Batch: 30, Loss: 0.7594
Epoch: 1, Batch: 0, Loss: 0.2908
Epoch: 1, Batch: 10, Loss: 0.5341
跳过异常批次 17, Loss: 1.1910
Epoch: 1, Batch: 20, Loss: 0.0666
Epoch: 1, Batch: 30, Loss: 0.1576
Epoch: 2, Batch: 0, Loss: 0.0947
跳过异常批次 1, Loss: 0.3853
跳过异常批次 2, Loss: 0.5008
跳过异常批次 4, Loss: 0.7187
Epoch: 2, Batch: 10, Loss: 0.2373
Epoch: 2, Batch: 20, Loss: 0.3805
Epoch: 2, Batch: 30, Loss: 0.4984
Epoch: 3, Batch: 0, Loss: 0.2165
Epoch: 3, Batch: 10, Loss: 0.1267
跳过异常批次 20, Loss: 0.9899
Epoch: 3, Batch: 30, Loss: 0.3664
Epoch: 4, Batch: 0, Loss: 0.1687
跳过异常批次 3, Loss: 1.0163
Epoch: 4, Batch: 10, Loss: 0.8756
Epoch: 4, Batch: 20, Loss: 0.3338
跳过异常批次 23, Loss: 1.2446
Epoch: 4, Batch: 30, Loss: 0.1727
PS C:\Users\lenovo\Desktop\大一下\ai-self\co-w\Lora_exper2-retry\Lora_exper2> ^C
PS C:\Users\lenovo\Desktop\大一下\ai-self\co-w\Lora_exper2-retry\Lora_exper2>
PS C:\Users\lenovo\Desktop\大一下\ai-self\co-w\Lora_exper2-retry\Lora_exper2>  c:; cd 'c:\Users\lenovo\Desktop\大一下\ai-self\co-w\Lora_exper2-retry\Lora_exper2'; & 'c:\Users\lenovo\.conda\envs\mysd\python.exe' 'c:\Users\lenovo\.vscode\extensions\ms-python.debugpy-2025.6.0-win32-x64\bundled\libs\debugpy\launcher' '56527' '--' 'C:\Users\lenovo\Desktop\大一下\ai-self\co-w\Lora_exper2-retry\Lora_exper2\edited_train.py' 
Couldn't connect to the Hub: (ProtocolError('Connection aborted.', ConnectionResetError(10054, '远程主机强迫关闭了一个现有的连接。', None, 10054, None)), '(Request ID: 801682c1-21c0-44d0-9faf-bc02fafd7d49)').
Will try to load from local cache.
Loading pipeline components...: 100%|██████████████████████████| 6/6 [00:14<00:00,  2.41s/it]
You have disabled the safety checker for <class 'diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline'> by passing `safety_checker=None`. Ensure that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered results in services or applications open to the public. Both the diffusers team and Hugging Face strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling it only for use-cases that involve analyzing network behavior or auditing its results. For more information, please have a look at https://github.com/huggingface/diffusers/pull/254 .       
wandb: Currently logged in as: 3922909893 (3922909893-peking-university) to https://api.wandb.ai. Use `wandb login --relogin` to force relogin
wandb: Tracking run with wandb version 0.19.10
wandb: Run data is saved locally in C:\Users\lenovo\Desktop\大一下\ai-self\co-w\Lora_exper2-retry\Lora_exper2\wandb\run-20250504_172330-zbpl74yz
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run galactic-bantha-6
wandb:  View project at https://wandb.ai/3922909893-peking-university/lora_finetuning
wandb:  View run at https://wandb.ai/3922909893-peking-university/lora_finetuning/runs/zbpl74yz
Epoch: 0, Batch: 0, Loss: 0.2163
跳过异常批次 5, Loss: 1.1364
Epoch: 0, Batch: 10, Loss: 0.4074
Epoch: 0, Batch: 20, Loss: 0.3324
Epoch: 0, Batch: 30, Loss: 0.3414
updated_loss, on 0 epoch
Epoch: 1, Batch: 0, Loss: 0.0919
跳过异常批次 4, Loss: 0.9669
跳过异常批次 9, Loss: 1.1047
Epoch: 1, Batch: 10, Loss: 0.3405
Epoch: 1, Batch: 20, Loss: 0.2897
Epoch: 1, Batch: 30, Loss: 0.8933
updated_loss, on 1 epoch
Epoch: 2, Batch: 0, Loss: 0.1080
跳过异常批次 3, Loss: 1.2982
跳过异常批次 4, Loss: 1.2893
Epoch: 2, Batch: 10, Loss: 0.0606
Epoch: 2, Batch: 20, Loss: 0.2576
Epoch: 2, Batch: 30, Loss: 0.0772
Epoch: 3, Batch: 0, Loss: 0.1232
跳过异常批次 1, Loss: 0.5581
跳过异常批次 9, Loss: 0.9400
Epoch: 3, Batch: 10, Loss: 0.0466
跳过异常批次 12, Loss: 1.1250
Epoch: 3, Batch: 20, Loss: 0.3610
Epoch: 3, Batch: 30, Loss: 0.1428
跳过异常批次 33, Loss: 1.1333
Epoch: 4, Batch: 0, Loss: 0.1081
跳过异常批次 4, Loss: 1.0914
跳过异常批次 6, Loss: 0.9019
Epoch: 4, Batch: 10, Loss: 0.0593
Epoch: 4, Batch: 20, Loss: 0.1209
跳过异常批次 23, Loss: 1.2439
Epoch: 4, Batch: 30, Loss: 0.0883
跳过异常批次 31, Loss: 1.0418
updated_loss, on 4 epoch
Epoch: 5, Batch: 0, Loss: 0.1469
跳过异常批次 4, Loss: 1.0939
Epoch: 5, Batch: 10, Loss: 0.8986
Epoch: 5, Batch: 20, Loss: 0.1810
Epoch: 5, Batch: 30, Loss: 0.0892
Epoch: 6, Batch: 0, Loss: 0.1411
跳过异常批次 3, Loss: 0.9446
Epoch: 6, Batch: 10, Loss: 0.0779
Epoch: 6, Batch: 20, Loss: 0.2528
Epoch: 6, Batch: 30, Loss: 0.1117
Epoch: 7, Batch: 0, Loss: 0.4332
Epoch: 7, Batch: 10, Loss: 0.0717
Epoch: 7, Batch: 20, Loss: 0.0728
Epoch: 7, Batch: 30, Loss: 1.0033
Epoch: 8, Batch: 0, Loss: 0.0865
跳过异常批次 1, Loss: 0.6609
跳过异常批次 4, Loss: 0.7630
跳过异常批次 5, Loss: 1.1329
Epoch: 8, Batch: 10, Loss: 0.2711
Epoch: 8, Batch: 20, Loss: 0.1246
Epoch: 8, Batch: 30, Loss: 0.1961
Epoch: 9, Batch: 0, Loss: 0.3362
跳过异常批次 3, Loss: 1.1299
Epoch: 9, Batch: 10, Loss: 0.0992
跳过异常批次 14, Loss: 1.2841
Epoch: 9, Batch: 20, Loss: 0.2281
Epoch: 9, Batch: 30, Loss: 0.8960
PS C:\Users\lenovo\Desktop\大一下\ai-self\co-w\Lora_exper2-retry\Lora_exper2> ^C
PS C:\Users\lenovo\Desktop\大一下\ai-self\co-w\Lora_exper2-retry\Lora_exper2>
PS C:\Users\lenovo\Desktop\大一下\ai-self\co-w\Lora_exper2-retry\Lora_exper2>  c:; cd 'c:\Users\lenovo\Desktop\大一下\ai-self\co-w\Lora_exper2-retry\Lora_exper2'; & 'c:\Users\lenovo\.conda\envs\mysd\python.exe' 'c:\Users\lenovo\.vscode\extensions\ms-python.debugpy-2025.6.0-win32-x64\bundled\libs\debugpy\launcher' '64393' '--' 'C:\Users\lenovo\Desktop\大一下\ai-self\co-w\Lora_exper2-retry\Lora_exper2\edited_train.py' 
Couldn't connect to the Hub: (ProtocolError('Connection aborted.', ConnectionResetError(10054, '远程主机强迫关闭了一个现有的连接。', None, 10054, None)), '(Request ID: fd131cef-4d8b-41bb-82e6-c2b18682bc17)').
Will try to load from local cache.
尝试 1/3 失败: Cannot load model IDEA-CCNL/Taiyi-Stable-Diffusion-1B-chinese-v0.1: model is not cached locally and an error occurred while trying to fetch metadata from the Hub. Please check out the root cause in the stacktrace above.
Couldn't connect to the Hub: (ProtocolError('Connection aborted.', ConnectionResetError(10054, '远程主机强迫关闭了一个现有的连接。', None, 10054, None)), '(Request ID: b339fd8d-6f29-4b18-adaa-185722a2b7ef)').
Will try to load from local cache.
尝试 2/3 失败: Cannot load model IDEA-CCNL/Taiyi-Stable-Diffusion-1B-chinese-v0.1: model is not cached locally and an error occurred while trying to fetch metadata from the Hub. Please check out the root cause in the stacktrace above.
Couldn't connect to the Hub: (ProtocolError('Connection aborted.', ConnectionResetError(10054, '远程主机强迫关闭了一个现有的连接。', None, 10054, None)), '(Request ID: 11cceb05-6289-4404-a302-4c251a2bcd56)').
Will try to load from local cache.
尝试 3/3 失败: Cannot load model IDEA-CCNL/Taiyi-Stable-Diffusion-1B-chinese-v0.1: model is not cached locally and an error occurred while trying to fetch metadata from the Hub. Please check out the root cause in the stacktrace above.
PS C:\Users\lenovo\Desktop\大一下\ai-self\co-w\Lora_exper2-retry\Lora_exper2> ^C
PS C:\Users\lenovo\Desktop\大一下\ai-self\co-w\Lora_exper2-retry\Lora_exper2>
PS C:\Users\lenovo\Desktop\大一下\ai-self\co-w\Lora_exper2-retry\Lora_exper2>  c:; cd 'c:\Users\lenovo\Desktop\大一下\ai-self\co-w\Lora_exper2-retry\Lora_exper2'; & 'c:\Users\lenovo\.conda\envs\mysd\python.exe' 'c:\Users\lenovo\.vscode\extensions\ms-python.debugpy-2025.6.0-win32-x64\bundled\libs\debugpy\launcher' '65234' '--' 'C:\Users\lenovo\Desktop\大一下\ai-self\co-w\Lora_exper2-retry\Lora_exper2\edited_train.py' 
Couldn't connect to the Hub: (ProtocolError('Connection aborted.', ConnectionResetError(10054, '远程主机强迫关闭了一个现有的连接。', None, 10054, None)), '(Request ID: 8dd572a5-74c5-4245-ab0e-60288338d296)').
Will try to load from local cache.
尝试 1/3 失败: Cannot load model IDEA-CCNL/Taiyi-Stable-Diffusion-1B-chinese-v0.1: model is not cached locally and an error occurred while trying to fetch metadata from the Hub. Please check out the root cause in the stacktrace above.
Couldn't connect to the Hub: (ProtocolError('Connection aborted.', ConnectionResetError(10054, '远程主机强迫关闭了一个现有的连接。', None, 10054, None)), '(Request ID: cb737119-b1ea-45c5-a8fe-95d6e1fc50c2)').
Will try to load from local cache.
尝试 2/3 失败: Cannot load model IDEA-CCNL/Taiyi-Stable-Diffusion-1B-chinese-v0.1: model is not cached locally and an error occurred while trying to fetch metadata from the Hub. Please check out the root cause in the stacktrace above.
Couldn't connect to the Hub: (ProtocolError('Connection aborted.', ConnectionResetError(10054, '远程主机强迫关闭了一个现有的连接。', None, 10054, None)), '(Request ID: 2764c97a-7791-4533-ad99-d022d5b0d232)').
Will try to load from local cache.
尝试 3/3 失败: Cannot load model IDEA-CCNL/Taiyi-Stable-Diffusion-1B-chinese-v0.1: model is not cached locally and an error occurred while trying to fetch metadata from the Hub. Please check out the root cause in the stacktrace above.
PS C:\Users\lenovo\Desktop\大一下\ai-self\co-w\Lora_exper2-retry\Lora_exper2> ^C
PS C:\Users\lenovo\Desktop\大一下\ai-self\co-w\Lora_exper2-retry\Lora_exper2>
PS C:\Users\lenovo\Desktop\大一下\ai-self\co-w\Lora_exper2-retry\Lora_exper2>  c:; cd 'c:\Users\lenovo\Desktop\大一下\ai-self\co-w\Lora_exper2-retry\Lora_exper2'; & 'c:\Users\lenovo\.conda\envs\mysd\python.exe' 'c:\Users\lenovo\.vscode\extensions\ms-python.debugpy-2025.6.0-win32-x64\bundled\libs\debugpy\launcher' '65322' '--' 'C:\Users\lenovo\Desktop\大一下\ai-self\co-w\Lora_exper2-retry\Lora_exper2\edited_train.py' 
Couldn't connect to the Hub: (ProtocolError('Connection aborted.', ConnectionResetError(10054, '远程主机强迫关闭了一个现有的连接。', None, 10054, None)), '(Request ID: 65799ec4-50bd-4a68-ad1e-4be9ae69a768)').
Will try to load from local cache.
尝试 1/3 失败: Cannot load model IDEA-CCNL/Taiyi-Stable-Diffusion-1B-chinese-v0.1: model is not cached locally and an error occurred while trying to fetch metadata from the Hub. Please check out the root cause in the stacktrace above.
Couldn't connect to the Hub: (ProtocolError('Connection aborted.', ConnectionResetError(10054, '远程主机强迫关闭了一个现有的连接。', None, 10054, None)), '(Request ID: 1eccdd0b-59f2-4a00-85e1-3037766a3b58)').
Will try to load from local cache.
尝试 2/3 失败: Cannot load model IDEA-CCNL/Taiyi-Stable-Diffusion-1B-chinese-v0.1: model is not cached locally and an error occurred while trying to fetch metadata from the Hub. Please check out the root cause in the stacktrace above.
Couldn't connect to the Hub: (ProtocolError('Connection aborted.', ConnectionResetError(10054, '远程主机强迫关闭了一个现有的连接。', None, 10054, None)), '(Request ID: 19c58e5b-6c5a-49c7-bb95-620e9fb22efd)').
Will try to load from local cache.
尝试 3/3 失败: Cannot load model IDEA-CCNL/Taiyi-Stable-Diffusion-1B-chinese-v0.1: model is not cached locally and an error occurred while trying to fetch metadata from the Hub. Please check out the root cause in the stacktrace above.
PS C:\Users\lenovo\Desktop\大一下\ai-self\co-w\Lora_exper2-retry\Lora_exper2> ^C
PS C:\Users\lenovo\Desktop\大一下\ai-self\co-w\Lora_exper2-retry\Lora_exper2>
PS C:\Users\lenovo\Desktop\大一下\ai-self\co-w\Lora_exper2-retry\Lora_exper2>  c:; cd 'c:\Users\lenovo\Desktop\大一下\ai-self\co-w\Lora_exper2-retry\Lora_exper2'; & 'c:\Users\lenovo\.conda\envs\mysd\python.exe' 'c:\Users\lenovo\.vscode\extensions\ms-python.debugpy-2025.6.0-win32-x64\bundled\libs\debugpy\launcher' '49289' '--' 'C:\Users\lenovo\Desktop\大一下\ai-self\co-w\Lora_exper2-retry\Lora_exper2\edited_train.py'
model_index.json: 100%|██████████████████████████████████████| 537/537 [00:00<00:00, 517kB/s]
c:\Users\lenovo\.conda\envs\mysd\lib\site-packages\huggingface_hub\file_download.py:144: UserWarning: `huggingface_hub` cache-system uses symlinks by default to efficiently store duplicated files but your machine does not support them in C:\Users\lenovo\Desktop\大一下\ai-self\co-w\Lora_exper2-retry\Lora_exper2\model_cache\models--IDEA-CCNL--Taiyi-Stable-Diffusion-1B-chinese-v0.1. Caching files will still work but in a degraded version that might require more space on your disk. This warning can be disabled by setting the `HF_HUB_DISABLE_SYMLINKS_WARNING` environment variable. For more details, see https://huggingface.co/docs/huggingface_hub/how-to-cache#limitations.
To support symlinks on Windows, you either need to activate Developer Mode or to run Python as an administrator. In order to activate developer mode, see this article: https://docs.microsoft.com/en-us/windows/apps/get-started/enable-your-device-for-development
  warnings.warn(message)
尝试 1/3 失败: You are trying to load model files of the `variant=fp16`, but no such modeling files are available.
尝试 2/3 失败: You are trying to load model files of the `variant=fp16`, but no such modeling files are available.
尝试 3/3 失败: You are trying to load model files of the `variant=fp16`, but no such modeling files are available.
PS C:\Users\lenovo\Desktop\大一下\ai-self\co-w\Lora_exper2-retry\Lora_exper2> ^C
PS C:\Users\lenovo\Desktop\大一下\ai-self\co-w\Lora_exper2-retry\Lora_exper2>
PS C:\Users\lenovo\Desktop\大一下\ai-self\co-w\Lora_exper2-retry\Lora_exper2>  c:; cd 'c:\Users\lenovo\Desktop\大一下\ai-self\co-w\Lora_exper2-retry\Lora_exper2'; & 'c:\Users\lenovo\.conda\envs\mysd\python.exe' 'c:\Users\lenovo\.vscode\extensions\ms-python.debugpy-2025.6.0-win32-x64\bundled\libs\debugpy\launcher' '49375' '--' 'C:\Users\lenovo\Desktop\大一下\ai-self\co-w\Lora_exper2-retry\Lora_exper2\edited_train.py'
尝试 1/3 失败: You are trying to load model files of the `variant=fp16`, but no such modeling files are available.
尝试 2/3 失败: You are trying to load model files of the `variant=fp16`, but no such modeling files are available.
尝试 3/3 失败: You are trying to load model files of the `variant=fp16`, but no such modeling files are available.
PS C:\Users\lenovo\Desktop\大一下\ai-self\co-w\Lora_exper2-retry\Lora_exper2> ^C
PS C:\Users\lenovo\Desktop\大一下\ai-self\co-w\Lora_exper2-retry\Lora_exper2>
PS C:\Users\lenovo\Desktop\大一下\ai-self\co-w\Lora_exper2-retry\Lora_exper2>  c:; cd 'c:\Users\lenovo\Desktop\大一下\ai-self\co-w\Lora_exper2-retry\Lora_exper2'; & 'c:\Users\lenovo\.conda\envs\mysd\python.exe' 'c:\Users\lenovo\.vscode\extensions\ms-python.debugpy-2025.6.0-win32-x64\bundled\libs\debugpy\launcher' '49435' '--' 'C:\Users\lenovo\Desktop\大一下\ai-self\co-w\Lora_exper2-retry\Lora_exper2\edited_train.py' 
preprocessor_config.json: 100%|██████████████████████████████| 342/342 [00:00<00:00, 169kB/s]
c:\Users\lenovo\.conda\envs\mysd\lib\site-packages\huggingface_hub\file_download.py:144: UserWarning: `huggingface_hub` cache-system uses symlinks by default to efficiently store duplicated files but your machine does not support them in C:\Users\lenovo\Desktop\大一下\ai-self\co-w\Lora_exper2-retry\Lora_exper2\model_cache\models--IDEA-CCNL--Taiyi-Stable-Diffusion-1B-chinese-v0.1. Caching files will still work but in a degraded version that might require more space on your disk. This warning can be disabled by setting the `HF_HUB_DISABLE_SYMLINKS_WARNING` environment variable. For more details, see https://huggingface.co/docs/huggingface_hub/how-to-cache#limitations.
To support symlinks on Windows, you either need to activate Developer Mode or to run Python as an administrator. In order to activate developer mode, see this article: https://docs.microsoft.com/en-us/windows/apps/get-started/enable-your-device-for-development
  warnings.warn(message)
config.json: 100%|██████████████████████████████████████████████| 4.67k/4.67k [00:00<?, ?B/s]
config.json: 100%|██████████████████████████████████████████████████| 884/884 [00:00<?, ?B/s] 
special_tokens_map.json: 100%|██████████████████████████████████████| 186/186 [00:00<?, ?B/s] 
vocab.txt: 100%|███████████████████████████████████████████| 110k/110k [00:00<00:00, 181kB/s] 
config.json: 100%|██████████████████████████████████████████████████| 793/793 [00:00<?, ?B/s] 
tokenizer_config.json: 100%|████████████████████████████████████████| 555/555 [00:00<?, ?B/s] 
config.json: 100%|██████████████████████████████████████████████████| 600/600 [00:00<?, ?B/s] 
scheduler_config.json: 100%|████████████████████████████████████████| 298/298 [00:00<?, ?B/s] 
pytorch_model.bin: 100%|██████████████████████████████████| 409M/409M [01:20<00:00, 5.11MB/s] 
diffusion_pytorch_model.bin: 100%|████████████████████████| 335M/335M [01:38<00:00, 3.41MB/s]
diffusion_pytorch_model.bin: 100%|██████████████████████| 3.44G/3.44G [12:00<00:00, 4.77MB/s] 
Fetching 13 files: 100%|█████████████████████████████████████| 13/13 [12:02<00:00, 55.59s/it] 
Loading pipeline components...:   0%|                                  | 0/6 [00:00<?, ?it/s]An error occurred while trying to fetch ./model_cache\models--IDEA-CCNL--Taiyi-Stable-Diffusion-1B-chinese-v0.1\snapshots\8de752b2bb2f8a951d16654254a3a46568c8584b\unet: Error no file named diffusion_pytorch_model.safetensors found in directory ./model_cache\models--IDEA-CCNL--Taiyi-Stable-Diffusion-1B-chinese-v0.1\snapshots\8de752b2bb2f8a951d16654254a3a46568c8584b\unet.     
Defaulting to unsafe serialization. Pass `allow_pickle=False` to raise an error instead.      
Loading pipeline components...:  67%|█████████████████▎        | 4/6 [00:00<00:00,  4.74it/s]An error occurred while trying to fetch ./model_cache\models--IDEA-CCNL--Taiyi-Stable-Diffusion-1B-chinese-v0.1\snapshots\8de752b2bb2f8a951d16654254a3a46568c8584b\vae: Error no file named diffusion_pytorch_model.safetensors found in directory ./model_cache\models--IDEA-CCNL--Taiyi-Stable-Diffusion-1B-chinese-v0.1\snapshots\8de752b2bb2f8a951d16654254a3a46568c8584b\vae.       
Defaulting to unsafe serialization. Pass `allow_pickle=False` to raise an error instead.      
Loading pipeline components...: 100%|██████████████████████████| 6/6 [00:01<00:00,  5.85it/s]
Expected types for text_encoder: (<class 'transformers.models.clip.modeling_clip.CLIPTextModel'>,), got <class 'transformers.models.bert.modeling_bert.BertModel'>.
You have disabled the safety checker for <class 'diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline'> by passing `safety_checker=None`. Ensure that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered results in services or applications open to the public. Both the diffusers team and Hugging Face strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling it only for use-cases that involve analyzing network behavior or auditing its results. For more information, please have a look at https://github.com/huggingface/diffusers/pull/254 .       
wandb: Currently logged in as: 3922909893 (3922909893-peking-university) to https://api.wandb.ai. Use `wandb login --relogin` to force relogin
wandb: Tracking run with wandb version 0.19.10
wandb: Run data is saved locally in C:\Users\lenovo\Desktop\大一下\ai-self\co-w\Lora_exper2-retry\Lora_exper2\wandb\run-20250504_205608-628a7qmv
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run galactic-fleet-7
wandb:  View project at https://wandb.ai/3922909893-peking-university/lora_finetuning
wandb:  View run at https://wandb.ai/3922909893-peking-university/lora_finetuning/runs/628a7qmv
We strongly recommend passing in an `attention_mask` since your input_ids may be padded. See https://huggingface.co/docs/transformers/troubleshooting#incorrect-output-when-padding-tokens-arent-masked.
You may ignore this warning if your `pad_token_id` (0) is identical to the `bos_token_id` (0), `eos_token_id` (2), or the `sep_token_id` (None), and your input is not padded.
Epoch: 0, Batch: 0, Loss: 0.4093
Epoch: 0, Batch: 10, Loss: 0.1020
Epoch: 0, Batch: 20, Loss: 0.8868
Epoch: 0, Batch: 30, Loss: 1.0155
updated_loss, on 0 epoch
Epoch: 1, Batch: 0, Loss: 0.0902
跳过异常批次 6, Loss: 0.7097
Epoch: 1, Batch: 10, Loss: 0.4725
跳过异常批次 17, Loss: 0.9275
Epoch: 1, Batch: 20, Loss: 0.0882
Epoch: 1, Batch: 30, Loss: 0.3710
updated_loss, on 1 epoch
Epoch: 2, Batch: 0, Loss: 0.4497
Epoch: 2, Batch: 10, Loss: 0.1381
Epoch: 2, Batch: 20, Loss: 0.0598
Epoch: 2, Batch: 30, Loss: 0.1752
跳过异常批次 33, Loss: 1.0660
Epoch: 3, Batch: 0, Loss: 0.3691
Epoch: 3, Batch: 10, Loss: 1.0682
Epoch: 3, Batch: 20, Loss: 0.5333
Epoch: 3, Batch: 30, Loss: 0.0548
Epoch: 4, Batch: 0, Loss: 0.1050
跳过异常批次 2, Loss: 0.5127
Epoch: 4, Batch: 10, Loss: 0.2075
跳过异常批次 18, Loss: 1.1133
Epoch: 4, Batch: 20, Loss: 0.2133
Epoch: 4, Batch: 30, Loss: 0.2324
Epoch: 5, Batch: 0, Loss: 0.0921
跳过异常批次 5, Loss: 0.9211
Epoch: 5, Batch: 10, Loss: 0.4474
跳过异常批次 18, Loss: 1.0279
Epoch: 5, Batch: 20, Loss: 0.2525
Epoch: 5, Batch: 30, Loss: 0.0683
updated_loss, on 5 epoch
Epoch: 6, Batch: 0, Loss: 0.0742
跳过异常批次 1, Loss: 0.3617
跳过异常批次 7, Loss: 0.7584
Epoch: 6, Batch: 10, Loss: 0.0589
Epoch: 6, Batch: 20, Loss: 0.0703
跳过异常批次 25, Loss: 1.0637
Epoch: 6, Batch: 30, Loss: 0.1179
updated_loss, on 6 epoch
Epoch: 7, Batch: 0, Loss: 0.4433
Epoch: 7, Batch: 10, Loss: 0.4047
Epoch: 7, Batch: 20, Loss: 0.2876
Epoch: 7, Batch: 30, Loss: 0.0592
Epoch: 8, Batch: 0, Loss: 0.0978
跳过异常批次 5, Loss: 0.6278
跳过异常批次 8, Loss: 0.9536
Epoch: 8, Batch: 10, Loss: 0.1206
Epoch: 8, Batch: 20, Loss: 0.7995
跳过异常批次 22, Loss: 1.2582
Epoch: 8, Batch: 30, Loss: 0.1818
Epoch: 9, Batch: 0, Loss: 0.0670
跳过异常批次 1, Loss: 0.4065
Epoch: 9, Batch: 10, Loss: 0.0950
跳过异常批次 18, Loss: 0.7940
Epoch: 9, Batch: 20, Loss: 0.0906
Epoch: 9, Batch: 30, Loss: 0.6982