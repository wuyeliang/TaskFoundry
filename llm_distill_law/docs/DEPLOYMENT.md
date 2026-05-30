# TaskFoundry 部署文档

## 1. 目标

TaskFoundry 用于在离线或受限网络环境中完成两阶段 LoRA 蒸馏：

1. 将垂直领域数据整理为 Alpaca 格式
2. 用较大的 teacher 模型做第一阶段 LoRA SFT
3. 用 teacher 为蒸馏 prompt 生成 response-based 对齐数据
4. 用这些对齐数据训练较小的 student 模型
5. 使用 LLaMA-Factory WebChat 做交互验证

## 2. 环境要求

- Linux 主机
- NVIDIA 驱动可用
- Docker 可用
- Docker 支持 `--gpus all`
- 本地已有或可离线导入 `hiyouga/llamafactory:0.9.4`

推荐目录：

```text
/root/taskfoundry
```

## 3. 数据准备

项目自带的下载脚本会从 ModelScope 拉取：

- teacher 基座模型
- student 基座模型
- 一个可运行的示例数据集

如果你要切换成自己的领域数据集，通常只需要：

1. 修改 `scripts/download_modelscope_assets.py` 的默认数据集参数，或直接在运行时传参
2. 调整 `scripts/prepare_domain_dataset.py` 的系统提示词和过滤规则
3. 更新 `data/dataset_info.json` 中的数据文件映射

转换后的默认输出：

- `data/domain_train_alpaca.json`
- `data/distill_prompts.json`
- `data/domain_teacher_aligned.json`

## 4. 训练配置

关键配置文件：

- `configs/qwen3_teacher_lora_sft.yaml`
- `configs/qwen3_student_distill_lora_sft.yaml`

建议先确认这几项：

- `model_name_or_path`
- `dataset`
- `cutoff_len`
- `per_device_train_batch_size`
- `gradient_accumulation_steps`
- `gradient_checkpointing`
- `output_dir`

当前模板默认值偏向高显存单卡。如果显存更紧张，优先降低：

- `per_device_train_batch_size`
- `per_device_eval_batch_size`
- `cutoff_len`

并保留：

- `gradient_checkpointing: true`

## 5. 执行方式

进入项目目录：

```bash
cd /root/taskfoundry
chmod +x run_all.sh run_chat.sh
```

启动完整流水线：

```bash
./run_all.sh
```

该脚本依次执行：

1. 下载模型和示例数据
2. 生成 Alpaca 训练数据和蒸馏 prompts
3. 训练 teacher LoRA
4. 生成 teacher 对齐数据
5. 训练 student LoRA

## 6. 交互验证

训练完成后：

```bash
cd /root/taskfoundry
./run_chat.sh
```

默认加载：

- `/workspace/modelscope/models/Qwen__Qwen3-1.7B`
- `/workspace/outputs/student-distill-lora`

如需验证 teacher：

```bash
MODEL=/workspace/modelscope/models/Qwen__Qwen3-4B-Instruct-2507 \
ADAPTER=/workspace/outputs/teacher-lora \
./run_chat.sh
```

## 7. 恢复与续训

若训练中断，可使用 LLaMA-Factory 的 checkpoint 续训。例如：

```bash
docker run --rm --gpus all --ipc=host \
  -v /root/taskfoundry:/workspace \
  -w /app \
  hiyouga/llamafactory:0.9.4 \
  llamafactory-cli train /workspace/configs/qwen3_teacher_lora_sft.yaml \
  resume_from_checkpoint=/workspace/outputs/teacher-lora/checkpoint-XXXX
```

## 8. 交付建议

建议把最终交付物分成两层：

1. 基础离线包
   - Docker 镜像
   - 项目代码
   - 模型与原始数据

2. 训练结果包
   - `outputs/teacher-lora/`
   - `outputs/student-distill-lora/`
   - 训练日志
   - 评测结果

这样可以把“可复用工程模板”和“某次具体实验产物”分开管理。
