# 离线包清单

本目录用于在无外网环境中恢复 TaskFoundry 运行所需的基础资产。包内通常包括：

- LLaMA-Factory Docker 镜像
- 项目代码
- 训练配置
- ModelScope 模型
- 示例数据集或业务方自备数据

## 文件说明

- `llamafactory-0.9.4.tar`
  - `hiyouga/llamafactory:0.9.4` 的 Docker 镜像离线包

- `taskfoundry_assets.tar.zst`
  - 项目代码、脚本、配置、模型和数据等资产
  - 建议不要把未完成训练的 checkpoint、日志和临时缓存混进基础包

- `offline_restore.sh`
  - 离线恢复脚本
  - 会加载 Docker 镜像、解压资产，并检查关键文件

- `SHA256SUMS`
  - 离线包校验文件

## 推荐恢复步骤

```bash
cd /root/taskfoundry_bundle
bash offline_restore.sh
```

恢复完成后：

```bash
cd /root/taskfoundry
./run_all.sh
```

## 打包建议

建议把交付内容拆成两类压缩包：

1. 基础运行包
   - Docker 镜像
   - 项目代码
   - 模型和原始数据

2. 训练结果包
   - `outputs/teacher-lora/`
   - `outputs/student-distill-lora/`
   - 日志
   - 评测结果

这样更利于版本管理，也更适合后续重复复用。
