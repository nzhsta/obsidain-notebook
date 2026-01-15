
- 官方文档地址：[基于人类反馈的强化学习 (RLHF) — OpenRLHF 0.8 文档 - OpenRLHF 框架](https://openrlhf.cn/en/latest/rl.html)
- 代码仓库：[GitHub - OpenRLHF](https://github.com/OpenRLHF/OpenRLHF/tree/main)

# 1 SFT
## 1.1 基础参数
1. logging_steps
2. **input_key**：JSON dataset key 
3. **packing_samples**：🚀packing SFT samples without CrossAttention（打包SFT样本时不使用CrossAttention​）
	- packing的好处
		- **减少padding浪费**：传统方法中，短样本需要padding到max_length，造成大量计算浪费
		- **提高GPU利用率**：将多个短样本打包到一个batch中，充分利用序列长度
		- **加速训练**：可以提升 2-5倍 的训练吞吐量
	- without CrossAttention
			- 防止样本间信息泄露
			- 梯度来自多个样本的混合信号 → 更新方向可能冲突
			- 推理时实际情况：模型处理的是独立的输入序列
	- ❗ 谨慎使用（可能需CrossAttention）
		- **长文档摘要**：文档被分割成多个块
		- **书籍续写**：章节间有强连续性
		- **多模态任务**：文本和图像交错
4. flash_attention
   **FlashAttention**​ 是一种革命性的注意力机制优化技术，专门解决传统注意力机制在大模型训练中的**内存和计算瓶颈**问题。
5. local_rank for deepspeed
   local_rank是 DeepSpeed分布式训练中的一个关键环境变量，用于标识当前进程在单个节点（机器）内的相对位置
	```python
	# 分布式训练层级
	# 假设你有4个节点（机器），每个节点有8个GPU
	
	# 全局视角
	world_size = 32  # 总GPU数 = 4节点 × 8GPU
	rank = 0-31      # 全局排名，每个GPU的唯一ID
	
	# 节点视角
	node_0: GPU[0,1,2,3,4,5,6,7]  # local_rank = 0-7
	node_1: GPU[8,9,10,11,12,13,14,15]  # local_rank = 0-7
	node_2: GPU[16,17,18,19,20,21,22,23]  # local_rank = 0-7
	node_3: GPU[24,25,26,27,28,29,30,31]  # local_rank = 0-7
	
	
	# 分布式训练的三个rank
import os

# 2 local_rank: 节点内GPU编号 (0-7)
local_rank = int(os.environ["LOCAL_RANK"])  # 0,1,2,...,7

# 3 global_rank: 全局GPU编号 (0-31)
global_rank = int(os.environ["RANK"])       # 0-31

# 4 world_size: 总GPU数量
world_size = int(os.environ["WORLD_SIZE"])  # 32
	```


