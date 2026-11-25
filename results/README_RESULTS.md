# 实验结果文件说明

本目录包含 54 名被试严格跨天（Day 1 训练 → Day 2 测试）完整基准评测输出：
- `benchmark_report.md`：SSVEP 与 P300 全量评测结果汇总报告
- `ssvep_summary.csv` / `ssvep_per_subject.csv`：CCA、FBCCA、eTRCA 在 0.5s–2.0s 六个窗长下的准确率与 ITR
- `ssvep_harmonic_snr.csv`：Oz 导联刺激基频及 2/3 次倍频谐波 SNR（dB）统计
- `p300_summary.csv` / `p300_per_subject.csv`：xDAWN+黎曼切空间 与 加权交叉熵 EEGNet 的跨天 AUC、Balanced Accuracy、F1 及 1–5 轮叠加拼写准确率
- `p300_erp_stats.csv`：中线/顶枕导联 250–450 ms ERP 波幅与潜伏期统计
- `experiment_summary.json`：结构化汇总指标
