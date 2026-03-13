import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
import math

class SRTProcessor:
    @staticmethod
    def time_to_seconds(time_str):
        """将 00:00:02,340 转换为秒数"""
        h, m, s = time_str.replace(',', '.').split(':')
        return int(h) * 3600 + int(m) * 60 + float(s)

    @staticmethod
    def seconds_to_time(seconds):
        """将秒数转换回 00:00:02,340 格式"""
        if math.isnan(seconds) or seconds < 0:
            seconds = 0
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int(round((seconds - int(seconds)) * 1000))
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    @staticmethod
    def parse_srt(file_path):
        """读取并解析 SRT 文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip().replace('\r\n', '\n')
        
        blocks = content.split('\n\n')
        subs = []
        for block in blocks:
            lines = block.split('\n')
            if len(lines) >= 3:
                # 提取时间轴
                times = re.findall(r'(\d{2}:\d{2}:\d{2},\d{3})', lines[1])
                if len(times) == 2:
                    text = '\n'.join(lines[2:])
                    subs.append({
                        'start': SRTProcessor.time_to_seconds(times[0]),
                        'end': SRTProcessor.time_to_seconds(times[1]),
                        'text': text
                    })
        return subs

    @staticmethod
    def export_srt(subs, output_path):
        """保存为新的 SRT 文件"""
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, sub in enumerate(subs):
                start_str = SRTProcessor.seconds_to_time(sub['start'])
                end_str = SRTProcessor.seconds_to_time(sub['end'])
                f.write(f"{i+1}\n")
                f.write(f"{start_str} --> {end_str}\n")
                f.write(f"{sub['text']}\n\n")

def dtw_align(eng_subs, chn_subs, eng_embeddings, chn_embeddings):
    """核心 DTW 算法：对齐两条时间轴"""
    n, m = len(eng_subs), len(chn_subs)
    
    # 计算所有句子之间的余弦相似度矩阵
    # similarity 越高 (越接近1)，距离越小
    sim_matrix = cosine_similarity(eng_embeddings, chn_embeddings)
    dist_matrix = 1.0 - sim_matrix 
    
    # 初始化 DP 矩阵
    dp = np.full((n + 1, m + 1), np.inf)
    dp[0, 0] = 0
    traceback = np.zeros((n + 1, m + 1), dtype=int)
    
    # 填充 DP 矩阵
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = dist_matrix[i-1, j-1]
            choices = [dp[i-1, j-1], dp[i-1, j], dp[i, j-1]]
            min_prev = min(choices)
            dp[i, j] = cost + min_prev
            traceback[i, j] = choices.index(min_prev) # 0: 匹配, 1: 英文多对一中文, 2: 英文一对多中文
            
    # 回溯寻找最优路径
    i, j = n, m
    path = []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        if traceback[i, j] == 0:
            i -= 1; j -= 1
        elif traceback[i, j] == 1:
            i -= 1
        else:
            j -= 1
            
    # 根据路径重新分配中文字幕时间
    aligned_chn_subs = []
    for c_idx in range(m):
        # 找到这个中文字幕匹配到的所有英文索引
        matched_eng_indices = [p[0] for p in path if p[1] == c_idx]
        if matched_eng_indices:
            first_match = min(matched_eng_indices)
            last_match = max(matched_eng_indices)
            aligned_chn_subs.append({
                'start': eng_subs[first_match]['start'],
                'end': eng_subs[last_match]['end'],
                'text': chn_subs[c_idx]['text']
            })
        else:
            # 万一没匹配上，保留原样（按理说 DTW 必然会匹配）
            aligned_chn_subs.append(chn_subs[c_idx])
            
    return aligned_chn_subs

def main():
    # 1. 设置文件路径 (请替换成你真实的电脑文件路径)
    eng_path = "english.srt"
    chn_path = "chinese.srt"
    out_path = "aligned_chinese.srt"

    print("📄 1. 正在解析 SRT 文件...")
    eng_subs = SRTProcessor.parse_srt(eng_path)
    chn_subs = SRTProcessor.parse_srt(chn_path)
    
    if not eng_subs or not chn_subs:
        print("❌ 错误：字幕文件为空或解析失败！")
        return

    print("🧠 2. 正在加载多语言 AI 模型 (首次运行会自动从 HuggingFace 下载)...")
    # 这个模型专门针对多语言优化，中英对齐效果极佳
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    print(f"🧮 3. 正在计算外文向量 ({len(eng_subs)} 行)...")
    eng_texts = [sub['text'] for sub in eng_subs]
    eng_embeddings = model.encode(eng_texts, show_progress_bar=True)

    print(f"🧮 4. 正在计算中文向量 ({len(chn_subs)} 行)...")
    chn_texts = [sub['text'] for sub in chn_subs]
    chn_embeddings = model.encode(chn_texts, show_progress_bar=True)

    print("⚡️ 5. 正在运行 DTW 语义对齐算法...")
    aligned_chn = dtw_align(eng_subs, chn_subs, eng_embeddings, chn_embeddings)

    print("💾 6. 正在导出新文件...")
    SRTProcessor.export_srt(aligned_chn, out_path)
    
    print(f"🎉 大功告成！完美对齐的中文字幕已保存至: {out_path}")

if __name__ == "__main__":
    main()