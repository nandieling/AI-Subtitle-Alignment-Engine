import sys
import os
import numpy as np
import re
import math
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QFileDialog)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# SRT 处理引擎
# ==========================================
class SRTProcessor:
    @staticmethod
    def time_to_seconds(time_str):
        h, m, s = time_str.replace(',', '.').split(':')
        return int(h) * 3600 + int(m) * 60 + float(s)

    @staticmethod
    def seconds_to_time(seconds):
        if math.isnan(seconds) or seconds < 0: seconds = 0
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int(round((seconds - int(seconds)) * 1000))
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    @staticmethod
    def parse_srt(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip().replace('\r\n', '\n')
        blocks = content.split('\n\n')
        subs = []
        for block in blocks:
            lines = block.split('\n')
            if len(lines) >= 3:
                times = re.findall(r'(\d{2}:\d{2}:\d{2},\d{3})', lines[1])
                if len(times) == 2:
                    subs.append({
                        'start': SRTProcessor.time_to_seconds(times[0]),
                        'end': SRTProcessor.time_to_seconds(times[1]),
                        'text': '\n'.join(lines[2:])
                    })
        return subs, None # None 是为了和 ASS 返回格式保持一致

    @staticmethod
    def export_srt(subs, original_lines, output_path):
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, sub in enumerate(subs):
                start_str = SRTProcessor.seconds_to_time(sub['start'])
                end_str = SRTProcessor.seconds_to_time(sub['end'])
                f.write(f"{i+1}\n{start_str} --> {end_str}\n{sub['text']}\n\n")

# ==========================================
# ASS 处理引擎 (新增核心功能)
# ==========================================
class ASSProcessor:
    @staticmethod
    def ass_time_to_seconds(time_str):
        # ASS 格式: H:MM:SS.cs (例如 0:02:41.03)
        h, m, s = time_str.split(':')
        return int(h) * 3600 + int(m) * 60 + float(s)

    @staticmethod
    def seconds_to_ass_time(seconds):
        if math.isnan(seconds) or seconds < 0: seconds = 0
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        cs = int(round((seconds - int(seconds)) * 100)) # ASS 是百分之一秒
        return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

    @staticmethod
    def clean_ass_text(text):
        # 剥离所有 { \an8 } 等 ASS 特效标签，防止干扰 AI 语义理解
        return re.sub(r'\{[^}]*\}', '', text)

    @staticmethod
    def parse_ass(file_path):
        subs = []
        original_lines = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                original_lines.append(line)
                if line.startswith('Dialogue:'):
                    # Dialogue: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
                    parts = line.strip().split(',', 9)
                    if len(parts) >= 10:
                        start_time = ASSProcessor.ass_time_to_seconds(parts[1])
                        end_time = ASSProcessor.ass_time_to_seconds(parts[2])
                        raw_text = parts[9]
                        clean_text = ASSProcessor.clean_ass_text(raw_text)
                        
                        subs.append({
                            'start': start_time,
                            'end': end_time,
                            'text': clean_text,          # 纯净版喂给AI
                            'original_line_index': len(original_lines) - 1,
                            'parts_before_time': parts[0],
                            'parts_after_time': parts[3:] # 包含原来的原始文本和特效
                        })
        return subs, original_lines

    @staticmethod
    def export_ass(aligned_subs, original_lines, output_path):
        for sub in aligned_subs:
            idx = sub['original_line_index']
            start_str = ASSProcessor.seconds_to_ass_time(sub['start'])
            end_str = ASSProcessor.seconds_to_ass_time(sub['end'])
            after_time_str = ','.join(sub['parts_after_time'])
            # 原汁原味拼装回去
            new_line = f"{sub['parts_before_time']},{start_str},{end_str},{after_time_str}\n"
            original_lines[idx] = new_line
            
        with open(output_path, 'w', encoding='utf-8') as f:
            f.writelines(original_lines)

# ==========================================
# 核心算法区 (DTW对齐 + 异常时长修剪)
# ==========================================
def dtw_align(eng_subs, chn_subs, eng_embeddings, chn_embeddings):
    n, m = len(eng_subs), len(chn_subs)
    dist_matrix = 1.0 - cosine_similarity(eng_embeddings, chn_embeddings)
    
    dp = np.full((n + 1, m + 1), np.inf)
    dp[0, 0] = 0
    traceback = np.zeros((n + 1, m + 1), dtype=int)
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = dist_matrix[i-1, j-1]
            choices = [dp[i-1, j-1], dp[i-1, j], dp[i, j-1]]
            min_prev = min(choices)
            dp[i, j] = cost + min_prev
            traceback[i, j] = choices.index(min_prev)
            
    i, j, path = n, m, []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        if traceback[i, j] == 0: i -= 1; j -= 1
        elif traceback[i, j] == 1: i -= 1
        else: j -= 1
            
    MAX_DURATION = 6.0 
    aligned_chn_subs = []
    
    for c_idx in range(m):
        matched_eng = [p[0] for p in path if p[1] == c_idx]
        
        # 巧妙使用 copy，无论 SRT 还是 ASS 的元数据都能被完美保留传递
        new_sub = chn_subs[c_idx].copy() 
        
        if matched_eng:
            start_time = eng_subs[min(matched_eng)]['start']
            end_time = eng_subs[max(matched_eng)]['end']
        else:
            start_time = chn_subs[c_idx]['start']
            end_time = chn_subs[c_idx]['end']
            
        # 异常长字幕一刀切断
        if (end_time - start_time) > MAX_DURATION:
            end_time = start_time + MAX_DURATION
            
        new_sub['start'] = start_time
        new_sub['end'] = end_time
        aligned_chn_subs.append(new_sub)
        
    return aligned_chn_subs

# ==========================================
# 后台工作线程
# ==========================================
class AlignWorker(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(str, bool)

    def __init__(self, eng_path, chn_path, out_path):
        super().__init__()
        self.eng_path = eng_path
        self.chn_path = chn_path
        self.out_path = out_path

    def parse_file(self, path):
        ext = os.path.splitext(path)[1].lower()
        if ext == '.ass':
            return ASSProcessor.parse_ass(path)
        else:
            return SRTProcessor.parse_srt(path)

    def run(self):
        try:
            self.progress.emit("📄 1/6 正在智能解析字幕文件...")
            eng_subs, _ = self.parse_file(self.eng_path)
            chn_subs, chn_original_lines = self.parse_file(self.chn_path)

            self.progress.emit("🧠 2/6 正在加载 AI 模型 (这可能需要几秒钟)...")
            model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

            self.progress.emit(f"🧮 3/6 正在转换外文向量 ({len(eng_subs)} 行)...")
            eng_embeddings = model.encode([s['text'] for s in eng_subs])

            self.progress.emit(f"🧮 4/6 正在转换中文向量 ({len(chn_subs)} 行)...")
            chn_embeddings = model.encode([s['text'] for s in chn_subs])

            self.progress.emit("⚡️ 5/6 正在运行 DTW 语义对齐算法...")
            aligned_chn = dtw_align(eng_subs, chn_subs, eng_embeddings, chn_embeddings)

            self.progress.emit("💾 6/6 正在封装并导出新文件...")
            out_ext = os.path.splitext(self.out_path)[1].lower()
            if out_ext == '.ass':
                ASSProcessor.export_ass(aligned_chn, chn_original_lines, self.out_path)
            else:
                SRTProcessor.export_srt(aligned_chn, None, self.out_path)

            self.finished.emit(f"✅ 大功告成！文件已保存:\n{os.path.basename(self.out_path)}", True)
        except Exception as e:
            self.finished.emit(f"❌ 发生错误: {str(e)}", False)

# ==========================================
# 优雅的 UI 界面区
# ==========================================
class DropZone(QLabel):
    file_dropped = pyqtSignal(str)

    def __init__(self, title):
        super().__init__()
        self.title = title
        self.file_path = None
        self.setText(f"点击或拖拽\n{title}\n(.srt 或 .ass)")
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(250, 150)
        self.setAcceptDrops(True)
        self.setCursor(Qt.PointingHandCursor)
        self.set_default_style()

    def set_default_style(self):
        self.setStyleSheet("border: 2px dashed #aaa; border-radius: 10px; background-color: #f9f9f9; font-size: 16px; color: #666;")

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            file_path = event.mimeData().urls()[0].toLocalFile().lower()
            if file_path.endswith('.srt') or file_path.endswith('.ass'):
                event.accept()
                self.setStyleSheet("border: 2px dashed #4a90e2; border-radius: 10px; background-color: #e6f3ff; font-size: 16px; color: #4a90e2;")
                return
        event.ignore()

    def dragLeaveEvent(self, event):
        if not self.file_path: self.set_default_style()

    def dropEvent(self, event):
        file_path = event.mimeData().urls()[0].toLocalFile()
        self.process_file(file_path)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            file_path, _ = QFileDialog.getOpenFileName(
                self, f"选择 {self.title}", "", "Subtitles (*.srt *.ass)"
            )
            if file_path:
                self.process_file(file_path)
                
    def process_file(self, file_path):
        self.file_path = file_path
        self.setText(f"✅ {self.title} 已加载\n\n{os.path.basename(file_path)}")
        self.setStyleSheet("border: 2px solid #4a90e2; border-radius: 10px; background-color: #e6f3ff; font-size: 14px; color: #333;")
        self.file_dropped.emit(file_path)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI 字幕对齐引擎 Pro")
        self.resize(650, 450)
        
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(20)

        title = QLabel("AI 字幕语义对齐")
        title.setFont(QFont("Arial", 28, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        subtitle = QLabel("支持 SRT 与 ASS 混搭，自动剥离并保护特效代码")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("color: #666; font-size: 14px;")
        layout.addWidget(subtitle)

        drop_layout = QHBoxLayout()
        drop_layout.setSpacing(30)
        self.eng_zone = DropZone("外文字幕 (基准时间)")
        self.eng_zone.file_dropped.connect(self.check_ready)
        self.chn_zone = DropZone("中文字幕 (待调整)")
        self.chn_zone.file_dropped.connect(self.check_ready)
        drop_layout.addWidget(self.eng_zone)
        drop_layout.addWidget(self.chn_zone)
        layout.addLayout(drop_layout)

        self.status_label = QLabel("请放入字幕文件 (支持 .srt / .ass)...")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("color: #666; font-size: 14px;")
        layout.addWidget(self.status_label)

        self.start_btn = QPushButton("选择保存位置并开始对齐")
        self.start_btn.setMinimumHeight(45)
        self.start_btn.setFont(QFont("Arial", 14, QFont.Bold))
        self.start_btn.setEnabled(False)
        self.start_btn.clicked.connect(self.start_process)
        self.start_btn.setStyleSheet("""
            QPushButton { background-color: #4a90e2; color: white; border-radius: 8px; }
            QPushButton:disabled { background-color: #ccc; }
        """)
        layout.addWidget(self.start_btn)

    def check_ready(self):
        if self.eng_zone.file_path and self.chn_zone.file_path:
            self.start_btn.setEnabled(True)
            self.status_label.setText("文件已就绪，点击按钮开始对齐")

    def start_process(self):
        # 智能判断：如果放入的是 ASS，默认保存的也是 ASS
        default_ext = os.path.splitext(self.chn_zone.file_path)[1].lower()
        if not default_ext: default_ext = ".srt"
        
        filter_str = f"Subtitles (*{default_ext})"
        out_path, _ = QFileDialog.getSaveFileName(self, "保存对齐后的中文字幕", f"Aligned_Chinese{default_ext}", filter_str)
        if not out_path: return

        self.start_btn.setEnabled(False)
        self.eng_zone.setEnabled(False)
        self.chn_zone.setEnabled(False)
        
        self.worker = AlignWorker(self.eng_zone.file_path, self.chn_zone.file_path, out_path)
        self.worker.progress.connect(self.status_label.setText)
        self.worker.finished.connect(self.task_finished)
        self.worker.start()

    def task_finished(self, msg, success):
        self.status_label.setText(msg)
        self.status_label.setStyleSheet("color: green; font-size: 14px;" if success else "color: red; font-size: 14px;")
        self.start_btn.setEnabled(True)
        self.eng_zone.setEnabled(True)
        self.chn_zone.setEnabled(True)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())