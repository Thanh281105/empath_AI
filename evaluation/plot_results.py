import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import argparse

def plot_results(csv_path=None):
    if not csv_path:
        files = glob.glob("evaluation/results/summary_*.csv")
        if not files:
            print("❌ Không tìm thấy file summary CSV nào trong evaluation/results/")
            return
        csv_path = max(files, key=os.path.getctime)
    
    print(f"📊 Đang vẽ biểu đồ từ dữ liệu: {csv_path}")
    df = pd.read_csv(csv_path)

    # Rút gọn tên kiến trúc cho dễ nhìn trên trục X
    name_map = {
        "Req1 | Groq base (no RAG)": "Req1 (Groq)",
        "Req2 | Vertex fine-tuned (no RAG)": "Req2 (Vertex FT)",
        "Req3 | Groq base + RAG": "Req3 (Groq + RAG)",
        "Req4 | Vertex fine-tuned + RAG": "Req4 (Vertex FT + RAG)"
    }
    df['Architecture'] = df['Architecture'].map(lambda x: name_map.get(x, x))

    # Cài đặt style hiện đại cho seaborn
    sns.set_theme(style="whitegrid")

    # ==========================================
    # 1. Biểu đồ chất lượng ngôn ngữ và truy xuất (BLEU, ROUGE, BERTScore, Recall@5)
    # ==========================================
    metrics = ['BLEU', 'ROUGE-L', 'BERTScore', 'Recall@5']
    df_metrics = df.melt(id_vars='Architecture', value_vars=metrics, var_name='Metric', value_name='Score')
    # Chuyển đổi sang số, những chữ 'N/A' hoặc 'None' sẽ thành NaN
    df_metrics['Score'] = pd.to_numeric(df_metrics['Score'], errors='coerce')

    plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=df_metrics, x='Metric', y='Score', hue='Architecture', palette='viridis')
    plt.title('So sánh Chất lượng Phản hồi & Truy xuất (BLEU, ROUGE-L, BERTScore, Recall@5)', fontsize=15, pad=15, fontweight='bold')
    plt.ylim(0, max(df_metrics['Score'].max() + 10, 100)) # Mở rộng trục Y
    plt.ylabel('Điểm (0-100)', fontsize=12)
    plt.xlabel('Chỉ số đánh giá', fontsize=12)
    plt.legend(title='Kiến trúc (Model)', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Hiển thị số liệu trực tiếp trên từng cột bar, bỏ qua NaN (ví dụ: mô hình không có RAG)
    for container in ax.containers:
        labels = [f"{v.get_height():.1f}" if pd.notna(v.get_height()) and v.get_height() > 0 else "" for v in container]
        ax.bar_label(container, labels=labels, padding=3, fontsize=10)

    plt.tight_layout()
    out_path1 = "evaluation/results/plot_quality.png"
    plt.savefig(out_path1, dpi=300)
    print(f"✅ Đã lưu: {out_path1}")

    # ==========================================
    # 2. Biểu đồ Tốc độ phản hồi (Latency)
    # ==========================================
    plt.figure(figsize=(10, 5))
    ax2 = sns.barplot(data=df, x='Architecture', y='Avg latency (s)', hue='Architecture', palette='rocket', legend=False)
    plt.title('So sánh Thời gian phản hồi (Avg Latency)', fontsize=15, pad=15, fontweight='bold')
    plt.ylabel('Thời gian (giây)', fontsize=12)
    plt.xlabel('Kiến trúc (Model)', fontsize=12)
    
    # Hiển thị số liệu trên cột
    for container in ax2.containers:
        ax2.bar_label(container, fmt='%.1fs', padding=3, fontsize=11)

    plt.tight_layout()
    out_path2 = "evaluation/results/plot_latency.png"
    plt.savefig(out_path2, dpi=300)
    print(f"✅ Đã lưu: {out_path2}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", help="Đường dẫn đến file summary CSV (nếu không truyền sẽ lấy file mới nhất)")
    args = parser.parse_args()
    plot_results(args.csv)
