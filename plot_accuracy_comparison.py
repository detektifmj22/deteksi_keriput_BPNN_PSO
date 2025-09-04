import matplotlib.pyplot as plt
import numpy as np

def plot_accuracy_comparison(baseline_acc, optimized_acc, save_path='output/accuracy_comparison.png'):
    labels = ['Baseline BPNN', 'Optimized BPNN-PSO']
    accuracies = [baseline_acc, optimized_acc]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, accuracies, color=['gray', 'blue'])
    ax.set_ylim([0, 1])
    ax.set_ylabel('Accuracy')
    ax.set_title('Perbandingan Akurasi Model (Gambar 6)')
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2%}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
    plt.savefig(save_path)
    plt.close()
    print(f"Grafik perbandingan akurasi disimpan di {save_path}")

if __name__ == "__main__":
    # Contoh data akurasi, sesuaikan dengan hasil Anda
    baseline_accuracy = 0.715  # misal 71.5%
    optimized_accuracy = 0.7955  # hasil BPNN-PSO
    plot_accuracy_comparison(baseline_accuracy, optimized_accuracy)
