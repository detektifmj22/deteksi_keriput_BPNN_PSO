import matplotlib.pyplot as plt

def plot_bpnn_architecture(input_size, hidden_layers, output_size, save_path='output/bpnn_architecture.png'):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis('off')

    layer_sizes = [input_size] + list(hidden_layers) + [output_size]
    n_layers = len(layer_sizes)

    v_spacing = 1.0 / float(max(layer_sizes))
    h_spacing = 1.0 / float(n_layers - 1)

    # Nodes with ellipsis for large layers
    for i, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing * (layer_size - 1) / 2.0 + 0.1
        if layer_size > 10:
            # Draw first 5 neurons
            for j in range(5):
                circle = plt.Circle((i * h_spacing + 0.1, layer_top - j * v_spacing), v_spacing / 4,
                                    color='w', ec='k', zorder=4)
                ax.add_artist(circle)
            # Draw ellipsis
            ax.text(i * h_spacing + 0.1, layer_top - 5.5 * v_spacing, '...', ha='center', va='center', fontsize=12)
            # Draw last 5 neurons
            for j in range(layer_size - 5, layer_size):
                circle = plt.Circle((i * h_spacing + 0.1, layer_top - j * v_spacing), v_spacing / 4,
                                    color='w', ec='k', zorder=4)
                ax.add_artist(circle)
        else:
            for j in range(layer_size):
                circle = plt.Circle((i * h_spacing + 0.1, layer_top - j * v_spacing), v_spacing / 4,
                                    color='w', ec='k', zorder=4)
                ax.add_artist(circle)

    # Edges
    for i, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layer_top_a = v_spacing * (layer_size_a - 1) / 2.0 + 0.1
        layer_top_b = v_spacing * (layer_size_b - 1) / 2.0 + 0.1
        # Connect first 5 neurons
        for j in range(min(5, layer_size_a)):
            for k in range(min(5, layer_size_b)):
                line = plt.Line2D([i * h_spacing + 0.1, (i + 1) * h_spacing + 0.1],
                                  [layer_top_a - j * v_spacing, layer_top_b - k * v_spacing], c='k')
                ax.add_artist(line)
        # Connect last 5 neurons
        for j in range(max(0, layer_size_a - 5), layer_size_a):
            for k in range(max(0, layer_size_b - 5), layer_size_b):
                line = plt.Line2D([i * h_spacing + 0.1, (i + 1) * h_spacing + 0.1],
                                  [layer_top_a - j * v_spacing, layer_top_b - k * v_spacing], c='k')
                ax.add_artist(line)
        # Draw ellipsis in the middle of edges
        ax.text((i + 0.5) * h_spacing + 0.1, 0.5, '...', ha='center', va='center', fontsize=14)

    ax.set_title('Skema Arsitektur BPNN Hasil Optimasi')
    plt.savefig(save_path)
    plt.close()
    print(f"Skema arsitektur BPNN disimpan di {save_path}")

if __name__ == "__main__":
    # Contoh penggunaan
    input_size = 6  # jumlah fitur input, sesuaikan dengan dataset Anda
    hidden_layers = (50, 37)  # hasil optimasi PSO
    output_size = 2  # jumlah kelas output, sesuaikan dengan dataset Anda
    plot_bpnn_architecture(input_size, hidden_layers, output_size)
