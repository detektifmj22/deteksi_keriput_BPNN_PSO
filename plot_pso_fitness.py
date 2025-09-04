import matplotlib.pyplot as plt

def plot_fitness_convergence(fitness_history, output_path='output/pso_fitness_convergence.png'):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(fitness_history) + 1), fitness_history, marker='o', linestyle='-')
    plt.title('PSO Fitness Convergence Curve Over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Best Fitness Score')
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()
    print(f"PSO fitness convergence curve saved to {output_path}")

if __name__ == "__main__":
    # Example usage: load fitness_history from a file or PSO run
    import numpy as np
    # Dummy data for testing
    fitness_history = np.random.rand(30).tolist()
    plot_fitness_convergence(fitness_history)
