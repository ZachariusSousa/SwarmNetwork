import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm  # Optional: for progress bar

# --- Reproducibility ---
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# --- Device Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# --- Hyperparameters ---
input_dim = 28 * 28  # MNIST image size (flattened)
latent_dim = 50  # Dimension of latent space
num_classes = 10  # Digits 0-9
population_size = 5000
num_generations = 200
selection_size = 10  # Top individuals for reproduction
mutation_std_initial = 0.02  # Initial mutation noise
mutation_decay = 0.9999  # Decay rate for mutation noise per generation
num_train_samples = 4000  # Subset for training evaluations
num_test_samples = 500  # Subset for testing evaluations

# --- Data Preparation ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))  # Flatten the image
])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

x_train = train_dataset.data[:num_train_samples].float().view(-1, input_dim) / 255.0
y_train = train_dataset.targets[:num_train_samples]
x_test = test_dataset.data[:num_test_samples].float().view(-1, input_dim) / 255.0
y_test = test_dataset.targets[:num_test_samples]

x_train, y_train = x_train.to(device), y_train.to(device)
x_test, y_test = x_test.to(device), y_test.to(device)


# --- Define the Individual Classifier ---
class Individual:
    def __init__(self, mutation_std):
        self.mutation_std = mutation_std
        self.projection = torch.randn(latent_dim, input_dim, device=device) * 0.1
        self.centroids = torch.randn(num_classes, latent_dim, device=device) * 0.1

    def predict(self, X):
        with torch.no_grad():
            latent = X.matmul(self.projection.t())
            distances = torch.sum((latent.unsqueeze(1) - self.centroids.unsqueeze(0)) ** 2, dim=2)
            predictions = torch.argmin(distances, dim=1)
        return predictions

    def evaluate(self, X, y):
        with torch.no_grad():
            preds = self.predict(X)
            accuracy = torch.mean((preds == y).float()).item()
        return accuracy

    def mutate(self):
        child = Individual(self.mutation_std)
        child.projection = self.projection + torch.randn_like(self.projection) * self.mutation_std
        child.centroids = self.centroids + torch.randn_like(self.centroids) * self.mutation_std
        return child


# --- Main Evolutionary Loop ---
def main():
    population = [Individual(mutation_std_initial) for _ in range(population_size)]
    best_fitness_history = []
    best_individual = None
    current_mutation_std = mutation_std_initial

    for generation in range(num_generations):
        # Evaluate fitness for each individual
        fitness_scores = [ind.evaluate(x_train, y_train) for ind in population]
        fitness_scores = np.array(fitness_scores)
        best_idx = np.argmax(fitness_scores)
        best_fitness = fitness_scores[best_idx]
        best_fitness_history.append(best_fitness)
        best_individual = population[best_idx]

        print(f"Generation {generation + 1:02d} | Best training accuracy: {best_fitness:.4f}")

        # Select the top individuals (elitism)
        top_indices = np.argsort(fitness_scores)[-selection_size:]
        selected = [population[i] for i in top_indices]

        new_population = [best_individual]  # Elitism: carry forward the best individual

        # Generate new offspring through mutation
        while len(new_population) < population_size:
            parent = np.random.choice(selected)
            child = parent.mutate()
            new_population.append(child)

        population = new_population

        # Decay mutation rate to fine-tune later generations
        current_mutation_std *= mutation_decay
        for ind in population:
            ind.mutation_std = current_mutation_std

    # Evaluate the best individual on test data
    test_accuracy = best_individual.evaluate(x_test, y_test)
    print(f"\nFinal Test Accuracy: {test_accuracy:.4f}")

    # Plot training accuracy over generations
    plt.figure(figsize=(8, 4))
    plt.plot(np.arange(1, num_generations + 1), best_fitness_history, marker='o')
    plt.title("Best Training Accuracy Over Generations")
    plt.xlabel("Generation")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.show()

    # Save best individual's parameters
    torch.save({'projection': best_individual.projection,
                'centroids': best_individual.centroids}, 'best_individual.pth')
    print("Best individual parameters saved to best_individual.pth")


if __name__ == "__main__":
    main()