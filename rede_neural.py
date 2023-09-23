import time
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

class MLP():

    def __init__(self, solver, alpha, neurons, verbose:bool, tol, epochs) -> MLPClassifier:
        self.solver = solver
        self.alpha = alpha
        self.neurons = neurons
        self.verbose = verbose
        self.tol = tol
        self._total_time = 0
        self.model = MLPClassifier(solver = self.solver, alpha = self.alpha, hidden_layer_sizes = self.neurons, tol = self.tol, random_state = 1, max_iter = epochs)
    
    def train_model(self, X_train, y_train):
        epochs = self.model.max_iter
        iteration_times = np.empty(epochs, dtype = float)
        classes = np.unique(y_train)
        
        start = time.time()
        for i in range(epochs):
            start_time = time.time()
            self.model.partial_fit(X_train, y_train, classes = classes)
            end_time = time.time()
            if(self.verbose):
                print(F'ÉPOCA {i + 1}')
                print(f'LOSS: {self.get_current_loss()}')
                print(f'DURAÇÃO: {end_time - start_time:.2f} SEGUNDOS\n')
            
            iteration_times[i] = end_time - start_time

        if(self.verbose):
            print('Fim do treinamento\n')    

        self._interations_time = iteration_times
        self._total_time = time.time() - start

        return iteration_times

    def get_total_time(self):
        return self._total_time
    
    def get_mean_time(self):
        return np.mean(self._interations_time)
    
    def get_loss_curve(self):
        return self.model.loss_curve_
    
    def get_current_loss(self):
        return self.model.loss_

    def get_neurons(self):
        return self.neurons

def make_plot(results):
    num_plots = len(results)

    fig, axes = plt.subplots(1, num_plots, figsize=(15, 5))

    for i, result in enumerate(results):
        color = "#{:02x}{:02x}{:02x}".format(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        ax = axes[i]
        ax.plot(result[0], label=f"Modelo {result[1]}\nLoss atual: {result[2]:.5f}\nDuração média entre treinamentos: {result[3]:.2f}s\nDuração total: {result[4]:.2f}s", color=color)
        ax.set_xlabel('Época', weight='bold')
        ax.set_ylabel('Função de Perda (Loss)', weight='bold')
        ax.set_title('Evolução da função de perda', weight='bold')
        ax.legend(loc='upper right', title='Informações\n', title_fontsize='large')

    plt.tight_layout()

    combined_fig, combined_ax = plt.subplots(figsize=(15, 5))
    for result in results:
        combined_ax.plot(result[0], label=f"Modelo {result[1]}")

    combined_ax.set_xlabel('Época', weight='bold')
    combined_ax.set_ylabel('Função de Perda (Loss)', weight='bold')
    combined_ax.set_title('Evolução da função de perda', weight='bold')
    combined_ax.legend(loc='upper right', title='Modelos\n', title_fontsize='large')

    plt.show()

if __name__ == "__main__":
    digits = load_digits()
    X = digits.data
    y = digits.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    model_1 = {
        'solver': 'adam',
        'alpha': 1e-4,
        'neurons': (20, 10),
        'verbose': True,
        'tol': 1e-5,
        'epochs': 200
    }

    model_2 = {
        'solver': 'adam',
        'alpha': 1e-4,
        'neurons': (50, 25, 10),
        'verbose': True,
        'tol': 1e-5,
        'epochs': 200
    }

    models = [model_1, model_2]
    results = []
    for model in models:
        mlp_classifier = MLP(solver=model['solver'],
            alpha=model['alpha'],
            neurons=model['neurons'],
            verbose=model['verbose'],
            tol=model['tol'],
            epochs=model['epochs']
        )
        mlp_classifier.train_model(X_train, y_train)

        total_time = mlp_classifier.get_total_time()
        mean_time = mlp_classifier.get_mean_time()
        loss_curve = mlp_classifier.get_loss_curve()
        current_loss = mlp_classifier.get_current_loss()
        neurons = mlp_classifier.get_neurons()

        results.append([loss_curve, neurons, current_loss, mean_time, total_time])

    make_plot(results)

    quit()
