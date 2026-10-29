import numpy as np

from main import Brain

NAMES = ("COMER", "DORMIR", "JUGAR")


def train(hidden_size=6, epochs=1200, learning_rate=0.1):
    brain = Brain(hidden_size=hidden_size)
    loss = brain.train(epochs=epochs, learning_rate=learning_rate, verbose=False)
    accuracy, test_targets = brain.test_accuracy()
    return brain, loss, accuracy, test_targets


def show(prob):
    return " | ".join(f"{n}: {p * 100:5.1f}%" for n, p in zip(NAMES, prob))


base, loss, acc, test_targets = train()
print("=== RED BASE (6 neuronas, 1200 épocas, lr 0.1) ===")
print(f"Pérdida: {loss:.4f} | Precisión: {acc:.2%}")
print("Ejemplos de prueba por clase (comer, dormir, jugar):", np.bincount(test_targets))

print("\n=== EXP 1: energía = 0.5, aburrimiento = 0.3, se varía el hambre ===")
for hunger in np.arange(0, 1.01, 0.1):
    prob = base.predict(np.array([[hunger, 0.5, 0.3]]))[0]
    print(f"Hambre {hunger:.1f} -> {show(prob)} -> {NAMES[prob.argmax()]}")

print("\n=== EXP 2: prioridades casi iguales ===")
for needs in ([0.5, 0.5, 0.5], [0.51, 0.5, 0.5], [0.6, 0.4, 0.6], [0.9, 0.1, 0.9],
              [0.9, 0.9, 0.1]):
    prob = base.predict(np.array([needs]))[0]
    print(f"{needs} -> {show(prob)}")

print("\n=== EXP 3: neuronas en la capa oculta ===")
for hidden in (3, 6, 10, 20):
    _, loss, acc, _ = train(hidden_size=hidden)
    print(f"{hidden:>2} neuronas -> Pérdida: {loss:.4f} | Precisión: {acc:.2%}")

print("\n=== EXP 4a: épocas ===")
for epochs in (200, 1200, 5000):
    _, loss, acc, _ = train(epochs=epochs)
    print(f"{epochs:>4} épocas -> Pérdida: {loss:.4f} | Precisión: {acc:.2%}")

print("\n=== EXP 4b: tasa de aprendizaje ===")
for lr in (0.01, 0.1, 1.0, 5.0):
    _, loss, acc, _ = train(learning_rate=lr)
    print(f"lr {lr:<4} -> Pérdida: {loss:.4f} | Precisión: {acc:.2%}")