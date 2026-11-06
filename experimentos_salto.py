"""Experimentos de Neurito 2.0 (salto) sin abrir la ventana de Pygame.

Usa la misma red (salto_modelo.py) y la misma física (salto_juego.py) que
saltar.py. Cada partida dura FOTOGRAMAS fotogramas en modo automático.

Ejecuta: python experimentos_salto.py
"""
import numpy as np

from salto_juego import JuegoSalto
from salto_modelo import RedSalto

FOTOGRAMAS = 3000  # 100 segundos de juego a 30 FPS


def crear_red(ocultas=12, tasa=0.8):
    """Crea la red y los datos en el mismo orden que saltar.py."""
    red = RedSalto(ocultas=ocultas, tasa=tasa)
    X, y = red.generar_datos(5000)
    X_test, y_test = red.generar_datos(1500)
    return red, X, y, X_test, y_test


def jugar(red, fotogramas=FOTOGRAMAS):
    """Juega en modo automático y devuelve (esquivados, choques, saltos)."""
    juego = JuegoSalto()
    saltos = 0
    for _ in range(fotogramas):
        if juego.en_suelo and red.probabilidad(juego.distancia, juego.velocidad) >= 0.5:
            saltos += juego.saltar()
        juego.actualizar()
    return juego.esquivados, juego.choques, saltos


def tasa_exito(esquivados, choques):
    total = esquivados + choques
    return esquivados / total if total else 0.0


def fila(red, X, y, X_test, y_test):
    esq, cho, saltos = jugar(red)
    perdida = f"{red.historial[-1]:.4f}" if red.historial else "sin entrenar"
    return (f"pérdida {perdida:>12} | prec. entrenamiento {red.precision(X, y):6.1%}"
            f" | prec. prueba {red.precision(X_test, y_test):6.1%}"
            f" | esquivados {esq:3d} | choques {cho:3d} | éxito {tasa_exito(esq, cho):6.1%}"
            f" | saltos {saltos}")


def entrenar(red, X, y, epocas):
    for _ in range(epocas):
        red.entrenar_epoca(X, y)


# ---------------------------------------------------------------------------
print("=== EXP 1: juego sin entrenar ===")
red, X, y, X_test, y_test = crear_red()
print(fila(red, X, y, X_test, y_test))
print(f"Etiquetas 'saltar' en entrenamiento: {y.mean():.1%} | en prueba: {y_test.mean():.1%}")

# ---------------------------------------------------------------------------
print("\n=== EXP 2: entrenamiento de cien en cien épocas (12 ocultas, tasa 0.8) ===")
for total in range(100, 1001, 100):
    entrenar(red, X, y, 100)
    print(f"{total:5d} épocas | {fila(red, X, y, X_test, y_test)}")

# ---------------------------------------------------------------------------
print("\n=== EXP 3: neuronas ocultas (1000 épocas, tasa 0.8) ===")
for ocultas in (3, 12, 24):
    red, X, y, X_test, y_test = crear_red(ocultas=ocultas)
    entrenar(red, X, y, 1000)
    print(f"{ocultas:2d} ocultas | {fila(red, X, y, X_test, y_test)}")

# ---------------------------------------------------------------------------
print("\n=== EXP 4: tasa de aprendizaje (12 ocultas) ===")
for tasa in (0.01, 0.8, 1.2):
    red, X, y, X_test, y_test = crear_red(tasa=tasa)
    for total in (100, 500, 1000):
        entrenar(red, X, y, total - red.epocas)
        print(f"tasa {tasa:<4} | {total:4d} épocas | {fila(red, X, y, X_test, y_test)}")
    h = np.array(red.historial)
    subidas = int(np.sum(np.diff(h) > 0))
    print(f"           veces que la pérdida SUBIÓ entre épocas: {subidas} de {len(h) - 1}")

# ---------------------------------------------------------------------------
print("\n=== PREGUNTA CENTRAL: ¿qué ventana de salto funciona de verdad en la física? ===")
for v in (4.5, 5.5, 6.5, 7.5, 8.5):
    buenos = []
    for ttc in np.arange(0, 40, 0.5):
        juego = JuegoSalto()
        juego.velocidad = v
        juego.obstaculo_x = 168 + v * ttc  # 168 = JUGADOR_X + JUGADOR_ANCHO
        juego.saltar()
        while not juego.terminado and juego.esquivados == 0:
            juego.actualizar()
        if not juego.terminado:
            buenos.append(ttc)
    rango = f"{min(buenos):.1f} a {max(buenos):.1f}" if buenos else "ninguno"
    print(f"Velocidad {v}: saltar sirve si el tiempo al obstáculo está entre {rango} fotogramas")
