"""Experimentos de Neurito 3.0 (laberinto) sin abrir la ventana de Pygame.

Usa la misma red (laberinto_red.py) y el mismo mundo (laberinto_mundo.py) que
laberinto.py. Cada reto se juega como en el juego: máximo 45 intentos
(movimientos + choques); si no llega a la comida, cuenta como atasco.

Los retos de navegación son TODOS los pares (mapa, inicio, meta) del conjunto
de prueba: combinaciones que la red nunca vio en entrenamiento.

Ejecuta: uv run experimentos_laberinto.py
"""
from collections import Counter
import heapq

import numpy as np

from laberinto_mundo import (MAPAS, NOMBRES, Laberinto, libres, distancias_bfs, accion_experta,
                             vecino, valido, elegir_accion)
from laberinto_red import RedLaberinto, generar_datos, codificar

LIMITE = 45  # igual que laberinto.py


# ---------------------------------------------------------------------------
# Simulación de un reto (misma lógica que el bucle de laberinto.py)
# ---------------------------------------------------------------------------
class Uniforme:
    """'Red' que no sabe nada: 25 % a cada acción. Sirve para aislar las reglas."""
    def probabilidades(self, indice, posicion, meta):
        return np.full(4, 0.25)


def jugar_reto(red, indice, inicio, meta, asistido):
    mundo = Laberinto(indice)
    mundo.posicion, mundo.meta = inicio, meta
    mundo.visitadas = {inicio: 1}
    while mundo.intentos < LIMITE:
        accion, _ = elegir_accion(red, mundo, mundo.visitadas, asistido)
        if mundo.mover(accion) == 'comida':
            return 'comida', mundo.pasos, mundo.choques
    return 'atasco', mundo.pasos, mundo.choques


def navegar(red, casos, asistido):
    exitos, pasos_exito, optimos, choques, atasco_pared = 0, 0, 0, 0, 0
    for indice, inicio, meta, _ in casos:
        resultado, pasos, ch = jugar_reto(red, indice, inicio, meta, asistido)
        choques += ch
        if resultado == 'comida':
            exitos += 1
            pasos_exito += pasos
            optimos += distancias_bfs(MAPAS[indice], meta)[inicio]
        elif ch > 0:
            atasco_pared += 1
    n = len(casos)
    return {
        'exito': exitos / n,
        'atasco_pared': atasco_pared / n,
        'atasco_ciclo': (n - exitos - atasco_pared) / n,
        'choques': choques / n,
        'eficiencia': optimos / pasos_exito if pasos_exito else 0.0,  # 1.0 = ruta óptima
    }


def fila(red, ent, prueba, casos, asistido=False):
    perdida = f"{red.historial[-1]:.4f}" if red.historial else "sin entrenar"
    nav = navegar(red, casos, asistido)
    return (f"pérdida {perdida:>12} | prec. ent. {red.exactitud(*ent):6.1%}"
            f" | prec. prueba {red.exactitud(*prueba):6.1%}"
            f" | comida {nav['exito']:6.1%} | atasco pared {nav['atasco_pared']:6.1%}"
            f" | atasco ciclo {nav['atasco_ciclo']:6.1%} | choques/reto {nav['choques']:5.2f}"
            f" | eficiencia ruta {nav['eficiencia']:6.1%}")


def entrenar(red, ent, epocas):
    for _ in range(epocas):
        red.entrenar_epoca(*ent)
    return red


ent, prueba, casos = generar_datos(semilla=42)
print(f"Ejemplos: {len(ent[1])} entrenamiento | {len(prueba[1])} prueba | retos de navegación: {len(casos)}")
print("Etiquetas en entrenamiento:",
      {NOMBRES[k]: f"{v / len(ent[1]):.1%}" for k, v in sorted(Counter(ent[1].tolist()).items())})

# ---------------------------------------------------------------------------
print("\n=== EXP 1: Neurito sin entrenar ===")
red = RedLaberinto()
print("Red pura:  ", fila(red, ent, prueba, casos))
acciones = Counter(int(red.adelante(prueba[0])[2].argmax(axis=1)[i]) for i in range(len(prueba[1])))
print("Acción elegida (argmax) en prueba:", {NOMBRES[k]: f"{v / len(prueba[1]):.1%}" for k, v in sorted(acciones.items())})
probs = red.adelante(prueba[0])[2]
print(f"Probabilidad máxima promedio: {probs.max(axis=1).mean():.1%} (azar = 25 %)")
# ¿Cuántas veces el argmax inicial apunta a una pared?
contra_pared = sum(not valido(MAPAS[i], vecino(p, int(red.probabilidades(i, p, m).argmax()))) for i, p, m, _ in casos)
print(f"Primer movimiento contra una pared: {contra_pared / len(casos):.1%}")

# ---------------------------------------------------------------------------
print("\n=== EXP 2: 100, 200 y 400 épocas (96 ocultas, tasa 0.45) ===")
red = RedLaberinto()
for total in (100, 200, 400):
    entrenar(red, ent, total - red.epocas)
    print(f"{total:4d} épocas | {fila(red, ent, prueba, casos)}")
red400 = red

# ---------------------------------------------------------------------------
print("\n=== EXP 3: red pura vs navegación asistida ===")
sin_entrenar = RedLaberinto()
for nombre, r in (("solo reglas (probs. uniformes)", Uniforme()),
                  ("red sin entrenar", sin_entrenar), ("red 400 épocas", red400)):
    for asistido in (False, True):
        nav = navegar(r, casos, asistido)
        print(f"{nombre:32s} | {'ASISTIDO' if asistido else 'PURA    '} | comida {nav['exito']:6.1%}"
              f" | atasco pared {nav['atasco_pared']:6.1%} | atasco ciclo {nav['atasco_ciclo']:6.1%}"
              f" | choques/reto {nav['choques']:5.2f} | eficiencia ruta {nav['eficiencia']:6.1%}")

# ---------------------------------------------------------------------------
print("\n=== EXP 4: neuronas ocultas (400 épocas, tasa 0.45) ===")
for ocultas in (16, 32, 96, 128):
    red = entrenar(RedLaberinto(ocultas=ocultas), ent, 400)
    print(f"{ocultas:3d} ocultas | {fila(red, ent, prueba, casos)}")
    print(f"            asistido: comida {navegar(red, casos, True)['exito']:6.1%}")

# ---------------------------------------------------------------------------
print("\n=== EXP 5: tasa de aprendizaje (96 ocultas) ===")
for tasa in (0.05, 0.45, 1.5, 3.0, 6.0, 10.0):
    red = RedLaberinto(tasa=tasa)
    for total in (100, 400):
        entrenar(red, ent, total - red.epocas)
        print(f"tasa {tasa:<4} | {total:4d} épocas | {fila(red, ent, prueba, casos)}")
    h = np.array(red.historial)
    print(f"           la pérdida SUBIÓ entre épocas {int(np.sum(np.diff(h) > 0))} de {len(h) - 1} veces"
          f" | pérdida época 1: {h[0]:.4f} | mínima: {h.min():.4f}")

# ---------------------------------------------------------------------------
print("\n=== EXP 6: por mapa (red 400 épocas) y retos entrenados vs no entrenados ===")
for indice in range(len(MAPAS)):
    sub = [c for c in casos if c[0] == indice]
    sel = np.array([c[0] == indice for c in casos])
    acc = red400.exactitud(prueba[0][sel], prueba[1][sel])
    pura, asis = navegar(red400, sub, False), navegar(red400, sub, True)
    print(f"Mapa {indice + 1} | retos {len(sub):3d} | prec. prueba {acc:6.1%} | comida pura {pura['exito']:6.1%}"
          f" | comida asistida {asis['exito']:6.1%} | eficiencia pura {pura['eficiencia']:6.1%}")
# Retos que SÍ estuvieron en entrenamiento (misma cantidad, para comparar)
todos = []
for indice, mapa in enumerate(MAPAS):
    celdas = libres(mapa)
    for meta in celdas:
        for p in celdas:
            if p != meta:
                todos.append((indice, p, meta, None))
claves_prueba = {(i, p, m) for i, p, m, _ in casos}
entrenados = [c for c in todos if c[:3] not in claves_prueba][:len(casos)]
print(f"Retos entrenados    | comida pura {navegar(red400, entrenados, False)['exito']:6.1%}")
print(f"Retos no entrenados | comida pura {navegar(red400, casos, False)['exito']:6.1%}")


# ---------------------------------------------------------------------------
# EXP 7: cambiar el experto BFS por DFS o A*
# ---------------------------------------------------------------------------
def ruta_dfs(mapa, origen, meta):
    """DFS iterativo con el mismo orden de direcciones. Devuelve la PRIMERA ruta encontrada."""
    pila = [(origen, [origen])]
    vistos = set()
    expandidos = 0
    while pila:
        actual, camino = pila.pop()
        if actual in vistos:
            continue
        vistos.add(actual)
        expandidos += 1
        if actual == meta:
            return camino, expandidos
        for a in reversed(range(4)):  # así ARRIBA se explora primero
            s = vecino(actual, a)
            if valido(mapa, s) and s not in vistos:
                pila.append((s, camino + [s]))
    return None, expandidos


def ruta_astar(mapa, origen, meta):
    """A* con heurística Manhattan (admisible: siempre da la ruta más corta)."""
    h = lambda p: abs(p[0] - meta[0]) + abs(p[1] - meta[1])
    abiertos = [(h(origen), 0, origen)]
    g = {origen: 0}
    padre = {origen: None}
    cerrados = set()
    expandidos = 0
    while abiertos:
        _, costo, actual = heapq.heappop(abiertos)
        if actual in cerrados:
            continue
        cerrados.add(actual)
        expandidos += 1
        if actual == meta:
            camino = []
            while actual is not None:
                camino.append(actual)
                actual = padre[actual]
            return camino[::-1], expandidos
        for a in range(4):
            s = vecino(actual, a)
            if valido(mapa, s) and costo + 1 < g.get(s, 10**9):
                g[s] = costo + 1
                padre[s] = actual
                heapq.heappush(abiertos, (costo + 1 + h(s), costo + 1, s))
    return None, expandidos


def ruta_bfs(mapa, origen, meta):
    dist = distancias_bfs(mapa, meta)
    camino = [origen]
    while camino[-1] != meta:
        camino.append(vecino(camino[-1], accion_experta(mapa, camino[-1], dist)))
    return camino, len(dist)  # BFS desde la meta explora todo el mapa


def accion_de(camino):
    df, dc = camino[1][0] - camino[0][0], camino[1][1] - camino[0][1]
    return {(-1, 0): 0, (1, 0): 1, (0, -1): 2, (0, 1): 3}[(df, dc)]


def datos_con_experto(planificador, semilla=42, fraccion_prueba=0.2):
    """Igual que generar_datos (mismo orden y misma división), pero con otro experto."""
    muestras, largo, optimo, expandidos, coinciden = [], 0, 0, 0, 0
    for indice, mapa in enumerate(MAPAS):
        celdas = libres(mapa)
        for meta in celdas:
            dist = distancias_bfs(mapa, meta)
            for posicion in celdas:
                if posicion == meta or posicion not in dist:
                    continue
                camino, exp = planificador(mapa, posicion, meta)
                a = accion_de(camino)
                largo += len(camino) - 1
                optimo += dist[posicion]
                expandidos += exp
                coinciden += dist[vecino(posicion, a)] == dist[posicion] - 1  # ¿acerca a la meta?
                muestras.append((indice, posicion, meta, a))
    n = len(muestras)
    rng = np.random.default_rng(semilla)
    rng.shuffle(muestras)
    corte = int(n * (1 - fraccion_prueba))
    arr = lambda f: (np.stack([codificar(i, p, m) for i, p, m, a in f]),
                     np.asarray([a for *_, a in f], dtype=np.int64))
    info = {'largo': largo / optimo, 'expandidos': expandidos / n, 'optimas': coinciden / n}
    return arr(muestras[:corte]), arr(muestras[corte:]), muestras[corte:], info


def consistencia(muestras_todas):
    """Si sigo la etiqueta del experto desde cada casilla, ¿llego a la meta? (sin red)"""
    tabla = {(i, p, m): a for i, p, m, a in muestras_todas}
    llega = 0
    for (i, p, m) in tabla:
        actual, pasos = p, 0
        while actual != m and pasos < LIMITE:
            actual = vecino(actual, tabla[(i, actual, m)])
            pasos += 1
        llega += actual == m
    return llega / len(tabla)


print("\n=== EXP 7: experto BFS vs DFS vs A* (96 ocultas, tasa 0.45, 400 épocas) ===")
for nombre, plan in (("BFS", ruta_bfs), ("DFS", ruta_dfs), ("A*", ruta_astar)):
    e, p, c, info = datos_con_experto(plan)
    # Etiquetas de TODOS los pares para medir la consistencia del experto
    tabla_ent = [(i, pos, m, a) for (i, pos, m, a) in c]
    todas = []
    for indice, mapa in enumerate(MAPAS):
        for meta in libres(mapa):
            for pos in libres(mapa):
                if pos != meta:
                    todas.append((indice, pos, meta, accion_de(plan(mapa, pos, meta)[0])))
    red = entrenar(RedLaberinto(), e, 400)
    # La navegación se juega sobre los mismos retos de prueba en los tres casos
    nav = navegar(red, casos, False)
    asis = navegar(red, casos, True)
    print(f"{nombre:3s} | ruta experto / óptima {info['largo']:5.2f}x | 1er paso óptimo {info['optimas']:6.1%}"
          f" | nodos expandidos/consulta {info['expandidos']:5.1f}"
          f" | seguir etiquetas llega {consistencia(todas):6.1%}")
    # ¿La etiqueta depende de la meta? (acciones distintas por casilla, sobre todas las metas)
    por_casilla = {}
    for i, pos, m, a in todas:
        por_casilla.setdefault((i, pos), set()).add(a)
    ref = {(i, pos, m): accion_experta(MAPAS[i], pos, distancias_bfs(MAPAS[i], m)) for i, pos, m, _ in todas}
    iguales = np.mean([a == ref[(i, pos, m)] for i, pos, m, a in todas])
    print(f"    | etiquetas iguales a BFS {iguales:6.1%}"
          f" | acciones distintas por casilla {np.mean([len(s) for s in por_casilla.values()]):4.2f}")
    print(f"    | pérdida {red.historial[-1]:.4f} | prec. prueba (vs su experto) {red.exactitud(*p):6.1%}"
          f" | comida pura {nav['exito']:6.1%} | comida asistida {asis['exito']:6.1%}"
          f" | eficiencia ruta {nav['eficiencia']:6.1%}")
