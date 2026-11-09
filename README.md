# Neurito – Mi cerebro artificial

Neurito es una mascota virtual controlada por una pequeña red neuronal escrita desde cero con NumPy. La red recibe tres necesidades (hambre, energía y aburrimiento) y decide entre tres acciones: **comer**, **dormir** o **jugar**. La interfaz está hecha con Pygame.

## Estructura del repositorio

| Archivo | Descripción |
|---|---|
| `main.py` | Red neuronal + juego. Los parámetros de los experimentos están al inicio (`HIDDEN_SIZE`, `EPOCHS`, `LEARNING_RATE`). |
| `experimentos.py` | Ejecuta todos los experimentos y muestra los resultados en consola (sin abrir el juego). |
| `assets/` | Imágenes de Neurito (neutral, comer, dormir, jugar). |
| `requirements.txt` | Librerías necesarias. |

## Cómo ejecutarlo

```bash
pip install -r requirements.txt
python main.py        # entrena la red y abre el juego
python experimentos.py   # corre los experimentos
```

**Controles:** H/J hambre +/- · E/D energía +/- · A/S aburrimiento +/- · R aleatorio · N estado tranquilo · ESC salir.

## Cambios realizados al código

- Se agregaron al inicio de `main.py` las variables `HIDDEN_SIZE`, `EPOCHS` y `LEARNING_RATE` para cambiar fácilmente el número de neuronas ocultas, las épocas y la tasa de aprendizaje.
- La red se organizó en una clase `Brain` con los métodos `train`, `predict` y `test_accuracy`, para poder entrenar varias redes con distintas configuraciones.
- El juego quedó dentro de la función `run_game`, así el archivo `experimentos.py` puede usar la red sin abrir la ventana.
- Se creó `experimentos.py`, que realiza los experimentos de forma automática.
- La interfaz quedó en español por defecto (`LANGUAGE = "es"`).

## Resultados de los experimentos

Todos los resultados usan la semilla 42. La red base tiene 6 neuronas ocultas, 1200 épocas y tasa de aprendizaje 0.1, y obtiene **pérdida 0.1225 y precisión 99.40 %**.

### 1. Mantener dos necesidades fijas y variar la tercera

Energía fija en 0.5 (cansancio = 0.5) y aburrimiento fijo en 0.3. Se sube el hambre:

| Hambre | Comer | Dormir | Jugar | Acción |
|---|---|---|---|---|
| 0.0 | 0.0 % | 97.5 % | 2.5 % | Dormir |
| 0.3 | 2.3 % | 95.6 % | 2.0 % | Dormir |
| 0.4 | 13.3 % | 85.0 % | 1.7 % | Dormir |
| **0.5** | **49.9 %** | **49.1 %** | 0.9 % | **Comer** |
| 0.6 | 86.6 % | 13.2 % | 0.2 % | Comer |
| 1.0 | 100.0 % | 0.0 % | 0.0 % | Comer |

**Conclusión:** la acción cambia de dormir a comer cuando el hambre alcanza al cansancio (0.5). La red aprendió la regla de entrenamiento: gana la necesidad más alta. Cerca de ese punto el cambio es gradual; lejos de él, la red está casi 100 % segura.

### 2. Prioridades casi iguales

| Hambre, energía, aburrimiento | Comer | Dormir | Jugar |
|---|---|---|---|
| 0.5, 0.5, 0.5 | 35.2 % | 33.4 % | 31.3 % |
| 0.51, 0.5, 0.5 | 39.7 % | 31.3 % | 29.1 % |
| 0.9, 0.1, 0.9 | 35.9 % | 33.2 % | 30.9 % |
| 0.9, 0.9, 0.1 (una necesidad domina) | 100.0 % | 0.0 % | 0.0 % |

**Conclusión:** cuando las necesidades están casi empatadas, Softmax reparte la probabilidad casi en partes iguales (alrededor de 33 % cada una): la red está insegura. Un cambio pequeño, como subir el hambre de 0.50 a 0.51, inclina la decisión. Cuando una necesidad domina, una salida se acerca al 100 %.

### 3. Neuronas en la capa oculta

| Neuronas | Pérdida | Precisión |
|---|---|---|
| 3 | 0.1618 | 96.30 % |
| 6 | 0.1225 | 99.40 % |
| 10 | 0.1262 | 99.30 % |
| 20 | 0.1220 | 98.80 % |

**Conclusión:** con 3 neuronas la red tiene poca capacidad y comete más errores. A partir de 6 neuronas los resultados son casi iguales; como el problema es sencillo, más neuronas no ayudan mucho y solo hacen la red más grande.

### 4. Épocas y tasa de aprendizaje

| Configuración | Pérdida | Precisión | Interpretación |
|---|---|---|---|
| 200 épocas | 0.4984 | 91.80 % | Le falta entrenamiento |
| 1200 épocas | 0.1225 | 99.40 % | Buen resultado |
| 5000 épocas | 0.0535 | 99.60 % | La pérdida baja, pero la precisión casi no mejora: se estanca |
| lr = 0.01 | 0.7376 | 82.90 % | Aprende demasiado lento |
| lr = 0.1 | 0.1225 | 99.40 % | Buen resultado |
| lr = 1.0 | 0.0447 | 98.70 % | Aprende rápido y funciona |
| lr = 5.0 | 1.0979 | 33.80 % | Inestable: los pasos son tan grandes que no aprende (33 % es el azar con 3 clases) |

**Conclusión:** más épocas mejoran el entrenamiento hasta que se estanca. Una tasa de aprendizaje muy pequeña hace que aprenda lento y una muy grande lo vuelve inestable.

### 5. ¿Qué se necesitaría para una cuarta acción?

Por ejemplo, la acción **bañarse**, asociada a una nueva necesidad, **suciedad**:

- **Datos:** una cuarta columna en `X` con el nivel de suciedad (`np.random.rand(3000, 4)`).
- **Etiquetas:** incluir la suciedad en la política para que `targets` pueda valer 0, 1, 2 o 3:
  `np.argmax(np.column_stack((X[:,0], 1 - X[:,1], X[:,2], X[:,3])), axis=1)`.
- **Red:** `W1` pasa a ser de 4×6, `W2` de 6×4 y `b2` de 1×4 (cuatro salidas en Softmax).
- **Ejemplos suficientes** de cada una de las cuatro acciones, para que las clases estén balanceadas.
- **Interfaz:** una nueva barra, teclas para cambiarla, textos e imágenes de Neurito bañándose.

## Explicación de los resultados

**1. ¿En qué se diferencian pesos y sesgos?**
Los pesos (`W1`, `W2`) multiplican las entradas e indican cuánto influye cada una en cada neurona, por ejemplo cuánto pesa el hambre en la decisión de comer. Los sesgos (`b1`, `b2`) se suman después y no dependen de la entrada: funcionan como un umbral que mueve el punto en que una neurona se activa y le dan a la red una preferencia de base.

**2. ¿Por qué el cansancio es `1 - energía`?**
Para que las tres necesidades signifiquen lo mismo: un valor alto indica "necesito hacer esto". Con hambre alta, Neurito debe comer; con aburrimiento alto, jugar. Pero con energía alta **no** necesita dormir. Al invertirla, un cansancio alto indica que debe dormir, y así `argmax` puede comparar las tres necesidades de forma justa.

**3. ¿Qué hace ReLU y qué hace Softmax?**
- **ReLU** (`max(0, x)`) deja pasar los valores positivos y convierte los negativos en 0. Añade no linealidad, lo que permite que la red aprenda decisiones más complejas que una simple combinación lineal.
- **Softmax** transforma las tres salidas finales en probabilidades positivas que suman 100 %. Son los porcentajes de comer, dormir y jugar que aparecen en pantalla.

**4. ¿Por qué una precisión alta puede ser engañosa?**
- El 99.4 % solo mide qué tan bien la red copia una regla que inventamos nosotros. No demuestra que Neurito tome buenas decisiones en una situación real.
- Los datos de prueba se generan con la misma fórmula y distribución que los de entrenamiento; con datos diferentes la red podría fallar.
- Los errores se concentran en los casos casi empatados, que son justamente los más difíciles.
- Si las clases estuvieran desbalanceadas (por ejemplo, 90 % "comer"), una red que siempre responde "comer" tendría 90 % de precisión sin aprender nada. En este caso las clases están balanceadas (338 / 320 / 342 ejemplos de prueba).
- El estado "tranquilo" no lo aprende la red: lo decide una regla fija en el código.
---

# Neurito 2.0 – Aprende a saltar

En la segunda parte, Neurito aprende **cuándo saltar** los obstáculos que se acercan a distintas velocidades. La red tiene arquitectura **3 → 12 → 1**: recibe la distancia, la velocidad y el tiempo estimado hasta el obstáculo, tiene 12 neuronas ocultas con ReLU y una salida con sigmoide. Si la salida es ≥ 0.5 y Neurito está en el suelo, salta. Las etiquetas dicen "saltar" cuando el tiempo estimado está entre **7 y 13 fotogramas**.

## Archivos nuevos

| Archivo | Descripción |
|---|---|
| `salto_modelo.py` | Red neuronal manual (`RedSalto`) y generación de datos. |
| `salto_juego.py` | Física del salto y detección de choques. |
| `saltar.py` | Juego con Pygame y visualización del entrenamiento. |
| `experimentos_salto.py` | Corre los experimentos sin abrir la ventana y muestra los resultados en consola. |
| `GUIA_SALTO.md` | Guía de la actividad. |

```bash
uv sync
uv run saltar.py               # juego (1: +1 época, 2: +100 épocas, 3: continuo)
uv run experimentos_salto.py   # experimentos
```

## Cómo se midió

`experimentos_salto.py` usa exactamente la misma red, los mismos datos y la misma física que `saltar.py`, pero sin dibujar. Cada partida dura **3000 fotogramas** (100 segundos a 30 FPS) en modo automático, y se cuentan los obstáculos esquivados y los choques. La precisión se mide con 1500 ejemplos de prueba distintos de los 5000 de entrenamiento. Semilla 42.

## Resultados

### 1. Juego sin entrenar

| Pérdida | Precisión (prueba) | Esquivados | Choques | Saltos |
|---|---|---|---|---|
| sin entrenar | 44.9 % | 0 | 20 | 0 |

**Conclusión:** con pesos aleatorios, la red nunca da una probabilidad ≥ 0.5, así que Neurito **nunca salta** y choca con todos los obstáculos. Su 44.9 % de precisión viene de responder siempre "esperar": acierta todos los ejemplos de esperar (el 45 % de los datos) y falla todos los de saltar (el 55 %).

### 2. Entrenamiento de cien en cien épocas (12 ocultas, tasa 0.8)

| Épocas | Pérdida | Precisión (prueba) | Esquivados | Choques |
|---|---|---|---|---|
| 100 | 0.2601 | 91.2 % | 22 | 1 |
| 200 | 0.2306 | 91.9 % | 24 | 0 |
| 300 | 0.2066 | 92.3 % | 24 | 0 |
| 500 | 0.1368 | 94.1 % | 24 | 0 |
| 700 | 0.0858 | 96.5 % | 24 | 0 |
| 1000 | 0.0574 | 97.8 % | 24 | 0 |

**Conclusión:** la pérdida baja y la precisión sube en cada bloque de 100 épocas. **El comportamiento en el juego mejora muy rápido**: con solo 100 épocas ya esquiva 22 de 23 obstáculos, y desde 200 épocas no choca. Después de eso, la pérdida sigue bajando, pero el juego ya no cambia. Lo que sí cambia es *cuándo* salta: con 100 épocas salta a unos 16 fotogramas del obstáculo y con 1000 épocas a unos 13.5, más cerca de la ventana de 7 a 13 de las etiquetas.

### 3. Neuronas ocultas (1000 épocas, tasa 0.8)

| Ocultas | Pérdida | Prec. entrenamiento | Prec. prueba | Esquivados | Choques |
|---|---|---|---|---|---|
| 3 | 0.2060 | 93.7 % | 93.2 % | 24 | 0 |
| 12 | 0.0574 | 98.0 % | 97.8 % | 24 | 0 |
| 24 | 0.0502 | 98.3 % | 97.7 % | 24 | 0 |

**Conclusión (generalización):** en los tres casos, la precisión de prueba es casi igual a la de entrenamiento (diferencia de 0.2 a 0.6 puntos). La red **generaliza bien** y no hay sobreajuste. Con 3 neuronas falta capacidad para dibujar bien la ventana de salto (93 %). Pasar de 12 a 24 casi no cambia nada en la prueba: más neuronas solo hacen la red más grande. En el juego, las tres configuraciones esquivan todo, porque basta con acertar el momento de saltar de forma aproximada.

### 4. Tasa de aprendizaje (12 ocultas)

| Tasa | Épocas | Pérdida | Prec. prueba | Esquivados | Choques |
|---|---|---|---|---|---|
| 0.01 | 100 | 0.6449 | 90.7 % | 18 | 4 |
| 0.01 | 500 | 0.5149 | 85.9 % | **0** | **18** |
| 0.01 | 1000 | 0.4302 | 86.5 % | **0** | **18** |
| 0.8 | 100 | 0.2601 | 91.2 % | 22 | 1 |
| 0.8 | 1000 | 0.0574 | 97.8 % | 24 | 0 |
| 1.2 | 100 | 0.2560 | 89.3 % | 18 | 4 |
| 1.2 | 1000 | 0.0392 | 98.3 % | 24 | 0 |

Número de veces que la pérdida **subió** de una época a la siguiente (de 999): tasa 0.01 → 0, tasa 0.8 → 0, tasa 1.2 → **292**.

**Conclusión (convergencia):**
- **0.01:** converge de forma estable pero **muy lenta**. Después de 1000 épocas, la pérdida (0.43) sigue lejos de la que logra 0.8. En este caso la red salta demasiado pronto y choca con *todos* los obstáculos, aunque su precisión sea de 86 %.
- **0.8:** converge rápido y sin oscilaciones. Es un buen equilibrio.
- **1.2:** llega a la pérdida más baja (0.039), pero el camino es **inestable**: la pérdida sube casi 3 de cada 10 épocas porque los pasos son tan grandes que a veces "se pasan" del mínimo. Con una tasa todavía mayor podría divergir.

## Pregunta central: ¿por qué una red puede clasificar con buena precisión y aun así chocar?

Porque **la precisión mide otra cosa diferente a lo que importa en el juego**. Los experimentos lo muestran:

| Configuración | Precisión (prueba) | Choques |
|---|---|---|
| tasa 0.8, 100 épocas | 91.2 % | 1 |
| tasa 0.01, 100 épocas | 90.7 % | 4 |
| tasa 0.01, 500 épocas | 85.9 % | **18 de 18** |

Con tasa 0.01 y 500 épocas, la red acierta casi 9 de cada 10 ejemplos, pero choca **siempre**. Las razones son:

1. **En el juego solo importa una decisión: el primer momento en que la probabilidad llega a 0.5.** Neurito salta en el primer fotograma que la red dice "saltar". Los errores en ejemplos muy lejanos o muy cercanos no importan, pero un error justo en el **borde de entrada** de la ventana es fatal. La red con tasa 0.01 salta cuando faltan **22.4 fotogramas** para llegar al obstáculo, cuando debería saltar entre 7 y 13. La precisión promedia todos los ejemplos por igual y no ve que ese error en particular es el que decide el choque.
2. **Las etiquetas no son la física.** La ventana de 7 a 13 fotogramas es una regla que nosotros inventamos. Al simular el salto real (impulso -12, gravedad 0.65), el salto funciona si se hace entre **4 y 14.5 fotogramas** antes a velocidad 4.5, y entre **4 y 22.5** a velocidad 8.5. A velocidad 5.5 el límite es 18, así que saltar a los 22.4 hace caer a Neurito encima del obstáculo. La red copia la regla, no la física.
3. **Los ejemplos de entrenamiento son independientes; el juego es una secuencia.** Cada ejemplo se evalúa por separado, pero en el juego una decisión cambia las siguientes: después de saltar, Neurito pasa unos 37 fotogramas en el aire sin poder volver a saltar.
4. **La distribución de los datos no es la del juego.** Los datos de prueba tienen 55 % de ejemplos de "saltar", mientras que en el juego la mayoría de los fotogramas son de "esperar". Además, la red solo vio velocidades entre 4.5 y 8.5.

En resumen, una precisión alta indica que la red imita bien las etiquetas, pero no garantiza que salte en el momento exacto que exige la física. Para evaluar a Neurito hay que medir también **choques y obstáculos esquivados en el juego**.

---

# Neurito 3.0 – Aprende a encontrar comida en un laberinto

En la tercera parte, Neurito busca comida dentro de tres laberintos de 9×9. La red tiene arquitectura **165 → 96 → 4**: recibe el mapa (one-hot de 3), su posición (one-hot de 81) y la meta (one-hot de 81); tiene 96 neuronas ocultas con ReLU y 4 salidas con Softmax (arriba, abajo, izquierda, derecha). Las etiquetas las da un **experto BFS**, que conoce la ruta más corta. Es **aprendizaje supervisado por imitación**, no aprendizaje por refuerzo: durante el juego la red decide sola, sin consultar BFS.

## Archivos nuevos

| Archivo | Descripción |
|---|---|
| `laberinto_mundo.py` | Mapas, movimiento, experto BFS y navegación asistida. |
| `laberinto_red.py` | Red neuronal manual (`RedLaberinto`) y generación de datos. |
| `laberinto.py` | Juego con Pygame, sprites y gráfica de la pérdida. |
| `test_laberinto.py` | Pruebas automáticas (las 5 pasan). |
| `experimentos_laberinto.py` | Corre los experimentos sin abrir la ventana y muestra los resultados en consola. |
| `GUIA_LABERINTO.md` | Guía de la actividad. |

```bash
uv sync
uv run laberinto.py                              # juego
uv run python -m unittest -v test_laberinto.py   # pruebas
uv run experimentos_laberinto.py                 # experimentos
```

## Cómo se midió

`experimentos_laberinto.py` usa exactamente la misma red, los mismos datos y la misma regla de decisión (`elegir_accion`) que `laberinto.py`, pero sin dibujar. Semilla 42.

- **Datos:** 3576 ejemplos (mapa, posición, meta → movimiento de BFS): 2860 de entrenamiento y 716 de prueba. Las clases están balanceadas (arriba 26.7 %, abajo 26.3 %, izquierda 21.4 %, derecha 25.5 %).
- **Navegación:** se juegan los **716 retos de prueba**, es decir, combinaciones de inicio y meta que la red nunca vio. Igual que en el juego, cada reto tiene un máximo de **45 intentos** (movimientos + choques); si no llega a la comida, es un **atasco**.
- **Métricas:**
  - *Comida*: porcentaje de retos completados.
  - *Atasco pared*: no llegó y chocó con paredes. En red pura, chocar significa quedarse en la misma casilla y repetir la misma decisión para siempre.
  - *Atasco ciclo*: no llegó y nunca chocó, así que estuvo dando vueltas.
  - *Eficiencia de ruta*: pasos óptimos de BFS / pasos reales en los retos completados (100 % = ruta más corta).

## Resultados

### 1. Neurito sin entrenar: ¿por qué se atasca?

| Pérdida | Prec. prueba | Comida | Atasco pared | Atasco ciclo | Choques por reto |
|---|---|---|---|---|---|
| sin entrenar | 27.7 % | 3.1 % | 82.3 % | 14.7 % | 36.4 |

- Con pesos aleatorios, las salidas Softmax son casi iguales (la probabilidad máxima promedio es 29.7 %, cerca del 25 % del azar), y la precisión es de azar (27.7 %).
- Pero la red no es aleatoria en cada paso: es **determinista**. Para la misma entrada siempre elige lo mismo, y además tiene un sesgo: elige arriba o abajo en el 86 % de los casos.
- En el 48 % de los retos, el primer movimiento ya apunta a una pared. Al chocar, Neurito **no cambia de casilla**, así que la entrada a la red es la misma y la red vuelve a elegir la misma pared. Se queda chocando hasta gastar los 45 intentos (36 choques por reto en promedio).
- Si no choca, termina oscilando entre dos casillas (14.7 % de ciclos), porque cada casilla "manda" a la otra.
- Solo llega en el 3 % de los casos, cuando la comida está muy cerca y la dirección elegida coincide por suerte.

### 2. Entrenamiento: 100, 200 y 400 épocas (96 ocultas, tasa 0.45)

| Épocas | Pérdida | Prec. entrenamiento | Prec. prueba | Comida | Atasco pared | Atasco ciclo | Choques por reto | Eficiencia |
|---|---|---|---|---|---|---|---|---|
| 0 | — | 25.3 % | 27.7 % | 3.1 % | 82.3 % | 14.7 % | 36.37 | — |
| 100 | 0.1352 | 94.1 % | 84.9 % | 51.3 % | 2.1 % | 46.6 % | 0.91 | 98.3 % |
| 200 | 0.0376 | 99.9 % | 90.1 % | 81.3 % | 0.7 % | 18.0 % | 0.31 | 99.4 % |
| 400 | 0.0107 | 100.0 % | 91.5 % | 82.7 % | 0.4 % | 16.9 % | 0.19 | 99.3 % |

**Conclusión:**
- La pérdida baja de forma clara: de 0.135 a 0.011. La precisión de entrenamiento llega al 100 %: la red **memoriza** todos los ejemplos de entrenamiento.
- La precisión de prueba sube menos (de 84.9 % a 91.5 %). La diferencia de casi 9 puntos entre entrenamiento y prueba muestra **algo de sobreajuste**.
- La **precisión no equivale al porcentaje de laberintos completados**. Con 100 épocas la red acierta el 85 % de los movimientos, pero solo completa el 51 % de los retos. Un reto exige acertar *varias decisiones seguidas*, y un solo error puede bastar para entrar en un ciclo.
- Entre 100 y 200 épocas el juego mejora mucho (de 51 % a 81 %). Entre 200 y 400 la pérdida sigue bajando, pero el juego casi no cambia (de 81 % a 83 %).
- Cuando llega a la comida, lo hace por una ruta casi óptima (eficiencia del 99 %): la red sí aprendió el camino más corto de BFS.
- Después de entrenar, casi no choca con paredes (0.19 choques por reto). El fallo típico es el **ciclo**: Neurito va y viene entre dos casillas.

### 3. Red pura vs navegación asistida: ¿qué parte es del modelo y qué parte de las reglas?

| Cerebro | Modo | Comida | Atasco pared | Atasco ciclo | Choques por reto | Eficiencia |
|---|---|---|---|---|---|---|
| Solo reglas (probabilidades iguales, 25 %) | Pura | 3.8 % | 96.2 % | 0.0 % | 42.05 | — |
| Solo reglas (probabilidades iguales, 25 %) | Asistido | 93.0 % | 0.0 % | 7.0 % | 0.00 | 31.8 % |
| Red sin entrenar | Pura | 3.1 % | 82.3 % | 14.7 % | 36.37 | — |
| Red sin entrenar | Asistido | 87.8 % | 0.0 % | 12.2 % | 0.00 | 34.2 % |
| Red 400 épocas | Pura | 82.7 % | 0.4 % | 16.9 % | 0.19 | 99.3 % |
| Red 400 épocas | Asistido | **100.0 %** | 0.0 % | 0.0 % | 0.00 | 91.6 % |

Para separar el modelo de las reglas, se probó un "cerebro" que no sabe nada: da 25 % a cada acción. Todo lo que logre en modo asistido lo hacen las reglas.

**Conclusión:**
- **Las reglas, por sí solas, ya encuentran la comida el 93 % de las veces.** Descartar paredes y penalizar casillas visitadas convierte a Neurito en un explorador que recorre el laberinto hasta tropezar con la comida. Como los mapas son pequeños (≈40 casillas), 45 intentos suelen alcanzar. Pero lo hace dando muchas vueltas: sus rutas son **unas 3 veces más largas** que la óptima (eficiencia del 32 %).
- **La red entrenada aporta la dirección:** sola completa el 82.7 % por rutas casi óptimas (99 %), pero a veces se atasca en ciclos.
- **Juntas se complementan:** las reglas eliminan choques y ciclos, y la red señala el camino. Así se llega al 100 % con rutas buenas (92 %).
- Por eso el 100 % del modo asistido **no se puede atribuir solo a la red**. Para evaluar lo que aprendió el modelo hay que mirar la red pura (82.7 %). Las reglas son conocimiento que programamos nosotros, no conocimiento aprendido.

### 4. Neuronas ocultas (400 épocas, tasa 0.45)

| Ocultas | Pérdida | Prec. entrenamiento | Prec. prueba | Comida (pura) | Choques por reto | Comida (asistido) |
|---|---|---|---|---|---|---|
| 16 | 0.0342 | 99.4 % | 87.2 % | 74.3 % | 1.36 | 100 % |
| 32 | 0.0161 | 100.0 % | 89.2 % | 78.4 % | 0.75 | 100 % |
| 96 | 0.0107 | 100.0 % | 91.5 % | 82.7 % | 0.19 | 100 % |
| 128 | 0.0103 | 100.0 % | 90.8 % | 85.3 % | 0.62 | 100 % |

**Conclusión:** aquí sí se nota la capacidad de la red, a diferencia de Neurito 1.0 y 2.0. El problema es más difícil: 165 entradas, 4 clases y una respuesta que depende de la combinación de posición *y* meta. Con 16 neuronas la red casi memoriza el entrenamiento (99.4 %), pero generaliza peor (87.2 %) y completa solo el 74 % de los retos. Al subir a 96 o 128 mejoran la prueba y el juego. Entre 96 y 128 la diferencia es pequeña y mixta: 128 completa más retos, pero tiene menor precisión de prueba y choca más; más neuronas ya no ayudan mucho. Con asistencia todos llegan al 100 %, lo que confirma que las reglas tapan las diferencias entre modelos.

### 5. Tasa de aprendizaje (96 ocultas)

| Tasa | Épocas | Pérdida | Prec. prueba | Comida (pura) | ¿Converge? | Veces que la pérdida subió (de 399) |
|---|---|---|---|---|---|---|
| 0.05 | 100 | 0.6063 | 74.4 % | 24.2 % | | |
| 0.05 | 400 | 0.2829 | 83.2 % | 46.1 % | Sí, muy lento | 1 |
| **0.45** | 100 | 0.1352 | 84.9 % | 51.3 % | | |
| **0.45** | 400 | 0.0107 | 91.5 % | 82.7 % | **Sí** | 85 |
| 1.5 | 100 | 0.0121 | 91.8 % | 82.8 % | | |
| 1.5 | 400 | 0.0016 | 93.4 % | 86.9 % | Sí, rápido | 46 |
| 3.0 | 100 | 0.0046 | 92.9 % | 87.2 % | | |
| 3.0 | 400 | 0.0007 | 93.9 % | 87.7 % | Sí, más rápido | 20 |
| 6.0 | 400 | 1.4093 | 23.6 % | 4.5 % | **No** | 203 |
| 10.0 | 400 | 1.6597 | 26.3 % | 3.9 % | **No** | 193 |

**Conclusión (convergencia):**
- **Con 0.45 sí converge.** La pérdida baja de 1.37 a 0.011. Sube 85 veces de 399, pero son pequeñas oscilaciones del descenso por mini-lotes cuando la pérdida ya está cerca de cero; la tendencia es siempre a bajar.
- **0.05** converge de forma muy estable (solo sube una vez), pero es lenta: tras 400 épocas su pérdida (0.28) es peor que la de 0.45 a las 100 épocas, y completa solo el 46 % de los retos.
- **1.5 y 3.0** convergen aún más rápido y llegan a una pérdida más baja. Para este problema, 0.45 es una tasa **conservadora**: funciona, pero no es la más rápida.
- **6.0 y 10.0 no convergen:** la pérdida se queda en 1.4 o más (el azar con 4 clases es ln 4 ≈ 1.39), sube la mitad de las veces y la precisión queda en ≈25 %, como si no hubiera entrenado. Los pasos son tan grandes que la red "salta" por encima del mínimo en lugar de bajar hacia él. Neurito se comporta como sin entrenar: se atasca contra las paredes el 95 % de las veces.

### 6. Cambiar de mapa (red 400 épocas)

| Mapa | Retos de prueba | Prec. prueba | Comida (pura) | Comida (asistido) | Eficiencia (pura) |
|---|---|---|---|---|---|
| 1 | 210 | 92.4 % | 85.7 % | 100 % | 99.8 % |
| 2 | 251 | 90.0 % | 78.5 % | 100 % | 98.3 % |
| 3 | 255 | 92.2 % | 84.3 % | 100 % | 99.7 % |

| Retos | Comida (pura) |
|---|---|
| Inicio y meta **entrenados** (716 retos) | 92.5 % |
| Inicio y meta **no entrenados** (716 retos de prueba) | 82.7 % |

**Conclusión:**
- La red funciona en los tres mapas porque los tres forman parte del entrenamiento: conoce el identificador de cada mapa. El **mapa 2** es el más difícil (78.5 %); tiene muchas bifurcaciones y caminos alternativos, así que hay más casillas donde un pequeño error lleva a un ciclo.
- En los retos con combinaciones de inicio y meta **no entrenadas** completa el 82.7 %: la red **generaliza** a pares nuevos, aunque 10 puntos por debajo de los pares entrenados.
- Incluso en los retos entrenados no llega al 100 %, a pesar de tener 100 % de precisión de entrenamiento. Solo el *primer* paso es un ejemplo entrenado; después de moverse, el par (nueva posición, meta) puede pertenecer al conjunto de prueba.
- **Limitación:** esto *no* demuestra que Neurito funcione en un laberinto nuevo. La red no ve las paredes: solo conoce el número del mapa. Con un cuarto mapa no sabría dónde están las paredes. Para eso habría que darle la cuadrícula de paredes como entrada y evaluar con mapas no usados en el entrenamiento.

### 7. Cambiar el experto: BFS vs DFS vs A*

Se reemplazó el profesor BFS por DFS (búsqueda en profundidad, con el mismo orden de direcciones) y por A* (con distancia Manhattan como heurística). Con cada experto se generaron nuevas etiquetas, con la misma división entrenamiento/prueba, y se entrenó una red igual (96 ocultas, tasa 0.45, 400 épocas). La navegación se mide en los mismos 716 retos de prueba.

**El experto:**

| Experto | Ruta vs óptima | Primer paso óptimo | Nodos explorados por consulta | Seguir sus etiquetas llega a la meta | Acciones distintas por casilla |
|---|---|---|---|---|---|
| BFS | 1.00× | 100 % | 35.1 | 100 % | 2.21 |
| DFS | **2.24×** | 50.3 % | 18.6 | **7.7 %** | **1.04** |
| A* | 1.00× | 100 % | **10.1** | 100 % | 2.21 |

**Neurito entrenado con cada experto:**

| Experto | Pérdida | Prec. prueba (vs su experto) | Comida (pura) | Comida (asistido) | Eficiencia |
|---|---|---|---|---|---|
| BFS | 0.0107 | 91.5 % | 82.7 % | 100 % | 99.3 % |
| DFS | 0.0012 | **99.9 %** | **7.5 %** | **44.8 %** | — |
| A* | 0.0110 | 92.6 % | **86.3 %** | 100 % | 99.6 % |

**Conclusión:**
- **DFS arruina a Neurito**, aunque la red tiene la mejor precisión (99.9 %) y la pérdida más baja. Es el caso más claro de que la precisión puede engañar:
  - DFS no busca la ruta más corta: sigue la primera que encuentra, que en promedio mide 2.24 veces la óptima.
  - Sus etiquetas son **inconsistentes entre sí**: la ruta DFS desde una casilla y la ruta DFS desde la siguiente no son la misma. Si sigues las etiquetas de DFS paso a paso (sin red), solo llegas a la meta el 7.7 % de las veces: das vueltas.
  - Con un orden fijo (primero arriba), DFS da casi siempre **la misma acción en cada casilla sin importar dónde está la comida** (1.04 acciones distintas por casilla, frente a 2.21 de BFS). La red aprende esa regla fácil, "en esta casilla ve hacia arriba", e **ignora la meta**. Por eso acierta el 99.9 %: imitar a DFS es trivial, pero inútil.
  - Ni la asistencia lo salva (44.8 %, peor que las reglas solas con 93 %): las probabilidades de la red empujan en una dirección que no tiene que ver con la comida.
- **A* mejora un poco a Neurito** (86.3 % frente a 82.7 %) y sigue siendo óptimo. A* y BFS dan rutas igual de cortas, pero cuando hay varias rutas del mismo largo pueden elegir distinto (coinciden en el 96.3 % de las etiquetas). La heurística de A* favorece las casillas más cercanas a la comida en línea recta, así que sus etiquetas siguen un patrón más regular ("acércate a la comida"), probablemente más fácil de generalizar para la red.
- **Costo del experto:** A* explora 10 nodos por consulta, contra 35 de BFS (que recorre todo el mapa desde la meta) y 19 de DFS. Sin embargo, un BFS desde la meta calcula de una vez la etiqueta de *todas* las posiciones para esa meta, así que para generar el conjunto de datos completo BFS sigue siendo barato.
- **Lección:** en el aprendizaje por imitación, la red solo puede ser tan buena como su profesor. Un experto óptimo y consistente (BFS, A*) produce un buen Neurito; un experto no óptimo e inconsistente (DFS) produce un Neurito que imita perfectamente un mal comportamiento.

## Conclusión general

1. **Precisión ≠ éxito en el laberinto.** Llegar a la comida exige una *secuencia* de decisiones correctas, y un solo error puede dejar a Neurito en un ciclo. Con 85 % de precisión completa solo el 51 % de los retos. El caso de DFS (99.9 % de precisión y 7.5 % de éxito) muestra que la precisión solo mide qué tan bien se imita al experto, no si el experto es bueno.
2. **Las reglas del modo asistido hacen mucho trabajo.** Por sí solas encuentran la comida el 93 % de las veces, aunque con rutas tres veces más largas. Lo que aporta la red es la *dirección* hacia la comida.
3. **La red generaliza a combinaciones nuevas de inicio y meta** (82.7 %), pero solo dentro de los tres mapas conocidos: no aprendió a "ver" paredes.
4. **La calidad del profesor importa más que el tamaño de la red.** Cambiar BFS por DFS tiene más efecto que cualquier cambio de neuronas o de tasa de aprendizaje.
