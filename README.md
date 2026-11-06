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

> Nota: cada partida tiene solo unos 20 a 24 obstáculos. Con otra semilla o partidas más largas, los números pueden cambiar un poco, pero las tendencias se mantienen.
