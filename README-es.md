# Clasificación de Artículos Médicos por Categoría

Proyecto de NLP que compara distintas representaciones de texto y modelos de clasificación para categorizar artículos científicos médicos (Nutrición, Ejercicio, Ayuno) a partir de un corpus reducido de documentos PDF.

## El problema real detrás del proyecto

Más allá de "clasificar texto", el objetivo central de este proyecto fue metodológico: **con un corpus muy pequeño (16 documentos), ¿qué tan confiables son realmente los resultados de un modelo de clasificación?** El proyecto está diseñado de punta a punta para responder eso con rigor, no solo para maximizar una métrica.

## Fase 1 — Exploración del corpus y calidad de datos

Corpus original de 20 documentos PDF, reducido a 16 tras detectar y eliminar un duplicado exacto. La justificación es importante: mantener un documento duplicado en el corpus genera *data leakage* si, por azar, una copia cae en el set de entrenamiento y la otra en el de prueba — el modelo memorizaría el texto exacto en vez de aprender a generalizar, inflando artificialmente el resultado.

## Fase 2 — Preprocesamiento (con justificación de cada técnica)

Cada técnica de limpieza de texto (minúsculas, eliminación de caracteres especiales, etc.) se documentó explicando qué problema resuelve y cómo afecta el resultado final — no se aplicó preprocesamiento "por costumbre" sin entender su efecto.

## Fase 3 — Representaciones textuales

Se construyeron y compararon 4 representaciones distintas del mismo texto:

| Representación | Tipo | Idea central |
|---|---|---|
| **BoW** (Bolsa de Palabras) | Dispersa | Cuenta frecuencia de palabras, simple y robusta |
| **TF-IDF** | Dispersa | Como BoW, pero le baja peso a palabras muy comunes entre documentos |
| **Word2Vec** | Densa | Embeddings semánticos entrenados sobre el propio corpus |
| **BERT** | Densa | Embeddings preentrenados con transferencia de aprendizaje |

## Fase 4 — Modelado

- **Regresión Logística** sobre cada una de las 4 representaciones (vector fijo por texto, sin noción de orden entre palabras)
- **RNN y LSTM** como modelos secuenciales, procesando el texto palabra por palabra y manteniendo un estado interno — en teoría capaces de capturar orden y contexto, pero con más parámetros para aprender con muy pocos datos disponibles

## Fase 5 — Evaluación

Se midieron Accuracy, Precision, Recall y F1, con una decisión metodológica clave: dado que las clases están desbalanceadas (Nutrición ≫ Ejercicio ≫ Ayuno), Precision/Recall/F1 se calculan con **promedio macro**, no weighted ni micro. Esto le da el mismo peso a cada categoría sin importar cuántos ejemplos tenga, evitando que un buen desempeño en la clase mayoritaria oculte un mal desempeño en las minoritarias.

## Fase 6 — Resultados y conclusiones

**Con split único (train/test):**

| Modelo | F1 (macro) |
|---|---|
| BERT | 0.849 |
| BoW | 0.834 |
| TF-IDF | 0.631 |
| RNN | 0.294 |
| LSTM | 0.275 |
| Word2Vec | 0.262 |

**Hallazgo clave:** LSTM tuvo un Accuracy relativamente alto (0.701) a pesar de su F1 bajísimo (0.275). Esto pasa porque el modelo colapsó a predecir casi siempre la clase mayoritaria ("Nutrición"), acertando por pura frecuencia sin distinguir realmente entre categorías — un ejemplo real de por qué el Accuracy solo puede ser engañoso con clases desbalanceadas, y por qué se priorizó F1 macro desde el diseño del experimento.

**Validación adicional con Leave-One-Out Cross-Validation** (evaluando documento por documento, no con un solo split): el accuracy real de BoW y BERT cayó a 0.661 y 0.632 respectivamente, con altísima variabilidad entre documentos. Esto confirma que el buen resultado del split único estuvo parcialmente influenciado por qué documentos específicos cayeron en train vs. test, y que no es una medida confiable de qué tan bien generalizaría el modelo con datos nuevos.

**Conclusión general:** con un corpus de solo 16 documentos, las representaciones simples y robustas (BoW) o preentrenadas con transferencia de aprendizaje (BERT) superaron ampliamente a las que necesitan aprender una representación desde cero con pocos datos (Word2Vec, RNN, LSTM). Un corpus más grande sería necesario para conclusiones más sólidas sobre qué enfoque es realmente "mejor".

## Stack técnico

- **Python** — pandas, numpy, scikit-learn
- **sentence-transformers** (BERT) para embeddings preentrenados
- **TensorFlow/Keras** para RNN y LSTM
- **scikit-learn** — `LogisticRegression`, `LeaveOneGroupOut`, métricas de clasificación

## Por qué lo incluyo en mi portafolio

No por el resultado del mejor modelo, sino por el proceso: identificar riesgo de data leakage antes de modelar, justificar cada decisión de preprocesamiento, elegir la métrica correcta para el problema (macro F1 sobre Accuracy en un dataset desbalanceado), y validar los resultados con una segunda metodología (LOO-CV) en lugar de confiar en un único split train/test. Esa disciplina de cuestionar los propios resultados es, en mi opinión, más valiosa que cualquier número de accuracy aislado.
