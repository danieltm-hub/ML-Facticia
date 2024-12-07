# Análisis de textos e imágenes para establecer conexiones y relaciones entre personalidades
## Enfoques basados en textos
### Subproblemas:
- Uso de modelos para analizar la estructura de los textos para tratar de encontrar relaciones entre las personalidades mencionadas en los textos
- Detección de entidades nombradas 
- Clasificación de relaciones

### Propuestas:
#### Modelos para encontrar relaciones entre personalidades
1. Modelos Basados en Word Embeddings

    Word2Vec: Este modelo convierte palabras en vectores densos, permitiendo que palabras con significados similares tengan representaciones cercanas en el espacio vectorial. Esto facilita la identificación de relaciones semánticas entre términos relacionados con personas1
    3
    .
    GloVe (Global Vectors for Word Representation): Similar a Word2Vec, GloVe genera representaciones vectoriales que capturan las relaciones contextuales entre palabras, lo que ayuda a identificar conexiones entre nombres y roles en un texto.

2. Redes Neuronales Recurrentes (RNN)

    Descripción: Las RNN son adecuadas para procesar secuencias de texto y pueden ser utilizadas para modelar relaciones temporales o contextuales entre personas mencionadas en un documento. Esto es útil para tareas como la extracción de información y el análisis de sentimientos.
    LSTM (Long Short-Term Memory): Una variante de RNN que maneja mejor las dependencias a largo plazo, permitiendo capturar relaciones complejas en textos narrativos.

3. Transformers

    BERT (Bidirectional Encoder Representations from Transformers): Este modelo utiliza atención bidireccional para comprender el contexto de las palabras dentro de una oración, lo que mejora la identificación de relaciones entre personas al considerar el contexto completo2
    4
    .
    GPT-3: Un modelo generativo que puede entender y generar texto coherente, siendo útil para inferir relaciones a partir del contexto textual1
    .

4. Clasificación y Análisis de Sentimientos

    Naive Bayes: Utilizado para clasificar textos en categorías relacionadas con personas, como identificar el tono o la intención detrás de las menciones5
    .
    Análisis de Sentimientos: Esta técnica permite determinar la polaridad (positiva, negativa o neutral) de las interacciones entre personas mencionadas en los textos, ayudando a entender la naturaleza de sus relaciones.

5. Extracción de Relaciones

    Modelos de Extracción de Relaciones: Se utilizan técnicas específicas para identificar y clasificar relaciones entre entidades mencionadas en un texto. Esto puede incluir métodos supervisados que requieren conjuntos de datos etiquetados o enfoques no supervisados que analizan patrones en los datos.
    N-gramas: Se utilizan para capturar secuencias contiguas de palabras que pueden indicar relaciones específicas entre personas, mejorando la precisión en la identificación de interacciones3
    .

6. Análisis Contextual

    Análisis Semántico: Utiliza técnicas avanzadas para comprender el significado detrás del lenguaje, lo que permite identificar cómo se relacionan las personas en diferentes contextos.
    Modelos Basados en Atención: Estos modelos permiten a los sistemas centrarse en partes específicas del texto donde se mencionan las relaciones entre personas, mejorando la capacidad para extraer información relevante.

#### Detección de entidades nombradas
1. Métodos Basados en Reglas

    Descripción: Estos métodos se fundamentan en reglas lingüísticas y patrones específicos definidos manualmente. Utilizan expresiones regulares y diccionarios para identificar entidades.
    Ventajas: Son efectivos en dominios específicos donde las entidades son bien definidas, como el lenguaje médico.
    Desventajas: Tienen limitaciones en escalabilidad y flexibilidad, ya que pueden no adaptarse bien a conjuntos de datos variados o grandes

2. Modelos de Aprendizaje Automático

    Descripción: Utilizan algoritmos de aprendizaje supervisado para entrenar modelos en conjuntos de datos anotados. Estos modelos aprenden a identificar y clasificar entidades basándose en características extraídas del texto.
    Ejemplos Comunes:
        Máquinas de Vectores de Soporte (SVM): Utilizadas para clasificar entidades basándose en características extraídas.
        Árboles de Decisión: Para clasificar entidades según reglas aprendidas durante el entrenamiento

3. Redes Neuronales

    Descripción: Los modelos de redes neuronales, especialmente las redes neuronales recurrentes (RNN) y las redes neuronales convolucionales (CNN), son utilizados para NER debido a su capacidad para capturar patrones complejos en los datos.
    Transformers: Modelos como BERT (Bidirectional Encoder Representations from Transformers) han demostrado ser altamente efectivos al considerar el contexto completo de las palabras, mejorando la precisión en la identificación de entidades

#### Clasificación de relaciones
Usar los modelos estudiados en clases

## Enfoques basados en imagenes
### Subproblemas:
- Detección de objetos para identificar figuras humanas
- Segmentación de las imágenes para dividirlas en regiones significativas
- Aplicar técnicas de clasificación para determinar a quien pertenece una determinada silueta en la imagen
- Análisis de distribución espacial para determinar relaciones visualmente en las imagenes

### Propuestas:
#### Identificación de figuras humanas:
- YOLO [https://visionplatform.ai/es/yolov8-deteccion-de-objetos-de-ultima-generacion-en-reconocimiento-de-imagenes-computer-vision/] (https://visionplatform.ai/es/yolov8-deteccion-de-objetos-de-ultima-generacion-en-reconocimiento-de-imagenes-computer-vision/), un modelo de detección de objetos que permite la identificación y localización de múltiples objetos en tiempo real. Se basa en una única red neuronal que predice simultáneamente las clases y las cajas delimitadoras de los objetos en una sola pasada.
- Los modelos actuales más reconocidos utlizan CNN (Redes Neuronales Convolucionales) para esta tarea o R-CNN (Region-based Convolutional Neural Networks) 

#### Segmentación de imágenes para dividirlas en regiones significativas:
1. Segmentación por Umbral

    Descripción: Esta técnica implica establecer un umbral para los valores de intensidad de los píxeles, separando así diferentes clases de objetos en la imagen. Es especialmente efectiva en imágenes con alto contraste.
    Algoritmos Comunes:
        Algoritmo de Otsu
        Método de Entropía Máxima

2. Segmentación Basada en Regiones

    Descripción: Agrupa píxeles en regiones homogéneas basándose en características similares, como color o textura. Se utilizan métodos como el crecimiento de regiones y la agrupación.
    Algoritmos Comunes:
        Crecimiento de Regiones
        K-means Clustering

3. Segmentación por Aprendizaje Profundo

    Descripción: Utiliza redes neuronales convolucionales (CNN) para aprender representaciones jerárquicas de las características visuales. Estos modelos son capaces de segmentar objetos a nivel de píxel y realizar segmentación semántica y de instancias.
    Modelos Destacados:
        U-Net: Diseñado para segmentación médica, permite una segmentación precisa a partir de imágenes complejas.
        SegNet: Utiliza una arquitectura encoder-decoder que es eficiente en la segmentación semántica.
        Mask R-CNN: Extensión de Faster R-CNN que permite la segmentación a nivel de instancia, generando máscaras para cada objeto detectado.

4. Segmentación Basada en Bordes

    Descripción: Se centra en detectar y seguir los bordes y contornos dentro de una imagen, utilizando algoritmos específicos para identificar cambios bruscos en la intensidad.
    Algoritmos Comunes:
        Canny Edge Detector
        Sobel Filter

5. Segmentación Basada en Color

    Descripción: Separa objetos según sus características cromáticas, utilizando modelos de color como RGB o HSV para identificar segmentos basados en similitudes de color.
    Técnicas Comunes:
        Umbralización basada en el color
        Histogramas de color

#### Técnicas de clasificación para determinar a quien pertenece una determinada silueta en la imagen:
Técnicas de Reconocimiento Facial
1. Detección Facial

    Algoritmo Viola-Jones: Este algoritmo es uno de los más utilizados para la detección de rostros en imágenes. Utiliza un conjunto de características en cascada (Haar features) para identificar rápidamente la presencia de un rostro
    
    Histograma de Gradientes Orientados (HOG): Este método se utiliza para detectar objetos, incluyendo rostros, al capturar la información de gradientes en la imagen, lo que ayuda a resaltar las características relevantes
    

2. Extracción de Características

    Análisis de Componentes Principales (PCA): Esta técnica reduce la dimensionalidad de los datos, extrayendo las características más importantes de las imágenes faciales, lo que permite una representación más eficiente
    
    Análisis Discriminante Lineal (LDA): A diferencia del PCA, LDA busca maximizar la separación entre diferentes clases en el espacio de características, mejorando la clasificación
    

3. Reconocimiento Facial Basado en Modelos

    Modelos Holísticos: Estos modelos consideran toda la imagen del rostro y crean un vector de características que se compara con otros vectores en una base de datos
    
    Modelos Geométricos: Se centran en las relaciones espaciales entre los rasgos faciales (como la distancia entre los ojos y la boca) para identificar a una persona
    

4. Métodos Basados en Aprendizaje Profundo

    Redes Neuronales Convolucionales (CNN): Estas redes son muy eficaces para el reconocimiento facial, ya que pueden aprender automáticamente las características relevantes a partir de grandes conjuntos de datos
    
    FaceNet: Este sistema utiliza un enfoque basado en embeddings que mapea imágenes faciales a un espacio euclidiano donde las distancias representan similitudes entre rostros.

#### Análisis de distribución espacial para determinar relaciones visualmente en las imagenes
1. Análisis de Proximidad

    Descripción: Evalúa la cercanía entre diferentes ubicaciones y su accesibilidad, utilizando métricas como la distancia euclidiana o la red de caminos.
    Aplicación: Utilizado para determinar cómo la proximidad entre objetos afecta su interacción y relación.

2. Análisis de Densidad

    Descripción: Estudia la concentración de eventos o características en un área específica para identificar patrones de distribución.
    Métodos Comunes:
        Kernel Density Estimation (KDE): Estima la densidad de puntos en un área, ayudando a visualizar concentraciones.
        Interpolación Espacial: Métodos como Kriging o distancia inversa que estiman valores en ubicaciones no muestreadas basándose en datos conocidos.

3. Análisis de Patrones Espaciales

    Descripción: Identifica y evalúa la aleatoriedad o agrupamiento de eventos mediante métodos estadísticos.
    Ejemplos:
        Índice de Moran: Mide la autocorrelación espacial para determinar si los valores similares están agrupados.
        Diagrama de Dispersión Espacial: Visualiza la relación entre dos variables en un contexto espacial.

4. Clasificación Espacial

    Descripción: Utiliza algoritmos para clasificar datos espaciales basándose en características observadas.
    Métodos:
        Clasificación Supervisada: Requiere datos etiquetados para entrenar el modelo (ej. Máxima Verosimilitud).
        Clasificación No Supervisada: Agrupa datos sin etiquetas previas (ej. K-Means).

5. Transformaciones y Filtrado de Imágenes

    Descripción: Modifica imágenes para resaltar características importantes y reducir el ruido.
    Técnicas:
        Filtrado Espacial: Suaviza o agudiza imágenes mediante técnicas matemáticas aplicadas a los píxeles vecinos.
        Transformaciones de Color: Ayuda a distinguir objetos mediante la manipulación del espectro de color.


## Enfoques híbridos
Vincular las técnicas de ambos enfoques para obtener mejores resultados

### Propuestas:
- Utilización de técnicas como CLIP (Contrastive Lenguage-Image Pre-training)
- Análisis multimodal (Modelos que puedan procesar tanto texto como imágenes simultaneamente)
