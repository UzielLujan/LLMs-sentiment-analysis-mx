# **Análisis de Sentimientos para Turismo Mexicano con Aprendizaje Multitarea**

## **1\. Resumen**

Este repositorio documenta un proyecto avanzado de análisis de sentimientos sobre reseñas de turismo en México, desarrollado originalmente para la competencia REST-MEX 2025\. El proyecto evoluciona a partir de un enfoque de clasificación de tarea única (single-task) a un pipeline de **Aprendizaje Multitarea (Multi-Task Learning \- MTL)** robusto, modular y optimizado para su ejecución en clústeres de supercómputo.

El objetivo principal fue mejorar el desempeño en la clasificación de **Polaridad** (sentimiento) forzando a modelos Transformer (BETO y MarIA) a aprender simultáneamente dos tareas auxiliares: la predicción del **Tipo** de establecimiento en el que fue emitida la reseña (Type) y la identificación del **Pueblo Mágico** (Town). La hipótesis central, validada con éxito, fue que este enfoque multitarea proporcionaría un contexto más rico, mejorando el rendimiento en todas las tareas.

La contribución estratégica clave del proyecto fue la implementación de una función de pérdida alineada con la métrica de evaluación oficial de la competencia. Este pivote estratégico, denominado **"Score-Optimized" (SO)**, resultó en un salto de rendimiento notable para esta tarea con el modelo **BETO (MTL-SO)**, que alcanzó un **Score oficial de 0.7822**.

## **2\. Estructura del Proyecto**

LLMs-sentiment-analysis-mx/  
├── data/               \# Contiene los datasets de entrenamiento y prueba  
├── notebooks/          \# Jupyter notebooks para EDA y prototipado inicial  
├── src/                \# Código fuente modular del pipeline  
│   ├── data\_loader.py  
│   ├── load\_test.py  
│   ├── train\_mtl.py  
│   ├── eval\_utils\_mtl.py  
│   └── inference.py  
├── models/             \# Checkpoints de los modelos guardados (ignorado por Git)  
├── results/            \# Métricas y matrices de confusión (ignorado por Git)  
├── submissions/        \# Archivos .txt de predicción finales (ignorado por Git)  
├── .gitignore  
├── requirements.txt    \# Dependencias del proyecto  
├── README.md           \# Este archivo  
└── run\_\*.sh            \# Scripts de lanzamiento para el clúster SLURM

**Nota sobre los Scripts de Lanzamiento:** Los scripts run\_\*.sh se encuentran intencionadamente en la raíz del proyecto. Nuestra extensa depuración en el clúster SLURM de CIMAT demostró que esta es la forma más robusta de evitar problemas complejos de rutas relativas, garantizando un flujo de trabajo estable y reproducible.

## **3\. Metodología y Evolución Estratégica**

El proyecto se abordó como una serie de experimentos iterativos, cada uno construido sobre los aprendizajes del anterior.

### **Fase 1: Baselines y Optimización del Pipeline**

* **Modelos Base:** Se establecieron baselines con **BETO** (dccuchile/bert-base-spanish-wwm-cased) y **MarIA** (BSC-TeMU/roberta-base-bne) en un enfoque de tarea única.  
* **Optimización para HPC:** Se construyó un pipeline modular y se optimizó para el clúster Lab-SB de CIMAT, resolviendo numerosos desafíos de entorno y dependencias.  
* **Entrenamiento Distribuido:** Se implementó torchrun con Distributed Data Parallel (DDP) para entrenar en 2 GPUs, logrando una **aceleración de \~3x** (de \~1.5 horas a \~30 minutos por ejecución).

### **Fase 2: El Salto al Aprendizaje Multitarea (MTL)**

* **Hipótesis:** Entrenar para las tres tareas (Polarity, Type, Town) simultáneamente mejoraría el rendimiento.  
* **Implementación:** Se desarrolló un MultiTaskModel con tres cabezas de clasificación. La función de pérdida inicial se enfocó en Polarity.  
* **Resultado:** Éxito rotundo. El F1-Score de Polaridad saltó de \~0.73 a **\~0.765**, validando la estrategia MTL.

### **Fase 3: El Pivote Estratégico (Score-Optimized)**

* El Descubrimiento: Un análisis detallado de la documentación de la competencia reveló que la métrica de evaluación final no era el F1 de Polaridad, sino un "Score" ponderado:  
  Score=62×F1polarity​+1×F1type​+3×F1town​​  
* **La Nueva Hipótesis:** Alinear la función de pérdida del entrenamiento con esta fórmula debería maximizar el Score final.  
* **Implementación Final:** Se modificó la función de pérdida en train\_mtl.py para reflejar la ponderación 2-1-3 y se ajustó la métrica de optimización del Trainer para que guardara el checkpoint con el Score más alto.

## **4\. Resultados Finales**

La estrategia "Score-Optimized" (SO) fue decisiva y reveló a **BETO** como el campeón indiscutible.

| Modelo | Estrategia | F1 Polarity | F1 Type | F1 Town | Score Oficial |
| :---- | :---- | :---- | :---- | :---- | :---- |
| BETO (MTL) | Foco en Polaridad | 0.7656 | 0.9770 | 0.6894 | 0.7627 |
| **BETO (MTL-SO)** | **Score-Optimized** | 0.7592 | 0.9782 | **0.7322** | **0.7822 (+0.0195)** |

El modelo SO sacrificó inteligentemente un rendimiento mínimo en Polarity para lograr una ganancia masiva en Town, la tarea más valiosa, lo que resultó en un aumento drástico del Score global.

## **5\. Uso e Instrucciones**

### **5.1. Configuración del Entorno**

Se recomienda usar conda para gestionar el entorno.

\# 1\. Clonar el repositorio  
git clone \[https://github.com/UzielLujan/LLMs-sentiment-analysis-mx.git\](https://github.com/UzielLujan/LLMs-sentiment-analysis-mx.git)  
cd LLMs-sentiment-analysis-mx

\# 2\. Crear y activar el entorno  
conda create \--name llms-mx-env python=3.10  
conda activate llms-mx-env

\# 3\. Instalar dependencias  
pip install \-r requirements.txt  
\# (Opcional, para GPU local)  
\# conda install pytorch torchvision torchaudio pytorch-cuda=12.1 \-c pytorch \-c nvidia

**Nota:** Para el clúster de CIMAT, se requiere una instalación manual más detallada de las librerías.

### **5.2. Entrenamiento en el Clúster**

Para lanzar un entrenamiento optimizado para el Score con BETO durante 6 épocas:

sbatch run\_mtl\_2gpu.sh "models/BETO\_local" "BETO\_MTL\_SO\_final" 6

### **5.3. Inferencia y Sumisión**

Para generar el archivo de predicciones con el modelo campeón (BETO\_MTL\_SO):

sbatch run\_prediction.sh "models/BETO\_MTL\_SO" "BETO\_final\_submission" "dccuchile/bert-base-spanish-wwm-cased"

El archivo CorpusChristi\_Run.txt se generará en la carpeta submissions/BETO\_final\_submission/.

## **6\. Conclusión **

Este proyecto demuestra el poder del Aprendizaje Multitarea y, de forma más crítica, la importancia de **alinear los objetivos de entrenamiento con las métricas de evaluación específicas**. Esta alineación fue la clave que desbloqueó el máximo potencial del modelo.

Uziel Isaí Lujan López — M.Sc. in Statistical Computing at CIMAT  
[LinkedIn](https://www.linkedin.com/in/uziel-lujan/) | [GitHub](https://github.com/UzielLujan)