 <h1 align="center">
 <img width="124" height="124" alt="icon" src="https://github.com/user-attachments/assets/6987432f-5f69-460c-b75a-ac6d7adef434" />
<p>SOL: Simulación Física de Dinámica Estelar y Nuclear</p>

   ![Python](https://img.shields.io/badge/Python-100%25-3776AB?logo=python&logoColor=white)
 ![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Research_Preview-orange)
</h1>

>**"La física dicta la estética."**

## 🔭 Sobre el Proyecto

**S0L** es una simulación física avanzada desarrollada integramente en Python. A diferencia de las visualizaciones astronómicas tradicionales que utilizan animaciones pre-renderizadas, este proyecto implementa modelos matemáticos de primeros principios para simular procesos de dinámica nuclear, gravitacional y termodinámica en tiempo real.

El proyecto se centra en los sistemas **Metriplécticos**, modelando el Sol no solo como un cuerpo gravitatorio, sino como una máquina térmica que procesa entropía e información.

>## 🚀 Funcionalidades Principales

* **Simulación Física Nuclear y Gravitacional:** Cálculo de fuerzas, densidades de energía y dinámica de fluidos estelares.
* **Motor Físico Personalizado:** Módulo central (`physics/sun_3d.py`) que resuelve ecuaciones de campo y trayectorias de partículas.
* **Visualización Científica:** Renderizado de campos escalares (densidad) y vectoriales (viento solar/evaporación) basado en datos reales de la simulación.
* **Interfaz de Usuario Interactiva:** Frontend desarrollado en Streamlit para manipular parámetros de la simulación en tiempo real.
* **Validación de Modelos:** Documentación y scripts dedicados a contrastar los resultados de la simulación con datos teóricos (ej. Precesión de Mercurio).

>## 🛠️ Pila Tecnológica

* **Lenguaje:** Python 3.12 (100%)
* **Arquitectura:** Modular con separación de responsabilidades (Physics, UI, Tests, Docs).
* **Librerías Clave:**
    * `numpy` (Cálculo vectorial)
    * `matplotlib` (Visualización de campos)
    * `streamlit` (Interfaz de usuario)

>## 📂 Estructura del Proyecto

```html
Gravity/
├── physics/            # Motor físico y lógica matemática
│   ├── sun_3d.py       # Simulación del núcleo y halo solar
│   ├── perihelium.py   # Cálculos orbitales y relativistas
│   └── ...
├── docs/               # Documentación científica y validación
│   ├── validation.py
│   └── analysis.py
├── tests/              # Suite de pruebas automatizadas
├── frontend.py         # Punto de entrada para la interfaz visual
├── main.py             # Script principal de ejecución
└── README.md           # Este archivo
```

>## 💻 Instalación y Uso

Clonar el repositorio:
```text
Bash



git clone [https://github.com/tu-usuario/proyecto-sol.git](https://github.com/tu-usuario/proyecto-sol.git)cd Gravity
```
Crear y activar entorno virtual (recomendado):
```text
Bash



python -m venv env# En Windows

.\env\Scripts\activate# En Linux/Macsource env/bin/activate
```
Instalar dependencias:
```text
Bash



pip install -r requirements.txt
```
Ejecutar la simulación y visualizar la interfaz gráfica interactiva:
```text
Bash



streamlit run frontend.py
```
>🧪 Tests

El proyecto incluye una suite de pruebas para validar la integridad de los cálculos físicos.
```text
Bash



pytest tests/
```
>📄  Licencia
<div align="center"
 
```html
Este proyecto se distribuye bajo la licencia MIT.
Esto permite el uso comercial, modificación y distribución, siempre y cuando se proporcione la atribución correspondiente al autor original.
Autor: Jacobo Tlacaelel Mina Rodriguez ("Jako")
Desarrollado porSmopsys QuoreMind.

```
</div>
