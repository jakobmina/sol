"""
Visualización de un 'Sol Metripléctico':
Núcleo Hamiltoniano denso rodeado por un Halo Disipativo turbulento
que evapora materia entrante.

Inspirado en conceptos de: Jako (Reducción Fragmentada Sistemática Causal, 
Segunda Cuantización Cuasiperiódica y Sistemas Metriplécticos).

CORRECCIONES Y MEJORAS:
- Anotaciones de tipo (type hints) completas
- Función auxiliar para separación de concernencias
- Parámetros configurables en dataclass para análisis de sensibilidad
- Mejor manejo de dimensionalidad para compatibilidad con Nivel 2 (Isomorfismo Dimensional)
"""

from typing import Tuple
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


@dataclass
class ParametrosMetriplecticos:
    """Parámetros físicos del modelo metripléctico.
    
    Mantienen compatibilidad dimensional (Nivel 2 de Isomorfismo):
    - Distancias en unidades normalizadas (1 = L_característica)
    - Densidades en unidades normalizadas (1 = ρ_máxima)
    - Velocidades de flujo en unidades de escape
    """
    N: int = 500                    # Resolución del grid [adimensional]
    L: float = 10.0                 # Tamaño del espacio observable [L_char]
    R_core: float = 3.0             # Radio del núcleo estable [L_char]
    R_halo: float = 5.5             # Radio exterior del halo [L_char]
    
    # Parámetros de densidad (Hamiltoniano)
    factor_densidad_core: float = 1.5  # Amplificación del núcleo [adimensional]
    factor_densidad_halo: float = 0.8  # Peso del halo [adimensional]
    
    # Parámetros de turbulencia (Disipativo)
    amplitud_textura_radial: float = 0.4   # Amplitud de cos(4*R*φ) [adimensional]
    amplitud_textura_angular: float = 0.3  # Amplitud de cos(7*Θ)*sin(...) [adimensional]
    sharpness_core: float = 8.0         # Pronunciación del borde del núcleo [adimensional]
    width_halo: float = 1.2             # Ancho de la capa del halo [L_char]
    offset_halo: float = 1.0            # Desplazamiento radial del halo [L_char]
    
    # Parámetro de evaporación
    factor_evaporacion: float = 1.0     # Intensidad del viento disipativo [adimensional]
    
    @property
    def phi(self) -> float:
        """Razón áurea como parámetro invariante."""
        return (1.0 + 5.0**0.5) / 2.0


def calcular_campos_basicos(params: ParametrosMetriplecticos) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calcula el sistema de coordenadas y campos básicos.
    
    Returns:
        Tuple de (X, Y, R, Theta) todas con shape (N, N)
        - X, Y: coordenadas cartesianas [L_char]
        - R: distancia radial [L_char]
        - Theta: ángulo polar [radianes]
    """
    x: np.ndarray = np.linspace(-params.L, params.L, params.N)
    y: np.ndarray = np.linspace(-params.L, params.L, params.N)
    X, Y = np.meshgrid(x, y)
    
    R: np.ndarray = np.sqrt(X**2 + Y**2)
    Theta: np.ndarray = np.arctan2(Y, X)
    
    return X, Y, R, Theta


def calcular_densidad_core(R: np.ndarray, params: ParametrosMetriplecticos) -> np.ndarray:
    """
    Calcula la componente Hamiltoniana (núcleo conservativo).
    
    Usa sigmoide inversa para representar un condensado de alta densidad
    con transición suave pero definida.
    
    Args:
        R: campo de distancias radiales [L_char]
        params: parámetros del modelo
        
    Returns:
        Densidad del núcleo, normalizada a [0, 1]
    """
    densidad_core: np.ndarray = 1.0 / (1.0 + np.exp(params.sharpness_core * (R - params.R_core)))
    return densidad_core


def calcular_densidad_halo(R: np.ndarray, Theta: np.ndarray, params: ParametrosMetriplecticos) -> np.ndarray:
    """
    Calcula la componente Disipativa (halo turbulento).
    
    Estructura cuasiperiódica usando números irracionales (razón áurea)
    para evitar periodicidad espuria. Esto representa caos determinista,
    no ruido aleatorio.
    
    Args:
        R: campo de distancias radiales [L_char]
        Theta: campo de ángulos polares [radianes]
        params: parámetros del modelo
        
    Returns:
        Densidad del halo, normalizada a [0, 1]
    """
    phi = params.phi
    
    # Gaussiana centrada en el borde del núcleo
    capa_base_halo: np.ndarray = np.exp(
        -0.5 * ((R - (params.R_core + params.offset_halo)) / params.width_halo)**2
    )
    
    # Textura cuasiperiódica (caos determinista)
    textura_radial: np.ndarray = np.cos(4.0 * R * phi)
    textura_angular: np.ndarray = np.cos(7.0 * Theta) * np.sin(7.0 * Theta * phi)
    
    # Máscara para evitar turbulencia dentro del núcleo
    mask_halo_externo: np.ndarray = np.where(R > params.R_core * 0.8, 1.0, 0.0)
    
    # Combinación modulada
    densidad_halo: np.ndarray = (
        capa_base_halo * 
        (1.0 + params.amplitud_textura_radial * textura_radial + 
               params.amplitud_textura_angular * textura_angular) * 
        mask_halo_externo
    )
    
    return densidad_halo


def calcular_densidad_total(densidad_core: np.ndarray, densidad_halo: np.ndarray, 
                             params: ParametrosMetriplecticos) -> np.ndarray:
    """
    Combina las componentes Hamiltoniana y Disipativa.
    
    Args:
        densidad_core: componente conservativa
        densidad_halo: componente disipativa
        params: parámetros del modelo
        
    Returns:
        Densidad total clipeada en [0, ∞)
    """
    densidad_total: np.ndarray = (
        params.factor_densidad_core * densidad_core + 
        params.factor_densidad_halo * densidad_halo
    )
    
    # Eliminar valores negativos por la textura
    densidad_total = np.clip(densidad_total, 0.0, None)
    
    return densidad_total


def calcular_campo_evaporacion(capa_base_halo: np.ndarray, R: np.ndarray, 
                                Theta: np.ndarray, params: ParametrosMetriplecticos) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calcula el campo de flujo disipativo (evaporación radial).
    
    Representa el "viento" que expulsa materia entrante. Es más intenso
    en la zona del halo, simulando la disipación energética.
    
    Args:
        capa_base_halo: perfil Gaussiano del halo
        R: distancia radial
        Theta: ángulo polar
        params: parámetros del modelo
        
    Returns:
        Tuple (U_flow, V_flow) componentes cartesianas del flujo [L_char/t_char]
    """
    fuerza_evaporacion: np.ndarray = (
        params.factor_evaporacion * capa_base_halo * (R / params.R_halo)
    )
    
    # Componentes vectoriales (flujo radialmente hacia afuera)
    U_flow: np.ndarray = fuerza_evaporacion * np.cos(Theta)
    V_flow: np.ndarray = fuerza_evaporacion * np.sin(Theta)
    
    return U_flow, V_flow


def generar_sol_metriplectico(params: ParametrosMetriplecticos | None = None, show_plot: bool = True) -> Tuple[plt.Figure, plt.Axes]:
    """
    Genera y visualiza el Sol Metripléctico.
    
    Args:
        params: Objeto ParametrosMetriplecticos. Si es None, usa valores por defecto.
        show_plot: Si es True, muestra la figura (bloqueante). Si es False, solo la retorna.
    """
    if params is None:
        params = ParametrosMetriplecticos()
    
    # --- Cálculos de campos ---
    X, Y, R, Theta = calcular_campos_basicos(params)
    
    densidad_core = calcular_densidad_core(R, params)
    densidad_halo = calcular_densidad_halo(R, Theta, params)
    densidad_total = calcular_densidad_total(densidad_core, densidad_halo, params)
    
    # Recalculamos la capa base del halo para el campo de evaporación
    capa_base_halo: np.ndarray = np.exp(
        -0.5 * ((R - (params.R_core + params.offset_halo)) / params.width_halo)**2
    )
    U_flow, V_flow = calcular_campo_evaporacion(capa_base_halo, R, Theta, params)
    
    # --- Visualización ---
    fig, ax = plt.subplots(figsize=(11, 11), dpi=100)
    
    # Mapa de color personalizado
    cmap = plt.cm.inferno
    
    # Densidad de energía/masa
    im = ax.imshow(densidad_total, extent=[-params.L, params.L, -params.L, params.L], 
                   origin='lower', cmap=cmap, vmin=0, vmax=1.8)
    
    # Campo de flujo de evaporación (Streamplot con muestreo)
    skip = (slice(None, None, 20), slice(None, None, 20))
    # 1. Asignamos el plot a una variable 'strm' y QUITAMOS alpha de los argumentos
    strm = ax.streamplot(X[skip], Y[skip], U_flow[skip], V_flow[skip],
                     color='cyan', linewidth=1, arrowsize=1.5, density=0.8)

    # 2. Aplicamos la transparencia (alpha) a las líneas y flechas después de crearlas
    strm.lines.set_alpha(0.6)
    strm.arrows.set_alpha(0.6)
    # Círculos de referencia para los radios característicos
    circle_core = plt.Circle((0, 0), params.R_core, fill=False, 
                             edgecolor='yellow', linestyle='--', linewidth=1.5, alpha=0.5)
    circle_halo = plt.Circle((0, 0), params.R_halo, fill=False, 
                             edgecolor='lime', linestyle='--', linewidth=1.5, alpha=0.5)
    ax.add_patch(circle_core)
    ax.add_patch(circle_halo)
    
    # Decoración
    ax.set_facecolor('black')
    ax.set_title("Sol Metripléctico: Isomorfismo Físico de Conservación-Disipación\n"
                 "Núcleo Hamiltoniano (Amarillo) | Halo Disipativo (Verde) | Flujo (Cian)",
                 color='white', fontsize=13, fontweight='bold', pad=15)
    
    ax.set_xlabel('x [L_char]', color='white')
    ax.set_ylabel('y [L_char]', color='white')
    ax.tick_params(colors='white')
    
    # Anotaciones
    ax.annotate('Núcleo Condensado\n(Hamiltoniano, Conservativo)',
                xy=(0, 0), xytext=(-3.5, -8),
                arrowprops=dict(facecolor='yellow', shrink=0.05, width=1.5),
                color='yellow', ha='center', fontsize=10, fontweight='bold')
    
    ax.annotate('Halo de Evaporación\n(Disipativo, Entrópico)',
                xy=(params.R_core+0.3, params.R_core+0.3), xytext=(6.5, 7),
                arrowprops=dict(facecolor='cyan', shrink=0.05, width=1.5),
                color='cyan', ha='center', fontsize=10, fontweight='bold')
    
    # Barra de color
    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label('Densidad de Energía [ρ_norm]', color='white', rotation=270, labelpad=20)
    cbar.ax.tick_params(colors='white')
    
    ax.set_xlim(-params.L, params.L)
    ax.set_ylim(-params.L, params.L)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    print("Generando visualización del sistema solar metripléctico...")
    print(f"Parámetros utilizados:")
    print(f"  - Radio del núcleo (Hamiltoniano): {params.R_core} L_char")
    print(f"  - Radio del halo (Disipativo): {params.R_halo} L_char")
    print(f"  - Factor de amplificación del núcleo: {params.factor_densidad_core}")
    print(f"  - Razón áurea (φ): {params.phi:.10f}")
    
    if show_plot:
        plt.show()
    
    return fig, ax


if __name__ == "__main__":
    # Ejecución con parámetros por defecto
    generar_sol_metriplectico()
    
    # Ejemplo: análisis de sensibilidad (descomenta para explorar)
    # params_alt = ParametrosMetriplecticos(
    #     factor_evaporacion=1.5,
    #     amplitud_textura_radial=0.6
    # )
    # generar_sol_metriplectico(params_alt)
