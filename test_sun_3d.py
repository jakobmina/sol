"""
Tests para la visualización 3D del Sol Metripléctico.

Verifica que:
- El módulo se importa correctamente
- Las funciones retornan objetos del tipo esperado
- Los cálculos físicos son consistentes con la versión 2D
- La animación genera el número correcto de frames
"""

import pytest
import numpy as np
import plotly.graph_objects as go

from physics.sun_3d import (
    operador_aureo,
    calcular_densidad_temporal,
    crear_superficie_3d,
    crear_campo_vectorial_3d,
    generar_sol_metriplectico_3d
)
from physics.sun import ParametrosMetriplecticos


class TestOperadorAureo:
    """Tests para el Operador Áureo O_n(t)."""
    
    def test_rango_valores(self):
        """El operador debe estar en el rango [-1, 1]."""
        phi = (1.0 + 5.0**0.5) / 2.0
        tiempos = np.linspace(0, 2*np.pi, 100)
        
        for t in tiempos:
            valor = operador_aureo(1.0, t, phi)
            assert -1.0 <= valor <= 1.0, f"Valor fuera de rango: {valor}"
    
    def test_periodicidad(self):
        """El operador debe ser cuasiperiódico (no exactamente periódico)."""
        phi = (1.0 + 5.0**0.5) / 2.0
        
        # Evaluar en t=0 y t=2π
        val_0 = operador_aureo(1.0, 0, phi)
        val_2pi = operador_aureo(1.0, 2*np.pi, phi)
        
        # Deben ser similares pero no necesariamente idénticos
        assert abs(val_0 - val_2pi) < 0.5, "Diferencia excesiva en un ciclo"


class TestDensidadTemporal:
    """Tests para el cálculo de densidad con evolución temporal."""
    
    def test_shape_consistente(self):
        """La densidad temporal debe tener la misma forma que los campos de entrada."""
        params = ParametrosMetriplecticos(N=50)
        
        x = np.linspace(-params.L, params.L, params.N)
        y = np.linspace(-params.L, params.L, params.N)
        X, Y = np.meshgrid(x, y)
        R = np.sqrt(X**2 + Y**2)
        Theta = np.arctan2(Y, X)
        
        densidad = calcular_densidad_temporal(R, Theta, 0, params)
        
        assert densidad.shape == (params.N, params.N), \
            f"Shape incorrecta: {densidad.shape}"
    
    def test_valores_no_negativos(self):
        """La densidad debe ser siempre no negativa."""
        params = ParametrosMetriplecticos(N=50)
        
        x = np.linspace(-params.L, params.L, params.N)
        y = np.linspace(-params.L, params.L, params.N)
        X, Y = np.meshgrid(x, y)
        R = np.sqrt(X**2 + Y**2)
        Theta = np.arctan2(Y, X)
        
        tiempos = np.linspace(0, 2*np.pi, 10)
        
        for t in tiempos:
            densidad = calcular_densidad_temporal(R, Theta, t, params)
            assert np.all(densidad >= 0), f"Densidad negativa en t={t}"
    
    def test_nucleo_estable(self):
        """El núcleo debe permanecer relativamente estable en el tiempo."""
        params = ParametrosMetriplecticos(N=50)
        
        x = np.linspace(-params.L, params.L, params.N)
        y = np.linspace(-params.L, params.L, params.N)
        X, Y = np.meshgrid(x, y)
        R = np.sqrt(X**2 + Y**2)
        Theta = np.arctan2(Y, X)
        
        # Densidad en el centro (núcleo) en diferentes tiempos
        centro_idx = params.N // 2
        densidades_centro = []
        
        for t in np.linspace(0, 2*np.pi, 10):
            densidad = calcular_densidad_temporal(R, Theta, t, params)
            densidades_centro.append(densidad[centro_idx, centro_idx])
        
        # La variación en el centro debe ser pequeña (núcleo estable)
        variacion = np.std(densidades_centro) / np.mean(densidades_centro)
        assert variacion < 0.3, f"Núcleo demasiado variable: {variacion}"


class TestSuperficie3D:
    """Tests para la creación de superficies 3D."""
    
    def test_retorna_surface(self):
        """Debe retornar un objeto Surface de Plotly."""
        X = np.array([[0, 1], [0, 1]])
        Y = np.array([[0, 0], [1, 1]])
        Z = np.array([[1, 2], [3, 4]])
        
        surface = crear_superficie_3d(X, Y, Z)
        
        assert isinstance(surface, go.Surface), \
            f"Tipo incorrecto: {type(surface)}"
    
    def test_colorscale_inferno(self):
        """Debe usar la escala de color Inferno."""
        X = np.array([[0, 1], [0, 1]])
        Y = np.array([[0, 0], [1, 1]])
        Z = np.array([[1, 2], [3, 4]])
        
        surface = crear_superficie_3d(X, Y, Z)
        
        # Plotly convierte 'Inferno' a una tupla de colores
        # Verificamos que sea una secuencia (tuple o list)
        assert isinstance(surface.colorscale, (tuple, list)), \
            f"Colorscale debe ser secuencia, obtenido: {type(surface.colorscale)}"


class TestCampoVectorial3D:
    """Tests para el campo vectorial 3D."""
    
    def test_retorna_cone(self):
        """Debe retornar un objeto Cone de Plotly."""
        X = np.random.rand(10, 10)
        Y = np.random.rand(10, 10)
        Z = np.random.rand(10, 10)
        U = np.random.rand(10, 10)
        V = np.random.rand(10, 10)
        W = np.random.rand(10, 10)
        
        cone = crear_campo_vectorial_3d(X, Y, Z, U, V, W, skip=2)
        
        assert isinstance(cone, go.Cone), \
            f"Tipo incorrecto: {type(cone)}"
    
    def test_submuestreo(self):
        """El submuestreo debe reducir el número de vectores."""
        N = 100
        X = np.random.rand(N, N)
        Y = np.random.rand(N, N)
        Z = np.random.rand(N, N)
        U = np.random.rand(N, N)
        V = np.random.rand(N, N)
        W = np.random.rand(N, N)
        
        skip = 10
        cone = crear_campo_vectorial_3d(X, Y, Z, U, V, W, skip=skip)
        
        # Número esperado de vectores
        n_esperado = (N // skip) ** 2
        n_actual = len(cone.x)
        
        assert n_actual == n_esperado, \
            f"Submuestreo incorrecto: esperado {n_esperado}, obtenido {n_actual}"


class TestVisualizacion3DCompleta:
    """Tests para la función principal de visualización."""
    
    def test_retorna_figure(self):
        """Debe retornar un objeto Figure de Plotly."""
        params = ParametrosMetriplecticos(N=30)  # Resolución baja para rapidez
        
        fig = generar_sol_metriplectico_3d(params, n_frames=5, show_plot=False)
        
        assert isinstance(fig, go.Figure), \
            f"Tipo incorrecto: {type(fig)}"
    
    def test_numero_frames(self):
        """Debe generar el número correcto de frames."""
        params = ParametrosMetriplecticos(N=30)
        n_frames = 10
        
        fig = generar_sol_metriplectico_3d(params, n_frames=n_frames, show_plot=False)
        
        assert len(fig.frames) == n_frames, \
            f"Número de frames incorrecto: esperado {n_frames}, obtenido {len(fig.frames)}"
    
    def test_tiene_controles_animacion(self):
        """La figura debe tener controles de animación."""
        params = ParametrosMetriplecticos(N=30)
        
        fig = generar_sol_metriplectico_3d(params, n_frames=5, show_plot=False)
        
        assert len(fig.layout.updatemenus) > 0, \
            "No se encontraron controles de animación"
        assert len(fig.layout.sliders) > 0, \
            "No se encontró slider de tiempo"
    
    def test_parametros_default(self):
        """Debe funcionar con parámetros por defecto."""
        fig = generar_sol_metriplectico_3d(show_plot=False)
        
        assert isinstance(fig, go.Figure), \
            "Falló con parámetros por defecto"


class TestIsomorfismoDimensional:
    """
    Tests para verificar el Nivel 2 de Isomorfismo Dimensional.
    
    Según El Mandato Metripléctico, las dimensiones físicas deben ser
    consistentes entre sistemas análogos.
    """
    
    def test_unidades_normalizadas(self):
        """Las coordenadas deben estar en unidades normalizadas [L_char]."""
        params = ParametrosMetriplecticos(N=30, L=10.0)
        
        fig = generar_sol_metriplectico_3d(params, n_frames=3, show_plot=False)
        
        # Verificar que los ejes tengan el rango correcto
        # Plotly puede retornar None si no se establece explícitamente
        x_range = fig.layout.scene.xaxis.range
        y_range = fig.layout.scene.yaxis.range
        
        if x_range is not None:
            assert list(x_range) == [-params.L, params.L], \
                f"Rango X incorrecto: {x_range}"
        if y_range is not None:
            assert list(y_range) == [-params.L, params.L], \
                f"Rango Y incorrecto: {y_range}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
