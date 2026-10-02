# -*- coding: utf-8 -*-
"""
Método de Euler - Ecuaciones Diferenciales
Aplicación en Streamlit para resolver dy/dx = f(x, y) mediante el método
numérico de Euler, con tabla de iteraciones y campo direccional.

El usuario puede ingresar su propia ecuación f(x, y) desde la interfaz
(por defecto: 2x + 3, con y(1) = 4).
"""

import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations, implicit_multiplication_application,
    convert_xor,
)

X_SYM, Y_SYM = sp.symbols("x y")

# Funciones y constantes permitidas dentro de la ecuación ingresada por el
# usuario. Se restringe deliberadamente el espacio de nombres para no
# ejecutar código arbitrario (no hay acceso a __builtins__).
ALLOWED_NAMES = {
    "x": X_SYM, "y": Y_SYM,
    "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
    "asin": sp.asin, "acos": sp.acos, "atan": sp.atan,
    "sinh": sp.sinh, "cosh": sp.cosh, "tanh": sp.tanh,
    "exp": sp.exp, "log": sp.log, "ln": sp.log,
    "sqrt": sp.sqrt, "Abs": sp.Abs, "abs": sp.Abs,
    "pi": sp.pi, "E": sp.E,
}

TRANSFORMATIONS = standard_transformations + (
    implicit_multiplication_application, convert_xor,
)

# Entorno "seguro" para evaluar la ecuación ingresada por el usuario: incluye
# los nombres internos que SymPy necesita (Integer, Symbol, Float, etc.) pero
# bloquea explícitamente los builtins de Python (open, __import__, eval,
# exec, ...) para que no se pueda ejecutar código arbitrario con la entrada
# de texto.
_SAFE_GLOBALS = {}
exec("from sympy import *", _SAFE_GLOBALS)
_SAFE_GLOBALS["__builtins__"] = {}


# =============================================================================
# 1. FUNCIONES DE LÓGICA MATEMÁTICA
# =============================================================================

def parse_equation(expr_str: str):
    """
    Convierte el texto ingresado por el usuario (ej. "2*x + 3", "x + y",
    "x**2 - y", "sin(x) - y") en una expresión simbólica de SymPy y en una
    función numérica f(x, y) evaluable con numpy.

    Lanza ValueError con un mensaje claro si la ecuación no es válida o
    usa símbolos distintos de x, y.
    """
    if not expr_str or not expr_str.strip():
        raise ValueError("Debes ingresar una expresión para f(x, y).")

    try:
        expr = parse_expr(
            expr_str,
            local_dict=ALLOWED_NAMES,
            global_dict=_SAFE_GLOBALS,
            transformations=TRANSFORMATIONS,
            evaluate=True,
        )
    except Exception as exc:
        raise ValueError(f"No se pudo interpretar la ecuación: {exc}")

    simbolos_usados = expr.free_symbols
    simbolos_invalidos = simbolos_usados - {X_SYM, Y_SYM}
    if simbolos_invalidos:
        nombres = ", ".join(str(s) for s in simbolos_invalidos)
        raise ValueError(
            f"La ecuación solo puede depender de x e y. "
            f"Símbolo(s) no reconocido(s): {nombres}"
        )

    f_num = sp.lambdify((X_SYM, Y_SYM), expr, modules=["numpy"])
    return expr, f_num


def differential_function(x, y, f_num):
    """
    Evalúa f(x, y) usando la función numérica ya generada a partir de la
    ecuación ingresada por el usuario (ver parse_equation).
    """
    return f_num(x, y)


def _evaluar_en_malla(f_num, X, Y):
    """
    Evalúa f_num sobre una malla (X, Y). Si la ecuación no depende de x
    y/o de y, lambdify puede devolver un escalar en lugar de un arreglo del
    mismo tamaño que la malla; esta función normaliza ese caso.
    """
    resultado = f_num(X, Y)
    resultado = np.asarray(resultado, dtype=float)
    if resultado.shape != X.shape:
        resultado = np.full_like(X, float(resultado), dtype=float)
    return resultado


def euler_method(x0: float, y0: float, x_final: float, h: float, f_num) -> pd.DataFrame:
    """
    Aplica el método de Euler explícito para dy/dx = f(x, y):
        y_{n+1} = y_n + h * f(x_n, y_n)
        x_{n+1} = x_n + h

    f_num es la función numérica obtenida de parse_equation().

    Retorna un DataFrame con el detalle de cada iteración:
    Iteración, x_n, y_n, f(x_n, y_n), h*f(x_n, y_n), y_{n+1}
    """
    if h == 0:
        raise ValueError("El tamaño de paso h no puede ser igual a 0.")
    if h < 0:
        raise ValueError(
            "El tamaño de paso h no puede ser negativo "
            "(este programa no implementa el cálculo hacia atrás)."
        )
    if x_final <= x0:
        raise ValueError("x_final debe ser mayor que x0.")

    n_iter = (x_final - x0) / h
    if n_iter > 100000:
        raise ValueError(
            "El número de iteraciones es excesivamente grande "
            "(más de 100,000). Aumenta h o reduce el intervalo."
        )

    n_steps = int(round(n_iter))

    filas = []
    x_n = float(x0)
    y_n = float(y0)

    for i in range(n_steps):
        f_xy = float(differential_function(x_n, y_n, f_num))
        if not np.isfinite(f_xy):
            raise ValueError(
                f"La ecuación produce un valor no finito (división por cero, "
                f"logaritmo de un número negativo, etc.) cerca de x = {x_n:.4f}, "
                f"y = {y_n:.4f}. Ajusta la ecuación, la condición inicial o el "
                f"intervalo."
            )
        incremento = h * f_xy
        y_next = y_n + incremento
        x_next = x_n + h

        filas.append({
            "Iteración": i,
            "x_n": x_n,
            "y_n": y_n,
            "f(x_n, y_n)": f_xy,
            "h * f(x_n, y_n)": incremento,
            "y_(n+1)": y_next,
        })

        x_n = x_next
        y_n = y_next

    # Se agrega el último punto alcanzado (x_final, y_final) como fila de
    # cierre, útil para graficar: es el resultado real de la última
    # iteración, no un dato inventado.
    filas.append({
        "Iteración": n_steps,
        "x_n": x_n,
        "y_n": y_n,
        "f(x_n, y_n)": np.nan,
        "h * f(x_n, y_n)": np.nan,
        "y_(n+1)": np.nan,
    })

    return pd.DataFrame(filas)


# =============================================================================
# 2. FUNCIÓN DE GRAFICACIÓN
# =============================================================================

def plot_direction_field(df: pd.DataFrame, x0: float, y0: float,
                          x_final: float, f_num, ecuacion_legible: str):
    """
    Dibuja el campo direccional de dy/dx = f(x, y) y superpone la
    solución aproximada de Euler.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    y_min = df["y_n"].min()
    y_max = df["y_n"].max()
    margen_y = max((y_max - y_min) * 0.3, 2)

    x_grid = np.linspace(x0, x_final, 20)
    y_grid = np.linspace(y_min - margen_y, y_max + margen_y, 20)
    X, Y = np.meshgrid(x_grid, y_grid)

    pendiente = _evaluar_en_malla(f_num, X, Y)
    pendiente = np.nan_to_num(pendiente, nan=0.0, posinf=1e6, neginf=-1e6)

    dx = np.ones_like(pendiente)
    dy = pendiente
    norma = np.sqrt(dx ** 2 + dy ** 2)
    norma[norma == 0] = 1.0
    dx_norm = dx / norma
    dy_norm = dy / norma

    ax.quiver(X, Y, dx_norm, dy_norm, color="gray", alpha=0.6,
              angles="xy", pivot="middle", width=0.003)

    ax.plot(df["x_n"], df["y_n"], color="#1f77b4", linewidth=2.5,
            label="Solución de Euler")

    ax.scatter([x0], [y0], color="black", zorder=5,
               label=f"Condición inicial (x₀={x0}, y₀={y0})")

    ax.set_title(f"Campo Direccional — dy/dx = {ecuacion_legible}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.3)

    from matplotlib.lines import Line2D
    leyenda_extra = Line2D([0], [0], color="gray", alpha=0.6, label="Campo direccional")
    handles, labels = ax.get_legend_handles_labels()
    handles.append(leyenda_extra)
    ax.legend(handles=handles, loc="best", fontsize=8)

    return fig


# =============================================================================
# 3. INTERFAZ GRÁFICA (STREAMLIT)
# =============================================================================

def render_table(df: pd.DataFrame):
    """
    Muestra un DataFrame como una tabla HTML real (filas y columnas con
    bordes visibles), construida a mano con estilos EN LÍNEA en cada
    celda. Se evita deliberadamente st.dataframe/st.table, porque ambas
    requieren PyArrow internamente, lo que falla en equipos donde una
    política de Control de aplicaciones bloquea ese archivo nativo. También
    se evita depender de un bloque <style> aparte (que algunos entornos
    pueden recortar), por eso cada <th>/<td> lleva su propio estilo.
    """
    th_style = (
        "border:1px solid #ccc; padding:6px 10px; background-color:#f0f2f6; "
        "text-align:right; white-space:nowrap; position:sticky; top:0;"
    )
    td_style = "border:1px solid #ccc; padding:6px 10px; text-align:right; white-space:nowrap;"

    encabezado = "".join(f'<th style="{th_style}">{col}</th>' for col in df.columns)

    filas_html = []
    for i, (_, fila) in enumerate(df.iterrows()):
        fondo = " background-color:#fafafa;" if i % 2 == 1 else ""
        celdas = "".join(
            f'<td style="{td_style}{fondo}">'
            f'{"—" if pd.isna(val) else val}</td>'
            for val in fila
        )
        filas_html.append(f"<tr>{celdas}</tr>")

    tabla_html = (
        '<div style="overflow-x:auto; overflow-y:auto; max-height:480px; '
        'border:1px solid #ccc; border-radius:4px;">'
        '<table style="border-collapse:collapse; width:100%; font-size:0.9rem;">'
        f"<thead><tr>{encabezado}</tr></thead>"
        f"<tbody>{''.join(filas_html)}</tbody>"
        "</table></div>"
    )
    st.markdown(tabla_html, unsafe_allow_html=True)


EJEMPLOS_ECUACIONES = {
    "2x + 3 (lineal, por defecto)": "2*x + 3",
    "x + y (lineal en x e y)": "x + y",
    "-y (decaimiento exponencial)": "-y",
    "x*y": "x*y",
    "y - x**2": "y - x**2",
    "sin(x)": "sin(x)",
}


def main():
    st.set_page_config(page_title="Método de Euler", layout="wide")
    st.title("Método de Euler - Ecuaciones Diferenciales")

    st.markdown(
        """
        Esta herramienta resuelve numéricamente una ecuación diferencial de
        primer orden **dy/dx = f(x, y)** mediante el **método de Euler**, y
        muestra la tabla de iteraciones junto con el campo direccional.
        """
    )

    # --- Sección: ecuación ---
    st.subheader("1. Ecuación diferencial")

    ejemplo_sel = st.selectbox(
        "Elige un ejemplo o escribe el tuyo abajo",
        options=list(EJEMPLOS_ECUACIONES.keys()),
        index=0,
    )

    ecuacion_texto = st.text_input(
        "f(x, y)  —  la ecuación a resolver es dy/dx = f(x, y)",
        value=EJEMPLOS_ECUACIONES[ejemplo_sel],
        help=(
            "Usa x e y como variables. Operadores: + - * / ** (o ^). "
            "Funciones disponibles: sin, cos, tan, exp, log, sqrt, sinh, "
            "cosh, tanh, Abs. Ejemplos: 2*x+3 · x+y · -y · x*y · sin(x)-y"
        ),
    )
    st.caption("La multiplicación implícita también funciona: puedes escribir 2x en vez de 2*x.")

    # --- Sección: parámetros ---
    st.subheader("2. Parámetros del método")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        x0 = st.number_input("x₀ (valor inicial de x)", value=1.0, step=0.1, format="%.4f")
    with col2:
        y0 = st.number_input("y₀ (valor inicial de y)", value=4.0, step=0.1, format="%.4f")
    with col3:
        x_final = st.number_input("x_final", value=5.0, step=0.1, format="%.4f")
    with col4:
        h = st.number_input("h (tamaño del paso)", value=0.1, step=0.01, format="%.4f")

    decimales = st.slider("Decimales a mostrar en la tabla", min_value=2, max_value=8, value=4)

    calcular = st.button("Calcular", type="primary")

    if "resultado" not in st.session_state:
        st.session_state.resultado = None

    if calcular:
        try:
            expr, f_num = parse_equation(ecuacion_texto)
            df_euler = euler_method(x0, y0, x_final, h, f_num)

            st.session_state.resultado = {
                "df": df_euler,
                "expr": expr,
                "f_num": f_num,
                "x0": x0, "y0": y0, "x_final": x_final, "h": h,
            }
            st.success(f"Cálculo completado: {len(df_euler) - 1} iteraciones.")
        except ValueError as e:
            st.error(f"Error en los datos ingresados: {e}")
            st.session_state.resultado = None

    resultado = st.session_state.resultado

    if resultado is not None:
        df = resultado["df"]
        expr = resultado["expr"]
        f_num = resultado["f_num"]
        x0r, y0r, x_finalr = resultado["x0"], resultado["y0"], resultado["x_final"]
        ecuacion_legible = str(expr)

        # --- Sección: tabla de resultados ---
        st.subheader("3. Tabla de valores (X, Y)")
        df_mostrar = df.round(decimales)
        render_table(df_mostrar)

        csv_buffer = io.StringIO()
        df_mostrar.to_csv(csv_buffer, index=False)
        st.download_button(
            label="Exportar tabla a CSV",
            data=csv_buffer.getvalue(),
            file_name="euler_resultados.csv",
            mime="text/csv",
        )

        # --- Sección: campo direccional ---
        st.subheader("4. Campo direccional")
        fig = plot_direction_field(df, x0r, y0r, x_finalr, f_num, ecuacion_legible)
        st.pyplot(fig)
    else:
        st.info("Configura la ecuación y los parámetros, luego presiona **Calcular** para ver los resultados.")


if __name__ == "__main__":
    main()