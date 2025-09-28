# --------------------------------------------------------------------
# IMPORTACIÓN DE LIBRERÍAS NECESARIAS
# --------------------------------------------------------------------
import streamlit as st
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from math import floor, ceil
import numpy_financial as npf

# --------------------------------------------------------------------
# CONFIGURACIÓN DE PÁGINA
# --------------------------------------------------------------------
st.set_page_config(page_title="Análisis de Inversión", layout="wide")
st.title("🏡 Análisis de Inversión")

# --------------------------------------------------------------------
# VARIABLES DE ESTILO (TAMAÑOS DE FUENTE BASE)
# --------------------------------------------------------------------
TITLE_SZ = 14
LABEL_SZ = 12

# ====================================================================
#                         ESTILOS CSS BASE
# ====================================================================
st.markdown("""
    <style>
    /* ===================================================================
       SECCIÓN: TABLAS
       =================================================================== */
    .stTable, .dataframe {
        border: 1px solid #e0e0e0 !important;
        border-radius: 10px !important;
        overflow: hidden !important;
        font-size: 14px !important;
    }
    .stTable thead th, .dataframe thead th {
        background: #004080 !important;
        color: #fff !important;
        text-align: center !important;
        font-size: 13px !important;
        padding: 6px 10px !important;
    }
    .stTable tbody td, .dataframe tbody td {
        text-align: center !important;
        vertical-align: middle !important;
        padding: 6px 8px !important;
        white-space: nowrap !important;
    }

    /* Ajuste de columnas específicas para tablas de plan de pagos */
    .stTable tbody td:nth-child(1),
    .dataframe tbody td:nth-child(1) { width: 40%; text-align: left !important; }
    .stTable tbody td:nth-child(2),
    .dataframe tbody td:nth-child(2) { width: 20%; }
    .stTable tbody td:nth-child(3),
    .dataframe tbody td:nth-child(3) { width: 25%; font-weight: bold; color: #004080; }
    .stTable tbody td:nth-child(4),
    .dataframe tbody td:nth-child(4) { width: 15%; }

    /* ===================================================================
       SECCIÓN: MÉTRICAS EN SIDEBAR
       =================================================================== */
    section[data-testid="stSidebar"] div[data-testid="stMetricValue"] {
        color: #004080 !important;
        font-weight: 800 !important;
        font-size: 20px !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stMetricLabel"] {
        color: #333 !important;
        font-size: 13px !important;
        font-weight: 600 !important;
    }
    </style>
""", unsafe_allow_html=True)

# ====================================================================
#                         MODO OSCURO / CLARO
# ====================================================================

# Estado inicial de modo oscuro
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

# Checkbox en sidebar
st.sidebar.checkbox("🌙 Dark Mode", key='dark_mode')

# Estilos modo oscuro
if st.session_state.dark_mode:
    st.markdown("""
        <style>
        body, .stApp {
            background-color: #121212 !important;
            color: #f5f5f5 !important;
        }
        h1, h2, h3, h4, h5, h6, p, label, span, div {
            color: #f5f5f5 !important;
        }
        section[data-testid="stSidebar"] {
            background-color: #1e1e1e !important;
        }
        .stTable thead th, .dataframe thead th {
            background: #333 !important;
            color: #fff !important;
        }
        .stTable tbody td, .dataframe tbody td {
            color: #e0e0e0 !important;
            background-color: #2C2C2C !important;
        }
        .stTable tbody tr:hover, .dataframe tbody tr:hover {
            background-color: #3A3A3A !important;
        }
        div[data-testid="stMetricValue"] {
            color: #00e676 !important; /* Verde para KPIs */
        }
        div[data-testid="stMetricLabel"] {
            color: #f5f5f5 !important;
        }
        input, textarea {
            background-color:#2C2C2C !important; color:#f5f5f5 !important;
        }
        </style>
    """, unsafe_allow_html=True)

# Estilos modo claro
else:
    st.markdown("""
        <style>
        body, .stApp { background-color: #ffffff !important; color: #000000 !important; }
        h1,h2,h3,h4,h5,h6,p,label,span,div { color: #000000 !important; }
        section[data-testid="stSidebar"] { background-color: #F8F9FA !important; }
        .stTable thead th, .dataframe thead th { background: #004080 !important; color: #fff !important; }
        .stTable tbody td, .dataframe tbody td { color: #000000 !important; background-color: #FFFFFF !important; }
        .stTable tbody tr:hover, .dataframe tbody tr:hover { background-color: #f1f1f1 !important; }
        input, textarea { background-color: #fffdf0 !important; color:#000000 !important; }
        div[data-testid="stMetricValue"] { color: #004080 !important; }
        div[data-testid="stMetricLabel"] { color: #333 !important; }
        </style>
    """, unsafe_allow_html=True)

# ====================================================================
#                         FUNCIONES AUXILIARES
# ====================================================================

def safe_number(x, default=0.0):
    """Convierte en float seguro para evitar crashes por NaN o None."""
    try:
        v = float(x)
        if pd.isna(v):
            return float(default)
        return v
    except Exception:
        return float(default)

def roi_equity_desde_flujo(df_num: pd.DataFrame):
    """ROI sobre equity usando pico de caja negativa acumulada como base."""
    if df_num.empty:
        return 0.0, 0.0
    ganancia_total = float(safe_number(df_num["Flujo Neto"].sum()))
    pico_equity = max(0.0, -float(safe_number(df_num["Flujo Acumulado"].min())))
    roi_equity = (ganancia_total / pico_equity * 100.0) if pico_equity > 0 else 0.0
    return roi_equity, pico_equity

def tasa_anual_a_mensual_compuesta(tasa_anual_pct: float) -> float:
    """Convierte tasa anual % a tasa mensual compuesta (decimal)."""
    return (1 + tasa_anual_pct/100.0) ** (1/12) - 1

def calcular_van(df_flujo: pd.DataFrame, tasa_descuento_anual_pct: float) -> float:
    """VAN descontando flujos netos mes a mes."""
    if df_flujo.empty:
        return 0.0
    tasa_mensual = tasa_anual_a_mensual_compuesta(tasa_descuento_anual_pct)
    flujos = df_flujo["Flujo Neto"].values
    return sum(flujo / ((1 + tasa_mensual) ** t) for t, flujo in enumerate(flujos))

def calcular_tir(df_flujo: pd.DataFrame) -> float:
    """
    TIR (anual %) a partir de flujos mensuales:
      1) irr mensual con numpy_financial.irr
      2) anualiza: (1 + irr_mensual)^12 - 1
    """
    if df_flujo.empty:
        return 0.0
    flujos = df_flujo["Flujo Neto"].values
    try:
        irr_mensual = float(npf.irr(flujos))
        if irr_mensual is None or pd.isna(irr_mensual):
            return 0.0
        irr_anual = (1 + irr_mensual) ** 12 - 1
        return irr_anual * 100.0
    except Exception:
        return 0.0

def generar_flujo_caja_desde_schedule(
    pagos_crudos_ingreso,
    egresos_por_mes: dict,
    tiempo_ejecucion_: int,
    fecha_inicio_,
    tasa_inflacion_anual: float,
    aplicar_inflacion_en_roi: bool = False
) -> pd.DataFrame:
    """
    Construye DataFrame con: Mes, Fecha, Ingresos, Egresos, Flujo Neto, Flujo Acumulado.
    - pagos_crudos_ingreso: lista [(mes:int, monto:float), ...] (0..N)
    - egresos_por_mes: dict {mes:int -> monto:float}
    - aplicar_inflacion_en_roi: infla egresos con tasa mensual compuesta si True
    """
    tasa_mensual = tasa_anual_a_mensual_compuesta(tasa_inflacion_anual)
    # Asegurar longitud hasta tiempo_ejecucion_
    ingresos_por_mes = [0.0] * (max(0, int(tiempo_ejecucion_)) + 1)
    for mes, monto in pagos_crudos_ingreso or []:
        mes = int(max(0, min(tiempo_ejecucion_, int(mes))))
        ingresos_por_mes[mes] += float(safe_number(monto))

    rows = []
    acumulado = 0.0
    for mes in range(0, int(tiempo_ejecucion_) + 1):
        ingresos = float(ingresos_por_mes[mes])
        egresos = float(safe_number(egresos_por_mes.get(mes, 0.0)))
        if aplicar_inflacion_en_roi:
            egresos *= (1 + tasa_mensual) ** mes
        neto = ingresos - egresos
        acumulado += neto
        # Fecha al último día del mes correspondiente
        fecha = (pd.to_datetime(fecha_inicio_) + pd.DateOffset(months=mes)).to_period('M').end_time.date()
        rows.append({
            "Mes": mes,
            "Fecha": fecha,
            "Ingresos": ingresos,
            "Egresos": egresos,
            "Flujo Neto": neto,
            "Flujo Acumulado": acumulado
        })
    return pd.DataFrame(rows)

def mes_por_avance(tiempo_total_meses: int, avance_pct: float) -> int:
    """Convierte % de avance en mes absoluto dentro del horizonte (redondeo floor)."""
    avance_pct = max(0.0, min(100.0, float(safe_number(avance_pct))))
    mes = int((avance_pct / 100.0) * int(max(0, tiempo_total_meses)))
    return min(max(mes, 0), int(tiempo_total_meses))

def monto_finiquito_calc(ingreso_escenario: float, base: str, pct: float, costo_total_: float) -> float:
    """Calcula finiquito dependiendo de base 'Venta' o 'Costo'."""
    base_val = float(ingreso_escenario) if base == "Venta" else float(costo_total_)
    return base_val * (float(safe_number(pct)) / 100.0)

def calcular_plan_pago(ingreso_escenario, pct_pago_inicial_, pct_finiquito_, base_finiquito_, costo_total_):
    """Devuelve (pago_inicial, remanente, finiquito)."""
    ingreso_escenario = float(safe_number(ingreso_escenario))
    pago_inicial = ingreso_escenario * (float(safe_number(pct_pago_inicial_)) / 100.0)
    finiquito = monto_finiquito_calc(ingreso_escenario, base_finiquito_, pct_finiquito_, float(safe_number(costo_total_)))
    remanente = max(0.0, ingreso_escenario - pago_inicial - finiquito)
    return pago_inicial, remanente, finiquito

def meses_bimensuales(tiempo_ejecucion_: int, mes_inicio: int) -> list:
    """Genera meses cada 2 a partir de mes_inicio+2 hasta < tiempo_ejecucion_."""
    salida = []
    m = int(mes_inicio) + 2
    while m < int(tiempo_ejecucion_):
        salida.append(m)
        m += 2
    return salida

def generar_plan_pagos_detallado(
    ingreso_escenario,
    pct_pago_inicial_,
    pct_finiquito_,
    base_finiquito_,
    costo_total_,
    tiempo_ejecucion_,
    fecha_inicio_,
    simbolo_,
    mes_pago_inicial=None,
    unico_final=False
):
    """
    Devuelve:
      - df_plan: DataFrame formateado para mostrar
      - crudo: lista [(mes, monto), ...] para alimentar el flujo
    """
    plan, crudo = [], []

    # ÚNICO PAGO FINAL
    if unico_final:
        mes_final = int(tiempo_ejecucion_)
        fecha_final = (pd.to_datetime(fecha_inicio_) + pd.DateOffset(months=mes_final)).to_period('M').end_time.date()
        plan.append({
            "Tipo de Pago": "Pago Único Final",
            "Fecha": fecha_final,
            "Monto": float(safe_number(ingreso_escenario)),
            "Porcentaje": 100.0
        })
        crudo.append((mes_final, float(safe_number(ingreso_escenario))))
        df_ = pd.DataFrame(plan)
        df_["Monto"] = df_["Monto"].apply(lambda x: f"{simbolo_}{x:,.0f}")
        df_["Porcentaje"] = df_["Porcentaje"].apply(lambda x: f"{x:,.2f}%")
        return df_, crudo

    # PAGO INICIAL + INTERMEDIOS + FINIQUITO
    pago_inicial, remanente, finiquito = calcular_plan_pago(
        ingreso_escenario, pct_pago_inicial_, pct_finiquito_, base_finiquito_, costo_total_
    )

    # Evitar forzar mes 0 cuando se espera un pago inicial por avance
    if mes_pago_inicial is None:
        mes_pago_inicial = max(1, int(tiempo_ejecucion_ // 2))
    # que quede entre 1 y tiempo_ejecucion_-1
    mes_pago_inicial = max(1, min(int(tiempo_ejecucion_) - 1, int(mes_pago_inicial)))

    if pago_inicial > 0:
        fecha_ini = (pd.to_datetime(fecha_inicio_) + pd.DateOffset(months=mes_pago_inicial)).to_period('M').end_time.date()
        plan.append({
            "Tipo de Pago": "Pago Inicial (mes escenario)",
            "Fecha": fecha_ini,
            "Monto": float(pago_inicial),
            "Porcentaje": (float(pago_inicial) / float(safe_number(ingreso_escenario)) * 100.0 if safe_number(ingreso_escenario) else 0.0)
        })
        crudo.append((mes_pago_inicial, float(pago_inicial)))

    meses_intermedios = meses_bimensuales(int(tiempo_ejecucion_), mes_pago_inicial)
    monto_intermedio = (float(remanente) / len(meses_intermedios)) if len(meses_intermedios) > 0 else 0.0
    for idx, mes_abs in enumerate(meses_intermedios):
        fecha_pago = (pd.to_datetime(fecha_inicio_) + pd.DateOffset(months=mes_abs)).to_period('M').end_time.date()
        plan.append({
            "Tipo de Pago": f"Pago Intermedio {idx+1}",
            "Fecha": fecha_pago,
            "Monto": float(monto_intermedio),
            "Porcentaje": (float(monto_intermedio) / float(safe_number(ingreso_escenario)) * 100.0 if safe_number(ingreso_escenario) else 0.0)
        })
        crudo.append((mes_abs, float(monto_intermedio)))

    mes_final = int(tiempo_ejecucion_)
    fecha_final = (pd.to_datetime(fecha_inicio_) + pd.DateOffset(months=mes_final)).to_period('M').end_time.date()
    if finiquito > 0:
        plan.append({
            "Tipo de Pago": "Pago Final (Contra Entrega)",
            "Fecha": fecha_final,
            "Monto": float(finiquito),
            "Porcentaje": (float(finiquito) / float(safe_number(ingreso_escenario)) * 100.0 if safe_number(ingreso_escenario) else 0.0)
        })
        crudo.append((mes_final, float(finiquito)))

    df_plan = pd.DataFrame(plan).sort_values("Fecha")
    df_plan["Monto"] = df_plan["Monto"].apply(lambda x: f"{simbolo_}{x:,.0f}")
    df_plan["Porcentaje"] = df_plan["Porcentaje"].apply(lambda x: f"{x:,.2f}%")
    return df_plan, crudo

def formatear_flujo(df_num: pd.DataFrame, simbolo_: str):
    df = df_num.copy()
    for c in ["Ingresos", "Egresos", "Flujo Neto", "Flujo Acumulado"]:
        df[c] = df[c].apply(lambda x: f"{simbolo_}{x:,.0f}")
    return df

def format_metric(label, value, simbolo="%"):
    color = "green" if float(safe_number(value)) >= 0 else "red"
    return f"<span class='kpi-roi' style='color:{color};'>{float(safe_number(value)):.2f}{simbolo}</span>"

BASE_DEFAULTS_12 = {
    "Terreno":      [30,0,30,0,40,0,0,0,0,0,0,0],
    "Construcción": [15,10,8,8,8,8,10,10,9,5,5,4],
    "Adicionales":  [20,20,20,40,0,0,0,0,0,0,0,0],
    "Misceláneos":  [0,0,0,0,0,0,0,0,0,0,100,0],
    "Comisión":     [0,0,0,0,0,0,0,0,0,0,0,100],
}

def resample_distribution(dist12, n_periodos):
    """
    Remuestrea la distribución de 12 buckets a n_periodos, conservando proporciones.
    Devuelve lista de porcentajes que suman 100 (aprox).
    """
    n_periodos = int(max(1, n_periodos))
    if n_periodos == 12:
        return dist12[:]
    out = []
    for k in range(n_periodos):
        start = k * 12.0 / n_periodos
        end   = (k + 1) * 12.0 / n_periodos
        total = 0.0
        i0 = int(floor(start))
        i1 = int(ceil(end))
        for i in range(i0, i1):
            if 0 <= i < 12:
                left, right = max(start, i), min(end, i + 1)
                overlap = max(0.0, right - left)
                total += dist12[i] * overlap
        out.append(total)
    s = sum(out)
    if s > 0:
        out = [round(x * 100.0 / s, 2) for x in out]
    drift = round(100.0 - sum(out), 2)
    if abs(drift) >= 0.01 and len(out) > 0:
        peso_total = sum(out) if sum(out) > 0 else 1.0
        ajuste = [round((x / peso_total) * drift, 2) for x in out]
        out = [round(x + a, 2) for x, a in zip(out, ajuste)]
        diff = round(100.0 - sum(out), 2)
        if abs(diff) >= 0.01:
            out[-1] = round(out[-1] + diff, 2)
    return out

def altura_dinamica(df, fila_px=40, extra=60):
    """Altura recomendada de dataframe para no cortar filas."""
    return min(600, len(df) * fila_px + extra)

def agregar_totales(df_num: pd.DataFrame, simbolo_: str):
    """
    Agrega fila TOTAL al final; setea Flujo Acumulado TOTAL = 0
    para cumplir con la presentación solicitada.
    """
    if df_num.empty:
        return pd.DataFrame()
    df_total = df_num.copy()
    total_ing = float(safe_number(df_total["Ingresos"].sum()))
    total_egr = float(safe_number(df_total["Egresos"].sum()))
    total_neto = float(safe_number(df_total["Flujo Neto"].sum()))
    df_total.loc["TOTAL"] = {
        "Mes": "",
        "Fecha": "",
        "Ingresos": total_ing,
        "Egresos": total_egr,
        "Flujo Neto": total_neto,
        "Flujo Acumulado": 0
    }
    df_fmt = formatear_flujo(df_total, simbolo_)
    df_fmt = df_fmt.style.apply(
        lambda x: ["font-weight: bold" if x.name == "TOTAL" else "" for _ in x],
        axis=1
    )
    return df_fmt

# ===============================================================
#                       SIDEBAR: PARÁMETROS
# ===============================================================

st.sidebar.header("📌 Parámetros del Proyecto")

# 1. Premisas básicas
st.sidebar.subheader("1. 📝 Premisas del Proyecto")
nombre_proyecto = st.sidebar.text_input("Nombre del Proyecto", "Villa de Lujo en Corales 79")
ubicacion_proyecto = st.sidebar.text_input("Ubicación", "Punta Cana, República Dominicana")
tiempo_ejecucion = st.sidebar.number_input("Tiempo de ejecución (meses)", min_value=1, value=24, step=1)
fecha_inicio = st.sidebar.date_input("Fecha de Inicio", datetime.date.today())
moneda = st.sidebar.selectbox("Moneda", ["USD", "EUR"], index=0)
simbolo_moneda = "$" if moneda == "USD" else "€"

# 2. Terreno
st.sidebar.subheader("2. 🌱 Inversión en Terreno")
precio_m2_solar = st.sidebar.number_input("Precio m² solar", min_value=0.0, value=360.0, step=10.0)
superficie_solar = st.sidebar.number_input("Superficie solar (m²)", min_value=0.0, value=4100.0, step=50.0)
costo_solar = float(safe_number(precio_m2_solar * superficie_solar))
st.sidebar.metric("Costo Total Terreno", f"{simbolo_moneda}{costo_solar:,.0f}")

# 3. Construcción
st.sidebar.subheader("3. 🏗️ Inversión en Construcción")
area_construida = st.sidebar.number_input("Área construida (m²)", min_value=0.0, value=1250.0, step=20.0)
costo_m2 = st.sidebar.number_input("Costo m² construcción", min_value=0.0, value=1750.0, step=50.0)
area_humeda = st.sidebar.number_input("Área húmeda (m²)", min_value=0.0, value=500.0, step=5.0)
costo_m2_humeda = st.sidebar.number_input("Costo m² área húmeda", min_value=0.0, value=900.0, step=50.0)
costo_construccion = float(safe_number(area_construida * costo_m2 + area_humeda * costo_m2_humeda))
st.sidebar.metric("Costo Total Construcción", f"{simbolo_moneda}{costo_construccion:,.0f}")

# 4. Complementarios
st.sidebar.subheader("4. 📑 Gastos Complementarios")
pct_permisos = st.sidebar.slider("Permisos (%)", 0.0, 10.0, 1.5, step=0.1)
pct_diseno = st.sidebar.slider("Diseño (%)", 0.0, 10.0, 3.0, step=0.1)
pct_legales = st.sidebar.slider("Legales (%)", 0.0, 5.0, 0.5, step=0.1)
pct_imprevistos = st.sidebar.slider("Imprevistos (%)", 0.0, 10.0, 2.0, step=0.1)
pct_adicionales = float(safe_number(pct_permisos + pct_diseno + pct_legales + pct_imprevistos))
costo_adicionales = (costo_construccion + costo_solar) * pct_adicionales / 100.0
st.sidebar.metric("Costo Total Complementarios", f"{simbolo_moneda}{costo_adicionales:,.0f}")

# 5. Acabados y extras
st.sidebar.subheader("5. 🎨 Acabados y Extras")
indoor_furniture_default = float(safe_number(area_construida * 300))
use_indoor = st.sidebar.checkbox("Incluir Indoor Furnitures", value=False)
indoor_furniture = st.sidebar.number_input(
    "Indoor Furnitures ($)", min_value=0.0, value=float(indoor_furniture_default), step=10000.0
) if use_indoor else 0.0

use_lighting = st.sidebar.checkbox("Incluir Lighting", value=False)
lighting = st.sidebar.number_input(
    "Lighting (interior y exterior) ($)", min_value=0.0, value=25000.0, step=5000.0
) if use_lighting else 0.0

use_closets = st.sidebar.checkbox("Incluir Closets y Wardrobes", value=False)
closets = st.sidebar.number_input(
    "Interior Closets and Wardrobes ($)", min_value=0.0, value=57600.0, step=5000.0
) if use_closets else 0.0

use_automation = st.sidebar.checkbox("Incluir Home Automation", value=True)
automation = st.sidebar.number_input(
    "Basic Home Automation ($)", min_value=0.0, value=10000.0, step=2000.0
) if use_automation else 0.0

use_marketing = st.sidebar.checkbox("Incluir Marketing", value=True)
marketing = st.sidebar.number_input(
    "Marketing ($)", min_value=0.0, value=2500.0, step=500.0
) if use_marketing else 0.0

use_paisajismo = st.sidebar.checkbox("Incluir Paisajismo", value=True)
paisajismo_pct = st.sidebar.number_input(
    "Paisajismo (% sobre costo base construcción)", min_value=0.0, max_value=100.0, value=4.5, step=0.1
) if use_paisajismo else 0.0
costo_base_construccion = float(safe_number(area_construida * costo_m2))
paisajismo = costo_base_construccion * (float(safe_number(paisajismo_pct)) / 100.0) if use_paisajismo else 0.0

costo_misc = float(safe_number(indoor_furniture + lighting + closets + automation + marketing + paisajismo))
st.sidebar.metric("Costo Total Acabados y Extras", f"{simbolo_moneda}{costo_misc:,.0f}")

# 6. Costo total
costo_total = float(safe_number(costo_solar + costo_construccion + costo_adicionales + costo_misc))

# 7. Proyección de Venta
st.sidebar.subheader("6. 📈 Proyección de Venta")
precio_venta_m2 = st.sidebar.number_input("Precio venta m²", min_value=0.0, value=4400.0, step=100.0)
ingreso_total = float(safe_number(precio_venta_m2 * area_construida))
st.sidebar.metric("Precio Total de Venta", f"{simbolo_moneda}{ingreso_total:,.0f}")

# 8. Comisiones de corretaje
st.sidebar.subheader("7. 💼 Comisiones de Corretaje")
pct_comision = st.sidebar.number_input("Comisión de corretaje (%)", min_value=0, max_value=100, value=8, step=1)
comision_venta = float(safe_number(ingreso_total * (pct_comision / 100.0)))
st.sidebar.metric("Total Comisión", f"{simbolo_moneda}{comision_venta:,.0f}")

# 9. Plan de pagos
st.sidebar.subheader("8. 💰 Plan de Pagos")
pct_pago_inicial = st.sidebar.number_input("Pago inicial (% sobre Venta)", min_value=0, max_value=100, value=30, step=1)
pct_finiquito   = st.sidebar.number_input("Finiquito (% base seleccionada)", min_value=0, max_value=100, value=20, step=1)
base_finiquito  = st.sidebar.selectbox("Base del finiquito", ["Venta", "Costo"], index=0)
avance_normal   = st.sidebar.slider("Avance escenario NORMAL (%)", 0, 100, 70)
avance_optimista= st.sidebar.slider("Avance escenario OPTIMISTA (%)", 0, 100, 50)

# 10. Variables financieras
st.sidebar.subheader("10. 📉 Variables Financieras")
tasa_inflacion_anual = st.sidebar.number_input("Inflación anual (%)", min_value=0.0, value=2.0, step=0.1)
aplicar_inflacion_en_roi = st.sidebar.checkbox("Aplicar inflación en ROI", value=False)

with st.sidebar:
    if aplicar_inflacion_en_roi:
        inflacion_acumulada = (1 + tasa_inflacion_anual/100.0) ** (float(safe_number(tiempo_ejecucion))/12.0)
        ingreso_real_ajustado = ingreso_total / inflacion_acumulada if inflacion_acumulada else ingreso_total
        st.metric("Ingreso Real (ajustado por inflación)", f"{simbolo_moneda}{ingreso_real_ajustado:,.0f}")
    else:
        st.markdown("<div style='opacity:0.5'>Ingreso Real (ajustado por inflación)</div>", unsafe_allow_html=True)
        st.metric(" ", "—")

# 10b. Financiamiento (definir antes de usar)
st.sidebar.subheader("10b. 🏦 Financiamiento")

usar_financiamiento = st.sidebar.checkbox("Usar financiamiento bancario", value=False)
pct_financiamiento  = st.sidebar.number_input("Porcentaje a financiar (%)", min_value=0.0, max_value=100.0, value=20.0, step=1.0)
tasa_interes_anual  = st.sidebar.number_input("Tasa interés anual (%)", min_value=0.0, max_value=50.0, value=10.0, step=0.5)
plazo_financiamiento= st.sidebar.number_input("Plazo (meses)", min_value=0, max_value=360, value=12, step=1)

# 10c. Costo de Oportunidad
st.sidebar.subheader("10c. 💸 Costo de Oportunidad")
usar_costo_oportunidad = st.sidebar.checkbox("Calcular Costo de Oportunidad", value=False)
tasa_costo_oportunidad = st.sidebar.number_input(
    "Tasa anual de costo de oportunidad (%)",
    min_value=0.0, max_value=50.0, value=5.0, step=0.1
)

# espacio justo debajo de la tasa
co_placeholder = st.sidebar.container()

# 11. Tasa de descuento
st.sidebar.subheader("11. 💰 Tasa de Descuento")
tasa_descuento = st.sidebar.number_input(
    "Tasa de descuento (%)",
    min_value=0.0, value=15.0, step=0.5,
    help="La tasa de descuento refleja el costo de capital o la rentabilidad mínima exigida. Se usa para traer los flujos mensuales a valor presente."
)

# Pre-cálculo de ganancia para usar en sidebar
ingreso_bruto = float(safe_number(ingreso_total))
ingreso_neto = float(safe_number(ingreso_bruto - comision_venta))
ganancia_simple = float(safe_number(ingreso_neto - costo_total))

# ===============================================================
#        ISR, Valor Residual y Sensibilidad Rápida
# ===============================================================

st.sidebar.subheader("12. 🏦 Ajustes Fiscales y Residuales")

# --- ISR ---
usar_isr = st.sidebar.checkbox("Aplicar ISR sobre Ganancia", value=False)
tasa_isr = st.sidebar.number_input("Tasa ISR (%)", min_value=0.0, max_value=50.0, value=27.0, step=0.5)
monto_isr = 0.0
if usar_isr and ganancia_simple > 0:
    monto_isr = ganancia_simple * (tasa_isr / 100.0)
    st.sidebar.metric("ISR Estimado", f"{simbolo_moneda}{monto_isr:,.0f}")
else:
    st.sidebar.markdown("<div style='opacity:0.5'>ISR Estimado</div>", unsafe_allow_html=True)
    st.sidebar.metric(" ", "—")

# --- Valor Residual Terreno ---
usar_residual = st.sidebar.checkbox("Incluir Valor Residual Terreno", value=False)  # antes value=True
valor_residual = costo_solar if usar_residual else 0.0
if usar_residual:
    st.sidebar.metric("Valor Residual Terreno", f"{simbolo_moneda}{valor_residual:,.0f}")
else:
    st.sidebar.markdown("<div style='opacity:0.5'>Valor Residual Terreno</div>", unsafe_allow_html=True)
    st.sidebar.metric(" ", "—")

# --- Sensibilidad Rápida ---
st.sidebar.subheader("12. ⚡ Sensibilidad Rápida")

sens_precio = st.sidebar.slider("Precio de Venta (±%)", -20, 20, 0, step=1)
sens_costo  = st.sidebar.slider("Costo de Construcción (±%)", -20, 20, 0, step=1)

# Recalcular variables con sensibilidad aplicada
precio_venta_sens = ingreso_total * (1 + sens_precio/100.0)
costo_total_sens  = costo_total * (1 + sens_costo/100.0)

# Calculate sensitive commission
comision_sens = float(safe_number(precio_venta_sens * (pct_comision / 100.0)))

# Calculate sensitive simple profit for ISR
ingreso_bruto_sens = float(safe_number(precio_venta_sens))
ingreso_neto_sens = float(safe_number(ingreso_bruto_sens - comision_sens))
ganancia_simple_sens = float(safe_number(ingreso_neto_sens - costo_total_sens))

# Calculate sensitive ISR
monto_isr_sens = 0.0
if usar_isr and ganancia_simple_sens > 0:
    monto_isr_sens = ganancia_simple_sens * (tasa_isr / 100.0)

ganancia_sens     = precio_venta_sens - costo_total_sens - comision_sens - monto_isr_sens + valor_residual

st.sidebar.metric("Venta Sensible", f"{simbolo_moneda}{precio_venta_sens:,.0f}")
st.sidebar.metric("Costo Sensible", f"{simbolo_moneda}{costo_total_sens:,.0f}")
st.sidebar.metric("Ganancia Sensible", f"{simbolo_moneda}{ganancia_sens:,.0f}")

# (Opcional) Propagar sensibilidad al resto del modelo
ingreso_total  = precio_venta_sens
costo_total    = costo_total_sens
comision_venta = comision_sens
monto_isr      = monto_isr_sens

capital_banco = 0.0
cuota_banco = 0.0
interes_total = 0.0
if usar_financiamiento and pct_financiamiento > 0 and plazo_financiamiento > 0:
    capital_banco = float(safe_number(costo_total * (pct_financiamiento / 100.0)))
    tasa_mensual = tasa_anual_a_mensual_compuesta(tasa_interes_anual)
    if tasa_mensual > 0 and plazo_financiamiento > 0:
        cuota_banco = npf.pmt(tasa_mensual, plazo_financiamiento, -capital_banco)
        interes_total = cuota_banco * plazo_financiamiento - capital_banco
    else: # Handle zero interest or zero term cases
        cuota_banco = capital_banco / plazo_financiamiento if plazo_financiamiento > 0 else 0.0
        interes_total = 0.0 # No interest if rate is zero
    st.sidebar.metric("Monto Financiado", f"{simbolo_moneda}{capital_banco:,.0f}")
    st.sidebar.metric("Total Intereses a Pagar (compuesto)", f"{simbolo_moneda}{interes_total:,.0f}")
    st.sidebar.metric("Cuota Mensual (compuesta)", f"{simbolo_moneda}{cuota_banco:,.0f}")

# ===============================================================
#                        RESUMEN PRINCIPAL
# ===============================================================

ingreso_bruto = float(safe_number(ingreso_total))
ingreso_neto = float(safe_number(ingreso_bruto - comision_venta))
ganancia_simple = float(safe_number(ingreso_neto - costo_total))

st.markdown(f"## 📌 Proyecto: **{nombre_proyecto}**")
st.markdown(f"### 🌍 Ubicación: *{ubicacion_proyecto}*")
st.markdown(f"### 📐 Superficie del Lote: **{superficie_solar:,.0f} m²**")
st.markdown(f"### 🏗️ Superficie de Construcción: **{area_construida:,.0f} m²**")
st.markdown(f"### 💵 Precio de Venta: **{simbolo_moneda}{ingreso_bruto:,.0f}**")

st.markdown("---")
st.markdown("<h3 style='font-size: 36px;'>📊 Resumen del Proyecto</h3>", unsafe_allow_html=True)
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Costo Total de Inversión", f"{simbolo_moneda}{costo_total:,.0f}")
k2.metric("Ingreso Bruto por Venta", f"{simbolo_moneda}{ingreso_bruto:,.0f}")
k3.metric("Ingreso Neto (después de comisión)", f"{simbolo_moneda}{ingreso_neto:,.0f}")
k4.metric("Ganancia Neta (simple)", f"{simbolo_moneda}{ganancia_simple:,.0f}")
roi_placeholder = k5.empty()

# ===============================================================
#        DISTRIBUCIÓN EDITABLE DE EGRESOS POR RUBROS/PERÍODOS
# ===============================================================

st.subheader("📉 Flujo de Caja de Egresos - Distribución por Rubros")

frecuencia_egresos = st.selectbox(
    "Frecuencia de desembolsos",
    ["Mensual", "Bimestral", "Trimestral"],
    index=1
)

if frecuencia_egresos == "Mensual":
    n_periodos = int(safe_number(tiempo_ejecucion))
    periodo_len = 1
elif frecuencia_egresos == "Bimestral":
    n_periodos = max(1, int(safe_number(tiempo_ejecucion)) // 2)
    periodo_len = 2
else:
    n_periodos = max(1, int(safe_number(tiempo_ejecucion)) // 3)
    periodo_len = 3

st.markdown(f"**Número de períodos: {n_periodos}**")

# Rubros y costos base
costo_rubros = {
    "Terreno": float(safe_number(costo_solar)),
    "Construcción": float(safe_number(costo_construccion)),
    "Adicionales": float(safe_number(costo_adicionales)),
    "Misceláneos": float(safe_number(costo_misc)),
    "Comisión": float(safe_number(comision_venta)),
}
rubros = list(costo_rubros.keys())

# Defaults re-muestreados
data_default = {}
for rubro in rubros:
    base12 = BASE_DEFAULTS_12.get(rubro, [0]*12)
    data_default[rubro] = resample_distribution(base12, n_periodos)

df_base_pct = pd.DataFrame(data_default, index=[f"P{i+1}" for i in range(n_periodos)]).T

# Editor de la distribución
if hasattr(st, "data_editor"):
    edit_df_pct = st.data_editor(df_base_pct, num_rows="dynamic", use_container_width=True)
else:
    edit_df_pct = st.experimental_data_editor(df_base_pct, num_rows="dynamic", use_container_width=True)

# Validación de totales 100% por rubro
valid = True
for rubro in edit_df_pct.index:
    total = float(safe_number(edit_df_pct.loc[rubro].sum()))
    if abs(total - 100) > 0.01:
        st.warning(f"La distribución de {rubro} no suma 100% (actual: {total:.2f}%)")
        valid = False
if valid:
    st.success("✅ Todas las distribuciones suman 100%")

# Conversión a valores monetarios
df_valores = edit_df_pct.copy()
for rubro in rubros:
    df_valores.loc[rubro] = edit_df_pct.loc[rubro].astype(float) * float(costo_rubros[rubro]) / 100.0

# Construcción de egresos por mes respetando frecuencia
egresos_por_mes = {}
for j, col in enumerate(df_valores.columns, start=1):
    monto_periodo = float(safe_number(df_valores[col].sum()))
    if n_periodos > 1:
        mes = j * periodo_len
    else:
        mes = int(safe_number(tiempo_ejecucion))
    if monto_periodo > 0:
        egresos_por_mes[mes] = egresos_por_mes.get(mes, 0.0) + monto_periodo

egresos_base = egresos_por_mes.copy()
egresos_con_intereses = egresos_base.copy()  # Placeholder por si luego prorrateas intereses en el flujo

# -- Cálculo del costo de oportunidad (compuesto por egreso hasta el final del proyecto)
intereses_oportunidad = 0.0
if usar_costo_oportunidad and tasa_costo_oportunidad > 0:
    tasa_mensual_oport = (1 + tasa_costo_oportunidad/100.0) ** (1/12) - 1
    for mes, egreso in egresos_base.items():
        if egreso > 0:
            meses_restantes = max(0, int(safe_number(tiempo_ejecucion)) - int(safe_number(mes)))
            intereses_oportunidad += egreso * ((1 + tasa_mensual_oport) ** meses_restantes - 1)

    with co_placeholder:
        st.metric("Intereses Generados (Costo de Oportunidad)",
                  f"{simbolo_moneda}{intereses_oportunidad:,.0f}")
else:
    with co_placeholder:
        st.markdown("<div style='opacity:0.5'>Intereses Generados (Costo de Oportunidad)</div>", unsafe_allow_html=True)
        st.metric(" ", "—")

# Gráfico pastel
st.markdown("### 🥧 Distribución porcentual de Egresos")
col_pie1, col_pie2 = st.columns([1,1])
with col_pie1:
    fig_pie, ax_pie = plt.subplots(figsize=(2.2,2.2), dpi=300)
    egresos_totales = {rubro: float(safe_number(df_valores.loc[rubro].sum())) for rubro in rubros}
    wedges, texts, autotexts = ax_pie.pie(
        egresos_totales.values(),
        autopct='%1.1f%%',
        startangle=90,
        colors=plt.cm.Pastel1.colors,
        pctdistance=0.6,
        textprops={'fontsize': 7}
    )
    for autotext in autotexts:
        autotext.set_color("black")
        autotext.set_fontsize(7)
        autotext.set_fontweight("bold")
    ax_pie.legend(
        wedges, egresos_totales.keys(),
        title="Rubros",
        loc="center right",
        bbox_to_anchor=(-0.25, 0.5),
        fontsize=6,
        title_fontsize=7
    )
    ax_pie.axis("equal")
    st.pyplot(fig_pie)

with col_pie2:
    egresos_sorted = sorted(egresos_totales.items(), key=lambda x: x[1], reverse=True)[:3]
    st.markdown("### 🔝 Top 3 Rubros de Egresos")
    for rubro, monto in egresos_sorted:
        pct = (monto / max(1e-9, sum(egresos_totales.values()))) * 100
        st.markdown(
            f"<div style='font-size:20px; font-weight:bold; color:#004080;'>"
            f"▪ {rubro}: {simbolo_moneda}{monto:,.0f} <span style='color:#222;'>({pct:.1f}%)</span>"
            f"</div>",
            unsafe_allow_html=True
        )

# Gráfico de barras apiladas por período
st.markdown("### 📊 Distribución visual de Egresos")
df_graph = df_valores.drop("TOTAL US$", axis=0, errors="ignore").drop("TOTAL US$", axis=1, errors="ignore")
fig, ax = plt.subplots(figsize=(12,6))
colors = plt.cm.Set3.colors
try:
    df_graph.T.plot(kind="bar", stacked=True, ax=ax, color=colors, edgecolor="black", linewidth=0.7)
except Exception:
    # Fallback muy raro; stackplot es continuo, pero lo dejamos para evitar crash.
    ax.stackplot(range(len(df_graph.columns)), df_graph.values, labels=df_graph.index, colors=colors)

totals = df_graph.T.sum(axis=1)
for i, total in enumerate(totals):
    ax.text(i, total + (total*0.02 if total else 0), f"{simbolo_moneda}{float(safe_number(total)):,.0f}",
            ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.set_ylabel("Egresos", fontsize=LABEL_SZ, fontweight="bold")
ax.set_xlabel("Periodo", fontsize=LABEL_SZ, fontweight="bold")
freq_label = "Mensual" if frecuencia_egresos == "Mensual" else "Bimestral" if frecuencia_egresos == "Bimestral" else "Trimestral"
ax.set_title(f"Flujo de Egresos {freq_label}", fontsize=14, fontweight="bold", pad=15)
ax.legend(title="Rubros", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
ax.grid(axis="y", linestyle="--", alpha=0.7)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
st.pyplot(fig)

# ===============================================================
#                 ESCENARIOS DE VENTA Y PLAN DE PAGOS
# ===============================================================

st.subheader("📈 Escenarios de Venta y Plan de Pagos")

mostrar_plan_pagos_detallado = st.checkbox("Mostrar Plan de Pagos Detallado", value=True)
mostrar_flujo_caja_escenarios = st.checkbox("Mostrar Flujo de Caja por Escenario", value=True)

ingreso_pesimista = float(safe_number(ingreso_total))
ingreso_normal = float(safe_number(ingreso_total))
ingreso_optimista = float(safe_number(ingreso_total))

# Planes de pago
plan_det_pes, pagos_pes = generar_plan_pagos_detallado(
    ingreso_pesimista,
    pct_pago_inicial_=0, pct_finiquito_=0, base_finiquito_=base_finiquito,
    costo_total_=costo_total, tiempo_ejecucion_=tiempo_ejecucion, fecha_inicio_=fecha_inicio,
    simbolo_=simbolo_moneda, mes_pago_inicial=None, unico_final=True
)
mes_ini_nor = mes_por_avance(int(safe_number(tiempo_ejecucion)), float(safe_number(avance_normal)))
plan_det_nor, pagos_nor = generar_plan_pagos_detallado(
    ingreso_normal,
    pct_pago_inicial_=pct_pago_inicial,
    pct_finiquito_=pct_finiquito,
    base_finiquito_=base_finiquito,
    costo_total_=costo_total,
    tiempo_ejecucion_=tiempo_ejecucion,
    fecha_inicio_=fecha_inicio,
    simbolo_=simbolo_moneda,
    mes_pago_inicial=mes_ini_nor if mes_ini_nor > 0 else None,
    unico_final=False
)
mes_ini_opt = mes_por_avance(int(safe_number(tiempo_ejecucion)), float(safe_number(avance_optimista)))
plan_det_opt, pagos_opt = generar_plan_pagos_detallado(
    ingreso_optimista, pct_pago_inicial_=pct_pago_inicial, pct_finiquito_=pct_finiquito,
    base_finiquito_=base_finiquito, costo_total_=costo_total, tiempo_ejecucion_=tiempo_ejecucion,
    fecha_inicio_=fecha_inicio, simbolo_=simbolo_moneda, mes_pago_inicial=mes_ini_opt, unico_final=False
)

# Incorporar valor residual al flujo de ingresos si está activado
if usar_residual and valor_residual > 0:
    ultimo_mes = int(safe_number(tiempo_ejecucion))
    # Add to each scenario's payments
    pagos_pes.append((ultimo_mes, valor_residual))
    pagos_nor.append((ultimo_mes, valor_residual))
    pagos_opt.append((ultimo_mes, valor_residual))

# Horizonte máximo (considera también el plazo de financiación si quieres extenderlo)
horizonte = max(int(safe_number(tiempo_ejecucion)), int(safe_number(plazo_financiamiento)) if usar_financiamiento else 0)

# Flujos nominales por escenario
df_f_pes_num = generar_flujo_caja_desde_schedule(
    pagos_pes, egresos_base, horizonte, fecha_inicio, float(safe_number(tasa_inflacion_anual)), aplicar_inflacion_en_roi=False
)
df_f_nor_num = generar_flujo_caja_desde_schedule(
    pagos_nor, egresos_con_intereses, horizonte, fecha_inicio, float(safe_number(tasa_inflacion_anual)), aplicar_inflacion_en_roi
)
df_f_opt_num = generar_flujo_caja_desde_schedule(
    pagos_opt, egresos_con_intereses, horizonte, fecha_inicio, float(safe_number(tasa_inflacion_anual)), aplicar_inflacion_en_roi
)

# ROI nominal por escenario
roi_pes, equity_pes = roi_equity_desde_flujo(df_f_pes_num)
roi_nor, equity_nor = roi_equity_desde_flujo(df_f_nor_num)
roi_opt, equity_opt = roi_equity_desde_flujo(df_f_opt_num)

# VAN por escenario
van_pes = calcular_van(df_f_pes_num, float(safe_number(tasa_descuento)))
van_nor = calcular_van(df_f_nor_num, float(safe_number(tasa_descuento)))
van_opt = calcular_van(df_f_opt_num, float(safe_number(tasa_descuento)))

# TIR por escenario
tir_pes = calcular_tir(df_f_pes_num)
tir_nor = calcular_tir(df_f_nor_num)
tir_opt = calcular_tir(df_f_opt_num)

# ===============================================================
#                           KPIs
# ===============================================================

st.markdown("---")
st.markdown("<h3 style='font-size: 28px;'>💹 KPIs Financieros</h3>", unsafe_allow_html=True)

escenario_kpi = st.selectbox(
    "Escenario base para ROI",
    ["Pesimista", "Normal", "Optimista"],
    index=1
)

if escenario_kpi == "Pesimista":
    pagos_sel, egresos_sel_base = pagos_pes, egresos_base
elif escenario_kpi == "Optimista":
    pagos_sel, egresos_sel_base = pagos_opt, egresos_con_intereses
else:
    pagos_sel, egresos_sel_base = pagos_nor, egresos_con_intereses

# Congelar ROI nominal sin ajustes (para KPI principal)
df_nominal_base = generar_flujo_caja_desde_schedule(
    pagos_sel, egresos_sel_base, horizonte, fecha_inicio, float(safe_number(tasa_inflacion_anual)), aplicar_inflacion_en_roi=False
)
roi_nominal, equity_nominal = roi_equity_desde_flujo(df_nominal_base)

# ROI Ajustado (inflación, financiación, costo oportunidad)
df_ajustado = df_nominal_base.copy()
roi_ajustado_calc = roi_nominal
ajustes = []

# Inflación
if aplicar_inflacion_en_roi:
    df_ajustado = generar_flujo_caja_desde_schedule(
        pagos_sel, egresos_sel_base, horizonte, fecha_inicio, float(safe_number(tasa_inflacion_anual)), aplicar_inflacion_en_roi=True
    )
    roi_ajustado_calc, _ = roi_equity_desde_flujo(df_ajustado)
    ajustes.append("inflación")

# Financiamiento simple (resta intereses totales a la ganancia)
if usar_financiamiento and interes_total > 0:
    ganancia_ajustada = float(safe_number(df_ajustado["Flujo Neto"].sum() - interes_total))
    roi_ajustado_calc = (ganancia_ajustada / float(safe_number(equity_nominal)) * 100.0) if float(safe_number(equity_nominal)) > 0 else 0.0
    ajustes.append("financiamiento simple")

# Costo de oportunidad (solo informativo)
if usar_costo_oportunidad and intereses_oportunidad > 0:
    st.markdown("### 📌 Comparativo Costo de Oportunidad")
    st.markdown(
        f"Si en lugar de invertir en el proyecto invertías cada egreso en un instrumento al "
        f"**{float(safe_number(tasa_costo_oportunidad)):.2f}% anual**, "
        f"habrías generado aproximadamente "
        f"**{simbolo_moneda}{float(safe_number(intereses_oportunidad)):,.0f}** en el mismo período."
    )

ajustes_txt = f" ({' + '.join(ajustes)})" if ajustes else ""
roi_ajustado = roi_ajustado_calc

colf1, colf2, colf3 = st.columns(3)
with colf1:
    st.markdown(f"**ROI {escenario_kpi} (Nominal)**")
    st.markdown(format_metric(f"ROI {escenario_kpi}", roi_nominal), unsafe_allow_html=True)

with colf2:
    st.markdown(f"**ROI Ajustado ({escenario_kpi}{ajustes_txt})**")
    st.markdown(format_metric("ROI Ajustado", roi_ajustado), unsafe_allow_html=True)

with colf3:
    if usar_financiamiento and pct_financiamiento > 0:
        colf3.metric("Monto Financiado", f"{simbolo_moneda}{capital_banco:,.0f}")
        colf3.metric("Cuota Mensual", f"{simbolo_moneda}{cuota_banco:,.0f}")
    else:
        colf3.metric("Monto Financiado", "—")
        colf3.metric("Cuota Mensual", "—")

# Gráfico comparativo de ROI por escenario
st.markdown("### 📊 Comparativo de ROI por Escenario")
col_comp1, col_comp2 = st.columns([2,1])
with col_comp1:
    roi_labels = ["Pesimista", "Normal", "Optimista"]
    roi_values = [roi_pes, roi_nor, roi_opt]
    hay_ajuste = aplicar_inflacion_en_roi or (usar_financiamiento and pct_financiamiento > 0)

    if hay_ajuste:
        roi_labels.append(f"{escenario_kpi} (Ajustado)")
        roi_values.append(roi_ajustado)

    colors_bar = ["#2ecc71" if v >= 0 else "#e74c3c" for v in roi_values]

    fig_roi, ax_roi = plt.subplots(figsize=(4,2.6), dpi=300)
    bars = ax_roi.barh(roi_labels, roi_values, color=colors_bar, edgecolor="black")
    for bar, val in zip(bars, roi_values):
        ax_roi.text(bar.get_width()/2, bar.get_y() + bar.get_height()/2,
                    f"{float(safe_number(val)):.2f}%", va="center", ha="center",
                    fontsize=8, fontweight="bold", color="white")
    ax_roi.axvline(0, color="black", linewidth=1)
    ax_roi.set_xlabel("ROI (%)", fontsize=9, fontweight="bold")
    ax_roi.grid(axis="x", linestyle="--", alpha=0.4)
    for spine in ["top", "right"]:
        ax_roi.spines[spine].set_visible(False)
    st.pyplot(fig_roi)

# Función global para ROI ajustado por escenario (reutilizable)
def calcular_roi_ajustado_local(pagos, egresos, horizonte_, fecha_inicio_, tasa_inflacion_, aplicar_inflacion_, interes_total_):
    df_base_loc = generar_flujo_caja_desde_schedule(
        pagos, egresos, int(safe_number(horizonte_)), fecha_inicio_, float(safe_number(tasa_inflacion_)),
        aplicar_inflacion_en_roi=bool(aplicar_inflacion_)
    )
    roi_calc, eq_calc = roi_equity_desde_flujo(df_base_loc)
    if interes_total_ and float(safe_number(interes_total_)) > 0:
        ganancia_ajustada = float(safe_number(df_base_loc["Flujo Neto"].sum() - interes_total_))
        roi_calc = (ganancia_ajustada / float(safe_number(eq_calc)) * 100.0) if float(safe_number(eq_calc)) > 0 else 0.0
    return roi_calc, float(safe_number(eq_calc))

with col_comp2:
    st.markdown("### 📌 ROI y Capital Propio (Ajustado)")
    roi_pes_adj, eq_pes_adj = calcular_roi_ajustado_local(pagos_pes, egresos_base, horizonte, fecha_inicio, tasa_inflacion_anual, aplicar_inflacion_en_roi, interes_total if usar_financiamiento else 0.0)
    st.markdown(f"**Pesimista (Ajustado):** {roi_pes_adj:.2f}% | {simbolo_moneda}{eq_pes_adj:,.0f}")

    roi_nor_adj, eq_nor_adj = calcular_roi_ajustado_local(pagos_nor, egresos_con_intereses, horizonte, fecha_inicio, tasa_inflacion_anual, aplicar_inflacion_en_roi, interes_total if usar_financiamiento else 0.0)
    st.markdown(f"**Normal (Ajustado):** {roi_nor_adj:.2f}% | {simbolo_moneda}{eq_nor_adj:,.0f}")

    roi_opt_adj, eq_opt_adj = calcular_roi_ajustado_local(pagos_opt, egresos_con_intereses, horizonte, fecha_inicio, tasa_inflacion_anual, aplicar_inflacion_en_roi, interes_total if usar_financiamiento else 0.0)
    st.markdown(f"**Optimista (Ajustado):** {roi_opt_adj:.2f}% | {simbolo_moneda}{eq_opt_adj:,.0f}")

# calcula payback buscando el PRIMER cruce de <0 a ≥0
payback_nor = None
fa = df_f_nor_num["Flujo Acumulado"].astype(float).values
meses = df_f_nor_num["Mes"].astype(int).values
for i in range(1, len(fa)):
    if fa[i-1] < 0 <= fa[i]:
        # interpolación lineal para un mes "decimal" (opcional)
        if fa[i] != fa[i-1]:
            frac = (-fa[i-1]) / (fa[i] - fa[i-1])
            payback_nor = meses[i-1] + frac
        else:
            payback_nor = meses[i]
        break

if payback_nor is not None:
    st.markdown(f"**⏱️ Payback (Normal):** {payback_nor:.1f} meses")
else:
    st.markdown("**⏱️ Payback (Normal):** No se recupera dentro del horizonte")

if usar_costo_oportunidad and (float(safe_number(tasa_costo_oportunidad)) > 0):
    st.markdown(f"**📉 Costo de oportunidad (referencia):** {float(safe_number(tasa_costo_oportunidad)):.2f}% anual")

if usar_financiamiento and float(safe_number(interes_total)) > 0:
    st.markdown(f"**💸 Intereses totales a pagar (simple):** {simbolo_moneda}{float(safe_number(interes_total)):,.0f}")

# VAN y TIR resumen
st.markdown("### 💵 VAN y TIR por Escenario")
cv1, cv2, cv3 = st.columns(3)
cv1.metric("VAN Pesimista", f"{simbolo_moneda}{van_pes:,.0f}")
cv2.metric("VAN Normal", f"{simbolo_moneda}{van_nor:,.0f}")
cv3.metric("VAN Optimista", f"{simbolo_moneda}{van_opt:,.0f}")
ct1, ct2, ct3 = st.columns(3)
ct1.metric("TIR Pesimista (anual)", f"{float(safe_number(tir_pes)):.2f}%")
ct2.metric("TIR Normal (anual)", f"{float(safe_number(tir_nor)):.2f}%")
ct3.metric("TIR Optimista (anual)", f"{float(safe_number(tir_opt)):.2f}%")

if usar_costo_oportunidad and float(safe_number(intereses_oportunidad)) > 0:
    st.info(
        f"Comparativo: al {float(safe_number(tasa_costo_oportunidad)):.2f}% anual, "
        f"los egresos habrían generado ≈ {simbolo_moneda}{float(safe_number(intereses_oportunidad)):,.0f}."
    )

# ===============================================================
#             TABLA DE ESCENARIOS NOMINALES INFORMATIVA
# ===============================================================

p_ini_base = float(safe_number(ingreso_total * pct_pago_inicial / 100.0))
finiq_base = float(safe_number((ingreso_total if base_finiquito == "Venta" else costo_total) * pct_finiquito / 100.0))
df_esc = pd.DataFrame({
    "Escenario": ["Pesimista","Normal","Optimista"],
    "Ingreso por Venta": [ingreso_pesimista, ingreso_normal, ingreso_optimista],
    f"Pago Inicial ({pct_pago_inicial}% Venta)": [0.0, p_ini_base, p_ini_base],
    "Remanente a Distribuir": [
        0.0,
        max(0.0, float(safe_number(ingreso_total - p_ini_base - finiq_base))),
        max(0.0, float(safe_number(ingreso_total - p_ini_base - finiq_base))),
    ],
    f"Finiquito ({pct_finiquito}% {base_finiquito.upper()})": [0.0, finiq_base, finiq_base]
})
st.table(df_esc.style.format({
    "Ingreso por Venta": simbolo_moneda+"{:,.0f}",
    f"Pago Inicial ({pct_pago_inicial}% Venta)": simbolo_moneda+"{:,.0f}",
    "Remanente a Distribuir": simbolo_moneda+"{:,.0f}",
    f"Finiquito ({pct_finiquito}% {base_finiquito.upper()})": simbolo_moneda+"{:,.0f}"
}))

# ===============================================================
#      PLANES DE PAGO DETALLADOS Y FLUJOS CON TOTALES AL FINAL
# ===============================================================

if mostrar_plan_pagos_detallado:
    st.markdown("### 📑 Planes de Pagos Detallados por Escenario")
    st.markdown("<h4 style='text-align:center;'>Escenario Pesimista</h4>", unsafe_allow_html=True)
    st.dataframe(plan_det_pes, use_container_width=True, height=altura_dinamica(plan_det_pes))

    st.markdown("<h4 style='text-align:center;'>Escenario Normal</h4>", unsafe_allow_html=True)
    st.dataframe(plan_det_nor, use_container_width=True, height=altura_dinamica(plan_det_nor))

    st.markdown("<h4 style='text-align:center;'>Escenario Optimista</h4>", unsafe_allow_html=True)
    st.dataframe(plan_det_opt, use_container_width=True, height=altura_dinamica(plan_det_opt))

def tabla_con_totales(df_num, simbolo_):
    return agregar_totales(df_num, simbolo_)

if mostrar_flujo_caja_escenarios:
    st.subheader("💰 Flujo de Caja por Escenario")
    st.write("### Escenario Pesimista")
    st.dataframe(tabla_con_totales(df_f_pes_num, simbolo_moneda), use_container_width=True)

    st.write("### Escenario Normal")
    st.dataframe(tabla_con_totales(df_f_nor_num, simbolo_moneda), use_container_width=True)

    st.write("### Escenario Optimista")
    st.dataframe(tabla_con_totales(df_f_opt_num, simbolo_moneda), use_container_width=True)

# ===============================================================
#                          KPI PRINCIPAL
# ===============================================================

roi_placeholder.metric(
    f"ROI ({escenario_kpi}, sobre equity)",
    f"{float(safe_number(roi_nominal)):.2f}%",
    delta=f"Equity usado: {simbolo_moneda}{float(safe_number(equity_nominal)):,.0f}"
)

# =========================
# 📊 Comparativo Final – VAN por Escenario
# =========================
st.markdown("---")
st.markdown("<h3 style='font-size: 30px;'>📊 Comparativo Final de VAN</h3>", unsafe_allow_html=True)

# Recalcular VAN por escenario (idempotente)
van_pes = calcular_van(df_f_pes_num, float(safe_number(tasa_descuento)))
van_nor = calcular_van(df_f_nor_num, float(safe_number(tasa_descuento)))
van_opt = calcular_van(df_f_opt_num, float(safe_number(tasa_descuento)))

col_izq, col_der = st.columns([2,1])

with col_izq:
    # --- Tabla Resumen VAN ---
    df_van = pd.DataFrame({
        "Escenario": ["Pesimista", "Normal", "Optimista"],
        "VAN (Valor Actual Neto)": [van_pes, van_nor, van_opt]
    })
    st.table(df_van.style.format({"VAN (Valor Actual Neto)": simbolo_moneda+"{:,.0f}"}))

with col_der:
    st.markdown("### 📘 Leyenda – Interpretación del VAN")
    st.markdown("""
    - **VAN > 0** 👉 El proyecto **crea valor**:  
      los flujos descontados superan la inversión inicial.  
    - **VAN = 0** 👉 El proyecto **empata**:  
      recupera la inversión, sin generar ni destruir valor.  
    - **VAN < 0** 👉 El proyecto **destruye valor**:  
      no alcanza la rentabilidad mínima exigida (tasa de descuento).  
    """)

# --- Gráfico de barras ---
st.markdown("### 📈 Gráfico Comparativo de VAN")

van_labels = ["Pesimista", "Normal", "Optimista"]
van_values = [van_pes, van_nor, van_opt]
colors = ["#e74c3c" if float(safe_number(v)) < 0 else "#2ecc71" for v in van_values]

fig_van, ax_van = plt.subplots(figsize=(6,3), dpi=300)
bars = ax_van.bar(van_labels, van_values, color=colors, edgecolor="black")

# Etiquetas arriba de las barras
for bar, val in zip(bars, van_values):
    ax_van.text(
        bar.get_x() + bar.get_width()/2, bar.get_height(),
        f"{simbolo_moneda}{float(safe_number(val)):,.0f}",
        ha="center", va="bottom", fontsize=9, fontweight="bold"
    )

ax_van.axhline(0, color="black", linewidth=1)
ax_van.set_ylabel(f"VAN ({simbolo_moneda})", fontsize=10, fontweight="bold")
ax_van.set_title("Comparativo de VAN por Escenario", fontsize=12, fontweight="bold")
ax_van.grid(axis="y", linestyle="--", alpha=0.5)
for spine in ["top", "right"]:
    ax_van.spines[spine].set_visible(False)

st.pyplot(fig_van)

# ====================================================================
#                  NOTAS ACLARATORIAS Y SUPUESTOS
# ====================================================================

with st.expander("📌 Notas de cálculo y supuestos"):
    st.markdown("""
- **ROI sobre equity**: usa el **pico de caja negativa acumulada** como base.
- **Payback**: primer mes donde el **Flujo Acumulado ≥ 0**.
- **VAN**: se descuenta con tasa mensual compuesta.
- **TIR**: se anualiza a partir de la tasa interna mensual.
- **Inflación en ROI**: infla egresos mes a mes si se activa.
- **Financiamiento simple**: resta intereses totales a la ganancia neta.
- **Valor Residual**: se incorpora como un ingreso final en el flujo, representando la recuperación del terreno al cierre del proyecto.
- **Sensibilidad Rápida**: permite variar ±% el precio de venta y el costo de construcción para analizar de inmediato el impacto en ingresos, costos y ganancia, sin rehacer todo el modelo.
- **Fila TOTAL**: el acumulado se fuerza a 0 en tablas, por presentación.
- **Sanitización**: se usan conversiones "seguras" para evitar NaN/None en cálculos.
    """)
