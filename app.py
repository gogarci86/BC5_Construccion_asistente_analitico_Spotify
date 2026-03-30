# ============================================================
# CABECERA
# ============================================================
# Alumno: Nombre Apellido
# URL Streamlit Cloud: https://...streamlit.app
# URL GitHub: https://github.com/...

# ============================================================
# IMPORTS
# ============================================================
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from openai import OpenAI
import json

# ============================================================
# CONSTANTES
# ============================================================
MODEL = "gpt-4.1-mini"

# -------------------------------------------------------
# >>> SYSTEM PROMPT — TU TRABAJO PRINCIPAL ESTÁ AQUÍ <<<
# -------------------------------------------------------
SYSTEM_PROMPT = """
Eres un analista de datos especializado en hábitos de escucha de Spotify.
Tu misión es responder preguntas sobre el historial de escucha de un usuario
generando código Python con Plotly para producir visualizaciones.

═══════════════════════════════════════════════
CONTEXTO DEL DATASET
═══════════════════════════════════════════════
El dataset contiene {num_rows} reproducciones de música (los podcasts ya fueron filtrados).
Rango temporal: del {fecha_min} al {fecha_max} ({num_months} meses completos).
Meses disponibles: {meses_disponibles}.

Resumen del usuario:
- Total de horas escuchadas: {total_hours} h
- Artistas únicos: {num_artists}
- Canciones únicas: {num_tracks}
- Álbumes únicos: {num_albums}
- Tasa de skip: {skip_rate}% de las reproducciones fueron saltadas
- Uso de shuffle: {shuffle_rate}% de las reproducciones con modo aleatorio

Top 5 artistas por reproducciones: {top5_artists}
Distribución por plataforma: {platform_dist}

═══════════════════════════════════════════════
ESTRUCTURA DEL DATAFRAME `df`
═══════════════════════════════════════════════
Columnas disponibles:
| Columna           | Tipo     | Descripción                                                |
|-------------------|----------|------------------------------------------------------------|
| ts                | datetime | Timestamp de fin de reproducción (UTC)                     |
| track             | str      | Nombre de la canción ({num_tracks} valores únicos)         |
| artist            | str      | Artista principal ({num_artists} valores únicos)           |
| album             | str      | Nombre del álbum ({num_albums} valores únicos)             |
| spotify_track_uri | str      | Identificador único de la canción                          |
| ms_played         | int      | Milisegundos de reproducción efectiva                      |
| min_played        | float    | Minutos de reproducción (= ms_played / 60000)              |
| hours_played      | float    | Horas de reproducción (= ms_played / 3600000)              |
| hour              | int      | Hora del día (0-23)                                        |
| day_of_week       | str      | Día en inglés: Monday, Tuesday, ..., Sunday                |
| day_num           | int      | Día numérico (0=Monday ... 6=Sunday), útil para ordenar    |
| month             | str      | Formato "YYYY-MM" (ej: "2025-03"). Útil para eje X temporal|
| month_name        | str      | Nombre en inglés: January, February, ...                   |
| month_num         | int      | Número del mes (1-12). Útil para filtrar por período       |
| is_weekend        | bool     | True = sábado o domingo                                    |
| shuffle           | bool     | True = modo aleatorio activado                             |
| skipped           | bool     | True = canción saltada, False = no saltada                 |
| reason_start      | str      | Motivo de inicio. Valores: {reason_start_values}           |
| reason_end        | str      | Motivo de fin. Valores: {reason_end_values}                |
| platform          | str      | Plataforma. Valores: {plataformas}                         |
| semester          | str      | "1er semestre" (ene-jun) o "2do semestre" (jul-dic)        |
| season            | str      | "Invierno", "Primavera", "Verano", "Otoño"                |

═══════════════════════════════════════════════
FORMATO DE RESPUESTA
═══════════════════════════════════════════════
Responde SIEMPRE con un JSON válido y NADA más. Sin texto antes ni después.
Dos tipos posibles:

1. Pregunta sobre el historial de escucha:
{{"tipo": "grafico", "codigo": "<código Python>", "interpretacion": "<texto breve>"}}

2. Pregunta fuera de alcance (no relacionada con el historial de escucha):
{{"tipo": "fuera_de_alcance", "codigo": "", "interpretacion": "<respuesta amable explicando que solo puedes analizar datos de escucha>"}}

═══════════════════════════════════════════════
REGLAS PARA EL CÓDIGO
═══════════════════════════════════════════════
- El código DEBE crear una variable `fig` usando plotly.express (px) o plotly.graph_objects (go).
- Tienes disponibles: df, pd, px, go. NO importes ninguna otra librería.
- NO modifiques el DataFrame `df`. Usa variables intermedias para agrupaciones y filtros.
- Los títulos de los gráficos deben ser descriptivos y en español.
- Las etiquetas de los ejes deben ser legibles y en español.
- Si un gráfico de barras tiene muchas categorías (>8), usa barras horizontales.
- Usa fig.update_layout() para mejorar la legibilidad (tamaño de fuente, márgenes, etc.).

═══════════════════════════════════════════════
CRITERIOS DE ANÁLISIS POR TIPO DE PREGUNTA
═══════════════════════════════════════════════

A. RANKINGS Y FAVORITOS ("top", "más escuchado", "favoritos"):
   - Barras horizontales, ordenadas de mayor a menor.
   - Por defecto top 10 si el usuario no especifica N.
   - Ten en cuenta que hay {num_artists} artistas y {num_tracks} canciones en el dataset.

B. EVOLUCIÓN TEMPORAL ("por mes", "evolución", "tendencia"):
   - Gráfico de líneas con `month` en eje X. Los {num_months} meses están en formato YYYY-MM y ya se ordenan alfabéticamente.
   - Para "canciones nuevas" o "descubrimientos" por mes: cuenta valores únicos de `spotify_track_uri` por mes.

C. PATRONES DE USO ("a qué hora", "qué día", "entre semana vs fin de semana"):
   - Barras verticales. Ordena días de la semana usando `day_num` (0=Monday a 6=Sunday).
   - Para horas: eje X de 0 a 23, agrupa por `hour`.
   - Para plataforma por período: recuerda que las plataformas son {plataformas}.

D. COMPORTAMIENTO ("skips", "shuffle", "saltadas"):
   - Pie chart o barras con porcentajes.
   - Tasa de skip global: {skip_rate}%. Tasa de shuffle global: {shuffle_rate}%.
   - Para skips por artista/canción: agrupa y calcula la media de `skipped`.

E. COMPARACIÓN ENTRE PERÍODOS ("verano vs invierno", "primer vs segundo semestre"):
   - Barras agrupadas (barmode="group") con color por período.
   - Usa la columna `semester` para semestres y `season` para estaciones.

Criterios de medida:
- "más escuchado" sin especificar = número de reproducciones (.value_counts() o .groupby().size()).
- Si menciona "horas" o "tiempo" = usa `hours_played` con .sum().
- Si menciona "minutos" = usa `min_played` con .sum().
- "escucha real" = filtrar ms_played >= 30000 (30 segundos).
- "canciones únicas/distintas" = .nunique() sobre `spotify_track_uri`.
- El usuario tiene {total_hours} horas totales repartidas en {num_rows} reproducciones.

Períodos temporales:
- Verano: junio, julio, agosto (month_num en [6, 7, 8]) → season == "Verano"
- Invierno: diciembre, enero, febrero (month_num en [12, 1, 2]) → season == "Invierno"
- Primavera: marzo, abril, mayo → season == "Primavera"
- Otoño: septiembre, octubre, noviembre → season == "Otoño"
- Primer semestre: enero-junio → semester == "1er semestre"
- Segundo semestre: julio-diciembre → semester == "2do semestre"
- Entre semana: is_weekend == False. Fin de semana: is_weekend == True.

═══════════════════════════════════════════════
INTERPRETACIÓN
═══════════════════════════════════════════════
- Escribe 2-3 frases en español que destaquen el hallazgo principal.
- Sé específico: menciona nombres de artistas, cifras concretas y tendencias.
- Puedes usar los datos del contexto (top 5 artistas, total horas, etc.) para enriquecer la interpretación.
- No repitas la pregunta del usuario. Ve directo al insight.

═══════════════════════════════════════════════
GUARDRAILS
═══════════════════════════════════════════════
- Si la pregunta no tiene relación con hábitos de escucha musical, devuelve tipo "fuera_de_alcance".
- Si la pregunta es ambigua, interpreta la intención más razonable y genera el gráfico.
- NUNCA generes código que use print(), input(), st., os, sys, open(), eval() o subprocess.
- NUNCA generes código que intente acceder a Internet o a ficheros externos.
- El código solo debe manipular `df` y crear `fig`. Nada más.
"""


# ============================================================
# CARGA Y PREPARACIÓN DE DATOS
# ============================================================
@st.cache_data
def load_data():
    df = pd.read_json("streaming_history.json")

    # ----------------------------------------------------------
    # >>> PREPARACIÓN DE DATOS <<<
    # ----------------------------------------------------------

    # 1. Convertir timestamp a datetime
    df["ts"] = pd.to_datetime(df["ts"])

    # 2. Renombrar columnas largas para simplificar el código del LLM
    df = df.rename(columns={
        "master_metadata_track_name": "track",
        "master_metadata_album_artist_name": "artist",
        "master_metadata_album_album_name": "album",
    })

    # 3. Filtrar podcasts (filas sin artista/track — son episodios de podcast)
    df = df[df["artist"].notna()].copy()

    # 4. Convertir ms_played a unidades más legibles
    df["min_played"] = df["ms_played"] / 60_000
    df["hours_played"] = df["ms_played"] / 3_600_000

    # 5. Columnas temporales derivadas
    df["hour"] = df["ts"].dt.hour
    df["day_of_week"] = df["ts"].dt.day_name()
    df["day_num"] = df["ts"].dt.dayofweek  # 0=Monday, 6=Sunday
    df["month"] = df["ts"].dt.strftime("%Y-%m")
    df["month_name"] = df["ts"].dt.month_name()
    df["month_num"] = df["ts"].dt.month
    df["is_weekend"] = df["ts"].dt.dayofweek >= 5

    # 6. Columnas de período para comparaciones
    df["semester"] = df["month_num"].apply(lambda m: "1er semestre" if m <= 6 else "2do semestre")
    df["season"] = df["month_num"].map({
        12: "Invierno", 1: "Invierno", 2: "Invierno",
        3: "Primavera", 4: "Primavera", 5: "Primavera",
        6: "Verano", 7: "Verano", 8: "Verano",
        9: "Otoño", 10: "Otoño", 11: "Otoño",
    })

    # 7. Limpiar skipped: NaN → False, 1.0 → True
    df["skipped"] = df["skipped"].fillna(False).astype(bool)

    return df


def build_prompt(df):
    """
    Inyecta información dinámica del dataset en el system prompt.
    Todos los valores se calculan a partir del DataFrame real,
    de modo que el prompt siempre refleja el estado actual de los datos.
    """
    # --- Rango temporal ---
    fecha_min = df["ts"].min().strftime("%Y-%m-%d")
    fecha_max = df["ts"].max().strftime("%Y-%m-%d")
    meses_disponibles = sorted(df["month"].unique().tolist())
    num_months = len(meses_disponibles)

    # --- Valores categóricos ---
    plataformas = df["platform"].unique().tolist()
    reason_start_values = df["reason_start"].unique().tolist()
    reason_end_values = df["reason_end"].unique().tolist()

    # --- Estadísticas generales ---
    num_rows = len(df)
    num_artists = df["artist"].nunique()
    num_tracks = df["track"].nunique()
    num_albums = df["album"].nunique()
    total_hours = round(df["hours_played"].sum(), 1)
    skip_rate = round(df["skipped"].mean() * 100, 1)
    shuffle_rate = round(df["shuffle"].mean() * 100, 1)

    # --- Top 5 artistas (nombre: reproducciones) ---
    top5 = df["artist"].value_counts().head(5)
    top5_artists = ", ".join([f"{a} ({c} reproducciones)" for a, c in top5.items()])

    # --- Distribución por plataforma ---
    plat_dist = df["platform"].value_counts()
    platform_dist = ", ".join([f"{p}: {c}" for p, c in plat_dist.items()])

    return SYSTEM_PROMPT.format(
        fecha_min=fecha_min,
        fecha_max=fecha_max,
        meses_disponibles=meses_disponibles,
        num_months=num_months,
        plataformas=plataformas,
        reason_start_values=reason_start_values,
        reason_end_values=reason_end_values,
        num_rows=num_rows,
        num_artists=num_artists,
        num_tracks=num_tracks,
        num_albums=num_albums,
        total_hours=total_hours,
        skip_rate=skip_rate,
        shuffle_rate=shuffle_rate,
        top5_artists=top5_artists,
        platform_dist=platform_dist,
    )


# ============================================================
# FUNCIÓN DE LLAMADA A LA API
# ============================================================
def get_response(user_msg, system_prompt):
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content


# ============================================================
# PARSING DE LA RESPUESTA
# ============================================================
def parse_response(raw):
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

    return json.loads(cleaned)


# ============================================================
# EJECUCIÓN DEL CÓDIGO GENERADO
# ============================================================
def execute_chart(code, df):
    local_vars = {"df": df, "pd": pd, "px": px, "go": go}
    exec(code, {}, local_vars)
    return local_vars.get("fig")


# ============================================================
# INTERFAZ STREAMLIT
# ============================================================
st.set_page_config(page_title="Spotify Analytics", layout="wide")

# --- Control de acceso ---
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("🔒 Acceso restringido")
    pwd = st.text_input("Contraseña:", type="password")
    if pwd:
        if pwd == st.secrets["PASSWORD"]:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Contraseña incorrecta.")
    st.stop()

# --- App principal ---
st.title("🎵 Spotify Analytics Assistant")
st.caption("Pregunta lo que quieras sobre tus hábitos de escucha")

df = load_data()
system_prompt = build_prompt(df)

if prompt := st.chat_input("Ej: ¿Cuál es mi artista más escuchado?"):

    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analizando..."):
            try:
                raw = get_response(prompt, system_prompt)
                parsed = parse_response(raw)

                if parsed["tipo"] == "fuera_de_alcance":
                    st.write(parsed["interpretacion"])
                else:
                    fig = execute_chart(parsed["codigo"], df)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                        st.write(parsed["interpretacion"])
                        st.code(parsed["codigo"], language="python")
                    else:
                        st.warning("El código no produjo ninguna visualización. Intenta reformular la pregunta.")
                        st.code(parsed["codigo"], language="python")

            except json.JSONDecodeError:
                st.error("No he podido interpretar la respuesta. Intenta reformular la pregunta.")
            except Exception as e:
                st.error("Ha ocurrido un error al generar la visualización. Intenta reformular la pregunta.")


# ============================================================
# REFLEXIÓN TÉCNICA (máximo 30 líneas)
# ============================================================
#
# 1. ARQUITECTURA TEXT-TO-CODE
#    Mi aplicación sigue una arquitectura text-to-code: el LLM (gpt-4.1-mini) recibe
#    el system prompt — que describe la estructura del DataFrame (columnas, tipos, valores
#    posibles) y las reglas de formato — junto con la pregunta del usuario. Devuelve un JSON
#    con código Python que se ejecuta localmente mediante exec() contra el DataFrame real.
#    El LLM nunca ve los 14.294 registros: solo sabe que existe una columna "artist" de tipo
#    string. Esto ahorra tokens (coste), evita superar el contexto y es más seguro, porque
#    los datos no salen del entorno local.
#
# 2. EL SYSTEM PROMPT COMO PIEZA CLAVE
#    Le proporciono: la lista exacta de columnas con tipos y descripciones, los valores
#    posibles de campos categóricos (plataformas, reason_start/end), el rango de fechas,
#    criterios de análisis (qué gráfico usar para cada tipo de pregunta), y definiciones
#    de períodos (verano = jun-ago, semestre, fin de semana). Ejemplo que funciona:
#    "Compara mis artistas de verano vs invierno" funciona porque el prompt define qué meses
#    son verano/invierno y la columna `season` ya existe. Ejemplo que fallaría: si quito la
#    regla de que "más escuchado" sin especificar usa conteo de reproducciones, la pregunta
#    "¿Cuál es mi artista más escuchado?" podría generar código que sume ms_played o que no
#    agrupe correctamente, produciendo resultados inconsistentes.
#
# 3. EL FLUJO COMPLETO
#    1) El usuario escribe una pregunta en st.chat_input. 2) Se envía al LLM junto con el
#    system prompt vía get_response (llamada a la API de OpenAI). 3) El LLM devuelve un
#    string con un JSON. 4) parse_response limpia posibles backticks markdown y lo convierte
#    en un diccionario Python con json.loads. 5) Si tipo=="grafico", execute_chart ejecuta el
#    código con exec(), que crea la variable fig (una figura Plotly). 6) Streamlit muestra el
#    gráfico con st.plotly_chart, la interpretación textual, y el código fuente. 7) Si
#    tipo=="fuera_de_alcance", solo se muestra la interpretación. 8) Si el JSON no se parsea
#    o el código falla, los bloques except capturan el error y muestran un mensaje amigable.