# app.py
import streamlit as st
import io
import re
import time
import requests
import base64
from datetime import datetime
import pdfplumber
from pdf2image import convert_from_bytes
try:
    import pytesseract
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False
from PIL import Image
import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER

# -------------------- CONFIGURACIÓN --------------------
# Tu API key (la solicitaste incrustada)
GEMINI_API_KEY = "AIzaSyBBxiisLsoPKLvKdWpjcE7cTtyXsRWQN7s"
# Modelo que usaremos vía REST
GEMINI_REST_MODEL = "gemini-1.5-flash"
# Cuántas páginas procesar por PDF (puedes aumentar si tu app permite más tokens)
PAGES_TO_PROCESS = 5

# -------------------- UI / STYLES --------------------
st.set_page_config(page_title="Auto-Grader (Gemini) - n8n style", layout="wide")
st.markdown("""
<style>
    .main-header { background: linear-gradient(90deg,#667eea 0%,#764ba2 100%); padding: 18px; border-radius:10px; color:white; text-align:center;}
    .stButton>button { background: linear-gradient(90deg,#667eea 0%,#764ba2 100%); color:white; font-weight:bold; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header"><h2>📝 Sistema de Calificación Automática (Google Gemini)</h2><div style="font-size:14px">Analiza PDFs con respuestas marcadas con "X" — Resultados, estadísticas y reporte PDF</div></div>', unsafe_allow_html=True)
st.write(" ")

# -------------------- HELPERS --------------------
def parse_answer_key(raw: str) -> dict:
    """Parsea entrada '1:a, 2:d, 3:e, 4:v, 5:f' -> {1:'a', 2:'d', ...}"""
    ans = {}
    if not raw:
        return ans
    candidates = re.findall(r'(\d{1,4})\s*[:\-]?\s*([a-eA-EvVfF])', raw)
    for n, c in candidates:
        ans[int(n)] = c.lower()
    return ans

def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """Extrae texto con pdfplumber; si muy corto y hay OCR disponible, hace OCR sobre primeras páginas."""
    text = ""
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for i, page in enumerate(pdf.pages):
                if i >= PAGES_TO_PROCESS: break
                ptext = page.extract_text()
                if ptext:
                    text += ptext + "\n"
    except Exception as e:
        st.debug(f"pdfplumber error: {e}")

    # Si no hay texto suficiente y OCR disponible, usar OCR
    if (not text or len(text.strip()) < 60) and OCR_AVAILABLE:
        try:
            images = convert_from_bytes(pdf_bytes, dpi=200, first_page=1, last_page=PAGES_TO_PROCESS)
            ocr_text = ""
            for img in images:
                ocr_text += pytesseract.image_to_string(img, lang='spa') + "\n"
            if len(ocr_text.strip()) > len(text.strip()):
                text = ocr_text
        except Exception as e:
            st.debug(f"OCR error: {e}")
    return text

def call_gemini_rest(prompt: str) -> str:
    """
    Llama a la API REST de Google Gemini correctamente (v1beta formato contents).
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_REST_MODEL}:generateContent"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}]
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 512
        }
    }
    try:
        resp = requests.post(f"{url}?key={GEMINI_API_KEY}", headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        # Extraer texto según estructura moderna
        text = (
            data.get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "")
        )
        return text.strip()
    except Exception as e:
        st.error(f"Error llamando a Gemini: {e}")
        return ""


def parse_gemini_json_response(gemini_text: str) -> dict:
    """Intentar extraer un JSON {\"1\":\"a\",...} o pares '1:a' de la respuesta generada."""
    if not gemini_text:
        return {}
    # Buscar primero JSON
    m = re.search(r'\{[^{}]*\}', gemini_text)
    if m:
        try:
            import json
            data = json.loads(m.group())
            # convert keys to int, values to lowercase
            parsed = {}
            for k, v in data.items():
                try:
                    parsed[int(k)] = str(v).lower().strip()
                except:
                    continue
            return parsed
        except Exception:
            pass
    # Si no JSON, buscar pares '1:a' o '1: a'
    pairs = re.findall(r'(\d{1,4})\s*[:\-]\s*([a-eA-EvVfF])', gemini_text)
    parsed = {}
    for n, c in pairs:
        parsed[int(n)] = c.lower()
    return parsed

def grade_answers(student: dict, key: dict, total_q: int) -> (float,int,int):
    """Devuelve (nota_0_20, correctas, incorrectas)"""
    if total_q <= 0:
        return 0.0, 0, 0
    correct = 0
    for q in range(1, total_q+1):
        if q in student and q in key and student[q] == key[q]:
            correct += 1
    nota = round((correct / total_q) * 20, 2)
    incorrect = total_q - correct
    return nota, correct, incorrect

def generar_reporte_pdf(resultados: list, curso_nombre: str, curso_codigo: str, clave: dict) -> io.BytesIO:
    """Genera PDF con reportlab y devuelve buffer."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=2*cm, bottomMargin=2*cm)
    elementos = []
    styles = getSampleStyleSheet()
    titulo_style = ParagraphStyle('T', parent=styles['Heading1'], fontSize=16, alignment=TA_CENTER, textColor=colors.HexColor('#667eea'))
    elementos.append(Paragraph("📝 SISTEMA DE CALIFICACIÓN AUTOMÁTICA", titulo_style))
    elementos.append(Paragraph("Reporte de Resultados", styles['Heading2']))
    elementos.append(Spacer(1, 0.3*cm))

    info_data = [
        ['Curso:', curso_nombre or '-'],
        ['Código:', curso_codigo or '-'],
        ['Fecha:', datetime.now().strftime('%d/%m/%Y %H:%M:%S')],
        ['Total preguntas:', str(len(clave))]
    ]
    info_table = Table(info_data, colWidths=[4*cm, 12*cm])
    info_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#667eea')),
        ('TEXTCOLOR', (0,0), (0,-1), colors.whitesmoke),
        ('FONTNAME',(0,0),(-1,-1),'Helvetica-Bold'),
        ('GRID',(0,0),(-1,-1),0.5,colors.black)
    ]))
    elementos.append(info_table)
    elementos.append(Spacer(1, 0.4*cm))

    # Estadísticas
    df = pd.DataFrame(resultados)
    notas = df['nota'].values if not df.empty else [0]
    aprobados = len(df[df['nota'] >= 14]) if not df.empty else 0
    desaprobados = len(df[df['nota'] < 14]) if not df.empty else 0

    stats_data = [
        ['Métrica', 'Valor'],
        ['Promedio General', f"{notas.mean():.2f}"],
        ['Promedio Aprobados', f"{df[df['nota']>=14]['nota'].mean():.2f}" if aprobados>0 else "N/A"],
        ['Nota Máx', f"{notas.max():.2f}"],
        ['Nota Mín', f"{notas.min():.2f}"],
        ['Aprobados', f"{aprobados}"],
        ['Desaprobados', f"{desaprobados}"]
    ]
    stats_table = Table(stats_data, colWidths=[8*cm, 8*cm])
    stats_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#667eea')),
        ('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),
        ('ALIGN',(0,0),(-1,-1),'CENTER'),
        ('GRID',(0,0),(-1,-1),0.5,colors.black)
    ]))
    elementos.append(stats_table)
    elementos.append(Spacer(1, 0.4*cm))

    # Detalle
    elementos.append(Paragraph("Detalle de calificaciones", styles['Heading2']))
    notas_data = [['#','Archivo','Correctas','Incorrectas','Nota','Estado']]
    for i, r in enumerate(resultados, start=1):
        estado = 'Aprobado' if r['nota'] >= 14 else 'Desaprobado'
        notas_data.append([str(i), r['nombre_pdf'], str(r['correctas']), str(r['incorrectas']), f"{r['nota']:.2f}", estado])
    notas_table = Table(notas_data, colWidths=[1.2*cm, 7*cm, 2.2*cm, 2.2*cm, 2*cm, 3.4*cm])
    notas_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#667eea')),
        ('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),
        ('GRID',(0,0),(-1,-1),0.5,colors.black)
    ]))
    elementos.append(notas_table)

    doc.build(elementos)
    buffer.seek(0)
    return buffer

# -------------------- INTERFAZ --------------------
with st.sidebar:
    st.header("Configuración")
    st.write("Clave Gemini: **incrustada** (usa st.secrets si prefieres cambiar).")
    st.write(f"Modelo REST usado: {GEMINI_REST_MODEL}")
    st.markdown("---")
    st.info("Notas: Para OCR en PDFs escaneados habilita poppler + tesseract (Dockerfile incluido).")

# Paso 1 - datos del curso y clave
st.header("1) Datos del curso y clave de respuestas")
col1, col2 = st.columns(2)
with col1:
    curso_nombre = st.text_input("Nombre del curso", "")
with col2:
    curso_codigo = st.text_input("Código del curso", "")

st.text("Formato de clave: `1:a, 2:d, 3:e, 4:v, 5:f` (use v/f para verdadero/falso)")
clave_raw = st.text_area("Hoja de respuestas (formato)", height=90, placeholder="1:a, 2:b, 3:c, 4:v, 5:f")
clave = parse_answer_key(clave_raw) if clave_raw else {}

total_questions = st.number_input("Total de preguntas (si no lo indicas se toma len(clave))", min_value=1, value=len(clave) if clave else 5)

st.markdown("---")
# Paso 2 - carga PDFs
st.header("2) Subir PDFs de exámenes (máx 30)")
uploaded = st.file_uploader("Sube hasta 30 PDFs (cada archivo = 1 estudiante)", type=["pdf"], accept_multiple_files=True)
if uploaded and len(uploaded) > 30:
    st.error("Máximo 30 archivos. Reduce la cantidad.")
    uploaded = uploaded[:30]

if uploaded:
    with st.expander("Ver archivos subidos"):
        for i, f in enumerate(uploaded, start=1):
            st.write(f"{i}. {f.name} — {f.size/1024:.1f} KB")

st.markdown("---")
# Paso 3 - botón analizar (simula n8n)
st.header("3) Análisis automatizado")
# Sólo habilitamos si todo necesario está presente
puede = bool(uploaded and clave and curso_nombre and curso_codigo)

if not puede:
    faltan = []
    if not curso_nombre: faltan.append("Nombre curso")
    if not curso_codigo: faltan.append("Código curso")
    if not clave: faltan.append("Clave respuestas")
    if not uploaded: faltan.append("PDFs")
    st.warning("Completa: " + ", ".join(faltan))

boton = st.button("🔎 Analizar en n8n", disabled=not puede)

if boton:
    st.info("Iniciando flujo (simulado) — procesando con Google Gemini...")
    progreso = st.progress(0)
    resultados = []
    total = len(uploaded)
    for idx, pdf_file in enumerate(uploaded):
        progreso.progress(int((idx/total)*100))
        st.write(f"Procesando: **{pdf_file.name}** — ({idx+1}/{total})")
        # leer bytes
        pdf_bytes = pdf_file.read()
        # extraer texto (pdfplumber o OCR)
        text = extract_text_from_pdf_bytes(pdf_bytes)
        if not text:
            st.warning("No se pudo extraer texto con pdfplumber; si es PDF escaneado asegúrate de usar Docker con OCR.")
            text = ""
        # preparar prompt para Gemini: pedimos JSON puro con pares pregunta:opcion
        short_text = text if len(text) < 20000 else text[:20000]
        prompt = f"""
Analiza el siguiente contenido (texto extraído del examen). Busca preguntas numeradas y las respuestas marcadas (marcas: X, círculo o resaltado).
Devuelve SOLO un JSON válido con pares "numero":"opcion" por ejemplo:
{{"1":"a","2":"d","3":"v"}}
Acepta opciones: a,b,c,d,e para multiple choice y v/f para verdadero/falso.
Si no encuentra respuestas devuelve {{}}.

TEXTO:
{short_text}
"""
        gemini_out = call_gemini_rest(prompt)
        student_answers = parse_gemini_json_response(gemini_out)
        # Si Gemini no encontró nada, tratar de heurística local rápida (buscar '1 a x' etc)
        if not student_answers:
            # heurística simple: buscar '1 a X' o '1. a X' o 'a) X' with nearby numbers
            lines = text.splitlines()
            local_answers = {}
            for line in lines:
                m = re.search(r'(\d{1,4})\D{0,4}([a-eA-EvVfF])\D{0,6}[xX]', line)
                if m:
                    local_answers[int(m.group(1))] = m.group(2).lower()
            student_answers = local_answers
        # Calificar
        total_q = total_questions if total_questions else len(clave)
        nota, correctas, incorrectas = grade_answers(student_answers, clave, total_q)
        resultados.append({
            "nombre_pdf": pdf_file.name,
            "nota": nota,
            "correctas": correctas,
            "incorrectas": incorrectas,
            "respuestas": student_answers
        })
        time.sleep(0.6)  # efecto visual
        progreso.progress(int(((idx+1)/total)*100))

    st.success("Procesamiento finalizado ✅")
    st.session_state.resultados = resultados
    st.session_state.clave = clave
    st.session_state.curso_nombre = curso_nombre
    st.session_state.curso_codigo = curso_codigo

# Paso 4 - mostrar resultados + estadísticas
if st.session_state.get("resultados"):
    st.markdown("---")
    st.header("4) Resultados y estadísticas")
    df = pd.DataFrame(st.session_state.resultados)
    if df.empty:
        st.info("No hay resultados.")
    else:
        st.dataframe(df[['nombre_pdf','correctas','incorrectas','nota']].sort_values('nota', ascending=False).reset_index(drop=True))
        avg = df['nota'].mean()
        approved = df[df['nota']>=14]
        disapproved = df[df['nota']<14]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Promedio general", f"{avg:.2f}")
        col2.metric("Aprobados", f"{len(approved)}")
        col3.metric("Desaprobados", f"{len(disapproved)}")
        col4.metric("Total estudiantes", f"{len(df)}")
        st.write(f"Mayor nota: {df['nota'].max():.2f} — Menor nota: {df['nota'].min():.2f}")

    # Paso 5 exportar reporte
    st.markdown("---")
    st.header("5) Exportar reporte")
    if st.button("📄 Generar reporte PDF"):
        with st.spinner("Generando reporte PDF..."):
            pdf_buf = generar_reporte_pdf(st.session_state.resultados, st.session_state.curso_nombre, st.session_state.curso_codigo, st.session_state.clave)
            now = datetime.now().strftime("%Y%m%d_%H%M%S")
            nombre = f"reporte_{st.session_state.curso_codigo or 'curso'}_{now}.pdf"
            st.download_button("⬇️ Descargar reporte", data=pdf_buf, file_name=nombre, mime="application/pdf")

