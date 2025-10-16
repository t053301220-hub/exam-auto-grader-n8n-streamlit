# streamlit_exam_grader_app.py
# Aplicación Streamlit para calificar PDFs de exámenes (MCQ y VF) marcados con X o círculo.
# - El usuario ingresa la clave en formato: 1:a, 2:d, 3:e, 4:v, 5:f
# - Sube hasta 30 PDFs. La app intentará extraer las respuestas y compararlas con la clave.\# - Simula un "análisis en n8n" con barra de progreso.
# - Permite exportar un reporte PDF con notas y estadísticas.

import re
import io
import os
import tempfile
from collections import defaultdict
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
from fpdf import FPDF
import matplotlib.pyplot as plt

# Librerías que se intentarán usar para extracción de texto/imágenes de PDFs
# Todas son opcionales: el app intentará múltiples estrategias y fallará elegantemente si no están disponibles.
try:
    import pdfplumber
except Exception:
    pdfplumber = None

try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

try:
    import pytesseract
    from PIL import Image
except Exception:
    pytesseract = None
    Image = None

# ----------------------------- UTILIDADES -----------------------------

def parse_key_string(key_str):
    """Parsea la cadena de claves del formato '1:a, 2:d, 3:e, 4:v, 5:f' a dict {1: 'a', ...}"""
    key_str = key_str.strip()
    if not key_str:
        return {}
    pairs = re.split(r"[,;]+", key_str)
    keys = {}
    for p in pairs:
        p = p.strip()
        if not p:
            continue
        m = re.match(r"^(\d+)\s*[:\-\)]\s*([a-evfvAEFV])$", p)
        if not m:
            # intentar formatos alternativos como '1 a' o '1:a'
            m = re.match(r"^(\d+)\s*[.:]?\s*([a-e])$", p, re.IGNORECASE)
        if m:
            q = int(m.group(1))
            ans = m.group(2).lower()
            # normalizar v/f
            if ans in ("v", "f"):
                ans = 'v' if ans == 'v' else 'f'
            keys[q] = ans
        else:
            # intentar separado por espacio
            parts = p.split()
            if len(parts) == 2 and parts[0].isdigit():
                keys[int(parts[0])] = parts[1].lower()[0]
            else:
                # ignorar si no pudo parsear
                continue
    return keys


def extract_text_with_pdfplumber(pdf_bytes):
    if not pdfplumber:
        return ""
    out = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            try:
                out.append(page.extract_text() or "")
            except Exception:
                out.append("")
    return "\n".join(out)


def render_pdf_pages_to_images(pdf_bytes):
    """Usa PyMuPDF para renderizar páginas a PIL Images. Retorna lista de PIL.Image o [] si no disponible."""
    if not fitz:
        return []
    imgs = []
    try:
        doc = fitz.open(stream=pdf_bytes, filetype='pdf')
        for page in doc:
            mat = fitz.Matrix(2, 2)  # render a mayor resolución
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img_bytes = pix.tobytes()
            try:
                from PIL import Image as PILImage
                img = PILImage.open(io.BytesIO(img_bytes))
                imgs.append(img.convert('RGB'))
            except Exception:
                continue
    except Exception:
        return []
    return imgs


def ocr_image_to_text(img):
    if not pytesseract:
        return ""
    try:
        return pytesseract.image_to_string(img, lang='eng')
    except Exception:
        return ""


def find_answers_in_text(text):
    """Busca patrones de respuesta dentro del texto extraido.
    Retorna dict {preg: alternatica} detectadas.
    Soporta formatos comunes como '1. a) X', '1) X a', '1. X a)'.
    """
    answers = {}
    if not text:
        return answers

    # Normalizar guiones y paréntesis
    text = text.replace('\r', '\n')

    # Regexes para capturar líneas como "1. a) X b)" o "1 a) X" o "1. X a)"
    # Buscaremos tres cosas: número de pregunta, la alternativa marcada (a-e o v/f)

    # 1) Buscar patrones tipo '1. a) b) c) d) e) -- con X o similar cerca'
    # Buscaremos por cada pregunta número las opciones con alguna marca
    # Build a simple token list
    tokens = re.split(r"[\n\t]+", text)

    # Pattern to find explicit '1 a) X' type
    for line in tokens:
        # buscar número de pregunta al inicio
        m = re.match(r"^\s*(\d{1,3})\b(.*)$", line)
        if not m:
            continue
        qnum = int(m.group(1))
        rest = m.group(2)
        # buscar 'X' o 'x' o '○' o 'o' cerca de opción
        # ejemplos: 'a) X', 'X a)', 'a) (X)'
        # construir patrones para cada alternativa a-e y v,f
        for opt in ['a','b','c','d','e','v','f']:
            # varias variantes
            pat1 = rf"{opt}\)\s*[^A-Za-z0-9\S]*[Xx○o●]"  # 'a) X'
            pat2 = rf"[Xx○o●]\s*{opt}\)"  # 'X a)'
            pat3 = rf"{opt}\)\s*\([^)]*[Xx○o●]"  # 'a) (X)'
            if re.search(pat1, rest) or re.search(pat2, rest) or re.search(pat3, rest):
                answers[qnum] = opt
                break
        # si no se detectó, intentar buscar '1. X a)'
        if qnum not in answers:
            m2 = re.search(r"[Xx○o●]" , rest)
            if m2:
                # intentar encontrar letra más cercana a la X
                # localizar posiciones
                pos = m2.start()
                # buscar letras a-e o v/f en un radio cercano
                window = rest[max(0,pos-12):pos+12]
                mm = re.search(r"([a-evfv])\)" , window, re.IGNORECASE)
                if mm:
                    answers[qnum] = mm.group(1).lower()
    # Como alternativa, buscar en todo texto patrones '1:a' '1: a X'
    # patrones tipo '1: a X' o '1-a X'
    extra = re.findall(r"(\d{1,3})\s*[:\-\)]\s*([a-evfv])\b.*?[Xx○o●]", text, re.IGNORECASE)
    for q, opt in extra:
        answers[int(q)] = opt.lower()

    return answers


def grade_single_pdf(pdf_bytes, answer_key):
    """Devuelve dict con las respuestas detectadas y la nota en 0-20 (no penaliza)."""
    # 1) Intentar extraer texto
    text = extract_text_with_pdfplumber(pdf_bytes) if pdfplumber else ""
    detected = find_answers_in_text(text)

    # 2) Si no detectó nada o detectó pocas preguntas, intentar OCR de imágenes
    if len(detected) < max(1, len(answer_key)//2):
        images = render_pdf_pages_to_images(pdf_bytes)
        for img in images:
            ocr_text = ocr_image_to_text(img) if pytesseract and Image else ""
            if ocr_text:
                more = find_answers_in_text(ocr_text)
                for k,v in more.items():
                    if k not in detected:
                        detected[k] = v
            # si ya detectamos todas las claves, podemos salir
            if len(detected) >= len(answer_key):
                break

    # 3) Ahora calcular nota: contar correctas sobre total de preguntas en la clave
    total_q = len(answer_key)
    if total_q == 0:
        nota = 0.0
    else:
        correct = 0
        for q, correct_ans in answer_key.items():
            student_ans = detected.get(q)
            if student_ans == correct_ans:
                correct += 1
        nota = (correct / total_q) * 20.0
        nota = round(nota, 2)

    return {
        'detected': detected,
        'score': nota,
        'correct_count': sum(1 for q in answer_key if detected.get(q)==answer_key[q]),
        'total': total_q
    }

# ----------------------------- STREAMLIT UI -----------------------------

st.set_page_config(page_title="Exam Auto Grader - Simulación n8n", layout='wide')
st.title("Exam Auto Grader — Simulación n8n")
st.caption("Procesa PDFs de exámenes (marcas X o círculo). Escala 0-20. Nota mínima aprobatoria: 14")

with st.form(key='key_form'):
    col1, col2 = st.columns([3,1])
    with col1:
        course_name = st.text_input('Nombre del curso', value='Curso X')
        course_code = st.text_input('Código del curso', value='C-001')
        key_input = st.text_input('Clave de respuestas (ej: 1:a, 2:d, 3:e, 4:v, 5:f)')
    with col2:
        st.markdown('''**Configuración**\n- Escala: 0-20\n- Aprobación: 14\n- Máx PDFs: 30''')
        min_aprob = st.number_input('Nota mínima aprobatoria', value=14.0, step=0.5)
    submitted = st.form_submit_button('Guardar clave')

if submitted:
    st.success('Clave guardada')

# Parsear clave cada vez (si vacía, será {})
answer_key = parse_key_string(key_input or "")

st.write('Clave parseada:', answer_key)

st.markdown('---')

uploaded = st.file_uploader('Sube hasta 30 archivos PDF (selección múltiple)', accept_multiple_files=True, type=['pdf'])
if uploaded:
    if len(uploaded) > 30:
        st.error('Máximo 30 PDFs. Sólo se procesarán los primeros 30.')
        uploaded = uploaded[:30]

# Área de simulación n8n
simulate_col = st.container()

results = []

# Botón simulado que 'conecta' con n8n (pero en realidad todo corre localmente)
if st.button('Analizar en n8n (simulado)'):
    if not uploaded:
        st.warning('Sube al menos 1 PDF para analizar')
    elif not answer_key:
        st.warning('Ingresa la clave primero (guardar)')
    else:
        progress_bar = st.progress(0)
        log_area = st.empty()
        total = len(uploaded)
        for i, up in enumerate(uploaded, start=1):
            # leer bytes
            pdf_bytes = up.read()
            # procesar
            res = grade_single_pdf(pdf_bytes, answer_key)
            res['filename'] = up.name
            results.append(res)
            # update fake logs and progress
            log_area.text(f"Procesando {up.name} ({i}/{total}) — usando modelo: Google Gemini 1.5 (simulado)")
            progress = int(i/total * 100)
            progress_bar.progress(progress)
        progress_bar.progress(100)
        st.success('Análisis completado (simulado en n8n)')

# Si ya hay resultados (por haber corrido el análisis), mostrar tabla
if results:
    df = pd.DataFrame([{'pdf': r['filename'], 'nota': r['score'], 'correctas': r['correct_count'], 'total': r['total']} for r in results])
    st.subheader('Resultados individuales')
    st.dataframe(df.sort_values('nota', ascending=False))

    # Estadísticas
    promedio = round(df['nota'].mean(),2)
    aprobados = df[df['nota']>=min_aprob]
    desaprobados = df[df['nota']<min_aprob]
    pct_aprob = round(len(aprobados)/len(df)*100,2) if len(df)>0 else 0
    mayor = df['nota'].max()
    menor = df['nota'].min()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric('Promedio general', promedio)
    col2.metric('Promedio aprobados', round(aprobados['nota'].mean(),2) if len(aprobados)>0 else 0)
    col3.metric('% Aprobados', f"{pct_aprob}%")
    col4.metric('Mayor / Menor', f"{mayor} / {menor}")

    st.markdown('---')

    # Histogram
    fig, ax = plt.subplots(figsize=(6,3))
    ax.hist(df['nota'], bins=10)
    ax.set_title('Distribución de notas')
    ax.set_xlabel('Nota (0-20)')
    ax.set_ylabel('Cantidad de PDFs')
    st.pyplot(fig)

    # Mostrar lista detallada con respuestas detectadas
    expander = st.expander('Ver detealles por PDF')
    with expander:
        for r in results:
            st.write(f"**{r['filename']}** — Nota: {r['score']}")
            st.write('Respuestas detectadas:', r['detected'])

    # Botón para exportar reporte en PDF
    def generate_report_pdf(results_list, key, course_name, course_code, min_aprob):
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0,10, f'Exam Auto Grader - Reporte', ln=True, align='C')
        pdf.set_font('Arial', '', 10)
        pdf.cell(0,6, f'Curso: {course_name}    Codigo: {course_code}    Fecha: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', ln=True)
        pdf.ln(4)

        # Tabla de notas
        pdf.set_font('Arial','B',12)
        pdf.cell(60,8,'PDF',1)
        pdf.cell(30,8,'Nota',1)
        pdf.cell(30,8,'Correctas',1)
        pdf.cell(30,8,'Total',1)
        pdf.ln()
        pdf.set_font('Arial','',10)
        for r in results_list:
            pdf.cell(60,8, r['filename'][:40],1)
            pdf.cell(30,8, str(r['score']),1)
            pdf.cell(30,8, str(r['correct_count']),1)
            pdf.cell(30,8, str(r['total']),1)
            pdf.ln()

        # Stats summary
        df2 = pd.DataFrame([{'nota': r['score']} for r in results_list])
        promedio = round(df2['nota'].mean(),2)
        mayor = df2['nota'].max()
        menor = df2['nota'].min()
        aprobados = df2[df2['nota']>=min_aprob]

        pdf.ln(6)
        pdf.set_font('Arial','B',12)
        pdf.cell(0,6,'Resumen de Estadísticas', ln=True)
        pdf.set_font('Arial','',11)
        pdf.cell(0,6, f'Promedio general: {promedio}', ln=True)
        pdf.cell(0,6, f'Mayor nota: {mayor}', ln=True)
        pdf.cell(0,6, f'Menor nota: {menor}', ln=True)
        pdf.cell(0,6, f'Aprobados: {len(aprobados)} / {len(df2)}', ln=True)

        # Guardar temporalmente
        tmpf = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        pdf.output(tmpf.name)
        return tmpf.name

    report_file = generate_report_pdf(results, answer_key, course_name, course_code, min_aprob)
    with open(report_file, 'rb') as f:
        st.download_button('Descargar reporte PDF', data=f, file_name=f"reporte_{course_code}.pdf", mime='application/pdf')

    st.info('El reporte incluye: notas por PDF, estadísticas (promedio, mayor, menor) y conteo aprobados/desaprobados.')

else:
    st.info('Aún no hay resultados. Ingresa la clave y sube PDFs, luego presiona "Analizar en n8n (simulado)"')

# ----------------------------- Notas e instrucciones -----------------------------
st.markdown('---')
st.header('Notas técnicas y dependencias')
st.markdown('''
- Este script intenta múltiples estrategias para extraer respuestas: extracción de texto (pdfplumber), renderizado a imágenes (PyMuPDF) y OCR (pytesseract).
- **Recomendado** instalar: `pdfplumber`, `PyMuPDF` (`fitz`), `pytesseract`, `Pillow`, `fpdf`.
- Si no deseas o no puedes instalar `tesseract-ocr` en el servidor, la app intentará extraer texto directamente del PDF. Para PDFs escaneados, la precisión dependerá de tener OCR disponible.

Instalación (ejemplo):
```
pip install streamlit pdfplumber pymupdf pytesseract pillow fpdf
# y en el servidor (si quieres OCR): instalar tesseract (sistema operativo)
```

Despliegue en Streamlit Cloud / Streamlit Community:
1. Crea un repo en GitHub con este archivo `streamlit_exam_grader_app.py`.
2. En Streamlit Cloud (https://streamlit.io), conecta tu repo y selecciona el archivo como `app.py` (o renombral0 como `app.py`).
3. Asegúrate de declarar las dependencias en `requirements.txt`.

''')

# Fin del archivo
