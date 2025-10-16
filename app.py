import streamlit as st
import pdfplumber
import re
import io
import statistics
from fpdf import FPDF
import google.generativeai as genai

# ================================
# CONFIGURACIÓN INICIAL GEMINI
# ================================
genai.configure(api_key="AIzaSyBBxiisLsoPKLvKdWpjcE7cTtyXsRWQN7s")
model = genai.GenerativeModel("gemini-1.5-flash")

# ================================
# FUNCIÓN PARA EXTRAER TEXTO DE PDF
# ================================
def extract_text_from_pdf(file):
    with pdfplumber.open(file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

# ================================
# FUNCIÓN PARA DETECTAR RESPUESTAS MARCADAS CON X
# ================================
def detect_answers_with_gemini(text):
    prompt = f"""
    Analiza el siguiente texto de un examen y devuelve solo las respuestas seleccionadas,
    en formato: 1:a, 2:c, 3:e, 4:v, 5:f. Solo devuelve eso.

    Examen:
    {text}
    """
    response = model.generate_content(prompt)
    return response.text.strip()

# ================================
# FUNCIÓN PARA COMPARAR RESPUESTAS
# ================================
def compare_answers(correct_answers, student_answers):
    correct_dict = {k.strip(): v.strip().lower() for k, v in [x.split(":") for x in correct_answers.split(",")]}
    student_dict = {k.strip(): v.strip().lower() for k, v in [x.split(":") for x in student_answers.split(",") if ":" in x]}

    score = 0
    total = len(correct_dict)

    for q, ans in correct_dict.items():
        if q in student_dict and student_dict[q] == ans:
            score += 1

    grade = round((score / total) * 20, 2)
    return grade

# ================================
# FUNCIÓN PARA GENERAR REPORTE PDF
# ================================
def generate_pdf_report(results, course_name, course_code):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 14)
    pdf.cell(200, 10, f"Reporte de Resultados - {course_name} ({course_code})", ln=True, align="C")
    pdf.ln(10)
    pdf.set_font("Arial", "", 12)

    for r in results:
        pdf.cell(200, 10, f"{r['filename']}: {r['grade']}", ln=True)

    pdf.ln(10)
    grades = [r['grade'] for r in results]
    avg = round(statistics.mean(grades), 2)
    passed = len([g for g in grades if g >= 10.5])
    failed = len(grades) - passed

    pdf.multi_cell(200, 10, f"""
Promedio General: {avg}
Aprobados: {passed}
Desaprobados: {failed}
Mayor nota: {max(grades)}
Menor nota: {min(grades)}
    """)

    pdf_output = io.BytesIO()
    pdf.output(pdf_output)
    pdf_output.seek(0)
    return pdf_output

# ================================
# INTERFAZ STREAMLIT
# ================================
st.set_page_config(page_title="Auto Calificador de Exámenes", layout="wide")

st.title("📘 Auto Calificador de Exámenes (Google Gemini + Streamlit + n8n)")

course_name = st.text_input("📘 Nombre del curso:")
course_code = st.text_input("🔢 Código del curso:")
correct_answers = st.text_area("✅ Ingrese las respuestas correctas (ej: 1:a, 2:c, 3:e, 4:v, 5:f)")

uploaded_files = st.file_uploader("📂 Subir hasta 30 exámenes en PDF", type="pdf", accept_multiple_files=True)

if st.button("🚀 Analizar en n8n (Simulación)"):
    if not correct_answers or not uploaded_files:
        st.error("Por favor, ingrese las respuestas correctas y suba al menos un PDF.")
    else:
        st.info("🔄 Enviando tareas a n8n... (simulación)")
        st.toast("Procesando PDFs con Gemini...", icon="🤖")

        results = []
        for file in uploaded_files:
            text = extract_text_from_pdf(file)
            student_answers = detect_answers_with_gemini(text)
            grade = compare_answers(correct_answers, student_answers)
            results.append({"filename": file.name, "grade": grade})

        grades = [r["grade"] for r in results]
        avg = round(statistics.mean(grades), 2)
        passed = len([g for g in grades if g >= 10.5])
        failed = len(grades) - passed

        st.success("✅ Análisis completado con éxito (simulación n8n terminada)")
        st.subheader("📊 Resultados")
        st.dataframe(results)

        st.markdown(f"""
        **Promedio general:** {avg}  
        **Aprobados:** {passed}  
        **Desaprobados:** {failed}  
        **Mayor nota:** {max(grades)}  
        **Menor nota:** {min(grades)}
        """)

        pdf_report = generate_pdf_report(results, course_name, course_code)
        st.download_button("📄 Descargar reporte PDF", pdf_report, file_name="reporte_resultados.pdf")
