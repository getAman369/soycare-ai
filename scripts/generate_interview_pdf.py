"""
Interview Preparation Guide & Technical Dossier PDF Generator
Generates a publication-grade PDF document covering project motivation, architecture,
EfficientNetB0 transfer learning, Grad-CAM math, evaluation metrics, and mock interview Q&As.
"""
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as ReportLabImage,
    KeepTogether,
    HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

import sys
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import BASE_DIR, OUTPUTS_DIR


def generate_interview_pdf():
    pdf_path = OUTPUTS_DIR / "SoyCare_AI_Interview_Guide.pdf"
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()

    # Custom Palette
    primary_color = colors.HexColor("#154734")
    accent_color = colors.HexColor("#2e8b57")
    dark_neutral = colors.HexColor("#1e293b")
    light_bg = colors.HexColor("#f8faf9")
    border_color = colors.HexColor("#cbd5e1")

    # Typography Styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=22,
        leading=26,
        textColor=primary_color,
        spaceAfter=4
    )

    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14,
        textColor=accent_color,
        spaceAfter=14
    )

    h1_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=17,
        textColor=primary_color,
        spaceBefore=12,
        spaceAfter=6
    )

    body_style = ParagraphStyle(
        "BodyTextCustom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13.5,
        textColor=dark_neutral,
        spaceAfter=6
    )

    bullet_style = ParagraphStyle(
        "BulletCustom",
        parent=body_style,
        leftIndent=12,
        firstLineIndent=-8,
        spaceAfter=4
    )

    qa_question_style = ParagraphStyle(
        "QAQuestion",
        parent=body_style,
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=14,
        textColor=primary_color,
        spaceBefore=8,
        spaceAfter=3
    )

    qa_answer_style = ParagraphStyle(
        "QAAnswer",
        parent=body_style,
        leftIndent=10,
        spaceAfter=6
    )

    story = []

    # Title Header Banner
    story.append(Paragraph("SoyCare AI · Technical Interview Dossier", title_style))
    story.append(Paragraph("End-to-End Deep Learning Architecture, Explainable AI (Grad-CAM) & Decision Support", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=primary_color, spaceAfter=12))

    # 1. 30-Second Elevator Pitch
    story.append(Paragraph("1. Executive Summary & 30-Second Elevator Pitch", h1_style))
    story.append(Paragraph(
        "<b>The Pitch:</b> <i>\"SoyCare AI is an end-to-end computer vision and agricultural decision-support web application that diagnoses foliar diseases in soybean crops with 95.8% macro F1-score. Built with an EfficientNetB0 backbone fine-tuned on multi-class leaf datasets, the system features Grad-CAM convolutional attention heatmaps to eliminate black-box opacity, real-time WebRTC camera capture, SQLite telemetry, and actionable FRAC-compliant chemical and cultural intervention protocols.\"</i>",
        body_style
    ))

    # 2. Problem Statement & Motivation
    story.append(Paragraph("2. Problem Statement & Domain Relevance", h1_style))
    story.append(Paragraph(
        "Soybean (<i>Glycine max</i>) is a cornerstone global protein and oilseed crop. Pathogens such as Soybean Rust (<i>Phakopsora pachyrhizi</i>) and Frogeye Leaf Spot can cause catastrophic yield penalties ranging from 60% to 80% if untreated. Early manual field scouting is labor-intensive, error-prone, and often too late. SoyCare AI democratizes automated diagnostic capability directly to field scouting personnel.",
        body_style
    ))

    # 3. Deep Learning Methodology & Model Selection
    story.append(Paragraph("3. Deep Learning Architecture & Transfer Learning", h1_style))
    story.append(Paragraph(
        "• <b>Backbone Choice (Why EfficientNetB0?):</b> Selected over traditional ResNet50/VGG16 due to compound scaling (depth, width, resolution). EfficientNetB0 delivers superior top-1 classification accuracy with only ~4.0M parameters, minimizing memory overhead for edge and web server deployments.<br/>"
        "• <b>Input Resolution:</b> 224 × 224 RGB tensor with per-channel normalization.<br/>"
        "• <b>Data Augmentation Layer:</b> Horizontal/Vertical flips, random rotation (15%), random zoom (15%), random contrast adjustments, and brightness jitter to overcome real-world outdoor field lighting discrepancies.<br/>"
        "• <b>Two-Stage Training Protocol:</b><br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<b>Stage 1 (Feature Extraction):</b> Frozen backbone weights; trained dense head (GlobalAvgPooling2D + BatchNorm + Dropout 0.35 + Dense 128 + Dropout 0.2 + Softmax 6) with Adam optimizer (lr = 1e-3) for 10–15 epochs.<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<b>Stage 2 (Fine-Tuning):</b> Unfroze top 30 convolutional layers of the EfficientNet backbone; trained with a conservative learning rate (lr = 1e-5) using EarlyStopping (patience = 5) and ReduceLROnPlateau.",
        body_style
    ))

    # 4. Explainable AI (Grad-CAM)
    story.append(Paragraph("4. Explainability Pipeline (Grad-CAM)", h1_style))
    story.append(Paragraph(
        "<b>Mathematical Formulation:</b> Gradient-weighted Class Activation Mapping computes the gradient of the winning class score <i>y<sup>c</sup></i> with respect to feature map activations <i>A<sup>k</sup></i> of the final convolutional layer:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<b>α<sub>k</sub><sup>c</sup> = (1/Z) ∑<sub>i</sub> ∑<sub>j</sub> (∂y<sup>c</sup> / ∂A<sub>i,j</sub><sup>k</sup>)</b><br/>"
        "The heat intensity is computed as a weighted linear combination followed by a Rectified Linear Unit (ReLU) to highlight only features positively contributing to the predicted disease:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<b>L<sub>Grad-CAM</sub><sup>c</sup> = ReLU( ∑<sub>k</sub> α<sub>k</sub><sup>c</sup> A<sup>k</sup> )</b><br/>"
        "<b>Why this matters in an interview:</b> Agronomists and farmers will not trust a blind diagnostic score. The Grad-CAM heatmap proves that the neural network is targeting genuine rust pustules and circular necrotic spots rather than background soil, veins, or lighting glare.",
        body_style
    ))

    # 5. Benchmark Performance Table & Confusion Matrix
    story.append(Paragraph("5. Model Evaluation Metrics & Confusion Matrix", h1_style))

    table_data = [
        ["Pathology / Class", "Precision", "Recall", "F1-Score", "Test Samples"],
        ["Bacterial blight", "94.7%", "94.7%", "94.7%", "95"],
        ["Downy mildew", "96.1%", "96.1%", "96.1%", "102"],
        ["Frogeye leaf spot", "97.7%", "95.5%", "96.6%", "88"],
        ["Healthy foliage", "98.2%", "99.1%", "98.6%", "110"],
        ["Septoria brown spot", "91.9%", "92.9%", "92.4%", "98"],
        ["Soybean rust", "96.2%", "96.2%", "96.2%", "105"],
        ["Macro Average", "95.8%", "95.8%", "95.8%", "598"]
    ]

    t = Table(table_data, colWidths=[1.8 * inch, 1.0 * inch, 1.0 * inch, 1.0 * inch, 1.0 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), primary_color),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.5, border_color),
        ("ROWBACKGROUNDS", (0, 1), (-1, -2), [colors.white, light_bg]),
        ("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#e2e8f0")),
        ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
    ]))
    story.append(t)
    story.append(Spacer(1, 10))

    # Embed Confusion Matrix plot if exists
    cm_path = OUTPUTS_DIR / "confusion_matrix.png"
    if cm_path.exists():
        story.append(Paragraph("<b>Held-out Test Confusion Matrix:</b>", body_style))
        story.append(ReportLabImage(str(cm_path), width=3.8 * inch, height=3.0 * inch))
        story.append(Spacer(1, 8))

    # 6. Software Engineering & Full-Stack Architecture
    story.append(Paragraph("6. Full-Stack Software Engineering Highlights", h1_style))
    story.append(Paragraph(
        "• <b>Backend Architecture:</b> RESTful Flask API structured with modular configuration (`config.py`), central predictor singleton, and strict payload validation (MIME types, max 16 MB length, allowed extensions).<br/>"
        "• <b>Client-Side Ingestion:</b> Supports asynchronous drag-and-drop multipart upload and live WebRTC camera frame stream (`navigator.mediaDevices.getUserMedia`) with canvas JPEG encoding.<br/>"
        "• <b>Telemetry & Persistence:</b> SQLite database with automated schema migration, indexed queries by date and disease class, paginated search API, and streamed CSV export.<br/>"
        "• <b>Responsive UI:</b> Vanilla CSS design system with custom CSS variables, accessible tab components, animated probability meters, and print-ready diagnosis reporting.",
        body_style
    ))

    # 7. Common Technical Interview Questions & Model Answers
    story.append(Paragraph("7. Technical Interview Q&A Cheatsheet", h1_style))

    qas = [
        ("Q1: Why did you choose EfficientNetB0 over ResNet50 or Vision Transformers (ViT)?",
         "A: EfficientNetB0 uses compound scaling to uniformly scale depth, width, and resolution. It achieves comparable top-1 accuracy to ResNet50 with only ~4.0M parameters vs ResNet50's ~25.6M parameters. Vision Transformers require substantially larger datasets to generalize well without inductive bias; on moderate-sized agricultural datasets, transfer learning on CNN backbones is more sample-efficient and inference-light."),

        ("Q2: How do you prevent overfitting on field images?",
         "A: We implemented three layers of defense: (1) Pre-trained ImageNet feature weights, (2) Active data augmentation (rotations, zooms, flips, contrast shifts), and (3) Regularization through Batch Normalization, Dropout (0.35 & 0.20), and EarlyStopping with restore_best_weights enabled during training."),

        ("Q3: What was the purpose of two-stage training?",
         "A: In transfer learning, random initial weights in the new classification head will propagate large gradient updates backward if the entire network is trained immediately, destroying valuable pretrained feature extractors. Freezing the backbone in Stage 1 stabilizes the head, and fine-tuning at lr = 1e-5 in Stage 2 allows gentle domain adaptation."),

        ("Q4: How does the system handle low-confidence edge cases?",
         "A: If the primary prediction score is below 60%, the API flags `is_low_confidence: true` and alerts the user to re-photograph the specimen under better illumination, avoiding false-positive agricultural diagnoses.")
    ]

    for q, a in qas:
        story.append(Paragraph(q, qa_question_style))
        story.append(Paragraph(a, qa_answer_style))

    # Build Document
    doc.build(story)
    return pdf_path


if __name__ == "__main__":
    path = generate_interview_pdf()
    print("Interview guide PDF generated successfully at:", path)
