import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# ==== CONFIG ====
GEMINI_API_KEY = "placeholder"
genai.configure(api_key=GEMINI_API_KEY)

# ==== EMBEDDING MODEL ====
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# ==== PLACEHOLDER DOCUMENTS ====
documents = [
    # ---------------- PM Context ----------------
    "PM: Proyek NusantaraPay (2023) mengembangkan aplikasi mobile e-wallet dengan fitur QRIS payment, top-up saldo, dan cashback program. Total anggaran Rp650.000.000, selesai dalam 6 bulan.",
    "PM: Proyek AgroInsight (2024) adalah dashboard SaaS untuk analitik pertanian. Fitur meliputi integrasi IoT sensor, visualisasi data panen, dan multi-tenant support. Anggaran Rp950.000.000, mundur 1 bulan karena perubahan requirement.",
    "PM: Proyek EduLink (2022) membangun platform pembelajaran sosial dengan forum diskusi, peer review, dan gamification badge. Anggaran Rp450.000.000, selesai 2 minggu lebih cepat.",
    "PM: Standar alokasi anggaran biasanya: 20% untuk desain, 50% untuk pengembangan, 30% untuk testing & project overhead.",
    "PM: Estimasi anggaran dibagi ke dalam kategori: kecil (≤Rp300 juta), menengah (Rp300–700 juta), besar (≥Rp700 juta).",

    # ---------------- UX / Design Context ----------------
    "UX: Desain mobile-first diprioritaskan. Tim UX pernah menangani aksesibilitas WCAG, dukungan multi-bahasa (Indonesia & Inggris), dan layout responsif.",
    "UX: Dalam Proyek NusantaraPay, keputusan UX utama adalah penggunaan bottom navigation untuk mempercepat akses ke menu pembayaran.",
    "UX: Dalam Proyek EduLink, implementasi gamification badge meningkatkan daily active user sebesar 25%.",
    "UX: Toolkit standar meliputi Figma untuk wireframing, Material Design, dan Tailwind untuk komponen UI.",
    "UX: User testing dilakukan per sprint (2 minggu) dengan minimal 5 partisipan setiap siklus.",

    # ---------------- Dev Context ----------------
    "Dev: Anggota tim termasuk Andi (Backend, Node.js & PostgreSQL), Sari (Frontend, React & Next.js), dan Budi (Mobile, Flutter).",
    "Dev: Praktik keamanan meliputi autentikasi JWT, role-based access control, dan enkripsi end-to-end untuk data sensitif.",
    "Dev: Dalam Proyek AgroInsight, Sari mengimplementasikan dashboard interaktif dengan Recharts.js untuk filtering data multi-tenant.",
    "Dev: Budi mengembangkan aplikasi mobile lintas platform menggunakan Flutter dengan 85% shared codebase untuk iOS & Android.",
    "Dev: Andi menangani optimisasi query PostgreSQL hingga 2 juta record dengan caching & indexing.",
    "Dev: Stack utama mencakup React (frontend), Node.js (backend), PostgreSQL (database), dan Docker (deployment).",

    # ---------------- Alignment Context ----------------
    "ALIGN: Setiap proyek dimulai dengan discovery workshop antara PM, UX, dan Dev untuk menyamakan kebutuhan klien dan feasibility teknis.",
    "ALIGN: Dokumentasi requirement ditulis oleh PM, lalu diterjemahkan UX menjadi user flow, sebelum Dev menentukan estimasi effort.",
    "ALIGN: Budget forecasting disetujui setelah validasi lintas tim, dengan trade-off feature vs biaya selalu dibahas bersama.",
]


doc_embeddings = embed_model.encode(documents)
dimension = doc_embeddings.shape[1]

# ==== CREATE FAISS VECTOR DB ====
index = faiss.IndexFlatL2(dimension)
index.add(np.array(doc_embeddings))

# ==== RETRIEVAL FUNCTION ====
def retrieve(query, top_k=3):
    query_embedding = embed_model.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)
    return [documents[i] for i in indices[0]]

# ==== PROMPT TEMPLATE ====
def build_prompt(query, retrieved_chunks):
    context = "\n".join(retrieved_chunks)

    return f"""
You are an AI assistant that helps software houses accelerate the initial product requirement stage of the software development lifecycle.

Your tasks:
1. Translate client requirements into a structured Project Requirement Document (PRD).
2. Ensure the PRD includes:
   - Executive Summary
   - Project Objectives
   - Scope of Work
   - Stakeholder Requirements (PM, UX, Dev)
   - Functional Requirements
   - Non-Functional Requirements
   - Proposed Solution Options
   - Budget and Resource Forecast
   - Risks & Open Questions
   - Alignment Notes (ensuring all stakeholders are on the same page)

Client Query:
{query}

Relevant Context Documents:
{context}

Final Output (PRD format):
---
# Project Requirement Document

**1. Executive Summary**  
- Concise overview of client request and business context.

**2. Project Objectives**  
- Key goals the project aims to achieve.

**3. Scope of Work**  
- Features and deliverables to be included.  
- Explicitly mention what is out of scope.

**4. Stakeholder Requirements**  
- **Project Manager (PM):** Action items, solutioning, resource/budget considerations.  
- **UX / Design (UX):** Action items, design considerations, solutioning.  
- **Development (Dev):** Action items, technical considerations, solutioning.

**5. Functional Requirements**  
- List of clear, testable system functionalities.

**6. Non-Functional Requirements**  
- Performance, scalability, security, usability, etc.

**7. Proposed Solution Options**  
- Alternative approaches with trade-offs.

**8. Budget & Resource Forecast**  
- High-level cost estimate (in IDR).  
- Roles required and approximate effort.

**9. Risks & Open Questions**  
- Known uncertainties or missing information.

**10. Alignment Notes**  
- Key points to ensure PM, UX, and Dev perspectives stay coordinated.
---

"""

# ==== RAG SIMULATION ====
def rag_query(query):
    retrieved_chunks = retrieve(query, top_k=3)
    prompt = build_prompt(query, retrieved_chunks)
    response = genai.GenerativeModel("gemini-1.5-flash").generate_content(prompt)
    return response.text

# ==== PLACEHOLDER INPUT QUERY ====
client_query = "Nurabites, sebuah UMKM kuliner yang berfokus pada produk camilan sehat berbahan alami, membutuhkan landing page sederhana dan menarik untuk memperkenalkan brand serta produk unggulannya kepada calon pelanggan. Landing page ini diharapkan dapat menampilkan identitas brand yang ramah dan modern, menonjolkan keunggulan produk (sehat, enak, praktis), serta menyediakan call-to-action yang jelas seperti “Beli Sekarang” atau “Hubungi Kami” untuk mendorong konversi."
print(rag_query(client_query))
