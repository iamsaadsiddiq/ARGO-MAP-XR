# App.py
"""
ARGO MAP XR — Single-file Streamlit app
Advanced Research & Guidance for Outer-space — Mission Analytics & Planning with Xtended Reality

Instructions:
 - Place a space background MP4 at assets/videos/video.mp4 (or change VIDEO_PATH)
 - Replace SKETCHFAB_MODEL_SRC with a public Sketchfab embed URL if desired
 - Install dependencies as needed. Minimal set: streamlit, pandas, numpy, plotly, scikit-learn, networkx, scipy, requests
 - Optional extras: sentence-transformers (better semantic search), reportlab/unidecode/fpdf (better PDF)
 - Run: streamlit run App.py
"""
import os
import io
import math
import json
import time
import tempfile
from datetime import datetime, timezone
from typing import List, Dict, Any

import streamlit as st
st.set_page_config(page_title="ARGO MAP XR", layout="wide", initial_sidebar_state="expanded")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.integrate import odeint
from scipy import stats
import requests

# -------------------------
# Optional heavy libs (graceful fallback)
# -------------------------
HAS_STRANS = False
EMBED_MODEL = None
try:
    from sentence_transformers import SentenceTransformer
    EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    HAS_STRANS = True
except Exception:
    HAS_STRANS = False

HAS_REPORTLAB = False
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    HAS_REPORTLAB = True
except Exception:
    HAS_REPORTLAB = False

HAS_UNIDECODE = False
try:
    from unidecode import unidecode
    HAS_UNIDECODE = True
except Exception:
    HAS_UNIDECODE = False

HAS_FPDF = False
try:
    from fpdf import FPDF
    HAS_FPDF = True
except Exception:
    HAS_FPDF = False

# -------------------------
# Constants / planetary data
# -------------------------
G_CONST = 6.67430e-11
C_LIGHT = 299792458.0
PLANETS = {
    "Sun": {"mass": 1.98847e30, "radius_km": 695700.0},
    "Earth": {"mass": 5.97219e24, "radius_km": 6371.0},
    "Moon": {"mass": 7.342e22, "radius_km": 1737.4},
    "Mars": {"mass": 6.4171e23, "radius_km": 3389.5}
}

# -------------------------
# Utility helpers
# -------------------------
def now_utc_iso() -> str:
    """Return timezone-aware ISO timestamp."""
    return datetime.now(timezone.utc).isoformat()

def safe_for_pdf(s: str) -> str:
    """Return a PDF-safe string: use reportlab if available, else transliterate with unidecode, else ascii replace."""
    if HAS_REPORTLAB:
        return s
    if HAS_UNIDECODE:
        return unidecode(s)
    return s.encode("ascii", errors="replace").decode("ascii")

# -------------------------
# Demo dataset generator
# -------------------------
@st.cache_data(show_spinner=False)
def generate_demo_publications(n=200, seed=42) -> pd.DataFrame:
    np.random.seed(seed)
    topics = [
        "Bone Health", "Radiation Effects", "Plant Growth",
        "Circadian Rhythm", "Microbiome", "Muscle Atrophy",
        "Immune Response", "Psychology", "Food Systems"
    ]
    contexts = ["ISS", "Low-Earth Orbit", "Ground Analog", "Lunar Habitat", "Mars Analog"]
    rows = []
    for i in range(n):
        t = np.random.choice(topics)
        ctx = np.random.choice(contexts)
        nsub = int(np.random.exponential(8)) + 4
        eff = float(np.round(np.random.normal(0, 8), 2))
        year = int(np.random.choice(range(1995, 2025)))
        title = f"{t} study in {ctx} — #{i+1}"
        abstract = (
            f"This study investigates {t.lower()} under {ctx.lower()} conditions. "
            f"n={nsub}. Intervention: {'resistive exercise' if 'Bone' in t else 'radiation shielding' if 'Radiation' in t else 'LED lighting' if 'Plant' in t else 'sleep intervention'}. "
            f"Outcome observed: approximately {eff}% change in primary endpoint."
        )
        rows.append({
            "id": f"S{i+1000}",
            "title": title,
            "year": year,
            "authors": f"Researcher {chr(65 + (i % 26))} et al.",
            "context": ctx,
            "intervention": "resistive exercise" if "Bone" in t else ("radiation shielding" if "Radiation" in t else "lighting"),
            "outcome": "bone mineral density" if "Bone" in t else ("DNA damage markers" if "Radiation" in t else "plant biomass"),
            "abstract": abstract,
            "sample_size": nsub,
            "effect_pct": eff,
            "doi": f"10.5555/demo.{i}"
        })
    return pd.DataFrame(rows)

# -------------------------
# Embedding & Summarization
# -------------------------
@st.cache_data(show_spinner=False)
def embed_texts(texts: List[str]):
    """Return embeddings - use sentence-transformers if available else TF-IDF dense vectors."""
    if HAS_STRANS and EMBED_MODEL is not None:
        try:
            emb = EMBED_MODEL.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            return emb
        except Exception:
            pass
    vect = TfidfVectorizer(max_features=1024, stop_words='english')
    X = vect.fit_transform(texts).toarray()
    return X

def tfidf_extractive_summary(text: str, n_sent: int = 3) -> str:
    sents = [s.strip() for s in text.split('.') if len(s.strip()) > 20]
    if not sents:
        return text if len(text) < 400 else text[:400] + "..."
    vect = TfidfVectorizer(stop_words='english')
    X = vect.fit_transform(sents)
    scores = X.sum(axis=1).A1
    idx = scores.argsort()[::-1][:n_sent]
    return ". ".join([sents[i] for i in sorted(idx)]) + "."

# -------------------------
# Knowledge graph helpers (Plotly fallback)
# -------------------------
def build_cooccurrence_graph(docs: pd.Series, top_k: int = 6) -> nx.Graph:
    vect = TfidfVectorizer(max_features=2000, stop_words='english', ngram_range=(1,2))
    X = vect.fit_transform(docs.fillna('').astype(str))
    terms = vect.get_feature_names_out()
    top_idx = np.argsort(X.toarray(), axis=1)[:, ::-1][:, :top_k]
    G = nx.Graph()
    for row in top_idx:
        tnames = [terms[i] for i in row if i < len(terms)]
        for t in tnames:
            if not G.has_node(t):
                G.add_node(t, count=1)
            else:
                G.nodes[t]['count'] += 1
        for i in range(len(tnames)):
            for j in range(i+1, len(tnames)):
                u, v = tnames[i], tnames[j]
                if G.has_edge(u, v):
                    G[u][v]['weight'] += 1
                else:
                    G.add_edge(u, v, weight=1)
    to_remove = [n for n, d in G.nodes(data=True) if d.get('count', 0) < 2]
    G.remove_nodes_from(to_remove)
    return G

def plot_network_plotly(G: nx.Graph, height: int = 650) -> go.Figure:
    if G.number_of_nodes() == 0:
        fig = go.Figure()
        fig.update_layout(template='plotly_dark', height=height)
        return fig
    pos = nx.spring_layout(G, seed=42, k=0.6)
    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
    node_x, node_y, node_text, node_size = [], [], [], []
    for n, d in G.nodes(data=True):
        x, y = pos[n]
        node_x.append(x); node_y.append(y)
        node_text.append(f"{n} (count={d.get('count',1)})")
        node_size.append(min(max(d.get('count',1)*4, 6), 36))
    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(width=0.5, color='#AAAAAA'), hoverinfo='none')
    node_trace = go.Scatter(x=node_x, y=node_y, mode='markers+text',
                            marker=dict(size=node_size, color=node_size, colorscale='Viridis', showscale=False),
                            text=[n for n in G.nodes()], textposition="top center", hovertext=node_text)
    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(template='plotly_dark', height=height, showlegend=False, margin=dict(l=20, r=20, t=30, b=20))
    return fig

# -------------------------
# Orbital physics helpers
# -------------------------
def orbital_velocity(mass_kg: float, radius_m: float) -> float:
    """Circular orbital velocity: v = sqrt(G*M / r)"""
    return math.sqrt(G_CONST * mass_kg / radius_m)

def orbital_energy_per_unit_mass(mass_kg: float, r_m: float) -> float:
    """Specific orbital energy for circular orbit: - GM / (2r)"""
    return -G_CONST * mass_kg / (2.0 * r_m)

def compute_time_dilation_factor(velocity_m_s: float) -> float:
    """Lorentz gamma factor for small velocities; returns gamma."""
    v = abs(velocity_m_s)
    beta2 = (v / C_LIGHT) ** 2
    if beta2 >= 1.0:
        return float('inf')
    return 1.0 / math.sqrt(1.0 - beta2)

# -------------------------
# PDF report generation (safe)
# -------------------------
def generate_pdf_report_from_card(card: Dict[str, Any]) -> bytes:
    """Generate a UTF-8-safe PDF report from an insight card with fallback strategies."""
    text = f"Insight Report {card.get('id','')}\n\nGenerated: {now_utc_iso()}\n\nQuery: {card.get('query','')}\n\nSummary:\n{card.get('summary','')}\n\nSupporting papers:\n"
    for s in card.get('supporting', []):
        text += f"- {s.get('id','')}: {s.get('title','')} ({s.get('year','')}) n={s.get('sample_size','')}, effect={s.get('effect_pct','')}\n"
    safe_text = text
    if HAS_UNIDECODE and not HAS_REPORTLAB:
        safe_text = unidecode(safe_text)
    # reportlab preferred
    if HAS_REPORTLAB:
        try:
            # try register a TTF if exists on system
            candidates = [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
                "C:\\Windows\\Fonts\\DejaVuSans.ttf",
                "C:\\Windows\\Fonts\\Arial.ttf"
            ]
            font_file = None
            for c in candidates:
                if os.path.exists(c):
                    font_file = c
                    break
            styles = getSampleStyleSheet()
            if font_file:
                try:
                    pdfmetrics.registerFont(TTFont("SelectedFont", font_file))
                    styleN = ParagraphStyle('Normal', parent=styles['Normal'], fontName='SelectedFont', fontSize=10, leading=12)
                except Exception:
                    styleN = styles['Normal']
            else:
                styleN = styles['Normal']
            buffer = io.BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=letter)
            story = []
            for para in safe_text.split('\n\n'):
                story.append(Paragraph(para.replace('\n','<br/>'), styleN))
                story.append(Spacer(1, 8))
            doc.build(story)
            pdf_bytes = buffer.getvalue()
            buffer.close()
            return pdf_bytes
        except Exception as e:
            print("reportlab failed:", e)
    # fallback: FPDF with transliteration
    if HAS_FPDF:
        try:
            safe_ascii = safe_text
            if HAS_UNIDECODE:
                safe_ascii = unidecode(safe_text)
            pdf = FPDF()
            pdf.add_page()
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.set_font("Arial", size=11)
            for line in safe_ascii.splitlines():
                pdf.multi_cell(0, 6, line)
            pdf_bytes = pdf.output(dest='S').encode('latin-1', errors='ignore')
            return pdf_bytes
        except Exception as e:
            print("FPDF fallback failed:", e)
    # final fallback: plain UTF-8 bytes
    return safe_text.encode('utf-8', errors='replace')

# -------------------------
# UI: styles and media
# -------------------------
st.markdown("""
<style>
/* background video */
.video-bg {
  position: fixed;
  right: 0;
  bottom: 0;
  min-width: 100%;
  min-height: 100%;
  width: auto;
  height: auto;
  z-index: -1;
  object-fit: cover;
  filter: brightness(0.45) saturate(1.02);
}
.header-title { font-size: 36px; font-weight:700; color: #FFFFFF; margin-bottom: 2px; }
.header-sub { color:#9fb0da; margin-top: 2px; margin-bottom: 10px; }
.section-card { background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); border-radius: 10px; padding: 12px; box-shadow: 0 6px 20px rgba(2,6,23,0.6); }
.small-muted { color: #9fb0da; font-size:13px; }
</style>
""", unsafe_allow_html=True)


# Top header / hero small
st.markdown("<div style='padding:14px 18px 0px 18px;'><div class='header-title'>ARGO MAP XR</div><div class='header-sub'>Advanced Research & Guidance for Outer-space — Mission Analytics & Planning with Xtended Reality</div></div>", unsafe_allow_html=True)



# -------------------------
# Sidebar navigation
# -------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "Overview",
    "AI & ML Insights",
    "Bioscience Publications",
    "Knowledge Graph",
    "3D Simulations (Human)",
    "Astrophysics & Orbits",
    "Space Weather & Forecasting",
    "Report Generator",
    "VR Experience"
])

# Dataset: upload or demo
st.sidebar.markdown("---")
use_demo = st.sidebar.checkbox("Use demo publications dataset", value=True)
uploaded_file = st.sidebar.file_uploader("Upload NASA publications CSV", type=["csv"], accept_multiple_files=False)

if use_demo or uploaded_file is None:
    df = generate_demo_publications(n=160)
    st.sidebar.text("Demo dataset loaded")
else:
    try:
        df = pd.read_csv(uploaded_file)
        # ensure standard columns exist
        for c in ["id","title","year","authors","abstract","intervention","outcome","context","sample_size","effect_pct","doi"]:
            if c not in df.columns:
                df[c] = ""
        st.sidebar.success("CSV loaded")
    except Exception as e:
        st.sidebar.error("Failed to load CSV: using demo dataset")
        df = generate_demo_publications(n=160)

# normalize numeric columns
df['year'] = pd.to_numeric(df['year'], errors='coerce').fillna(df['year'].median() if 'year' in df and df['year'].notna().any() else 2008).astype(int)
df['sample_size'] = pd.to_numeric(df['sample_size'], errors='coerce').fillna(6).astype(int)
df['effect_pct'] = pd.to_numeric(df['effect_pct'], errors='coerce').fillna(0.0).astype(float)

# -------------------------
# Pages
# -------------------------
if page == "Overview":
    st.header("Overview — ARGO MAP XR")
    st.markdown("A unified platform for summarizing NASA bioscience publications, exploring knowledge graphs, simulating human health impacts in space, and visualizing mission-level insights.")
    st.markdown("### Quick stats")
    c1,c2,c3 = st.columns(3)
    c1.metric("Publications", len(df))
    c2.metric("Time span", f"{df['year'].min()} — {df['year'].max()}")
    c3.metric("Unique outcomes", df['outcome'].nunique())
    st.markdown("---")
    st.markdown("### Embedded Human Anatomy (Sketchfab)")
    st.markdown("Replace `SKETCHFAB_MODEL_SRC` with any public Sketchfab embed URL to swap the model.")
    SKETCHFAB_MODEL_SRC = "https://sketchfab.com/models/9b0b079953b840bc9a13f524b60041e4/embed"
    sketchfab_html = f"""
    <div style="display:flex;gap:12px;flex-wrap:wrap;">
      <iframe title="Human Anatomy Sketchfab" width="640" height="480"
        src="{SKETCHFAB_MODEL_SRC}" frameborder="0" allow="autoplay; fullscreen; vr" allowfullscreen mozallowfullscreen="true" webkitallowfullscreen="true"></iframe>
      <div style="flex:1; min-width:300px; color: white;">
        <h3>Human Digital Twin</h3>
        <p>Interactive 3D anatomy model embedded via Sketchfab. Use the UI to simulate effects below: bone/muscle mass, radiation accumulated dose, oxygen consumption.</p>
        <p class="small-muted">Tip: click the model's "Download" or "Share" on Sketchfab if you want the glTF to host locally.</p>
      </div>
    </div>
    """
    st.components.v1.html(sketchfab_html, height=520, scrolling=True)

elif page == "AI & ML Insights":
    st.header("AI & ML Insights")
    st.markdown("Upload dataset of publications or use the demo. The system will extract summaries, trends, and enable semantic search.")
    col1, col2 = st.columns([2,1])
    with col1:
        st.subheader("Dataset preview")
        st.dataframe(df.head(12))
        st.subheader("Document clustering (semantic)")
        abstracts = df['abstract'].astype(str).tolist()
        with st.spinner("Computing embeddings..."):
            embs = embed_texts(abstracts)
            try:
                n_clusters = min(8, max(2, int(len(df)/15)))
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                labs = kmeans.fit_predict(embs)
                pca = PCA(n_components=2, random_state=42)
                coords = pca.fit_transform(embs)
                plot_df = pd.DataFrame({"x": coords[:,0], "y": coords[:,1], "title": df['title'], "cluster": labs})
                fig = px.scatter(plot_df, x='x', y='y', color='cluster', hover_data=['title'], title="Semantic clusters (2D PCA)")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error("Clustering failed: " + str(e))
    with col2:
        st.subheader("Top terms (global)")
        vect = TfidfVectorizer(stop_words='english', max_features=80)
        try:
            X = vect.fit_transform(df['abstract'].astype(str))
            terms = vect.get_feature_names_out()
            freqs = X.toarray().sum(axis=0)
            top_idx = np.argsort(freqs)[::-1][:20]
            top_terms = [(terms[i], int(freqs[i])) for i in top_idx]
            term_df = pd.DataFrame(top_terms, columns=['term','count'])
            fig2 = px.bar(term_df, x='term', y='count', title="Top terms by TF-IDF frequency")
            st.plotly_chart(fig2, use_container_width=True)
        except Exception as e:
            st.info("Term extraction failed: " + str(e))
        st.subheader("Semantic Search")
        query = st.text_input("Ask a question (e.g., 'radiation effects on plants')", value="radiation effects on plant growth")
        topk = st.slider("Top K results", 1, 8, 4)
        if st.button("Search"):
            try:
                texts = df['abstract'].astype(str).tolist()
                embs = embed_texts(texts)
                if HAS_STRANS and EMBED_MODEL is not None:
                    qemb = EMBED_MODEL.encode([query], convert_to_numpy=True)[0]
                else:
                    vect_q = TfidfVectorizer(max_features=1024, stop_words='english')
                    vect_q.fit(texts)
                    embs = vect_q.transform(texts).toarray()
                    qemb = vect_q.transform([query]).toarray()[0]
                sims = cosine_similarity([qemb], embs)[0]
                idxs = np.argsort(sims)[::-1][:topk]
                st.markdown("Top matches:")
                for i in idxs:
                    r = df.iloc[int(i)]
                    st.markdown(f"- **{r['title']}** ({int(r['year'])}) — effect {r['effect_pct']}% — n={int(r['sample_size'])}")
                    st.caption(r['abstract'][:400] + "...")
                combined = " ".join(df.iloc[idxs]['abstract'].astype(str).tolist())
                summary = tfidf_extractive_summary(combined, n_sent=4)
                st.markdown("**Extractive summary (combined)**")
                st.write(summary)
            except Exception as e:
                st.error("Search failed: " + str(e))

elif page == "Bioscience Publications":
    st.header("Bioscience Publications Explorer")
    st.markdown("Upload the NASA publications CSV or use the demo. Explore extracted fields, trends, and run meta-analyses.")
    st.subheader("Dataset summary")
    st.write(df[['id','title','year','context','intervention','outcome']].head(20))
    st.subheader("Trends & charts")
    col1, col2 = st.columns(2)
    with col1:
        counts = df.groupby('year').size().reset_index(name='count')
        st.plotly_chart(px.bar(counts, x='year', y='count', title="Publications per Year", template='plotly_dark'), use_container_width=True)
    with col2:
        ctx_counts = df['context'].value_counts().reset_index()
        ctx_counts.columns = ['context','count']
        st.plotly_chart(px.pie(ctx_counts, values='count', names='context', title="Context distribution", template='plotly_dark'), use_container_width=True)
    st.subheader("Word cloud (approx via top terms)")
    try:
        vect = TfidfVectorizer(stop_words='english', max_features=80)
        X = vect.fit_transform(df['abstract'].astype(str))
        terms = vect.get_feature_names_out()
        freqs = X.toarray().sum(axis=0)
        freq_df = pd.DataFrame({'term': terms, 'count': freqs})
        freq_df = freq_df.sort_values('count', ascending=False).head(40)
        st.plotly_chart(px.bar(freq_df, x='term', y='count', title="Top terms (wordcloud-like bar chart)", template='plotly_dark'), use_container_width=True)
    except Exception as e:
        st.info("Failed to compute wordcloud-like chart: " + str(e))

elif page == "Knowledge Graph":
    st.header("Knowledge Graph Visualization")
    st.markdown("A keyword co-occurrence graph is constructed and visualized.")
    top_k_terms = st.slider("Top terms per doc (graph sensitivity)", min_value=3, max_value=12, value=6)
    with st.spinner("Building graph..."):
        G = build_cooccurrence_graph(df['abstract'].astype(str), top_k=top_k_terms)
    st.markdown(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    fig_net = plot_network_plotly(G, height=650)
    st.plotly_chart(fig_net, use_container_width=True)

elif page == "3D Simulations (Human)":
    st.header("3D Human Digital Twin — Simulation Dashboard")
    st.markdown("Simulate bone, muscle, and radiation accumulation; visualize median & uncertainty.")
    col1, col2 = st.columns([2,1])
    with col1:
        st.subheader("Simulation Controls")
        duration_days = st.slider("Mission duration (days)", 7, 540, 180)
        exercise_adherence = st.slider("Exercise adherence (0-1)", 0.0, 1.0, 0.7)
        radiation_rate = st.slider("Ambient radiation rate (mGy/day, relative)", 0.0, 5.0, 0.8)
        shielding_factor = st.slider("Shielding efficiency (0-1)", 0.0, 0.99, 0.5)
    with col2:
        st.subheader("Model Parameters")
        bone_k = st.number_input("Bone loss decay k0 (per day)", min_value=0.0001, max_value=0.02, value=0.001)
        muscle_k = st.number_input("Muscle atrophy rate (per day)", min_value=0.0001, max_value=0.02, value=0.002)
        seed = st.number_input("Random seed", value=42)
    np.random.seed(int(seed))
    days = np.arange(0, duration_days+1)
    n_mc = 120
    bone_runs = []
    muscle_runs = []
    dose_runs = []
    for i in range(n_mc):
        k0p = np.random.normal(bone_k, bone_k*0.1)
        km_p = np.random.normal(muscle_k, muscle_k*0.12)
        ex = np.clip(np.random.normal(exercise_adherence, 0.08), 0, 1)
        shield = np.clip(np.random.normal(shielding_factor, 0.05), 0, 1)
        def bone_ode(B,t):
            k_ex = 0.05
            return -abs(k0p)*B + k_ex*ex
        B0=100.0
        B = odeint(lambda y,t: bone_ode(y,t), B0, days).flatten()
        M = B0 * np.exp(-abs(km_p)*days) + 5.0*ex
        dose = np.cumsum(np.maximum(0.0, np.random.normal(radiation_rate, 0.05, size=len(days))) * (1.0-shield))
        bone_runs.append(B); muscle_runs.append(M); dose_runs.append(dose)
    bone_arr=np.vstack(bone_runs); muscle_arr=np.vstack(muscle_runs); dose_arr=np.vstack(dose_runs)
    bone_med=np.median(bone_arr,axis=0); bone_p025=np.percentile(bone_arr,2.5,axis=0); bone_p975=np.percentile(bone_arr,97.5,axis=0)
    muscle_med=np.median(muscle_arr,axis=0)
    dose_med=np.median(dose_arr,axis=0)
    fig_bone = go.Figure()
    fig_bone.add_trace(go.Scatter(x=days, y=bone_med, mode='lines', name='Bone median'))
    fig_bone.add_trace(go.Scatter(x=np.concatenate([days, days[::-1]]),
                                  y=np.concatenate([bone_p975, bone_p025[::-1]]),
                                  fill='toself', fillcolor='rgba(66,135,245,0.12)', line=dict(color='rgba(0,0,0,0)'), name='95% CI'))
    fig_bone.update_layout(title="Predicted Bone Index (median & 95% CI)", template='plotly_dark', xaxis_title='Days')
    st.plotly_chart(fig_bone, use_container_width=True)
    fig_mus = go.Figure()
    fig_mus.add_trace(go.Scatter(x=days, y=muscle_med, mode='lines', name='Muscle median'))
    fig_mus.update_layout(title="Predicted Muscle (median)", template='plotly_dark', xaxis_title='Days')
    st.plotly_chart(fig_mus, use_container_width=True)
    fig_dose = go.Figure()
    fig_dose.add_trace(go.Scatter(x=days, y=dose_med, mode='lines', name='Cumulative dose (mGy)'))
    fig_dose.update_layout(title="Predicted cumulative radiation dose (demo units)", template='plotly_dark', xaxis_title='Days')
    st.plotly_chart(fig_dose, use_container_width=True)
    st.markdown("### Interpretation")
    st.markdown(f"- Estimated median bone index change: **{bone_med[-1]-bone_med[0]:.2f}** units (lower = loss).")
    st.markdown(f"- Estimated median cumulative dose after {duration_days} days: **{dose_med[-1]:.2f}** (relative units).")

elif page == "Astrophysics & Orbits":
    st.header("Astrophysics & Orbital Calculators")
    st.markdown("Compute orbital parameters and visualize a simple 3D solar system snapshot.")
    t_days = st.slider("Simulation day (relative)", 0, 365*2, 0)
    earth_r = 1.0; moon_r = 1.2; mars_r = 1.52
    theta_e = 2*math.pi*(t_days/365.25)
    theta_m = 2*math.pi*(t_days/27.3)
    theta_ma = 2*math.pi*(t_days/687.0)
    sun = dict(x=[0], y=[0], z=[0], size=[30], name="Sun")
    earth = dict(x=[earth_r*math.cos(theta_e)], y=[earth_r*math.sin(theta_e)], z=[0], size=[8], name="Earth")
    moon = dict(x=[earth['x'][0] + 0.05*math.cos(theta_m)], y=[earth['y'][0] + 0.05*math.sin(theta_m)], z=[0], size=[3], name="Moon")
    mars = dict(x=[mars_r*math.cos(theta_ma)], y=[mars_r*math.sin(theta_ma)], z=[0], size=[6], name="Mars")
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=sun['x'], y=sun['y'], z=sun['z'], mode='markers', marker=dict(size=sun['size'], color='yellow'), name='Sun'))
    fig.add_trace(go.Scatter3d(x=earth['x'], y=earth['y'], z=earth['z'], mode='markers+text', marker=dict(size=earth['size'], color='blue'), text=['Earth'], name='Earth'))
    fig.add_trace(go.Scatter3d(x=moon['x'], y=moon['y'], z=moon['z'], mode='markers+text', marker=dict(size=moon['size'], color='lightgray'), text=['Moon'], name='Moon'))
    fig.add_trace(go.Scatter3d(x=mars['x'], y=mars['y'], z=mars['z'], mode='markers+text', marker=dict(size=mars['size'], color='red'), text=['Mars'], name='Mars'))
    fig.update_layout(template='plotly_dark', height=650, title="Simplified Solar System (animated)")
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Orbital calculators")
    body = st.selectbox("Central body", options=["Earth","Moon","Mars","Sun"], index=0)
    alt_km = st.number_input("Orbit altitude (km above body surface)", min_value=0.0, value=400.0)
    body_mass = PLANETS.get(body, PLANETS['Earth'])['mass']
    body_radius_km = PLANETS.get(body, PLANETS['Earth'])['radius_km']
    r_m = (body_radius_km + alt_km) * 1000.0
    v = orbital_velocity(body_mass, r_m)
    energy = orbital_energy_per_unit_mass(body_mass, r_m)
    gamma = compute_time_dilation_factor(v)
    st.markdown(f"Orbital velocity: **{v/1000.0:.3f} km/s**")
    st.markdown(f"Specific orbital energy (per unit mass): **{energy:.3e} J/kg**")
    st.markdown(f"Lorentz gamma for speed: **{gamma:.12f}** (time dilation negligible at orbital velocities)")

elif page == "Space Weather & Forecasting":
    st.header("Space Weather — Synthetic Index & Demo Forecast")
    days = st.slider("Days history", 30, 730, 180)
    t = np.arange(days)
    solar_index = 40 + 15 * np.sin(2 * np.pi * t / 27) + 6 * np.sin(2 * np.pi * t / 11) + np.random.normal(0, 2.5, size=days)
    df_sw = pd.DataFrame({"date": pd.date_range(end=pd.Timestamp.now(), periods=days), "index": solar_index})
    st.plotly_chart(px.line(df_sw, x='date', y='index', title="Synthetic solar index (demo)", template='plotly_dark'), use_container_width=True)
    if st.button("Forecast 30 days"):
        idx = np.arange(len(solar_index))
        slope, intercept, r, p, se = stats.linregress(idx, solar_index)
        future_idx = np.arange(len(solar_index), len(solar_index)+30)
        forecast = intercept + slope * future_idx
        combined_series = np.concatenate([solar_index, forecast])
        idx_dates = pd.date_range(end=pd.Timestamp.now() + pd.Timedelta(days=30), periods=len(combined_series))
        st.plotly_chart(px.line(x=idx_dates, y=combined_series, title="Historical + linear forecast", template='plotly_dark'), use_container_width=True)

elif page == "Report Generator":
    st.header("Report Generator (UTF-8 safe)")
    st.markdown("Build an insight card (summary + supporting studies) and export as JSON, Markdown, or PDF.")
    query_text = st.text_input("Query for insight card", value="resistive exercise bone loss")
    topk = st.slider("Top supporting studies", 1, 6, 3)
    if st.button("Generate Insight Card"):
        texts = df['abstract'].astype(str).tolist()
        embs = embed_texts(texts)
        if HAS_STRANS and EMBED_MODEL is not None:
            qemb = EMBED_MODEL.encode([query_text], convert_to_numpy=True)[0]
        else:
            vect = TfidfVectorizer(max_features=1024, stop_words='english')
            vect.fit(texts)
            embs = vect.transform(texts).toarray()
            qemb = vect.transform([query_text]).toarray()[0]
        sims = cosine_similarity([qemb], embs)[0]
        idxs = np.argsort(sims)[::-1][:topk]
        supporting = df.iloc[idxs]
        summary = tfidf_extractive_summary(" ".join(supporting['abstract'].astype(str).tolist()), n_sent=4)
        card = {
            "id": f"IC-{int(time.time())}",
            "query": query_text,
            "generated_at": now_utc_iso(),
            "summary": summary,
            "confidence": float(np.round(float(sims[idxs[0]]) if len(idxs)>0 else 0.0, 3)),
            "supporting": supporting[['id','title','year','sample_size','effect_pct','doi','context']].to_dict(orient='records')
        }
        st.json(card)
        st.download_button("Download JSON", data=json.dumps(card, indent=2), file_name=f"{card['id']}.json", mime="application/json")
        st.download_button("Download Markdown", data=("\n".join([
            f"# Insight {card['id']}",
            f"**Query**: {card['query']}",
            f"**Generated**: {card['generated_at']}",
            f"**Summary**:\n{card['summary']}",
            f"**Confidence**: {card['confidence']}",
            "\n**Supporting papers:**"
        ] + [f"- {s['id']}: {s['title']} ({s['year']}) — n={s['sample_size']}, effect={s['effect_pct']}%" for s in card['supporting']])), file_name=f"{card['id']}.md", mime="text/markdown")
        try:
            pdf_bytes = generate_pdf_report_from_card(card)
            st.download_button("Download PDF", data=pdf_bytes, file_name=f"{card['id']}.pdf", mime="application/pdf")
            st.success("PDF ready for download")
        except Exception as e:
            st.error("PDF generation failed: " + str(e))

elif page == "VR Experience":
    st.header("VR Preview (A-Frame)")
    st.markdown("Embedded A-Frame scene for immersive preview (open on mobile with WebXR support for best experience).")
    aframe = """
    <script src="https://aframe.io/releases/1.4.0/aframe.min.js"></script>
    <a-scene embedded background="color: #01020a">
      <a-camera position="0 1.6 0"></a-camera>
      <a-sphere position="0 1.25 -3" radius="1.2" color="#6A89CC"></a-sphere>
      <a-box position="1 0.5 -2" rotation="0 45 0" color="#3AAFA9"></a-box>
      <a-text value="ARGO MAP XR — VR Preview" position="-1 2 -3" color="#e6eef8"></a-text>
    </a-scene>
    """
    st.components.v1.html(aframe, height=520, scrolling=False)

# final footer / notes
st.markdown("---")
st.markdown("<div class='small-muted'>Notes: This demo app uses synthetic demo data when no CSV is provided. For full evaluation, upload the NASA bioscience CSV (608 entries) and adjust parsing to map columns: id,title,year,authors,abstract,intervention,outcome,context,sample_size,effect_pct,doi. Optional recommended installs: sentence-transformers, reportlab, unidecode, fpdf, faiss-cpu.</div>", unsafe_allow_html=True)
