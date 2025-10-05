# ARGO MAP XR – The Intelligent Dashboard for Human Space Exploration  

---

##  Overview  
**ARGO MAP XR** is an advanced AI-powered visualization and analysis dashboard built to explore NASA’s bioscience data.  
It enables scientists, researchers, and mission planners to study how humans, plants, and microbes adapt to microgravity environments.  
By integrating artificial intelligence, machine learning, and immersive 3D simulations, ARGO MAP XR turns complex space research into accessible insights that empower future Moon and Mars missions.  

This tool summarizes NASA’s bioscience publications, identifies research trends, highlights knowledge gaps, and provides interactive visual exploration—all in one unified platform.  

---

## Key Features  

### 1. AI Research Analyzer  
- Processes and analyzes NASA bioscience datasets automatically.  
- Extracts key themes and findings using transformer-based AI models.  
- Natural language semantic search (e.g., “Show me studies about radiation effects on plants”).  

### 2. Knowledge Graph Explorer  
- Interactive PyVis-based graph visualization of NASA bioscience papers.  
- Clickable nodes display abstracts and metadata.  
- Color-coded relationships between species, experiment types, and outcomes.  

### 3. 3D Space Simulation Dashboard  
- Real-time solar system model rendered in 3D using Plotly.  
- Visualizes orbital mechanics, gravitational effects, and mission trajectories.  
- Includes ISS, Moon, and Mars with physics-based simulations.  

### 4. Predictive Analytics  
- Machine learning models forecast:  
  - Space radiation exposure  
  - Crew health risk index  
  - Mission success probabilities  

### 5. Visualization Suite  
- Dozens of graph types including:  
  - 2D/3D scatter plots  
  - Heatmaps  
  - Line and radar charts  
  - Animated data dashboards  

### 6. Human Digital Twin  
- Interactive 3D anatomical model (via Sketchfab).  
- Simulates zero-gravity physiological effects such as muscle loss and oxygen variation.  

### 7. AI-Powered Report Generator  
- Automatically generates multilingual, research-grade PDF reports.  
- Fixed encoding for Unicode (UTF-8) to avoid formatting errors.  
- Adds time-stamped report titles using `datetime.datetime.now(datetime.UTC)`.  

### 8. NASA API Integration  
- Pulls live data from:  
  - NASA OSDR (Open Science Data Repository)  
  - Space Life Sciences Library  
  - NASA Task Book  

### 9. Space Weather & Real-Time Analytics  
- Displays live solar activity, CME index, and radiation flux.  
- Interactive charts showing dynamic updates.  

### 10. Immersive UI and 3D Landing Page  
- 3D animated galaxy background integrated with Three.js and Plotly.  
- Floating astronaut with NASA reflection animation.  
- Parallax scrolling effects for Moon and Mars sections.  
- VR-ready visuals and cinematic transitions.  

---

## How It Works  
1. User uploads NASA bioscience datasets (CSV).  
2. Data is automatically cleaned, tokenized, and analyzed by an AI summarizer.  
3. Extracted knowledge is represented visually via graphs and knowledge networks.  
4. Simulations and predictive models help interpret findings.  
5. Users can generate analytical reports or explore interactive 3D dashboards.  

---

## Technology Stack  
**Frontend:** Streamlit, Plotly, PyVis, Three.js  
**Backend:** Python  
**AI / ML:** Transformers, TensorFlow, Scikit-learn, SpaCy, BERT  
**Visualization:** Matplotlib, Plotly, WordCloud  
**APIs:** NASA OSDR, Space Life Sciences Library, NASA Task Book  
**Deployment:** Streamlit Cloud or Docker  

---

## Setup and Installation  

### Step 1: Install Python  
Ensure Python 3.10 or newer is installed.  
```bash
python --version
````

### Step 2: Clone the Repository

```bash
git clone https://github.com/your-username/ARGO-MAP-XR.git
cd ARGO-MAP-XR
```

### Step 3: Create and Activate Virtual Environment

```bash
python -m venv venv
# Activate:
venv\Scripts\activate   # On Windows  
source venv/bin/activate  # On macOS/Linux
```

### Step 4: Install Dependencies

If `requirements.txt` is present:

```bash
pip install -r requirements.txt
```

Otherwise, install manually:

```bash
pip install streamlit pandas numpy matplotlib plotly pyvis transformers torch tensorflow scikit-learn wordcloud requests beautifulsoup4 reportlab pypandoc spacy opencv-python seaborn
```

### Step 5: Run the Application

```bash
streamlit run main.py
```

Your app will open automatically at:
**[http://localhost:8501](http://localhost:8501)**

---

## 🧬 How to Use

1. Launch the Streamlit app.
2. Upload NASA bioscience dataset in the AI Analyzer section.
3. Explore insights using the knowledge graph and visual dashboards.
4. Run 3D simulations and analyze radiation, orbit, and biology metrics.
5. Generate detailed PDF summaries or live analytics visualizations.

---

## Future Enhancements

* Integration of FAISS for ultra-fast semantic search.
* Voice-based assistant for querying publications.
* AI narration and storytelling for visual summaries.
* Real-time VR environment for multi-user exploration.
* Automated discovery of knowledge gaps using GPT-based reasoning.

---

## License

This project is released under the **MIT License**.

---

## Author

Developed, designed, and implemented by **Muhammad Saad**.

---

## Project Summary

**ARGO MAP XR** represents a leap in how space bioscience data is visualized and understood.
It merges AI, data science, and interactive simulation to provide a complete ecosystem for exploring life sciences in space.
By transforming NASA’s vast bioscience archive into an intelligent and intuitive interface, it empowers research and planning for sustainable human exploration beyond Earth.

```
```
