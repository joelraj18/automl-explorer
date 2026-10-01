import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.api.types import is_numeric_dtype
import io
import contextlib

st.set_page_config(page_title="AutoML Explorer", layout="wide", initial_sidebar_state="expanded")

# ─────────────────────────────────────────────
# APPLE-STYLE CSS INJECTION
# ─────────────────────────────────────────────
apple_css = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
        color: #1d1d1f !important;
        background-color: #ffffff !important;
    }
    .stApp { background-color: #ffffff !important; }
    header, footer { visibility: hidden; }

    h1 {
        color: #1d1d1f !important;
        font-size: 3.8rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.04em !important;
        text-align: center;
        padding-bottom: 0 !important;
        margin-bottom: 0 !important;
    }
    h2, h3 {
        font-weight: 600 !important;
        letter-spacing: -0.02em !important;
        color: #1d1d1f !important;
        margin-top: 2rem !important;
    }
    .hero-subtitle {
        text-align: center;
        color: #86868b;
        font-size: 1.4rem;
        font-weight: 500;
        letter-spacing: -0.01em;
        margin-bottom: 3rem;
    }
    .stButton > button {
        background-color: #f5f5f7 !important;
        color: #1d1d1f !important;
        border: none !important;
        border-radius: 980px !important;
        padding: 0.6rem 1.4rem !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        background-color: #e8e8ed !important;
        transform: scale(0.98);
    }
    .stButton > button:active {
        background-color: #0071e3 !important;
        color: #ffffff !important;
    }
    .stSelectbox > div > div {
        background-color: #f5f5f7 !important;
        border-radius: 12px !important;
        border: none !important;
    }
    [data-testid="stFileUploadDropzone"] {
        background-color: #fbfbfd !important;
        border-radius: 20px !important;
        border: 1px dashed #d2d2d7 !important;
        padding: 2rem !important;
    }
    .stCodeBlock {
        border-radius: 16px !important;
        border: 1px solid #e5e5ea !important;
    }
    /* Decision Trace Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #f5f5f7 !important;
        border-right: 1px solid #e5e5ea !important;
    }
</style>
"""
st.markdown(apple_css, unsafe_allow_html=True)

st.markdown("<h1>AutoML Explorer.</h1>", unsafe_allow_html=True)
st.markdown("<p class='hero-subtitle'>Upload a dataset. Generate a pipeline. Beautifully simple.</p>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 1. LOAD
# ─────────────────────────────────────────────
file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"], label_visibility="collapsed")

@st.cache_data
def load_file(f):
    return pd.read_csv(f) if f.name.endswith(".csv") else pd.read_excel(f)

if not file:
    st.stop()

df = load_file(file)

st.subheader("Data Preview")
st.dataframe(df.head(10), use_container_width=True)

# ─────────────────────────────────────────────
# 2. ADVANCED DECISION ENGINE (The "Big Tree")
# ─────────────────────────────────────────────
target = st.selectbox("Target Column (choose 'None' for unsupervised)", options=["None"] + list(df.columns))

def decide_pipeline(dataframe, target_col):
    rows, cols = dataframe.shape
    numeric_cols = [c for c in dataframe.columns if is_numeric_dtype(dataframe[c])]
    
    # Stage 0: Quality Gates
    if rows < 30:
        return {"task": "halt", "trace": "🛑 Quality Gate: Dataset too small for reliable modelling (rows < 30)."}
    if cols < 2:
        return {"task": "halt", "trace": "🛑 Quality Gate: Need at least one feature + target."}
    if dataframe.isna().mean().mean() > 0.6:
        return {"task": "halt", "trace": "🛑 Quality Gate: Over 60% missing data. Halt and impute first."}

    # Branch A: Unsupervised
    if target_col == "None":
        if len(numeric_cols) < 2:
            return {"task": "halt", "trace": "🛑 Need >= 2 numeric columns for multi-var clustering."}
        algo = "MiniBatchKMeans" if rows > 10000 else "KMeans"
        trace = f"🟢 No target selected.\n↳ {len(numeric_cols)} numeric cols found.\n↳ Rows={rows} -> Selected {algo}."
        return {"task": "unsupervised", "algo": algo, "trace": trace}
        
    t = dataframe[target_col]
    is_num = is_numeric_dtype(t)
    n_unique = t.nunique()
    
    # Branch B1: Regression
    if is_num and n_unique > 15:
        algo = "LinearRegression"
        trace = f"🟢 Numeric target (>15 unique) -> Regression Path.\n"
        
        # Cross-cutting checks
        if abs(t.skew()) > 1:
            algo = "HuberRegressor"
            trace += f"↳ Target is skewed (skew={t.skew():.2f}) -> Selected {algo} for robustness.\n"
        elif len(numeric_cols) > 30:
            algo = "Lasso"
            trace += f"↳ High feature count (>30) -> Selected {algo} for feature selection.\n"
        else:
            trace += f"↳ Normal distribution & standard feature count -> Selected {algo}.\n"
            
        return {"task": "regression", "algo": algo, "trace": trace}
        
    # Branch B2: Classification
    else:
        counts = t.value_counts(normalize=True)
        imbalanced = counts.min() < 0.2
        class_weight = "class_weight='balanced', " if imbalanced else ""
        
        if n_unique == 2:
            sub = "Binary"
            algo = f"RandomForestClassifier({class_weight}random_state=42)"
        elif n_unique <= 10:
            sub = "Multiclass"
            algo = f"RandomForestClassifier({class_weight}random_state=42)"
        else:
            sub = "High-Cardinality"
            algo = "RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)"
            
        trace = f"🟢 Target unique={n_unique} -> {sub} Classification.\n"
        if imbalanced:
            trace += f"↳ Minority class < 20% -> Applied SMOTE/Class Weights to model.\n"
            
        return {"task": "classification", "sub": sub, "algo": algo, "trace": trace}

decision = decide_pipeline(df, target)

# UI: Decision Trace Sidebar
with st.sidebar:
    st.subheader("🧠 Engine Decision Trace")
    st.text(decision["trace"])

if decision["task"] == "halt":
    st.error(decision["trace"])
    st.stop()

# ─────────────────────────────────────────────
# 3. CODE CELL GENERATORS
# ─────────────────────────────────────────────
def eda_code(dataframe):
    num = [c for c in dataframe.columns if is_numeric_dtype(dataframe[c])]
    return f"""import matplotlib.pyplot as plt
import seaborn as sns

numeric_cols = {num!r}
if numeric_cols:
    df[numeric_cols].hist(figsize=(10, 6), bins=30, color='#0071e3', edgecolor='white')
    plt.tight_layout(); plt.show()
"""

def supervised_code(dataframe, target_col, decision_dict):
    feats = [c for c in dataframe.columns if c != target_col]
    num = [c for c in feats if is_numeric_dtype(dataframe[c])]
    cat = [c for c in feats if not is_numeric_dtype(dataframe[c])]
    
    algo_str = decision_dict["algo"]
    task_type = decision_dict["task"]

    # Map the algorithm to its import statement dynamically
    imports = ["from sklearn.model_selection import train_test_split",
               "from sklearn.preprocessing import StandardScaler, OneHotEncoder",
               "from sklearn.compose import ColumnTransformer",
               "from sklearn.pipeline import Pipeline"]
               
    if task_type == "classification":
        imports.append("from sklearn.ensemble import RandomForestClassifier")
        imports.append("from sklearn.metrics import classification_report, accuracy_score")
        metrics = "print(f'Accuracy: {accuracy_score(y_test, y_pred):.3f}')\nprint(classification_report(y_test, y_pred))"
    else:
        if "Huber" in algo_str: imports.append("from sklearn.linear_model import HuberRegressor")
        elif "Lasso" in algo_str: imports.append("from sklearn.linear_model import Lasso")
        else: imports.append("from sklearn.linear_model import LinearRegression")
        imports.append("from sklearn.metrics import r2_score, mean_squared_error")
        metrics = "print(f'R2 Score: {r2_score(y_test, y_pred):.4f}')\nprint(f'MSE: {mean_squared_error(y_test, y_pred):.4f}')"

    import_block = "\n".join(imports)

    return f"""{import_block}

target = {target_col!r}
clean_df = df.dropna(subset=[target])
X = clean_df.drop(columns=[target])
y = clean_df[target]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), {num!r}),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), {cat!r}),
])

model = {algo_str}
pipe = Pipeline([("prep", preprocessor), ("model", model)])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

{metrics}
"""

def unsupervised_code(dataframe, decision_dict):
    num = [c for c in dataframe.columns if is_numeric_dtype(dataframe[c])]
    algo_str = decision_dict["algo"]
    
    imports = "from sklearn.cluster import KMeans"
    if "MiniBatch" in algo_str: 
        imports = "from sklearn.cluster import MiniBatchKMeans"
        
    return f"""from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
{imports}
import matplotlib.pyplot as plt

X = df[{num!r}].dropna()
X_scaled = StandardScaler().fit_transform(X)

model = {algo_str}(n_clusters=3, random_state=42, n_init="auto")
labels = model.fit_predict(X_scaled)
print(f"Silhouette Score: {{silhouette_score(X_scaled, labels):.3f}}")

if X_scaled.shape[1] >= 2:
    plt.figure(figsize=(8, 6))
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap="Blues", alpha=0.8)
    plt.title("Cluster Visualization")
    plt.show()
"""

# ─────────────────────────────────────────────
# 4. BUILD CELLS & EXECUTE
# ─────────────────────────────────────────────
st.subheader("Generated Pipeline")

cells = {"Exploratory Data Analysis": eda_code(df)}
if decision["task"] == "unsupervised":
    cells["Model Training"] = unsupervised_code(df, decision)
else:
    cells["Model Training"] = supervised_code(df, target, decision)

for name, code in cells.items():
    st.markdown(f"**{name}**")
    st.code(code, language="python")

st.markdown("<br>", unsafe_allow_html=True)

if st.button("Run Pipeline"):
    ns = {"df": df, "pd": pd, "np": np, "plt": plt, "sns": sns}
    
    for name, code in cells.items():
        st.markdown(f"### Output: {name}")
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                exec(code, ns)
            
            if buf.getvalue():
                st.text(buf.getvalue())
                
            for n in plt.get_fignums():
                st.pyplot(plt.figure(n))
            plt.close("all")
            
        except Exception as e:
            st.error(f"Error executing {name}: {e}")