import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import LatentDirichletAllocation

st.set_page_config(
    page_title="Sentiment Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.stApp { background-color: #0b1120; }

section[data-testid="stSidebar"] {
    background-color: #0f1629;
    border-right: 0.5px solid rgba(255,255,255,0.06);
}

.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

h1, h2, h3 { color: #e2e8f0; font-weight: 500; }
p, label, li { color: #94a3b8; }

.hero {
    background: linear-gradient(135deg, #1e1b4b, #312e81);
    padding: 28px 32px;
    border-radius: 14px;
    margin-bottom: 24px;
}
.hero h1 { color: #e0e7ff; margin-bottom: 6px; font-size: 1.5rem; }
.hero p  { color: #a5b4fc; font-size: 14px; margin: 0; }

[data-testid="metric-container"] {
    background: #111827;
    border: 0.5px solid rgba(255,255,255,0.06);
    border-radius: 12px;
    padding: 18px 20px;
}
[data-testid="metric-container"] label {
    color: #64748b !important;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: .05em;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #f1f5f9;
}

.stButton > button {
    background: #4f46e5;
    color: white;
    border: none;
    border-radius: 10px;
    padding: 10px 20px;
    font-weight: 500;
    font-size: 14px;
}
.stButton > button:hover { background: #4338ca; }

.stTextArea textarea {
    background-color: #111827;
    color: #e2e8f0;
    border-radius: 10px;
    border: 0.5px solid rgba(255,255,255,0.08);
}

.stTextInput input {
    background-color: #111827;
    color: #e2e8f0;
    border-radius: 10px;
    border: 0.5px solid rgba(255,255,255,0.08);
}

[data-baseweb="select"] { background-color: #111827 !important; border-radius: 10px !important; }
[data-baseweb="select"] * { color: #e2e8f0 !important; }

[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }

.model-badge {
    margin-top: 2rem;
    padding: 12px 16px;
    background: rgba(99,102,241,0.08);
    border-radius: 10px;
    border: 0.5px solid rgba(99,102,241,0.2);
}
.model-badge p {
    font-size: 11px;
    color: #475569;
    margin-bottom: 4px;
    text-transform: uppercase;
    letter-spacing: .05em;
}
.model-badge span { font-size: 13px; color: #a5b4fc; }

.info-box {
    background: #111827;
    border: 0.5px solid rgba(255,255,255,0.06);
    border-left: 3px solid #4f46e5;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 20px;
}
.info-box p { margin: 0; font-size: 13px; color: #94a3b8; line-height: 1.6; }
</style>
""", unsafe_allow_html=True)


# ── Helpers layout ─────────────────────────────────────────
def base_layout(**extra):
    layout = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#94a3b8",
        margin=dict(t=28, b=24, l=10, r=10),
    )
    layout.update(extra)
    return layout


COLOR_MAP = {
    "Positif": "#4ade80",
    "Negatif": "#f87171",
    "Netral":  "#818cf8",
}
URUTAN = ["Positif", "Netral", "Negatif"]


# ── Load data & model ──────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("shopee_reviews_bersih.csv")
    df["at"] = pd.to_datetime(df["at"], errors="coerce")
    return df

@st.cache_resource
def load_models():
    model_nb = joblib.load("model/model_nb.pkl")
    model_lr = joblib.load("model/model_lr.pkl")
    tfidf    = joblib.load("model/tfidf.pkl")
    return model_nb, model_lr, tfidf

@st.cache_data
def get_test_predictions():
    df = load_data()
    model_nb, model_lr, tfidf = load_models()
    X_all = tfidf.transform(df["teks_bersih"].fillna(""))
    y_all = df["sentimen"]
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
    )
    y_pred_nb = model_nb.predict(X_test)
    y_pred_lr = model_lr.predict(X_test)
    return y_test, y_pred_nb, y_pred_lr

@st.cache_data
def get_lda_model(n_topics):
    df = load_data()
    _, _, tfidf = load_models()
    X_lda = tfidf.transform(df["teks_bersih"].fillna(""))
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42, max_iter=10)
    lda.fit(X_lda)
    doc_topics = lda.transform(X_lda)
    return lda, doc_topics

df         = load_data()
model_nb, model_lr, tfidf = load_models()


# ── Sidebar ────────────────────────────────────────────────
with st.sidebar:
    st.markdown("**Sentiment Dashboard**")
    st.caption("Shopee Review Analyzer")
    st.write("")

    menu = st.radio(
        "Menu",
        [
            "Overview",
            "Distribusi & Tren",
            "Word Cloud & Top Kata",
            "Pemodelan ML",
            "Confusion Matrix",
            "Topik LDA",
            "Eksplorasi Data",
        ],
        label_visibility="collapsed",
    )

    st.markdown("""
    <div class="model-badge">
        <p>Model aktif</p>
        <span>&#9679; Naive Bayes + Logistic Regression</span>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# 1. OVERVIEW
# ══════════════════════════════════════════════════════════
if menu == "Overview":

    st.markdown("""
    <div class="hero">
        <h1>Shopee Review Sentiment Analysis</h1>
        <p>Dashboard interaktif analisis sentimen ulasan Google Play Store · Text Mining + Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)

    total   = len(df)
    positif = len(df[df["sentimen"] == "Positif"])
    negatif = len(df[df["sentimen"] == "Negatif"])
    netral  = len(df[df["sentimen"] == "Netral"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Review", f"{total:,}")
    c2.metric("Positif",  f"{positif:,} ({positif/total*100:.1f}%)")
    c3.metric("Negatif",  f"{negatif:,} ({negatif/total*100:.1f}%)")
    c4.metric("Netral",   f"{netral:,} ({netral/total*100:.1f}%)")

    st.write("")
    col_left, col_right = st.columns([1, 1.4])

    with col_left:
        st.subheader("Distribusi Sentimen")
        fig_pie = go.Figure(go.Pie(
            labels=URUTAN,
            values=[positif, netral, negatif],
            hole=0.62,
            marker_colors=[COLOR_MAP[s] for s in URUTAN],
            textfont_size=13,
        ))
        fig_pie.update_layout(
            **base_layout(),
            showlegend=True,
            legend=dict(orientation="h", y=-0.1),
        )
        fig_pie.update_traces(hovertemplate="%{label}<br>%{value:,} ulasan (%{percent})")
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_right:
        st.subheader("Sebaran Rating")
        score_counts = df["score"].value_counts().sort_index().reset_index()
        score_counts.columns = ["Rating", "Jumlah"]
        fig_bar = px.bar(
            score_counts, x="Rating", y="Jumlah",
            color="Rating",
            color_continuous_scale=["#f87171", "#fb923c", "#facc15", "#4ade80", "#22c55e"],
            text="Jumlah",
        )
        fig_bar.update_traces(texttemplate="%{text:,}", textposition="outside")
        fig_bar.update_layout(
            **base_layout(
                xaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
                coloraxis_showscale=False,
            )
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("Ringkasan Statistik")
    cs1, cs2, cs3 = st.columns(3)
    cs1.metric("Rata-rata Rating",        f"{df['score'].mean():.2f} / 5.00")
    cs2.metric("Rata-rata Thumbs Up",     f"{df['thumbsUpCount'].mean():.1f}")
    cs3.metric("Rata-rata Panjang Review",f"{df['panjang_asli'].mean():.0f} kata")


# ══════════════════════════════════════════════════════════
# 2. DISTRIBUSI & TREN
# ══════════════════════════════════════════════════════════
elif menu == "Distribusi & Tren":

    st.title("Distribusi & Tren")

    st.subheader("Distribusi Panjang Review per Sentimen")
    fig_hist = go.Figure()
    for s in URUTAN:
        subset = df[df["sentimen"] == s]["panjang_asli"].clip(upper=100)
        fig_hist.add_trace(go.Histogram(
            x=subset, name=s,
            marker_color=COLOR_MAP[s],
            opacity=0.65, nbinsx=30,
        ))
    fig_hist.update_layout(
        **base_layout(
            barmode="overlay",
            xaxis=dict(title="Jumlah Kata", gridcolor="rgba(255,255,255,0.04)"),
            yaxis=dict(title="Frekuensi",   gridcolor="rgba(255,255,255,0.04)"),
            legend=dict(orientation="h", y=1.08),
        )
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    st.subheader("Perbandingan Panjang Review (Boxplot)")
    fig_box = go.Figure()
    for s in URUTAN:
        fig_box.add_trace(go.Box(
            y=df[df["sentimen"] == s]["panjang_asli"].clip(upper=150),
            name=s,
            marker_color=COLOR_MAP[s],
            boxmean=True,
        ))
    fig_box.update_layout(
        **base_layout(
            yaxis=dict(title="Jumlah Kata", gridcolor="rgba(255,255,255,0.04)"),
            showlegend=False,
        )
    )
    st.plotly_chart(fig_box, use_container_width=True)

    st.subheader("Tren Ulasan per Waktu")
    df_tren = df.dropna(subset=["at"]).copy()
    df_tren["tanggal"] = df_tren["at"].dt.date
    tren = df_tren.groupby(["tanggal", "sentimen"]).size().unstack(fill_value=0)
    for col in URUTAN:
        if col not in tren.columns:
            tren[col] = 0
    tren = tren[URUTAN].reset_index()

    fig_tren = go.Figure()
    for s in URUTAN:
        fig_tren.add_trace(go.Scatter(
            x=tren["tanggal"], y=tren[s],
            name=s, mode="lines+markers",
            line=dict(color=COLOR_MAP[s], width=2),
            marker=dict(size=4),
        ))
    fig_tren.update_layout(
        **base_layout(
            xaxis=dict(title="Tanggal", gridcolor="rgba(255,255,255,0.04)"),
            yaxis=dict(title="Jumlah Ulasan", gridcolor="rgba(255,255,255,0.04)"),
            legend=dict(orientation="h", y=1.08),
        )
    )
    st.plotly_chart(fig_tren, use_container_width=True)

    st.subheader("Statistik Deskriptif per Sentimen")
    stat = df.groupby("sentimen")[["panjang_asli", "panjang_bersih", "thumbsUpCount"]].agg(
        ["mean", "median", "std"]
    ).round(2)
    st.dataframe(stat, use_container_width=True)


# ══════════════════════════════════════════════════════════
# 3. WORD CLOUD & TOP KATA
# ══════════════════════════════════════════════════════════
elif menu == "Word Cloud & Top Kata":

    st.title("Word Cloud & Top Kata")

    pilihan = st.selectbox("Sentimen", URUTAN)
    cmap_map = {"Positif": "Greens", "Netral": "Oranges", "Negatif": "Reds"}

    subset_text = df[df["sentimen"] == pilihan]["teks_bersih"].dropna().astype(str)
    text = " ".join(subset_text)

    if not text.strip():
        st.warning("Tidak ada teks untuk sentimen ini.")
    else:
        st.subheader(f"Word Cloud — {pilihan}")
        wc = WordCloud(
            width=1400, height=560,
            background_color="#0b1120",
            colormap=cmap_map[pilihan],
            max_words=200,
            collocations=False,
            prefer_horizontal=0.8,
        ).generate(text)

        fig_wc, ax = plt.subplots(figsize=(16, 6))
        fig_wc.patch.set_facecolor("#0b1120")
        ax.set_facecolor("#0b1120")
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig_wc)
        plt.close(fig_wc)

        st.subheader(f"Top 20 Kata — {pilihan}")
        top_kata = Counter(text.split()).most_common(20)
        kata_df  = pd.DataFrame(top_kata, columns=["Kata", "Frekuensi"])

        fig_top = px.bar(
            kata_df.sort_values("Frekuensi"),
            x="Frekuensi", y="Kata",
            orientation="h",
            text="Frekuensi",
            color="Frekuensi",
            color_continuous_scale=cmap_map[pilihan],
        )
        fig_top.update_traces(textposition="outside")
        fig_top.update_layout(
            **base_layout(
                xaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
                coloraxis_showscale=False,
                height=520,
            )
        )
        st.plotly_chart(fig_top, use_container_width=True)


# ══════════════════════════════════════════════════════════
# 4. PEMODELAN ML
# ══════════════════════════════════════════════════════════
elif menu == "Pemodelan ML":

    st.title("Pemodelan ML")

    st.markdown("""
    <div class="info-box">
        <p>Model dilatih menggunakan TF-IDF (max 1.000 fitur) dengan split data 80:20 (train:test).
        Dua model dibandingkan: Naive Bayes dan Logistic Regression.</p>
    </div>
    """, unsafe_allow_html=True)

    y_test, y_pred_nb, y_pred_lr = get_test_predictions()

    def get_metrics(y_true, y_pred):
        return {
            "Accuracy":  round(accuracy_score(y_true, y_pred), 4),
            "Precision": round(precision_score(y_true, y_pred, average="weighted", zero_division=0), 4),
            "Recall":    round(recall_score(y_true, y_pred, average="weighted", zero_division=0), 4),
            "F1 Score":  round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4),
        }

    m_nb = get_metrics(y_test, y_pred_nb)
    m_lr = get_metrics(y_test, y_pred_lr)

    st.subheader("Perbandingan Performa Model")
    col_nb, col_lr = st.columns(2)

    with col_nb:
        st.caption("Naive Bayes")
        a, b, c, d = st.columns(4)
        a.metric("Accuracy",  m_nb["Accuracy"])
        b.metric("Precision", m_nb["Precision"])
        c.metric("Recall",    m_nb["Recall"])
        d.metric("F1 Score",  m_nb["F1 Score"])

    with col_lr:
        st.caption("Logistic Regression")
        a, b, c, d = st.columns(4)
        a.metric("Accuracy",  m_lr["Accuracy"])
        b.metric("Precision", m_lr["Precision"])
        c.metric("Recall",    m_lr["Recall"])
        d.metric("F1 Score",  m_lr["F1 Score"])

    st.write("")

    # Bar chart perbandingan — TANPA yaxis di CHART_LAYOUT, pakai base_layout
    labels = list(m_nb.keys())
    fig_cmp = go.Figure()
    fig_cmp.add_trace(go.Bar(
        name="Naive Bayes", x=labels, y=list(m_nb.values()),
        marker_color="#818cf8",
        text=[str(v) for v in m_nb.values()],
        textposition="outside",
    ))
    fig_cmp.add_trace(go.Bar(
        name="Logistic Regression", x=labels, y=list(m_lr.values()),
        marker_color="#4ade80",
        text=[str(v) for v in m_lr.values()],
        textposition="outside",
    ))
    fig_cmp.update_layout(
        **base_layout(
            barmode="group",
            xaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
            yaxis=dict(range=[0, 1.12], gridcolor="rgba(255,255,255,0.04)"),
            legend=dict(orientation="h", y=1.08),
        )
    )
    st.plotly_chart(fig_cmp, use_container_width=True)

    # Classification report
    st.subheader("Classification Report")
    tab_nb, tab_lr = st.tabs(["Naive Bayes", "Logistic Regression"])

    with tab_nb:
        rpt = classification_report(y_test, y_pred_nb, output_dict=True)
        st.dataframe(pd.DataFrame(rpt).T.round(4), use_container_width=True)

    with tab_lr:
        rpt = classification_report(y_test, y_pred_lr, output_dict=True)
        st.dataframe(pd.DataFrame(rpt).T.round(4), use_container_width=True)

    # Prediksi teks baru
    st.subheader("Coba Prediksi Teks Baru")
    user_input   = st.text_area(
        "Teks ulasan",
        placeholder="Contoh: Barangnya bagus, pengiriman cepat dan packagingnya aman.",
        height=120,
        label_visibility="collapsed",
    )
    model_pilih = st.selectbox("Pilih Model", ["Naive Bayes", "Logistic Regression", "Keduanya"])

    if st.button("Analisis Sentimen"):
        if not user_input.strip():
            st.warning("Silakan masukkan teks ulasan terlebih dahulu.")
        else:
            vec = tfidf.transform([user_input])

            def tampilkan_hasil(mdl, nama):
                hasil  = mdl.predict(vec)[0]
                proba  = mdl.predict_proba(vec)[0]
                classes = mdl.classes_
                if hasil == "Positif":
                    st.success(f"[{nama}]  Sentimen Positif")
                elif hasil == "Negatif":
                    st.error(f"[{nama}]  Sentimen Negatif")
                else:
                    st.warning(f"[{nama}]  Sentimen Netral")

                prob_df = pd.DataFrame({"Sentimen": classes, "Probabilitas": proba})
                prob_df = prob_df.sort_values("Probabilitas")
                fig_p = px.bar(
                    prob_df, x="Probabilitas", y="Sentimen", orientation="h",
                    color="Sentimen", color_discrete_map=COLOR_MAP,
                    text=prob_df["Probabilitas"].apply(lambda x: f"{x:.1%}"),
                    range_x=[0, 1],
                )
                fig_p.update_traces(textposition="outside")
                fig_p.update_layout(
                    **base_layout(
                        xaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
                        yaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
                        showlegend=False,
                        height=180,
                    )
                )
                st.plotly_chart(fig_p, use_container_width=True)

            if model_pilih == "Naive Bayes":
                tampilkan_hasil(model_nb, "Naive Bayes")
            elif model_pilih == "Logistic Regression":
                tampilkan_hasil(model_lr, "Logistic Regression")
            else:
                cr1, cr2 = st.columns(2)
                with cr1:
                    tampilkan_hasil(model_nb, "Naive Bayes")
                with cr2:
                    tampilkan_hasil(model_lr, "Logistic Regression")


# ══════════════════════════════════════════════════════════
# 5. CONFUSION MATRIX
# ══════════════════════════════════════════════════════════
elif menu == "Confusion Matrix":

    st.title("Confusion Matrix")

    y_test, y_pred_nb, y_pred_lr = get_test_predictions()
    classes = model_nb.classes_

    def plot_cm(y_true, y_pred, model_name, colorscale):
        cm = confusion_matrix(y_true, y_pred, labels=classes)
        fig = go.Figure(go.Heatmap(
            z=cm, x=list(classes), y=list(classes),
            text=cm, texttemplate="%{text}",
            colorscale=colorscale, showscale=True,
        ))
        fig.update_layout(
            **base_layout(
                title=dict(text=model_name, font=dict(color="#e2e8f0", size=14)),
                xaxis=dict(title="Predicted Label"),
                yaxis=dict(title="True Label"),
                height=380,
            )
        )
        return fig

    col_cm1, col_cm2 = st.columns(2)
    with col_cm1:
        st.plotly_chart(plot_cm(y_test, y_pred_nb, "Naive Bayes", "Blues"), use_container_width=True)
    with col_cm2:
        st.plotly_chart(plot_cm(y_test, y_pred_lr, "Logistic Regression", "Greens"), use_container_width=True)

    st.subheader("Ringkasan Prediksi")
    cm_nb = confusion_matrix(y_test, y_pred_nb, labels=classes)
    cm_lr = confusion_matrix(y_test, y_pred_lr, labels=classes)

    ci1, ci2 = st.columns(2)
    ci1.metric("Naive Bayes — Prediksi Benar",
               f"{int(np.diag(cm_nb).sum()):,} / {int(cm_nb.sum()):,}")
    ci2.metric("Logistic Regression — Prediksi Benar",
               f"{int(np.diag(cm_lr).sum()):,} / {int(cm_lr.sum()):,}")


# ══════════════════════════════════════════════════════════
# 6. TOPIK LDA
# ══════════════════════════════════════════════════════════
elif menu == "Topik LDA":

    st.title("Topik LDA")

    st.markdown("""
    <div class="info-box">
        <p>LDA (Latent Dirichlet Allocation) digunakan untuk menemukan topik utama yang sering muncul
        dalam ulasan pengguna secara otomatis.</p>
    </div>
    """, unsafe_allow_html=True)

    col_sl1, col_sl2 = st.columns(2)
    with col_sl1:
        n_topics = st.slider("Jumlah Topik", min_value=2, max_value=6, value=3)
    with col_sl2:
        n_words = st.slider("Jumlah Kata per Topik", min_value=5, max_value=20, value=10)

    # Tombol untuk trigger training — bukan otomatis dari slider
    if st.button("Latih Model LDA"):
        st.session_state["lda_trained"]  = True
        st.session_state["lda_topics"]   = n_topics
        st.session_state["lda_words"]    = n_words

    run_topics = st.session_state.get("lda_topics", n_topics)
    run_words  = st.session_state.get("lda_words", n_words)

    if not st.session_state.get("lda_trained", False):
        st.info("Atur jumlah topik dan kata, lalu tekan **Latih Model LDA** untuk melihat hasilnya.")
    else:
        with st.spinner("Melatih model LDA..."):
            lda, doc_topics = get_lda_model(run_topics)

        feature_names = tfidf.get_feature_names_out()
        palette = ["#818cf8", "#4ade80", "#f87171", "#fb923c", "#38bdf8", "#e879f9"]

        n_cols = min(run_topics, 3)
        cols   = st.columns(n_cols)

        for idx, topic in enumerate(lda.components_):
            col = cols[idx % n_cols]
            top_idx    = topic.argsort()[-run_words:][::-1]
            top_words  = [feature_names[i] for i in top_idx]
            top_scores = [topic[i] for i in top_idx]

            fig_lda = px.bar(
                x=top_scores[::-1], y=top_words[::-1],
                orientation="h",
                labels={"x": "Bobot", "y": "Kata"},
                color_discrete_sequence=[palette[idx % len(palette)]],
            )
            fig_lda.update_layout(
                **base_layout(
                    title=dict(text=f"Topik {idx+1}", font=dict(color="#e2e8f0", size=13)),
                    xaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
                    yaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
                    showlegend=False,
                    height=360,
                )
            )
            col.plotly_chart(fig_lda, use_container_width=True)

        st.subheader("Distribusi Topik per Sentimen")
        df_topics = df[["sentimen"]].copy()
        for i in range(run_topics):
            df_topics[f"Topik {i+1}"] = doc_topics[:, i]

        topic_by_sent = df_topics.groupby("sentimen")[
            [f"Topik {i+1}" for i in range(run_topics)]
        ].mean().round(4)

        fig_heat = px.imshow(
            topic_by_sent,
            color_continuous_scale="Blues",
            text_auto=True, aspect="auto",
        )
        fig_heat.update_layout(**base_layout(height=280))
        st.plotly_chart(fig_heat, use_container_width=True)


# ══════════════════════════════════════════════════════════
# 7. EKSPLORASI DATA
# ══════════════════════════════════════════════════════════
elif menu == "Eksplorasi Data":

    st.title("Eksplorasi Data")

    col_s, col_sent, col_dl = st.columns([2, 1, 1])
    with col_s:
        search = st.text_input(
            "Cari teks",
            placeholder="Ketik kata kunci...",
            label_visibility="collapsed",
        )
    with col_sent:
        filter_sent = st.selectbox(
            "Filter Sentimen",
            ["Semua"] + URUTAN,
            label_visibility="collapsed",
        )
    with col_dl:
        st.download_button(
            "Download CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="shopee_reviews_bersih.csv",
            mime="text/csv",
            use_container_width=True,
        )

    display_df = df.copy()
    if filter_sent != "Semua":
        display_df = display_df[display_df["sentimen"] == filter_sent]
    if search:
        mask = (
            display_df["teks_bersih"].str.contains(search, case=False, na=False) |
            display_df["content_asli"].str.contains(search, case=False, na=False)
        )
        display_df = display_df[mask]

    st.caption(f"{len(display_df):,} dari {len(df):,} data")

    st.dataframe(
        display_df[[
            "content_asli", "teks_bersih", "score", "sentimen",
            "panjang_asli", "panjang_bersih", "thumbsUpCount", "at"
        ]],
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Statistik Kolom Numerik")
    st.dataframe(
        df[["score", "panjang_asli", "panjang_bersih", "thumbsUpCount"]].describe().round(2),
        use_container_width=True,
    )