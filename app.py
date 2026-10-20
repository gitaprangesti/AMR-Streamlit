from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ============================================================
# KONFIGURASI
# ============================================================
st.set_page_config(
    page_title="AMR Surveillance | Data Explorer",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent

FILES = {
    "model": BASE_DIR / "best_amr_model.joblib",
    "predictions": BASE_DIR / "amr_predictions.csv",
    "cv": BASE_DIR / "cv_table.csv",
    "importance": BASE_DIR / "feature_importance.csv",
    "residual": BASE_DIR / "country_residual.csv",
    "holdout": BASE_DIR / "holdout_metrics.csv",
    "features": BASE_DIR / "model_features.json",
}

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.6rem; padding-bottom: 2.5rem;}
    [data-testid="stMetric"] {
        background: #F5F8FC;
        border: 1px solid #E5EAF2;
        padding: 16px 18px;
        border-radius: 12px;
    }
    .insight {
        background: #F1F7F6;
        border-left: 4px solid #148477;
        padding: 13px 16px;
        border-radius: 7px;
        margin: 8px 0 16px 0;
    }
    .muted {color: #64748B; font-size: 0.9rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# PEMUATAN DATA
# ============================================================
@st.cache_resource
def load_model(path, modified):
    return joblib.load(path)


@st.cache_data
def load_csv(path, modified, **kwargs):
    return pd.read_csv(path, **kwargs)


@st.cache_data
def load_json(path, modified):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def file_version(path):
    return path.stat().st_mtime if path.exists() else None


def missing_files():
    return [name for name, path in FILES.items() if not path.exists()]


def clean_numeric(series):
    return pd.to_numeric(series, errors="coerce")


missing = missing_files()
if missing:
    st.error("Beberapa file yang dibutuhkan belum ditemukan.")
    st.write("Letakkan file berikut dalam folder yang sama dengan `app.py`:")
    for key in missing:
        st.code(FILES[key].name)
    st.info("Pastikan nama file sama persis, termasuk ekstensi `.csv`, `.joblib`, dan `.json`.")
    st.stop()

try:
    model = load_model(str(FILES["model"]), file_version(FILES["model"]))
    predictions = load_csv(
        str(FILES["predictions"]), file_version(FILES["predictions"])
    )
    cv_table = load_csv(str(FILES["cv"]), file_version(FILES["cv"]))
    importance = load_csv(
        str(FILES["importance"]), file_version(FILES["importance"])
    )
    country_residual = load_csv(
        str(FILES["residual"]), file_version(FILES["residual"])
    )
    holdout = load_csv(
        str(FILES["holdout"]), file_version(FILES["holdout"]), index_col=0
    )
    FEATURES = load_json(str(FILES["features"]), file_version(FILES["features"]))
except Exception as exc:
    st.error("Data atau model gagal dimuat.")
    st.exception(exc)
    st.stop()

# ============================================================
# VALIDASI KOLOM DAN PERSIAPAN DATA
# ============================================================
required_prediction_columns = {
    "country", "year", "resistance_rate", "predicted_resistance_rate"
}
missing_prediction_columns = required_prediction_columns - set(predictions.columns)
if missing_prediction_columns:
    st.error(
        "Kolom wajib tidak ditemukan pada `amr_predictions.csv`: "
        + ", ".join(sorted(missing_prediction_columns))
    )
    st.write("Kolom yang tersedia:", list(predictions.columns))
    st.stop()

predictions = predictions.copy()
predictions["year"] = clean_numeric(predictions["year"])
predictions["resistance_rate"] = clean_numeric(predictions["resistance_rate"])
predictions["predicted_resistance_rate"] = clean_numeric(
    predictions["predicted_resistance_rate"]
)
predictions = predictions.dropna(
    subset=["country", "year", "resistance_rate", "predicted_resistance_rate"]
)
predictions["year"] = predictions["year"].astype(int)
predictions["selisih"] = (
    predictions["resistance_rate"] - predictions["predicted_resistance_rate"]
)
predictions["besar_kesalahan"] = predictions["selisih"].abs()

if predictions.empty:
    st.error("Tidak ada baris data yang valid pada `amr_predictions.csv`.")
    st.stop()

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.title("🦠 AMR Surveillance")
st.sidebar.caption("Dashboard eksplorasi data dan hasil pemodelan")
page = st.sidebar.radio(
    "Pilih halaman",
    [
        "Ringkasan",
        "Tren Resistensi",
        "Negara yang Perlu Ditinjau",
        "Evaluasi Model",
        "Faktor Model",
        "Data dan Simulasi",
    ],
)
st.sidebar.divider()
st.sidebar.caption("AMR = Antimicrobial Resistance")
st.sidebar.caption("Hasil model merupakan estimasi, bukan diagnosis.")

# ============================================================
# UTILITAS TAMPILAN
# ============================================================
def show_plot(fig, height=420):
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=20, r=20, t=65, b=25),
        legend_title_text="",
        font=dict(family="Arial", size=13),
    )
    st.plotly_chart(fig, width="stretch")


def metric_value(value, decimals=2):
    if pd.isna(value):
        return "Tidak tersedia"
    return f"{value:,.{decimals}f}"


def find_metric_column(frame, candidates):
    for candidate in candidates:
        for column in frame.columns:
            if str(column).strip().lower() == candidate.lower():
                return column
    return None


# ============================================================
# HALAMAN 1 — RINGKASAN
# ============================================================
if page == "Ringkasan":
    st.title("Memahami Resistensi Antimikroba")
    st.write(
        "Ringkasan data resistensi antimikroba dan hasil prediksi model. "
        "Gunakan halaman lain untuk menelusuri tren, perbedaan prediksi, "
        "dan evaluasi model secara lebih rinci."
    )

    years = sorted(predictions["year"].unique())
    countries_count = predictions["country"].nunique()
    average_actual = predictions["resistance_rate"].mean()
    average_error = predictions["besar_kesalahan"].mean()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Negara dalam data", f"{countries_count:,}")
    col2.metric("Periode pengamatan", f"{min(years)}–{max(years)}")
    col3.metric("Rata-rata resistensi aktual", f"{average_actual:.2f}%")
    col4.metric("Rata-rata selisih absolut", f"{average_error:.2f} poin")

    st.caption(
        "Angka dihitung dari baris yang tersedia pada file prediksi. "
        "Rata-rata antarbaris tidak selalu sama dengan rata-rata populasi negara."
    )

    st.subheader("Apa yang terlihat dari data?")
    yearly = (
        predictions.groupby("year", as_index=False)
        .agg(
            aktual=("resistance_rate", "mean"),
            prediksi=("predicted_resistance_rate", "mean"),
            jumlah_data=("resistance_rate", "size"),
        )
        .sort_values("year")
    )

    if len(yearly) >= 2:
        first, last = yearly.iloc[0], yearly.iloc[-1]
        change = last["aktual"] - first["aktual"]
        direction = "meningkat" if change > 0 else "menurun" if change < 0 else "relatif tetap"
        st.markdown(
            f"""
            <div class="insight">
            <b>Perubahan rata-rata pada periode yang tersedia</b><br>
            Rata-rata nilai aktual {direction} sebesar <b>{abs(change):.2f} poin persentase</b>,
            dari {first['aktual']:.2f}% pada {int(first['year'])}
            menjadi {last['aktual']:.2f}% pada {int(last['year'])}.
            Ini menggambarkan perubahan dalam data yang tersedia, bukan bukti penyebab perubahan.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.info("Data hanya mencakup satu tahun, sehingga perubahan dari tahun ke tahun belum dapat dihitung.")

    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("Rata-rata resistensi per tahun")
        yearly_long = yearly.melt(
            id_vars="year",
            value_vars=["aktual", "prediksi"],
            var_name="jenis",
            value_name="resistance_rate",
        )
        yearly_long["jenis"] = yearly_long["jenis"].map(
            {"aktual": "Data aktual", "prediksi": "Prediksi model"}
        )
        fig = px.line(
            yearly_long,
            x="year",
            y="resistance_rate",
            color="jenis",
            markers=True,
            labels={
                "year": "Tahun",
                "resistance_rate": "Rata-rata resistensi (%)",
                "jenis": "Jenis data",
            },
        )
        show_plot(fig, 380)
    with col_right:
        st.subheader("Sebaran nilai aktual")
        fig = px.histogram(
            predictions,
            x="resistance_rate",
            nbins=25,
            labels={"resistance_rate": "Resistensi aktual (%)", "count": "Jumlah data"},
        )
        fig.update_layout(yaxis_title="Jumlah data")
        show_plot(fig, 380)

    st.subheader("Cara membaca dashboard")
    st.markdown(
        """
        - **Data aktual** adalah nilai yang tercatat pada dataset.
        - **Prediksi model** adalah perkiraan yang dihasilkan model machine learning.
        - **Selisih positif** berarti nilai aktual lebih tinggi daripada prediksi.
        - **Selisih negatif** berarti nilai aktual lebih rendah daripada prediksi.
        - Pola dalam data tidak dengan sendirinya membuktikan hubungan sebab-akibat.
        """
    )

# ============================================================
# HALAMAN 2 — TREN
# ============================================================
elif page == "Tren Resistensi":
    st.title("Tren Resistensi")
    st.write(
        "Telusuri perubahan nilai resistensi dalam data yang tersedia. "
        "Gunakan filter untuk mempersempit negara dan periode pengamatan."
    )

    countries = sorted(predictions["country"].astype(str).unique())
    selected_countries = st.multiselect(
        "Pilih negara (kosongkan untuk semua negara)",
        options=countries,
        default=[],
    )
    year_min, year_max = int(predictions["year"].min()), int(predictions["year"].max())
    selected_years = st.slider(
        "Rentang tahun",
        min_value=year_min,
        max_value=year_max,
        value=(year_min, year_max),
    )

    trend_data = predictions[
        predictions["year"].between(selected_years[0], selected_years[1])
    ].copy()
    if selected_countries:
        trend_data = trend_data[trend_data["country"].astype(str).isin(selected_countries)]

    if trend_data.empty:
        st.warning("Tidak ada data untuk filter yang dipilih.")
    else:
        trend = (
            trend_data.groupby("year", as_index=False)
            .agg(
                aktual=("resistance_rate", "mean"),
                prediksi=("predicted_resistance_rate", "mean"),
            )
            .sort_values("year")
        )
        st.subheader("Perubahan rata-rata per tahun")
        trend_long = trend.melt(
            id_vars="year",
            value_vars=["aktual", "prediksi"],
            var_name="jenis",
            value_name="nilai",
        )
        trend_long["jenis"] = trend_long["jenis"].map(
            {"aktual": "Data aktual", "prediksi": "Prediksi model"}
        )
        fig = px.line(
            trend_long,
            x="year",
            y="nilai",
            color="jenis",
            markers=True,
            labels={"year": "Tahun", "nilai": "Resistensi (%)", "jenis": "Jenis data"},
        )
        show_plot(fig)

        st.subheader("Perbandingan antarnegara")
        country_summary = (
            trend_data.groupby("country", as_index=False)
            .agg(
                aktual=("resistance_rate", "mean"),
                prediksi=("predicted_resistance_rate", "mean"),
                jumlah_pengamatan=("year", "size"),
            )
            .sort_values("aktual", ascending=False)
        )
        top_n = st.slider(
            "Jumlah negara yang ditampilkan",
            min_value=5,
            max_value=min(30, max(5, len(country_summary))),
            value=min(10, max(5, len(country_summary))),
        )
        chart_data = country_summary.head(top_n).melt(
            id_vars="country",
            value_vars=["aktual", "prediksi"],
            var_name="jenis",
            value_name="nilai",
        )
        chart_data["jenis"] = chart_data["jenis"].map(
            {"aktual": "Data aktual", "prediksi": "Prediksi model"}
        )
        fig = px.bar(
            chart_data,
            x="nilai",
            y="country",
            color="jenis",
            barmode="group",
            orientation="h",
            labels={"nilai": "Rata-rata resistensi (%)", "country": "Negara", "jenis": "Jenis data"},
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        show_plot(fig, max(420, top_n * 32))

        with st.expander("Lihat tabel ringkasan"):
            st.dataframe(country_summary, width="stretch", hide_index=True)

# ============================================================
# HALAMAN 3 — NEGARA YANG PERLU DITINJAU
# ============================================================
elif page == "Negara yang Perlu Ditinjau":
    st.title("Negara yang Perlu Ditinjau")
    st.write(
        "Halaman ini menyoroti lokasi dengan perbedaan besar antara nilai aktual "
        "dan hasil prediksi. Daftar ini membantu menentukan data yang perlu diperiksa "
        "lebih lanjut, bukan menetapkan peringkat risiko kesehatan secara definitif."
    )

    country_summary = (
        predictions.groupby("country", as_index=False)
        .agg(
            aktual=("resistance_rate", "mean"),
            prediksi=("predicted_resistance_rate", "mean"),
            selisih=("selisih", "mean"),
            besar_kesalahan=("besar_kesalahan", "mean"),
            jumlah_pengamatan=("year", "size"),
        )
    )
    country_summary["interpretasi"] = np.select(
        [
            country_summary["selisih"] > 0,
            country_summary["selisih"] < 0,
        ],
        [
            "Aktual rata-rata lebih tinggi daripada prediksi",
            "Aktual rata-rata lebih rendah daripada prediksi",
        ],
        default="Aktual rata-rata sama dengan prediksi",
    )

    ranking_mode = st.radio(
        "Urutkan berdasarkan",
        [
            "Selisih positif terbesar",
            "Perbedaan absolut terbesar",
        ],
        horizontal=True,
    )
    if ranking_mode == "Selisih positif terbesar":
        ranked = country_summary.sort_values("selisih", ascending=False).head(10)
        chart_title = "Negara dengan aktual paling tinggi dibandingkan prediksi"
        chart_x = "selisih"
        axis_title = "Selisih rata-rata (poin persentase)"
    else:
        ranked = country_summary.sort_values("besar_kesalahan", ascending=False).head(10)
        chart_title = "Negara dengan perbedaan aktual-prediksi terbesar"
        chart_x = "besar_kesalahan"
        axis_title = "Rata-rata selisih absolut (poin persentase)"

    st.subheader(chart_title)
    fig = px.bar(
        ranked.sort_values(chart_x),
        x=chart_x,
        y="country",
        orientation="h",
        labels={chart_x: axis_title, "country": "Negara"},
        hover_data=["aktual", "prediksi", "jumlah_pengamatan"],
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    show_plot(fig, max(420, len(ranked) * 42))

    st.markdown(
        """
        <div class="insight">
        <b>Interpretasi</b><br>
        Selisih positif menunjukkan nilai aktual rata-rata lebih tinggi daripada prediksi.
        Selisih negatif menunjukkan nilai aktual rata-rata lebih rendah daripada prediksi.
        Perbedaan ini dapat muncul karena keterbatasan model, kualitas data, atau faktor lain
        yang tidak tercakup dalam model.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.subheader("Tabel negara")
    st.dataframe(
        ranked.sort_values(chart_x, ascending=False),
        width="stretch",
        hide_index=True,
        column_config={
            "aktual": st.column_config.NumberColumn("Rata-rata aktual (%)", format="%.2f"),
            "prediksi": st.column_config.NumberColumn("Rata-rata prediksi (%)", format="%.2f"),
            "selisih": st.column_config.NumberColumn("Selisih (poin)", format="%.2f"),
            "besar_kesalahan": st.column_config.NumberColumn("Selisih absolut (poin)", format="%.2f"),
        },
    )

# ============================================================
# HALAMAN 4 — EVALUASI MODEL
# ============================================================
elif page == "Evaluasi Model":
    st.title("Seberapa Andal Prediksinya?")
    st.write(
        "Evaluasi membantu melihat seberapa dekat prediksi model dengan nilai aktual. "
        "Nilai evaluasi yang lebih baik tidak menjamin model akan akurat untuk setiap negara atau tahun."
    )

    st.subheader("Hasil evaluasi hold-out")
    holdout_display = holdout.copy()
    st.dataframe(holdout_display, width="stretch")

    st.markdown(
        """
        - **R²:** menunjukkan seberapa besar variasi target yang dapat dijelaskan model dalam data evaluasi. Nilai yang lebih tinggi biasanya lebih baik.
        - **MAE:** rata-rata besar kesalahan absolut. Jika target dinyatakan dalam persen, interpretasi satuannya perlu mengikuti definisi target.
        - **RMSE:** ukuran kesalahan yang memberi penalti lebih besar pada kesalahan besar. Nilai lebih rendah biasanya lebih baik.
        """
    )

    actual_col = "resistance_rate"
    pred_col = "predicted_resistance_rate"
    eval_data = predictions[[actual_col, pred_col, "country", "year"]].dropna()

    st.subheader("Nilai aktual dibandingkan prediksi")
    fig = px.scatter(
        eval_data,
        x=actual_col,
        y=pred_col,
        hover_data=["country", "year"],
        labels={
            actual_col: "Resistensi aktual (%)",
            pred_col: "Resistensi prediksi (%)",
        },
    )
    low = min(eval_data[actual_col].min(), eval_data[pred_col].min())
    high = max(eval_data[actual_col].max(), eval_data[pred_col].max())
    fig.add_trace(
        go.Scatter(
            x=[low, high],
            y=[low, high],
            mode="lines",
            name="Prediksi sama dengan aktual",
            line=dict(dash="dash"),
        )
    )
    show_plot(fig)

    st.caption(
        "Titik yang dekat dengan garis putus-putus menunjukkan prediksi yang dekat dengan nilai aktual. "
        "Grafik ini memakai baris pada file prediksi; pastikan file tersebut memang disiapkan untuk evaluasi yang dimaksud."
    )

    if not cv_table.empty:
        st.subheader("Perbandingan skema evaluasi")
        st.dataframe(cv_table, width="stretch", hide_index=True)
        if {"model", "R2"}.issubset(cv_table.columns):
            fig = px.bar(
                cv_table,
                x="model",
                y="R2",
                color="skema" if "skema" in cv_table.columns else None,
                barmode="group",
                labels={"model": "Model", "R2": "Nilai R²", "skema": "Skema evaluasi"},
            )
            show_plot(fig, 380)

# ============================================================
# HALAMAN 5 — FAKTOR MODEL
# ============================================================
elif page == "Faktor Model":
    st.title("Variabel yang Digunakan Model")
    st.write(
        "Permutation importance menunjukkan seberapa besar kinerja model berubah "
        "ketika suatu variabel diacak. Ini membantu memahami ketergantungan model pada "
        "variabel, tetapi tidak membuktikan bahwa variabel tersebut menyebabkan resistensi."
    )

    if {"fitur", "penurunan_R2"}.issubset(importance.columns):
        imp = importance.copy()
        imp["penurunan_R2"] = clean_numeric(imp["penurunan_R2"])
        imp = imp.dropna(subset=["penurunan_R2"]).sort_values("penurunan_R2", ascending=True)
        fig = px.bar(
            imp,
            x="penurunan_R2",
            y="fitur",
            orientation="h",
            labels={"penurunan_R2": "Penurunan R² saat variabel diacak", "fitur": "Variabel"},
        )
        show_plot(fig, max(420, len(imp) * 35))
        st.subheader("Cara membacanya")
        st.write(
            "Batang yang lebih panjang menunjukkan bahwa pengacakan variabel tersebut "
            "lebih menurunkan skor R² pada pengujian yang digunakan. Nilai kecil atau negatif "
            "tidak berarti variabel itu tidak penting dalam semua konteks."
        )
        st.dataframe(
            imp.sort_values("penurunan_R2", ascending=False),
            width="stretch",
            hide_index=True,
        )
    else:
        st.warning(
            "Kolom `fitur` dan/atau `penurunan_R2` tidak ditemukan pada feature_importance.csv. "
            f"Kolom tersedia: {list(importance.columns)}"
        )

# ============================================================
# HALAMAN 6 — DATA DAN SIMULASI
# ============================================================
elif page == "Data dan Simulasi":
    st.title("Eksplorasi Data dan Simulasi Prediksi")
    st.write(
        "Telusuri catatan yang tersedia atau buat simulasi menggunakan model. "
        "Hasil simulasi bukan pengganti pengukuran laboratorium atau data surveilans."
    )

    tab_data, tab_sim = st.tabs(["Jelajahi data", "Coba simulasi"])

    with tab_data:
        country_options = ["Semua"] + sorted(predictions["country"].astype(str).unique())
        chosen_country = st.selectbox("Negara", country_options)
        chosen_years = st.multiselect(
            "Tahun",
            sorted(predictions["year"].unique().tolist()),
            default=sorted(predictions["year"].unique().tolist()),
        )
        table_data = predictions[predictions["year"].isin(chosen_years)].copy()
        if chosen_country != "Semua":
            table_data = table_data[table_data["country"].astype(str) == chosen_country]

        st.caption(f"{len(table_data):,} baris sesuai filter.")
        st.dataframe(table_data, width="stretch", hide_index=True)
        st.download_button(
            "Unduh data yang ditampilkan (CSV)",
            data=table_data.to_csv(index=False).encode("utf-8"),
            file_name="amr_data_terfilter.csv",
            mime="text/csv",
        )

    with tab_sim:
        st.warning(
            "Gunakan rentang nilai yang masuk akal dan sesuai skala data pelatihan. "
            "Jika model dilatih dengan transformasi tertentu, input harus mengikuti format yang sama."
        )
        st.caption("Masukkan nilai variabel untuk melihat estimasi dari model.")
        defaults = {
            "antibiotic_use_ddd": 10.0,
            "access_percent": 60.0,
            "watch_percent": 30.0,
            "reserve_percent": 10.0,
            "log_population": 15.0,
            "infection_incidence": 1000.0,
            "mortality_rate": 10.0,
            "year": int(predictions["year"].max()),
        }
        input_labels = {
            "antibiotic_use_ddd": "Penggunaan antibiotik (DDD)",
            "access_percent": "Kelompok Access (%)",
            "watch_percent": "Kelompok Watch (%)",
            "reserve_percent": "Kelompok Reserve (%)",
            "log_population": "Log populasi",
            "infection_incidence": "Insidensi infeksi",
            "mortality_rate": "Angka mortalitas",
            "year": "Tahun",
        }
        inputs = {}
        left, right = st.columns(2)
        for index, feature in enumerate(FEATURES):
            container = left if index % 2 == 0 else right
            with container:
                label = input_labels.get(feature, feature.replace("_", " ").title())
                if feature == "year":
                    inputs[feature] = st.number_input(
                        label,
                        min_value=2000,
                        max_value=2100,
                        value=int(defaults.get(feature, 2025)),
                        step=1,
                    )
                else:
                    inputs[feature] = st.number_input(
                        label,
                        value=float(defaults.get(feature, 0.0)),
                    )

        if st.button("Hitung estimasi", type="primary"):
            try:
                input_frame = pd.DataFrame(
                    [{feature: inputs[feature] for feature in FEATURES}],
                    columns=FEATURES,
                )
                prediction_value = float(model.predict(input_frame)[0])
                st.metric("Estimasi tingkat resistensi", f"{prediction_value:.2f}%")
                st.info(
                    "Ini merupakan estimasi berdasarkan model dan nilai yang dimasukkan. "
                    "Hasil tidak membuktikan hubungan sebab-akibat dan bukan diagnosis klinis."
                )
            except Exception as exc:
                st.error(
                    "Prediksi tidak dapat dihitung. Periksa kesesuaian nama, urutan, "
                    "dan format fitur dengan data saat model dilatih."
                )
                st.exception(exc)

st.sidebar.divider()
st.sidebar.caption("AMR Surveillance · Machine Learning")
st.sidebar.caption("Gunakan hasil sebagai dukungan analisis, bukan keputusan klinis.")