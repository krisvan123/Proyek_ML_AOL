# ===================================================================
# CELL 4: BUAT APLIKASI STREAMLIT (app.py)
# ===================================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

# ── Konfigurasi Halaman ──────────────────────────────────────────
st.set_page_config(
    page_title="Prediksi PM2.5 | Air Quality",
    page_icon="🌍",
    layout="wide"
)

# ── Load Model Pipeline ──────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load('pipeline_pm25_final.pkl')

model = load_model()

# ── Mapping Wilayah WHO ─────────────────────────────────────────
# PENTING: value harus persis sama dengan data training!
WHO_REGION = {
    "Africa (AFRO)"                : "1_Afr",
    "Americas (AMRO)"               : "2_Amr",
    "South-East Asia (SEARO)"       : "3_Sear",
    "Europe (EURO)"                : "4_Eur",
    "Eastern Mediterranean (EMRO)" : "5_Emr",
    "Western Pacific (WPRO)"        : "6_Wpr",
    "Non-Member States"             : "7_NonMS",
}

# ── Sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Panduan Penggunaan")
    st.markdown("""
    1. Pilih **wilayah WHO** lokasi Anda
    2. Masukkan **koordinat** lokasi
    3. Isi **data demografis** area
    4. Opsional: masukkan data **PM10/NO₂**
    5. Tekan tombol **Prediksi**
    """)
    st.info("💡 Memasukkan data PM10 akan meningkatkan akurasi secara signifikan (korelasi 0.886 dengan PM2.5)")
    st.markdown("---")
    st.markdown("**📊 Standar WHO:**")
    st.markdown("""
    | Kategori | PM2.5 (µg/m³) |
    |----------|:---:|
    | 🟢 Sehat | ≤ 12.0 |
    | 🟡 Sedang | 12.1–35.4 |
    | 🟠 Tidak Sehat Sebagian | 35.5–55.4 |
    | 🔴 Berbahaya | > 55.4 |
    """)
    st.markdown("---")
    st.caption("COMP6577001 — Machine Learning\nModel: Random Forest Regressor\nDataset: WHO Global Air Quality 2010–2018")

# ── Header ───────────────────────────────────────────────────────
st.title("🌍 Prediksi Konsentrasi PM2.5")
st.markdown("Sistem prediksi kualitas udara berbasis **Random Forest** — dilatih dengan data WHO Global Air Quality.")
st.divider()

# ── Form Input ───────────────────────────────────────────────────
st.subheader("📝 Masukkan Data Lokasi & Lingkungan")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**🗺 Informasi Geografis**")
    who_region_label = st.selectbox(
        "Wilayah WHO",
        options=list(WHO_REGION.keys()),
        index=2,  # Default: South-East Asia (SEARO)
        help="Pilih wilayah administratif WHO"
    )
    who_region = WHO_REGION[who_region_label]

    latitude = st.number_input(
        "Latitude", min_value=-90.0, max_value=90.0,
        value=-6.2088, format="%.4f",
        help="Garis lintang (Jakarta: -6.2088)"
    )
    longitude = st.number_input(
        "Longitude", min_value=-180.0, max_value=180.0,
        value=106.8456, format="%.4f",
        help="Garis bujur (Jakarta: 106.8456)"
    )

with col2:
    st.markdown("**👥 Informasi Demografis & Temporal**")
    year = st.number_input("Tahun Analisis", min_value=2010, max_value=2030, value=2024)
    population = st.number_input(
        "Populasi Area (jiwa)",
        min_value=1_000, max_value=50_000_000,
        value=10_000_000, step=100_000, format="%d"
    )
    who_ms = st.radio(
        "Status Anggota WHO", options=[1, 0],
        format_func=lambda x: "✅ Anggota WHO" if x == 1 else "❌ Non-Anggota",
        horizontal=True
    )
    number_of_stations = st.number_input(
        "Jumlah Stasiun Pengukuran", min_value=1, max_value=200, value=3
    )

with col3:
    st.markdown("**🔬 Data Polutan _(opsional, meningkatkan akurasi)_**")

    use_pm10 = st.checkbox("Saya memiliki data PM10", value=False)
    if use_pm10:
        pm10 = st.number_input(
            "PM10 Concentration (µg/m³)",
            min_value=0.0, max_value=500.0, value=35.0, step=0.1
        )
        st.success("✅ PM10 akan digunakan — akurasi lebih tinggi")
    else:
        pm10 = np.nan
        st.info("ℹ️ PM10 tidak tersedia → nilai median training digunakan")

    use_no2 = st.checkbox("Saya memiliki data NO₂", value=False)
    if use_no2:
        no2 = st.number_input(
            "NO₂ Concentration (µg/m³)",
            min_value=0.0, max_value=300.0, value=20.0, step=0.1
        )
    else:
        no2 = np.nan

st.divider()

# ── Tombol Prediksi ───────────────────────────────────────────────
_, btn_col, _ = st.columns([2, 1, 2])
with btn_col:
    predict_btn = st.button("🔍 Prediksi PM2.5", use_container_width=True, type="primary")

# ── Hasil Prediksi ────────────────────────────────────────────────
if predict_btn:
    # Susun input agar kolom persis sama dengan saat training
    input_df = pd.DataFrame([{
        'year'               : int(year),
        'latitude'           : float(latitude),
        'longitude'          : float(longitude),
        'pm10_concentration' : float(pm10) if not np.isnan(float(pm10)) else np.nan,
        'no2_concentration'  : float(no2)  if not np.isnan(float(no2))  else np.nan,
        'number_of_stations' : int(number_of_stations),
        'who_ms'             : int(who_ms),
        'population'         : float(population),
        'who_region'         : who_region,   # Nilai asli: "3_Sear", "4_Eur", dst.
    }])

    # ✅ BENAR — gunakan variabel model yang sudah di-load
    hasil = float(model.predict(input_df)[0])
    hasil = max(0.0, hasil)   # PM2.5 tidak mungkin negatif

    # Tentukan kategori
    if hasil <= 12.0:
        kategori, emoji, color_fn = "Sehat (Good)",            "🟢", st.success
        pesan = f"Udara bersih! Konsentrasi {hasil:.2f} µg/m³ aman untuk semua aktivitas outdoor."
    elif hasil <= 35.4:
        kategori, emoji, color_fn = "Sedang (Moderate)",       "🟡", st.warning
        pesan = f"Kualitas sedang. Kelompok sensitif (lansia, asma, anak-anak) sebaiknya batasi aktivitas outdoor lama."
    elif hasil <= 55.4:
        kategori, emoji, color_fn = "Tidak Sehat Sebagian",    "🟠", st.error
        pesan = f"Tidak sehat untuk kelompok sensitif! Gunakan masker N95 di luar ruangan."
    else:
        kategori, emoji, color_fn = "Berbahaya (Hazardous)",   "🔴", st.error
        pesan = f"BERBAHAYA! Hindari semua aktivitas outdoor. Gunakan air purifier di dalam ruangan."

    # Tampilkan metrik
    st.subheader("📊 Hasil Prediksi")
    m1, m2, m3 = st.columns(3)
    m1.metric("Estimasi PM2.5",        f"{hasil:.2f} µg/m³")
    m2.metric("Kategori Kualitas Udara", f"{emoji} {kategori}")
    m3.metric("Persentase di atas batas sehat (12 µg/m³)",
              f"{max(0, hasil - 12.0):.2f} µg/m³")

    color_fn(f"**{emoji} {kategori}** — {pesan}")

    # Detail input
    with st.expander("🔍 Detail Input & Konfigurasi Model"):
        st.json({
            "Input": {
                "who_region": who_region,
                "year": int(year),
                "latitude": float(latitude),
                "longitude": float(longitude),
                "population": int(population),
                "pm10_concentration": float(pm10) if not np.isnan(float(pm10)) else "Imputasi median",
                "no2_concentration":  float(no2)  if not np.isnan(float(no2))  else "Imputasi median",
                "number_of_stations": int(number_of_stations),
                "who_ms": int(who_ms),
            },
            "Model": {
                "algorithm": "Random Forest Regressor",
                "n_estimators": 300,
                "preprocessing": "ColumnTransformer (MedianImputer + OneHotEncoder)",
            }
        })

st.divider()
st.caption("🔬 Dataset: WHO Global Air Quality 2010–2018 | COMP6577001 Machine Learning | Latency < 100ms")
