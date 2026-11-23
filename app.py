import os
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from train_model import train_and_save, FEATURES, DATA_PATH, MODEL_PATH

st.set_page_config(page_title="SDN DDoS Detection", page_icon="🛡️", layout="wide")
st.title("🛡️ SDN DDoS Attack Detection")
st.caption("Lightweight neural-network prototype • Synthetic demonstration data only")

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(DATA_PATH):
        train_and_save()
    return joblib.load(MODEL_PATH)

model_bundle = load_model()
model = model_bundle["model"]
scaler = model_bundle["scaler"]
df = pd.read_csv(DATA_PATH)

st.sidebar.header("Navigation")
page = st.sidebar.radio("Choose a page", ["Dashboard", "Test a traffic flow", "Dataset"])
st.sidebar.info("This prototype uses generated sample data. It does not monitor or block real network traffic.")

if page == "Dashboard":
    st.subheader("Detection overview")
    total = len(df)
    attack_count = int(df["label"].sum())
    normal_count = total - attack_count
    c1, c2, c3 = st.columns(3)
    c1.metric("Sample flows", f"{total:,}")
    c2.metric("Normal flows", f"{normal_count:,}")
    c3.metric("DDoS-like flows", f"{attack_count:,}")
    left, right = st.columns(2)
    with left:
        st.markdown("**Traffic class distribution**")
        fig, ax = plt.subplots()
        df["class_name"] = df["label"].map({0:"Normal", 1:"DDoS-like"})
        df["class_name"].value_counts().reindex(["Normal", "DDoS-like"]).plot(kind="bar", ax=ax)
        ax.set_ylabel("Number of flows")
        ax.set_xlabel("")
        st.pyplot(fig)
        plt.close(fig)
    with right:
        st.markdown("**Packets per second distribution**")
        fig, ax = plt.subplots()
        for label, name in [(0, "Normal"), (1, "DDoS-like")]:
            ax.hist(df.loc[df.label == label, "packets_per_second"], bins=25, alpha=.6, label=name)
        ax.set_xlabel("Packets per second")
        ax.set_ylabel("Count")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)
    st.markdown("**Model evaluation on held-out synthetic test data**")
    metrics = model_bundle["metrics"]
    cols = st.columns(4)
    for col, key in zip(cols, ["accuracy", "precision", "recall", "f1"]):
        col.metric(key.title(), f"{metrics[key]*100:.2f}%")
    st.caption("Scores are from a reproducible synthetic dataset split, not a real-world benchmark.")

elif page == "Test a traffic flow":
    st.subheader("Classify an example flow")
    st.write("Enter SDN flow-statistics features. Values are illustrative.")
    with st.form("flow_form"):
        a, b, c = st.columns(3)
        duration = a.number_input("Flow duration (seconds)", min_value=0.01, value=2.0, step=0.1)
        packets = b.number_input("Packets per second", min_value=0.0, value=120.0, step=10.0)
        bytes_sec = c.number_input("Bytes per second", min_value=0.0, value=80000.0, step=1000.0)
        d, e, f = st.columns(3)
        packet_size = d.number_input("Average packet size (bytes)", min_value=1.0, value=650.0, step=10.0)
        flow_count = e.number_input("Active flows at switch", min_value=1.0, value=15.0, step=1.0)
        syn_ratio = f.slider("SYN packet ratio", 0.0, 1.0, 0.2, 0.01)
        submitted = st.form_submit_button("Analyze flow")
    if submitted:
        row = pd.DataFrame([[duration, packets, bytes_sec, packet_size, flow_count, syn_ratio]], columns=FEATURES)
        prediction = int(model.predict(scaler.transform(row))[0])
        probability = float(model.predict_proba(scaler.transform(row))[0][1])
        if prediction == 1:
            st.error("DDoS-like traffic pattern detected")
        else:
            st.success("Traffic classified as normal")
        st.metric("Estimated DDoS-like probability", f"{probability*100:.1f}%")
        st.caption("This is a model estimate for the demo feature values, not a security verdict.")

else:
    st.subheader("Synthetic SDN flow dataset")
    st.write("The CSV contains generated flow-statistics examples for demonstration and classroom testing.")
    st.dataframe(df.drop(columns=["class_name"], errors="ignore"), use_container_width=True)
    st.download_button("Download CSV dataset", df.to_csv(index=False), "sdn_demo_dataset.csv", "text/csv")
    st.markdown("**Feature descriptions**")
    st.dataframe(pd.DataFrame({
        "Feature": FEATURES,
        "Meaning": [
            "Duration of the observed flow in seconds",
            "Packets observed per second",
            "Bytes observed per second",
            "Average packet size in bytes",
            "Approximate active flow count at a switch",
            "Fraction of packets represented as SYN in the demo"
        ]
    }), use_container_width=True)
