import streamlit as st
import pandas as pd
import joblib

# ⭐⭐⭐ 一定要放在所有 st.xxx() 之前 ⭐⭐⭐
st.set_page_config(
    page_title="Hogwarts Sorting - Voting Ensemble",
    page_icon="🧙‍♂️",
    layout="centered",
)

# 1️⃣ 只加载 pkl，不训练
@st.cache_resource
def load_model():
    model = joblib.load("voting_model.pkl")
    return model

model = load_model()

# 2️⃣ 页面标题 & 说明
st.title("🏰 Hogwarts Sorting Prediction")
st.markdown("""
Welcome to the **Hogwarts Sorting Hat** web app!  
This app uses a pre-trained **Voting Ensemble model (Random Forest + Gradient Boosting)**  
to predict which house a student belongs to based on their characteristics.
""")

# 3️⃣ 输入区
st.markdown("### Please enter the student's characteristics:")

col1, col2 = st.columns(2)

with col1:
    blood_status = st.selectbox("Blood Status", ["Half-blood", "Muggle-born", "Pure-blood"])
    bravery = st.slider("Bravery", 0, 10, 5)
    intelligence = st.slider("Intelligence", 0, 10, 5)
    loyalty = st.slider("Loyalty", 0, 10, 5)
    ambition = st.slider("Ambition", 0, 10, 5)

with col2:
    dark_arts = st.slider("Dark Arts Knowledge", 0, 10, 5)
    quidditch = st.slider("Quidditch Skills", 0, 10, 5)
    dueling = st.slider("Dueling Skills", 0, 10, 5)
    creativity = st.slider("Creativity", 0, 10, 5)

input_df = pd.DataFrame([{
    "Blood Status": blood_status,
    "Bravery": bravery,
    "Intelligence": intelligence,
    "Loyalty": loyalty,
    "Ambition": ambition,
    "Dark Arts Knowledge": dark_arts,
    "Quidditch Skills": quidditch,
    "Dueling Skills": dueling,
    "Creativity": creativity,
}])

st.markdown("**🧾 Input Summary:**")
st.dataframe(input_df)

# 4️⃣ 预测按钮
if st.button("🔮 Predict House"):
    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]

    st.markdown("---")
    st.subheader(f"🏆 The Sorting Hat chooses: **{pred}** 🪄")

    proba_df = pd.DataFrame({
        "House": model.classes_,
        "Probability": proba
    }).sort_values("Probability", ascending=False)

    st.markdown("### Class Probabilities:")
    st.dataframe(proba_df.reset_index(drop=True))

# 5️⃣ 页脚
st.markdown("---")
st.caption("Developed by Hogwarts Data Science Team 🧙‍♀️ | Pre-trained Voting Ensemble (RF + GB)")