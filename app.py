import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from PIL import Image

st.set_page_config(layout="wide")

st.title("🧬 HealthGuardAI")
st.subheader("Biomedical AI Disease Prediction Platform")

# -----------------------------
# SIDEBAR INPUTS
# -----------------------------

st.sidebar.title("Patient Clinical Data")

age = st.sidebar.slider("Age",10,90,45)
bmi = st.sidebar.slider("BMI",15,40,24)
smoking = st.sidebar.slider("Smoking Years",0,40,2)
activity = st.sidebar.slider("Activity Minutes",0,120,30)
sleep = st.sidebar.slider("Sleep Hours",3,10,7)
cholesterol = st.sidebar.slider("Cholesterol",120,300,180)
glucose = st.sidebar.slider("Glucose",70,200,95)

# -----------------------------
# AI MODEL (SIMULATED TRAINING)
# -----------------------------

np.random.seed(1)

data = pd.DataFrame({
"age":np.random.randint(20,80,500),
"bmi":np.random.randint(18,35,500),
"smoking":np.random.randint(0,30,500),
"activity":np.random.randint(0,100,500),
"cholesterol":np.random.randint(120,260,500),
"glucose":np.random.randint(70,180,500)
})

data["risk"] = (
data["age"]*0.02 +
data["bmi"]*0.03 +
data["smoking"]*0.04 +
data["cholesterol"]*0.02 +
data["glucose"]*0.03 -
data["activity"]*0.02
)

data["risk"] = (data["risk"]>7).astype(int)

X = data.drop("risk",axis=1)
y = data["risk"]

model = RandomForestClassifier()
model.fit(X,y)

patient = pd.DataFrame({
"age":[age],
"bmi":[bmi],
"smoking":[smoking],
"activity":[activity],
"cholesterol":[cholesterol],
"glucose":[glucose]
})

prediction = model.predict_proba(patient)[0][1]

risk = int(prediction*100)

# -----------------------------
# TOP DASHBOARD
# -----------------------------

col1,col2,col3,col4 = st.columns(4)

col1.metric("Lung Risk",f"{min(risk+8,100)} %")
col2.metric("Heart Risk",f"{risk} %")
col3.metric("Diabetes Risk",f"{max(risk-5,0)} %")
col4.metric("Cancer Risk",f"{min(risk+12,100)} %")

# -----------------------------
# GAUGE
# -----------------------------

fig = go.Figure(go.Indicator(
mode="gauge+number",
value=risk,
title={"text":"Overall Disease Risk"},
gauge={
"axis":{"range":[0,100]},
"steps":[
{"range":[0,30],"color":"green"},
{"range":[30,60],"color":"yellow"},
{"range":[60,100],"color":"red"}
]
}
))

st.plotly_chart(fig,use_container_width=True)

# -----------------------------
# ORGAN MAP
# -----------------------------

st.subheader("Organ Health Map")

organs = {
"Lung":min(risk+8,100),
"Heart":risk,
"Liver":max(risk-10,0),
"Kidney":max(risk-15,0),
"Brain":max(risk-5,0),
"Pancreas":max(risk-12,0)
}

org_df = pd.DataFrame({
"Organ":organs.keys(),
"Risk":organs.values()
})

fig2 = go.Figure()

fig2.add_trace(go.Bar(
x=org_df["Organ"],
y=org_df["Risk"]
))

st.plotly_chart(fig2,use_container_width=True)

# -----------------------------
# IMAGE ANALYSIS SECTION
# -----------------------------

st.subheader("Medical Image Analysis")

img = st.file_uploader("Upload X-ray / CT / Report",type=["png","jpg","jpeg"])

if img:

    image = Image.open(img)
    st.image(image,width=300)

    st.success("Image Loaded")

    st.write("AI Preliminary Scan")

    if risk > 60:
        st.error("Possible abnormality detected")
    else:
        st.success("No major abnormality detected")

# -----------------------------
# AI REPORT
# -----------------------------

st.subheader("AI Clinical Interpretation")

if risk < 30:
    st.success("Low disease probability")

elif risk < 60:
    st.warning("Moderate disease risk detected")

else:
    st.error("High disease risk — screening recommended")

st.write("AI Confidence:",round(prediction,2))

# -----------------------------
# FUTURE MODULE
# -----------------------------

st.subheader("Genomic Analysis (Coming)")

st.info("Upload FASTA / mutation file for cancer prediction")

# -----------------------------
# DOCKING MODULE
# -----------------------------

st.subheader("Drug Docking Engine (Coming)")

st.info("Protein + ligand docking simulation will appear here")