import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from io import StringIO
from streamlit_lottie import st_lottie
import requests

# --- Lottie Loader ---
def load_lottie_url(url):
    r = requests.get(url)
    if r.status_code == 200:
        return r.json()
    return None

lottie_url = "https://assets10.lottiefiles.com/packages/lf20_qp1q7mct.json"
lottie_anim = load_lottie_url(lottie_url)

# --- Streamlit Config & Style ---
st.set_page_config(page_title="Customer Churn Prediction", page_icon="📊", layout="wide")
st.markdown("""
    <style>
    .main { background: #f8f8fc; }
    .stButton>button {
        background-color: #1d3557;
        color: white !important;
        border-radius: 8px;
        font-size: 16px;
        padding: 6px 18px;
    }
    </style>
    """, unsafe_allow_html=True)

with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/000000/business-network.png", width=120)
    if lottie_anim:
        st_lottie(lottie_anim, speed=1, width=120, height=120)
    st.subheader("😊 Welcome!")
    st.markdown("This is an interactive tool to predict customer churn.")

st.title("📊 Customer Churn Prediction App")
st.markdown("**Predict customer churn and explore data-driven insights using the Telco Customer Churn dataset.**")

# --- Data Load and Preprocessing ---
@st.cache_data
def load_data():
    df = pd.read_csv("Telco-Customer-Churn.csv.csv")
    df.replace(" ", np.nan, inplace=True)
    df.drop("customerID", axis=1, inplace=True)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # Encode categorical columns
    le = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = le.fit_transform(df[col])
    return df

df = load_data()
X = df.drop("Churn", axis=1)
y = df["Churn"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, stratify=y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

if "history" not in st.session_state:
    st.session_state["history"] = []

# --- Tabs Layout ---
tab1, tab2, tab3 = st.tabs(["🔮 Predict Churn", "📈 Insights Dashboard", "ℹ️ About"])

# --- PREDICT TAB ---
with tab1:

    st.header("Real-Time Churn Prediction")

    with st.expander("Set Customer Features", expanded=True):
        gender = st.selectbox("Gender", ("Male", "Female"))
        senior_citizen = st.selectbox("Senior Citizen", ("Yes", "No"))
        partner = st.selectbox("Partner", ("Yes", "No"))
        dependents = st.selectbox("Dependents", ("Yes", "No"))
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        monthly_charges = st.slider("Monthly Charges ($)", 10, 150, 70)
        total_charges = st.slider("Total Charges ($)", 0, 8000, 2000)
        contract = st.selectbox("Contract Type", ("Month-to-month", "One year", "Two year"))
        internet_service = st.selectbox("Internet Service", ("DSL", "Fiber optic", "No"))
        payment_method = st.selectbox("Payment Method", ("Electronic check", "Mailed check", "Bank transfer", "Credit card"))

    # Real-time prediction logic
    input_data = {
        "gender": 1 if gender == "Male" else 0,
        "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
        "Partner": 1 if partner == "Yes" else 0,
        "Dependents": 1 if dependents == "Yes" else 0,
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "Contract": 0 if contract == "Month-to-month" else (1 if contract == "One year" else 2),
        "InternetService": 0 if internet_service == "DSL" else (1 if internet_service == "Fiber optic" else 2),
        "PaymentMethod": 0 if payment_method == "Electronic check" else (1 if payment_method == "Mailed check" else (2 if payment_method == "Bank transfer" else 3))
    }
    user_df = pd.DataFrame([input_data])
    for col in X.columns:
        if col not in user_df.columns:
            user_df[col] = 0
    user_df = user_df[X.columns]

    scaled_input = scaler.transform(user_df)
    prediction = model.predict(scaled_input)
    prob = model.predict_proba(scaled_input)[0][1]

    pred_status = "Churn" if prediction[0]==1 else "Not Churn"

    # Save history
    st.session_state["history"].append({
        **input_data,
        "Predicted Churn": pred_status,
        "Churn Probability (%)": round(prob*100,2)
    })

    st.subheader("Result")

    # Gauge Chart for Probability
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = round(prob*100,2),
        number = {'suffix': '%'},
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Churn Probability"},
        gauge = {
            'axis': {'range': [0,100]},
            'bar': {'color': "#ef476f" if prediction[0]==1 else "#06d6a0"},
            'steps' : [
                {'range': [0, 60], 'color': "#06d6a0"},
                {'range': [60, 80], 'color': "#ffd166"},
                {'range': [80, 100], 'color': "#ef476f"}
            ]
        }
    ))
    fig_gauge.update_layout(paper_bgcolor='#f8f8fc', plot_bgcolor='#f8f8fc')
    st.plotly_chart(fig_gauge, use_container_width=True)

    if prediction[0] == 1:
        st.error(f"⚠️ This customer is **likely to churn!** ({prob*100:.2f}%)")
        st.progress(int(prob*100))
    else:
        st.success(f"✅ This customer is **not likely to churn.** ({100-prob*100:.2f}%)")
        st.progress(int(100-prob*100))

    display_df = user_df.copy()
    display_df["Predicted Churn"] = pred_status
    display_df["Churn Probability (%)"] = round(prob*100,2)
    st.markdown("#### Prediction Details")
    st.dataframe(display_df.T, use_container_width=True)

    # History table
    with st.expander("Recent Predictions"):
        hist_df = pd.DataFrame(st.session_state["history"]).tail(5)
        st.dataframe(hist_df.T, use_container_width=True)

    # Download button
    csv_buffer = StringIO()
    display_df.to_csv(csv_buffer, index=False)
    st.download_button(
        label="⬇️ Download Prediction Report (CSV)",
        data=csv_buffer.getvalue().encode("utf-8"),
        file_name="churn_prediction_report.csv",
        mime="text/csv"
    )

# --- INSIGHTS TAB ---
with tab2:
    st.header("Data Insights Dashboard")

    # Pie chart for churn
    churn_counts = df["Churn"].value_counts()
    fig_pie = px.pie(
        churn_counts, 
        values=churn_counts.values, 
        names=["No Churn", "Churn"], 
        color_discrete_sequence=["#06d6a0","#ef476f"]
    )
    fig_pie.update_traces(textinfo='percent+label')
    fig_pie.update_layout(paper_bgcolor='#f8f8fc', plot_bgcolor='#f8f8fc')
    st.subheader("Churn Distribution")
    st.plotly_chart(fig_pie, use_container_width=True)

    # Stacked bar: Contract vs Churn
    st.subheader("Contract Type vs Churn Rate")
    ct = pd.crosstab(df["Contract"], df["Churn"], normalize="index")
    fig_bar = px.bar(
        ct, 
        barmode='stack', 
        labels={"value":"Proportion","Contract":"Contract Type"}, 
        color_discrete_map={0:"#06d6a0",1:"#ef476f"}
    )
    fig_bar.update_layout(paper_bgcolor='#f8f8fc', plot_bgcolor='#f8f8fc')
    st.plotly_chart(fig_bar, use_container_width=True)

    # Feature Importance
    st.subheader("Feature Importances")
    importances = pd.Series(model.feature_importances_, index=X.columns)
    top_features = importances.nlargest(10)
    fig_imp = px.bar(
        top_features.sort_values(), 
        orientation="h", 
        title="Top 10 Influencing Features", 
        labels={"value":"Importance","index":"Feature"},
        color=top_features.sort_values(),
        color_continuous_scale=["#06d6a0", "#ffd166", "#ef476f"]
    )
    fig_imp.update_layout(paper_bgcolor='#f8f8fc', plot_bgcolor='#f8f8fc')
    st.plotly_chart(fig_imp, use_container_width=True)

    if st.button("Show Data Summary"):
        st.dataframe(df.describe())

# --- ABOUT TAB ---
with tab3:
    st.header("About This App")
    st.markdown("""
    - **Developed by Aadil Hussain**
    - Powered by **Streamlit** & **Scikit-learn**
    - Dataset: Telco Customer Churn
    - Enhanced for real-time insights, modern visuals, and practical business utility.  
    """)

    st.info("For feedback or support, contact: aadilh118@gmail.com")

st.caption("© 2025 | Customer Churn Prediction App")
