import streamlit as st
import pickle
import matplotlib.pyplot as plt

# Load model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))

# Page config
st.set_page_config(
    page_title="NexaBank AI Support",
    page_icon="🏦",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main { background-color: #f0f4f8; }
    .result-box {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1a3c5e;
        margin: 10px 0;
    }
    .login-box {
        background-color: #ffffff;
        padding: 40px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        max-width: 450px;
        margin: auto;
    }
    .bank-header {
        background: linear-gradient(135deg, #1a3c5e, #4a6fa5);
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin-bottom: 30px;
    }
    </style>
""", unsafe_allow_html=True)

# Session state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user_name' not in st.session_state:
    st.session_state.user_name = ""
if 'history' not in st.session_state:
    st.session_state.history = []
if 'query_input' not in st.session_state:
    st.session_state.query_input = ""

# ─── LOGIN PAGE ───────────────────────────────────────────
if not st.session_state.logged_in:

    st.markdown("""
    <div class="bank-header">
        <h1>🏦 NexaBank</h1>
        <p style="font-size:1.1em; opacity:0.9;">
            AI-Powered Customer Support System
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_left, col_center, col_right = st.columns([1, 1.5, 1])
    with col_center:
        st.markdown("""
        <div style="text-align:center;">
            <h3 style="color:#1a3c5e;">👤 Customer Login</h3>
            <p style="color:#888; font-size:0.9em;">
                Please enter your details to continue
            </p>
        </div>
        """, unsafe_allow_html=True)

        name = st.text_input("Full Name", placeholder="e.g. John Smith")
        bank_id = st.text_input("Bank ID", placeholder="e.g. NXB123456")

        st.markdown("")
        login_btn = st.button("🔐 Login to Support Portal",
                               use_container_width=True, type="primary")

        if login_btn:
            if name and bank_id:
                if len(bank_id) >= 6:
                    st.session_state.logged_in = True
                    st.session_state.user_name = name
                    st.rerun()
                else:
                    st.error("Bank ID must be at least 6 characters!")
            else:
                st.warning("Please fill in both fields!")

        st.markdown("---")
        st.markdown("""
        <p style="text-align:center; color:#888; font-size:0.8em;">
            🔒 Secured by NexaBank AI Systems<br>
            For demo purposes only
        </p>
        """, unsafe_allow_html=True)

# ─── MAIN APP ─────────────────────────────────────────────
else:
    def predict(text):
        vec = tfidf.transform([text])
        prediction = model.predict(vec)[0]
        confidence = model.predict_proba(vec).max()
        probs = model.predict_proba(vec)[0]
        top5_idx = probs.argsort()[-5:][::-1]
        top5_labels = [model.classes_[i].replace('_', ' ').title()
                       for i in top5_idx]
        top5_probs = probs[top5_idx]
        return prediction, confidence, top5_labels, top5_probs

    # Navbar
    nav1, nav2 = st.columns([4, 1])
    with nav1:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #1a3c5e, #4a6fa5);
                    padding: 15px 25px; border-radius: 10px; color: white;">
            <h3 style="margin:0;">🏦 NexaBank AI Support</h3>
            <p style="margin:0; opacity:0.8; font-size:0.9em;">
                Welcome, {st.session_state.user_name}!
            </p>
        </div>
        """, unsafe_allow_html=True)
    with nav2:
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.user_name = ""
            st.session_state.history = []
            st.session_state.query_input = ""
            st.rerun()

    st.markdown("---")

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.markdown("### 💬 How can we help you today?")

        st.markdown("**Quick Questions:**")
        examples = [
            "Why was my card declined?",
            "I want to check my balance",
            "My transfer hasn't arrived yet",
            "I was charged twice",
            "How do I cancel my card?"
        ]

        example_cols = st.columns(2)
        for i, ex in enumerate(examples):
            with example_cols[i % 2]:
                if st.button(ex, key=f"ex_{i}", use_container_width=True):
                    st.session_state.query_input = ex

        user_input = st.text_input(
            "Or describe your issue:",
            value=st.session_state.query_input,
            placeholder="e.g. Why was my card blocked?"
        )

        predict_btn = st.button("🔍 Analyse Query",
                                 use_container_width=True, type="primary")

        should_predict = predict_btn and user_input
        auto_predict = st.session_state.query_input and not predict_btn

        if should_predict or auto_predict:
            query = user_input if should_predict else st.session_state.query_input
            prediction, confidence, top5_labels, top5_probs = predict(query)

            st.session_state.history.insert(0, {
                'query': query,
                'intent': prediction.replace('_', ' ').title(),
                'confidence': f"{confidence:.0%}"
            })
            if len(st.session_state.history) > 5:
                st.session_state.history.pop()

            st.markdown("---")
            st.markdown("### ✅ Query Analysis Result")

            if confidence > 0.7:
                badge = "🟢 High Confidence"
            elif confidence > 0.4:
                badge = "🟡 Medium Confidence"
            else:
                badge = "🔴 Low Confidence"

            st.markdown(f"""
            <div class="result-box">
                <h3 style="color:#1a3c5e; margin:0;">
                    {prediction.replace('_', ' ').title()}
                </h3>
                <p style="color:#4a6fa5; margin:5px 0;">
                    Confidence: <b>{confidence:.0%}</b> &nbsp; {badge}
                </p>
                <p style="color:#888; font-size:0.85em; margin:0;">
                    Your query has been routed to the relevant department.
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("### 📊 Top 5 Predicted Intents")
            fig, ax = plt.subplots(figsize=(7, 3))
            colors = ['#1a3c5e' if i == 0 else '#4a6fa5' for i in range(5)]
            bars = ax.barh(top5_labels[::-1], top5_probs[::-1] * 100,
                           color=colors[::-1])
            ax.set_xlabel('Confidence (%)')
            ax.set_xlim(0, 100)
            for bar, prob in zip(bars, top5_probs[::-1]):
                ax.text(bar.get_width() + 1,
                        bar.get_y() + bar.get_height()/2,
                        f'{prob:.0%}', va='center', fontsize=9)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)

        elif predict_btn and not user_input:
            st.warning("Please enter your query first!")

    with col2:
        st.markdown("### 📈 System Performance")
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Accuracy", "85.62%")
        with m2:
            st.metric("Intents", "77")
        with m3:
            st.metric("Trained On", "10K+")

        st.markdown("---")

        st.markdown("### 🕐 Your Recent Queries")
        if st.session_state.history:
            for item in st.session_state.history:
                st.markdown(f"""
                <div class="result-box">
                    <small style="color:#888;">
                        {item['query'][:45]}...
                    </small><br>
                    <b style="color:#1a3c5e;">{item['intent']}</b>
                    <span style="float:right; color:#4a6fa5;">
                        {item['confidence']}
                    </span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Your query history will appear here.")

        st.markdown("---")
        st.markdown("### ℹ️ About This System")
        st.markdown("""
        AI-powered intent classification system for
        banking customer support.

        **Tech Stack:**
        - Python · Scikit-learn
        - TF-IDF · Logistic Regression
        - Streamlit

        **Model Accuracy: 85.62%**
        """)

    st.markdown("---")
    st.caption("🏦 NexaBank AI Support | Powered by Machine Learning")