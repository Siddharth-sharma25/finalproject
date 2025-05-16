# app.py
import streamlit as st
from datetime import datetime
from analytics.analytics import InstagramAnalytics
from ml.predict_suggestion import predict_suggestion
import requests

# --- App Configuration ---
st.set_page_config(page_title="Instagram Analytics Pro", page_icon="📊", layout="wide")

@st.cache_resource
def init_analytics():
    analyzer = InstagramAnalytics("Instagram_dataset.csv")
    analyzer.load_data()
    analyzer.load_model()
    return analyzer

analyzer = init_analytics()

# --- AI Suggestion Function ---
def get_ai_suggestions(content_type, post_type, metrics):
    try:
        if 'OPENROUTER_API_KEY' not in st.secrets:
            st.error("API key not found.")
            return None

        prompt = f"""
        As an Instagram growth expert, provide specific recommendations for:
        - Content type: {content_type}
        - Post format: {post_type}
        - Current metrics: {metrics}

        Provide:
        1. Three technical optimizations
        2. Two content improvements  
        3. Two engagement strategies
        4. One experimental idea

        Format with bullet points and emojis.
        """

        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {st.secrets['OPENROUTER_API_KEY']}",
                "Content-Type": "application/json"
            },
            json={
                "model": "deepseek/deepseek-r1-zero:free",
                "messages": [
                    {"role": "system", "content": "You're a social media expert with 10+ years experience."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7,
                "max_tokens": 600
            },
            timeout=10
        )

        response.raise_for_status()
        result = response.json()
        print('finslresult0', result)
        return result

    except Exception as e:
        st.error(f"AI suggestion failed: {e}")
        return None


# --- Sidebar ---
with st.sidebar:
    st.title("Account Settings")
    st.selectbox("Audience Size", ["<1K", "1K-10K", "10K-50K", "50K-100K", "100K+"])
    st.selectbox("Primary Niche", ["Lifestyle", "Food", "Travel", "Fashion", "Tech", "Fitness", "Education"])
    st.divider()
    if st.button("🔄 Refresh Data"):
        st.cache_resource.clear()
        st.rerun()
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

# --- Main Tabs ---
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🔮 Predictions", "🤖 AI Recommendations"])

with tab1:
    st.header("Performance Analytics")
    if st.button("Run Full Analysis"):
        with st.spinner("Generating insights..."):
            analyzer.generate_visualizations()
            st.success("Analysis complete!")
            st.metric("Avg Engagement Rate", 
                      f"{(analyzer.data['Likes'].mean() / analyzer.data['Impressions'].mean()) * 100:.2f}%")

with tab2:
    st.header("Impressions Predictor")
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            content_type = st.selectbox("Content Type", ["Photo", "Reel", "Story", "Carousel"])
            likes = st.slider("Expected Likes", 0, 10000, 500)
        with col2:
            post_type = st.selectbox("Post Category", ["Educational", "Entertainment", "Promotional", "Personal"])
            comments = st.slider("Expected Comments", 0, 500, 20)
        if st.form_submit_button("Predict Impressions"):
            features = [[likes, 0, comments, 0, 0]]
            prediction = analyzer.predict(features)
            st.metric("Predicted Impressions", f"{prediction[0]:,.0f}")

with tab3:
    st.header("AI-Powered Optimization")
    with st.form("suggestion_form"):
        col1, col2 = st.columns(2)
        with col1:
            sug_content_type = st.selectbox("Content Type", ["Photo", "Reel", "Story"])
            recent_likes = st.number_input("Recent Likes Avg", 0, 10000, 300)
        with col2:
            sug_post_type = st.selectbox("Content Style", ["Tutorial", "Showcase", "BTS", "Q&A"])
            recent_comments = st.number_input("Recent Comments Avg", 0, 500, 20)
        if st.form_submit_button("Get Recommendations"):
            metrics = {
                "likes": recent_likes,
                "comments": recent_comments
            }
            with st.spinner("Consulting AI..."):
                suggestions = get_ai_suggestions(sug_content_type, sug_post_type, metrics)
                try:
                    ai_message = suggestions["choices"][0]["message"]
                    content = ai_message.get("content", "").strip()
                    reasoning = ai_message.get("reasoning", "").strip()

                    final_output = ""
                    if reasoning:
                        final_output += f"### 🤖 Reasoning\n{reasoning}\n"
                    if content:
                        final_output += f"\n### 📋 Recommendations\n{content}"

                    if final_output:
                        st.success("AI Recommendations")
                        st.markdown(final_output)
                    else:
                        st.warning("Fallback to local ML suggestions")
                        fallback = predict_suggestion(sug_content_type, sug_post_type)
                        st.markdown(f"**💡 {fallback}**")

                except Exception as e:
                    st.error("Error processing AI response")
                    st.text(str(e))


st.divider()
st.caption("Predictions are estimates. Always validate results through experimentation.")
