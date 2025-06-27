import streamlit as st
st.title("Multilabel Hate Speech Detection")

st.markdown("""
This project was developed in partial fulfillment for the course **COSC 304 – Introduction to Artificial Intelligence**. It serves as a practical application of AI techniques, specifically in the area of natural language processing, by building a multi-label hate speech detection system.
""")

st.markdown("""**Note:** *This model has been trained exclusively on hate speech datasets and may not accurately classify non-hate speech content.* """)

st.markdown("### Developers")

# CSS for developer cards
st.markdown("""
<style>
    .dev-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.15);
        margin: 0.5rem 0;
        color: white;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        height: 150px;
        text-align: center;
    }
    
    .dev-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.2);
    }
    
    .dev-name {
        font-size: 1.1rem;
        font-weight: bold;
        margin-bottom: 0.7rem;
        color: #ffffff;
    }
    
    .dev-info {
        font-size: 0.8rem;
        margin: 0.4rem 0;
        color: #f0f0f0;
        line-height: 1.3;
        text-align: left;
    }
    
    .dev-email {
        font-size: 0.6rem;
        margin: 0.2rem 0;
        color: #f0f0f0;
        line-height: 1.3;
        text-align: left;
    }
    
    .dev-icon {
        font-size: 0.9rem;
        margin-right: 0.3rem;
    }
    
    .github-link {
        color: #ffffff !important;
        text-decoration: none;
        border-bottom: 1px solid rgba(255,255,255,0.3);
        font-size: 0.75rem;
    }
    
    .github-link:hover {
        border-bottom: 1px solid #ffffff;
    }
</style>
""", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="dev-card">
        <div class="dev-name">Chynna Doria</div>
        <div class="dev-info"><span class="dev-icon">📱</span>09070377206</div>
        <div class="dev-email"><span class="dev-icon">📧</span>chynnadoria18@email.com</div>
        <div class="dev-info"><span class="dev-icon">🐙</span><a href="https://github.com/chynnadoria" class="github-link" target="_blank">github.com/johndoe</a></div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="dev-card">
        <div class="dev-name">Daniela Joaquin</div>
        <div class="dev-info"><span class="dev-icon">📱</span>09219402904</div>
        <div class="dev-email"><span class="dev-icon">📧</span>joaquindaniela2018@gmail.com</div>
        <div class="dev-info"><span class="dev-icon">🐙</span><a href="https://github.com/Thaniela" class="github-link" target="_blank">github.com/janesmith</a></div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="dev-card">
        <div class="dev-name">Angela Loro</div>
        <div class="dev-info"><span class="dev-icon">📱</span>09276355732</div>
        <div class="dev-email"><span class="dev-icon">📧</span>aloro152@gmail.com</div>
        <div class="dev-info"><span class="dev-icon">🐙</span><a href="https://github.com/llaight" class="github-link" target="_blank">github.com/alexjohnson</a></div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="dev-card">
        <div class="dev-name">Alexandra Panela</div>
        <div class="dev-info"><span class="dev-icon">📱</span>09398765432</div>
        <div class="dev-email"><span class="dev-icon">📧</span>panela.alex30@gmail.com</div>
        <div class="dev-info"><span class="dev-icon">🐙</span><a href="https://github.com/abclexd" class="github-link" target="_blank">github.com/mariagarcia</a></div>
    </div>
    """, unsafe_allow_html=True)

