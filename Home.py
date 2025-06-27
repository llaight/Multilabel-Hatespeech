import streamlit as st

st.set_page_config(
    page_title="Multilabel Hate Speech Detection",
    page_icon="📊",
    layout="wide"
)

st.title("Multilabel Hate Speech Detection")

# Top navigation bar
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("🤖 SGD Model", use_container_width=True):
        st.switch_page("pages/SGD-Hybrid_Model.py")
with col2:
    if st.button("🔍 SVM Model", use_container_width=True):
        st.switch_page("pages/SVM-Hybrid Model.py")
with col3:
    if st.button("🏠 About", use_container_width=True):
        st.switch_page("pages/About.py")

st.markdown("---")  

st.markdown("""
This is a multi-label hate speech detection tool designed to identify and categorize hateful content across multiple dimensions: age, gender, physical appearance, race, religion, politics, and others. It integrates two hybrid models: **SGD + rule-based** and **SVM + rule-based**, combining machine learning with expert-defined linguistic rules for more accurate and interpretable classification.
""")

st.markdown("### Classification Labels")
st.markdown("""
**Age**: Hate speech targeting individuals based on their age or generational status, from childhood to seniority.

**Gender**: Hate speech related to social constructs, roles, and identities such as man, woman, and non-binary individuals.

**Physical**: Hate speech aimed at external traits such as facial features, body size, clothing, or hygiene.

**Politics**: Hate speech related to political beliefs, government policies, or partisan affiliations.

**Race**: Hate speech based on social or physical constructs such as skin color, ethnicity, ancestry, or cultural background.

**Religion**: Hate speech aimed at individuals or groups based on their religious beliefs, practices, or affiliations.

**Others**: Hate speech not conforming to the generic categories such as age, gender, race, or religion. This may also involve attacks like that on social class, education, occupation, disability, or personal attributes groups which are not otherwise specified.
""")

