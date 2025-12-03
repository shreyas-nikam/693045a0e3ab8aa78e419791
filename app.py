
import streamlit as st

st.set_page_config(page_title="QuLab", layout="wide")
st.sidebar.image("https://www.quantuniversity.com/assets/img/logo5.jpg")
st.sidebar.divider()
st.title("QuLab")
st.divider()
# Your code starts here
st.markdown("""
In this lab, we present the **AI Career Navigator**, an interactive Streamlit application designed to help financial professionals assess their AI-Readiness Score ($AI-R$) and explore potential career pathways in the evolving landscape of AI-transformed finance.

This tool allows you to: 
*   **Understand your AI-Readiness Score ($AI-R$)**: A parametric framework that quantifies your preparedness for success in AI-enabled careers. It breaks down career opportunity into two key components: Idiosyncratic Readiness ($V^R$) and Systematic Opportunity ($H^R$).
*   **Evaluate your individual capabilities ($V^R$)**: Assess your AI-Fluency, Domain-Expertise, and Adaptive-Capacity.
*   **Analyze market opportunities ($H^R$)**: Explore job growth and demand for various financial AI roles.
*   **Simulate learning pathways**: See how educational interventions can enhance your $AI-R$.
*   **Compare career paths**: Get insights into different roles based on their $AI-R$ components.

Use the navigation in the sidebar to explore the application.
""")

page = st.sidebar.selectbox(label="Navigation", options=["AI Career Navigator Main"])
if page == "AI Career Navigator Main":
    from application_pages.ai_career_navigator_main import main
    main()

# Your code ends here
