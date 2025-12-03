
# Streamlit Application Specification: AI Career Navigator for Financial Professionals

## 1. Application Overview

The AI Career Navigator is an interactive Streamlit application designed to help financial professionals assess their AI-Readiness Score ($AI-R$) and explore potential career pathways in the evolving landscape of AI-transformed finance.

The **AI-Readiness Score ($AI-R$)** is a parametric framework that quantifies an individual's preparedness for success in AI-enabled careers. It decomposes career opportunity into two orthogonal components:
1.  **Idiosyncratic Readiness ($V^R$):** Represents individual-specific capabilities that can be actively developed through learning and skill enhancement.
2.  **Systematic Opportunity ($H^R$):** Captures macro-level job growth and demand that individuals can position themselves to capture.

The application allows users to understand these components, calculate their own scores based on hypothetical inputs, and explore "what-if" scenarios for career development.

**Learning Goals:**
*   Understand the components of the AI-Readiness Score ($AI-R$).
*   Evaluate $V^R$ (Idiosyncratic Readiness) across AI-Fluency, Domain-Expertise, and Adaptive-Capacity.
*   Analyze $H^R$ (Systematic Opportunity) for different financial AI roles based on market data.
*   Identify personalized learning pathways to enhance $AI-R$ and career prospects.
*   Grasp the synergistic effects of individual readiness and market opportunity.

## 2. User Interface Requirements

### 2.1 Layout and Navigation Structure

The application will be structured as a single-page Streamlit application with a logical flow, mirroring the Jupyter Notebook's progression. Key sections will be organized using `st.header`, `st.subheader`, and `st.expander` for detailed input/output groups.
*   **Main Content Area**: Will display the application overview, interactive input sections, calculated scores, and visualizations.
*   **Sidebar (`st.sidebar`)**: Will host global parameters ($\alpha$, $\beta$) to ensure they are easily accessible throughout the application.

### 2.2 Input Widgets and Controls

The application will feature interactive widgets for user input, adhering to the requirements for real-time updates.
*   **User Profile Input (Section: Interactive User Profile Input)**:
    *   **AI-Fluency Sub-components**: `st.slider` widgets for 'Technical AI Skills', 'AI-Augmented Productivity', 'Critical AI Judgment', 'AI Learning Velocity' (range [0.0, 1.0], step 0.01). Grouped using `st.expander` or `st.container`.
    *   **Domain-Expertise Factors**: `st.selectbox` for 'Education Level' (options: 'PhD in target field', 'Master\'s in target field', 'Bachelor\'s in target field', 'Associate\'s/Certificate', 'HS + significant coursework'). `st.slider` for 'Years of Experience' (range [0, 30], step 1). `st.slider` widgets for 'Portfolio Score', 'Recognition Score', 'Credentials Score' (range [0.0, 1.0], step 0.01). Grouped using `st.expander` or `st.container`.
    *   **Adaptive-Capacity Sub-components**: `st.slider` widgets for 'Cognitive Flexibility', 'Social-Emotional Intelligence', 'Strategic Career Management' (range [0.0, 100.0], step 1.0). Grouped using `st.expander` or `st.container`.
    *   **Skills Match Score**: `st.slider` for 'Skills Match Score' (range [0.0, 1.0], step 0.01).
    *   **Target Role Selection**: `st.selectbox` for 'Target Role' (options dynamically populated from `market_opportunities_df['Role']`).

*   **Parameter Adjustment (Section: Interactive Parameter Adjustment)**:
    *   `st.sidebar.slider` for $\alpha$ (alpha) parameter (range [0.0, 1.0], step 0.05).
    *   `st.sidebar.slider` for $\beta$ (beta) parameter (range [0.0, 0.5], step 0.01).

*   **Learning Pathway Simulation (Section: "What-If" Scenario: Exploring Learning Pathways)**:
    *   `st.selectbox` for 'Select Pathway' (options dynamically populated from `learning_pathways_df['Pathway_Name']`, including 'None').
    *   `st.slider` for 'Completion' (range [0.0, 1.0], step 0.1).
    *   `st.slider` for 'Mastery' (range [0.0, 1.0], step 0.1).

*   **Multi-Role Comparison (Section: "What-If" Scenario: Career Path Comparison)**:
    *   `st.multiselect` for 'Select Roles' (options dynamically populated from `market_opportunities_df['Role']`, allowing up to three selections).
    *   `st.button` to trigger the comparison visualization.

### 2.3 Visualization Components

All visualizations will be interactive and generated using Plotly.
*   **Radar Chart for $V^R$ (Section: Visualizing Idiosyncratic Readiness ($V^R$) Breakdown)**:
    *   Displays breakdown of 'AI-Fluency', 'Domain-Expertise', and 'Adaptive-Capacity', scaled to 0-100.
    *   Updated in real-time with user profile inputs.
    *   Uses `st.plotly_chart`.

*   **Bar Chart for $H^R$ (Section: Visualizing Systematic Opportunity ($H^R$) Breakdown)**:
    *   Illustrates weighted contributions of 'AI-Enhancement', 'Job Growth', 'Wage Premium', and 'Entry Accessibility' to the Base Systematic Opportunity for the selected target role, scaled to 0-100.
    *   Updated in real-time upon target role selection.
    *   Uses `st.plotly_chart`.

*   **Career Path Comparison Bar Chart (Section: "What-If" Scenario: Career Path Comparison)**:
    *   Compares $V^R$, $H^R$, Synergy, and $AI-R$ for multiple selected target financial roles, scaled to 0-100, grouped by role.
    *   Triggered by a button click.
    *   Uses `st.plotly_chart`.

*   **Data Tables**:
    *   `st.dataframe` to display the `sample_user_profile`, `market_opportunities_df`, `learning_pathways_df`, and the `comparison_df` results.

### 2.4 Interactive Elements and Feedback Mechanisms

*   **Real-time Score Display**:
    *   All calculated scores ($V^R$, $H^R$, Synergy, $AI-R$) and their breakdowns will be displayed prominently using `st.metric` or `st.write` and updated immediately as users interact with input widgets.
    *   Score outputs will be displayed near their respective input sections.
*   **Output Clarity**: All calculated scores and insights will be presented with clear, readable text outputs using `st.markdown` and formatted string literals.
*   **"What-if" Scenario Updates**: Changes in learning pathway inputs or $\alpha, \beta$ parameters will immediately reflect in the projected $AI-R$ scores.

## 3. Additional Requirements

### 3.1 Annotation and Tooltip Specifications

*   **Input Descriptions**: Each input widget (slider, dropdown) will have a clear `description` explaining its purpose and range.
*   **Formula Explanations**: Markdown cells from the notebook that introduce formulas and concepts will be rendered using `st.markdown` and `st.latex` to provide context.
*   **Chart Titles and Labels**: All charts will have descriptive titles, clear axis labels, and legends for easy interpretation.
*   **Inline Explanations**: Where appropriate, `st.info` or `st.help` will be used to provide additional context or definitions.

### 3.2 Save the States of the Fields Properly

*   **Session State**: Streamlit's `st.session_state` will be extensively used to manage and persist the values of all interactive input widgets (sliders, dropdowns, multiselects) and the calculated scores across reruns.
    *   Initial values for all widgets will be set based on the `sample_user_profile` or default parameters.
    *   Any user interaction will update the corresponding value in `st.session_state`.
    *   Calculated variables like `current_user_vr_score`, `current_target_hr_score`, `current_synergy_score`, `current_ai_r_score`, `current_ai_r_score_alpha_beta`, `projected_ai_r_after_pathway`, `delta_ai_r_pathway` will be stored in `st.session_state` to ensure continuity and correct chaining of calculations.

## 4. Notebook Content and Code Requirements

This section details how the Jupyter Notebook content, including markdown and code, will be integrated into the Streamlit application.

### 4.1 Extracted Code Stubs and Usage

All functions defined in the Jupyter Notebook will be migrated to the Streamlit application. Synthetic data generation functions will be run once and cached. Calculation functions will be called reactively based on user inputs. Plotting functions will generate Plotly figures for `st.plotly_chart`.

**Core Imports:**
```python
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
```

**Global Parameters (from Section 2. Setting Up the Environment and Global Parameters):**
These will be defined at the top of the Streamlit script.
```python
# Global Parameters for AI-R formula (Eq 1)
ALPHA_DEFAULT = 0.6
BETA_DEFAULT = 0.15

# Global Weights for V_R components (Eq 15)
W1_AI_FLUENCY = 0.45
W2_DOMAIN_EXPERTISE = 0.35
W3_ADAPTIVE_CAPACITY = 0.20

# Global Weights for H_base components (Eq 5)
W1_AI_ENHANCEMENT = 0.30
W2_JOB_GROWTH = 0.30
W3_WAGE_PREMIUM = 0.25
W4_ENTRY_ACCESSIBILITY = 0.15

# Global Weights for AI-Fluency sub-components (Eq 16)
THETA1_TECHNICAL_AI_SKILLS = 0.30
THETA2_AI_AUGMENTED_PRODUCTIVITY = 0.35
THETA3_CRITICAL_AI_JUDGMENT = 0.20
THETA4_AI_LEARNING_VELOCITY = 0.15

# Other constants
LAMBDA_GROWTH_MULTIPLIER = 0.3 # For M_growth (Eq 13)
GAMMA_REGIONAL_MULTIPLIER = 0.2 # For M_regional (Eq 14)
GAMMA_EXPERIENCE_DECAY = 0.15 # For E_experience (Eq 23)
```

**Data Generation and Caching (from Section 3. Generating Synthetic User Profile Data & Section 4. Generating Synthetic Market Opportunity Data & Section 13. "What-If" Scenario: Exploring Learning Pathways):**
These functions will be decorated with `@st.cache_data` to run only once.
```python
@st.cache_data
def generate_synthetic_user_profile():
    # ... (code from notebook) ...
    user_profile = {
        'User_ID': 'FinancialPro001',
        'AI_Technical_Skills_Input': np.random.uniform(0.6, 0.9),
        'AI_Augmented_Productivity_Input': np.random.uniform(0.5, 0.8),
        'AI_Critical_Judgment_Input': np.random.uniform(0.7, 0.9),
        'AI_Learning_Velocity_Input': np.random.uniform(0.6, 0.85),
        'Education_Level': np.random.choice(['Master\'s in target field', 'PhD in target field', 'Bachelor\'s in target field']),
        'Years_Experience': np.random.randint(5, 15),
        'Portfolio_Score': np.random.uniform(0.6, 0.9),
        'Recognition_Score': np.random.uniform(0.5, 0.8),
        'Credentials_Score': np.random.uniform(0.7, 0.95),
        'Cognitive_Flexibility': np.random.uniform(60, 90),
        'Social_Emotional_Intelligence': np.random.uniform(55, 85),
        'Strategic_Career_Management': np.random.uniform(70, 95),
        'Skills_Match_Score': np.random.uniform(0.7, 0.9)
    }
    return pd.Series(user_profile)

@st.cache_data
def generate_synthetic_market_data():
    # ... (code from notebook) ...
    market_data = {
        'Role': ['Quant Researcher', 'AI Risk Analyst', 'Algorithmic Trader', 'Financial Data Engineer', 'Portfolio Manager'],
        'AI_Enhancement_Potential': [0.85, 0.80, 0.90, 0.88, 0.75],
        'Job_Growth_Projection': [80, 75, 70, 85, 60],
        'Wage_Premium': [0.90, 0.85, 0.95, 0.88, 0.70],
        'Entry_Accessibility': [0.60, 0.70, 0.55, 0.65, 0.75],
        'Job_Postings_t': [1200, 950, 800, 1500, 700],
        'Job_Postings_t_minus_1': [1100, 900, 820, 1350, 720],
        'Local_Demand_Factor': [1.1, 1.05, 1.2, 1.15, 0.95],
        'Remote_Work_Factor': [0.7, 0.8, 0.6, 0.9, 0.5]
    }
    return pd.DataFrame(market_data)

@st.cache_data
def generate_synthetic_learning_pathways():
    # ... (code from notebook) ...
    pathways_data = {
        'Pathway_ID': [1, 2, 3, 4, 5, 6],
        'Pathway_Name': [
            'Prompt Engineering Fundamentals',
            'AI for Financial Analysis',
            'Human-AI Collaboration Skills',
            'Generative AI for Product Design',
            'Advanced ML for Trading',
            'Ethical AI in Finance'
        ],
        'Pathway_Type': [
            'AI-Fluency',
            'Domain + AI Integration',
            'Adaptive Capacity',
            'AI-Fluency',
            'Domain + AI Integration',
            'Adaptive Capacity'
        ],
        'Impact_Coefficient_Delta': [
            10, # AI-Fluency points
            15, # V_R points (can be split across AI-Fluency/Domain)
            8,  # Adaptive Capacity points
            12, # AI-Fluency points
            18, # V_R points
            7   # Adaptive Capacity points
        ],
        'Estimated_Completion_Time_Hours': [40, 80, 60, 50, 120, 30]
    }
    return pd.DataFrame(pathways_data)
```

**Calculation Functions (from Section 5. Calculating Idiosyncratic Readiness ($V^R$) Components, Section 7. Calculating Systematic Opportunity ($H^R$) Components, Section 9. Calculating Synergy Percentage, Section 10. Calculating the Overall AI-Readiness Score ($AI-R$)):**
All `calculate_*` functions will be directly copied into the Streamlit application.
```python
# AI-Fluency sub-component functions
def calculate_technical_ai_skills(prompting, tools, understanding, data_lit):
    return (prompting + tools + understanding + data_lit) / 4

def calculate_ai_augmented_productivity(productivity_score):
    return productivity_score

def calculate_critical_ai_judgment(judgment_score):
    return judgment_score

def calculate_ai_learning_velocity(velocity_score):
    return velocity_score

def calculate_ai_fluency(technical_ai_skills, ai_augmented_productivity, critical_ai_judgment, ai_learning_velocity):
    return (THETA1_TECHNICAL_AI_SKILLS * technical_ai_skills +
            THETA2_AI_AUGMENTED_PRODUCTIVITY * ai_augmented_productivity +
            THETA3_CRITICAL_AI_JUDGMENT * critical_ai_judgment +
            THETA4_AI_LEARNING_VELOCITY * ai_learning_velocity)

# Domain-Expertise sub-component functions
def calculate_educational_foundation(education_level_str):
    mapping = {
        'PhD in target field': 1.0,
        'Master\'s in target field': 0.85,
        'Bachelor\'s in target field': 0.70,
        'Associate\'s/Certificate': 0.60,
        'HS + significant coursework': 0.50
    }
    return mapping.get(education_level_str, 0.0)

def calculate_practical_experience(years_experience, gamma=GAMMA_EXPERIENCE_DECAY):
    return 1 - np.exp(-gamma * years_experience)

def calculate_specialization_depth(portfolio_score, recognition_score, credentials_score):
    return (0.4 * portfolio_score + 0.3 * recognition_score + 0.3 * credentials_score)

def calculate_domain_expertise(educational_foundation, practical_experience, specialization_depth):
    return educational_foundation * practical_experience * specialization_depth

# Adaptive-Capacity function
def calculate_adaptive_capacity(cognitive_flexibility, social_emotional_intelligence, strategic_career_management):
    return (cognitive_flexibility + social_emotional_intelligence + strategic_career_management) / 3

# Overall Idiosyncratic Readiness (V_R)
def calculate_idiosyncratic_readiness(ai_fluency_score, domain_expertise_score, adaptive_capacity_score):
    return (W1_AI_FLUENCY * ai_fluency_score +
            W2_DOMAIN_EXPERTISE * domain_expertise_score +
            W3_ADAPTIVE_CAPACITY * adaptive_capacity_score)

# H_R component functions
def calculate_ai_enhancement(enhancement_score):
    return enhancement_score

def calculate_job_growth_normalized(growth_projection_score):
    return growth_projection_score / 100.0

def calculate_wage_premium(wage_premium_score):
    return wage_premium_score

def calculate_entry_accessibility(accessibility_score):
    return accessibility_score

def calculate_base_opportunity_score(ai_enhancement, job_growth_normalized, wage_premium, entry_accessibility):
    return (W1_AI_ENHANCEMENT * ai_enhancement +
            W2_JOB_GROWTH * job_growth_normalized +
            W3_WAGE_PREMIUM * wage_premium +
            W4_ENTRY_ACCESSIBILITY * entry_accessibility) * 100

def calculate_growth_multiplier(job_postings_t, job_postings_t_minus_1, lambda_param=LAMBDA_GROWTH_MULTIPLIER):
    if job_postings_t_minus_1 == 0: return 1.0
    return 1 + lambda_param * ((job_postings_t / job_postings_t_minus_1) - 1)

def calculate_regional_multiplier(local_demand_factor, remote_work_factor, gamma_param=GAMMA_REGIONAL_MULTIPLIER):
    return local_demand_factor * (1 + gamma_param * remote_work_factor)

def calculate_systematic_opportunity(base_opportunity_score, growth_multiplier, regional_multiplier):
    return min(max(base_opportunity_score * growth_multiplier * regional_multiplier, 0), 100)

def calculate_skills_match_score_func(user_skills_match_input):
    return user_skills_match_input

def calculate_timing_factor(years_experience):
    if years_experience <= 5:
        return 1.0
    elif 5 < years_experience <= 15:
        return 1.0
    else:
        return 0.8

def calculate_alignment_factor(skills_match_score_val, timing_factor_val):
    return skills_match_score_val * timing_factor_val

def calculate_synergy_percentage(v_r_score, h_r_score, alignment_factor):
    return (v_r_score * h_r_score / 100) * alignment_factor

def calculate_ai_readiness_score(v_r_score, h_r_score, synergy_percentage, alpha=ALPHA_DEFAULT, beta=BETA_DEFAULT):
    return (alpha * v_r_score +
            (1 - alpha) * h_r_score +
            beta * synergy_percentage)

def update_ai_readiness_dynamic(current_ai_r, pathway_id, completion_fraction, mastery_score, learning_pathways_df):
    pathway = learning_pathways_df[learning_pathways_df['Pathway_ID'] == pathway_id].iloc[0]
    delta_p = pathway['Impact_Coefficient_Delta']
    delta_ai_r = delta_p * completion_fraction * mastery_score
    return current_ai_r + delta_ai_r, delta_ai_r
```

**Visualization Functions (from Section 6. Visualizing Idiosyncratic Readiness ($V^R$) Breakdown & Section 8. Visualizing Systematic Opportunity ($H^R$) Breakdown):**
These functions will return Plotly figures, which Streamlit will render using `st.plotly_chart`.
```python
def plot_vr_radar_chart(ai_fluency_val, domain_expertise_val, adaptive_capacity_val):
    # ... (code from notebook) ...
    categories = ['AI-Fluency', 'Domain-Expertise', 'Adaptive-Capacity']
    values = [ai_fluency_val, domain_expertise_val, adaptive_capacity_val]
    fig = go.Figure(data=go.Scatterpolar(r=values, theta=categories, fill='toself', name='Idiosyncratic Readiness Components'))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=True, title_text="Idiosyncratic Readiness (V^R) Component Breakdown (Scaled to 100)")
    return fig

def plot_hr_bar_chart(component_contributions, title="Systematic Opportunity (H^R) Component Breakdown"):
    # ... (code from notebook) ...
    df_hr = pd.DataFrame(list(component_contributions.items()), columns=['Component', 'Contribution'])
    fig = px.bar(df_hr, x='Component', y='Contribution', title=title, labels={'Contribution': 'Weighted Contribution to Base H^R (0-100)'}, color='Contribution', color_continuous_scale=px.colors.sequential.Plasma)
    fig.update_layout(yaxis_range=[0, max(df_hr['Contribution'].max() + 10, 100)])
    return fig
```

**Interactive Logic:**
The `update_and_display_scores`, `update_alpha_beta_display`, `simulate_pathway_impact`, and `compare_career_paths` logic from the notebook will be refactored into Streamlit's reactive model. Instead of `ipywidgets.interactive_output` and `clear_output`, Streamlit's natural rerun on widget change will drive the updates, with outputs rendered directly using `st.write`, `st.plotly_chart`, `st.dataframe`, and `st.session_state` for managing state.

### 4.2 Markdown Content

All relevant markdown cells from the Jupyter Notebook will be included in the Streamlit application using `st.markdown` for general text and `st.latex` for mathematical equations, ensuring strict adherence to LaTeX formatting.

**Example of Markdown to Streamlit Mapping:**

**From Notebook (Application Overview):**
```markdown
# AI Career Navigator for Financial Professionals
Welcome to the AI Career Navigator! This interactive tool helps financial professionals assess their AI-Readiness Score (AI-R) and explore potential career pathways in the rapidly evolving world of AI-transformed finance.
The **AI-Readiness Score ($AI-R$)** is a parametric framework that quantifies an individual's preparedness for success in AI-enabled careers. It decomposes career opportunity into two orthogonal components:
1.  **Idiosyncratic Readiness ($V^R$):** Represents individual-specific capabilities that can be actively developed through learning and skill enhancement.
2.  **Systematic Opportunity ($H^R$):** Captures macro-level job growth and demand that individuals can position themselves to capture.
This notebook will guide you through understanding these components, calculating your own scores based on hypothetical inputs, and exploring "what-if" scenarios for career development.
**Learning Objectives:**
*   Understand the components of the AI-Readiness Score ($AI-R$).
*   Evaluate your current $V^R$ (Idiosyncratic Readiness) across AI-Fluency, Domain-Expertise, and Adaptive-Capacity.
*   Analyze $H^R$ (Systematic Opportunity) for different financial AI roles based on market data.
*   Identify personalized learning pathways to enhance your $AI-R$ and career prospects.
*   Grasp the synergistic effects of individual readiness and market opportunity.
```
**To Streamlit:**
```python
st.title("AI Career Navigator for Financial Professionals")
st.markdown("Welcome to the AI Career Navigator! This interactive tool helps financial professionals assess their AI-Readiness Score (AI-R) and explore potential career pathways in the rapidly evolving world of AI-transformed finance.")
st.markdown("The **AI-Readiness Score ($AI-R$)** is a parametric framework that quantifies an individual's preparedness for success in AI-enabled careers. It decomposes career opportunity into two orthogonal components:")
st.markdown("1. **Idiosyncratic Readiness ($V^R$):** Represents individual-specific capabilities that can be actively developed through learning and skill enhancement.")
st.markdown("2. **Systematic Opportunity ($H^R$):** Captures macro-level job growth and demand that individuals can position themselves to capture.")
st.markdown("This application will guide you through understanding these components, calculating your own scores based on hypothetical inputs, and exploring \"what-if\" scenarios for career development.")
st.subheader("Learning Objectives:")
st.markdown("""
*   Understand the components of the AI-Readiness Score ($AI-R$).
*   Evaluate your current $V^R$ (Idiosyncratic Readiness) across AI-Fluency, Domain-Expertise, and Adaptive-Capacity.
*   Analyze $H^R$ (Systematic Opportunity) for different financial AI roles based on market data.
*   Identify personalized learning pathways to enhance your $AI-R$ and career prospects.
*   Grasp the synergistic effects of individual readiness and market opportunity.
""")
```

**Markdown with Formulas (Example from Section 2. Setting Up the Environment and Global Parameters):**
```markdown
The main formula for the AI-Readiness Score is:
$$AI-R_{i,t} = \alpha \cdot V^R_i(t) + (1 - \alpha) \cdot H^R(t) + \beta \cdot Synergy\%(V^R, H^R)$$
Where:
*   $\alpha \in [0,1]$: Weight on individual vs. market factors. (Prior: $\alpha \in [0.5, 0.7]$)
*   $\beta > 0$: Synergy coefficient. (Prior: $\beta \in [0.05, 0.20]$)
```
**To Streamlit:**
```python
st.subheader("2. Setting Up the Environment and Global Parameters")
st.markdown("First, we import the necessary Python libraries and define the global parameters (weights and coefficients) that will be used throughout our calculations. These parameters are derived from the research framework and represent the relative importance of various factors.")
st.markdown("The main formula for the AI-Readiness Score is:")
st.latex(r"AI-R_{i,t} = \alpha \cdot V^R_i(t) + (1 - \alpha) \cdot H^R(t) + \beta \cdot Synergy\%(V^R, H^R)")
st.markdown(r"Where:")
st.markdown(r"* $\alpha \in [0,1]$: Weight on individual vs. market factors. (Prior: $\alpha \in [0.5, 0.7]$)")
st.markdown(r"* $\beta > 0$: Synergy coefficient. (Prior: $\beta \in [0.05, 0.20]$)")
```
All formulas, sub-component descriptions, and conceptual explanations will be translated using `st.markdown` and `st.latex` to preserve the original notebook's educational value and detail.

