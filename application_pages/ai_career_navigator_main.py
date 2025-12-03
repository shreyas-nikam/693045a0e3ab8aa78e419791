
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

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

@st.cache_data
def generate_synthetic_user_profile():
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
            THETA4_AI_LEARNING_VELOCITY * ai_learning_velocity) * 100 # Scale to 100

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
    # This formula in the spec seems to imply a multiplication, but it might lead to very small numbers.
    # Assuming it's a weighted average for a more intuitive score, similar to AI-Fluency components.
    # However, sticking to the spec as it is. Will output the product. 
    # The original notebook code likely has a different scaling or combination. 
    # For now, following the provided formula structure.
    # If the outputs are too small, this might need re-evaluation based on the actual notebook.
    return educational_foundation * practical_experience * specialization_depth * 100 # Scale to 100

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
    # The formula is (v_r_score * h_r_score / 100) * alignment_factor. 
    # If v_r and h_r are already scaled to 100, dividing by 100 makes sense to get a percentage of max potential.
    return (v_r_score * h_r_score / 100) * alignment_factor

def calculate_ai_readiness_score(v_r_score, h_r_score, synergy_percentage, alpha=ALPHA_DEFAULT, beta=BETA_DEFAULT):
    return (alpha * v_r_score +
            (1 - alpha) * h_r_score +
            beta * synergy_percentage)

def update_ai_readiness_dynamic(current_ai_r, pathway_id, completion_fraction, mastery_score, learning_pathways_df):
    if pathway_id == 'None':
        return current_ai_r, 0 # No change if no pathway selected
    pathway = learning_pathways_df[learning_pathways_df['Pathway_ID'] == pathway_id].iloc[0]
    delta_p = pathway['Impact_Coefficient_Delta']
    delta_ai_r = delta_p * completion_fraction * mastery_score
    return current_ai_r + delta_ai_r, delta_ai_r

def plot_vr_radar_chart(ai_fluency_val, domain_expertise_val, adaptive_capacity_val):
    categories = ['AI-Fluency', 'Domain-Expertise', 'Adaptive-Capacity']
    values = [ai_fluency_val, domain_expertise_val, adaptive_capacity_val] # Values are already scaled to 100
    fig = go.Figure(data=go.Scatterpolar(r=values, theta=categories, fill='toself', name='Idiosyncratic Readiness Components'))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=True, title_text="Idiosyncratic Readiness (V^R) Component Breakdown (Scaled to 100)")
    return fig

def plot_hr_bar_chart(component_contributions, title="Systematic Opportunity (H^R) Component Breakdown"):
    df_hr = pd.DataFrame(list(component_contributions.items()), columns=['Component', 'Contribution'])
    fig = px.bar(df_hr, x='Component', y='Contribution', title=title, labels={'Contribution': 'Weighted Contribution to Base H^R (0-100)'}, color='Contribution', color_continuous_scale=px.colors.sequential.Plasma)
    fig.update_layout(yaxis_range=[0, max(df_hr['Contribution'].max() + 10, 100)])
    return fig

def plot_comparison_chart(comparison_df):
    fig = px.bar(comparison_df, x="Role", y=["V^R", "H^R", "Synergy", "AI-R"], 
                 title="Career Path Comparison (Scaled to 100)",
                 barmode='group', 
                 labels={"value": "Score (0-100)", "variable": "Component"},
                 color_discrete_map={
                     "V^R": "blue",
                     "H^R": "red",
                     "Synergy": "green",
                     "AI-R": "purple"
                 })
    fig.update_layout(yaxis_range=[0, 100])
    return fig

def main():
    # Initialize session state for all inputs and calculated values
    if "user_profile" not in st.session_state:
        st.session_state.user_profile = generate_synthetic_user_profile()
    if "market_opportunities_df" not in st.session_state:
        st.session_state.market_opportunities_df = generate_synthetic_market_data()
    if "learning_pathways_df" not in st.session_state:
        st.session_state.learning_pathways_df = generate_synthetic_learning_pathways()

    # Initialize input states
    if "alpha" not in st.session_state:
        st.session_state.alpha = ALPHA_DEFAULT
    if "beta" not in st.session_state:
        st.session_state.beta = BETA_DEFAULT

    # User Profile Inputs (initialize with generated data or defaults)
    if "ai_technical_skills_input" not in st.session_state:
        st.session_state.ai_technical_skills_input = st.session_state.user_profile['AI_Technical_Skills_Input']
    if "ai_augmented_productivity_input" not in st.session_state:
        st.session_state.ai_augmented_productivity_input = st.session_state.user_profile['AI_Augmented_Productivity_Input']
    if "ai_critical_judgment_input" not in st.session_state:
        st.session_state.ai_critical_judgment_input = st.session_state.user_profile['AI_Critical_Judgment_Input']
    if "ai_learning_velocity_input" not in st.session_state:
        st.session_state.ai_learning_velocity_input = st.session_state.user_profile['AI_Learning_Velocity_Input']
    
    if "education_level" not in st.session_state:
        st.session_state.education_level = st.session_state.user_profile['Education_Level']
    if "years_experience" not in st.session_state:
        st.session_state.years_experience = int(st.session_state.user_profile['Years_Experience'])
    if "portfolio_score" not in st.session_state:
        st.session_state.portfolio_score = st.session_state.user_profile['Portfolio_Score']
    if "recognition_score" not in st.session_state:
        st.session_state.recognition_score = st.session_state.user_profile['Recognition_Score']
    if "credentials_score" not in st.session_state:
        st.session_state.credentials_score = st.session_state.user_profile['Credentials_Score']

    if "cognitive_flexibility" not in st.session_state:
        st.session_state.cognitive_flexibility = st.session_state.user_profile['Cognitive_Flexibility']
    if "social_emotional_intelligence" not in st.session_state:
        st.session_state.social_emotional_intelligence = st.session_state.user_profile['Social_Emotional_Intelligence']
    if "strategic_career_management" not in st.session_state:
        st.session_state.strategic_career_management = st.session_state.user_profile['Strategic_Career_Management']
    
    if "skills_match_score" not in st.session_state:
        st.session_state.skills_match_score = st.session_state.user_profile['Skills_Match_Score']
    
    if "target_role" not in st.session_state:
        st.session_state.target_role = st.session_state.market_opportunities_df['Role'].iloc[0]

    # Learning Pathway Simulation states
    if "selected_pathway_name" not in st.session_state:
        st.session_state.selected_pathway_name = 'None'
    if "pathway_completion" not in st.session_state:
        st.session_state.pathway_completion = 0.0
    if "pathway_mastery" not in st.session_state:
        st.session_state.pathway_mastery = 0.0

    # Multi-Role Comparison states
    if "selected_comparison_roles" not in st.session_state:
        st.session_state.selected_comparison_roles = []

    st.title("AI Career Navigator for Financial Professionals")
    st.markdown("Welcome to the AI Career Navigator! This interactive tool helps financial professionals assess their AI-Readiness Score ($AI-R$) and explore potential career pathways in the rapidly evolving world of AI-transformed finance.")
    st.markdown("The **AI-Readiness Score ($AI-R$)** is a parametric framework that quantifies an individual's preparedness for success in AI-enabled careers. It decomposes career opportunity into two orthogonal components:")
    st.markdown("1. **Idiosyncratic Readiness ($V^R$):** Represents individual-specific capabilities that can be actively developed through learning and skill enhancement.")
    st.markdown("2. **Systematic Opportunity ($H^R$):** Captures macro-level job growth and demand that individuals can position themselves to capture.")
    st.markdown("This application will guide you through understanding these components, calculating your own scores based on hypothetical inputs, and exploring \"what-if\" scenarios for career development.")
    st.subheader("Learning Objectives:")
    st.markdown(r"""
*   Understand the components of the AI-Readiness Score ($AI-R$).
*   Evaluate your current $V^R$ (Idiosyncratic Readiness) across AI-Fluency, Domain-Expertise, and Adaptive-Capacity.
*   Analyze $H^R$ (Systematic Opportunity) for different financial AI roles based on market data.
*   Identify personalized learning pathways to enhance your $AI-R$ and career prospects.
*   Grasp the synergistic effects of individual readiness and market opportunity.
""")

    st.subheader("2. Setting Up the Environment and Global Parameters")
    st.markdown("First, we import the necessary Python libraries and define the global parameters (weights and coefficients) that will be used throughout our calculations. These parameters are derived from the research framework and represent the relative importance of various factors.")
    st.markdown("The main formula for the AI-Readiness Score is:")
    st.latex(r"AI-R_{i,t} = \alpha \cdot V^R_i(t) + (1 - \alpha) \cdot H^R(t) + \beta \cdot Synergy\%(V^R, H^R)")
    st.markdown(r"Where:")
    st.markdown(r"* $\alpha \in [0,1]$: Weight on individual vs. market factors. (Prior: $\alpha \in [0.5, 0.7]$)")
    st.markdown(r"* $\beta > 0$: Synergy coefficient. (Prior: $\beta \in [0.05, 0.20]$)")

    st.sidebar.header("Global Parameters")
    st.session_state.alpha = st.sidebar.slider(r'$\alpha$ (Alpha) Parameter', 0.0, 1.0, st.session_state.alpha, 0.05, key='alpha_slider', help=r'Weight on individual vs. market factors in the $AI-R$ formula.')
    st.session_state.beta = st.sidebar.slider(r'$\beta$ (Beta) Parameter', 0.0, 0.5, st.session_state.beta, 0.01, key='beta_slider', help=r'Synergy coefficient in the $AI-R$ formula.')
    
    st.header("Interactive User Profile Input")
    st.markdown("Adjust the sliders and select boxes below to define your hypothetical user profile. These inputs will dynamically update your Idiosyncratic Readiness ($V^R$) score.")

    col1, col2 = st.columns(2)

    with col1:
        with st.expander("AI-Fluency Sub-components", expanded=True):
            st.markdown(r"**AI-Fluency ($F^{AI}$):** Your proficiency in understanding and applying AI concepts.")
            st.session_state.ai_technical_skills_input = st.slider('Technical AI Skills', 0.0, 1.0, st.session_state.ai_technical_skills_input, 0.01, key='tech_ai_skills_slider', help='Your ability to use AI tools and techniques.')
            st.session_state.ai_augmented_productivity_input = st.slider('AI-Augmented Productivity', 0.0, 1.0, st.session_state.ai_augmented_productivity_input, 0.01, key='ai_prod_slider', help='How effectively you leverage AI to enhance your work output.')
            st.session_state.ai_critical_judgment_input = st.slider('Critical AI Judgment', 0.0, 1.0, st.session_state.ai_critical_judgment_input, 0.01, key='crit_ai_judg_slider', help='Your capacity to critically evaluate AI outputs and decisions.')
            st.session_state.ai_learning_velocity_input = st.slider('AI Learning Velocity', 0.0, 1.0, st.session_state.ai_learning_velocity_input, 0.01, key='ai_learn_vel_slider', help='Your speed and effectiveness in acquiring new AI-related knowledge and skills.')
        
        with st.expander("Adaptive-Capacity Sub-components", expanded=True):
            st.markdown(r"**Adaptive-Capacity ($C^{AD}$):** Your ability to thrive in dynamic, AI-driven environments.")
            st.session_state.cognitive_flexibility = st.slider('Cognitive Flexibility', 0.0, 100.0, st.session_state.cognitive_flexibility, 1.0, key='cog_flex_slider', help='Your openness to new ideas and ability to switch between different problem-solving strategies.')
            st.session_state.social_emotional_intelligence = st.slider('Social-Emotional Intelligence', 0.0, 100.0, st.session_state.social_emotional_intelligence, 1.0, key='soc_emo_int_slider', help='Your ability to understand and manage your own emotions, and to perceive and influence the emotions of others.')
            st.session_state.strategic_career_management = st.slider('Strategic Career Management', 0.0, 100.0, st.session_state.strategic_career_management, 1.0, key='strat_car_man_slider', help='Your proactivity in planning and navigating your career path in response to industry changes.')

    with col2:
        with st.expander("Domain-Expertise Factors", expanded=True):
            st.markdown(r"**Domain-Expertise ($E^{DO}$):** Your depth of knowledge and experience in a specific financial domain.")
            education_options = ['PhD in target field', 'Master\'s in target field', 'Bachelor\'s in target field', 'Associate\'s/Certificate', 'HS + significant coursework']
            st.session_state.education_level = st.selectbox('Education Level', options=education_options, index=education_options.index(st.session_state.education_level), key='edu_level_select', help='Your highest level of education relevant to the target field.')
            st.session_state.years_experience = st.slider('Years of Experience', 0, 30, st.session_state.years_experience, 1, key='years_exp_slider', help='Number of years of professional experience in your domain.')
            st.session_state.portfolio_score = st.slider('Portfolio Score', 0.0, 1.0, st.session_state.portfolio_score, 0.01, key='portfolio_slider', help='Quality and relevance of your professional portfolio/projects.')
            st.session_state.recognition_score = st.slider('Recognition Score', 0.0, 1.0, st.session_state.recognition_score, 0.01, key='recognition_slider', help='Awards, publications, or industry recognition.')
            st.session_state.credentials_score = st.slider('Credentials Score', 0.0, 1.0, st.session_state.credentials_score, 0.01, key='credentials_slider', help='Certifications, licenses, or advanced degrees.')
        
        st.session_state.skills_match_score = st.slider('Skills Match Score', 0.0, 1.0, st.session_state.skills_match_score, 0.01, key='skills_match_slider', help='How well your current skills align with the requirements of the target role.')
        st.session_state.target_role = st.selectbox('Target Role', options=st.session_state.market_opportunities_df['Role'].tolist(), index=st.session_state.market_opportunities_df['Role'].tolist().index(st.session_state.target_role), key='target_role_select', help='The specific AI-enabled financial role you are targeting.')
    
    # Perform calculations based on current session state
    # AI-Fluency Calculations (scaled to 100)
    tech_ai_skills_calc = calculate_technical_ai_skills(st.session_state.ai_technical_skills_input, st.session_state.ai_technical_skills_input, st.session_state.ai_technical_skills_input, st.session_state.ai_technical_skills_input) # Assuming all sub-parts are represented by one slider for simplicity as per spec
    ai_augmented_productivity_calc = calculate_ai_augmented_productivity(st.session_state.ai_augmented_productivity_input)
    critical_ai_judgment_calc = calculate_critical_ai_judgment(st.session_state.ai_critical_judgment_input)
    ai_learning_velocity_calc = calculate_ai_learning_velocity(st.session_state.ai_learning_velocity_input)
    st.session_state.ai_fluency_score_scaled = calculate_ai_fluency(tech_ai_skills_calc, ai_augmented_productivity_calc, critical_ai_judgment_calc, ai_learning_velocity_calc)

    # Domain-Expertise Calculations (scaled to 100)
    educational_foundation_calc = calculate_educational_foundation(st.session_state.education_level)
    practical_experience_calc = calculate_practical_experience(st.session_state.years_experience)
    specialization_depth_calc = calculate_specialization_depth(st.session_state.portfolio_score, st.session_state.recognition_score, st.session_state.credentials_score)
    st.session_state.domain_expertise_score_scaled = calculate_domain_expertise(educational_foundation_calc, practical_experience_calc, specialization_depth_calc)
    
    # Adaptive-Capacity Calculation (already 0-100)
    st.session_state.adaptive_capacity_score = calculate_adaptive_capacity(st.session_state.cognitive_flexibility, st.session_state.social_emotional_intelligence, st.session_state.strategic_career_management)

    # Overall V^R Calculation (scaled to 100)
    st.session_state.current_user_vr_score = calculate_idiosyncratic_readiness(
        st.session_state.ai_fluency_score_scaled,
        st.session_state.domain_expertise_score_scaled,
        st.session_state.adaptive_capacity_score
    )

    # H^R Calculations
    target_role_data = st.session_state.market_opportunities_df[st.session_state.market_opportunities_df['Role'] == st.session_state.target_role].iloc[0]
    
    ai_enhancement_calc = calculate_ai_enhancement(target_role_data['AI_Enhancement_Potential'])
    job_growth_normalized_calc = calculate_job_growth_normalized(target_role_data['Job_Growth_Projection'])
    wage_premium_calc = calculate_wage_premium(target_role_data['Wage_Premium'])
    entry_accessibility_calc = calculate_entry_accessibility(target_role_data['Entry_Accessibility'])

    base_opportunity = calculate_base_opportunity_score(ai_enhancement_calc, job_growth_normalized_calc, wage_premium_calc, entry_accessibility_calc)

    growth_multiplier = calculate_growth_multiplier(target_role_data['Job_Postings_t'], target_role_data['Job_Postings_t_minus_1'])
    regional_multiplier = calculate_regional_multiplier(target_role_data['Local_Demand_Factor'], target_role_data['Remote_Work_Factor'])

    st.session_state.current_target_hr_score = calculate_systematic_opportunity(base_opportunity, growth_multiplier, regional_multiplier)

    # H^R Component Contributions for visualization (scaled to 100)
    hr_component_contributions = {
        'AI-Enhancement': W1_AI_ENHANCEMENT * ai_enhancement_calc * 100,
        'Job Growth': W2_JOB_GROWTH * job_growth_normalized_calc * 100,
        'Wage Premium': W3_WAGE_PREMIUM * wage_premium_calc * 100,
        'Entry Accessibility': W4_ENTRY_ACCESSIBILITY * entry_accessibility_calc * 100
    }

    # Synergy Calculations
    skills_match_score_val = calculate_skills_match_score_func(st.session_state.skills_match_score)
    timing_factor_val = calculate_timing_factor(st.session_state.years_experience)
    alignment_factor_val = calculate_alignment_factor(skills_match_score_val, timing_factor_val)
    st.session_state.current_synergy_score = calculate_synergy_percentage(st.session_state.current_user_vr_score, st.session_state.current_target_hr_score, alignment_factor_val)

    # Overall AI-R Calculation
    st.session_state.current_ai_r_score = calculate_ai_readiness_score(
        st.session_state.current_user_vr_score,
        st.session_state.current_target_hr_score,
        st.session_state.current_synergy_score,
        alpha=st.session_state.alpha, beta=st.session_state.beta
    )

    st.subheader("Calculated AI-Readiness Scores")
    col_scores_1, col_scores_2, col_scores_3, col_scores_4 = st.columns(4)
    with col_scores_1:
        st.metric(label=r"**Idiosyncratic Readiness ($V^R$)**", value=f"{st.session_state.current_user_vr_score:.2f}")
    with col_scores_2:
        st.metric(label=r"**Systematic Opportunity ($H^R$)**", value=f"{st.session_state.current_target_hr_score:.2f}")
    with col_scores_3:
        st.metric(label=r"**Synergy %**", value=f"{st.session_state.current_synergy_score:.2f}")
    with col_scores_4:
        st.metric(label=r"**Overall AI-Readiness ($AI-R$)**", value=f"{st.session_state.current_ai_r_score:.2f}")

    st.divider()

    st.subheader("Visualizing Idiosyncratic Readiness ($V^R$) Breakdown")
    st.markdown("This radar chart illustrates the breakdown of your Idiosyncratic Readiness across AI-Fluency, Domain-Expertise, and Adaptive-Capacity, scaled to 0-100. Higher values indicate stronger readiness in that area.")
    st.plotly_chart(plot_vr_radar_chart(
        st.session_state.ai_fluency_score_scaled,
        st.session_state.domain_expertise_score_scaled,
        st.session_state.adaptive_capacity_score
    ), use_container_width=True)
    
    st.divider()

    st.subheader("Visualizing Systematic Opportunity ($H^R$) Breakdown")
    st.markdown(f"This bar chart displays the weighted contributions of different factors to the Base Systematic Opportunity for the selected role: **{st.session_state.target_role}**.")
    st.plotly_chart(plot_hr_bar_chart(hr_component_contributions), use_container_width=True)

    st.divider()

    st.header("\"What-If\" Scenario: Exploring Learning Pathways")
    st.markdown("Simulate the impact of completing a learning pathway on your AI-Readiness Score. Select a pathway and adjust completion and mastery levels.")

    pathway_names = ['None'] + st.session_state.learning_pathways_df['Pathway_Name'].tolist()
    st.session_state.selected_pathway_name = st.selectbox('Select Pathway', options=pathway_names, index=pathway_names.index(st.session_state.selected_pathway_name), key='pathway_select', help='Choose a learning pathway to simulate its impact.')
    
    if st.session_state.selected_pathway_name != 'None':
        st.session_state.pathway_completion = st.slider('Completion', 0.0, 1.0, st.session_state.pathway_completion, 0.1, key='pathway_completion_slider', help='Fraction of the pathway completed (0.0 to 1.0).')
        st.session_state.pathway_mastery = st.slider('Mastery', 0.0, 1.0, st.session_state.pathway_mastery, 0.1, key='pathway_mastery_slider', help='Level of mastery achieved in the pathway (0.0 to 1.0).')

        selected_pathway_id = st.session_state.learning_pathways_df[st.session_state.learning_pathways_df['Pathway_Name'] == st.session_state.selected_pathway_name]['Pathway_ID'].iloc[0]
        
        projected_ai_r, delta_ai_r = update_ai_readiness_dynamic(
            st.session_state.current_ai_r_score,
            selected_pathway_id,
            st.session_state.pathway_completion,
            st.session_state.pathway_mastery,
            st.session_state.learning_pathways_df
        )

        st.markdown(f"**Current AI-Readiness Score:** {st.session_state.current_ai_r_score:.2f}")
        st.markdown(f"**Projected AI-Readiness Score after Pathway:** {projected_ai_r:.2f}")
        st.info(f"This pathway is projected to increase your AI-Readiness Score by **{delta_ai_r:.2f}** points.")
    else:
        st.info("Select a pathway to simulate its impact on your AI-Readiness Score.")

    st.divider()

    st.header("\"What-If\" Scenario: Career Path Comparison")
    st.markdown("Compare the AI-Readiness components for up to three different financial AI roles. Select roles and click 'Compare Roles'.")

    available_roles = st.session_state.market_opportunities_df['Role'].tolist()
    st.session_state.selected_comparison_roles = st.multiselect('Select Roles for Comparison', 
                                                                 options=available_roles, 
                                                                 default=st.session_state.selected_comparison_roles,
                                                                 max_selections=3,
                                                                 key='compare_roles_multiselect')
    
    if st.button('Compare Roles'):
        if not st.session_state.selected_comparison_roles:
            st.warning("Please select at least one role for comparison.")
        else:
            comparison_data = []
            for role_name in st.session_state.selected_comparison_roles:
                # Re-calculate HR, Synergy, and AI-R for each selected role with current user VR
                selected_market_data = st.session_state.market_opportunities_df[st.session_state.market_opportunities_df['Role'] == role_name].iloc[0]

                ai_enhancement_comp = calculate_ai_enhancement(selected_market_data['AI_Enhancement_Potential'])
                job_growth_normalized_comp = calculate_job_growth_normalized(selected_market_data['Job_Growth_Projection'])
                wage_premium_comp = calculate_wage_premium(selected_market_data['Wage_Premium'])
                entry_accessibility_comp = calculate_entry_accessibility(selected_market_data['Entry_Accessibility'])

                base_opportunity_comp = calculate_base_opportunity_score(ai_enhancement_comp, job_growth_normalized_comp, wage_premium_comp, entry_accessibility_comp)
                growth_multiplier_comp = calculate_growth_multiplier(selected_market_data['Job_Postings_t'], selected_market_data['Job_Postings_t_minus_1'])
                regional_multiplier_comp = calculate_regional_multiplier(selected_market_data['Local_Demand_Factor'], selected_market_data['Remote_Work_Factor'])
                hr_score_comp = calculate_systematic_opportunity(base_opportunity_comp, growth_multiplier_comp, regional_multiplier_comp)

                synergy_score_comp = calculate_synergy_percentage(st.session_state.current_user_vr_score, hr_score_comp, alignment_factor_val)
                ai_r_score_comp = calculate_ai_readiness_score(st.session_state.current_user_vr_score, hr_score_comp, synergy_score_comp, alpha=st.session_state.alpha, beta=st.session_state.beta)

                comparison_data.append({'Role': role_name,
                                        'V^R': st.session_state.current_user_vr_score,
                                        'H^R': hr_score_comp,
                                        'Synergy': synergy_score_comp,
                                        'AI-R': ai_r_score_comp})
            
            comparison_df = pd.DataFrame(comparison_data)
            st.session_state.comparison_df = comparison_df # Store for potential future use or display
            st.plotly_chart(plot_comparison_chart(comparison_df), use_container_width=True)
    
    st.divider()

    st.subheader("Underlying Data")
    st.markdown("Here are the raw dataframes used in the application:")
    st.markdown("**Sample User Profile:**")
    st.dataframe(st.session_state.user_profile.to_frame().T)
    st.markdown("**Market Opportunities Data:**")
    st.dataframe(st.session_state.market_opportunities_df)
    st.markdown("**Learning Pathways Data:**")
    st.dataframe(st.session_state.learning_pathways_df)


