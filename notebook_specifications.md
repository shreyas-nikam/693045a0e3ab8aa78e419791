
# Technical Specification for Jupyter Notebook: AI Career Navigator for Financial Professionals

## 1. Notebook Overview

This Jupyter Notebook, "AI Career Navigator for Financial Professionals," is designed to provide a comprehensive and interactive tool for financial professionals to assess their AI-Readiness (AI-R) and explore career opportunities in the evolving landscape of AI-transformed finance.

**Learning Goals:**
1.  **Understand the components of the AI-Readiness Score ($AI-R$).** Users will grasp how individual capabilities and market opportunities contribute to their overall readiness.
2.  **Evaluate their current $V^R$ (Idiosyncratic Readiness) across AI-Fluency, Domain-Expertise, and Adaptive-Capacity.** The notebook will guide users through an assessment of their personal skills and attributes.
3.  **Analyze $H^R$ (Systematic Opportunity) for different financial AI roles based on market data.** Users will understand how market demand, growth, and accessibility shape career prospects.
4.  **Identify personalized learning pathways to enhance their $AI-R$ and career prospects.** The notebook will enable "what-if" scenario planning for skill development.
5.  **Grasp the synergistic effects of individual readiness and market opportunity.** Users will observe how alignment between their skills and market needs amplifies their $AI-R$.

**Who the notebook is targeted to:**
This notebook is specifically tailored for financial professionals, including CFA/PRM/FRM charter holders, aspiring Quants, AI Risk Analysts, Algorithmic Traders, Financial Data Engineers, and Portfolio Managers. It assumes a basic understanding of financial concepts and data analysis principles, but will comprehensively explain AI-related metrics and methodologies.

## 2. Code Requirements

### List of Expected Libraries

*   **Data Manipulation:** `pandas`
*   **Numerical Operations:** `numpy`
*   **Interactive Widgets:** `ipywidgets`
*   **Visualization:**
    *   `matplotlib.pyplot` (for static plots, e.g., radar chart base)
    *   `seaborn` (for enhanced statistical plots, e.g., bar charts)
    *   `plotly.express` (for interactive bar charts and comparison plots)
    *   `plotly.graph_objects` (for advanced interactive plots, e.g., radar charts if `matplotlib` isn't interactive enough)
    *   `mpl_toolkits.axisartist.angle_helper` (if using `matplotlib` for radar charts)

### List of Algorithms or Functions to be Implemented

All mathematical formulas provided in the input context will be implemented as Python functions.

1.  `generate_synthetic_user_profile(num_profiles=1)`: Generates a pandas DataFrame for user profiles with specific columns and realistic ranges.
    *   Columns: `User_ID`, `AI_Technical_Skills`, `AI_Augmented_Productivity`, `AI_Critical_Judgment`, `AI_Learning_Velocity`, `Education_Level`, `Years_Experience`, `Portfolio_Score`, `Recognition_Score`, `Credentials_Score`, `Cognitive_Flexibility`, `Social_Emotional_Intelligence`, `Strategic_Career_Management`, `Skills_Match_Score`.
2.  `generate_synthetic_market_data()`: Generates a pandas DataFrame with market opportunity data for several financial roles.
    *   Columns: `Role`, `AI_Enhancement_Potential`, `Job_Growth_Projection`, `Wage_Premium`, `Entry_Accessibility`, `Job_Postings_t`, `Job_Postings_t_minus_1`, `Local_Demand_Factor`, `Remote_Work_Factor`.
3.  `generate_synthetic_learning_pathways()`: Generates a pandas DataFrame for predefined learning pathways.
    *   Columns: `Pathway_ID`, `Pathway_Name`, `Pathway_Type`, `Impact_Coefficient_Delta`, `Estimated_Completion_Time_Hours`.
4.  `calculate_ai_enhancement(automation_tasks, ai_augmentation_tasks)`: Implements the AI-Enhancement potential formula (Eq 6).
5.  `calculate_job_growth_normalized(growth_rate)`: Implements the normalized job growth formula (Eq 8).
6.  `calculate_wage_premium(ai_skilled_wage, median_wage)`: Implements the wage premium formula (Eq 9).
7.  `calculate_entry_accessibility(education_years_required, experience_years_required)`: Implements the entry accessibility formula (Eq 10).
8.  `calculate_base_opportunity_score(ai_enhancement, job_growth_normalized, wage_premium, entry_accessibility)`: Implements the base opportunity score formula (Eq 5).
9.  `calculate_growth_multiplier(job_postings_t, job_postings_t_minus_1, lambda_param=0.3)`: Implements the growth multiplier formula (Eq 13).
10. `calculate_regional_multiplier(local_demand_factor, remote_work_factor, gamma_param=0.2)`: Implements the regional multiplier formula (Eq 14).
11. `calculate_systematic_opportunity(base_opportunity_score, growth_multiplier, regional_multiplier)`: Implements the systematic opportunity formula (Eq 4).
12. `calculate_technical_ai_skills(prompting, tools, understanding, data_lit)`: Implements the Technical AI Skills sub-component formula (Eq 17).
13. `calculate_ai_augmented_productivity(output_quality_with_ai, output_quality_without_ai, time_without_ai, time_with_ai)`: Implements the AI-Augmented Productivity sub-component formula (Eq 18).
14. `calculate_critical_ai_judgment(errors_caught, total_ai_errors, appropriate_trust_decisions, total_decisions)`: Implements the Critical AI Judgment sub-component formula (Eq 19).
15. `calculate_ai_learning_velocity(delta_proficiency, delta_t, hours_invested)`: Implements the AI Learning Velocity sub-component formula (Eq 20).
16. `calculate_ai_fluency(technical_ai_skills, ai_augmented_productivity, critical_ai_judgment, ai_learning_velocity)`: Implements the AI-Fluency factor formula (Eq 16).
17. `calculate_educational_foundation(education_level_str)`: Maps education level string to a numerical score (Eq 22).
18. `calculate_practical_experience(years_experience, gamma_exp=0.15)`: Implements the practical experience formula (Eq 23).
19. `calculate_specialization_depth(portfolio_score, recognition_score, credentials_score)`: Implements the specialization depth formula (Eq 24).
20. `calculate_domain_expertise(educational_foundation, practical_experience, specialization_depth)`: Implements the Domain-Expertise factor formula (Eq 21).
21. `calculate_adaptive_capacity(cognitive_flexibility, social_emotional_intelligence, strategic_career_management)`: Implements the Adaptive-Capacity factor formula (Eq 25).
22. `calculate_idiosyncratic_readiness(ai_fluency, domain_expertise, adaptive_capacity)`: Implements the idiosyncratic readiness formula (Eq 15).
23. `calculate_skills_match_score(individual_skill_levels, required_skill_levels, importance_weights)`: Implements the skills match score formula (Eq 28).
24. `calculate_timing_factor(years_experience)`: Implements the timing factor function (Eq 29).
25. `calculate_alignment_factor(skills_match_score, max_possible_match_score, timing_factor)`: Combines skills match and timing factor (Eq 27).
26. `calculate_synergy_percentage(v_r_score, h_r_score, alignment_factor)`: Implements the synergy function (Eq 26).
27. `calculate_ai_readiness_score(v_r_score, h_r_score, synergy_percentage, alpha, beta)`: Implements the main AI-Readiness Score formula (Eq 1).
28. `update_ai_readiness_dynamic(current_ai_r, pathways_undertaken, learning_pathways_df)`: Implements the dynamic update formula (Eq 34).
29. Visualization functions:
    *   `plot_vr_radar_chart(ai_fluency, domain_expertise, adaptive_capacity)`: Generates an interactive radar chart.
    *   `plot_hr_bar_chart(ai_enhancement, job_growth, wage_premium, entry_accessibility)`: Generates an interactive bar chart.
    *   `plot_career_comparison(comparison_df)`: Generates a bar chart comparing $AI-R$, $V^R$, and $H^R$ for different roles.

### Visualization like charts, tables, plots that should be generated

1.  **Radar Chart for $V^R$ Components:** Visualizing AI-Fluency, Domain-Expertise, and Adaptive-Capacity scores for a user.
2.  **Bar Chart for $H^R$ Components:** Displaying contributions of AI-Enhancement, Job Growth, Wage Premium, and Entry Accessibility to the `Base_Opportunity_Score` for a selected role.
3.  **Comparison Bar Charts/Tables:**
    *   Comparing $AI-R$, $V^R$, and $H^R$ for multiple target financial roles.
    *   Visualizing the change in $AI-R$ after applying learning pathways.
4.  **Interactive Tables/Displays:** To show synthetic data, user inputs, and calculated scores.

## 3. Notebook Sections (in detail)

The notebook will be structured into the following 15 sections. Each section will follow the Markdown, Code (function definition), Code (function execution), Markdown (explanation) pattern.

---

### Section 1: Introduction to AI Career Navigator

*   **Markdown Cell:**
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

---

### Section 2: Setting Up the Environment and Global Parameters

*   **Markdown Cell:**
    ```markdown
    ## 2. Setting Up the Environment and Global Parameters

    First, we import the necessary Python libraries and define the global parameters (weights and coefficients) that will be used throughout our calculations. These parameters are derived from the research framework and represent the relative importance of various factors.

    The main formula for the AI-Readiness Score is:
    $$AI-R_{i,t} = \alpha \cdot V^R_i(t) + (1 - \alpha) \cdot H^R(t) + \beta \cdot Synergy\%(V^R, H^R)$$
    Where:
    *   $\alpha \in [0,1]$: Weight on individual vs. market factors. (Prior: $\alpha \in [0.5, 0.7]$)
    *   $\beta > 0$: Synergy coefficient. (Prior: $\beta \in [0.05, 0.20]$)

    Initial weights for $V^R$ components are: $w_1 = 0.45$ for AI-Fluency, $w_2 = 0.35$ for Domain-Expertise, $w_3 = 0.20$ for Adaptive-Capacity.
    Initial weights for $H_{base}$ components are: $w_1 = 0.30$ for AI-Enhancement, $w_2 = 0.30$ for Job Growth, $w_3 = 0.25$ for Wage Premium, $w_4 = 0.15$ for Entry Accessibility.
    ```

*   **Code Cell (Function Definition):** (No function definition here, just library imports and parameter definitions)
    ```python
    # Code cell for importing libraries and defining global parameters
    ```

*   **Code Cell (Function Execution):**
    ```python
    import pandas as pd
    import numpy as np
    import ipywidgets as widgets
    from IPython.display import display, clear_output
    import matplotlib.pyplot as plt
    import seaborn as sns
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

    print("Libraries imported and global parameters defined.")
    ```

*   **Markdown Cell:**
    ```markdown
    The essential libraries for data manipulation, numerical computing, interactive elements, and visualization have been imported. The global parameters $\alpha$, $\beta$, and various weights ($w_x$, $\theta_y$) are set to their default values as specified in the framework, allowing for consistent calculations throughout the notebook. These parameters can be interactively adjusted in a later section.
    ```

---

### Section 3: Generating Synthetic User Profile Data

*   **Markdown Cell:**
    ```markdown
    ## 3. Generating Synthetic User Profile Data

    To make this notebook interactive and allow for consistent examples, we will generate a synthetic user profile. This profile simulates the various skill and experience levels that contribute to the **Idiosyncratic Readiness ($V^R$)** score.

    The user profile includes:
    *   **AI-Fluency Sub-components:** Technical AI Skills, AI-Augmented Productivity, Critical AI Judgment, AI Learning Velocity (all scored [0,1]).
    *   **Domain-Expertise Factors:** Education Level, Years of Experience, Portfolio, Recognition, Credentials.
    *   **Adaptive-Capacity Sub-components:** Cognitive Flexibility, Social-Emotional Intelligence, Strategic Career Management (all scored [0,100]).
    *   **Skills Match Score:** A hypothetical initial score reflecting general alignment to financial AI roles.

    These values will serve as inputs for calculating $V^R$.
    ```

*   **Code Cell (Function Definition):**
    ```python
    def generate_synthetic_user_profile():
        """
        Generates a synthetic user profile as a pandas Series.
        Scores are normalized to [0,1] or [0,100] as appropriate.
        """
        user_profile = {
            'User_ID': 'FinancialPro001',
            # AI-Fluency sub-components (0-1 scale)
            'AI_Technical_Skills_Input': np.random.uniform(0.6, 0.9),
            'AI_Augmented_Productivity_Input': np.random.uniform(0.5, 0.8),
            'AI_Critical_Judgment_Input': np.random.uniform(0.7, 0.9),
            'AI_Learning_Velocity_Input': np.random.uniform(0.6, 0.85),
            # Domain-Expertise factors
            'Education_Level': np.random.choice(['Master\'s in target field', 'PhD in target field', 'Bachelor\'s in target field']),
            'Years_Experience': np.random.randint(5, 15),
            'Portfolio_Score': np.random.uniform(0.6, 0.9), # Normalized [0,1]
            'Recognition_Score': np.random.uniform(0.5, 0.8), # Normalized [0,1]
            'Credentials_Score': np.random.uniform(0.7, 0.95), # Normalized [0,1]
            # Adaptive-Capacity sub-components (0-100 scale)
            'Cognitive_Flexibility': np.random.uniform(60, 90),
            'Social_Emotional_Intelligence': np.random.uniform(55, 85),
            'Strategic_Career_Management': np.random.uniform(70, 95),
            # General Skills Match Score for initial Alignment Factor (0-1 scale)
            'Skills_Match_Score': np.random.uniform(0.7, 0.9)
        }
        return pd.Series(user_profile)
    ```

*   **Code Cell (Function Execution):**
    ```python
    sample_user_profile = generate_synthetic_user_profile()
    print("Generated Sample User Profile:")
    print(sample_user_profile.to_string())
    ```

*   **Markdown Cell:**
    ```markdown
    Above is a sample synthetic user profile. For this demonstration, we assume specific values for a typical financial professional. In a real application, these would be derived from user inputs or assessments. This profile will be used to calculate the user's $V^R$.
    ```

---

### Section 4: Generating Synthetic Market Opportunity Data

*   **Markdown Cell:**
    ```markdown
    ## 4. Generating Synthetic Market Opportunity Data

    The **Systematic Opportunity ($H^R$)** component relies on macro-level market data for various financial roles. We will generate synthetic market data for several target roles relevant to financial professionals, such as 'Quant Researcher' and 'Financial Data Engineer'.

    This market data includes:
    *   **AI-Enhancement Potential:** How much AI augments rather than replaces tasks (0-1 scale).
    *   **Job Growth Projections:** Normalized 10-year BLS outlook (0-100 scale).
    *   **Wage Premium:** Compensation potential for AI-skilled roles (0-1 scale).
    *   **Entry Accessibility:** Ease of transition into the role (0-1 scale).
    *   **Dynamic Multipliers:** Job posting volumes and regional demand factors for temporal and geographic adjustments.

    This data will serve as inputs for calculating $H^R$ for each role.
    ```

*   **Code Cell (Function Definition):**
    ```python
    def generate_synthetic_market_data():
        """
        Generates synthetic market data for predefined financial AI roles.
        """
        market_data = {
            'Role': ['Quant Researcher', 'AI Risk Analyst', 'Algorithmic Trader', 'Financial Data Engineer', 'Portfolio Manager'],
            'AI_Enhancement_Potential': [0.85, 0.80, 0.90, 0.88, 0.75], # 0-1 scale
            'Job_Growth_Projection': [80, 75, 70, 85, 60], # 0-100 normalized
            'Wage_Premium': [0.90, 0.85, 0.95, 0.88, 0.70], # 0-1 scale
            'Entry_Accessibility': [0.60, 0.70, 0.55, 0.65, 0.75], # 0-1 scale
            'Job_Postings_t': [1200, 950, 800, 1500, 700], # Current job postings
            'Job_Postings_t_minus_1': [1100, 900, 820, 1350, 720], # Previous period job postings
            'Local_Demand_Factor': [1.1, 1.05, 1.2, 1.15, 0.95], # Relative to national avg
            'Remote_Work_Factor': [0.7, 0.8, 0.6, 0.9, 0.5] # 0-1 scale
        }
        return pd.DataFrame(market_data)
    ```

*   **Code Cell (Function Execution):**
    ```python
    market_opportunities_df = generate_synthetic_market_data()
    print("Generated Market Opportunities Data:")
    print(market_opportunities_df.to_string())
    ```

*   **Markdown Cell:**
    ```markdown
    This table provides synthetic market data for several key AI-enabled financial roles. Each row represents a target occupation, with metrics that will be used to calculate its **Systematic Opportunity ($H^R$)**. This data allows us to compare the attractiveness of different career paths.
    ```

---

### Section 5: Calculating Idiosyncratic Readiness ($V^R$) Components

*   **Markdown Cell:**
    ```markdown
    ## 5. Calculating Idiosyncratic Readiness ($V^R$) Components

    **Idiosyncratic Readiness ($V^R$)** measures an individual's specific preparation for AI-enabled careers. Unlike market factors, $V^R$ can be directly improved through deliberate learning. It is composed of three main factors: AI-Fluency, Domain-Expertise, and Adaptive-Capacity.

    The formula for $V^R$ is:
    $$V^R(t) = w_1 \cdot AI\text{-}Fluency_i(t) + w_2 \cdot Domain\text{-}Expertise_i(t) + w_3 \cdot Adaptive\text{-}Capacity_i(t)$$
    Where $w_1 = 0.45$, $w_2 = 0.35$, $w_3 = 0.20$.

    Each of these factors is further broken down into sub-components.

    ### AI-Fluency Factor
    AI-Fluency measures the ability to effectively use, understand, and collaborate with AI systems.
    $$AI\text{-}Fluency_i = \sum_{k=1}^{4} \theta_k \cdot S_{i,k}$$
    Sub-components ($S_{i,k}$ scores from 0-1, $\theta_k$ weights):
    1.  **Technical AI Skills ($\theta_1 = 0.30$):**
        $$S_{i,1} = \frac{1}{4} (Prompting_i + Tools_i + Understanding_i + DataLit_i)$$
    2.  **AI-Augmented Productivity ($\theta_2 = 0.35$):**
        $$S_{i,2} = \frac{Output\ Quality_{i,with\ AI}}{Output\ Quality_{i,without\ AI}} \cdot \frac{Time_{i,without\ AI}}{Time_{i,with\ AI}}$$
        (For simplicity, we use a single input score [0-1] representing this ratio, where 1 means perfect augmentation.)
    3.  **Critical AI Judgment ($\theta_3 = 0.20$):**
        $$S_{i,3} = \frac{Errors\ Caught_i}{Total\ AI\ Errors} + \frac{Appropriate\ Trust\ Decisions_i}{Total\ Decisions}$$
        (For simplicity, we use a single input score [0-1] representing overall judgment.)
    4.  **AI Learning Velocity ($\theta_4 = 0.15$):**
        $$S_{i,4} = \frac{\Delta Proficiency_i}{\Delta t} \cdot \frac{1}{Hours\ Invested}$$
        (For simplicity, we use a single input score [0-1] representing velocity.)

    ### Domain-Expertise Factor
    Domain-Expertise captures depth of knowledge in specific application areas.
    $$Domain\text{-}Expertise_i = E_{education} \cdot E_{experience} \cdot E_{specialization}$$
    1.  **Educational Foundation ($E_{education}$):**
        $$
        E_{education} =
        \begin{cases}
            1.0 & \text{PhD in target field} \\
            0.85 & \text{Master's in target field} \\
            0.70 & \text{Bachelor's in target field} \\
            0.60 & \text{Associate's/Certificate} \\
            0.50 & \text{HS + significant coursework}
        \end{cases}
        $$
    2.  **Practical Experience ($E_{experience}$):**
        $$E_{experience} = 1 - e^{-\gamma Years}$$
        Where $\gamma = 0.15$.
    3.  **Specialization Depth ($E_{specialization}$):**
        $$E_{specialization} = 0.4 \cdot Portfolio_i + 0.3 \cdot Recognition_i + 0.3 \cdot Credentials_i$$

    ### Adaptive-Capacity Factor
    Adaptive-Capacity measures meta-skills enabling successful navigation of AI-driven transitions.
    $$Adaptive\text{-}Capacity_i = \frac{1}{3} (C_{cognitive} + C_{social} + C_{strategic})$$
    (Each $C$ component is scored [0-100]).
    ```

*   **Code Cell (Function Definition):**
    ```python
    # AI-Fluency sub-component functions
    def calculate_technical_ai_skills(prompting, tools, understanding, data_lit):
        return (prompting + tools + understanding + data_lit) / 4

    def calculate_ai_augmented_productivity(productivity_score): # Simplified input as a single score
        return productivity_score

    def calculate_critical_ai_judgment(judgment_score): # Simplified input as a single score
        return judgment_score

    def calculate_ai_learning_velocity(velocity_score): # Simplified input as a single score
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
        return mapping.get(education_level_str, 0.0) # Default to 0 if not found

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
    ```

*   **Code Cell (Function Execution):**
    ```python
    # Calculate AI-Fluency components
    ai_tech_skills = calculate_technical_ai_skills(sample_user_profile['AI_Technical_Skills_Input'],
                                                   sample_user_profile['AI_Technical_Skills_Input'], # Using same for all 4 parts
                                                   sample_user_profile['AI_Technical_Skills_Input'],
                                                   sample_user_profile['AI_Technical_Skills_Input'])
    ai_augmented_prod = calculate_ai_augmented_productivity(sample_user_profile['AI_Augmented_Productivity_Input'])
    critical_ai_judg = calculate_critical_ai_judgment(sample_user_profile['AI_Critical_Judgment_Input'])
    ai_learning_vel = calculate_ai_learning_velocity(sample_user_profile['AI_Learning_Velocity_Input'])

    ai_fluency_score = calculate_ai_fluency(ai_tech_skills, ai_augmented_prod, critical_ai_judg, ai_learning_vel)

    # Calculate Domain-Expertise components
    edu_foundation = calculate_educational_foundation(sample_user_profile['Education_Level'])
    prac_experience = calculate_practical_experience(sample_user_profile['Years_Experience'])
    spec_depth = calculate_specialization_depth(sample_user_profile['Portfolio_Score'],
                                                sample_user_profile['Recognition_Score'],
                                                sample_user_profile['Credentials_Score'])
    domain_expertise_score = calculate_domain_expertise(edu_foundation, prac_experience, spec_depth)

    # Calculate Adaptive-Capacity
    adaptive_capacity_score = calculate_adaptive_capacity(sample_user_profile['Cognitive_Flexibility'],
                                                          sample_user_profile['Social_Emotional_Intelligence'],
                                                          sample_user_profile['Strategic_Career_Management'])

    # Calculate overall V_R
    user_vr_score = calculate_idiosyncratic_readiness(ai_fluency_score, domain_expertise_score, adaptive_capacity_score)

    print(f"Calculated AI-Fluency Score: {ai_fluency_score:.2f}")
    print(f"Calculated Domain-Expertise Score: {domain_expertise_score:.2f}")
    print(f"Calculated Adaptive-Capacity Score: {adaptive_capacity_score:.2f}")
    print(f"\nOverall Idiosyncratic Readiness (V^R) Score: {user_vr_score:.2f}")

    # Store component scores for visualization
    vr_components = {
        'AI-Fluency': ai_fluency_score * 100, # Scale to 0-100 for better visualization
        'Domain-Expertise': domain_expertise_score * 100,
        'Adaptive-Capacity': adaptive_capacity_score
    }
    ```

*   **Markdown Cell:**
    ```markdown
    The individual components for the synthetic user's $V^R$ have been calculated. We see the scores for AI-Fluency, Domain-Expertise, and Adaptive-Capacity, which are then combined using their respective weights to derive the overall Idiosyncratic Readiness score. This score reflects the individual's current capability to thrive in an AI-enabled financial role.
    ```

---

### Section 6: Visualizing Idiosyncratic Readiness ($V^R$) Breakdown

*   **Markdown Cell:**
    ```markdown
    ## 6. Visualizing Idiosyncratic Readiness ($V^R$) Breakdown

    To better understand the strengths and weaknesses contributing to an individual's $V^R$, we will visualize the scores of its three main components: AI-Fluency, Domain-Expertise, and Adaptive-Capacity, using a radar chart. This visual representation helps identify areas for improvement.
    ```

*   **Code Cell (Function Definition):**
    ```python
    def plot_vr_radar_chart(ai_fluency_val, domain_expertise_val, adaptive_capacity_val):
        """
        Generates an interactive radar chart for V^R components.
        Inputs should be scaled to a common range (e.g., 0-100).
        """
        categories = ['AI-Fluency', 'Domain-Expertise', 'Adaptive-Capacity']
        values = [ai_fluency_val, domain_expertise_val, adaptive_capacity_val]

        fig = go.Figure(data=go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Idiosyncratic Readiness Components'
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=True,
            title_text="Idiosyncratic Readiness (V^R) Component Breakdown (Scaled to 100)"
        )
        fig.show()
    ```

*   **Code Cell (Function Execution):**
    ```python
    # Note: vr_components stores scaled scores (AI-Fluency and Domain-Expertise were 0-1, scaled to 0-100 here)
    plot_vr_radar_chart(vr_components['AI-Fluency'], vr_components['Domain-Expertise'], vr_components['Adaptive-Capacity'])
    ```

*   **Markdown Cell:**
    ```markdown
    The radar chart provides a clear visual breakdown of the synthetic user's Idiosyncratic Readiness across its three core dimensions. A larger area covered by the polygon indicates higher overall readiness, while specific spikes or dips highlight areas of strength or weakness. This visualization is crucial for identifying targeted learning pathways.
    ```

---

### Section 7: Calculating Systematic Opportunity ($H^R$) Components

*   **Markdown Cell:**
    ```markdown
    ## 7. Calculating Systematic Opportunity ($H^R$) Components

    **Systematic Opportunity ($H^R$)** represents the macro-level demand and growth potential in AI-enabled occupations. It's a market-driven component that individuals position themselves to capture, rather than create.

    The main formula for $H^R$ is:
    $$H^R(t) = H_{base}(O_{target}) \cdot M_{growth}(t) \cdot M_{regional}(t)$$

    ### Base Opportunity Score ($H_{base}(o)$)
    The base opportunity score aggregates multiple dimensions of occupational attractiveness for a target role $o$.
    $$H_{base}(o) = \sum_{j \in J_o} w_j \cdot AI\text{-}Enhancement_j \cdot Growth_j$$
    The factors and their weights ($w_j$):
    1.  **AI-Enhancement Potential ($w_1 = 0.30$):** How much AI augments rather than replaces tasks.
        $$AI\text{-}Enhancement_o = \frac{1}{|T_o|} \sum_{t \in T_o} (1 - Automation_t) \cdot AI\text{-}Augmentation_t$$
        (For simplicity, we use a single input score [0-1] for each role representing this.)
    2.  **Job Growth Projections ($w_2 = 0.30$):** BLS 10-year outlook, normalized to [0,100].
        $$Growth_{normalized} = \frac{g + 0.5}{2.0} \times 100$$
        Where $g$ is the raw growth rate. (For simplicity, we use a single input score [0-100] directly).
    3.  **Wage Premium ($w_3 = 0.25$):** Compensation potential for AI-skilled roles (0-1 scale).
        $$Wage_o = \frac{AI\text{-}skilled\ wage_o - median\ wage_o}{median\ wage_o}$$
        (For simplicity, we use a single input score [0-1] directly).
    4.  **Entry Accessibility ($w_4 = 0.15$):** Ease of transition into role (0-1 scale).
        $$Access_o = 1 - \frac{Education\ Years\ Required + Experience\ Years\ Required}{10}$$
        (For simplicity, we use a single input score [0-1] directly).

    ### Dynamic Multipliers
    The base score is modulated by time-varying factors.
    1.  **Growth Multiplier ($M_{growth}(t)$):** Captures market momentum.
        $$M_{growth}(t) = 1 + \lambda \cdot \left( \frac{Job\ Postingso,t}{Job\ Postingso,t-1} - 1 \right)$$
        Where $\lambda = 0.3$.
    2.  **Regional Multiplier ($M_{regional}(t)$):** Adjusts for local labor markets.
        $$M_{regional}(t) = \frac{Local\ Demand_{i,t}}{National\ Avg\ Demand} \times (1 + \gamma \cdot Remote\ Work\ Factor_o)$$
        Where $\gamma = 0.2$.
    ```

*   **Code Cell (Function Definition):**
    ```python
    # H_R component functions
    def calculate_ai_enhancement(enhancement_score): # Simplified as direct input
        return enhancement_score

    def calculate_job_growth_normalized(growth_projection_score): # Simplified as direct input (0-100)
        return growth_projection_score / 100.0 # Normalize to 0-1 for formula

    def calculate_wage_premium(wage_premium_score): # Simplified as direct input (0-1)
        return wage_premium_score

    def calculate_entry_accessibility(accessibility_score): # Simplified as direct input (0-1)
        return accessibility_score

    def calculate_base_opportunity_score(ai_enhancement, job_growth_normalized, wage_premium, entry_accessibility):
        return (W1_AI_ENHANCEMENT * ai_enhancement +
                W2_JOB_GROWTH * job_growth_normalized +
                W3_WAGE_PREMIUM * wage_premium +
                W4_ENTRY_ACCESSIBILITY * entry_accessibility) * 100 # Scale to 0-100

    def calculate_growth_multiplier(job_postings_t, job_postings_t_minus_1, lambda_param=LAMBDA_GROWTH_MULTIPLIER):
        if job_postings_t_minus_1 == 0: return 1.0 # Avoid division by zero
        return 1 + lambda_param * ((job_postings_t / job_postings_t_minus_1) - 1)

    def calculate_regional_multiplier(local_demand_factor, remote_work_factor, gamma_param=GAMMA_REGIONAL_MULTIPLIER):
        return local_demand_factor * (1 + gamma_param * remote_work_factor)

    def calculate_systematic_opportunity(base_opportunity_score, growth_multiplier, regional_multiplier):
        return min(max(base_opportunity_score * growth_multiplier * regional_multiplier, 0), 100) # Cap at 0-100
    ```

*   **Code Cell (Function Execution):**
    ```python
    # Let's select 'Financial Data Engineer' as the target role for initial H_R calculation
    target_role_data = market_opportunities_df[market_opportunities_df['Role'] == 'Financial Data Engineer'].iloc[0]

    # Calculate Base Opportunity Score components
    ai_enhancement = calculate_ai_enhancement(target_role_data['AI_Enhancement_Potential'])
    job_growth_normalized = calculate_job_growth_normalized(target_role_data['Job_Growth_Projection'])
    wage_premium = calculate_wage_premium(target_role_data['Wage_Premium'])
    entry_accessibility = calculate_entry_accessibility(target_role_data['Entry_Accessibility'])

    base_opportunity_score = calculate_base_opportunity_score(ai_enhancement, job_growth_normalized, wage_premium, entry_accessibility)

    # Calculate Dynamic Multipliers
    growth_multiplier = calculate_growth_multiplier(target_role_data['Job_Postings_t'], target_role_data['Job_Postings_t_minus_1'])
    regional_multiplier = calculate_regional_multiplier(target_role_data['Local_Demand_Factor'], target_role_data['Remote_Work_Factor'])

    # Calculate overall H_R
    target_hr_score = calculate_systematic_opportunity(base_opportunity_score, growth_multiplier, regional_multiplier)

    print(f"Target Role: {target_role_data['Role']}")
    print(f"  AI-Enhancement Potential: {ai_enhancement:.2f}")
    print(f"  Job Growth (Normalized): {job_growth_normalized * 100:.2f}")
    print(f"  Wage Premium: {wage_premium:.2f}")
    print(f"  Entry Accessibility: {entry_accessibility:.2f}")
    print(f"  Base Opportunity Score: {base_opportunity_score:.2f}")
    print(f"  Growth Multiplier: {growth_multiplier:.2f}")
    print(f"  Regional Multiplier: {regional_multiplier:.2f}")
    print(f"\nOverall Systematic Opportunity (H^R) Score: {target_hr_score:.2f}")

    # Store component scores for visualization
    hr_components = {
        'AI-Enhancement': ai_enhancement * W1_AI_ENHANCEMENT * 100,
        'Job Growth': job_growth_normalized * W2_JOB_GROWTH * 100,
        'Wage Premium': wage_premium * W3_WAGE_PREMIUM * 100,
        'Entry Accessibility': entry_accessibility * W4_ENTRY_ACCESSIBILITY * 100
    }
    ```

*   **Markdown Cell:**
    ```markdown
    The Systematic Opportunity ($H^R$) for the 'Financial Data Engineer' role has been calculated. We've first determined the Base Opportunity Score from static market attributes, then adjusted it using dynamic growth and regional factors. The overall $H^R$ reflects the current attractiveness and potential of this role in the market.
    ```

---

### Section 8: Visualizing Systematic Opportunity ($H^R$) Breakdown

*   **Markdown Cell:**
    ```markdown
    ## 8. Visualizing Systematic Opportunity ($H^R$) Breakdown

    Similar to $V^R$, visualizing the components of $H^R$ provides insights into which market factors are most influential for a given role. We will use a bar chart to display the contribution of each component to the Base Opportunity Score.
    ```

*   **Code Cell (Function Definition):**
    ```python
    def plot_hr_bar_chart(component_contributions, title="Systematic Opportunity (H^R) Component Breakdown"):
        """
        Generates an interactive bar chart for H^R component contributions.
        Inputs should be the weighted contributions scaled to a common range (e.g., 0-100).
        """
        df_hr = pd.DataFrame(list(component_contributions.items()), columns=['Component', 'Contribution'])

        fig = px.bar(df_hr, x='Component', y='Contribution',
                     title=title,
                     labels={'Contribution': 'Weighted Contribution to Base H^R (0-100)'},
                     color='Contribution',
                     color_continuous_scale=px.colors.sequential.Plasma)
        fig.update_layout(yaxis_range=[0, max(df_hr['Contribution'].max() + 10, 100)])
        fig.show()
    ```

*   **Code Cell (Function Execution):**
    ```python
    plot_hr_bar_chart(hr_components, title=f"Systematic Opportunity (H^R) Component Breakdown for {target_role_data['Role']}")
    ```

*   **Markdown Cell:**
    ```markdown
    This bar chart illustrates the weighted contributions of AI-Enhancement, Job Growth, Wage Premium, and Entry Accessibility to the target role's Base Systematic Opportunity. It clearly shows which market aspects make a career path more or less appealing, helping financial professionals identify roles that align with their market preferences.
    ```

---

### Section 9: Calculating Synergy Percentage

*   **Markdown Cell:**
    ```markdown
    ## 9. Calculating Synergy Percentage

    The relationship between individual readiness and market opportunity is not merely additive; it can be multiplicative. The **Synergy Function** captures this compounding benefit when an individual's preparation ($V^R$) aligns with a high-opportunity field ($H^R$) at the right time.

    The synergy term in the $AI-R$ equation is:
    $$Synergy\%(V^R, H^R) = \frac{V^R \times H^R}{100} \times Alignment_i$$
    Both $V^R$ and $H^R$ are normalized to $[0,100]$, and $Alignment_i \in [0,1]$ ensures $Synergy\% \in [0,100]$.

    The **Alignment Factor** measures how well individual skills match occupation requirements and career stage:
    $$Alignment_i = \frac{Skills\ Match\ Score}{Maximum\ Possible\ Match} \times Timing\ Factor$$
    Where both components are bounded to ensure $Alignment_i \in [0,1]$.

    ### Skills Match Score
    Using O*NET task-skill mappings, we compute:
    $$Match_i = \sum_{s \in S} \min(Individual\ Skill_{i,s}, Required\ Skill_{o,s}) \cdot Importance_s$$
    (For simplicity, we use the pre-generated `Skills_Match_Score` from the user profile, representing the overall match for a given role normalized to [0,1].)

    ### Timing Factor
    Career stage affects transition ease:
    $$
    Timing(y) =
    \begin{cases}
        1.0 & y \in [0,5] \text{ (early career)} \\
        1.0 & y \in (5,15] \text{ (mid-career)} \\
        0.8 & y > 15 \text{ (late career, transition friction)}
    \end{cases}
    $$
    Where $y$ is years of experience.
    ```

*   **Code Cell (Function Definition):**
    ```python
    def calculate_skills_match_score(user_skills_match_input): # Simplified as direct input (0-1)
        return user_skills_match_input

    def calculate_timing_factor(years_experience):
        if years_experience <= 5:
            return 1.0 # Early career
        elif 5 < years_experience <= 15:
            return 1.0 # Mid-career
        else:
            return 0.8 # Late career, transition friction

    def calculate_alignment_factor(skills_match_score_val, timing_factor_val):
        # Assuming Max_Possible_Match is integrated into skills_match_score (0-1)
        return skills_match_score_val * timing_factor_val

    def calculate_synergy_percentage(v_r_score, h_r_score, alignment_factor):
        return (v_r_score * h_r_score / 100) * alignment_factor
    ```

*   **Code Cell (Function Execution):**
    ```python
    # Using the calculated V_R and H_R scores
    skills_match_score = calculate_skills_match_score(sample_user_profile['Skills_Match_Score'])
    timing_factor = calculate_timing_factor(sample_user_profile['Years_Experience'])
    alignment_factor = calculate_alignment_factor(skills_match_score, timing_factor)

    synergy_score = calculate_synergy_percentage(user_vr_score, target_hr_score, alignment_factor)

    print(f"User's Years of Experience: {sample_user_profile['Years_Experience']}")
    print(f"Calculated Timing Factor: {timing_factor:.2f}")
    print(f"User's Skills Match Score: {skills_match_score:.2f}")
    print(f"Calculated Alignment Factor: {alignment_factor:.2f}")
    print(f"\nCalculated Synergy Percentage: {synergy_score:.2f}")
    ```

*   **Markdown Cell:**
    ```markdown
    The Synergy Percentage has been calculated by combining the user's Idiosyncratic Readiness ($V^R$), the Systematic Opportunity ($H^R$) of the target role, and an Alignment Factor. The Alignment Factor incorporates both the skills match and career timing, illustrating how well the individual's profile aligns with the market opportunity. A higher synergy indicates a stronger multiplicative benefit.
    ```

---

### Section 10: Calculating the Overall AI-Readiness Score ($AI-R$)

*   **Markdown Cell:**
    ```markdown
    ## 10. Calculating the Overall AI-Readiness Score ($AI-R$)

    Finally, we combine all calculated components—Idiosyncratic Readiness ($V^R$), Systematic Opportunity ($H^R$), and the Synergy Percentage—to determine the overall AI-Readiness Score ($AI-R$). This score provides a comprehensive measure of an individual's career opportunity in the AI era.

    The master formula for $AI-R$ is:
    $$AI-R_{i,t} = \alpha \cdot V^R_i(t) + (1 - \alpha) \cdot H^R(t) + \beta \cdot Synergy\%(V^R, H^R)$$
    Where $\alpha$ and $\beta$ are weighting parameters that determine the relative importance of individual readiness, market opportunity, and their synergy.
    ```

*   **Code Cell (Function Definition):**
    ```python
    def calculate_ai_readiness_score(v_r_score, h_r_score, synergy_percentage, alpha=ALPHA_DEFAULT, beta=BETA_DEFAULT):
        return (alpha * v_r_score +
                (1 - alpha) * h_r_score +
                beta * synergy_percentage)
    ```

*   **Code Cell (Function Execution):**
    ```python
    initial_ai_r_score = calculate_ai_readiness_score(user_vr_score, target_hr_score, synergy_score)

    print(f"Parameters used: Alpha = {ALPHA_DEFAULT}, Beta = {BETA_DEFAULT}")
    print(f"Individual Readiness (V^R): {user_vr_score:.2f}")
    print(f"Systematic Opportunity (H^R): {target_hr_score:.2f}")
    print(f"Synergy Percentage: {synergy_score:.2f}")
    print(f"\nOverall AI-Readiness Score (AI-R): {initial_ai_r_score:.2f}")
    ```

*   **Markdown Cell:**
    ```markdown
    The AI-Readiness Score for the synthetic user targeting 'Financial Data Engineer' has been calculated. This single score summarizes the individual's preparedness, the market's opportunity, and the alignment between them. It serves as a benchmark for understanding career positioning in AI-transformed finance.
    ```

---

### Section 11: Interactive User Profile Input and Initial AI-R Calculation

*   **Markdown Cell:**
    ```markdown
    ## 11. Interactive User Profile Input and Initial AI-R Calculation

    This section allows you to interactively input or adjust the details of a professional profile. Use the widgets below to modify the scores for AI-Fluency sub-components, Domain-Expertise factors, and Adaptive-Capacity. Observe how these changes immediately impact your Idiosyncratic Readiness ($V^R$) and the overall AI-Readiness Score ($AI-R$).
    ```

*   **Code Cell (Function Definition):**
    ```python
    # Function to update and display scores based on interactive inputs
    def update_and_display_scores(
        tech_ai_skills_input, ai_augmented_prod_input, critical_ai_judg_input, ai_learning_vel_input,
        education_level, years_experience, portfolio_score, recognition_score, credentials_score,
        cognitive_flexibility, social_emotional_intelligence, strategic_career_management,
        skills_match_score_input, target_role_selected
    ):
        global current_user_vr_score, current_target_hr_score, current_synergy_score, current_ai_r_score

        # Recalculate V_R based on interactive inputs
        ai_tech_skills = calculate_technical_ai_skills(tech_ai_skills_input, tech_ai_skills_input, tech_ai_skills_input, tech_ai_skills_input)
        ai_augmented_prod = calculate_ai_augmented_productivity(ai_augmented_prod_input)
        critical_ai_judg = calculate_critical_ai_judgment(critical_ai_judg_input)
        ai_learning_vel = calculate_ai_learning_velocity(ai_learning_vel_input)
        current_ai_fluency_score = calculate_ai_fluency(ai_tech_skills, ai_augmented_prod, critical_ai_judg, ai_learning_vel)

        current_edu_foundation = calculate_educational_foundation(education_level)
        current_prac_experience = calculate_practical_experience(years_experience)
        current_spec_depth = calculate_specialization_depth(portfolio_score, recognition_score, credentials_score)
        current_domain_expertise_score = calculate_domain_expertise(current_edu_foundation, current_prac_experience, current_spec_depth)

        current_adaptive_capacity_score = calculate_adaptive_capacity(cognitive_flexibility, social_emotional_intelligence, strategic_career_management)

        current_user_vr_score = calculate_idiosyncratic_readiness(current_ai_fluency_score, current_domain_expertise_score, current_adaptive_capacity_score)

        # Retrieve H_R for selected target role (recalculate if needed, but for now just retrieve)
        selected_role_data = market_opportunities_df[market_opportunities_df['Role'] == target_role_selected].iloc[0]
        # Recalculate H_R in case market data was interactive, but for now it's static
        current_hr_ai_enhancement = calculate_ai_enhancement(selected_role_data['AI_Enhancement_Potential'])
        current_hr_job_growth_normalized = calculate_job_growth_normalized(selected_role_data['Job_Growth_Projection'])
        current_hr_wage_premium = calculate_wage_premium(selected_role_data['Wage_Premium'])
        current_hr_entry_accessibility = calculate_entry_accessibility(selected_role_data['Entry_Accessibility'])
        current_base_opportunity_score = calculate_base_opportunity_score(current_hr_ai_enhancement, current_hr_job_growth_normalized, current_hr_wage_premium, current_hr_entry_accessibility)
        current_growth_multiplier = calculate_growth_multiplier(selected_role_data['Job_Postings_t'], selected_role_data['Job_Postings_t_minus_1'])
        current_regional_multiplier = calculate_regional_multiplier(selected_role_data['Local_Demand_Factor'], selected_role_data['Remote_Work_Factor'])
        current_target_hr_score = calculate_systematic_opportunity(current_base_opportunity_score, current_growth_multiplier, current_regional_multiplier)


        # Recalculate Synergy and AI-R
        current_skills_match_score = calculate_skills_match_score(skills_match_score_input)
        current_timing_factor = calculate_timing_factor(years_experience)
        current_alignment_factor = calculate_alignment_factor(current_skills_match_score, current_timing_factor)
        current_synergy_score = calculate_synergy_percentage(current_user_vr_score, current_target_hr_score, current_alignment_factor)
        current_ai_r_score = calculate_ai_readiness_score(current_user_vr_score, current_target_hr_score, current_synergy_score)

        with output_widget:
            clear_output(wait=True)
            print(f"--- Current Profile & Scores for '{target_role_selected}' ---")
            print(f"AI-Fluency: {current_ai_fluency_score:.2f}")
            print(f"Domain-Expertise: {current_domain_expertise_score:.2f}")
            print(f"Adaptive-Capacity: {current_adaptive_capacity_score:.2f}")
            print(f"V^R (Idiosyncratic Readiness): {current_user_vr_score:.2f}")
            print(f"H^R (Systematic Opportunity): {current_target_hr_score:.2f}")
            print(f"Synergy Percentage: {current_synergy_score:.2f}")
            print(f"Overall AI-Readiness Score (AI-R): {current_ai_r_score:.2f}")

            # Update V_R radar chart
            plot_vr_radar_chart(current_ai_fluency_score * 100, current_domain_expertise_score * 100, current_adaptive_capacity_score)
    ```

*   **Code Cell (Function Execution):**
    ```python
    # Initial values from the sample user profile
    initial_tech_ai_skills = sample_user_profile['AI_Technical_Skills_Input']
    initial_ai_augmented_prod = sample_user_profile['AI_Augmented_Productivity_Input']
    initial_critical_ai_judg = sample_user_profile['AI_Critical_Judgment_Input']
    initial_ai_learning_vel = sample_user_profile['AI_Learning_Velocity_Input']
    initial_education_level = sample_user_profile['Education_Level']
    initial_years_experience = sample_user_profile['Years_Experience']
    initial_portfolio_score = sample_user_profile['Portfolio_Score']
    initial_recognition_score = sample_user_profile['Recognition_Score']
    initial_credentials_score = sample_user_profile['Credentials_Score']
    initial_cognitive_flexibility = sample_user_profile['Cognitive_Flexibility']
    initial_social_emotional_intelligence = sample_user_profile['Social_Emotional_Intelligence']
    initial_strategic_career_management = sample_user_profile['Strategic_Career_Management']
    initial_skills_match_score = sample_user_profile['Skills_Match_Score']
    initial_target_role = market_opportunities_df['Role'].iloc[0] # Default to first role

    # Create ipywidgets
    ai_fluency_group = widgets.VBox([
        widgets.FloatSlider(min=0, max=1, step=0.01, value=initial_tech_ai_skills, description='Tech AI Skills (0-1)'),
        widgets.FloatSlider(min=0, max=1, step=0.01, value=initial_ai_augmented_prod, description='AI Prod (0-1)'),
        widgets.FloatSlider(min=0, max=1, step=0.01, value=initial_critical_ai_judg, description='AI Judg (0-1)'),
        widgets.FloatSlider(min=0, max=1, step=0.01, value=initial_ai_learning_vel, description='AI Velocity (0-1)')
    ], layout=widgets.Layout(border='2px solid lightgray', padding='10px'))

    domain_expertise_group = widgets.VBox([
        widgets.Dropdown(options=list(set(sample_user_profile.index) & set(['PhD in target field', 'Master\'s in target field', 'Bachelor\'s in target field', 'Associate\'s/Certificate', 'HS + significant coursework'])), value=initial_education_level, description='Education'),
        widgets.IntSlider(min=0, max=30, step=1, value=initial_years_experience, description='Years Exp.'),
        widgets.FloatSlider(min=0, max=1, step=0.01, value=initial_portfolio_score, description='Portfolio (0-1)'),
        widgets.FloatSlider(min=0, max=1, step=0.01, value=initial_recognition_score, description='Recognition (0-1)'),
        widgets.FloatSlider(min=0, max=1, step=0.01, value=initial_credentials_score, description='Credentials (0-1)')
    ], layout=widgets.Layout(border='2px solid lightgray', padding='10px'))

    adaptive_capacity_group = widgets.VBox([
        widgets.FloatSlider(min=0, max=100, step=1, value=initial_cognitive_flexibility, description='Cognitive Flex (0-100)'),
        widgets.FloatSlider(min=0, max=100, step=1, value=initial_social_emotional_intelligence, description='Social-EQ (0-100)'),
        widgets.FloatSlider(min=0, max=100, step=1, value=initial_strategic_career_management, description='Strategic CM (0-100)')
    ], layout=widgets.Layout(border='2px solid lightgray', padding='10px'))

    skills_match_widget = widgets.FloatSlider(min=0, max=1, step=0.01, value=initial_skills_match_score, description='Skills Match (0-1)')
    target_role_dropdown = widgets.Dropdown(options=market_opportunities_df['Role'].tolist(), value=initial_target_role, description='Target Role')

    output_widget = widgets.Output()

    # Link widgets to the update function
    interactive_vr_inputs = widgets.interactive_output(update_and_display_scores, {
        'tech_ai_skills_input': ai_fluency_group.children[0],
        'ai_augmented_prod_input': ai_fluency_group.children[1],
        'critical_ai_judg_input': ai_fluency_group.children[2],
        'ai_learning_vel_input': ai_fluency_group.children[3],
        'education_level': domain_expertise_group.children[0],
        'years_experience': domain_expertise_group.children[1],
        'portfolio_score': domain_expertise_group.children[2],
        'recognition_score': domain_expertise_group.children[3],
        'credentials_score': domain_expertise_group.children[4],
        'cognitive_flexibility': adaptive_capacity_group.children[0],
        'social_emotional_intelligence': adaptive_capacity_group.children[1],
        'strategic_career_management': adaptive_capacity_group.children[2],
        'skills_match_score_input': skills_match_widget,
        'target_role_selected': target_role_dropdown
    })

    display(widgets.HBox([widgets.VBox([ai_fluency_group, domain_expertise_group, adaptive_capacity_group]),
                          widgets.VBox([skills_match_widget, target_role_dropdown])]), output_widget)
    ```

*   **Markdown Cell:**
    ```markdown
    By manipulating the sliders and dropdowns, you can simulate different professional profiles and immediately see the recalculated $V^R$, $H^R$, Synergy, and $AI-R$ scores. The updated radar chart provides real-time feedback on your $V^R$ component strengths. This interactive exploration helps in understanding the sensitivity of your AI-Readiness to various personal attributes.
    ```

---

### Section 12: Interactive Parameter Adjustment ($\alpha$ and $\beta$)

*   **Markdown Cell:**
    ```markdown
    ## 12. Interactive Parameter Adjustment ($\alpha$ and $\beta$)

    The parameters $\alpha$ and $\beta$ play a crucial role in weighting the influence of Idiosyncratic Readiness ($V^R$), Systematic Opportunity ($H^R$), and Synergy on the final $AI-R$ score.
    *   $\alpha$: Controls the balance between individual readiness ($V^R$) and market opportunity ($H^R$). A higher $\alpha$ means $V^R$ has more weight.
    *   $\beta$: Determines the impact of the Synergy function. A higher $\beta$ means greater emphasis on the multiplicative benefits of alignment.

    Adjust the sliders below to see how different strategic emphases (e.g., prioritizing individual skill-building versus market alignment) affect the overall $AI-R$ for your current profile.
    ```

*   **Code Cell (Function Definition):**
    ```python
    def update_alpha_beta_display(alpha_val, beta_val):
        global current_ai_r_score_alpha_beta

        current_ai_r_score_alpha_beta = calculate_ai_readiness_score(
            current_user_vr_score, current_target_hr_score, current_synergy_score,
            alpha=alpha_val, beta=beta_val
        )

        with alpha_beta_output:
            clear_output(wait=True)
            print(f"Current Alpha: {alpha_val:.2f}, Current Beta: {beta_val:.2f}")
            print(f"Recalculated AI-Readiness Score (AI-R): {current_ai_r_score_alpha_beta:.2f}")
    ```

*   **Code Cell (Function Execution):**
    ```python
    alpha_slider = widgets.FloatSlider(min=0.0, max=1.0, step=0.05, value=ALPHA_DEFAULT, description='Alpha (α)')
    beta_slider = widgets.FloatSlider(min=0.0, max=0.5, step=0.01, value=BETA_DEFAULT, description='Beta (β)')
    alpha_beta_output = widgets.Output()

    widgets.interactive_output(update_alpha_beta_display, {
        'alpha_val': alpha_slider,
        'beta_val': beta_slider
    })

    display(widgets.VBox([alpha_slider, beta_slider, alpha_beta_output]))
    ```

*   **Markdown Cell:**
    ```markdown
    This interactive section demonstrates the sensitivity of the $AI-R$ to the weighting parameters $\alpha$ and $\beta$. Financial professionals can explore how different strategic perspectives (e.g., focusing more on personal development vs. market timing) would alter their perceived AI-Readiness, aiding in strategic career planning.
    ```

---

### Section 13: "What-If" Scenario: Exploring Learning Pathways

*   **Markdown Cell:**
    ```markdown
    ## 13. "What-If" Scenario: Exploring Learning Pathways

    A key feature of the AI-Readiness framework is its ability to model the impact of learning pathways on your $AI-R$. This allows for proactive career planning.

    The **Dynamic Update** formula calculates the projected change in $AI-R$ after undertaking learning pathways:
    $$AI-R_{i,t+1} = AI-R_{i,t} + \sum_{p \in P} \Delta_p \cdot Completion_p \cdot Mastery_p$$
    Where:
    *   $P$: Set of learning pathways undertaken.
    *   $\Delta_p$: Pre-calibrated impact coefficient for pathway $p$.
    *   $Completion_p \in [0,1]$: Fraction of pathway completed.
    *   $Mastery_p \in [0,1]$: Assessment performance score.

    Select learning pathways and simulate your completion and mastery levels to observe the projected improvement in your $AI-R$.
    ```

*   **Code Cell (Function Definition):**
    ```python
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

    def update_ai_readiness_dynamic(current_ai_r, pathway_id, completion_fraction, mastery_score, learning_pathways_df):
        pathway = learning_pathways_df[learning_pathways_df['Pathway_ID'] == pathway_id].iloc[0]
        delta_p = pathway['Impact_Coefficient_Delta']
        delta_ai_r = delta_p * completion_fraction * mastery_score
        return current_ai_r + delta_ai_r, delta_ai_r

    def simulate_pathway_impact(selected_pathway_id, completion, mastery):
        global projected_ai_r_after_pathway, delta_ai_r_pathway
        if selected_pathway_id == 'None':
            projected_ai_r_after_pathway = current_ai_r_score_alpha_beta if 'current_ai_r_score_alpha_beta' in globals() else initial_ai_r_score
            delta_ai_r_pathway = 0
        else:
            projected_ai_r_after_pathway, delta_ai_r_pathway = update_ai_readiness_dynamic(
                current_ai_r_score_alpha_beta if 'current_ai_r_score_alpha_beta' in globals() else initial_ai_r_score,
                selected_pathway_id, completion, mastery, learning_pathways_df
            )
        with pathway_output:
            clear_output(wait=True)
            print(f"Initial AI-R: {current_ai_r_score_alpha_beta if 'current_ai_r_score_alpha_beta' in globals() else initial_ai_r_score:.2f}")
            print(f"Selected Pathway: {learning_pathways_df[learning_pathways_df['Pathway_ID'] == selected_pathway_id]['Pathway_Name'].iloc[0] if selected_pathway_id != 'None' else 'N/A'}")
            print(f"Completion: {completion:.0%}, Mastery: {mastery:.0%}")
            print(f"Projected AI-R: {projected_ai_r_after_pathway:.2f}")
            print(f"Change in AI-R: {delta_ai_r_pathway:.2f}")
    ```

*   **Code Cell (Function Execution):**
    ```python
    learning_pathways_df = generate_synthetic_learning_pathways()

    pathway_options = ['None'] + learning_pathways_df['Pathway_ID'].tolist()
    pathway_names = ['None'] + learning_pathways_df['Pathway_Name'].tolist()
    pathway_map = dict(zip(pathway_names, pathway_options))

    pathway_dropdown = widgets.Dropdown(options=pathway_map, description='Select Pathway:')
    completion_slider = widgets.FloatSlider(min=0, max=1, step=0.1, value=1.0, description='Completion:')
    mastery_slider = widgets.FloatSlider(min=0, max=1, step=0.1, value=0.8, description='Mastery:')
    pathway_output = widgets.Output()

    widgets.interactive_output(simulate_pathway_impact, {
        'selected_pathway_id': pathway_dropdown,
        'completion': completion_slider,
        'mastery': mastery_slider
    })

    display(widgets.VBox([pathway_dropdown, completion_slider, mastery_slider, pathway_output]))
    print("\nAvailable Learning Pathways:")
    print(learning_pathways_df[['Pathway_Name', 'Pathway_Type', 'Impact_Coefficient_Delta']].to_string())
    ```

*   **Markdown Cell:**
    ```markdown
    By choosing a learning pathway and adjusting the completion and mastery sliders, you can simulate how educational investments translate into a higher $AI-R$. This "what-if" analysis is invaluable for prioritizing learning efforts and visualizing tangible career benefits.
    ```

---

### Section 14: "What-If" Scenario: Career Path Comparison

*   **Markdown Cell:**
    ```markdown
    ## 14. "What-If" Scenario: Career Path Comparison

    One of the most powerful applications of the AI-Readiness framework is to compare multiple target financial roles. This allows you to visualize how your current (or projected) $V^R$ aligns with different market opportunities ($H^R$) and how this impacts your overall $AI-R$ for each path.

    Select up to three target roles to compare their $H^R$, your $V^R$, and the resulting $AI-R$.
    ```

*   **Code Cell (Function Definition):**
    ```python
    def compare_career_paths(roles_to_compare):
        comparison_results = []
        current_ai_r_val = current_ai_r_score_alpha_beta if 'current_ai_r_score_alpha_beta' in globals() else initial_ai_r_score
        current_vr_val = current_user_vr_score if 'current_user_vr_score' in globals() else user_vr_score

        for role_name in roles_to_compare:
            if role_name == 'None': continue

            # Get H_R for the role
            role_data = market_opportunities_df[market_opportunities_df['Role'] == role_name].iloc[0]
            hr_ai_enhancement = calculate_ai_enhancement(role_data['AI_Enhancement_Potential'])
            hr_job_growth_normalized = calculate_job_growth_normalized(role_data['Job_Growth_Projection'])
            hr_wage_premium = calculate_wage_premium(role_data['Wage_Premium'])
            hr_entry_accessibility = calculate_entry_accessibility(role_data['Entry_Accessibility'])
            base_opportunity = calculate_base_opportunity_score(hr_ai_enhancement, hr_job_growth_normalized, hr_wage_premium, hr_entry_accessibility)
            growth_mult = calculate_growth_multiplier(role_data['Job_Postings_t'], role_data['Job_Postings_t_minus_1'])
            regional_mult = calculate_regional_multiplier(role_data['Local_Demand_Factor'], role_data['Remote_Work_Factor'])
            role_hr_score = calculate_systematic_opportunity(base_opportunity, growth_mult, regional_mult)

            # Calculate Alignment Factor (skills match is assumed general for the user's profile across roles, or can be dynamic)
            # For simplicity, we use the skills_match_score from the interactive profile input.
            current_skills_match = skills_match_widget.value if 'skills_match_widget' in globals() else sample_user_profile['Skills_Match_Score']
            current_timing = timing_factor_widget.value if 'timing_factor_widget' in globals() else calculate_timing_factor(sample_user_profile['Years_Experience'])
            alignment_factor_for_role = calculate_alignment_factor(current_skills_match, current_timing)
            synergy_for_role = calculate_synergy_percentage(current_vr_val, role_hr_score, alignment_factor_for_role)

            ai_r_for_role = calculate_ai_readiness_score(
                current_vr_val, role_hr_score, synergy_for_role,
                alpha=alpha_slider.value if 'alpha_slider' in globals() else ALPHA_DEFAULT,
                beta=beta_slider.value if 'beta_slider' in globals() else BETA_DEFAULT
            )

            comparison_results.append({
                'Role': role_name,
                'V^R': current_vr_val,
                'H^R': role_hr_score,
                'Synergy%': synergy_for_role,
                'AI-R': ai_r_for_role
            })

        comparison_df = pd.DataFrame(comparison_results)
        with comparison_output:
            clear_output(wait=True)
            if not comparison_df.empty:
                print("--- Career Path Comparison ---")
                print(comparison_df.round(2).to_string(index=False))
                fig = px.bar(comparison_df.melt(id_vars='Role', value_vars=['V^R', 'H^R', 'AI-R'], var_name='Metric', value_name='Score'),
                             x='Role', y='Score', color='Metric', barmode='group',
                             title=f"AI-Readiness Components Across Target Roles (User V^R: {current_vr_val:.2f})",
                             labels={'Score': 'Score (0-100)'},
                             color_discrete_map={'V^R': 'skyblue', 'H^R': 'lightcoral', 'AI-R': 'mediumseagreen'})
                fig.update_layout(yaxis_range=[0, 100])
                fig.show()
            else:
                print("Please select at least one role for comparison.")
    ```

*   **Code Cell (Function Execution):**
    ```python
    role_options = ['None'] + market_opportunities_df['Role'].tolist()
    role_selector = widgets.SelectMultiple(
        options=role_options,
        value=[market_opportunities_df['Role'].iloc[0], market_opportunities_df['Role'].iloc[1]],
        description='Select Roles',
        disabled=False
    )
    compare_button = widgets.Button(description="Compare Roles")
    comparison_output = widgets.Output()

    def on_compare_button_clicked(b):
        compare_career_paths(list(role_selector.value))

    compare_button.on_click(on_compare_button_clicked)

    display(widgets.VBox([role_selector, compare_button, comparison_output]))
    ```

*   **Markdown Cell:**
    ```markdown
    The career path comparison visualization allows you to see how your individual readiness aligns with different market opportunities. This comparison highlights which roles offer the highest $AI-R$ given your current profile and helps in making informed decisions about career transitions or specialized skill development.
    ```

---

### Section 15: Conclusion and Next Steps

*   **Markdown Cell:**
    ```markdown
    ## 15. Conclusion and Next Steps

    This Jupyter Notebook has provided a detailed, interactive exploration of the AI-Readiness Score ($AI-R$) framework. You've learned how to:
    *   Calculate and understand **Idiosyncratic Readiness ($V^R$)** based on your skills and experience.
    *   Assess **Systematic Opportunity ($H^R$)** for various financial AI roles using market data.
    *   Quantify the **Synergy** between your readiness and market opportunities.
    *   Determine the overall **AI-Readiness Score ($AI-R$)** by integrating all components.
    *   Utilize "what-if" scenarios to simulate the impact of learning pathways and compare different career paths.

    For financial professionals like Financial Data Engineers and Portfolio Managers, understanding your AI-Readiness is crucial for navigating the evolving job market. This framework empowers you to:
    *   **Identify Skill Gaps:** Pinpoint specific areas within AI-Fluency, Domain-Expertise, or Adaptive-Capacity that need improvement.
    *   **Strategic Learning:** Select learning pathways that offer the highest impact on your $AI-R$.
    *   **Informed Career Decisions:** Compare roles based on objective metrics to choose paths with the best fit and opportunity.

    **Next Steps:**
    *   Consider which components of your $V^R$ you want to enhance and explore relevant learning pathways.
    *   Research specific roles that align with high $H^R$ scores and your preferences.
    *   Reflect on how different weighting of $\alpha$ and $\beta$ reflects your personal career philosophy (e.g., prioritizing personal growth vs. market demand).

    By continuously assessing and improving your AI-Readiness, you can proactively position yourself for success in the age of Artificial Intelligence.
    ```

