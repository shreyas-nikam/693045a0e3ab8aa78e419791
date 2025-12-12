
# Navigating the AI Frontier: An AI-Readiness Score Assessment for Financial Professionals

## Introduction

Welcome to the AI Career Navigator for Financial Professionals! In today's rapidly evolving financial landscape, Artificial Intelligence (AI) is transforming roles and creating new opportunities. For professionals like **Alex Chen**, a seasoned Financial Data Engineer at **FinTech Innovators Inc.**, understanding how to adapt and thrive in this AI-driven world is crucial.

Alex currently manages large financial datasets, builds data pipelines, and supports quantitative analysts. However, he sees the shift towards AI-powered insights and algorithmic decision-making. He aspires to transition into more AI-centric roles, such as an **AI-Risk Analyst** or an **Algorithmic Trading Engineer**, to stay ahead in his career and contribute more strategically to FinTech Innovators Inc.'s innovative projects.

This notebook will guide Alex through a practical, step-by-step workflow to assess his current AI-Readiness Score (AI-R). We will use a parametric framework that quantifies his preparedness and evaluates market opportunities. By the end of this journey, Alex will have a clear understanding of his strengths, areas for development, and a personalized roadmap to achieve his career goals within the AI-transformed finance industry.

**Alex's Goal:** To assess his current AI-Readiness for target roles and identify concrete learning pathways to enhance his career prospects at FinTech Innovators Inc.

## 1. Environment Setup

This section installs necessary libraries and imports dependencies to perform AI-Readiness calculations and visualizations.

### 1.1. Install Required Libraries

```python
!pip install pandas numpy ipywidgets matplotlib seaborn plotly scikit-learn
```

### 1.2. Import Required Dependencies

```python
import pandas as pd
import numpy as np
import ipywidgets as widgets
from IPython.display import display, Markdown
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
```

## 2. Defining Alex's Idiosyncratic Readiness ($V^R$)

**Story + Context + Real-World Relevance**

Alex's first step is to quantify his individual capabilities, which are captured by the Idiosyncratic Readiness ($V^R$). This component reflects his unique skills, knowledge, and adaptive traits that can be actively developed. At FinTech Innovators Inc., management is increasingly evaluating employees based on their AI-fluency and ability to integrate AI into existing financial processes. Alex needs a clear breakdown of his own readiness to identify areas where he can improve to meet these new expectations.

The Idiosyncratic Readiness ($V^R$) is composed of three main factors: AI-Fluency, Domain-Expertise, and Adaptive-Capacity. Each factor is assigned a weight, reflecting its relative importance:

$$
V^R(t) = w_1 \cdot AI\text{-}Fluency_i(t) + w_2 \cdot Domain\text{-}Expertise_i(t) + w_3 \cdot Adaptive\text{-}Capacity_i(t)
$$

where $w_1 = 0.45$, $w_2 = 0.35$, $w_3 = 0.20$.

Let's delve into each sub-component of Alex's profile.

### 2.1. AI-Fluency Sub-Components

AI-Fluency measures Alex's ability to effectively use, understand, and collaborate with AI systems. This is critical for his desired AI-Risk Analyst role, where he'll need to interpret AI model outputs and build AI-driven risk frameworks.

$$
AI\text{-}Fluency_i = \sum_{k=1}^4 \theta_k \cdot S_{i,k}
$$

The sub-components and their weights ($\theta_k$) are:
1.  **Technical AI Skills ($S_{i,1}$, $\theta_1 = 0.30$):** Alex's proficiency in prompt engineering, AI tools, understanding AI concepts, and data literacy.
    $$
    S_{i,1} = \frac{1}{4} (\text{Prompting}_i + \text{Tools}_i + \text{Understanding}_i + \text{DataLit}_i)
    $$
2.  **AI-Augmented Productivity ($S_{i,2}$, $\theta_2 = 0.35$):** How much AI augments his productivity.
    $$
    S_{i,2} = \frac{\text{Output Quality}_{i,\text{with AI}}}{\text{Output Quality}_{i,\text{without AI}}} \cdot \frac{\text{Time}_{i,\text{without AI}}}{\text{Time}_{i,\text{with AI}}}
    $$
3.  **Critical AI Judgment ($S_{i,3}$, $\theta_3 = 0.20$):** Alex's ability to identify AI errors and make appropriate trust decisions.
    $$
    S_{i,3} = \frac{\text{Errors Caught}_i}{\text{Total AI Errors}_i} + \frac{\text{Appropriate Trust Decisions}_i}{\text{Total Decisions}_i}
    $$
4.  **AI Learning Velocity ($S_{i,4}$, $\theta_4 = 0.15$):** His improvement rate in AI proficiency per unit time invested.
    $$
    S_{i,4} = \frac{\Delta\text{Proficiency}_i}{\Delta t} \cdot \frac{1}{\text{Hours Invested}}
    $$
Each $S_{i,k}$ is normalized to a $[0,1]$ scale.

### 2.2. Domain-Expertise Factor

Domain-Expertise captures Alex's deep knowledge in specific financial application areas, which is invaluable for translating AI insights into actionable financial strategies. This includes his educational background, practical experience, and specialization.

$$
\text{Domain-Expertise}_i = E_{\text{education}} \cdot E_{\text{experience}} \cdot E_{\text{specialization}}
$$

1.  **Educational Foundation ($E_{\text{education}}$):**
    *   PhD in target field: 1.0
    *   Master's in target field: 0.85
    *   Bachelor's in target field: 0.70
    *   Associate's/Certificate: 0.60
    *   HS + significant coursework: 0.50
2.  **Practical Experience ($E_{\text{experience}}$):**
    $$
    E_{\text{experience}} = 1 - e^{-\gamma \text{Years}}
    $$
    where $\gamma = 0.15$ ensures diminishing returns after approximately 15 years.
3.  **Specialization Depth ($E_{\text{specialization}}$):**
    $$
    E_{\text{specialization}} = 0.4 \cdot \text{Portfolio}_i + 0.3 \cdot \text{Recognition}_i + 0.3 \cdot \text{Credentials}_i
    $$
    Each sub-component (Portfolio, Recognition, Credentials) is on a $[0,1]$ scale.

### 2.3. Adaptive-Capacity Factor

Adaptive-Capacity measures meta-skills enabling Alex to successfully navigate AI-driven transitions, such as cognitive flexibility, social-emotional intelligence, and strategic career management. These soft skills are increasingly critical in roles requiring human-AI collaboration.

$$
\text{Adaptive-Capacity}_i = \frac{1}{3} (C_{\text{cognitive}} + C_{\text{social}} + C_{\text{strategic}})
$$

Where $C_{\text{cognitive}}$, $C_{\text{social}}$, and $C_{\text{strategic}}$ are scores on a $[0,100]$ scale, representing Cognitive Flexibility, Social-Emotional Intelligence, and Strategic Career Management, respectively. These are averaged and then scaled to $[0,1]$ for consistency.

### Code Cell: Calculate Alex's Idiosyncratic Readiness ($V^R$)

```python
# --- Define functions for VR sub-components ---

def calculate_ai_fluency(prompting, tools, understanding, datalit,
                         output_quality_with_ai, output_quality_without_ai,
                         time_without_ai, time_with_ai,
                         errors_caught, total_ai_errors, appropriate_trust_decisions, total_decisions,
                         delta_proficiency, hours_invested):
    """Calculates AI-Fluency score based on its sub-components."""
    # S1: Technical AI Skills (normalized to [0,1])
    s1 = (prompting + tools + understanding + datalit) / 4

    # S2: AI-Augmented Productivity (normalized to [0,1], assuming ratio for time/quality)
    # If output_quality_without_ai is 0, avoid division by zero.
    if output_quality_without_ai == 0:
        s2_quality_ratio = 1.0 # Assume no baseline, so 1.0 for augmentation
    else:
        s2_quality_ratio = output_quality_with_ai / output_quality_without_ai
    
    # If time_with_ai is 0, avoid division by zero.
    if time_with_ai == 0:
        s2_time_ratio = 1.0 # Assume no baseline, so 1.0 for efficiency
    else:
        s2_time_ratio = time_without_ai / time_with_ai
    
    s2 = s2_quality_ratio * s2_time_ratio
    s2 = min(max(s2, 0), 1) # Ensure S2 is within [0,1] reasonable bounds

    # S3: Critical AI Judgment (normalized to [0,1])
    s3_errors = errors_caught / total_ai_errors if total_ai_errors > 0 else 0
    s3_trust = appropriate_trust_decisions / total_decisions if total_decisions > 0 else 0
    s3 = (s3_errors + s3_trust) / 2 # Average of two parts, scaled to [0,1]

    # S4: AI Learning Velocity (normalized to [0,1])
    s4 = (delta_proficiency / hours_invested) if hours_invested > 0 else 0
    s4 = min(max(s4 * 5, 0), 1) # Scale S4 to be more meaningful, max possible proficiency 0.2/1 = 0.2, so *5 to reach 1

    # Weights for AI-Fluency sub-components
    theta = {'S1': 0.30, 'S2': 0.35, 'S3': 0.20, 'S4': 0.15}
    
    ai_fluency = (theta['S1'] * s1 + theta['S2'] * s2 +
                  theta['S3'] * s3 + theta['S4'] * s4)
    
    return ai_fluency, {'S1_TechAI_Skills': s1, 'S2_AI_Productivity': s2, 'S3_Critical_AI_Judgment': s3, 'S4_AI_Learning_Velocity': s4}

def calculate_domain_expertise(education_level, years_experience, portfolio, recognition, credentials):
    """Calculates Domain-Expertise score."""
    edu_map = {
        "PhD in target field": 1.0,
        "Master's in target field": 0.85,
        "Bachelor's in target field": 0.70,
        "Associate's/Certificate": 0.60,
        "HS + significant coursework": 0.50
    }
    edu = edu_map.get(education_level, 0.50)

    gamma_exp = 0.15
    exp = 1 - np.exp(-gamma_exp * years_experience)

    spec = (0.4 * portfolio + 0.3 * recognition + 0.3 * credentials)
    
    domain_expertise = edu * exp * spec
    return domain_expertise, {'E_education': edu, 'E_experience': exp, 'E_specialization': spec}

def calculate_adaptive_capacity(cognitive_flexibility, social_emotional_intelligence, strategic_career_management):
    """Calculates Adaptive-Capacity score (normalized to [0,1])."""
    # Components are expected to be on [0,100] scale, normalize to [0,1]
    c_cognitive_norm = cognitive_flexibility / 100
    c_social_norm = social_emotional_intelligence / 100
    c_strategic_norm = strategic_career_management / 100

    adaptive_capacity = (c_cognitive_norm + c_social_norm + c_strategic_norm) / 3
    return adaptive_capacity, {'C_Cognitive': c_cognitive_norm, 'C_Social_Emotional': c_social_norm, 'C_Strategic_Career': c_strategic_norm}

def calculate_vr(ai_fluency, domain_expertise, adaptive_capacity):
    """Calculates the overall Idiosyncratic Readiness (VR) score."""
    w1_ai_fluency = 0.45
    w2_domain_expertise = 0.35
    w3_adaptive_capacity = 0.20
    
    vr = (w1_ai_fluency * ai_fluency +
          w2_domain_expertise * domain_expertise +
          w3_adaptive_capacity * adaptive_capacity) * 100 # Scale to [0,100]
    
    return vr

# --- Alex's Profile Input (Synthetic Data) ---
# AI-Fluency Sub-components (normalized to [0,1] where applicable)
alex_prompting = 0.8
alex_tools = 0.7
alex_understanding = 0.6
alex_datalit = 0.7

alex_output_quality_with_ai = 1.2
alex_output_quality_without_ai = 1.0 # Baseline
alex_time_without_ai = 5.0 # Hours
alex_time_with_ai = 4.0 # Hours

alex_errors_caught = 75
alex_total_ai_errors = 100
alex_appropriate_trust_decisions = 80
alex_total_decisions = 100

alex_delta_proficiency = 0.15 # e.g., 15% increase in proficiency
alex_hours_invested = 20 # hours

# Domain-Expertise Factors
alex_education_level = "Master's in target field"
alex_years_experience = 8
alex_portfolio = 0.75 # [0,1] scale, e.g., strong portfolio
alex_recognition = 0.6 # [0,1] scale, e.g., some internal recognition
alex_credentials = 0.65 # [0,1] scale, e.g., CFA Level I, FRM

# Adaptive-Capacity Sub-components (raw scores, assumed [0,100])
alex_cognitive_flexibility = 70
alex_social_emotional_intelligence = 65
alex_strategic_career_management = 75

# --- Execute Calculations for Alex ---

# 1. Calculate AI-Fluency
alex_ai_fluency, ai_fluency_components = calculate_ai_fluency(
    alex_prompting, alex_tools, alex_understanding, alex_datalit,
    alex_output_quality_with_ai, alex_output_quality_without_ai,
    alex_time_without_ai, alex_time_with_ai,
    alex_errors_caught, alex_total_ai_errors, alex_appropriate_trust_decisions, alex_total_decisions,
    alex_delta_proficiency, alex_hours_invested
)

# 2. Calculate Domain-Expertise
alex_domain_expertise, domain_expertise_components = calculate_domain_expertise(
    alex_education_level, alex_years_experience, alex_portfolio, alex_recognition, alex_credentials
)

# 3. Calculate Adaptive-Capacity
alex_adaptive_capacity, adaptive_capacity_components = calculate_adaptive_capacity(
    alex_cognitive_flexibility, alex_social_emotional_intelligence, alex_strategic_career_management
)

# 4. Calculate overall VR
alex_vr = calculate_vr(alex_ai_fluency, alex_domain_expertise, alex_adaptive_capacity)

display(Markdown(f"### Alex's Calculated $V^R$ Components (normalized to [0,1]):"))
display(Markdown(f"- **AI-Fluency**: {alex_ai_fluency:.2f}"))
for k, v in ai_fluency_components.items():
    display(Markdown(f"  - {k}: {v:.2f}"))
display(Markdown(f"- **Domain-Expertise**: {alex_domain_expertise:.2f}"))
for k, v in domain_expertise_components.items():
    display(Markdown(f"  - {k}: {v:.2f}"))
display(Markdown(f"- **Adaptive-Capacity**: {alex_adaptive_capacity:.2f}"))
for k, v in adaptive_capacity_components.items():
    display(Markdown(f"  - {k}: {v:.2f}"))
display(Markdown(f"### Alex's Overall Idiosyncratic Readiness ($V^R$): **{alex_vr:.2f}** (on a scale of 0-100)"))

# --- Visualization for VR Components ---
vr_category_scores = {
    'AI-Fluency': alex_ai_fluency * 100, # Scale for visualization [0,100]
    'Domain-Expertise': alex_domain_expertise * 100,
    'Adaptive-Capacity': alex_adaptive_capacity * 100
}

fig = go.Figure(data=go.Scatterpolar(
  r=list(vr_category_scores.values()),
  theta=list(vr_category_scores.keys()),
  fill='toself',
  name='Alex Chen'
))

fig.update_layout(
  polar=dict(
    radialaxis=dict(
      visible=True,
      range=[0, 100]
    )),
  showlegend=True,
  title="Alex's Idiosyncratic Readiness ($V^R$) Component Breakdown"
)
fig.show()
```

**Explanation of Execution**

The output shows Alex's calculated scores for each sub-component and his overall Idiosyncratic Readiness ($V^R$). His $V^R$ of **`alex_vr:.2f`** indicates his current individual preparedness. The radar chart visually represents how he scores across AI-Fluency, Domain-Expertise, and Adaptive-Capacity. For instance, a high score in "Domain-Expertise" might reflect his years of experience as a Financial Data Engineer, while "AI-Fluency" might be an area for growth given his aspiration to AI-centric roles. This visual breakdown helps Alex quickly pinpoint his strengths and weaknesses from an internal capabilities perspective.

## 3. Market Pulse: Assessing Systematic Opportunity ($H^R$)

**Story + Context + Real-World Relevance**

Beyond his individual capabilities, Alex needs to understand the external market demand and growth potential for his target roles. This is the Systematic Opportunity ($H^R$), which captures macro-level job growth and demand that he can position himself to capture. FinTech Innovators Inc. operates in a competitive talent market, and Alex wants to ensure his career transition aligns with high-demand roles that promise significant growth and wage premiums. He is considering two primary target roles: **AI-Risk Analyst** and **Algorithmic Trading Engineer**.

The Systematic Opportunity is defined as:

$$
H^R(t) = H_{\text{base}}(O_{\text{target}}) \cdot M_{\text{growth}}(t) \cdot M_{\text{regional}}(t)
$$

where $H_{\text{base}}(o)$ is the base opportunity score, $M_{\text{growth}}(t)$ captures temporal market momentum, and $M_{\text{regional}}(t)$ adjusts for geographic factors.

The base opportunity score aggregates multiple dimensions of occupational attractiveness:

$$
H_{\text{base}}(o) = \sum_{j \in J_o} w_j \cdot AI\text{-}Enhancement_j \cdot Growth_j
$$

The factors and their weights are:
1.  **AI-Enhancement Potential ($w_1 = 0.30$):** How much AI augments rather than replaces tasks in the role.
    $$
    AI\text{-}Enhancement_o = \frac{1}{|T_o|} \sum_{t \in T_o} (1 - \text{Automation}_t) \cdot \text{AI-Augmentation}_t
    $$
    This value is scaled to $[0,1]$.
2.  **Job Growth Projections ($w_2 = 0.30$):** BLS 10-year outlook for the occupation. The raw growth rate $g \in [-0.5, 1.5]$ is normalized via:
    $$
    \text{Growth}_{\text{normalized}} = \frac{g+0.5}{2.0} \times 100
    $$
3.  **Wage Premium ($w_3 = 0.25$):** Compensation potential for AI-skilled roles relative to median wage.
    $$
    \text{Wage}_o = \frac{\text{AI-skilled wage}_o - \text{median wage}_o}{\text{median wage}_o}
    $$
    This value is scaled to $[0,1]$ before use in $H_{\text{base}}$.
4.  **Entry Accessibility ($w_4 = 0.15$):** Ease of transition into the role.
    $$
    \text{Access}_o = 1 - \frac{\text{Education Years Required} + \text{Experience Years Required}}{10}
    $$
    This value is scaled to $[0,1]$ before use in $H_{\text{base}}$.

The dynamic multipliers are:
*   **Growth Multiplier ($M_{\text{growth}}(t)$):** Captures market momentum based on job postings.
    $$
    M_{\text{growth}}(t) = 1 + \lambda \cdot \left(\frac{\text{Job Postings}_{o,t}}{\text{Job Postings}_{o,t-1}} - 1\right)
    $$
    where $\lambda = 0.3$ dampens volatility.
*   **Regional Multiplier ($M_{\text{regional}}(t)$):** Adjusts for local labor markets.
    $$
    M_{\text{regional}}(t) = \frac{\text{Local Demand}_{i,t}}{\text{National Avg Demand}} \times (1 + \gamma \cdot \text{Remote Work Factor}_o)
    $$
    where $\gamma = 0.2$ and $\text{Remote Work Factor} \in [0,1]$.

### Code Cell: Calculate Systematic Opportunity ($H^R$) for Target Roles

```python
# --- Define functions for HR components ---

def normalize_to_100(value, min_val, max_val):
    """Normalizes a value from a given range to [0, 100]."""
    if max_val == min_val:
        return 0
    return ((value - min_val) / (max_val - min_val)) * 100

def calculate_h_base(ai_enhancement, job_growth_raw, wage_premium_raw, entry_accessibility_raw):
    """Calculates the base opportunity score (H_base)."""
    # Weights for H_base components
    w1_ai_enhancement = 0.30
    w2_job_growth = 0.30
    w3_wage_premium = 0.25
    w4_entry_accessibility = 0.15

    # Normalize raw inputs to a [0,1] scale before applying weights
    # AI-Enhancement is already [0,1]
    norm_ai_enhancement = ai_enhancement

    # Job Growth Normalization (g in [-0.5, 1.5] -> [0,100])
    norm_job_growth = ((job_growth_raw + 0.5) / 2.0) # Map to [0,1] for calculation
    
    # Wage Premium Normalization (scaled to [0,1], assuming typical premiums are 0-1)
    # If wage_premium_raw is negative (lower than median), cap at 0 for opportunity score
    norm_wage_premium = max(0, min(1, wage_premium_raw))
    
    # Entry Accessibility Normalization (Access_o formula already maps to a [0,1] range directly, if denominator is 10)
    norm_entry_accessibility = max(0, min(1, entry_accessibility_raw)) # Ensure between 0 and 1

    h_base_score = (w1_ai_enhancement * norm_ai_enhancement +
                    w2_job_growth * norm_job_growth +
                    w3_wage_premium * norm_wage_premium +
                    w4_entry_accessibility * norm_entry_accessibility)
    
    return h_base_score * 100 # Scale to [0,100]

def calculate_m_growth(job_postings_t, job_postings_t_minus_1, lambda_param=0.3):
    """Calculates the growth multiplier (M_growth)."""
    if job_postings_t_minus_1 == 0:
        return 1.0 # No previous data, assume no change
    
    m_growth = 1 + lambda_param * ((job_postings_t / job_postings_t_minus_1) - 1)
    return m_growth

def calculate_m_regional(local_demand, national_avg_demand, remote_work_factor, gamma_param=0.2):
    """Calculates the regional multiplier (M_regional)."""
    if national_avg_demand == 0:
        return 1.0 # Avoid division by zero
    
    m_regional = (local_demand / national_avg_demand) * (1 + gamma_param * remote_work_factor)
    return m_regional

def calculate_hr(h_base, m_growth, m_regional):
    """Calculates the overall Systematic Opportunity (HR) score."""
    hr = h_base * m_growth * m_regional
    return min(max(hr, 0), 100) # Ensure HR is within [0,100]

# --- Synthetic Market Data for Target Roles ---
market_data = {
    'Occupation': ['AI-Risk Analyst', 'Algorithmic Trading Engineer', 'Quant Researcher'],
    'AI_Enhancement_Potential': [0.85, 0.90, 0.88], # 0-1 scale
    'Job_Growth_Projection_Raw': [0.40, 0.55, 0.35], # e.g., 40% growth
    'Wage_Premium_Raw': [0.60, 0.80, 0.70], # e.g., 60% higher than median
    'Entry_Accessibility_Raw': [0.70, 0.50, 0.60], # 0-1 scale, higher is easier
    'Job_Postings_t': [500, 750, 600],
    'Job_Postings_t_minus_1': [450, 700, 580],
    'Local_Demand': [1.1, 1.2, 0.9], # Relative to national average
    'National_Avg_Demand': [1.0, 1.0, 1.0],
    'Remote_Work_Factor': [0.4, 0.2, 0.3] # 0-1 scale
}
market_data_df = pd.DataFrame(market_data)

# --- Execute Calculations for Market Data ---
target_roles = ['AI-Risk Analyst', 'Algorithmic Trading Engineer']
hr_scores = {}
hr_component_breakdowns = {}

for index, row in market_data_df[market_data_df['Occupation'].isin(target_roles)].iterrows():
    occupation = row['Occupation']

    # Calculate H_base
    h_base, h_base_components = calculate_h_base(
        row['AI_Enhancement_Potential'],
        row['Job_Growth_Projection_Raw'],
        row['Wage_Premium_Raw'],
        row['Entry_Accessibility_Raw']
    ), {
        'AI-Enhancement': row['AI_Enhancement_Potential'] * 100,
        'Job Growth': ((row['Job_Growth_Projection_Raw'] + 0.5) / 2.0) * 100,
        'Wage Premium': row['Wage_Premium_Raw'] * 100,
        'Entry Accessibility': row['Entry_Accessibility_Raw'] * 100
    }

    # Calculate M_growth
    m_growth = calculate_m_growth(row['Job_Postings_t'], row['Job_Postings_t_minus_1'])

    # Calculate M_regional
    m_regional = calculate_m_regional(row['Local_Demand'], row['National_Avg_Demand'], row['Remote_Work_Factor'])

    # Calculate overall HR
    hr = calculate_hr(h_base, m_growth, m_regional)
    hr_scores[occupation] = hr
    hr_component_breakdowns[occupation] = {
        'H_base': h_base,
        'M_growth': m_growth,
        'M_regional': m_regional,
        'AI-Enhancement': h_base_components['AI-Enhancement'],
        'Job Growth': h_base_components['Job Growth'],
        'Wage Premium': h_base_components['Wage Premium'],
        'Entry Accessibility': h_base_components['Entry Accessibility']
    }

display(Markdown(f"### Systematic Opportunity ($H^R$) for Target Roles (on a scale of 0-100):"))
for role, hr_score in hr_scores.items():
    display(Markdown(f"- **{role}**: {hr_score:.2f}"))
    for comp, val in hr_component_breakdowns[role].items():
        if comp not in ['H_base', 'M_growth', 'M_regional']: # Only show primary H_base factors
            display(Markdown(f"  - {comp}: {val:.2f}"))

# --- Visualization for HR Components ---
labels = ['AI-Enhancement', 'Job Growth', 'Wage Premium', 'Entry Accessibility']
fig = go.Figure()

for role, breakdown in hr_component_breakdowns.items():
    values = [breakdown[label] for label in labels]
    fig.add_trace(go.Bar(
        x=labels,
        y=values,
        name=role,
        marker_color=sns.color_palette("viridis")[list(hr_component_breakdowns.keys()).index(role)]
    ))

fig.update_layout(
    title="Systematic Opportunity ($H^R$) Component Breakdown by Target Role",
    yaxis_title="Score (0-100)",
    barmode='group'
)
fig.show()
```

**Explanation of Execution**

The bar chart vividly illustrates the Systematic Opportunity ($H^R$) breakdown for Alex's target roles. For example, "Algorithmic Trading Engineer" might show a higher wage premium and job growth, reflecting current market trends at FinTech Innovators Inc. and beyond. In contrast, "AI-Risk Analyst" might have a strong AI-enhancement potential, indicating it's a role where AI significantly augments human tasks rather than replacing them. This allows Alex to compare external opportunities and validate if his chosen career paths align with promising market dynamics.

## 4. Strategic Alignment: Calculating Synergy

**Story + Context + Real-World Relevance**

Alex now understands his individual readiness ($V^R$) and the market opportunity ($H^R$). However, simply having high scores in both doesn't guarantee success. The relationship isn't merely additive; when Alex's individual preparation perfectly aligns with market opportunity, the benefits compound multiplicatively. This compounding effect is captured by the **Synergy Function**. At FinTech Innovators Inc., Alex knows that internal promotions favor individuals whose skills not only meet job requirements but also fit the strategic timing of new initiatives.

The synergy term in the overall AI-Readiness Score is defined as:

$$
\text{Synergy}\%(V^R, H^R) = \frac{V^R \times H^R}{100} \times \text{Alignment}_i
$$

where both $V^R$ and $H^R$ are on a $[0,100]$ scale, and $\text{Alignment}_i \in [0,1]$ ensures $\text{Synergy}\% \in [0, 100]$.

The Alignment Factor measures how well Alex's individual skills match occupation requirements and the timing of his career transition:

$$
\text{Alignment}_i = \frac{\text{Skills Match Score}}{\text{Maximum Possible Match}} \times \text{Timing Factor}
$$

*   **Skills Match Score:** Uses O*NET task-skill mappings to compute.
    $$
    \text{Match}_i = \sum_{s \in S} \min(\text{Individual Skill}_{i,s}, \text{Required Skill}_{o,s}) \cdot \text{Importance}_s
    $$
    This formula ensures that excess skill in one area doesn't compensate for a deficiency in critical areas. Maximum Possible Match is the sum of (Required Skill * Importance) for all skills.

*   **Timing Factor:** Career stage affects transition ease based on years of experience ($y$).
    $$
    \text{Timing}(y) = \begin{cases}
    1.0 & y \in [0,5] \quad (\text{early career}) \\
    1.0 & y \in (5,15] \quad (\text{mid-career}) \\
    0.8 & y > 15 \quad (\text{late career, transition friction})
    \end{cases}
    $$

### Code Cell: Calculate Synergy for Alex

```python
# --- Define functions for Synergy components ---

def calculate_skills_match(individual_skills_df, required_skills_df, occupation):
    """Calculates the Skills Match Score and Maximum Possible Match."""
    
    # Filter required skills for the target occupation
    occupation_required_skills = required_skills_df[required_skills_df['Occupation'] == occupation]

    if occupation_required_skills.empty:
        return 0, 1 # No required skills defined, return 0 match and max_match of 1

    matched_score = 0
    max_possible_match = 0

    for idx, req_skill_row in occupation_required_skills.iterrows():
        skill_name = req_skill_row['Skill']
        required_level = req_skill_row['Required_Skill_Level'] # on [0,100] scale
        importance = req_skill_row['Importance'] # on [0,1] scale

        individual_level_row = individual_skills_df[individual_skills_df['Skill'] == skill_name]
        
        if not individual_level_row.empty:
            individual_level = individual_level_row['Individual_Skill_Level'].iloc[0] # on [0,100] scale
            matched_score += min(individual_level, required_level) * importance
        else:
            # If Alex doesn't have the skill, his level is 0 for min calculation
            matched_score += min(0, required_level) * importance
        
        max_possible_match += required_level * importance
    
    # Normalize matched_score and max_possible_match to handle cases where skill levels are not strictly 0-1
    # Assuming required_level and individual_level are from 0-100, and importance 0-1
    # Max score if all skills perfectly matched and importance is 1 for all.
    
    return matched_score, max_possible_match


def calculate_timing_factor(years_experience):
    """Calculates the Timing Factor based on years of experience."""
    if years_experience <= 5:
        return 1.0 # Early career
    elif 5 < years_experience <= 15:
        return 1.0 # Mid-career
    else:
        return 0.8 # Late career, transition friction

def calculate_alignment(years_experience, individual_skills_df, required_skills_df, occupation):
    """Calculates the Alignment Factor."""
    skills_match_score, max_possible_match = calculate_skills_match(individual_skills_df, required_skills_df, occupation)
    
    if max_possible_match == 0:
        skills_match_ratio = 0
    else:
        skills_match_ratio = skills_match_score / max_possible_match
    
    timing_factor = calculate_timing_factor(years_experience)
    
    alignment = skills_match_ratio * timing_factor
    return alignment, {'Skills_Match_Ratio': skills_match_ratio, 'Timing_Factor': timing_factor}

def calculate_synergy(vr_score, hr_score, alignment_score):
    """Calculates the Synergy percentage."""
    synergy_percent = (vr_score * hr_score / 100) * alignment_score
    return synergy_percent

# --- Synthetic Skills Data ---
alex_individual_skills = {
    'Skill': ['Python Programming', 'SQL', 'Financial Modeling', 'Risk Management', 'Machine Learning', 
              'Data Visualization', 'Generative AI', 'Quantitative Analysis', 'Statistical Modeling', 
              'Cloud Platforms (AWS/Azure)'],
    'Individual_Skill_Level': [90, 85, 70, 60, 55, 75, 40, 65, 60, 70] # on [0,100] scale
}
alex_individual_skills_df = pd.DataFrame(alex_individual_skills)

required_skills = {
    'Occupation': [
        'AI-Risk Analyst', 'AI-Risk Analyst', 'AI-Risk Analyst', 'AI-Risk Analyst', 'AI-Risk Analyst',
        'Algorithmic Trading Engineer', 'Algorithmic Trading Engineer', 'Algorithmic Trading Engineer', 
        'Algorithmic Trading Engineer', 'Algorithmic Trading Engineer'
    ],
    'Skill': [
        'Risk Management', 'Machine Learning', 'Statistical Modeling', 'Python Programming', 'Financial Regulations',
        'Python Programming', 'Quantitative Analysis', 'Machine Learning', 'Cloud Platforms (AWS/Azure)', 'Algorithmic Trading Strategies'
    ],
    'Required_Skill_Level': [80, 75, 70, 85, 70, 
                             95, 90, 80, 85, 75], # on [0,100] scale
    'Importance': [0.9, 0.8, 0.7, 0.7, 0.6,
                   0.9, 0.85, 0.8, 0.7, 0.9] # on [0,1] scale
}
required_skills_df = pd.DataFrame(required_skills)

# --- Execute Calculations for Synergy ---
alex_synergy_scores = {}
alex_alignment_details = {}

for role in target_roles:
    alignment_score, alignment_components = calculate_alignment(
        alex_years_experience, alex_individual_skills_df, required_skills_df, role
    )
    alex_alignment_details[role] = alignment_components
    
    vr_score_for_synergy = alex_vr
    hr_score_for_synergy = hr_scores[role]
    
    synergy = calculate_synergy(vr_score_for_synergy, hr_score_for_synergy, alignment_score)
    alex_synergy_scores[role] = synergy
    
    display(Markdown(f"### Synergy Calculation for {role}:"))
    display(Markdown(f"- **Alignment Score**: {alignment_score:.2f}"))
    for k, v in alignment_components.items():
        display(Markdown(f"  - {k}: {v:.2f}"))
    display(Markdown(f"- **Synergy Contribution**: {synergy:.2f}"))

```

**Explanation of Execution**

The synergy calculation output provides crucial insights. Alex can see how well his existing skills (like Python and SQL) align with the specific requirements of "AI-Risk Analyst" versus "Algorithmic Trading Engineer." A lower alignment score for one role might highlight a specific skill gap (e.g., "Algorithmic Trading Strategies") that could be a bottleneck, even if the $V^R$ and $H^R$ are high. The timing factor, considering Alex's 8 years of experience, suggests he's in a mid-career stage, which is generally favorable for transitions. This granular detail helps Alex understand *why* certain roles might lead to higher synergy and thus a greater overall AI-Readiness.

## 5. The AI-Readiness Score ($AI-R$): A Holistic View

**Story + Context + Real-World Relevance**

Now, Alex combines all the pieces: his individual readiness ($V^R$), the market opportunity ($H^R$), and their strategic alignment (Synergy). This gives him the comprehensive **AI-Readiness Score ($AI-R$)**, a single metric to guide his career decisions. FinTech Innovators Inc. uses this holistic view to identify high-potential employees for advanced AI projects. Alex also knows that the weighting of individual factors versus market factors can be adjusted (parameters $\alpha$ and $\beta$), reflecting different organizational priorities or personal career philosophies. He wants to explore how sensitive his $AI-R$ is to these parameters.

The overall AI-Readiness Score for individual $i$ at time $t$ is:

$$
AI-R_{i,t} = \alpha \cdot V^R_i(t) + (1 - \alpha) \cdot H^R(t) + \beta \cdot \text{Synergy}\%(V^R, H^R)
$$

where $\alpha \in [0,1]$ is the weight on individual vs. market factors (default $\alpha=0.6$), and $\beta > 0$ is the synergy coefficient (default $\beta=0.15$). Both $V^R$ and $H^R$ are normalized to $[0,100]$, and $\text{Synergy}\%$ is also on $[0,100]$.

### Code Cell: Calculate Initial $AI-R$ and Explore Sensitivity

```python
# --- Define function for AI-Readiness Score ---

def calculate_air(vr_score, hr_score, synergy_score, alpha=0.6, beta=0.15):
    """Calculates the AI-Readiness Score."""
    air = alpha * vr_score + (1 - alpha) * hr_score + beta * synergy_score
    return min(max(air, 0), 100) # Ensure AI-R is within [0,100]

# --- Execute Initial AI-R Calculation for Alex ---
alex_air_scores = {}
default_alpha = 0.6
default_beta = 0.15

for role in target_roles:
    current_hr = hr_scores[role]
    current_synergy = alex_synergy_scores[role]
    
    air = calculate_air(alex_vr, current_hr, current_synergy, default_alpha, default_beta)
    alex_air_scores[role] = air
    display(Markdown(f"### Initial AI-Readiness Score ($AI-R$) for {role}:"))
    display(Markdown(f"- $V^R$: {alex_vr:.2f}"))
    display(Markdown(f"- $H^R$: {current_hr:.2f}"))
    display(Markdown(f"- Synergy Contribution: {current_synergy:.2f}"))
    display(Markdown(f"- **Overall $AI-R$ (alpha={default_alpha}, beta={default_beta})**: **{air:.2f}**"))

# --- Interactive Widgets for Alpha and Beta Sensitivity ---
alpha_slider = widgets.FloatSlider(
    value=default_alpha,
    min=0.0,
    max=1.0,
    step=0.05,
    description='Alpha (VR weight):',
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='.2f',
)

beta_slider = widgets.FloatSlider(
    value=default_beta,
    min=0.0,
    max=0.5, # Extended max for demonstration
    step=0.01,
    description='Beta (Synergy weight):',
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='.2f',
)

air_output = widgets.Output()

def update_air_plot(alpha, beta):
    with air_output:
        air_output.clear_output(wait=True)
        scenario_air_scores = {}
        for role in target_roles:
            current_hr = hr_scores[role]
            current_synergy = alex_synergy_scores[role]
            scenario_air_scores[role] = calculate_air(alex_vr, current_hr, current_synergy, alpha, beta)
        
        roles = list(scenario_air_scores.keys())
        air_values = list(scenario_air_scores.values())

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=roles,
            y=air_values,
            name='AI-R Score',
            marker_color=sns.color_palette("viridis", len(roles))
        ))
        fig.update_layout(
            title=f"Alex's AI-Readiness Score ($AI-R$) by Role (Alpha={alpha:.2f}, Beta={beta:.2f})",
            yaxis_title="AI-Readiness Score (0-100)",
            xaxis_title="Target Role"
        )
        fig.show()

display(Markdown("### Adjust Alpha and Beta Parameters to See AI-R Sensitivity:"))
widgets.interactive_output(update_air_plot, {'alpha': alpha_slider, 'beta': beta_slider})
display(alpha_slider, beta_slider, air_output)
```

**Explanation of Execution**

The interactive sliders for $\alpha$ and $\beta$ provide immediate feedback on how Alex's $AI-R$ changes. If FinTech Innovators Inc. prioritizes individual capabilities more (higher $\alpha$), Alex might see his "AI-Risk Analyst" score improve more if his $V^R$ is higher than the market $H^R$. Conversely, if synergy is highly valued (higher $\beta$), roles with better skill alignment will see a boost. This helps Alex understand the "governance" behind the score and how different strategic lenses can affect his perceived readiness, preparing him for discussions with his manager about career progression.

## 6. "What-If" Scenario: Planning Learning Pathways

**Story + Context + Real-World Relevance**

With a clear baseline $AI-R$, Alex is ready to plan his next steps. He wants to know how investing in specific learning pathways will enhance his readiness for his primary target role: **AI-Risk Analyst**. FinTech Innovators Inc. offers various internal and external learning programs, and Alex needs to choose the most impactful ones. This "what-if" scenario planning allows him to simulate the impact of these pathways on his $V^R$ components and, ultimately, his overall $AI-R$.

The dynamic score update framework is:

$$
AI-R_{i,t+1} = AI-R_{i,t} + \sum_{p \in P} \Delta_p \cdot \text{Completion}_p \cdot \text{Mastery}_p
$$

where $P$ is the set of learning pathways undertaken, $\Delta_p$ is the pre-calibrated impact coefficient for pathway $p$, $\text{Completion}_p \in [0,1]$ is the fraction of pathway completed, and $\text{Mastery}_p \in [0,1]$ is the assessment performance score.

Pathways are categorized by their primary impact:
*   **Type 1 - AI-Fluency Pathways:** Directly improve AI-Fluency (e.g., $\Delta_{p,\text{fluency}} \in [5, 15]$ points in AI-Fluency).
*   **Type 2 - Domain + AI Integration:** Improve both Domain-Expertise and Alignment (e.g., $\Delta_{p,\text{domain}} \in [10,20]$ points across multiple components).
*   **Type 3 - Adaptive Capacity:** Improve meta-skills (e.g., $\Delta_{p,\text{adaptive}} \in [5, 10]$ points in Adaptive-Capacity).

### Code Cell: Simulate Learning Pathway Impact

```python
# --- Define learning pathways data ---
learning_pathways_data = {
    'Pathway_ID': [
        'PE101', 'MLOps201', 'GenAI_Fin', 'FinRisk_ML', 'Ethics_AI_Fin',
        'Cognitive_Flex', 'Strategic_Career', 'Fintech_Reg'
    ],
    'Category': [
        'AI-Fluency', 'AI-Fluency', 'AI-Fluency', 'Domain+AI Integration', 'Adaptive Capacity',
        'Adaptive Capacity', 'Adaptive Capacity', 'Domain+AI Integration'
    ],
    'Description': [
        'Prompt Engineering Fundamentals', 'MLOps for Financial Data', 'Generative AI for Finance',
        'Machine Learning in Financial Risk Management', 'AI Ethics and Responsible Use in Finance',
        'Cognitive Flexibility Training', 'Strategic Career Management in AI Age', 'Fintech Regulatory Landscape'
    ],
    'Delta_P_AI_Fluency': [8, 12, 10, 0, 0, 0, 0, 0], # Impact on AI-Fluency (points)
    'Delta_P_Domain_Expertise': [0, 0, 0, 15, 0, 0, 0, 10], # Impact on Domain-Expertise (points)
    'Delta_P_Adaptive_Capacity': [0, 0, 0, 0, 8, 7, 6, 0], # Impact on Adaptive-Capacity (points)
    'Delta_P_Alignment_Skills_Match': [0, 0, 0, 0.05, 0, 0, 0, 0.03], # Impact on Alignment Skills Match Ratio (0-1)
    'Estimated_Completion_Time_Hours': [40, 80, 60, 120, 30, 20, 20, 50]
}
learning_pathways_df = pd.DataFrame(learning_pathways_data)

# --- Scenario Analysis Engine Function ---
def simulate_learning_pathways(
    current_vr, current_ai_fluency_components, current_domain_components, current_adaptive_components,
    current_air, current_hr, current_synergy, current_alignment_details,
    selected_pathways_ids, completion_scores, mastery_scores,
    alpha=0.6, beta=0.15
):
    """
    Simulates the impact of learning pathways on VR and overall AI-R.
    completion_scores and mastery_scores are dicts mapping pathway_id to score.
    """
    projected_ai_fluency_components = current_ai_fluency_components.copy()
    projected_domain_components = current_domain_components.copy()
    projected_adaptive_components = current_adaptive_components.copy()
    projected_alignment_details = current_alignment_details.copy()

    total_air_delta = 0
    
    for pathway_id in selected_pathways_ids:
        pathway = learning_pathways_df[learning_pathways_df['Pathway_ID'] == pathway_id].iloc[0]
        
        completion = completion_scores.get(pathway_id, 1.0) # Default to 100% completion
        mastery = mastery_scores.get(pathway_id, 1.0) # Default to 100% mastery

        # Update V^R sub-components
        # Assuming Delta_P values are already in points for AI-Fluency, Domain, Adaptive
        # and that they add to the original component score on a 0-100 scale.
        # We need to scale these impacts to the [0,1] range of the components for VR calculation.
        
        # AI-Fluency (0-100 score from calculate_ai_fluency output)
        if pathway['Delta_P_AI_Fluency'] > 0:
            # We add to the overall AI-Fluency score from alex_ai_fluency (which is 0-1)
            # and ensure it does not exceed 1.0
            total_ai_fluency_impact = (pathway['Delta_P_AI_Fluency'] / 100) * completion * mastery
            alex_ai_fluency_updated = min(alex_ai_fluency + total_ai_fluency_impact, 1.0)
            
        # Domain-Expertise (0-100 score from calculate_domain_expertise output)
        if pathway['Delta_P_Domain_Expertise'] > 0:
            # Add to the Domain-Expertise score (which is 0-1)
            total_domain_expertise_impact = (pathway['Delta_P_Domain_Expertise'] / 100) * completion * mastery
            alex_domain_expertise_updated = min(alex_domain_expertise + total_domain_expertise_impact, 1.0)

        # Adaptive-Capacity (0-100 score from calculate_adaptive_capacity output)
        if pathway['Delta_P_Adaptive_Capacity'] > 0:
            # Add to the Adaptive-Capacity score (which is 0-1)
            total_adaptive_capacity_impact = (pathway['Delta_P_Adaptive_Capacity'] / 100) * completion * mastery
            alex_adaptive_capacity_updated = min(alex_adaptive_capacity + total_adaptive_capacity_impact, 1.0)

        # Update Alignment (Skills Match Ratio)
        if pathway['Delta_P_Alignment_Skills_Match'] > 0:
            # Add to the Skills_Match_Ratio (which is 0-1)
            projected_alignment_details['Skills_Match_Ratio'] = min(
                projected_alignment_details['Skills_Match_Ratio'] + pathway['Delta_P_Alignment_Skills_Match'] * completion * mastery, 1.0
            )

        # For simplicity in this simulation, we'll directly add impact to the final VR components
        # This is a simplified dynamic update. In a real system, the underlying sub-sub components would update.
        delta_ai_fluency_points = pathway['Delta_P_AI_Fluency'] * completion * mastery
        delta_domain_expertise_points = pathway['Delta_P_Domain_Expertise'] * completion * mastery
        delta_adaptive_capacity_points = pathway['Delta_P_Adaptive_Capacity'] * completion * mastery

        total_air_delta += (delta_ai_fluency_points * alpha * 0.45) # VR_w1 * AI_Fluency_w1
        total_air_delta += (delta_domain_expertise_points * alpha * 0.35) # VR_w2 * Domain_w2
        total_air_delta += (delta_adaptive_capacity_points * alpha * 0.20) # VR_w3 * Adaptive_w3

    # Re-calculate projected VR, Synergy and AIR based on new sub-component values
    # We must re-calculate the components of VR first and then the overall VR
    # For this simplified pathway simulation, we'll directly add the Delta_P values to the VR components.
    
    # We apply the delta values to the base scores, assuming delta values are "points" on the 0-100 scale
    # and then normalize them back to 0-1 for the calculate_vr function.
    projected_ai_fluency_scaled = min(alex_ai_fluency * 100 + sum(p['Delta_P_AI_Fluency'] for p in learning_pathways_df[learning_pathways_df['Pathway_ID'].isin(selected_pathways_ids)].to_dict('records')), 100) / 100.0
    projected_domain_expertise_scaled = min(alex_domain_expertise * 100 + sum(p['Delta_P_Domain_Expertise'] for p in learning_pathways_df[learning_pathways_df['Pathway_ID'].isin(selected_pathways_ids)].to_dict('records')), 100) / 100.0
    projected_adaptive_capacity_scaled = min(alex_adaptive_capacity * 100 + sum(p['Delta_P_Adaptive_Capacity'] for p in learning_pathways_df[learning_pathways_df['Pathway_ID'].isin(selected_pathways_ids)].to_dict('records')), 100) / 100.0
    
    projected_vr = calculate_vr(projected_ai_fluency_scaled, projected_domain_expertise_scaled, projected_adaptive_capacity_scaled)

    # Re-calculate alignment and synergy based on updated skills_match_ratio
    # For simplicity, we assume the skills_match_ratio update is cumulative and applied once.
    original_alignment_score, _ = calculate_alignment(
        alex_years_experience, alex_individual_skills_df, required_skills_df, target_role_for_pathways
    )

    projected_skills_match_ratio = original_alignment_score # Start with original ratio
    for pathway_id in selected_pathways_ids:
        pathway = learning_pathways_df[learning_pathways_df['Pathway_ID'] == pathway_id].iloc[0]
        completion = completion_scores.get(pathway_id, 1.0)
        mastery = mastery_scores.get(pathway_id, 1.0)
        projected_skills_match_ratio = min(projected_skills_match_ratio + pathway['Delta_P_Alignment_Skills_Match'] * completion * mastery, 1.0)

    projected_alignment_score = projected_skills_match_ratio * current_alignment_details['Timing_Factor'] # Timing factor remains unchanged
    
    projected_synergy = calculate_synergy(projected_vr, current_hr, projected_alignment_score)
    projected_air = calculate_air(projected_vr, current_hr, projected_synergy, alpha, beta)

    return projected_air, projected_vr, projected_synergy, {
        'AI-Fluency': projected_ai_fluency_scaled * 100,
        'Domain-Expertise': projected_domain_expertise_scaled * 100,
        'Adaptive-Capacity': projected_adaptive_capacity_scaled * 100
    }

# --- Interactive Pathway Selection ---
pathway_options = [(row['Description'], row['Pathway_ID']) for idx, row in learning_pathways_df.iterrows()]

pathways_selector = widgets.SelectMultiple(
    options=pathway_options,
    value=[],
    description='Select Pathways:',
    disabled=False
)

target_role_selector = widgets.Dropdown(
    options=target_roles,
    value='AI-Risk Analyst',
    description='Target Role:',
    disabled=False,
)

# Sliders for completion and mastery
completion_sliders = {}
mastery_sliders = {}
for pathway_id in learning_pathways_df['Pathway_ID']:
    completion_sliders[pathway_id] = widgets.FloatSlider(value=1.0, min=0.0, max=1.0, step=0.05, description=f'Completion ({pathway_id}):', readout=True, readout_format='.2f')
    mastery_sliders[pathway_id] = widgets.FloatSlider(value=1.0, min=0.0, max=1.0, step=0.05, description=f'Mastery ({pathway_id}):', readout=True, readout_format='.2f')

sliders_box = widgets.VBox([])

def on_pathways_change(change):
    selected_pathways_ids = pathways_selector.value
    new_sliders = []
    for pathway_id in learning_pathways_df['Pathway_ID']:
        if pathway_id in selected_pathways_ids:
            new_sliders.append(completion_sliders[pathway_id])
            new_sliders.append(mastery_sliders[pathway_id])
    sliders_box.children = new_sliders

pathways_selector.observe(on_pathways_change, names='value')

pathway_scenario_output = widgets.Output()

def run_pathway_scenario(selected_pathways, target_role, *args): # *args to capture dynamic sliders
    with pathway_scenario_output:
        pathway_scenario_output.clear_output(wait=True)

        completion_scores = {pid: s.value for pid, s in completion_sliders.items()}
        mastery_scores = {pid: s.value for pid, s in mastery_sliders.items()}
        
        # We need the current alignment details for the selected target role
        global target_role_for_pathways
        target_role_for_pathways = target_role # Store for use in simulation function
        current_alignment_for_role = alex_alignment_details[target_role]
        
        projected_air, projected_vr, projected_synergy, projected_vr_components = simulate_learning_pathways(
            alex_vr, ai_fluency_components, domain_expertise_components, adaptive_capacity_components,
            alex_air_scores[target_role], hr_scores[target_role], alex_synergy_scores[target_role],
            current_alignment_for_role,
            selected_pathways, completion_scores, mastery_scores,
            default_alpha, default_beta
        )
        
        display(Markdown(f"### Pathway Scenario for **{target_role}**"))
        display(Markdown(f"**Initial $AI-R$**: {alex_air_scores[target_role]:.2f}"))
        display(Markdown(f"**Projected $AI-R$**: {projected_air:.2f} (Change: {projected_air - alex_air_scores[target_role]:+.2f})"))
        display(Markdown(f"**Initial $V^R$**: {alex_vr:.2f}"))
        display(Markdown(f"**Projected $V^R$**: {projected_vr:.2f} (Change: {projected_vr - alex_vr:+.2f})"))
        
        # Visualization
        labels = list(projected_vr_components.keys())
        initial_values = [
            alex_ai_fluency * 100,
            alex_domain_expertise * 100,
            alex_adaptive_capacity * 100
        ]
        projected_values = list(projected_vr_components.values())

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=labels,
            y=initial_values,
            name='Initial VR Components',
            marker_color='skyblue'
        ))
        fig.add_trace(go.Bar(
            x=labels,
            y=projected_values,
            name='Projected VR Components',
            marker_color='lightcoral'
        ))
        fig.update_layout(
            title="Initial vs. Projected $V^R$ Component Breakdown after Learning Pathways",
            yaxis_title="Score (0-100)",
            barmode='group'
        )
        fig.show()

run_button = widgets.Button(description="Run Scenario")
def on_run_button_clicked(b):
    # Collect all slider values dynamically
    all_slider_values = {s.description: s for s in sliders_box.children}
    # Pass the selected pathways and target role, plus a dummy argument for each slider
    run_pathway_scenario(pathways_selector.value, target_role_selector.value, *all_slider_values.values())

run_button.on_click(on_run_button_clicked)

display(Markdown("### Select Target Role and Learning Pathways:"))
display(target_role_selector, pathways_selector, sliders_box, run_button, pathway_scenario_output)
```

**Explanation of Execution**

By selecting specific pathways like "Machine Learning in Financial Risk Management" and "AI Ethics and Responsible Use in Finance," Alex can see an immediate projection of how his $AI-R$ and $V^R$ components will change for the "AI-Risk Analyst" role. The bar chart shows the "Before vs. After" effect on his AI-Fluency, Domain-Expertise, and Adaptive-Capacity. For instance, completing "Machine Learning in Financial Risk Management" would significantly boost his "Domain-Expertise" and "AI-Fluency." This allows Alex to prioritize pathways that directly address his identified skill gaps and have the highest impact on his target role's readiness.

## 7. Optimizing Career Transition: Pathway Comparison

**Story + Context + Real-World Relevance**

Alex has explored the impact of individual pathways, but now he needs to make a strategic decision: which combination of pathways for which target role offers the best return on his learning investment? FinTech Innovators Inc. encourages a data-driven approach to career development, and Alex wants to present a clear case for his chosen learning plan. This section allows him to compare multiple "what-if" scenarios side-by-side, evaluating the total $AI-R$ and the improvement for each.

### Code Cell: Compare Multiple Pathway Scenarios

```python
# --- Define a function to encapsulate a full scenario run for comparison ---
def get_scenario_results(
    current_vr, current_ai_fluency_components, current_domain_components, current_adaptive_components,
    initial_air, initial_hr, initial_synergy, initial_alignment_details,
    selected_pathways_ids,
    alpha=0.6, beta=0.15
):
    """
    Runs a single scenario (with 100% completion/mastery for simplicity in comparison)
    and returns projected AI-R and VR.
    """
    
    projected_ai_fluency_scaled = min(alex_ai_fluency * 100 + sum(p['Delta_P_AI_Fluency'] for p in learning_pathways_df[learning_pathways_df['Pathway_ID'].isin(selected_pathways_ids)].to_dict('records')), 100) / 100.0
    projected_domain_expertise_scaled = min(alex_domain_expertise * 100 + sum(p['Delta_P_Domain_Expertise'] for p in learning_pathways_df[learning_pathways_df['Pathway_ID'].isin(selected_pathways_ids)].to_dict('records')), 100) / 100.0
    projected_adaptive_capacity_scaled = min(alex_adaptive_capacity * 100 + sum(p['Delta_P_Adaptive_Capacity'] for p in learning_pathways_df[learning_pathways_df['Pathway_ID'].isin(selected_pathways_ids)].to_dict('records')), 100) / 100.0
    
    projected_vr = calculate_vr(projected_ai_fluency_scaled, projected_domain_expertise_scaled, projected_adaptive_capacity_scaled)

    # Re-calculate alignment and synergy based on updated skills_match_ratio
    original_alignment_score, current_alignment_for_role_comps = calculate_alignment(
        alex_years_experience, alex_individual_skills_df, required_skills_df, initial_alignment_details['Target_Role']
    )

    projected_skills_match_ratio = current_alignment_for_role_comps['Skills_Match_Ratio'] # Start with original ratio
    for pathway_id in selected_pathways_ids:
        pathway = learning_pathways_df[learning_pathways_df['Pathway_ID'] == pathway_id].iloc[0]
        projected_skills_match_ratio = min(projected_skills_match_ratio + pathway['Delta_P_Alignment_Skills_Match'], 1.0) # Assume 100% completion/mastery for comparison

    projected_alignment_score = projected_skills_match_ratio * current_alignment_for_role_comps['Timing_Factor'] # Timing factor remains unchanged
    
    projected_synergy = calculate_synergy(projected_vr, initial_hr, projected_alignment_score)
    projected_air = calculate_air(projected_vr, initial_hr, projected_synergy, alpha, beta)

    return projected_air, projected_vr, projected_synergy

# --- Define pre-configured learning plans for comparison ---
comparison_scenarios = {
    "Plan A: AI-Risk Analyst Focus (ML + Ethics)": {
        "target_role": "AI-Risk Analyst",
        "pathways": ["MLOps201", "FinRisk_ML", "Ethics_AI_Fin"],
        "description": "Deep dive into ML Ops, Risk ML, and AI Ethics."
    },
    "Plan B: Algorithmic Trader Focus (GenAI + Quant)": {
        "target_role": "Algorithmic Trading Engineer",
        "pathways": ["GenAI_Fin", "MLOps201"], # Using MLOps for trading too
        "description": "Focus on Generative AI for trading and MLOps."
    },
    "Plan C: Adaptive Capacity Boost (Flexibility + Strategy)": {
        "target_role": "AI-Risk Analyst", # Can apply to any, choose one for H_R calc
        "pathways": ["Cognitive_Flex", "Strategic_Career"],
        "description": "Strengthen meta-skills for better adaptability."
    }
}

scenario_results = []

for scenario_name, details in comparison_scenarios.items():
    role = details["target_role"]
    pathways = details["pathways"]
    
    # Need to pass initial alignment details that include the target role
    initial_alignment_details_for_scenario = alex_alignment_details[role].copy()
    initial_alignment_details_for_scenario['Target_Role'] = role

    projected_air, projected_vr, projected_synergy = get_scenario_results(
        alex_vr, ai_fluency_components, domain_expertise_components, adaptive_capacity_components,
        alex_air_scores[role], hr_scores[role], alex_synergy_scores[role],
        initial_alignment_details_for_scenario,
        pathways, default_alpha, default_beta
    )
    
    scenario_results.append({
        "Scenario": scenario_name,
        "Target Role": role,
        "Pathways": ", ".join(details["pathways"]),
        "Initial AI-R": alex_air_scores[role],
        "Projected AI-R": projected_air,
        "AI-R Change": projected_air - alex_air_scores[role],
        "Projected VR": projected_vr,
        "VR Change": projected_vr - alex_vr
    })

results_df = pd.DataFrame(scenario_results)
display(Markdown("### Comparison of Learning Pathway Scenarios:"))
display(results_df.round(2))

# --- Visualization for Scenario Comparison ---
fig = go.Figure()

fig.add_trace(go.Bar(
    x=results_df['Scenario'],
    y=results_df['Initial AI-R'],
    name='Initial AI-R',
    marker_color='lightgrey'
))
fig.add_trace(go.Bar(
    x=results_df['Scenario'],
    y=results_df['Projected AI-R'],
    name='Projected AI-R',
    marker_color=sns.color_palette("rocket")[0]
))

fig.update_layout(
    title="Comparison of AI-Readiness Score Across Different Learning Pathway Scenarios",
    yaxis_title="AI-Readiness Score (0-100)",
    xaxis_title="Learning Scenario",
    barmode='group'
)
fig.show()

# Visualize AI-R Change
fig_change = go.Figure()
fig_change.add_trace(go.Bar(
    x=results_df['Scenario'],
    y=results_df['AI-R Change'],
    name='AI-R Change',
    marker_color=results_df['AI-R Change'].apply(lambda x: 'green' if x > 0 else 'red')
))
fig_change.update_layout(
    title="Improvement in AI-Readiness Score by Scenario",
    yaxis_title="AI-R Change (Points)",
    xaxis_title="Learning Scenario"
)
fig_change.show()
```

**Explanation of Execution**

The comparison table and bar charts clearly present the trade-offs between different learning pathway strategies. Alex can see that "Plan A: AI-Risk Analyst Focus (ML + Ethics)" yields the highest $AI-R$ increase for his initial target role. This concrete data allows him to justify his learning investments to FinTech Innovators Inc. management, showing how his chosen path directly contributes to enhancing his AI-Readiness for high-value AI-centric roles. He can demonstrate not just an increase in skills, but a measurable improvement in his overall career opportunity score within the company.
