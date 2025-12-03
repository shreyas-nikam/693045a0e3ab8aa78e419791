# AI Career Navigator for Financial Professionals

![Streamlit Logo](https://streamlit.io/images/brand/streamlit-mark-color.svg)

## Project Title and Description

The **AI Career Navigator** is an interactive Streamlit application designed for financial professionals to assess their AI-Readiness Score ($AI-R$) and explore potential career pathways in the rapidly evolving landscape of AI-transformed finance.

This application implements a parametric framework to quantify an individual's preparedness for success in AI-enabled careers. It decomposes career opportunity into two core components:

1.  **Idiosyncratic Readiness ($V^R$)**: Represents individual-specific capabilities that can be actively developed through learning and skill enhancement, encompassing AI-Fluency, Domain-Expertise, and Adaptive-Capacity.
2.  **Systematic Opportunity ($H^R$)**: Captures macro-level job growth and demand that individuals can position themselves to capture, influenced by factors like AI enhancement potential, job growth projections, and market accessibility.

The tool provides an intuitive interface to:
*   Understand the components contributing to your $AI-R$.
*   Evaluate your current $V^R$ across key dimensions.
*   Analyze $H^R$ for different financial AI roles based on synthetic market data.
*   Simulate the impact of personalized learning pathways on your $AI-R$.
*   Compare various career paths to inform strategic career development.

This project serves as a lab exercise to demonstrate the practical application of a complex scoring model using Streamlit's interactive capabilities and data visualization tools.

## Features

*   **Interactive User Profile Input**: Adjust sliders and select boxes to define a hypothetical user's AI-Fluency, Domain-Expertise, and Adaptive-Capacity.
*   **Dynamic AI-Readiness Calculation**: Instantly calculates and displays the $AI-R$, $V^R$, $H^R$, and Synergy scores based on user inputs and chosen target role.
*   **Configurable Global Parameters**: Adjust the $\alpha$ (alpha) and $\beta$ (beta) parameters that govern the weighting of individual readiness versus market opportunity and synergy in the $AI-R$ formula.
*   **Idiosyncratic Readiness ($V^R$) Visualization**: A radar chart provides a clear breakdown of AI-Fluency, Domain-Expertise, and Adaptive-Capacity scores, scaled from 0-100.
*   **Systematic Opportunity ($H^R$) Visualization**: A bar chart illustrates the weighted contributions of AI-Enhancement, Job Growth, Wage Premium, and Entry Accessibility to the base $H^R$ for the selected target role.
*   **"What-If" Learning Pathway Simulation**: Simulate the impact of completing specific learning pathways on the $AI-R$ score, with adjustable completion and mastery levels.
*   **"What-If" Career Path Comparison**: Compare the $V^R$, $H^R$, Synergy, and $AI-R$ scores across up to three different AI-enabled financial roles side-by-side.
*   **Underlying Data Display**: View the synthetic user profile, market opportunities, and learning pathways dataframes used by the application.
*   **Performance Optimization**: Utilizes `@st.cache_data` to optimize data generation and function calculations, ensuring a smooth user experience.

## Getting Started

Follow these instructions to get a copy of the project up and running on your local machine.

### Prerequisites

Make sure you have the following installed:

*   **Python**: Version 3.8 or higher. You can download it from [python.org](https://www.python.org/downloads/).
*   **pip**: Python's package installer, usually comes with Python.

### Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/ai-career-navigator.git
    cd ai-career-navigator
    ```
    *(Note: Replace `your-username/ai-career-navigator.git` with the actual repository URL if this project is hosted.)*

2.  **Create a virtual environment** (recommended):
    ```bash
    python -m venv venv
    ```

3.  **Activate the virtual environment**:
    *   **On Windows**:
        ```bash
        .\venv\Scripts\activate
        ```
    *   **On macOS/Linux**:
        ```bash
        source venv/bin/activate
        ```

4.  **Install dependencies**:
    Create a `requirements.txt` file in the root directory with the following content:
    ```
    streamlit
    pandas
    numpy
    plotly
    ```
    Then, install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Run the Streamlit application**:
    Ensure your virtual environment is activated and you are in the project's root directory.
    ```bash
    streamlit run app.py
    ```

2.  **Access the application**:
    Once started, your web browser will automatically open to `http://localhost:8501` (or another port if 8501 is in use).

3.  **Interact with the UI**:
    *   Use the **sidebar** to adjust global parameters ($\alpha$, $\beta$) and navigate between pages (though currently only one main page exists).
    *   Adjust **sliders** and **select boxes** in the main content area to define your hypothetical user profile and target role.
    *   Observe the **dynamically updated scores** and **visualizations**.
    *   Explore the **"What-If" Scenarios** to simulate learning pathways and compare different career roles.

## Project Structure

```
.
├── application_pages/
│   └── ai_career_navigator_main.py  # Main logic and UI for the AI Career Navigator page
├── app.py                           # Main Streamlit entry point, handles page navigation
├── requirements.txt                 # List of Python dependencies
└── README.md                        # Project README file
```

*   `app.py`: This is the entry point for the Streamlit application. It sets up the page configuration, displays initial welcome text, and imports/runs the specific application pages based on sidebar navigation.
*   `application_pages/ai_career_navigator_main.py`: Contains the core logic for the AI Career Navigator. This includes data generation, all calculation functions (for $V^R$, $H^R$, $AI-R$), visualization functions, and the Streamlit UI elements for user interaction and display.

## Technology Stack

*   **Python 3.x**: The primary programming language.
*   **Streamlit**: The open-source app framework used to turn Python scripts into interactive web applications.
*   **Pandas**: For data manipulation and analysis, especially with dataframes for market opportunities and learning pathways.
*   **NumPy**: For numerical operations, particularly in generating synthetic data and various formula calculations.
*   **Plotly Express / Plotly Graph Objects**: For creating interactive and insightful data visualizations (radar charts, bar charts).

## Contributing

This is a lab project, but contributions are welcome! If you have suggestions for improvements, bug fixes, or new features, feel free to:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/AmazingFeature`).
3.  Make your changes.
4.  Commit your changes (`git commit -m 'Add some AmazingFeature'`).
5.  Push to the branch (`git push origin feature/AmazingFeature`).
6.  Open a Pull Request.

Please ensure your code adheres to good practices and includes appropriate comments.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
*(Note: You will need to create a `LICENSE` file in your repository if you choose the MIT license.)*

## Contact

For any questions or feedback, please reach out via the GitHub repository issues page or directly to the project maintainer.

*   **Project Link**: `https://github.com/your-username/ai-career-navigator` *(Replace with your actual repo link)*