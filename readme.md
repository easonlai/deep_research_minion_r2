# AI Research Assistant R2 🔎🌏 with Thinking Process 🧠

## Overview
The AI Research Assistant R2 🔎🌏 with Thinking Process 🧠 (Deep Research Minion R2) is an AI-powered research assistant (AI Agent) application designed to facilitate deep research processes. It leverages Azure OpenAI for natural language processing and SerpAPI for web searches, providing users with an interactive interface to explore research topics iteratively.

You need to provide a research topic and configure their Azure OpenAI API settings in the sidebar. The AI Agent generates search queries, retrieves information from the web using SerpAPI, and summarizes the findings. It iteratively critiques the summary, identifies gaps, and generates new queries to refine the research until it reaches a comprehensive conclusion or the user-defined iteration limit. Throughout the process, the program transparently displays its thinking steps, including search queries, summaries, critiques, and updates, culminating in a detailed final report.

## Features
- Streamlit-based user interface for easy interaction.
- Iterative research process that includes generating search queries, summarizing results, and critiquing findings.
- Integration with Azure OpenAI for advanced language processing capabilities.
- Utilizes SerpAPI for real-time web searches to gather relevant information.

## Installation
To set up the project, follow these steps:

1. Clone the repository:
   ```
   git clone <repository-url>
   cd deep_research_minion_r2
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Configure your Azure OpenAI API settings in the Streamlit sidebar when you run the application.

## Usage
To run the application, execute the following command in your terminal:
```
streamlit run src/research_agent.py
```

Once the application is running, you can enter a research topic, configure your API settings, and start the research process. The AI will guide you through generating queries, summarizing results, and refining your research iteratively.

## Contributing
Contributions are welcome! If you have suggestions for improvements or new features, please open an issue or submit a pull request.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.

## Acknowledgements

- [Streamlit](https://streamlit.io/)
- [Azure OpenAI](https://azure.microsoft.com/en-us/products/ai-services/openai-service/)
- [SerpAPI](https://serpapi.com/)