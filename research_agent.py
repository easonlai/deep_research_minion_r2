import streamlit as st
from langchain_openai.chat_models import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_community.utilities import SerpAPIWrapper

# Streamlit App Layout
st.set_page_config(page_title="AI Research Agent 🔎🌏 with Thinking Process 🧠", layout="wide")
st.title("AI Research Agent 🔎🌏 with Thinking Process 🧠")
st.markdown("""
Enter a research topic below to start the deep research process. The AI will perform iterative research,
displaying its thinking process step-by-step, including search queries, summaries, and critiques.
""")

# User Inputs for API Configuration
st.sidebar.header("Azure OpenAI API Configuration")
user_openai_api_key = st.sidebar.text_input("Azure OpenAI API Key", value="", type="password", help="Enter your Azure OpenAI API Key.")
user_azure_endpoint = st.sidebar.text_input("Azure OpenAI Endpoint", value="https://example.openai.azure.com", help="Enter your Azure OpenAI Endpoint.")
user_azure_deployment = st.sidebar.text_input("Azure OpenAI Model Deployment Name", value="gpt-4o", help="Enter your Azure OpenAI Model Deployment Name.")
user_openai_api_version = st.sidebar.text_input("Azure OpenAI API Version", value="2025-01-01-preview", help="Enter the OpenAI API Version.")

# Validate API Configuration
if not user_openai_api_key or not user_azure_endpoint or not user_azure_deployment or not user_openai_api_version:
    st.sidebar.error("Please provide all Azure OpenAI API configuration details.")

# Use AzureChatOpenAI for GPT-4 chat completions
llm = AzureChatOpenAI(
    openai_api_key=user_openai_api_key,
    azure_endpoint=user_azure_endpoint,
    azure_deployment=user_azure_deployment,
    openai_api_type="azure",
    openai_api_version=user_openai_api_version,
    temperature=0
)

# Initialize SerpAPI for web searches
search = SerpAPIWrapper(serpapi_api_key=st.secrets["SERPAPI_API_KEY"])

# Define Prompt Templates for each step
query_template = PromptTemplate(
    input_variables=["query"],
    template="Given the research topic '{query}', generate 3 specific search queries to find relevant information on the internet."
)

summary_template = PromptTemplate(
    input_variables=["query", "search_results"],
    template="Based on the following web search results, provide a concise summary relevant to the topic '{query}':\n{search_results}"
)

critique_template = PromptTemplate(
    input_variables=["query", "summary"],
    template="Review the following summary for the topic '{query}':\n{summary}\nIdentify any gaps, inconsistencies, or areas that need further exploration. If more research is needed, suggest specific questions or topics to investigate next. If the summary is comprehensive, state 'No further research needed'."
)

new_queries_template = PromptTemplate(
    input_variables=["critique"],
    template="Based on the following critique, generate 3 new search queries to address the identified gaps:\n{critique}"
)

update_summary_template = PromptTemplate(
    input_variables=["summary", "new_search_results"],
    template="Update the following summary with the new information from the search results:\nCurrent summary: {summary}\nNew search results: {new_search_results}"
)

# Build chains using PromptTemplate | llm syntax
query_chain = query_template | llm
summary_chain = summary_template | llm
critique_chain = critique_template | llm
new_queries_chain = new_queries_template | llm
update_summary_chain = update_summary_template | llm

# User Inputs for Research
query = st.text_input("Research Topic:", "Impact of AI on healthcare")
max_iterations = st.slider("Maximum Iterations", 1, 5, 3, help="Set the maximum number of research iterations.")

# Research Process Trigger
if st.button("Start Research"):
    # Initialize a cache for search results to avoid redundant API calls
    search_cache = {}

    # Function to perform web searches with caching
    def search_tool(queries):
        results = []
        for q in queries:
            if q not in search_cache:
                try:
                    search_cache[q] = search.run(q)
                except ValueError as e:
                    st.write(f"SerpAPI error for query '{q}': {e}")
                    search_cache[q] = f"No results found for query: {q}"
            results.append(search_cache[q])
        return "\n".join(results)

    # Initial Research Phase
    with st.spinner("Performing initial research..."):
        st.markdown("### Initial Research")
        st.write("Generating search queries...")
        query_response = query_chain.invoke({"query": query})
        query_text = query_response.content.strip()
        search_queries = []
        for line in query_text.split("\n"):
            parts = line.split(". ", 1)
            if len(parts) == 2:
                search_queries.append(parts[1])

        st.write(f"Search queries: {', '.join(search_queries)}")

        # Fetch data from the web using SerpAPI
        st.write("Fetching data from the web...")
        search_results = search_tool(search_queries)

        # Summarize data
        st.write("Summarizing data...")
        summary_response = summary_chain.invoke({"query": query, "search_results": search_results})
        summary_text = summary_response.content
        st.write(f"**Initial Summary**: {summary_text}")

    # Iterative Research Loop
    for i in range(max_iterations):
        with st.spinner(f"Performing iteration {i+1}..."):
            st.markdown(f"### Iteration {i+1}")
            st.write("Critiquing current summary...")
            critique_response = critique_chain.invoke({"query": query, "summary": summary_text})
            critique_text = critique_response.content
            st.write(f"**Critique**: {critique_text}")

            if "No further research needed" in critique_text:
                st.write("Research is complete based on the critique.")
                break
            else:
                st.write("Generating new search queries...")
                new_queries_response = new_queries_chain.invoke({"critique": critique_text})
                new_queries_text = new_queries_response.content.strip()
                new_queries = []
                for line in new_queries_text.split("\n"):
                    parts = line.split(". ", 1)
                    if len(parts) == 2:
                        new_queries.append(parts[1])

                st.write(f"New search queries: {', '.join(new_queries)}")

                st.write("Fetching new data from the web...")
                new_search_results = search_tool(new_queries)

                st.write("Updating summary with new information...")
                updated_summary_response = update_summary_chain.invoke({
                    "summary": summary_text,
                    "new_search_results": new_search_results
                })
                summary_text = updated_summary_response.content
                st.write(f"**Updated Summary**: {summary_text}")

    # Display Final Report
    st.markdown("### Final Report")
    st.write("Here is the comprehensive summary after all research iterations:")
    st.write(summary_text)