from dotenv import load_dotenv
import os

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Set LangSmith project
os.environ['LANGCHAIN_PROJECT'] = 'sequential llm app'

# Load .env
load_dotenv()

# Prompt 1: Generate report
prompt1 = PromptTemplate(
    template="Generate a detailed report on {topic}",
    input_variables=["topic"]
)

# Prompt 2: Summarize text
prompt2 = PromptTemplate(
    template="Generate a 5 pointer summary from the following text:\n{text}",
    input_variables=["text"]
)

# Groq LLM (only one definition needed)
model = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.7,
)

# Output parser
parser = StrOutputParser()

# Chain: prompt1 → model → parser → prompt2 → model → parser
chain = prompt1 | model | parser | prompt2 | model | parser

# LangSmith config
config = {
    'tags': ['llm app', 'report generation', 'summarization'],
    'metadata': {
        'model': 'llama-3.3-70b-versatile',
        'model_temp': 0.7,
        'parser': 'StrOutputParser'
    }
}

# Run chain
result = chain.invoke({"topic": "Unemployment in India"}, config=config)

print(result)
