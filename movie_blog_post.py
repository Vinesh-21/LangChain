### LLM Wrapper/Chat Model
from langchain_google_genai import ChatGoogleGenerativeAI

# Output Parsing
from langchain_core.output_parsers import StrOutputParser

### Prompt Templating
from langchain_core.prompts import ChatPromptTemplate

### Chaining
from langchain_core.runnables import RunnableParallel, RunnableLambda

# Loading Env Variables
from dotenv import load_dotenv
load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# Prompt for summary
summary_template = ChatPromptTemplate.from_messages([
    ("system", "you are a movie critic."),
    ("human", "Provide a brief summary of the movie {movie}")
])

# --- Plot analysis template ---


def plot_analysis_template(summary) :
    template = ChatPromptTemplate.from_messages([
    ("system", "you are a movie critic."),
    ("human", "Analyze the plot of the movie:\n{summary}\nProvide strengths and weaknesses.")])

    return template.format_prompt(summary=summary)


plot_analysis_chain = RunnableLambda(lambda x:plot_analysis_template(x)) | model | StrOutputParser()

# --- Character analysis template ---

def character_analysis_template(summary):
    template = ChatPromptTemplate.from_messages([
    ("system", "you are a movie critic."),
    ("human", "Analyze the characters of the movie:\n{summary}\nProvide strengths and weaknesses.")])

    return template.format_prompt(summary=summary)

character_analysis_chain = RunnableLambda(lambda x: character_analysis_template(x)) | model | StrOutputParser()

# --- Combine outputs ---
def combine_analyses(plot_analysis, character_analysis):
    return f"Plot Analysis:\n{plot_analysis}\n\nCharacter Analysis:\n{character_analysis}"

# --- Main chain ---
chain = (
    summary_template
    | model
    | StrOutputParser()
    | RunnableParallel({
        "plot_analysis": plot_analysis_chain,
        "character_analysis": character_analysis_chain
    })
    | RunnableLambda(lambda x: combine_analyses(
        plot_analysis=x["plot_analysis"],
        character_analysis=x["character_analysis"]
    ))
)

# Run it
print(chain.invoke({"movie": "Inception"}))
