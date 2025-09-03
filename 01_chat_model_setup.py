
from langchain_google_genai import ChatGoogleGenerativeAI


from dotenv import load_dotenv


load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")


result = llm.invoke("what is the squre root of 49?")

print(result.content)

