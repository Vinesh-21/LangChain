from langchain_core.messages import HumanMessage,SystemMessage,AIMessage

from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv

load_dotenv()


llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash",
                             temperature=0,)

messages = [
    SystemMessage(content="You are a helpful AI assistant."),
    HumanMessage(content="Write a haiku about the ocean."),
    # AIMessage(content="""Vast blue waters gleam,Waves crash on the sandy shore,Ocean's deep secrets.""")
]

result = llm.invoke(messages)

print(result.content)
