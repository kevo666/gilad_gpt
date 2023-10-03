urls = [
    "https://www.pollenstreetgroup.com/latest-news/in-the-trenches-of-digital-transformation","https://www.pollenstreetgroup.com/latest-news/digital-transformation-in-private-equity-beyond-the-trenches/","https://www.pollenstreetgroup.com/latest-news/building-a-digital-lending-system-methods-learnings-and-insights/","https://www.pollenstreetgroup.com/latest-news/building-a-digital-lending-system-from-origination-to-monitoring/","https://www.elsewhen.com/blog/finding-strategic-clarity-in-the-new-financial-services-reality/","https://medium.com/@gilad.amir/dispelling-the-small-business-cashflow-myth-66f76a6f734","https://www.linkedin.com/pulse/security-tokens-back-future-4-gilad-g-amir/","https://www.pollenstreetgroup.com/latest-news/the-human-side-of-digital-transformation-building-a-culture-of-digital-success/"
]

from langchain.document_loaders import SeleniumURLLoader
loader = SeleniumURLLoader(urls=urls)

data = loader.load()

just_text = [d.page_content for d in data]

from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI

retriever_descriptions = []

for d in just_text:
    prompt = PromptTemplate.from_template("""You are an AI coordinating assistant.
    You are part of a team generating data for an AI to impersonate Gilad.
    Your task is to write a utility of what the following document Gilad created will be good for so that other AI assistants will know what information they can gain from it. 
    
    {document}
    
    Below are some examples of the output format:
    Good for answering questions about the 2023 State of the Union address.
    Good for answering questions about Paul Graham's essay on his career
    Good for answering questions about me
    
    Your output should just be "Good for..."
    """)
    #llm = ChatOpenAI(model='gpt-3.5-turbo-16k')
    #retriever_descriptions.append(llm.predict(prompt.format(document=d)))
    retriever_descriptions.append("")
 
for i,t in enumerate(just_text):
    with open(f"document_{i}.txt","w") as f:
        f.write(t)

from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.vectorstores import FAISS

retriever_infos = []
for i,t in enumerate(just_text):
    retriever_infos.append(
    {
        "name": urls[i], 
        "description": retriever_descriptions[i], 
        "retriever": FAISS.from_documents(TextLoader(f'document_{i}.txt').load_and_split(), OpenAIEmbeddings()).as_retriever()
    }
    )

from langchain.document_transformers import LongContextReorder

from langchain.chains.router import MultiRetrievalQAChain
from langchain.retrievers.merger_retriever import MergerRetriever
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_transformers import (
    EmbeddingsRedundantFilter,
    EmbeddingsClusteringFilter,
)
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.retrievers import ContextualCompressionRetriever

lotr = MergerRetriever(retrievers=[r["retriever"] for r in retriever_infos])
filter_embeddings = OpenAIEmbeddings()

reordering = LongContextReorder()
# If you want the final document to be ordered by the original retriever scores
# you need to add the "sorted" parameter.
filter_ordered_by_retriever = EmbeddingsClusteringFilter(
    embeddings=filter_embeddings,
    num_clusters=10,
    num_closest=1,
    sorted=True,
)

pipeline = DocumentCompressorPipeline(transformers=[reordering, filter_ordered_by_retriever])
compression_retriever = ContextualCompressionRetriever(
    base_compressor=pipeline, base_retriever=lotr
)

retriever = {"name": "gilad_info", "description":"Good for answering all questions", "retriever":compression_retriever}

from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain.agents import ConversationalChatAgent, AgentExecutor
from langchain.callbacks import StreamlitCallbackHandler

st.set_page_config(page_title="GiladGPT: Chat with Gilad", page_icon="🦜")
st.title("🦜 GiladGPT: Chat with Gilad")

msgs = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(
    chat_memory=msgs, return_messages=True, memory_key="chat_history", output_key="output"
)
if len(msgs.messages) == 0 or st.sidebar.button("Reset chat history"):
    msgs.clear()
    msgs.add_ai_message("Hi, I'm GiladGPT a bot of Gilad, How can I help you?")
    st.session_state.steps = {}

avatars = {"human": "user", "ai": "assistant"}

for idx, msg in enumerate(msgs.messages):
    with st.chat_message(avatars[msg.type]):
        # Render intermediate steps if any were saved
        for step in st.session_state.steps.get(str(idx), []):
            if step[0].tool == "_Exception":
                continue
            with st.status(f"**{step[0].tool}**: {step[0].tool_input}", state="complete"):
                st.write(step[0].log)
                st.write(step[1])
        st.write(msg.content)

chain = MultiRetrievalQAChain.from_retrievers(ChatOpenAI(model='gpt-3.5-turbo-16k'), [retriever], verbose=True)
from langchain.tools import Tool
if prompt := st.chat_input(placeholder="what are the keys to successful digital transformation in financial services?"):
    st.chat_message("user").write(prompt)

    #if not openai_api_key:
    #    st.info("Please add your OpenAI API key to continue.")
    #    st.stop()

    tools = [Tool.from_function(
        func=chain.run,
        name="gilad_info",
        description="useful for answering all questions"
    )]
    executor = AgentExecutor.from_agent_and_tools(
        agent=ConversationalChatAgent.from_llm_and_tools(llm=ChatOpenAI(model='gpt-3.5-turbo-16k'), tools=tools),
        tools=tools,
        memory=memory,
        return_intermediate_steps=True,
        handle_parsing_errors=True,
    )
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = executor(prompt, callbacks=[st_cb])
        st.write(response["output"])
        st.session_state.steps[str(len(msgs.messages) - 1)] = response["intermediate_steps"]
