import os
import uuid
import gradio as gr
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.utils import filter_complex_metadata
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import logging

load_dotenv()

os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
print(os.environ['LANGCHAIN_API_KEY']) 
print(os.environ["OPENAI_API_KEY"])

embeddings_model = HuggingFaceEmbeddings(model_name="HIT-TMG/KaLM-embedding-multilingual-mini-instruct-v1.5")

model = AutoModelForSequenceClassification.from_pretrained("facebook/bart-large-mnli")


classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def detect_intent(text):
    result = classifier(text, candidate_labels=["question", "greeting", "small talk", "feedback", "thanks"])
    label = result["labels"][0]
    return label.lower()

chroma_db_path = "./chroma_db"
chroma_client = chromadb.PersistentClient(path=chroma_db_path)

data = chroma_client.get_collection(name="my_dataaaa")
vectorstore = Chroma(
    collection_name="my_dataaaa",  
    persist_directory="./chroma_db",
    embedding_function=embeddings_model
)

#Create a retriever from chroma DATASTORE 
retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={'k': 6, 'lambda_mult': 0.25}
    )

reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def rerank_docs(query, docs, top_k=50):
    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)
    scored_docs = list(zip(docs, scores))
    scored_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)
    top_docs = [doc for doc, score in scored_docs[:top_k]]
    return top_docs
custom_prompt = PromptTemplate.from_template("""
You are a helpful assistant answering student questions based ONLY on the provided context.
You must read the entire context carefully and include all relevant information in your answer.
If multiple documents or requirements are mentioned, list them all clearly and completely.
If the answer is not found in the context, respond with: "je ne trouve pas la réponse."
Do not use your own knowledge for university-related questions. Only use what is in the context.
Structure the answer clearly and completely. Do not make any assumptions if the context does not have the answer.

Context:
{context}

Question:
{question}

Answer:
""")

llm = ChatOpenAI(model="gpt-3.5-turbo")

def format_docs(docs):
     return "\n\n".join(doc.page_content for doc in docs)

context = format_docs(docs)
context

rag_chain = (
    {
        "context": retriever
        | (lambda docs: rerank_docs(docs=docs, query="{question}"))  
        | format_docs,
        "question": RunnablePassthrough()
    }
    | custom_prompt
    | llm
    | StrOutputParser()
)


PENDING_QUESTIONS_FILE = "pending_questions.json"

def store_pending_question(user_email, question):
    q_id = str(uuid.uuid4())
    pending = {
        "id": q_id,
        "timestamp": datetime.utcnow().isoformat(),
        "user_email": user_email,
        "question": question
    }
    if os.path.exists(PENDING_QUESTIONS_FILE):
        with open(PENDING_QUESTIONS_FILE, "r") as f:
            data = json.load(f)
    else:
        data = []

    data.append(pending)
    with open(PENDING_QUESTIONS_FILE, "w") as f:
        json.dump(data, f, indent=4)
    return q_id



def send_question_to_admin(user_email, user_question,question_id):
    admin_email = "belhassen.esra@icloud.com"
    smtp_server = "smtp.gmail.com"
    smtp_port = 587
    sender_email = "fsmchatbot@gmail.com"
    sender_password = os.getenv("BOT_EMAIL_PASSWORD")  

    subject = f"Nouvelle question [{question_id}] "
    body = (
        f"Question ID: {question_id}\n"
        f"Question posée :\n\n{user_question}"
    )

    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = admin_email
    message["Reply-To"] = "fsmchatbot@gmail.com"
    message["Subject"] = subject
    
    message.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, admin_email, message.as_string())
        return True
    except Exception as e:
        print("Error sending email:", e)
        return False


def university_related(question):
    labels = ["university", "general knowledge"]
    result = classifier(question, candidate_labels=labels)
    top_label = result["labels"][0]
    return top_label.lower() == "university"

def uncertain(answer):
    uncertain_phrases = [
        "je ne trouve pas la réponse",
        "désolé, je ne peux pas vous aider"
    ]
    return any(phrase in answer.lower() for phrase in uncertain_phrases) or answer.strip() == ""

def handle_user_query(question, user_email=None):
    # using the classifier model
    intent = detect_intent(question.lower())

    if intent in ["greeting", "small talk"]:
        return "Salut 👋 ! Posez-moi une question précise sur les procédures universitaires 😊."
    if not university_related(question):
        return "Merci de poser une question sur les procédures universitaires 😊"
    # integration de RAG Pipeline
    answer = rag_chain.invoke(question)

    # making the llama know what to do if there are no relevant docs
    if uncertain(answer):
        if not user_email:
            return (
                "Je ne trouve pas la réponse à cette question. "
                "Veuillez me fournir votre adresse e-mail et la question en français pour que je puisse la transmettre à un administrateur.")
        
        q_id = store_pending_question(user_email, question)
        sent = send_question_to_admin(user_email, question, q_id)
        
        if sent:
            return "Votre question a été transmise à l'administration. Vous recevrez une réponse par e-mail dès que possible."
        else:
            return "Une erreur est survenue lors de l'envoi de votre question. Veuillez réessayer plus tard."
    else:
        return answer


user_email = ""

def chatbot_fn(message, history):
    global user_email
    if not user_email:
        if "@gmail.com" in message or "@fsm.rnu.tn" in message:
            user_email = message
            return "Merci ! Maintenant, posez-moi votre question 😊"
        else:
            return "Bienvenue 👋 Veuillez entrer votre adresse e-mail pour commencer."
    
    return handle_user_query(message, user_email)

with gr.Blocks() as chat:
    gr.ChatInterface(
        fn=chatbot_fn,
        title="Chatbot Universitaire 🤖 🧠",
        description="Commencez par entrer votre adresse e-mail. Ensuite, posez toutes vos questions sur les procédures universitaires !",
        examples=[
            ["Comment faire une demande de réinscription ?"],
            ["Quels sont les délais pour la soutenance ?"]
        ],
        submit_btn="Envoyer"
    )
    gr.Markdown("© 2025 Esra Belhassen. All rights reserved")

chat.launch(share=True)

