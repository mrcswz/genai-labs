import os
import yaml
import logging
import google.cloud.logging
from flask import Flask, render_template, request

from google.cloud import firestore
from google.cloud.firestore_v1.vector import Vector
from google.cloud.firestore_v1.base_vector_query import DistanceMeasure
from google.cloud.aiplatform_v1beta1.types.content import SafetySetting, HarmCategory

import vertexai
from vertexai.generative_models import GenerativeModel
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel
from langchain_google_vertexai import VertexAIEmbeddings

# Configure Cloud Logging
logging_client = google.cloud.logging.Client()
logging_client.setup_logging()
logging.basicConfig(level=logging.INFO)

# Read application variables from the config fle
BOTNAME = "FreshBot"
SUBTITLE = "Your Friendly Restaurant Safety Expert"

app = Flask(__name__)

# Initializing the Firebase client
db = firestore.Client()

# Instantiate a collection reference
collection = db.collection('food-safety')

# Instantiate an embedding model here
embedding_model = VertexAIEmbeddings(model_name="text-embedding-004")

safety_config = [
    SafetySetting(
        category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold=SafetySetting.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    )
]

# Instantiate a Generative AI model here
generation_config = {
    "temperature": 0,
}

generative_model = GenerativeModel(
    "gemini-pro",
    generation_config=generation_config
)

# Return relevant context from your vector database
def search_vector_database(query: str):

    context = ""

    query_embedding = embedding_model.embed_query(query)
  
    vector_query = collection.find_nearest(
        vector_field='embedding',
        query_vector=Vector(query_embedding),
        distance_measure=DistanceMeasure.EUCLIDEAN,
        limit=5,
    )
  
    docs = vector_query.stream()
  
    context = [result.to_dict()['content'] for result in docs]

    logging.info(
        context, extra={"labels": {"service": "cymbal-service", "component": "context"}}
    )
    return context

# Pass Gemini the context data, generate a response, and return the response text.
def ask_gemini(question):

    prompt_template = """
    You are an AI assistant. Use the provided context to answer the question as accurately and concisely as possible.
    
    Question: {question}
    Context: {context}
    
    Answer:
    """

    context = search_vector_database(question)

    formatted_prompt = prompt_template.format(question=question, context=context)
    
    response = generative_model.generate_content(formatted_prompt)
    
    return response

# The Home page route
@app.route("/", methods=["POST", "GET"])
def main():

    # The user clicked on a link to the Home page
    # They haven't yet submitted the form
    if request.method == "GET":
        question = ""
        answer = "Hi, I'm FreshBot, what can I do for you?"

    # The user asked a question and submitted the form
    # The request.method would equal 'POST'
    else:
        question = request.form["input"]
        # Do not delete this logging statement.
        logging.info(
            question,
            extra={"labels": {"service": "cymbal-service", "component": "question"}},
        )
        
        # Ask Gemini to answer the question using the data
        # from the database
        answer = ask_gemini(question)

    # Do not delete this logging statement.
    logging.info(
        answer, extra={"labels": {"service": "cymbal-service", "component": "answer"}}
    )
    print("Answer: " + answer)

    # Display the home page with the required variables set
    config = {
        "title": BOTNAME,
        "subtitle": SUBTITLE,
        "botname": BOTNAME,
        "message": answer,
        "input": question,
    }

    return render_template("index.html", config=config)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
