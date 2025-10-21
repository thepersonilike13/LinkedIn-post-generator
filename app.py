from fastapi import FastAPI
from pydantic import BaseModel
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools import DuckDuckGoSearchResults
from bs4 import BeautifulSoup
import requests
import os
from dotenv import load_dotenv
# from fastapi.middleware.cors import CORSMiddleware

# origins = [
#     "http://localhost.tiangolo.com",
#     "https://localhost.tiangolo.com",
#     "http://localhost",
#     "http://localhost:8080",
# ]


# ------------------ Setup ------------------
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "")
os.environ['GRPC_VERBOSITY'] = 'NONE'

llm = init_chat_model("google_genai:gemini-2.5-flash-lite")

# ------------------ FastAPI ------------------
app = FastAPI(title="Demanual AI - LinkedIn Post Generator",
              version="1.0.0")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )         

# Request body model
class TopicRequest(BaseModel):
    topic: str

# ------------------ Helper functions ------------------
def scrape_page(url):
    """Scrape the main textual content from a single page."""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=5)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        
        article = soup.find("article")
        if article:
            text = " ".join(p.get_text(strip=True) for p in article.find_all("p"))
        else:
            text = " ".join(p.get_text(strip=True) for p in soup.find_all("p"))
        
        return text
    except Exception as e:
        print(f"Failed to scrape {url}: {e}")
        return None

def scrape_links_fast(search_results):
    """Scrape all links from search results quickly (no recursion)."""
    scraped_data = {}
    for result in search_results:
        url = result.get("link")
        if url:
            content = scrape_page(url)
            scraped_data[url] = content
    return scraped_data

# ------------------ Endpoint -----------------
@app.get("/")
def read_root():
    return {"message": "Welcome to the Demanual AI LinkedIn Post Generator API! It works! (for now....)"}

@app.post("/generate-post")
def generate_post(request: TopicRequest):
    topic = request.topic

    # 1️⃣ Search for recent news/articles
    search = DuckDuckGoSearchResults(output_format="list")
    search_results = search.invoke(topic)

    # 2️⃣ Scrape content from links
    scraped_dict = scrape_links_fast(search_results)

    # 3️⃣ Combine text for LLM
    all_texts = [content for content in scraped_dict.values() if content]
    combined_text = "\n\n---\n\n".join(all_texts)

    # 4️⃣ LLM system + prompt templates
    system_template = """
    You are a professional content writer specialized in creating engaging LinkedIn posts.
    Your goal is to summarize multiple news articles into a concise, professional, and readable post.
    Follow these rules:
    - Summarize the main points from all articles.
    - Keep it professional, insightful, and in a very engaging tone.
    - Structure:
    - Use 2–3 paragraphs minimum write it big.
    - Highlight key facts or quotes.
    - Avoid copying verbatim; paraphrase and make it readable.
    - At the end, suggest a short concept for an image or illustration relevant to the post (1 sentence). it should be very creative and catchy.
    eg: [image suggestion: A futuristic cityscape powered by renewable energy, with wind turbines and solar panels integrated into sleek skyscrapers under a bright blue sky.]
    give me only the post and the image suggestion in the format below:
    LinkedIn Post:
    [post]
    Image Suggestion: 
    [image_suggestion]

    """

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("user", """
Here are the news articles on the topic "{topic}":

{articles}

Please generate a professional LinkedIn post summarizing these articles.
""")
    ])

    prompt = prompt_template.invoke({"topic": topic, "articles": combined_text})

    # 5️⃣ Get LLM response
    response = llm.invoke(prompt)
    linkedin_post = response.content

    # 6️⃣ Return JSON
    return {
        "topic": topic,
        "news_sources": list(scraped_dict.keys()),
        "linkedin_post": linkedin_post,
        # "image_suggestion": None  # Optional: parse from LLM if you include image instruction
    }

# ------------------ Run (for local testing) ------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
