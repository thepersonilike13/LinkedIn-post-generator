from fastapi import FastAPI
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langserve import add_routes 
import uvicorn 
import os
from langchain_core.messages import HumanMessage, SystemMessage
import getpass
from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchResults
import requests
# from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from bs4 import BeautifulSoup


load_dotenv()
os.environ["GOOGLE_API_KEY"] =os.getenv("GOOGLE_API_KEY", "")
print("GOOGLE_API_KEY:", os.environ["GOOGLE_API_KEY"][:8] + "..." if os.environ["GOOGLE_API_KEY"] else "Not Set")
os.environ['GRPC_VERBOSITY'] = 'NONE'



llm = init_chat_model("google_genai:gemini-2.5-flash-lite")


# ------------------------------------- test -------------------------------------

# system_template = "Translate the following from English into {language}"

# prompt_template = ChatPromptTemplate.from_messages(
#     [("system", system_template), ("user", "{text}")]
# )

# prompt = prompt_template.invoke({"language": "Italian", "text": "hi!"})

# print(prompt.to_messages())

# response = llm.invoke(prompt)
# print(response.content)


def scrape_page(url):
    """Scrape the main textual content from a single page."""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=5)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        
        # Try to get main article content first
        article = soup.find("article")
        if article:
            text = " ".join(p.get_text(strip=True) for p in article.find_all("p"))
        else:
            # fallback: all <p> text
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



topic = "Latest advancements in renewable energy technologies 2024"

search = DuckDuckGoSearchResults(output_format="list")
pit = search.invoke(topic)
print(pit)

scraped_dict = scrape_links_fast(pit)

# for link, content in val.items():
#     print(f"URL: {link}\nContent length: {len(content) if content else 'Failed'}\n")
#     print(f"Content Preview: {content[:200]}...\n")


all_texts = [content for content in scraped_dict.values() if content]

combined_text = "\n\n---\n\n".join(all_texts)

print(combined_text)
print("x"*50)
print(f"len:{len(combined_text)}")




system_template = """
You are a professional content writer specialized in creating engaging LinkedIn posts.
Your goal is to summarize multiple news articles into a concise, professional, and readable post.
Follow these rules:
- Summarize the main points from all articles.
- Keep it professional, insightful, and engaging.
- Use 2–3 paragraphs maximum.
- Highlight key facts or quotes.
- Avoid copying verbatim; paraphrase and make it readable.
- Optionally suggest a short image concept at the end (1 sentence).
"""

prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_template),
    ("user", """
Here are the news articles on the topic "{topic}":

{articles}

Please generate a professional LinkedIn post summarizing these articles.
""")
])

prompt = prompt_template.invoke({"topic": {topic}, "articles": {combined_text}})

print(prompt.to_messages())

response = llm.invoke(prompt)
print(response.content)
