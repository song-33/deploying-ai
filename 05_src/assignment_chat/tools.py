import requests
import chromadb
from chromadb.utils import embedding_functions
from langchain_core.tools import tool
import os
from ddgs import DDGS

# Service 1: Random Cocktail API
@tool
def get_random_cocktail() -> str:
    """
    Fetches a random cocktail recipe.
    Use this tool whenever the user asks for a random drink, a surprise, or doesn't know what to order.
    """
    url = "https://www.thecocktaildb.com/api/json/v1/1/random.php"
    response = requests.get(url)

    if response.status_code != 200:
        return "Sorry, I couldn't fetch a cocktail recipe at the moment."
    
    data = response.json()
    drink = data['drinks'][0]

    name = drink.get('strDrink','Unknown')
    glass = drink.get('strGlass','Unknown')
    instructions = drink.get('strInstructions','')

    # Pair ingredients with their measures
    ingredients_list = []
    for i in range(1,16):
        ingredient = drink.get(f'strIngredient{i}')
        measure = drink.get(f'strMeasure{i}')
        if ingredient:
            measure_str = f'({measure.strip()})' if measure else ''
            ingredients_list.append(f'{ingredient}{measure_str}')

    ingredients_str = ', '.join(ingredients_list)

    return f'Name: {name}\nGlass: {glass}\nIngredients: {ingredients_str}\nInstructions: {instructions}'

# Service 2: Semantic Query
@tool
def search_cocktails_by_flavor(query: str) -> str:
    """
    Searches the cocktail database for drinks matching a specific floavor profile, ingredient, or description (e.g., "citrusy", "strong rum drink", "sweet and sour").
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    docs_dir = os.path.join(current_dir, 'documents')
    chroma_path = os.path.join(docs_dir, 'chroma_db')
    default_ef = embedding_functions.DefaultEmbeddingFunction()
    chroma_client = chromadb.PersistentClient(path=chroma_path)
    collection = chroma_client.get_collection(name='cocktails', embedding_function = default_ef)

    results = collection.query(
        query_texts=[query],
        n_results=3
    )
    if not results['documents'][0]:
        return "Sorry, I couldn't find any cocktails matching that description."

    formatted_results = []
    for i,doc in enumerate(results['documents'][0]):
        formatted_results.append(f'Option{i+1}:\n{doc}\n')

    return "\n".join(formatted_results)

# Service 3: Web Search (requires instally DuckDuckGoSearch)
@tool
def search_cocktail_history(query: str) -> str:
    """
    Searches the web for historical facts, origins, or general trivia about cocktails.
    Returns the top 3 search results including titles, snippets, and source URLs.
    """
    try:
        results = DDGS().text(query, max_results=3)
        if not results:
            return "No results found."
        
        formatted_results = []
        for i,r in enumerate(results):
            title = r.get('title','Unknown title')
            snippet = r.get('body', 'No snippet available.')
            link = r.get('href', 'No link available.')

            formatted_results.append(
                f'Result {i+1}:\n'
                f'Title: {title}\n'
                f'Snippet: {snippet}\n'
                f'Source link: {link}\n'
            )
        return '\n'.join(formatted_results)
    except Exception as e:
        return f'Web search failed: {e}'

# Toolbox
tools = [get_random_cocktail, search_cocktails_by_flavor, search_cocktail_history]
tools_by_name = {tool.name: tool for tool in tools}
