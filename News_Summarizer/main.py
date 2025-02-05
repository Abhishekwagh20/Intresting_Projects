import openai
from dotenv import find_dotenv, load_dotenv
from datetime import datetime
import os
import requests
import json

load_dotenv()


news_api_key = os.environ.get("NEWS_API_Key")
client = openai.OpenAI()

def get_news(topic):
    """Get news information based on given topic."""
    url = (
        f"https://newsapi.org/v2/everything?q={topic}&apikey={news_api_key}&pageSize=5"
    )
    try:
        response = requests.get(url)
        
        news = json.dumps(response.json(), indent=3)
        news_json = json.loads(news) 
        data = news_json
        
        articles = data["articles"]
        final_news = []
        
        for article in articles:
            source_name = article["source"]["name"]
            author = article["author"]
            title = article["title"]
            description = article["description"]
            url = article["url"]
            
            title_description = f""" 
                Title: {title},
                Author: {author},
                source: {source_name},
                Description: {description},
                URL: {url}
            """
            final_news.append(title_description)
        return final_news
            
            
    except Exception as e:
        print("Something went wrong",e)
        
        
def news_summarizer(topic):
    messages = [{'role': 'user', 'content': topic}]
    functions = [
    {
        "name": "get_news",
        "description": "Get news title, author, description and url which is similar to the news topic provided by the user ",
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "news topic provided by the user",
                },
            },
            "required": ["topic"],
        },
    }
    ]
    
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        functions=functions,
        function_call="auto",
        max_tokens=200,
    )
    
    response_message = response.choices[0].message
    
    if response_message.function_call:
        available_functions = {
            "get_news": get_news
        }
        function_name = response_message.function_call.name
        function_to_call = available_functions[function_name]
        function_args = json.loads(response_message.function_call.arguments)
        function_response = function_to_call(function_args)

        messages.append(response_message.dict()) 
        messages.append(
            {
                "role": "function",
                "name": function_name,
                "content": str(function_response),
            }
        )
        second_response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
        )
        return second_response.choices[0].message.content
    else:
        return response_message["content"]

if __name__ == '__main__':
    print(news_summarizer("GenAI"))