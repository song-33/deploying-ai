import gradio as gr
from agent import agent

def chat_with_agent(message, history):
    """
    This function acts as the bridge between Gradio and LangGraph.
    """
    formatted_messages = []
    for msg in history:
        formatted_messages.append((msg["role"], msg["content"]))
    
    formatted_messages.append(('user', message))

    response_state = agent.invoke({'messages': formatted_messages})

    final_message = response_state['messages'][-1].content

    return final_message

chat = gr.ChatInterface(
    fn = chat_with_agent,
    type = 'messages',
    title = 'The AI Mixologist',
    description = 'Ask me for a random cocktail, search my recipe book by flavor, or dive into cocktail history!',
    examples = [
        'Surprise me with a random drink!',
        'Find me a fruity and sweet drink with rum.',
        'Who invented the Moscow Mule?'
    ],
    theme = 'ocean'
)

if __name__ == "__main__":
    chat.launch()
