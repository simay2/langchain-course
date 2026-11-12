import re
import inspect
from dotenv import load_dotenv
from langchain.tools import tool
import ollama

from langsmith import traceable

load_dotenv() 

MAX_ITERATIONS = 10
MODEL = "qwen3.5:0.8b"

@traceable(run_type="tool")
def get_product_price(product: str) -> float:
    """
    Look up the price of a product in the catalog.

    Args:
        product: Exact product name. Must be one of:
            - "laptop"
            - "headphones"
            - "keyboard"

    Do not include discount information in the product name. For example, for "laptop with gold discount", pass only "laptop" to this tool. The discount will be applied in a separate step using the apply_discount tool.          
    """
    # Simulate fetching product price from an external source
    print(f"   >>Executing get_product_price '{product}')")
    prices = {
        "laptop": 1299.99,
        "headphones": 149.95,
        "keyboard": 89.50,}
    return prices.get(product, 0)

@traceable(run_type="tool")
def apply_discount(price: float, discount_tier: str) -> float:
    """
    Apply a discount tier to a price and return the final price.
    Available discount tiers: bronze, silver, gold
    """
    print(f"   >>Executing apply_discount '{price}', '{discount_tier}'")
    discount_percentages = {
        "bronze": 5,
        "silver": 12,
        "gold": 23,
    }

    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)

tools = {
    "get_product_price": get_product_price,
    "apply_discount": apply_discount
}

def get_tool_description(tool_dict):
    descriptions = {}
    for tool_name, tool_func in tool_dict.items():
        original_function = getattr(tool_func, "__wrapped__", tool_func)
        signature = inspect.signature(original_function)
        docstring = inspect.getdoc(tool_func) or ""
        descriptions.append (f"{tool_name}{signature} - {docstring}")
    return "\n".join(descriptions)    


tools_description = get_tool_description(tools)
tool_names = ", ".join(tools.keys())

react_prompt = f"""
STRICT RULES — you must follow these exactly:
1. NEVER guess or assume any product price. You MUST call get_product_price first to get the real price.
2. Only call apply_discount AFTER you have received a price from get_product_price. Pass the exact price returned by get_product_price — do NOT pass a made-up number.
3. NEVER calculate discounts yourself using math. Always use the apply_discount tool.
4. If the user does not specify a discount tier, ask them which tier to use — do NOT assume one.

Answer the following questions as best you can. You have access to the following tools:

{tool_descriptions}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action, as comma separated values
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {{question}}
Thought:"""


@traceable(name="Ollama Chat", run_type="llm")
def ollama_chat_traced(model, messages, options):
    return ollama.chat(model=model, messages=messages, options=options)






@traceable(name="LangChain Agent Loop Example")
def run_agent(question: str):
    tools = [get_product_price, apply_discount]
    tools_dict = {
        "get_product_price" : get_product_price,
        "apply_discount" : apply_discount
    }

     
    print(f"Question: {question}")
    print("=" * 60)

    messages = [
        {"role": "system",
         "content": (
                "You are a helpful shopping assistent."
                "You have access to a product catalog tool"
                "and a discount tool. \n\n"
                "STRICT RULES - you must follow these exactly:\n"
                "1. NEVER guess or assume any product price."
                "You MUST call get_product_price first to get the real price. \n"
                "2. Only call apply_discount AFTER you have received."
                "a price from get_product_price. Pass the exact price"
                "returned by get_product_price - do NOT pass a made-up number .\n"
                "3. NEVER calculate discounts yourself using math."
                "Always use the apply_discount tool. \n"
                "4. If the user does not specify a discount tier, "
                "ask them which tier to use- do NOT assume one."
        )
        },
        {"role": "user", "content": question},
    ]

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n===Iteration {iteration}===")

        response = ollama_chat_traced(messages)
        ai_message = response.message 

        tool_calls = ai_message.tool_calls

        if not tool_calls:
            print(f"\nFinal Answer: {ai_message.content}")
            return ai_message.content

        tool_call = tool_calls[0]
        tool_name = tool.call.function.name
        tool_args = tool_call.function.arguments
        
        print(f"[Tool Selected] {tool_name} with args {tool_args}")

        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool '{tool_name}' not found.")

        observation = tool_to_use(**tool_args)

        print(f" [Tool Result] {observation}")
        messages.append(ai_message)
        messages.append(
            {
           "role": "tool",
           "content": str(observation),
           }
        )

    print("ERROR: Max iterations reached a final answer")
    return None

if __name__ == "__main__":
    print("Welcome to the LangChain Agent Loop Example!")
    print()
    result = run_agent("What is the price of a laptop with a gold discount?")