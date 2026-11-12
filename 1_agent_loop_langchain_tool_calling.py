from dotenv import load_dotenv
from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langsmith import traceable

load_dotenv() 

MAX_ITERATIONS = 10
MODEL = "qwen3.5:0.8b"

@tool
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

@tool
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



@traceable(name="LangChain Agent Loop Example")
def run_agent(question: str):
    tools = [get_product_price, apply_discount]
    tools_dict = {t.name: t for t in tools}

    llm = init_chat_model(f"ollama:{MODEL}", temperature=0)
    llm_with_tools = llm.bind_tools(tools)    
    print(f"Question: {question}")
    print("=" * 60)

    messages = [
        SystemMessage(
             content = (
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
        ),
        HumanMessage(content=question)
    ]

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n===Iteration {iteration}===")
        ai_message = llm_with_tools.invoke(messages)
        tool_calls = ai_message.tool_calls

        if not tool_calls:
            print(f"\nFinal Answer: {ai_message.content}")
            return ai_message.content

        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")

        print(f"[Tool Selected] {tool_name} with args {tool_args}")

        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool '{tool_name}' not found.")

        observation = tool_to_use.invoke(tool_args)

        print(f" [Tool Result] {observation}")
        messages.append(ai_message)
        messages.append(
            ToolMessage(content=str(observation), tool_name=tool_name, tool_call_id=tool_call_id)
        )

    print("ERROR: Max iterations reached a final answer")
    return None

if __name__ == "__main__":
    print("Welcome to the LangChain Agent Loop Example!")
    print()
    result = run_agent("What is the price of a laptop with a gold discount?")