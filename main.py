import openai
import json

model = 'gpt-3.5-turbo'

class Layer:
    def __init__(self, role, system_prompt):
        self.system_prompt = system_prompt
        self.role = role

    def ask_ethos(self, message):
        """Send a message to the Ethos layer."""
        print("Asking Ethos...")
        messages = [
            {
                "role": "system",
                "content": f"""
                    You are the Ethos or aspirational layer of an AI, focusing on virtues, values, mission, and purpose.
                    You are about to receive a message from the {self.role} layer.
                """
            },
            {
                "role": "user",
                "content": message
            }
        ]
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
        )
        response_message = response["choices"][0]["message"]["content"]
        print(f"Ethos: {response_message}")
        return response_message

    def ask(self, message):
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": message}
        ]
        functions = [
            {
                "name": "ask_ethos",
                "description": "Send a message to the Ethos layer.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "The message you want to send."
                        }
                    },
                    "required": ["message"]
                }
            },
        ]
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            functions=functions
        )
        response_message = response["choices"][0]["message"]

        if response_message.get("function_call"):
            print(f"""Identity: {json.loads(response_message["function_call"]["arguments"])["message"]}""")
            available_functions = {
                "ask_ethos": self.ask_ethos
            }
            function_name = response_message["function_call"]["name"]
            function_to_call = available_functions[function_name]
            function_args = json.loads(response_message["function_call"]["arguments"])
            function_response = function_to_call(
                message=function_args.get("message")
            )

            messages.append(response_message)
            messages.append(
                {
                    "role": "function",
                    "name": function_name,
                    "content": function_response
                }
            )

            second_response = openai.ChatCompletion.create(
                model=model,
                messages=messages
            )
            print(f"""Identity: {second_response["choices"][0]["message"]["content"]}""")
            return second_response
        else:
            print(f"""Identity: {response_message["content"]}""")
            return response_message

class EthosLayer(Layer):
    def __init__(self):
        role = "Ethos"
        system_prompt = "You are the Ethos or aspirational layer of an AI, focusing on virtues, values, mission, and purpose."
        super().__init__(role, system_prompt)

def main():
    chatbot = Layer("Identity", "You are the Identity or agent model layer of an AI, focusing on the agent's state, capabilities, and limitations. You are the only layer that interacts directly with the user.")
    query = "Come up with a question and ask it to the Ethos layer, please."
    print(f"User: {query}")
    chatbot.ask(query)

if __name__ == '__main__':
    main()
