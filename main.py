import openai
import json
from typing import Type, Union

class OpenAICommunicator:
    def __init__(self, model: str = 'gpt-3.5-turbo'):
        self.model = model

    def communicate(self, layer, messages: list) -> str:
        functions = [
            {
                "name": "ask_layer",
                "description": "Send a message to the specified layer.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target_layer_class": {
                            "type": "string",
                            "description": "The layer you want to talk to.",
                            "enum": ["EthosLayer", "StrategyLayer", "IdentityLayer", "PlanningLayer", "ControlLayer", "ExecutionLayer"]
                        },
                        "message": {
                            "type": "string",
                            "description": "The message you want to send."
                        }
                    },
                    "required": ["target_layer_class", "message"]
                }
            },
        ]
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=messages,
            functions=functions
        )
        response_message = response["choices"][0]["message"]

        if response_message.get("function_call"):
            print(f"""{layer.role}: {json.loads(response_message["function_call"]["arguments"])["message"]}""")
            available_functions = {
                "ask_layer": layer.ask_layer
            }
            function_name = response_message["function_call"]["name"]
            function_to_call = available_functions[function_name]
            function_args = json.loads(response_message["function_call"]["arguments"])
            function_response = function_to_call(
                target_layer_class=function_args.get("target_layer_class"),
                message=function_args.get("message")
            )

            messages.append(response_message)
            messages.append(
                {
                    "role": "function",
                    "name": function_name,
                    "content": function_response["content"]
                }
            )

            second_response = openai.ChatCompletion.create(
                model=self.model,
                messages=messages
            )
            second_response_message = second_response["choices"][0]["message"]["content"]
            print(f"""{layer.role}: {second_response_message}""")
            return second_response
        else:
            print(f"""{layer.role}: {response_message["content"]}""")
            return response_message

class Layer:
    openai_communicator = OpenAICommunicator()
    LayerType = Union['EthosLayer', 'StrategyLayer', 'IdentityLayer', 'PlanningLayer', 'ControlLayer', 'ExecutionLayer']

    def __init__(self, role: str, system_prompt: str):
        self.system_prompt = system_prompt
        self.role = role

    def ask_layer(self, target_layer_class: Type[LayerType], message: str) -> str:
        architecture = CognitiveArchitecture()
        layer_mapping = architecture.get_layer_mapping()
        target_layer = layer_mapping.get(target_layer_class)

        if target_layer_class not in layer_mapping:
            print(target_layer_class)
            print(layer_mapping)
            raise ValueError('Invalid target_layer_class provided.')

        messages = [
            {
                'role': 'system',
                'content': f'''You are the {target_layer.role} layer.
                    You are about to receive a message from the {self.role} layer.'''
            },
            {
                'role': 'user',
                'content': message
            }
        ]

        return self.communicate_with_openai(target_layer, messages)

    def communicate_with_openai(self, layer, messages: list) -> str:
        return self.openai_communicator.communicate(layer, messages)

class EthosLayer(Layer):
    def __init__(self):
        self.role = "Ethos"
        self.system_prompt = "You are the Ethos layer, one of six thought layers of a cognitive architecture. You are in charge of determining the core principles and values guiding the agent's behavior."
        super().__init__(self.role, self.system_prompt)

class StrategyLayer(Layer):
    def __init__(self):
        self.role = "Strategy"
        self.system_prompt = "You are the Strategy layer, one of six thought layers of a cognitive architecture. You focus on long-term goals and the overarching approach to achieve them."
        super().__init__(self.role, self.system_prompt)

class IdentityLayer(Layer):
    def __init__(self):
        self.role = "Identity"
        self.system_prompt = "You are the Identity layer, one of six thought layers of a cognitive architecture. You shape the agent's self-conception, including its role, capabilities, and inherent characteristics."
        super().__init__(self.role, self.system_prompt)

class PlanningLayer(Layer):
    def __init__(self):
        self.role = "Planning"
        self.system_prompt = "You are the Planning layer, one of six thought layers of a cognitive architecture. You craft detailed plans, setting milestones and determining the path forward."
        super().__init__(self.role, self.system_prompt)

class ControlLayer(Layer):
    def __init__(self):
        self.role = "Control"
        self.system_prompt = "You are the Control layer, one of six thought layers of a cognitive architecture. You oversee the execution of plans, making real-time adjustments as needed."
        super().__init__(self.role, self.system_prompt)

class ExecutionLayer(Layer):
    def __init__(self):
        self.role = "Execution"
        self.system_prompt = "You are the Execution layer, one of six thought layers of a cognitive architecture. You carry out the tasks, actions, and commands derived from the higher layers."
        super().__init__(self.role, self.system_prompt)

class CognitiveArchitecture:
    def __init__(self):
        self.ethos_layer = EthosLayer()
        self.strategy_layer = StrategyLayer()
        self.identity_layer = IdentityLayer()
        self.planning_layer = PlanningLayer()
        self.control_layer = ControlLayer()
        self.execution_layer = ExecutionLayer()

    def get_layer_mapping(self):
        return {
            "EthosLayer": self.ethos_layer,
            "StrategyLayer": self.strategy_layer,
            "IdentityLayer": self.identity_layer,
            "PlanningLayer": self.planning_layer,
            "ControlLayer": self.control_layer,
            "ExecutionLayer": self.execution_layer
        }

    def chat(self, message):
        chatbot = self.identity_layer
        messages = [
            {
                'role': 'system',
                'content': f'''{self.identity_layer.system_prompt}'''
            },
            {
                'role': 'user',
                'content': message
            }
        ]
        chatbot.communicate_with_openai(chatbot, messages)

def main():
    user_input = input("User: ")
    CognitiveArchitecture().chat(user_input)

if __name__ == '__main__':
    main()
