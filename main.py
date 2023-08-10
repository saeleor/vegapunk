import openai
import json
from typing import Union

model = 'gpt-3.5-turbo'

ethos_layer = None
strategy_layer = None
identity_layer = None
planning_layer = None
control_layer = None
execution_layer = None

def get_layer_mapping():
    global ethos_layer, strategy_layer, identity_layer, planning_layer, control_layer, execution_layer

    ethos_layer = EthosLayer()
    strategy_layer = StrategyLayer()
    identity_layer = IdentityLayer()
    planning_layer = PlanningLayer()
    control_layer = ControlLayer()
    execution_layer = ExecutionLayer()

    return {
        "EthosLayer": ethos_layer,
        "StrategyLayer": strategy_layer,
        "IdentityLayer": identity_layer,
        "PlanningLayer": planning_layer,
        "ControlLayer": control_layer,
        "ExecutionLayer": execution_layer
    }

class Layer:
    layer_union_type = Union['StrategyLayer', 'IdentityLayer', 'PlanningLayer',
                             'ControlLayer', 'ExecutionLayer', 'EthosLayer']

    def __init__(self, role, system_prompt):
        self.system_prompt = system_prompt
        self.role = role

    def ask_layer(self, target_layer: layer_union_type, message):
        """Send a message to the specified layer."""
        layer_mapping = get_layer_mapping()

        if isinstance(target_layer, str):
            target_layer = layer_mapping.get(target_layer)

        if not isinstance(target_layer, (EthosLayer, StrategyLayer, IdentityLayer, PlanningLayer, ControlLayer, ExecutionLayer)):
            raise ValueError("target_layer must be an instance of one of the Layer subclasses.")

        messages = [
            {
                "role": "system",
                "content": f"""
                    You are the {target_layer.role} layer.
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

        print(f"{target_layer.role} (to {self.role}): {response_message}")
        return response_message

    def chat(self, message):
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": message}
        ]
        functions = [
            {
                "name": "ask_layer",
                "description": "Send a message to the specified layer.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target_layer": {
                            "type": "string",
                            "description": "The layer you want to talk to.",
                            "enum": ["EthosLayer", "StrategyLayer", "IdentityLayer", "PlanningLayer", "ControlLayer", "ExecutionLayer"]
                        },
                        "message": {
                            "type": "string",
                            "description": "The message you want to send."
                        }
                    },
                    "required": ["target_layer", "message"]
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
            print(f"""{self.role}: {json.loads(response_message["function_call"]["arguments"])["message"]}""")
            available_functions = {
                "ask_layer": self.ask_layer
            }
            function_name = response_message["function_call"]["name"]
            function_to_call = available_functions[function_name]
            function_args = json.loads(response_message["function_call"]["arguments"])
            function_response = function_to_call(
                target_layer=function_args.get("target_layer"),
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
            second_response_message = second_response["choices"][0]["message"]["content"]
            print(f"""{self.role}: {second_response_message}""")
            return second_response
        else:
            print(f"""{self.role}: {response_message["content"]}""")
            return response_message

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

def main():
    get_layer_mapping()
    user_input = input("User: ")
    identity_layer.chat(user_input)

if __name__ == '__main__':
    main()
