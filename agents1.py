import asyncio
import time
import os
import json
from typing import Dict, List, Any, Optional, Callable, Type
import logging
from rich.console import Console
from rich.logging import RichHandler
import gc
import tiktoken
from itertools import zip_longest
from rich.progress import Progress, TaskID
from rich.logging import RichHandler
from rich.table import Table
from code_interpriter_03 import RobustCodeInterpreter

import openai

console = Console()

class BaseAgent:
    def __init__(self, name: str, model: str = "gpt-4", debug: bool = False, 
                 working_directory: str = "./", save_history: bool = False,
                 temperature: float = 0, max_tokens: int = 4000,
                 system_prompt: str = ""):
        self.name = name
        self.model = model
        self.debug = debug
        self.logger = self._setup_logger()
        self.tools: Dict[str, Dict[str, Any]] = {}
        self.conversation_history: List[Dict[str, str]] = []
        self.subagents: List[BaseAgent] = []
        self._temp_files: List[str] = []
        self.working_directory = working_directory
        self.save_history = save_history if debug else False
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.stats = {
            "total_tokens": 0,
            "prompt_tokens": 0,
            "response_tokens": 0,
            "total_time": 0,
            "num_responses": 0
        }
        self.tokenizer = tiktoken.encoding_for_model(model)

        # We'll add the system prompt with tools later, after tools are registered
        self.system_prompt = system_prompt

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger(self.name)
        logger.setLevel(logging.DEBUG if self.debug else logging.INFO)
        rich_handler = RichHandler(rich_tracebacks=True)
        rich_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(rich_handler)
        return logger

    def register_tool(self, tool: Callable) -> None:
        self.tools[tool.__name__] = {"function": tool, "description": tool.__doc__}

    def _add_system_prompt_with_tools(self) -> None:
        if self.tools:
            tools_prompt = "\n".join([f"{name}: {spec['description']}" for name, spec in self.tools.items()])
            full_system_prompt = f"{self.system_prompt}\n\nAvailable tools:\n{tools_prompt}\n\nDecide if you need to use any tools to complete the task."
        else:
            full_system_prompt = self.system_prompt

        if full_system_prompt:
            self.add_to_history({"role": "system", "content": full_system_prompt})

    async def use_tool(self, tool_name: str, *args: Any, **kwargs: Any) -> Any:
        if tool_name in self.tools:
            try:
                start_time = time.time()
                result = await self.tools[tool_name]["function"](*args, **kwargs)
                elapsed_time = time.time() - start_time
                self.stats["total_time"] += elapsed_time
                self.logger.debug(f"Tool {tool_name} executed successfully in {elapsed_time:.2f} seconds")
                return result
            except Exception as e:
                self.logger.error(f"Error executing tool {tool_name}: {str(e)}")
                raise
        else:
            self.logger.error(f"Tool {tool_name} not found")
            raise ValueError(f"Tool {tool_name} not found")

    def add_to_history(self, message: Dict[str, str]) -> None:
        self.conversation_history.append(message)
        if self.save_history:
            self._append_to_history_file(message)

    def _append_to_history_file(self, message: Dict[str, str]) -> None:
        history_file = os.path.join(self.working_directory, f"{self.name}_history.jsonl")
        with open(history_file, 'a') as f:
            json.dump(message, f)
            f.write('\n')
        if history_file not in self._temp_files:
            self._temp_files.append(history_file)
            self.logger.debug(f"Registered history file: {history_file}")

    def get_history(self) -> List[Dict[str, str]]:
        return self.conversation_history

    def get_last_message(self) -> Optional[Dict[str, str]]:
        return self.conversation_history[-1] if self.conversation_history else None

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    async def generate_response(self, prompt: str, guided_json: bool = False, task_description: str = "Generating") -> str:
        self.add_to_history({"role": "user", "content": prompt})
        
        try:
            start_time = time.time()
            prompt_tokens = self.count_tokens(prompt)
            self.stats["prompt_tokens"] += prompt_tokens
            
            full_response = ""
            while True:
                messages = self.conversation_history.copy()
                if full_response:
                    messages.append({"role": "assistant", "content": full_response})
                    messages.append({"role": "user", "content": "continue"})

                response = await openai.ChatCompletion.acreate(
                    model=self.model,
                    messages=messages,
                    response_format={"type": "json_object"} if guided_json else None,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                #usage = response.usage
                #console.print(f"Input Tokens: {usage.prompt_tokens}, Output Tokens: {usage.completion_tokens}, Total Tokens: {usage.total_tokens}")

                content = response.choices[0].message['content']
                full_response += content
                usage = response.usage
                response_tokens = usage.completion_tokens # self.count_tokens(full_response)
                self.stats["response_tokens"] += response_tokens
                self.stats["total_tokens"] += usage.prompt_tokens + response_tokens
                self.stats["num_responses"] += 1
                if response_tokens < self.max_tokens:
                    break

            elapsed_time = time.time() - start_time
            self.stats["total_time"] += elapsed_time
            
            self.add_to_history({"role": "assistant", "content": full_response})
            
            
            
            self.logger.info(f"{self.name} ({task_description}): {prompt_tokens + response_tokens} tokens, {elapsed_time:.2f} seconds")
            
            if self.debug:
                self.logger.debug(f"{self.name} Response ({task_description}):\n{full_response}")
            
            return full_response
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            raise

    async def create_subagent(self, agent_class: Type[BaseAgent], *args: Any, **kwargs: Any) -> BaseAgent:
        try:
            subagent = agent_class(*args, **kwargs)
            self.subagents.append(subagent)
            self.logger.debug(f"Created subagent of type {agent_class.__name__}")
            return subagent
        except Exception as e:
            self.logger.error(f"Error creating subagent of type {agent_class.__name__}: {str(e)}")
            raise

    def register_temp_file(self, file_path: str) -> None:
        """Register a temporary file for cleanup."""
        full_path = os.path.join(self.working_directory, file_path)
        if os.path.exists(full_path):
            self._temp_files.append(full_path)
            self.logger.debug(f"Registered temporary file: {full_path}")
        else:
            self.logger.warning(f"Attempted to register non-existent file: {full_path}")

    async def cleanup(self) -> None:
        """
        Thorough cleanup of resources.
        """
        self.logger.info(f"Starting cleanup for {self.name}")

        # Clean up conversation history
        self.conversation_history.clear()
        self.logger.debug("Cleared conversation history")

        # Clean up tools
        self.tools.clear()
        self.logger.debug("Cleared tools")

        # Clean up temporary files
        # for file_path in self._temp_files:
        #     try:
        #         os.remove(file_path)
        #         self.logger.debug(f"Removed temporary file: {file_path}")
        #     except Exception as e:
        #         self.logger.error(f"Error removing temporary file {file_path}: {str(e)}")
        # self._temp_files.clear()

        # Clean up subagents
        for subagent in self.subagents:
            await subagent.cleanup()
        self.subagents.clear()
        self.logger.debug("Cleaned up and cleared subagents")

        # Explicitly call garbage collection
        gc.collect()
        self.logger.debug("Explicitly called garbage collection")

        self.logger.info(f"Cleanup completed for {self.name}")

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the agent's operations."""
        stats = self.stats.copy()
        stats["name"] = self.name
        stats["subagents"] = [subagent.get_stats() for subagent in self.subagents]
        
        # Aggregate stats from subagents
        for subagent_stats in stats["subagents"]:
            for key in ["total_tokens", "prompt_tokens", "response_tokens", "total_time", "num_responses"]:
                stats[key] += subagent_stats[key]
        
        return stats

    async def __aenter__(self):
        self._add_system_prompt_with_tools()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup()

# Usage example:
# async def main():
#     async with BaseAgent("TestAgent", debug=True, working_directory="./temp", 
#                          save_history=True, temperature=0.7, max_tokens=2000,
#                          system_prompt="You are a helpful AI assistant.") as agent:
#         
#         def example_tool():
#             """This is an example tool."""
#             pass
#         
#         agent.register_tool(example_tool)
#         
#         response = await agent.generate_response("Tell me a story about a brave knight.")
#         print(response)
#         stats = agent.get_stats()
#         print(f"Agent stats: {stats}")

# if __name__ == "__main__":
#     asyncio.run(main())




class VariableIdentificationAgent(BaseAgent):
    def __init__(self, parser: SASParser, module_name: str, metadata: Dict[str, Any], model: str = "gpt-4", debug: bool = False):
        super().__init__(f"{module_name}_VariableIdentifier", model=model, debug=debug)
        self.parser = parser
        self.module_name = module_name
        self.metadata = metadata

    async def identify_variables(self) -> Dict[str, Any]:
        self.logger.info(f"Identifying variables for module: {self.module_name}")
        context = self._prepare_context()
        prompt = self._create_prompt(context)
        response = await self.generate_response(prompt, guided_json=True, task_description=f"Identifying variables for {self.module_name}")
        variables_info = self._parse_response(response)
        
        if self.debug:
            self._display_variable_comparison(variables_info)
        
        return self.parser.get_module_comprehensive_metadata(self.module_name, variables_info)

    def _prepare_context(self) -> str:
        context = f"########### Main Module ({self.module_name}): ############\n{self.parser.sas_modules[self.module_name]}\n\n"
        for ext_file, ext_metadata in self.metadata['metadata']['external_files'].items():
            if ext_file in self.parser.sas_modules:
                context += f"########### External File ({ext_file}):############\n{self.parser.sas_modules[ext_file]}\n\n"
        return context

    def _create_prompt(self, context: str) -> str:
        return f"""
        Analyze the following SAS code and identify all variables used in the main module and its referenced external files.
        Provide the output in JSON format as follows:
        {{
            "variables": [list of all variables in the main module],
            "external": {{
                "filename1": [list of variables in this external file],
                "filename2": [list of variables in this external file],
                ...
            }}
        }}

        Context:
        {context}

        Please provide your analysis in the specified JSON format.
        """

    def _parse_response(self, response: str) -> Dict[str, Any]:
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            self.logger.error(f"Failed to parse JSON response for module {self.module_name}")
            return {"variables": [], "external": {}}

    def _display_variable_comparison(self, variables_info: Dict[str, Any]) -> None:
        initial_vars = set(self.metadata['metadata']['global'] + self.metadata['metadata']['local'] + self.metadata['metadata']['used'])
        llm_vars = set(variables_info['variables'])
        new_vars = llm_vars - initial_vars
        missing_vars = initial_vars - llm_vars
        
        if new_vars or missing_vars:
            table = Table(title=f"Variable Comparison for {self.module_name}")
            table.add_column("New Variables", style="green")
            table.add_column("Missing Variables", style="red")
            for new, missing in zip_longest(sorted(new_vars), sorted(missing_vars), fillvalue=""):
                table.add_row(new, missing)
            console.print(table)

    async def cleanup(self) -> None:
        await super().cleanup()
        # Add any specific cleanup for VariableIdentificationAgent here

# Usage example:
# async def identify_variables_for_module(parser: SASParser, module_name: str, metadata: Dict[str, Any]):
#     agent = VariableIdentificationAgent(parser, module_name, metadata, debug=True)
#     async with agent:
#         variables_info = await agent.identify_variables()
#     return variables_info




class MetadataExtractionAgent(BaseAgent):
    def __init__(self, sas_modules: Dict[str, str], model: str = "gpt-4", debug: bool = False):
        super().__init__("MetadataExtractor", model=model, debug=debug)
        self.sas_modules = sas_modules
        self.parser = SASParser(sas_modules)
        self.metadata: Dict[str, Any] = {}
        self.last_run_time: Dict[str, float] = self._load_last_run_time()
        self.token_usage: Dict[str, int] = {
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0
        }
        
    def _load_last_run_time(self) -> Dict[str, float]:
        if os.path.exists("last_run_time.json"):
            with open("last_run_time.json", 'r') as f:
                return json.load(f)
        return {}

    def _save_last_run_time(self) -> None:
        with open("last_run_time.json", 'w') as f:
            json.dump(self.last_run_time, f)

    async def run_extraction(self) -> Dict[str, Any]:
        self.logger.info("Starting metadata extraction process")
        with Progress() as progress:
            task = progress.add_task("Extracting metadata...", total=None)
            self.metadata = await self.extract_metadata(progress, task)
        self.logger.info("Metadata extraction process completed")
        self.logger.info(f"Token usage: {self.token_usage}")
        return self.metadata

    async def extract_metadata(self, progress: Progress, task: TaskID) -> Dict[str, Any]:
        tasks = []
        total_modules = len([m for m in self.parser.meta_data if "macro" not in m.lower()])
        progress.update(task, total=total_modules)

        for module_name, metadata in self.parser.meta_data.items():
            if "macro" not in module_name.lower():
                task = self.process_module(module_name, metadata, progress, task)
                tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for module_name, result in zip([m for m in self.parser.meta_data if "macro" not in m.lower()], results):
            if isinstance(result, Exception):
                self.logger.error(f"Error processing module {module_name}: {str(result)}")
            else:
                self.metadata[module_name] = result['metadata']
        self._update_token_usage()

        self._save_last_run_time()
        self.logger.info(f"Metadata extraction completed for {len(self.metadata)} modules")
        return self.metadata

    async def process_module(self, module_name: str, metadata: Dict[str, Any], 
                             progress: Progress, task: TaskID) -> Dict[str, Any]:
        self.logger.debug(f"Processing module: {module_name}")
        
        current_time = time.time()
        
        try:
            variable_identifier = await self.create_subagent(
                VariableIdentificationAgent,
                self.parser,
                module_name,
                metadata,
                self.model,
                self.debug
            )
            extraction_result = await variable_identifier.identify_variables()
            
            validated_metadata = self._validate_metadata(extraction_result['metadata'])
            
            self.last_run_time[module_name] = current_time
            
            progress.update(task, advance=1)
            return {
                'metadata': validated_metadata,
                'token_usage': extraction_result['token_usage']
            }
        except Exception as e:
            self.logger.error(f"Error processing module {module_name}: {str(e)}")
            progress.update(task, advance=1)
            raise

    def _validate_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        required_keys = ['variables', 'external_files', 'proc_statements', 'data_steps']
        for key in required_keys:
            if key not in metadata:
                raise ValueError(f"Missing required key in metadata: {key}")
        return metadata

    def _update_token_usage(self, usage: Dict[str, int]) -> None:
        self.get_stats()
        for key in self.token_usage:
            self.token_usage[key] += usage.get(key, 0)

    async def cleanup(self) -> None:
        await super().cleanup()
        self._save_last_run_time()

    def get_extraction_stats(self) -> Dict[str, Any]:
        return {
            "total_modules": len(self.metadata),
            "processed_modules": len(self.last_run_time),
            "last_run_time": self.last_run_time,
            "token_usage": self.token_usage
        }

# Usage example:
# async def main():
#     sas_modules = {...}  # Your SAS modules dictionary
#     async with MetadataExtractionAgent(sas_modules, debug=True) as agent:
#         metadata = await agent.run_extraction()
#         print(agent.get_extraction_stats())

# if __name__ == "__main__":
#     asyncio.run(main())


class ReflectorAgent(BaseAgent):
    def __init__(self, module_name: str, model: str = "gpt-4", debug: bool = False):
        super().__init__(f"Reflector_{module_name}", model=model, debug=debug)
        self.module_name = module_name

    async def reflect(self, history: List[Dict[str, str]], original_module: str, 
                      original_macros: Dict[str, str], converted_code: Dict[str, str], 
                      metadata: Dict[str, Any]) -> Dict[str, Any]:
        self.logger.info(f"Reflecting on initial conversion for module: {self.module_name}")
        prompt = self._create_initial_reflection_prompt(history, original_module, original_macros, converted_code, metadata)
        response = await self.generate_response(prompt, task_description=f"Initial reflection on {self.module_name}")
        return self._parse_reflection_response(response)

    async def reflect_on_update(self, last_message: str) -> Dict[str, Any]:
        self.logger.info(f"Reflecting on update for module: {self.module_name}")
        prompt = self._create_update_reflection_prompt(last_message)
        response = await self.generate_response(prompt, task_description=f"Reflecting on update for {self.module_name}")
        return self._parse_reflection_response(response)

    def _create_initial_reflection_prompt(self, history: List[Dict[str, str]], original_module: str, 
                                          original_macros: Dict[str, str], converted_code: Dict[str, str], 
                                          metadata: Dict[str, Any]) -> str:
        prompt = f"""
        Analyze the following SAS to Python conversion process:

        Conversion History:
        {self._format_history(history)}

        Original SAS Module:
        {original_module}

        Original SAS Macros:
        {original_macros}

        Latest Converted Python Code:
        {converted_code}

        Metadata:
        {metadata}

        Your task is to:
        1. Determine if the conversion is satisfactory (pass) or needs improvement (fail).
        2. If improvement is needed, provide specific suggestions to improve the code, change something, or add functionality.
        3. Ensure all SAS functionality has been properly translated.
        4. Verify that all variables from the metadata are accounted for.
        5. Check that Python best practices and error handling are followed.
        6. Consider the full conversion history and requirements when making your assessment.

        Provide your reflection in the following format:
        PASS: [True/False]
        SUGGESTIONS:
        - [Suggestion 1]
        - [Suggestion 2]
        ...
        EXPLANATION: [Brief explanation of your decision and suggestions]
        """
        return prompt

    def _create_update_reflection_prompt(self, last_message: str) -> str:
        prompt = f"""
        Analyze the following update to the Python conversion:

        Latest Update:
        {last_message}

        Based on the previous recommendations, new code was generated. Your task is to:
        1. Determine if the conversion is now satisfactory (pass) or still needs improvement (fail).
        2. If improvement is still needed, provide specific suggestions to improve the code, change something, or add functionality.
        3. Ensure all previously identified issues have been addressed.
        4. Check that the new changes haven't introduced any new problems.

        Provide your reflection in the following format:
        PASS: [True/False]
        SUGGESTIONS:
        - [Suggestion 1]
        - [Suggestion 2]
        ...
        EXPLANATION: [Brief explanation of your decision and suggestions]
        """
        return prompt

    def _format_history(self, history: List[Dict[str, str]]) -> str:
        formatted_history = ""
        for entry in history:
            formatted_history += f"{entry['role'].capitalize()}: {entry['content']}\n\n"
        return formatted_history

    def _parse_reflection_response(self, response: str) -> Dict[str, Any]:
        lines = response.split('\n')
        result = {
            'pass': False,
            'suggestions': [],
            'explanation': ''
        }
        
        for line in lines:
            if line.startswith('PASS:'):
                result['pass'] = line.split(':')[1].strip().lower() == 'true'
            elif line.startswith('SUGGESTIONS:'):
                continue
            elif line.startswith('-'):
                result['suggestions'].append(line[1:].strip())
            elif line.startswith('EXPLANATION:'):
                result['explanation'] = line.split(':')[1].strip()

        return result

    async def cleanup(self) -> None:
        await super().cleanup()
        # Add any specific cleanup for ReflectorAgent here

# Usage of this class is typically through the CodeConversionSubAgent

from typing import Dict, Any, List
from rich.console import Console



console = Console()

class CodeConversionSubAgent(BaseAgent):
    def __init__(self, module_name: str, module_code: str, macro_codes: Dict[str, str], 
                 metadata: Dict[str, Any], model: str = "gpt-4", debug: bool = False, max_iterations:int =5):
        super().__init__(f"Conversion Sub-Agent: {module_name}", model=model, debug=debug)
        self.module_name = module_name
        self.module_code = module_code
        self.macro_codes = macro_codes
        self.metadata = metadata
        self.max_iterations = max_iterations 
        self.reflector: ReflectorAgent = None

    async def convert(self) -> Dict[str, Any]:
        self.logger.info(f"Starting conversion for module: {self.module_name}")
        converted_code = await self._initial_conversion()
        self.reflector = await self.create_subagent(
            ReflectorAgent,
            self.module_name,
            self.model,
            self.debug
        )

        iteration = 0
        max_iterations = 5  # Set a maximum number of iterations to prevent infinite loops
        while iteration < self.max_iterations:
            if iteration == 0:
                reflection = await self.reflector.reflect(self.get_history(), self.module_code, self.macro_codes, converted_code, self.metadata)
            else:
                reflection = await self.reflector.reflect_on_update(self.get_last_message()['content'])

            if self.debug:
                console.print(f"Iteration {iteration + 1}\nPass: {reflection['pass']}\nSuggestions: {reflection['suggestions']}")
            
            if reflection['pass']:
                break
            
            converted_code = await self._improve_code(converted_code, reflection['suggestions'])
            iteration += 1

        if iteration == max_iterations:
            self.logger.warning(f"Reached maximum iterations for module {self.module_name}")

        reflector_stats = self.reflector.get_stats()

        return {
            'converted_code': {
                'module': converted_code['module'],
                **{macro: converted_code[macro] for macro in self.macro_codes}
            },
            'reflector_stats': reflector_stats
        }

    async def _initial_conversion(self) -> Dict[str, str]:
        prompt = self._create_conversion_prompt()
        response = await self.generate_response(prompt, guided_json=True, task_description=f"Converting {self.module_name}")
        return self._parse_conversion_response(response)

    def _create_conversion_prompt(self) -> str:
        prompt = f"Convert the following SAS module and its macros to Python:\n\nMain Module ({self.module_name}):\n{self.module_code}\n\n"
        for macro_name, macro_code in self.macro_codes.items():
            prompt += f"Macro ({macro_name}):\n{macro_code}\n\n"
        prompt += f"Metadata:\n{self.metadata}\n\nProvide the converted Python code for the main module and each macro separately.\n\n"
        prompt += """
                    Provide the output in JSON format as follows:        
                    {
                            'code': python code,
                            'external': {"filename1.py": python code,
                                        "filename2.py": python code
                                        }
                    }
                  """
        return prompt

    def _parse_conversion_response(self, response: str) -> Dict[str, str]:
        try:
            parsed_response = json.loads(response)
            return {
                'module': parsed_response['code'],
                **parsed_response['external']
            }
        except json.JSONDecodeError:
            self.logger.error(f"Failed to parse JSON response for module {self.module_name}")
            return {'module': '', 'external': {}}

    async def _improve_code(self, code: Dict[str, str], suggestions: List[str]) -> Dict[str, str]:
        prompt = f"Improve the following Python code based on these suggestions:\n\nCode:\n{code}\n\nSuggestions:\n" + "\n".join(suggestions)
        response = await self.generate_response(prompt, guided_json=True, task_description=f"Improving {self.module_name}")
        return self._parse_conversion_response(response)

    async def cleanup(self) -> None:
        await super().cleanup()
        if self.reflector:
            await self.reflector.cleanup()

# Usage of this class is typically through the ConversionAgent

import asyncio
from typing import Dict, Any, List, Tuple
from rich.progress import Progress, TaskID



class ConversionAgent(BaseAgent):
    def __init__(self, modules: Dict[str, str], metadata: Dict[str, Any], model: str = "gpt-4", debug: bool = False,max_iterations=5):
        super().__init__("Conversion Manager", model=model, debug=debug)
        self.modules = modules
        self.metadata = metadata
        self.max_iterations=max_iterations
        self.converted_modules: Dict[str, Dict[str, str]] = {}
        self.reflector_stats: Dict[str, Any] = {
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_calls": 0
        }

    async def run_conversion(self, progress: Progress, task: TaskID) -> Dict[str, Dict[str, str]]:
        self.logger.info("Starting code conversion process")
        tasks = []
        total_modules = len([m for m in self.metadata if "macro" not in m.lower()])
        progress.update(task, total=total_modules)

        for module_name, module_metadata in self.metadata.items():
            if "macro" not in module_name.lower():
                task = self._convert_module(module_name, module_metadata, progress, task)
                tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for module_name, result in zip([m for m in self.metadata if "macro" not in m.lower()], results):
            if isinstance(result, Exception):
                self.logger.error(f"Error converting module {module_name}: {str(result)}")
            else:
                self.converted_modules[module_name] = result['converted_code']
                self._update_reflector_stats(result['reflector_stats'])

        self.logger.info(f"Code conversion completed for {len(self.converted_modules)} modules")
        return self.converted_modules

    async def _convert_module(self, module_name: str, module_metadata: Dict[str, Any], 
                              progress: Progress, task: TaskID) -> Dict[str, Any]:
        self.logger.debug(f"Converting module: {module_name}")
        
        try:
            module_code = self.modules[module_name]
            macro_codes = self._get_macro_codes(module_metadata)
            
            conversion_subagent = await self.create_subagent(
                CodeConversionSubAgent,
                module_name,
                module_code,
                macro_codes,
                module_metadata,
                self.model,
                self.debug,
                self.max_iterations
            )

            result = await conversion_subagent.convert()
            
            progress.update(task, advance=1)
            return result
        except Exception as e:
            self.logger.error(f"Error converting module {module_name}: {str(e)}")
            progress.update(task, advance=1)
            raise

    def _get_macro_codes(self, module_metadata: Dict[str, Any]) -> Dict[str, str]:
        macro_codes = {}
        for macro_name in module_metadata['metadata']['external_files']:
            if macro_name in self.modules:
                macro_codes[macro_name] = self.modules[macro_name]
        return macro_codes

    def _update_reflector_stats(self, stats: Dict[str, Any]) -> None:
        for key in self.reflector_stats:
            self.reflector_stats[key] += stats[key]

    async def cleanup(self) -> None:
        await super().cleanup()
        # Add any specific cleanup for ConversionAgent here

    def get_conversion_stats(self) -> Dict[str, Any]:
        base_stats = self.get_stats()
        return {
            **base_stats,
            "total_modules": len(self.metadata),
            "converted_modules": len(self.converted_modules),
            "conversion_rate": len(self.converted_modules) / len(self.metadata) if len(self.metadata) > 0 else 0,
            "reflector_stats": self.reflector_stats
        }

# Usage example:
# async def main():
#     modules = {...}  # Your SAS modules dictionary
#     metadata = {...}  # Your metadata dictionary
#     async with ConversionAgent(modules, metadata, debug=True) as agent:
#         with Progress() as progress:
#             task = progress.add_task("Converting code...", total=None)
#             converted_modules = await agent.run_conversion(progress, task)
#         print(agent.get_conversion_stats())

# if __name__ == "__main__":
#     asyncio.run(main())


class ExecutionAgent(BaseAgent):
    def __init__(self, working_folder: str, max_fix_iterations: int = 3, model: str = "gpt-4", debug: bool = False):
        super().__init__("ExecutionAgent", model=model, debug=debug)
        self.working_folder = working_folder
        self.max_fix_iterations = max_fix_iterations
        self.interpreter = RobustCodeInterpreter(working_folder)
        self.fixer_stats: Dict[str, Any] = {
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_calls": 0
        }

    async def execute(self, python_codes: Dict[str, Dict[str, Any]], sas_codes: Dict[str, str], metadata: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        self.logger.info("Starting execution process")
        results = {}
        
        for module_name, module_data in python_codes.items():
            self.interpreter.add_module(module_name, module_data['code'])
            for ext_name, ext_code in module_data.get('external', {}).items():
                self.interpreter.add_module(ext_name, ext_code)

            result = await self.execute_module(module_name, sas_codes, python_codes, metadata)
            results[module_name] = result

            if result["success"]:
                await self.save_output(module_name, module_data['code'])
                for ext_name, ext_code in module_data.get('external', {}).items():
                    await self.save_output(ext_name, ext_code)
            else:
                self.logger.warning(f"Execution failed for module: {module_name}")

            self.interpreter.reset()

        self.logger.info("Execution and fixing process completed")
        return results

    async def execute_module(self, module_name: str, sas_codes: Dict[str, str], python_codes: Dict[str, Dict[str, Any]], 
                             metadata: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        self.logger.debug(f"Executing module: {module_name}")
        module_metadata = metadata.get(module_name, {})
        module_data = python_codes[module_name]

        fixer = CodeFixerSubagent(self.model, self.debug)

        for attempt in range(self.max_fix_iterations):
            self.logger.debug(f"Execution attempt {attempt + 1} for module: {module_name}")

            result = self.interpreter.execute_module(module_name)

            if result["success"]:
                self.logger.info(f"Module {module_name} executed successfully")
                return result

            self.logger.warning(f"Execution failed. Attempting to fix. Error: {result['error']}")

            fix_result = await fixer.fix_code(
                module_name,
                sas_codes[module_name],
                module_data['code'],
                sas_codes,
                module_data.get('external', {}),
                module_metadata,
                result["error"],
                attempt
            )

            self._update_fixer_stats(fix_result['stats'])

            fixed_code = fix_result['fixed_code']
            if fixed_code['module'] == module_data['code']:
                self.logger.warning("Code fixer couldn't improve the code. Stopping attempts.")
                return result

            self.interpreter.reset()
            self.interpreter.add_module(module_name, fixed_code['module'])
            for ext_name, ext_code in fixed_code['external'].items():
                self.interpreter.add_module(ext_name, ext_code)

            module_data['code'] = fixed_code['module']
            module_data['external'] = fixed_code['external']

        self.logger.error(f"Failed to fix module {module_name} after {self.max_fix_iterations} attempts")
        return result

    async def save_output(self, module_name: str, code: str) -> None:
        output_path = os.path.join(self.working_folder, module_name)
        with open(output_path, "w") as f:
            f.write(code)
        self.logger.debug(f"Output for module {module_name} saved to {output_path}")

    def _update_fixer_stats(self, stats: Dict[str, Any]) -> None:
        for key in self.fixer_stats:
            self.fixer_stats[key] += stats[key]

    async def cleanup(self) -> None:
        await super().cleanup()
        # Add any specific cleanup for ExecutionAgent here

    def get_execution_stats(self) -> Dict[str, Any]:
        base_stats = self.get_stats()
        return {
            **base_stats,
            "fixer_stats": self.fixer_stats
        }
class CodeFixerSubagent(BaseAgent):
    def __init__(self, model: str = "gpt-4", debug: bool = False):
        super().__init__("CodeFixerSubagent", model=model, debug=debug)
        self.initial_prompt = None

    async def fix_code(self, module_name: str, sas_code: str, python_code: str, 
                       sas_codes: Dict[str, str], python_macros: Dict[str, str], 
                       metadata: Dict[str, Any], error: str, attempt: int) -> Dict[str, Any]:
        self.logger.debug(f"Attempting to fix code for module: {module_name}")

        if attempt == 0:
            self.initial_prompt = self._create_initial_prompt(module_name, sas_code, python_code, sas_codes, python_macros, metadata)
            prompt = self.initial_prompt + f"\n\nError message:\n{error}\n\nPlease provide the corrected Python code for the module and its external files."
        else:
            prompt = f"""
            Previous attempt to fix the code was unsuccessful. Here's the current state of the Python module:

            ```python
            {self._add_line_numbers(python_code)}
            ```

            Error message:
            {error}

            Please provide an updated version of the Python code that addresses this error.
            """

        response = await self.generate_response(prompt, task_description=f"Fixing code for {module_name}")
        
        fixed_code = self._parse_fix_response(response)

        if self.debug:
            console.print(Panel(json.dumps(fixed_code, indent=2), title="[bold green]Fixed Code[/bold green]", expand=False))

        return {
            'fixed_code': fixed_code,
            'stats': self.get_stats()
        }

    def _create_initial_prompt(self, module_name: str, sas_code: str, python_code: str, 
                               sas_codes: Dict[str, str], python_macros: Dict[str, str], 
                               metadata: Dict[str, Any]) -> str:
        prompt = f"""
        Fix the following Python code based on the original SAS code and the metadata:

        Module name: {module_name}

        Original SAS module:
        ```sas
        {sas_code}
        ```

        Python module to be fixed:
        ```python
        {self._add_line_numbers(python_code)}
        ```

        SAS Macros:
        {self._format_macros(sas_codes, 'sas')}

        Python Macros:
        {self._format_macros(python_macros, 'python')}

        Metadata:
        {metadata}

        Please provide the fixed Python code for the main module and its external files in the following JSON format:
        {{
            "module": "fixed Python code for the main module",
            "external": {{
                "filename1.py": "fixed Python code for external file 1",
                "filename2.py": "fixed Python code for external file 2",
                ...
            }}
        }}
        """
        return prompt

    def _add_line_numbers(self, code: str) -> str:
        lines = code.split('\n')
        return '\n'.join(f"{i+1}: {line}" for i, line in enumerate(lines))

    def _format_macros(self, macros: Dict[str, str], language: str) -> str:
        formatted = ""
        for macro_name, macro_code in macros.items():
            formatted += f"\n{macro_name}:\n```{language}\n{self._add_line_numbers(macro_code)}\n```\n"
        return formatted

    def _parse_fix_response(self, response: str) -> Dict[str, Any]:
        try:
            fixed_code = json.loads(response)
            if not isinstance(fixed_code, dict) or 'module' not in fixed_code or 'external' not in fixed_code:
                raise ValueError("Invalid response format")
            return fixed_code
        except json.JSONDecodeError:
            self.logger.error("Failed to parse JSON response")
            return {"module": "", "external": {}}
        except ValueError as e:
            self.logger.error(f"Invalid response format: {str(e)}")
            return {"module": "", "external": {}}
    
   
    async def cleanup(self) -> None:
        await super().cleanup()
        # Add any specific cleanup for CodeFixerSubagent here