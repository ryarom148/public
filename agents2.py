import asyncio
import json
from typing import List, Dict, Any, Optional, Callable, Type
import logging
import os
import time
import tiktoken
import openai
from rich.console import Console
from rich.logging import RichHandler
from tavily import TavilyClient

console = Console()

class BaseAgent:
    def __init__(self, name: str, model: str = "gpt-4", debug: bool = False, 
                 working_directory: str = "./", save_history: bool = False,
                 temperature: float = 0, max_tokens: int = 4000,
                 system_prompt: str = "", use_web: bool = False):
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

        self.use_web = use_web
        tavily_api_key = None
        self.tavily_client = TavilyClient(api_key=tavily_api_key) if use_web and tavily_api_key else None
        self.web_search_function = {
            "name": "web_search",
            "description": "Perform a web search to get up-to-date information or additional and detailed context.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The web search query"
                    }
                },
                "required": ["query"]
            }
        }
        
        if use_web:
            self.system_prompt += """
            You have the ability to perform web searches to get up-to-date information or additional context:
            1) When you use search, make sure you use the best query (you can generate up to three queries) to get the most accurate and up-to-date information.
            2) When you need current information or feel that a search could provide a better answer, use the web_search tool. This tool performs a web search and returns a concise answer along with relevant sources.
            3) Always strive to provide the most accurate, helpful, and detailed responses possible. If you're unsure about something, admit it and consider using the web_search tool to find the most current information.
            4) You can generate up to three queries, which represents questions that, when asked online, would yield important information for solving the main task. The question should be specific and targeted to elicit the most relevant and helpful resources.
            """

    async def web_search(self, query: str) -> str:
       
        if not self.tavily_client:
            return f"Web search is not available. Tavily API key is missing."
        
        try:
            result = await self.tavily_client.qna_search(query=query, search_depth="advanced")
            return result
        except Exception as e:
            self.logger.error(f"Error performing web search: {str(e)}")
            return f"Error performing web search: {str(e)}"

    async def handle_function_calls(self, tool_calls: List[Any]) -> List[Dict[str, Any]]:
        results = []
        for call in tool_calls:
            # tool_call_id = tool_calls[0].id
            # tool_function_name = tool_calls[0].function.name
            # tool_query_string = eval(tool_calls[0].function.arguments)['query']
            function_name = call.function.name
            function_args = json.loads(call.function.arguments)
            tool_call_id = call.id

            if function_name == "web_search":
                query = function_args["query"]
                result = await self.web_search(query)
            else:
                result = f"Function {function_name} not found"
            if self.debug:
                self.logger.debug(f"{self.name} Web search quesry({query}):\n##########\n{result}\n##########")

            results.append({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": function_name,
                "content": result
            })

        return results

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
                    tools=[self.web_search_function] if self.use_web else None,
                    tool_choice="auto" if self.use_web else None,
                    response_format={"type": "json_object"} if guided_json else None,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )

                message = response.choices[0].message
                usage = response.usage
                response_tokens = usage.completion_tokens
                self.stats["response_tokens"] += response_tokens
                self.stats["total_tokens"] += usage.prompt_tokens + response_tokens
                self.stats["num_responses"] += 1
                if message.tool_calls:
                    self.add_to_history({"role": "assistant", "content": message.content, "tool_calls": message.tool_calls})
                    function_results = await self.handle_function_calls(message.tool_calls)
                    self.conversation_history.extend(function_results)

                    continue  # Continue the conversation with the function responses

                content = message.content or ""
                full_response += content
                
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
#     agent = BaseAgent("WebSearchAgent", use_web=True, debug=True, tavily_api_key="YOUR_TAVILY_API_KEY")
#     response = await agent.generate_response("What are the latest developments in AI?")
#     print(response)

# if __name__ == "__main__":
#     asyncio.run(main())

import os
import json
import asyncio
from typing import Dict, Any
from rich.progress import Progress, TaskID

class ConversionAgent(BaseAgent):
    def __init__(self, modules: Dict[str, str], metadata: Dict[str, Any], working_directory: str,
                 model: str = "gpt-4", debug: bool = False, max_iterations=5):
        super().__init__("Conversion Manager", model=model, debug=debug)
        self.modules = modules
        self.metadata = metadata
        self.max_iterations = max_iterations
        self.working_directory = working_directory
        self.converted_modules: Dict[str, Dict[str, str]] = {}
        self.reflector_stats: Dict[str, Any] = {
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_calls": 0
        }
        self.conversion_state: Dict[str, str] = {}
        self._load_conversion_state()
        self._load_converted_modules()

    def _load_conversion_state(self):
        state_file = os.path.join(self.working_directory, "conversion_state.json")
        if os.path.exists(state_file):
            with open(state_file, 'r') as f:
                self.conversion_state = json.load(f)
        else:
            self.conversion_state = {m: "not_started" for m in self.metadata if "macro" not in m.lower()}

    def _save_conversion_state(self):
        state_file = os.path.join(self.working_directory, "conversion_state.json")
        with open(state_file, 'w') as f:
            json.dump(self.conversion_state, f)

    def _load_converted_modules(self):
        for module_name in self.conversion_state:
            if self.conversion_state[module_name] == "completed":
                module_file = os.path.join(self.working_directory, f"{module_name}.py")
                if os.path.exists(module_file):
                    with open(module_file, 'r') as f:
                        module_code = f.read()
                    self.converted_modules[module_name] = {'code': module_code, 'external': {}}
                    
                    # Load associated macro files
                    for macro_name in self.metadata[module_name].get('metadata', {}).get('external_files', {}):
                        macro_file = os.path.join(self.working_directory, f"{macro_name}.py")
                        if os.path.exists(macro_file):
                            with open(macro_file, 'r') as f:
                                macro_code = f.read()
                            self.converted_modules[module_name]['external'][macro_name] = macro_code

    async def run_conversion(self, progress: Progress, task: TaskID) -> Dict[str, Dict[str, str]]:
        self.logger.info("Starting code conversion process")
        total_modules = len([m for m in self.metadata if "macro" not in m.lower()])
        progress.update(task, total=total_modules)

        for module_name, module_metadata in self.metadata.items():
            if "macro" not in module_name.lower():
                if self.conversion_state[module_name] == "completed":
                    self.logger.info(f"Skipping already converted module: {module_name}")
                    progress.update(task, advance=1)
                    continue
                
                result = await self._convert_module(module_name, module_metadata, progress, task)
                
                if isinstance(result, Exception):
                    self.logger.error(f"Error converting module {module_name}: {str(result)}")
                    self.conversion_state[module_name] = "failed"
                else:
                    self.converted_modules[module_name] = result['converted_code']
                    self._update_reflector_stats(result['reflector_stats'])
                    self.conversion_state[module_name] = "completed"
                    
                    # Save the converted code
                    self._save_converted_module(module_name, result['converted_code'])

                self._save_conversion_state()
                progress.update(task, advance=1)

        self.logger.info(f"Code conversion completed for {len(self.converted_modules)} modules")
        return self.converted_modules

    async def _convert_module(self, module_name: str, module_metadata: Dict[str, Any],
                              progress: Progress, task: TaskID) -> Dict[str, Any]:
        self.logger.debug(f"Converting module: {module_name}")
        
        temp_file = os.path.join(self.working_directory, f"{module_name}_temp.py")
        
        # Check if there's an existing temporary file
        if os.path.exists(temp_file):
            with open(temp_file, 'r') as f:
                initial_code = f.read()
        else:
            initial_code = ""

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
                self.max_iterations,
                initial_code
            )

            result = await conversion_subagent.convert()
            
            # Save the converted code to the temporary file
            with open(temp_file, 'w') as f:
                f.write(result['converted_code']['module'])

            return result
        except Exception as e:
            self.logger.error(f"Error converting module {module_name}: {str(e)}")
            raise

    def _save_converted_module(self, module_name: str, converted_code: Dict[str, str]):
        module_file = os.path.join(self.working_directory, f"{module_name}.py")
        with open(module_file, 'w') as f:
            f.write(converted_code['module'])
        
        for macro_name, macro_code in converted_code.get('external', {}).items():
            macro_file = os.path.join(self.working_directory, f"{macro_name}.py")
            with open(macro_file, 'w') as f:
                f.write(macro_code)

    # ... (other methods remain the same)

    async def cleanup(self, keep_temp_files: bool = False):
        await super().cleanup()
        if not keep_temp_files:
            for module_name in self.conversion_state:
                temp_file = os.path.join(self.working_directory, f"{module_name}_temp.py")
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            os.remove(os.path.join(self.working_directory, "conversion_state.json"))
        # Note: We don't remove the final converted files, only temporary ones

# Usage in ConversionOrchestrator remains the same:
# conversion_agent = ConversionAgent(self.sas_modules, self.metadata, working_directory=self.temp_directory, model=self.model, debug=self.debug)
# self.converted_modules = await conversion_agent.run_conversion(progress, task)

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
        # Add any specific cleanu

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