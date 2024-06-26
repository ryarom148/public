import asyncio
import os
import time
from typing import Dict, Any, List
import logging
from rich.console import Console
from rich.progress import Progress, TaskID
from rich.logging import RichHandler
from rich.table import Table

from state_manager import StateManager
from agents1 import MetadataExtractionAgent
from agents1 import ConversionAgent
from agents1 import ExecutionAgent

console = Console()

class ConversionOrchestrator:
    def __init__(self, input_directory: str, output_directory: str, state_file: str, 
                 model: str = "gpt-4", debug: bool = False, start_fresh: bool = False):
        self.input_directory = input_directory
        self.output_directory = output_directory
        self.model = model
        self.debug = debug
        self.logger = self._setup_logger()
        self.state_manager = StateManager(state_file)

        if start_fresh:
            self.state_manager = StateManager(state_file)  # This will create a fresh state

        self.sas_modules: Dict[str, str] = {}
        self.metadata: Dict[str, Any] = {}
        self.converted_modules: Dict[str, Dict[str, str]] = {}
        self.execution_results: Dict[str, Any] = {}
        
        self.stats: Dict[str, Any] = {
            "input_processing": {},
            "metadata_extraction": {},
            "code_conversion": {},
            "code_execution": {},
            "overall": {
                "total_tokens": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_calls": 0,
                "total_time": 0
            }
        }

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger("ConversionOrchestrator")
        logger.setLevel(logging.DEBUG if self.debug else logging.INFO)
        rich_handler = RichHandler(rich_tracebacks=True)
        rich_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(rich_handler)
        return logger

    async def run_conversion(self) -> None:
        try:
            with Progress() as progress:
                overall_task = progress.add_task("[cyan]Converting SAS to Python...", total=4)
                await self._run_conversion_steps(progress, overall_task)
            console.print("[bold green]Conversion completed successfully![/bold green]")
            self.display_overall_stats()
        except Exception as e:
            self.logger.exception(f"An error occurred during conversion: {str(e)}")
            console.print(f"[bold red]Conversion failed. Error: {str(e)}[/bold red]")

    async def _run_conversion_steps(self, progress: Progress, overall_task: TaskID) -> None:
        steps = [
            ("input_processed", self._process_input),
            ("metadata_extracted", self._extract_metadata),
            ("code_converted", self._convert_code),
            ("code_executed", self._execute_and_fix_code)
        ]

        for step_name, step_func in steps:
            if not self.state_manager.is_completed(step_name):
                step_task = progress.add_task(f"[green]{step_name.replace('_', ' ').title()}", total=None)
                start_time = time.time()
                await step_func(progress, step_task)
                end_time = time.time()
                self.stats[step_name]["time"] = end_time - start_time
                self.display_step_stats(step_name)
            else:
                self.logger.info(f"Skipping {step_name} (already completed)")
            progress.update(overall_task, advance=1)

    async def _process_input(self, progress: Progress, task: TaskID) -> None:
        self.logger.info("Processing input files...")
        files = [f for f in os.listdir(self.input_directory) if f.endswith('.sas')]
        progress.update(task, total=len(files))

        for file in files:
            file_path = os.path.join(self.input_directory, file)
            with open(file_path, 'r') as f:
                self.sas_modules[file] = f.read()
            progress.update(task, advance=1)
        
        self.stats["input_processed"]["files_processed"] = len(self.sas_modules)
        self.state_manager.update_state("input_processed", True)
        self.logger.info(f"Processed {len(self.sas_modules)} SAS files.")

    async def _extract_metadata(self, progress: Progress, task: TaskID) -> None:
        self.logger.info("Extracting metadata...")
        metadata_agent = MetadataExtractionAgent(self.sas_modules, model=self.model, debug=self.debug)
        self.metadata = await metadata_agent.run_extraction(progress, task)
        self.stats["metadata_extracted"] = metadata_agent.get_stats()
        for module in self.metadata:
            self.state_manager.mark_module_completed("metadata_extracted", module)
        self.logger.info("Metadata extraction completed.")

    async def _convert_code(self, progress: Progress, task: TaskID) -> None:
        self.logger.info("Converting SAS code to Python...")
        conversion_agent = ConversionAgent(self.sas_modules, self.metadata, model=self.model, debug=self.debug)
        self.converted_modules = await conversion_agent.run_conversion(progress, task)
        self.stats["code_converted"] = conversion_agent.get_conversion_stats()
        for module in self.converted_modules:
            self.state_manager.mark_module_completed("code_converted", module)
        self.logger.info("Code conversion completed.")

    async def _execute_and_fix_code(self, progress: Progress, task: TaskID) -> None:
        self.logger.info("Executing, fixing, and saving converted code...")
        execution_agent = ExecutionAgent(self.output_directory, model=self.model, debug=self.debug)
        
        modules = [m for m in self.metadata if "macro" not in m.lower()]
        progress.update(task, total=len(modules))

        for module_name in modules:
            if not self.state_manager.is_completed("code_executed", module_name):
                self.logger.info(f"Executing module: {module_name}")
                python_code = self.converted_modules.get(module_name, {})
                sas_code = {module_name: self.sas_modules.get(module_name, "")}
                
                for macro_name in self.metadata[module_name].get('metadata', {}).get('external_files', {}):
                    sas_code[macro_name] = self.sas_modules.get(macro_name, "")
                
                try:
                    result = await execution_agent.execute(
                        python_code, 
                        sas_code, 
                        self.metadata[module_name]
                    )
                    self.execution_results[module_name] = result
                    if result["success"]:
                        self.state_manager.mark_module_completed("code_executed", module_name)
                        self.logger.info(f"Successfully executed and saved module: {module_name}")
                    else:
                        self.logger.error(f"Failed to execute module: {module_name}. Error: {result.get('error', 'Unknown error')}")
                except Exception as e:
                    self.logger.exception(f"An error occurred while executing module {module_name}: {str(e)}")
            else:
                self.logger.info(f"Skipping already executed module: {module_name}")
            
            progress.update(task, advance=1)
        
        self.stats["code_executed"] = execution_agent.get_execution_stats()
        self.logger.info("Code execution, fixing, and saving completed.")

    def display_step_stats(self, step_name: str) -> None:
        stats = self.stats[step_name]
        table = Table(title=f"{step_name.replace('_', ' ').title()} Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        for key, value in stats.items():
            if key == "time":
                table.add_row("Time", f"{value:.2f} seconds")
            elif isinstance(value, (int, float)):
                table.add_row(key.replace("_", " ").title(), str(value))
        
        console.print(table)

    def display_overall_stats(self) -> None:
        table = Table(title="Overall Conversion Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        overall_stats = self.get_overall_stats()
        for key, value in overall_stats.items():
            if key == "total_time":
                table.add_row(key.replace("_", " ").title(), f"{value:.2f} seconds")
            elif isinstance(value, (int, float)):
                table.add_row(key.replace("_", " ").title(), str(value))
        
        console.print(table)

    def get_overall_stats(self) -> Dict[str, Any]:
        overall_stats = self.stats["overall"]
        overall_stats["total_time"] = sum(step_stats.get("time", 0) for step_stats in self.stats.values() if isinstance(step_stats, dict))
        
        for step_stats in self.stats.values():
            if isinstance(step_stats, dict):
                for key in ["total_tokens", "prompt_tokens", "completion_tokens", "total_calls"]:
                    overall_stats[key] += step_stats.get(key, 0)
        
        overall_stats["total_modules"] = len(self.sas_modules)
        overall_stats["converted_modules"] = len(self.converted_modules)
        overall_stats["executed_modules"] = len(self.execution_results)
        
        return overall_stats

    async def cleanup(self) -> None:
        # Implement cleanup logic here
        pass

# Usage example:
async def main():
    input_dir = "path/to/input/sas/files"
    output_dir = "path/to/output/python/files"
    state_file = "conversion_state.json"
    start_fresh = False  # Set to True to ignore existing state and start fresh

    orchestrator = ConversionOrchestrator(input_dir, output_dir, state_file, debug=True, start_fresh=start_fresh)
    await orchestrator.run_conversion()

if __name__ == "__main__":
    asyncio.run(main())