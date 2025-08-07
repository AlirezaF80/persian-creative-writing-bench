import time
import logging
from typing import Dict, Any, List, Optional

class CreativeWritingTask:
    """
    Stores a single creative writing prompt, plus seed modifiers. For each seed modifier,
    we do a single test-model generation and then judge it with the judge model.
    Mirroring eqbench structure, but no multi-turn logic is needed.
    """

    def __init__(
        self,
        prompt_id: str,
        base_prompt: str,
        seed_modifiers: List[str],
        iteration_index: int,
        test_model: str,
        judge_model: str,
        test_max_tokens: Optional[int] = None,
        judge_max_tokens: Optional[int] = None,
        test_temperature: float = 0.7,
        judge_temperature: float = 0.0,
        test_min_p: Optional[float] = 0.1,
        judge_min_p: Optional[float] = None
    ):
        self.prompt_id = prompt_id
        self.base_prompt = base_prompt
        self.seed_modifiers = seed_modifiers
        self.iteration_index = iteration_index
        self.test_model = test_model
        self.judge_model = judge_model
        self.test_max_tokens = test_max_tokens
        self.judge_max_tokens = judge_max_tokens
        self.test_temperature = test_temperature
        self.judge_temperature = judge_temperature
        self.test_min_p = test_min_p
        self.judge_min_p = judge_min_p

        self.status = "initialized"
        self.start_time = None
        self.end_time = None
        self.error = None

        # For each seed-modifier:
        # { seed_mod: { "model_response": "...", "judge_scores": { ... }, "raw_judge_text": "..." } }
        self.results_by_modifier = {}

    def generate_creative_piece(self, api_clients, runs_file=None, run_key=None, save_interval=2):
        """
        For each seed modifier, if not already done, call the test model with base_prompt + seed.
        Retry up to 3 times if output is too short, then discard if still failing.
        """
        self.status = "in_progress"
        if not self.start_time:
            self.start_time = time.time()

        test_api = api_clients["test"]
        from utils.file_io import update_run_data

        for i, seed_modifier in enumerate(self.seed_modifiers):
            if seed_modifier in self.results_by_modifier and "model_response" in self.results_by_modifier[seed_modifier]:
                continue

            final_prompt = self.base_prompt.replace("<SEED>", seed_modifier)
            
            # Initialize result block
            self.results_by_modifier[seed_modifier] = {}
            
            # Try up to 3 times for short responses
            max_attempts = 3
            for attempt in range(1, max_attempts + 1):
                try:
                    response = test_api.generate(
                        self.test_model,
                        final_prompt,
                        max_tokens=self.test_max_tokens,
                        temperature=self.test_temperature,
                        min_p=self.test_min_p,
                        include_seed=False
                    )
                    
                    # Check if response is too short
                    if len(response.strip()) < 500:
                        if attempt < max_attempts:
                            logging.warning(f"Generated text too short ({len(response.strip())} chars), retry {attempt}/{max_attempts} "
                                        f"(prompt_id={self.prompt_id}, seed={seed_modifier})")
                            time.sleep(1)  # Brief pause before retry
                            continue
                        else:
                            # Final attempt also failed, mark as failed but don't treat error as content
                            logging.error(f"Generated text too short after {max_attempts} attempts "
                                        f"(prompt_id={self.prompt_id}, seed={seed_modifier})")
                            self.results_by_modifier[seed_modifier]["generation_failed"] = True
                            self.results_by_modifier[seed_modifier]["error_message"] = f"Text too short after {max_attempts} attempts"
                            break
                    
                    # Success - store the valid response and break retry loop
                    self.results_by_modifier[seed_modifier]["model_response"] = response.strip()
                    self.results_by_modifier[seed_modifier]["generation_failed"] = False
                    break
                    
                except Exception as e:
                    if attempt < max_attempts:
                        logging.warning(f"Generation error, retry {attempt}/{max_attempts} "
                                    f"(prompt_id={self.prompt_id}, seed={seed_modifier}): {str(e)}")
                        time.sleep(1)  # Brief pause before retry
                    else:
                        # Final attempt also failed, mark as failed but don't treat error as content
                        logging.error(f"Generation failed after {max_attempts} attempts "
                                    f"(prompt_id={self.prompt_id}, seed={seed_modifier}): {str(e)}")
                        self.results_by_modifier[seed_modifier]["generation_failed"] = True
                        self.results_by_modifier[seed_modifier]["error_message"] = str(e)

            # Save partial
            if runs_file and run_key and (i % save_interval == 0):
                conv_data = self.to_dict()
                update_run_data(runs_file, run_key, {
                    "creative_tasks": {
                        str(self.iteration_index): {
                            str(self.prompt_id): conv_data
                        }
                    }
                })

        self.status = "generated"
        self.end_time = time.time()
        if runs_file and run_key:
            conv_data = self.to_dict()
            update_run_data(runs_file, run_key, {
                "creative_tasks": {
                    str(self.iteration_index): {
                        str(self.prompt_id): conv_data
                    }
                }
            })

    def judge(self, api_clients, judge_prompt: str, creative_writing_criteria: Dict[str, str],
          negative_criteria: Dict[str, str], runs_file=None, run_key=None):
        """
        For each seed modifier, if there's a model_response and no generation failures,
        pass it to the judge model for scoring.
        """
        if self.status != "generated":
            logging.warning(f"Cannot judge a {self.status} CreativeWritingTask (prompt_id={self.prompt_id})")
            return

        judge_api = api_clients["judge"]
        from utils.file_io import update_run_data
        from core.scoring import parse_judge_scores_creative

        for seed_modifier, data_block in self.results_by_modifier.items():
            if "judge_scores" in data_block:
                continue

            # Skip judging if generation failed
            if data_block.get("generation_failed", False):
                data_block["judge_scores"] = {}
                data_block["raw_judge_text"] = f"[Skipping - generation error: {data_block.get('error_message', 'Unknown error')}]"
                continue

            model_text = data_block.get("model_response", "")
            if not model_text:
                # Skip if no model response (shouldn't happen with new logic but just in case)
                data_block["judge_scores"] = {}
                data_block["raw_judge_text"] = "[Skipping - empty generation]"
                continue
            
            # Build criterias
            higher_criteria_str = "\n".join(["- " + c for c in creative_writing_criteria])
            negative_criteria_str = "\n".join(["- " + c for c in negative_criteria])
            all_higher_criteria_desc_exist = all(v for v in creative_writing_criteria.values() if v is not None and v.strip())
            all_neg_criteria_desc_exist = all(v for v in negative_criteria.values() if v is not None and v.strip())
            if all_higher_criteria_desc_exist and all_neg_criteria_desc_exist:
                higher_criteria_str = "\n".join(
                    [f"- {c}: {creative_writing_criteria[c]}" for c in creative_writing_criteria]
                )
                negative_criteria_str = "\n".join(
                    [f"- {c}: {negative_criteria[c]}" for c in negative_criteria]
                )
            creative_criteria_str = "\n".join(
                [f"- {c}" for c in creative_writing_criteria] + [f"- {c}" for c in negative_criteria]
            )

            # Build final
            final_judge_prompt = judge_prompt.format(
                writing_prompt=self.base_prompt,
                test_model_response=model_text,
                higher_is_better_criteria=higher_criteria_str,
                lower_is_better_criteria=negative_criteria_str,
                creative_writing_criteria=creative_criteria_str
            )
            
            print(final_judge_prompt)

            try:
                judge_resp = judge_api.generate(
                    self.judge_model,
                    final_judge_prompt,
                    max_tokens=self.judge_max_tokens,
                    temperature=self.judge_temperature,
                    include_seed=True,
                    min_p=self.judge_min_p
                )
                scores_dict = parse_judge_scores_creative(judge_resp)
                data_block["judge_scores"] = scores_dict
                data_block["raw_judge_text"] = judge_resp
            except Exception as e:
                logging.error(f"[CreativeWritingTask] Judge error (prompt_id={self.prompt_id}, seed={seed_modifier}): {str(e)}")
                data_block["judge_scores"] = {}
                data_block["raw_judge_text"] = f"[ERROR: {str(e)}]"

            self.status = "completed"
            if runs_file and run_key:
                update_run_data(runs_file, run_key, {
                    "creative_tasks": {
                        str(self.iteration_index): {
                            str(self.prompt_id): self.to_dict()
                        }
                    }
                })

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt_id": self.prompt_id,
            "base_prompt": self.base_prompt,
            "seed_modifiers": self.seed_modifiers,
            "iteration_index": self.iteration_index,
            "test_model": self.test_model,
            "judge_model": self.judge_model,
            "test_max_tokens": self.test_max_tokens,
            "judge_max_tokens": self.judge_max_tokens,
            "test_temperature": self.test_temperature,
            "judge_temperature": self.judge_temperature,
            "test_min_p": self.test_min_p,
            "judge_min_p": self.judge_min_p,
            "status": self.status,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "error": self.error,
            "results_by_modifier": self.results_by_modifier
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        obj = cls(
            prompt_id=data["prompt_id"],
            base_prompt=data["base_prompt"],
            seed_modifiers=data["seed_modifiers"],
            iteration_index=data.get("iteration_index", 0),
            test_model=data["test_model"],
            judge_model=data["judge_model"]
        )
        obj.status = data.get("status", "initialized")
        obj.start_time = data.get("start_time")
        obj.end_time = data.get("end_time")
        obj.error = data.get("error")
        obj.results_by_modifier = data.get("results_by_modifier", {})
        # Backward compatibility: old runs won't have these fields
        obj.test_max_tokens = data.get("test_max_tokens")
        obj.judge_max_tokens = data.get("judge_max_tokens")
        # Backward compatibility defaults
        obj.test_temperature = data.get("test_temperature", 0.7)
        obj.judge_temperature = data.get("judge_temperature", 0.0)
        obj.test_min_p = data.get("test_min_p", 0.1)
        obj.judge_min_p = data.get("judge_min_p")
        return obj
