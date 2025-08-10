# Creative Writing Benchmark Documentation

## 1. Project Overview & Philosophy

This project is a benchmark designed to evaluate the creative writing capabilities of large language models (LLMs). It is the system used for the Creative Writing leaderboard on [EQ-Bench.com](https://eqbench.com/creative_writing.html).

The benchmark's philosophy is to provide a reliable *relative* ranking of models by combining two scoring methods:
*   **Rubric Scoring**: An initial assessment where each generated piece is scored in isolation against a detailed rubric. This provides insight into specific criteria but can saturate at high performance levels.
*   **Elo Scoring**: A more discriminative relative rating derived from pairwise comparisons against other models. This is the primary metric for leaderboard ranking.

The system uses a hybrid rubric and Elo scoring system designed for enhanced discrimination, especially at the top end of model performance. It uses prompts designed to challenge models in areas like humor, romance, spatial awareness, and unique perspectives.

## 2. Getting Started

### Prerequisites

*   Python 3.x
*   API keys for the test and judge models.
*   Required Python packages.

### Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/EQ-bench/creative-writing-bench.git
    cd creative-writing-bench
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Download NLTK data:**
    ```python
    import nltk
    nltk.download('punkt')
    nltk.download('cmudict')
    ```

4.  **Configure API Keys:**
    *   Copy the example environment file: `cp .env.example .env`
    *   Edit the `.env` file and add your API keys and desired endpoint URLs for the test and judge models.

### Running the Benchmark

Execute the main script with your desired parameters. For a leaderboard-comparable score, use the recommended judge model and the provided runs file.

```bash
python3 creative_writing_bench.py \
    --test-model "your-model-provider/your-model-name" \
    --judge-model "anthropic/claude-3.7-sonnet" \
    --runs-file "creative_bench_runs.json" \
    --iterations 3 \
    --threads 50 \
    --verbosity "INFO" \
    --test-max-tokens 4000 \
    --judge-max-tokens 1000
```

**Important Arguments:**

*   `--test-model`: Identifier for the model you want to evaluate.
*   `--judge-model`: Identifier for the judge model. Use `anthropic/claude-3.7-sonnet` for leaderboard-comparable scores.
*   `--runs-file`: Path to the JSON file storing run data. **To get an Elo score comparable to the EQ-Bench leaderboard, you must use the `creative_bench_runs.json` file provided in this repository**, as it contains the necessary historical data for Elo calculation.
*   `--iterations`: Number of generation iterations per prompt (default and recommended: 3).
*   `--threads`: Number of parallel threads for generation and judging. Adjust based on your API rate limits.
*   `--test-max-tokens`: Optional. Max tokens for test model generations. If omitted, defaults to 4000.
*   `--judge-max-tokens`: Optional. Max tokens for judge model scoring. If omitted, defaults to 1000.

### Output

*   **Console**: Progress will be logged to the console.
*   `creative_bench_runs.json`: This file is updated with detailed run data, including generated text and judge scores.
*   `elo_results.json`: This file stores the results of the Elo analysis, including pairwise comparisons and final ratings. The final `elo_norm` score is the primary leaderboard metric.

## 3. Overall Execution Workflow

A typical benchmark run proceeds as follows:
1.  **Generation**: The test model generates responses to 32 distinct writing prompts across 3 iterations (96 items total). Generation uses a `temperature` of 0.7 and `min_p` of 0.1 to encourage creativity. The generation output length is capped by `--test-max-tokens` (default 4000).
2.  **Rubric Scoring**: Each generated piece is individually assessed by a judge model against a comprehensive rubric.
3.  **Initial Elo Inference**: The aggregate rubric score is used to estimate an initial Elo rating for the model relative to existing models in `elo_results.json`.
4.  **Pairwise Matchups**: The model is compared against neighboring models on the leaderboard in pairwise matchups. The judge determines the better output across several criteria.
5.  **Glicko Calculation**: Elo scores are calculated using the Glicko-2 rating system, modified to incorporate the win margin from pairwise comparisons. This process loops until model positions stabilize.
6.  **Final Elo Calculation**: The definitive leaderboard Elo score is computed based on all comparisons.
7.  **Normalization**: Raw Elo scores are normalized by anchoring specific models (e.g., `deepseek/deepseek-r1` to 1500) to ensure comparability over time.

## 3.1 Detailed Execution Workflow

1.  The user executes `creative_writing_bench.py` with the required arguments (e.g., test model, judge model).
2.  `run_eq_bench_creative` in `benchmark.py` is called.
3.  A unique `run_key` is created, and a new entry is made in `creative_bench_runs.json`.
4.  Prompts and criteria are loaded from the `data/` directory.
5.  `CreativeWritingTask` objects are created for each prompt.
6.  The `ThreadPoolExecutor` starts multiple threads to process the tasks.
7.  **In each thread**:
    a.  A `CreativeWritingTask` object calls `generate_creative_piece()`.
    b.  The `APIClient` sends the prompt to the test model API.
    c.  The generated text is saved in the task object.
    d.  The task's status is updated to `generated`.
    e.  `update_run_data` is called to save the progress to the JSON file.
8.  After generation is complete, the threads move on to judging.
9.  **In each thread**:
    a.  A `CreativeWritingTask` object calls `judge()`.
    b.  The `APIClient` sends the generated text and the judging prompt to the judge model API (capped by `--judge-max-tokens`, default 1000).
    c.  The judge's response is parsed by `parse_judge_scores_creative`.
    d.  The scores are saved in the task object.
    e.  The task's status is updated to `completed`.
    f.  `update_run_data` is called to save the scores.
10. Once all tasks are judged, `compute_benchmark_results_creative` is called to calculate the final rubric score and bootstrap statistics.
11. If Elo is enabled, `run_elo_analysis_creative` is called to perform pairwise comparisons and calculate Elo ratings.
12. The final results are saved to `creative_bench_runs.json` and `elo_results.json`.

## 4. Core Components

The `core/` directory contains the main logic for the benchmark.

### `benchmark.py`

This is the main orchestrator of the benchmark. The `run_eq_bench_creative` function orchestrates the entire process:

-   **Initialization**: Creates or resumes a benchmark run, identified by a unique `run_key`.
-   **Data Loading**: Loads creative writing prompts, judging criteria, and other necessary data from the `data/` directory.
-   **Task Management**: Creates and manages `CreativeWritingTask` objects for each prompt and iteration.
-   **Execution Flow**:
    1.  Calls the `generate_creative_piece` method on each task to get the test model's output.
    2.  Calls the `judge` method to have the judge model score the output.
    3.  Computes the final rubric-based scores using functions from `core.scoring`.
    4.  Optionally, initiates the Elo analysis using `core.elo.run_elo_analysis_creative`.
-   **Concurrency**: Uses a `ThreadPoolExecutor` to run generation and judging tasks in parallel, significantly speeding up the process.

### `conversation.py`

This file defines the `CreativeWritingTask` class, which is the central data structure for a single task. An instance of this class represents one creative writing piece to be generated and judged.

-   **Attributes**:
    -   `prompt_id`, `base_prompt`, `seed_modifiers`: Information about the writing prompt.
    -   `test_model`, `judge_model`: The models being used.
    -   `status`: Tracks the state of the task (e.g., `initialized`, `in_progress`, `generated`, `completed`).
    -   `results_by_modifier`: A dictionary that stores the generated text, the judge's scores, and the raw text from the judge for each seed modifier.
-   **Methods**:
    -   `generate_creative_piece()`: Interacts with the test model via the API client to generate the creative text.
    -   `judge()`: Interacts with the judge model to get scores for the generated text.
    -   `to_dict()` and `from_dict()`: Methods for serializing and deserializing the task object, which is essential for saving and resuming runs.

### `scoring.py`

This module handles the parsing and calculation of scores.

-   `parse_judge_scores_creative()`: Uses regular expressions to parse the free-text response from the judge model and extract a dictionary of numerical scores for each criterion.
-   `invert_if_negative()`: Inverts scores for negative criteria (where a lower score is better) so that all scores are on a "higher is better" scale.
-   `compute_creative_scores()`: Aggregates all the individual rubric scores for a set of tasks to produce a single average score on a 0-20 scale.
-   `compute_single_benchmark_score_creative()`: Converts the 0-20 score into the final 0-100 `eqbench_creative_score`.
-   `bootstrap_benchmark_stability_creative()`: Performs a bootstrap analysis by resampling the tasks to calculate a confidence interval for the final score, which helps to understand the stability and reliability of the benchmark results.

### `elo.py`

This is a sophisticated module for ranking models using a rating system.
-   **Pairwise Comparison**: It orchestrates pairwise comparisons between the outputs of different models for the same prompt. The judge model is asked to decide which of the two pieces is better.
-   **Glicko-2 Implementation**: It uses the `glicko2` library to calculate Elo ratings for the models based on the outcomes of the pairwise comparisons. This is more advanced than a standard Elo system as it also considers the rating deviation (uncertainty) of a model's rating.
-   **Fractional Outcomes**: Instead of just win/loss, the system can calculate a fractional outcome based on the margin of victory, providing more nuanced data to the rating system.
-   **Normalization**: The final Elo scores are normalized against a set of anchor models (e.g., `deepseek/deepseek-r1` is anchored to 1500) to ensure comparability across different benchmark runs.
-   `run_elo_analysis_creative()`: The main function that manages the entire Elo analysis process.

### `metrics.py`

This module provides functions to calculate various objective metrics on the generated text. These metrics can be used for deeper analysis of a model's output.

-   `calculate_complexity_index()`: Measures the linguistic complexity of the text.
-   `calculate_slop_index()`: Calculates a "slop" score, which is a measure of how much the text contains common, uncreative, or undesirable phrases.
-   `calculate_repetition_metric()`: Measures the repetitiveness of the text by identifying words that are used more frequently than in standard English.
-   `get_multi_prompt_ngrams()`: Identifies common n-grams (phrases of n words) that a model uses across different prompts, which can be another indicator of repetitiveness.

## 5. Utility Modules

The `utils/` directory contains helper modules that provide common functionality used throughout the project.

### `api.py`

-   **`APIClient`**: A wrapper around the `requests` library for making calls to LLM APIs.
    - Respects CLI-specified token caps: includes `max_tokens` when provided; for models like `o3` that require `max_completion_tokens`, it automatically maps the value accordingly.
-   **Configuration**: It reads API keys and URLs from environment variables (`.env` file), allowing for easy configuration of different API providers for the test and judge models.
-   **Error Handling**: Includes robust retry logic with exponential backoff to handle transient network errors or API rate limits.
-   **Model-Specific Payloads**: Can adjust the API request payload for specific models.

### `file_io.py`

-   **Thread-Safe File Operations**: This module is crucial for the stability of the benchmark, especially when running with multiple threads. It provides thread-safe functions for reading and writing JSON files.
-   **File Locking**: It uses a per-file locking mechanism (`threading.Lock`) to prevent race conditions where multiple threads might try to write to the same file simultaneously.
-   `update_run_data()`: This function is particularly important. It performs a deep, nested merge of new data into the main runs file (`creative_bench_runs.json`). This allows different threads to update their specific part of the run data without overwriting the progress of other threads.
-   **Atomic Writes**: Uses an atomic write pattern (write to a temporary file, then rename) to prevent data corruption if the application crashes during a write operation.

### `logging_setup.py`

-   A simple module to configure the logging for the application. It allows setting the log level via a command-line argument or an environment variable.

## 6. Data Files

The `data/` directory contains the data used to run the benchmark.

-   `creative_writing_prompts_v3.json`: The main file containing the writing prompts.
-   `creative_writing_criteria.txt`: The list of criteria that the judge model uses to score the creative pieces.
-   `negative_criteria.txt`: A subset of the criteria where a lower score is better (e.g., "Verbosity").
-   `creative_writing_judging_prompt.txt`: The template for the prompt that is sent to the judge model.
-   `pairwise_prompt.txt`: The template for the prompt used in Elo pairwise comparisons.
-   `slop_list*.json`: Files containing lists of words and phrases used by the `calculate_slop_index` metric.

## 7. Bias Mitigation and Limitations

### Bias Mitigation

The benchmark attempts to control for several common biases in LLM judging:
*   **Length Bias**: Mitigated by truncating outputs to 4000 characters before judging.
*   **Position Bias**: Mitigated by running comparisons in both A/B and B/A orders and averaging the results.
*   **Verbosity/Poetic Incoherence Bias**: Addressed through specific judging criteria that penalize these traits.

Biases **not** explicitly controlled for include potential judge self-bias, stylistic preferences, and "slop" bias (favoring overused tropes).

### Limitations

*   **Subjectivity**: Creative quality is subjective; the judge's assessment may differ from human preferences.
*   **Judge Limitations**: The recommended judge (`anthropic/claude-3.7-sonnet`) is good but not infallible.
*   **English Only**: The benchmark currently evaluates English language writing only.
*   **Cost**: Running the benchmark involves API costs (approx. $10 per model using the recommended judge).

**Always view benchmark scores as a guide, not absolute truth. Read the sample outputs!**

## 8. How to Extend the Benchmark

### Adding a New Model

1.  **Configure API**: Add the API key and URL for the new model to your `.env` file. You can create new variables like `MY_MODEL_API_KEY` and `MY_MODEL_API_URL`.
2.  **Update `api.py` (if necessary)**: If the new model requires a different payload structure than the default OpenAI format, you will need to add a new condition in the `APIClient.generate` method to accommodate it.
3.  **Run the Benchmark**: Run `creative_writing_bench.py` and pass the new model's name as the `test_model` argument. You will also need to update your environment variables so that `TEST_API_KEY` and `TEST_API_URL` point to your new model's credentials.

### Adding New Prompts

1.  **Edit JSON**: Add new entries to the `data/creative_writing_prompts_v3.json` file. Follow the existing format, providing a `writing_prompt` and a list of `seed_modifiers`.

### Adding New Metrics

1.  **Create Function**: Add a new function to `core/metrics.py` that takes a string of text and returns a numerical score.
2.  **Integrate into Analysis**: To have this metric automatically calculated and displayed, you would need to modify the analysis part of the code, which is currently handled in the `generate_results_html.ipynb` notebook. You would add a call to your new metric function and display the result in the output HTML.

## 9. Citation

If you use this benchmark in your work, please cite the repository:

```bibtex
@misc{creative-writing-bench-v3,
  author = {Samuel J Paech},
  title = {EQ-Bench Creative Writing Benchmark v3},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/EQ-bench/creative-writing-bench}}
}
```
