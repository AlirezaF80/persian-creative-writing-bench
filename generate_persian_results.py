#!/usr/bin/env python3
"""
Persian Creative Writing Benchmark - Results Generator
Simplified version focusing on rubric scores only (no ELO analysis)
"""

import json
import pandas as pd
import numpy as np
from collections import defaultdict
import os
from typing import Dict, List, Any, Optional, Tuple

# Model family assignments for Persian benchmark models
model_to_family = {
    'google/gemma-3-12b-it:free': 'Google',
    'gemini-2.0-flash': 'Google',
    'c4ai-aya-expanse-32b': 'C4AI',
    'meta-llama/llama-3.1-405b-instruct:free': 'Meta-Llama',
    'meta-llama/llama-3.3-70b-instruct:free': 'Meta-Llama',
    'qwen/qwen3-235b-a22b:free': 'Qwen',
    'qwen/qwen3-14b:free': 'Qwen',
    'deepseek/deepseek-chat-v3-0324:free': 'DeepSeek',
    'gemini-2.5-pro': 'Google',
    'gemini-2.5-flash-lite': 'Google'
}

# Family colors for visualization
family_colors = {
    'Google':     '#8a5cf5',
    'C4AI':       '#ff6b6b',
    'Meta-Llama': '#1e3d59',
    'Qwen':       '#b2df8a',
    'DeepSeek':   '#1eb980',
    'Other':      '#cccccc'
}

# Model name substitutions for display
MODEL_NAME_SUBS = {
    'google/gemma-3-12b-it:free': 'Gemma 3 12B IT (Free)',
    'gemini-2.0-flash': 'Gemini 2.0 Flash',
    'c4ai-aya-expanse-32b': 'C4AI Aya Expanse 32B',
    'meta-llama/llama-3.1-405b-instruct:free': 'Llama 3.1 405B Instruct (Free)',
    'meta-llama/llama-3.3-70b-instruct:free': 'Llama 3.3 70B Instruct (Free)',
    'qwen/qwen3-235b-a22b:free': 'Qwen3 235B A22B (Free)',
    'qwen/qwen3-14b:free': 'Qwen3 14B (Free)',
    'deepseek/deepseek-chat-v3-0324:free': 'DeepSeek Chat V3 0324 (Free)',
    'gemini-2.5-pro': 'Gemini 2.5 Pro',
    'gemini-2.5-flash-lite': 'Gemini 2.5 Flash Lite'
}

# Configuration
PERSIAN_BENCH_FILE = "persian_bench.json"
RESULTS_DIR = "results_persian"

def get_updated_model_name(original: str) -> str:
    """Get the updated model name for display."""
    return MODEL_NAME_SUBS.get(original, original)

def sanitize_model_name(model_name: str) -> str:
    """Sanitize model name for use in filenames."""
    sanitized = model_name.replace("/", "__")
    unsafe_chars = r'<>:"|?*\\'
    for char in unsafe_chars:
        sanitized = sanitized.replace(char, '-')
    return sanitized

def load_json_file(file_path: str) -> Dict:
    """Load data from a JSON file."""
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return {}
    
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            print(f"Error decoding JSON from {file_path}")
            return {}

def calculate_creative_writing_scores(bench_data: Dict, model_name: str) -> Tuple[float, Dict]:
    """
    Calculate the creative writing scores for a model from Persian benchmark data.
    
    Args:
        bench_data: The dict containing all benchmark data (run_key -> run_data)
        model_name: The name of the model to calculate the score for
        
    Returns:
        Tuple of (overall_average_score, iterations_dict)
    """
    # Get negative criteria (where lower score is better)
    neg_criteria = [
        "Meandering", "Weak Dialogue", "Tell-Don't-Show", "Unsurprising or Uncreative",
        "Amateurish", "Purple Prose", "Overwrought", "Incongruent Ending Positivity",
        "Unearned Transformations", "Translated Feel", "Cultural Incongruity"
    ]
    
    # Find matching runs for the model
    matching_runs = []
    for run_key, run_data in bench_data.items():
        if run_data.get("test_model") == model_name:
            matching_runs.append((run_key, run_data))
    
    if not matching_runs:
        return 0.0, {}
    
    # Calculate scores by iteration
    iterations = {}
    total_score_sum = 0
    total_score_count = 0
    
    # Process all matching runs
    for run_key, run_data in matching_runs:
        creative_tasks = run_data.get("creative_tasks", {})
        
        for iter_idx, prompt_data in creative_tasks.items():
            if iter_idx not in iterations:
                iterations[iter_idx] = {
                    "score": 0,
                    "entries": [],
                    "prompt_count": 0
                }
            
            iter_score_sum = 0
            iter_score_count = 0
            
            for prompt_id, task_data in prompt_data.items():
                if task_data.get("status") not in ["completed", "judged"]:
                    continue
                
                # Extract entries from results_by_modifier
                results_by_mod = task_data.get("results_by_modifier", {})
                for seed_mod, result_block in results_by_mod.items():
                    judge_scores = result_block.get("judge_scores", {})
                    
                    # Create entry for this result
                    entry = {
                        "prompt_id": prompt_id,
                        "base_prompt": task_data.get("base_prompt", ""),
                        "seed_modifier": seed_mod,
                        "model_response": result_block.get("model_response", ""),
                        "judge_scores": judge_scores,
                        "raw_judge_text": result_block.get("raw_judge_text", "")
                    }
                    iterations[iter_idx]["entries"].append(entry)
                    iterations[iter_idx]["prompt_count"] += 1
                    
                    # Calculate scores for this entry
                    for metric, val in judge_scores.items():
                        if isinstance(val, (int, float)) and val <= 20:
                            # Invert negative criteria scores
                            score_val = (20 - val) if metric in neg_criteria else val
                            iter_score_sum += score_val
                            iter_score_count += 1
                            total_score_sum += score_val
                            total_score_count += 1
            
            # Update iteration score
            if iter_score_count > 0:
                iterations[iter_idx]["score"] = round(iter_score_sum / iter_score_count, 2)
    
    overall_avg_score = round(total_score_sum / total_score_count, 2) if total_score_count > 0 else 0.0
    return overall_avg_score, iterations

def generate_model_report(model_name: str, bench_data: Dict, save_to_file: bool = False) -> str:
    """
    Generate an HTML report for a specific model from Persian benchmark data.
    
    Args:
        model_name: The name of the model to generate the report for
        bench_data: The dict of benchmark data (run_key -> run_data)
        save_to_file: Whether to save the report to an HTML file
        
    Returns:
        HTML content as a string
    """
    # Calculate scores
    overall_avg_score, iterations = calculate_creative_writing_scores(bench_data, model_name)
    
    if not iterations:
        return f"<h2>No data found for model: {model_name}</h2>"
    
    display_model_name = get_updated_model_name(model_name)
    
    # Sort iterations by score (descending)
    sorted_iterations = sorted(iterations.items(), key=lambda x: x[1]["score"], reverse=True)
    
    # Generate HTML
    html_output = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Persian Creative Writing Results: {display_model_name}</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                color: #333;
                background-color: #f5f5f5;
                max-width: 1000px;
                margin: 20px auto;
                padding: 20px;
            }}
            h1 {{
                text-align: center;
                color: #2c3e50;
                border-bottom: 3px solid #3498db;
                padding-bottom: 10px;
            }}
            .summary {{
                background: white;
                padding: 20px;
                border-radius: 8px;
                margin: 20px 0;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .iteration-container {{
                background: white;
                margin: 20px 0;
                border-radius: 8px;
                overflow: hidden;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .iteration-header {{
                background: #3498db;
                color: white;
                padding: 15px 20px;
                cursor: pointer;
                font-weight: bold;
                font-size: 1.1em;
            }}
            .iteration-header:hover {{
                background: #2980b9;
            }}
            .content-block {{
                padding: 20px;
                border-top: 1px solid #ecf0f1;
            }}
            .prompt-text {{
                background: #f8f9fa;
                padding: 15px;
                border-left: 4px solid #3498db;
                margin-bottom: 15px;
                font-style: italic;
            }}
            .response-content {{
                white-space: pre-wrap;
                background: #f8f9fa;
                padding: 15px;
                border-radius: 4px;
                margin-bottom: 15px;
                border: 1px solid #e9ecef;
                max-height: 400px;
                overflow-y: auto;
            }}
            .judge-content {{
                background: #e8f4fd;
                padding: 15px;
                border-radius: 4px;
                border: 1px solid #bee5eb;
            }}
            .scores-container {{
                margin-top: 10px;
                font-size: 0.9em;
                color: #666;
            }}
            .collapsible-content {{
                display: none;
            }}
            .expanded {{
                display: block;
            }}
            .toggle-icon {{
                display: inline-block;
                width: 20px;
                text-align: center;
                margin-right: 10px;
            }}
            .back-button {{
                display: inline-block;
                padding: 10px 20px;
                background: #3498db;
                color: white;
                text-decoration: none;
                border-radius: 4px;
                margin-bottom: 20px;
            }}
            .back-button:hover {{
                background: #2980b9;
            }}
            .family-badge {{
                display: inline-block;
                padding: 4px 8px;
                background: #ecf0f1;
                border-radius: 4px;
                font-size: 0.8em;
                color: #666;
                margin-left: 10px;
            }}
        </style>
    </head>
    <body>
        <a href="index.html" class="back-button">← Back to Index</a>
        <h1>Persian Creative Writing Results: {display_model_name}</h1>
        
        <div class="summary">
            <h2>Summary</h2>
            <p><strong>Overall Average Score:</strong> {overall_avg_score:.2f}/20</p>
            <p><strong>Number of Iterations:</strong> {len(iterations)}</p>
            <p><strong>Model Family:</strong> <span class="family-badge">{model_to_family.get(model_name, 'Other')}</span></p>
        </div>
    """
    
    # Add iterations
    for display_idx, (iter_idx, iter_data) in enumerate(sorted_iterations):
        is_first = display_idx == 0
        entries = iter_data["entries"]
        
        html_output += f"""
        <div class="iteration-container">
            <div class="iteration-header" onclick="toggleContent('iteration-{iter_idx}')">
                <span class="toggle-icon">{'−' if is_first else '+'}</span>
                Iteration {display_idx + 1} — Score: {iter_data['score']:.2f}/20 ({len(entries)} entries)
            </div>
            <div id="iteration-{iter_idx}" class="collapsible-content {'expanded' if is_first else ''}">
                <div class="content-block">
        """
        
        # Add entries for this iteration
        for entry_idx, entry in enumerate(entries):
            if entry_idx > 0:
                html_output += "<hr style='border: none; border-top: 2px dashed #ccc; margin: 20px 0;'>"
            
            # Add prompt if available
            prompt_text = entry.get("base_prompt", "")
            seed_modifier = entry.get("seed_modifier", "")
            if prompt_text:
                display_prompt = prompt_text
                if seed_modifier:
                    display_prompt = f"{prompt_text}\n\n**Seed Modifier:** {seed_modifier}"
                
                html_output += f"""
                        <div class="prompt-text">
                            <strong>Prompt (ID: {entry.get('prompt_id', 'N/A')}):</strong><br>{display_prompt}
                        </div>"""
            
            # Add model response
            response_text = entry.get("model_response", "")
            if response_text:
                # Truncate very long responses for display
                if len(response_text) > 2000:
                    truncated_text = response_text[:2000] + "... [truncated]"
                else:
                    truncated_text = response_text
                
                html_output += f"""
                        <div class="response-content">
                            <strong>Model Response:</strong><br>{truncated_text}
                        </div>"""
            
            # Add judge scores
            judge_scores = entry.get("judge_scores", {})
            raw_judge_text = entry.get("raw_judge_text", "")
            if judge_scores:
                scores_list = []
                for metric, score in judge_scores.items():
                    if isinstance(score, (int, float)):
                        scores_list.append(f"{metric}: {score}")
                
                if scores_list:
                    html_output += f"""
                        <div class="judge-content">
                            <strong>Judge Evaluation:</strong><br>
                            <div class="scores-container">
                                <strong>Scores:</strong> {', '.join(scores_list)}
                            </div>"""
                    
                    # Add raw judge text if available
                    if raw_judge_text and len(raw_judge_text.strip()) > 0:
                        # Truncate long judge text
                        if len(raw_judge_text) > 1000:
                            truncated_judge = raw_judge_text[:1000] + "... [truncated]"
                        else:
                            truncated_judge = raw_judge_text
                        
                        html_output += f"""
                            <br><br>
                            <strong>Judge's Full Response:</strong><br>
                            <div style="font-size: 0.9em; color: #555; max-height: 200px; overflow-y: auto;">
                                {truncated_judge}
                            </div>"""
                    
                    html_output += "</div>"
        
        html_output += """
                </div>
            </div>
        </div>"""
    
    # Add JavaScript for toggling
    html_output += """
        <script>
            function toggleContent(id) {
                const element = document.getElementById(id);
                if (!element) return;
                
                const isExpanded = element.classList.contains('expanded');
                const header = element.previousElementSibling;
                const toggleIcon = header ? header.querySelector('.toggle-icon') : null;
                
                if (isExpanded) {
                    element.classList.remove('expanded');
                    if (toggleIcon) toggleIcon.textContent = '+';
                } else {
                    element.classList.add('expanded');
                    if (toggleIcon) toggleIcon.textContent = '−';
                }
            }
        </script>
    </body>
    </html>
    """
    
    if save_to_file:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        sanitized_name = sanitize_model_name(get_updated_model_name(model_name))
        filename = f"{RESULTS_DIR}/{sanitized_name}.html"
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(html_output)
            print(f"Report saved to {filename}")
        except IOError as e:
            print(f"Error saving report to {filename}: {e}")
    
    return html_output

def generate_index_page(model_scores: Dict, save_to_file: bool = True) -> str:
    """Generate an index page with links to all model reports."""
    
    # Sort models by score
    sorted_models = sorted(model_scores.items(), key=lambda x: x[1]['score'], reverse=True)
    
    html_output = f"""
           <!DOCTYPE html>
           <html lang="en">
           <head>
               <meta charset="UTF-8">
               <title>Persian Creative Writing Benchmark - Results Index</title>
               <meta name="viewport" content="width=device-width, initial-scale=1">
               <style>
                   body {{
                       font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                       line-height: 1.6;
                       color: #333;
                       background-color: #f5f5f5;
                       max-width: 1200px;
                       margin: 20px auto;
                       padding: 20px;
                   }}
                   h1 {{
                       text-align: center;
                       color: #2c3e50;
                       border-bottom: 3px solid #3498db;
                       padding-bottom: 10px;
                   }}
                   .summary-stats {{
                       background: white;
                       padding: 20px;
                       border-radius: 8px;
                       margin: 20px 0;
                       box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                   }}
                   .models-table {{
                       background: white;
                       border-radius: 8px;
                       overflow: hidden;
                       box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                       margin: 20px 0;
                   }}
                   table {{
                       width: 100%;
                       border-collapse: collapse;
                   }}
                   th, td {{
                       padding: 12px 15px;
                       text-align: left;
                       border-bottom: 1px solid #ecf0f1;
                   }}
                   th {{
                       background-color: #3498db;
                       color: white;
                       font-weight: bold;
                   }}
                   tr:hover {{
                       background-color: #f8f9fa;
                   }}
                   .model-link {{
                       color: #3498db;
                       text-decoration: none;
                       font-weight: bold;
                   }}
                   .model-link:hover {{
                       text-decoration: underline;
                   }}
                   .score-high {{ color: #27ae60; font-weight: bold; }}
                   .score-medium {{ color: #f39c12; font-weight: bold; }}
                   .score-low {{ color: #e74c3c; font-weight: bold; }}
                   .family-badge {{
                       display: inline-block;
                       padding: 4px 8px;
                       border-radius: 4px;
                       font-size: 0.8em;
                       color: white;
                       font-weight: bold;
                   }}
               </style>
           </head>
           <body>
               <h1>Persian Creative Writing Benchmark - Results Index</h1>
               
               <div class="summary-stats">
                   <h2>Summary Statistics</h2>
                   <p><strong>Total Models:</strong> {len(sorted_models)}</p>
                   <p><strong>Average Score:</strong> {np.mean([data['score'] for data in model_scores.values()]):.2f}/20</p>
                   <p><strong>Highest Score:</strong> {max([data['score'] for data in model_scores.values()]):.2f}/20</p>
                   <p><strong>Lowest Score:</strong> {min([data['score'] for data in model_scores.values()]):.2f}/20</p>
               </div>
               
               <div class="models-table">
                   <table>
                       <thead>
                           <tr>
                               <th>Rank</th>
                               <th>Model</th>
                               <th>Score</th>
                               <th>Iterations</th>
                               <th>Family</th>
                               <th>Report</th>
                           </tr>
                       </thead>
                       <tbody>
           """
    
    for rank, (model, data) in enumerate(sorted_models, 1):
        display_name = get_updated_model_name(model)
        score = data['score']
        iterations = data['iterations']
        family = data['family']
        
        # Score color coding
        if score >= 15:
            score_class = "score-high"
        elif score >= 10:
            score_class = "score-medium"
        else:
            score_class = "score-low"
        
        # Family color
        family_color = family_colors.get(family, family_colors['Other'])
        
        # Sanitized filename
        sanitized_name = sanitize_model_name(get_updated_model_name(model))
        report_filename = f"{sanitized_name}.html"
        
        html_output += f"""
                    <tr>
                        <td><strong>{rank}</strong></td>
                        <td>{display_name}</td>
                        <td class="{score_class}">{score:.2f}</td>
                        <td>{iterations}</td>
                        <td><span class="family-badge" style="background-color: {family_color}">{family}</span></td>
                        <td><a href="{report_filename}" class="model-link">View Report</a></td>
                    </tr>
        """
    
    html_output += """
                </tbody>
            </table>
        </div>
    </body>
    </html>
    """
    
    if save_to_file:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        filename = f"{RESULTS_DIR}/index.html"
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(html_output)
            print(f"Index page saved to {filename}")
        except IOError as e:
            print(f"Error saving index page to {filename}: {e}")
    
    return html_output

def main():
    """Main function to process Persian benchmark data and generate reports."""
    print("Persian Creative Writing Benchmark - Results Generator")
    print("=" * 60)
    
    # Load Persian benchmark data
    print(f"Loading Persian benchmark data from {PERSIAN_BENCH_FILE}...")
    bench_data = load_json_file(PERSIAN_BENCH_FILE)
    
    if not bench_data:
        print("No data found in Persian benchmark file.")
        return
    
    print(f"Loaded {len(bench_data)} entries from Persian benchmark.")
    
    # Get unique models
    unique_models = set()
    for run_key, run_data in bench_data.items():
        model = run_data.get("test_model")
        if model:
            unique_models.add(model)
    
    print(f"\nFound {len(unique_models)} unique models:")
    for model in sorted(unique_models):
        display_name = get_updated_model_name(model)
        family = model_to_family.get(model, "Other")
        print(f"  - {display_name} ({family})")
    
    # Calculate overall scores for all models
    print("\nCalculating overall scores...")
    model_scores = {}
    for model in unique_models:
        score, iterations = calculate_creative_writing_scores(bench_data, model)
        model_scores[model] = {
            'score': score,
            'iterations': len(iterations),
            'family': model_to_family.get(model, 'Other')
        }
    
    # Sort by score and display
    sorted_models = sorted(model_scores.items(), key=lambda x: x[1]['score'], reverse=True)
    
    print("\n--- Model Rankings (by Average Score) ---")
    print("Rank | Model | Score | Iterations | Family")
    print("-----|-------|-------|------------|--------")
    
    for rank, (model, data) in enumerate(sorted_models, 1):
        display_name = get_updated_model_name(model)
        score = data['score']
        iterations = data['iterations']
        family = data['family']
        print(f"{rank:4d} | {display_name[:30]:<30} | {score:5.2f} | {iterations:10d} | {family}")
    
    # Generate and save HTML reports for all models
    print("\nGenerating HTML reports for all models...")
    for model in unique_models:
        display_name = get_updated_model_name(model)
        print(f"Processing: {display_name}")
        try:
            generate_model_report(model, bench_data, save_to_file=True)
        except Exception as e:
            print(f"  ERROR generating report for {model}: {e}")
    
    # Generate index page
    print("\nGenerating index page...")
    generate_index_page(model_scores, save_to_file=True)
    
    print(f"\nAll reports saved to '{RESULTS_DIR}' directory.")
    
    # Generate summary statistics
    print("\n--- Summary Statistics ---")
    
    # Calculate statistics by family
    family_stats = defaultdict(list)
    for model, data in model_scores.items():
        family = data['family']
        family_stats[family].append(data['score'])
    
    print("\nPerformance by Family:")
    for family, scores in family_stats.items():
        avg_score = np.mean(scores)
        std_score = np.std(scores)
        print(f"{family}: {avg_score:.2f} ± {std_score:.2f} (n={len(scores)})")
    
    # Overall statistics
    all_scores = [data['score'] for data in model_scores.values()]
    print(f"\nOverall Statistics:")
    print(f"Mean Score: {np.mean(all_scores):.2f}")
    print(f"Std Score: {np.std(all_scores):.2f}")
    print(f"Min Score: {np.min(all_scores):.2f}")
    print(f"Max Score: {np.max(all_scores):.2f}")
    print(f"Total Models: {len(all_scores)}")
    
    print(f"\nResults are available in the '{RESULTS_DIR}' directory.")
    print("Open 'index.html' to view the main results page.")

if __name__ == "__main__":
    main()
