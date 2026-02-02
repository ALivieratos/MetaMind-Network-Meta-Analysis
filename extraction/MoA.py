import numpy as np
import fitz  # PyMuPDF for extracting text from PDFs
import aiohttp
import asyncio
import openai
import nest_asyncio
# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()
# Set your API keys
together_api_key = "Your Key"
openai.api_key = "Your Key"
# Together AI base URL for model inference
together_api_url = "https://api.together.xyz/v1/completions"
headers = {
    "Authorization": f"Bearer {together_api_key}",
    "Content-Type": "application/json"
}
# Generate model-specific prompts for clinical data extraction
def get_model_prompts(abstract_text):
    return {
        "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo": f"""
        Extract detailed clinical data from the text below:
        {abstract_text}
        Provide the following information:
        - Treatment names and endpoints.
        - Baseline clinical values (Mean, SD, or CI) and explicitly state if CI or SD is missing.
        - Final clinical remission values, including calculated Mean, SDs, or CIs, and explicitly indicate any missing values.
        - Weekly remission percentages (Weeks 4, 8, and 12) with SDs or CIs for all treatment groups.
        - Sample sizes for each treatment group.
        If any SD or CI values are missing, highlight this explicitly and suggest plausible methods for estimating them.
        Format the output as a structured table suitable for Network Meta-Analysis (NMA).
        """,
        "mistralai/Mistral-7B-Instruct-v0.3": f"""
        Extract clinical data for induction phase from the following text:
        {abstract_text}
        Include:
        - Treatment names and endpoints.
        - Baseline and final remission values with SDs or CIs, clearly identifying missing data.
        - Weekly remission percentages (Weeks 4, 8, and 12) with SDs or CIs for all treatment groups.
        - Sample sizes for each treatment group.
        For any missing SD or CI values, indicate their absence explicitly and propose how they might be calculated.
        Provide results in a structured table for NMA.
        """,
        "Qwen/Qwen2-72B-Instruct": f"""
        Extract induction phase clinical data from the following text:
        {abstract_text}
        Provide:
        - Baseline clinical values (Mean, SD, or CI) and clearly indicate if these values are missing.
        - Final clinical remission values with calculated Mean, SDs, or CIs, and highlight any missing values.
        - Weekly remission percentages (Weeks 4, 8, and 12) with SDs or CIs for all treatment groups.
        - Sample sizes for all treatment groups.
        Emphasize identifying and reporting missing SD or CI values, and suggest estimation methods if possible.
        Summarize in a structured format suitable for NMA.
        """
    }
# Aggregator prompt template
aggregator_prompt_template = """
    You are provided with multiple responses from different models analyzing clinical study data.
    Your task is to extract the following data for a Network Meta-Analysis (NMA):
    - Baseline clinical values for each treatment group (Mean, SD, or CI). Indicate if CI or SD values are missing.
    - Final clinical remission values for each treatment group, with calculated Mean, SDs, or CIs, and explicitly flag missing values.
    - Weekly remission percentages (Weeks 4, 8, 12) for each treatment group with SDs or CIs.
    - Sample sizes for all treatment groups.
    Highlight any missing SD or CI values and propose methods for estimation if feasible.
    Summarize the output in a clear, tabular format suitable for NMA.
"""
# Helper function to validate and log missing CI/SD values
def validate_ci_sd_extraction(batch_result):
    missing_data_flag = False
    if "CI" not in batch_result and "SD" not in batch_result:
        missing_data_flag = True
    return missing_data_flag, batch_result
async def run_together_model_with_retry(model, prompt):
    payload = {"model": model, "prompt": prompt, "max_tokens": 2048}
    async with aiohttp.ClientSession() as session:
        for _ in range(3):  # Retry logic
            async with session.post(together_api_url, json=payload, headers=headers) as response:
                if response.status == 200:
                    return await response.text()
    return None  # Return None if all retries fail
# Helper function to synthesize responses with GPT-4o
async def run_llm(model, prompt, prev_responses=None):
    system_prompt = "You are provided with multiple responses to a user query. Please synthesize these responses into coherent, accurate responses and calculate Mean clinical values, remission percentages, and Standard deviations (SDs) or confidence intervals (CIs) for each treatment group."
    try:
        if prev_responses:
            combined_responses = "\n\n".join(prev_responses)
            prompt = f"{system_prompt}\n\nPrevious responses:\n{combined_responses}"
        if model == "gpt-4o":
            # Await the coroutine to get the response object
            response = await openai.ChatCompletion.acreate(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2048,
                temperature=0.7,
                stream=True
            )
            # Collect the streamed output into a single string
            response_content = ""
            async for chunk in response:  # Iterate over the response object
                if "choices" in chunk:
                    response_content += chunk["choices"][0]["delta"].get("content", "")
            return response_content.strip()  # Return the full response as a string
        else:
            return await run_together_model_with_retry(model, prompt)
    except Exception as e:
        print(f"Error with model {model}: {e}")
        return None
def extract_text_from_pdfs_in_batches(pdf_path):
    with fitz.open(pdf_path) as pdf:
        # Extract all text from the PDF
        full_text = ""
        for page in pdf:
            full_text += page.get_text()
        # Split text into two roughly equal parts
        midpoint = len(full_text) // 2
        return [full_text[:midpoint], full_text[midpoint:]]
# Multi-layer aggregation with batch-level and overall streaming
async def process_pdf_batches(pdf_path):
    text_batches = extract_text_from_pdfs_in_batches(pdf_path)
    final_aggregated_data = []
    for i, batch_text in enumerate(text_batches):
        print(f"\nProcessing Batch {i + 1}/{len(text_batches)}...\n")
        model_prompts = get_model_prompts(batch_text)
        # Layer 1: Run reference models asynchronously
        initial_responses = await asyncio.gather(
            *[run_llm(model, model_prompts[model]) for model in model_prompts]
        )
        # Filter out None responses
        valid_responses = [r for r in initial_responses if r]
        # Layers 2 and 3: Iterative aggregation with GPT-4o
        for layer in range(2):
            print(f"\nRunning Aggregation Layer {layer + 2}...\n")
            valid_responses = await asyncio.gather(
                *[run_llm("gpt-4o", aggregator_prompt_template, prev_responses=valid_responses) for _ in model_prompts]
            )
            valid_responses = [r for r in valid_responses if r]  # Filter out None again
        # Validate for CI/SD presence and highlight missing data
        for response in valid_responses:
            missing_ci_sd, validated_result = validate_ci_sd_extraction(response)
            if missing_ci_sd:
                print(f"Warning: Missing CI/SD in Batch {i + 1}: {validated_result}\n")
        # Store aggregated batch result
        if valid_responses:
            batch_result = "\n".join(valid_responses)
            print(f"\nDetailed Batch {i + 1} Output:\n{batch_result}\n")
            final_aggregated_data.append(batch_result)
        else:
            print(f"Batch {i + 1} failed to produce valid responses.")
    # Combine all batch results for overall aggregation
    if final_aggregated_data:
        combined_final_prompt = "\n\n".join(final_aggregated_data)
        print("\n================== Final Document Aggregated Output ==================\n")
        final_stream = await openai.ChatCompletion.acreate(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": aggregator_prompt_template},
                {"role": "user", "content": combined_final_prompt}
            ],
            max_tokens=2048,
            temperature=0.7,
            stream=True
        )
        # Stream overall output
        final_output = ""
        async for chunk in final_stream:
            if "choices" in chunk:
                final_output += chunk["choices"][0]["delta"].get("content", "")
                print(chunk["choices"][0]["delta"].get("content", ""), end='', flush=True)
        return final_output.strip()
# Main async function
async def main():
    pdf_path = "/content/adalimumab.pdf"  # Update with the actual path
    await process_pdf_batches(pdf_path)
# Run the script
await main()


