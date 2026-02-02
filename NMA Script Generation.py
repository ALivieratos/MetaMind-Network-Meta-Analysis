import requests
import os
import subprocess
import re
API_URL = "https://api.openai.com/v1/chat/completions"
API_KEY = "Your Key"
# Pseudocode instructions for OpenAI to generate the R script
prompt_instructions = """
Generate an R script that follows these pseudocode instructions exactly:
Load necessary libraries: brms, dplyr, tibble.
Define a dataset (`UC_data`) with:
   - Study details (study_id).
   - Treatment groups (treatment).
   - Response counts (response_count).
   - Total sample sizes (total_sample_size).
Ensure placebo is the reference category for treatment comparisons.
```r
UC_data$treatment <- relevel(factor(UC_data$treatment), ref = "placebo")```
Fit a Bayesian random-effects model using:
   - **Binomial likelihood (logit link) for response counts over total sample size.**
   - **Fixed effects:** Treatment group comparisons (**placebo as reference**).
   - **Random effects:** Study-level variability, allowing **random intercepts and slopes** for treatment.
   - **Formula:**
     ```
     response_count | trials(total_sample_size) ~ treatment + (1 + treatment | study_id)
     ```
   - **Priors:**
     - Treatment effects: `Normal(0, 3.5)`
     - Study variance: `Normal(0, 0.5)`, constrained to non-negative values (`lb=0`).
   - **Sampling settings:**
     - `chains = 4`, `iter = 2000`, `warmup = 1000`, `seed = 123`, `adapt_delta = 0.98`
 Extracts posterior summaries:
       - Computes mean estimates, standard errors, and 95% credible intervals (2.5%, 50%, 97.5% quantiles) using:
         ```r
         posterior_summ <- as.data.frame(posterior_summary(model, probs = c(0.025, 0.5, 0.975)))
         ```
   - Extracts diagnostics:
       - Obtains summary of the fitted model:
         ```r
         model_summary <- summary(model)
         diagnostics <- as.data.frame(model_summary$fixed)
         ```
   - Converts row names to a column to facilitate merging of Rhat values:
       ```r
       posterior_summ <- posterior_summ %>% rownames_to_column("Parameter")
       diagnostics <- diagnostics %>% rownames_to_column("Parameter") %>% select(Parameter, Rhat)
       ```
   - Renames parameters for consistency by removing the `b_` prefix:
       ```r
       posterior_summ$Parameter <- gsub("^b_", "", posterior_summ$Parameter)
       ```
   - Merges posterior summaries with Rhat values:
       ```r
       posterior_summ_df <- left_join(posterior_summ, diagnostics, by = "Parameter") %>%
         filter(grepl("^treatment", Parameter)) %>%
         mutate(Parameter = gsub("^treatment", "d", Parameter)) %>%
         select(Parameter, Estimate, Est.Error, Q2.5, Q97.5, Rhat)
       ```
   - Prints the final posterior summary table:
       ```r
       cat("\nFinal Posterior Summary with Rhat Values:\n")
       print(posterior_summ_df)
       ```
   - Extracts and prints the between-study heterogeneity estimate (tau):
       ```r
       tau_value <- as.numeric(VarCorr(model)$study_id$sd[1])
       cat("\nTau (between-study heterogeneity):", tau_value, "\n")
       ```
   - Displays model diagnostics:
       ```r
       cat("\nSamples were drawn using NUTS(diag_e) at", Sys.time(), "\n")
       cat("For each parameter, n_eff is a crude measure of effective sample size,\n")
       cat("and Rhat is the potential scale reduction factor on split chains (at\n")
       cat("convergence, Rhat=1).\n")
       ```
2. Execute the R script inline using Python:
   - Use `subprocess.run()` to execute the R script using `Rscript -e`.
   - Capture both standard output and errors to verify correct execution.
   - Print all relevant outputs for user inspection.
3. Expected output:
   - A summary table displaying treatment effect estimates, confidence intervals, and Rhat values.
   - The between-study heterogeneity estimate (tau).
   - Convergence diagnostics including effective sample size and Rhat values.
Ensure the LLM generates the entire R script based on these instructions without explicitly providing any R code.
"""
# Construct request payload
payload = {
    "model": "gpt-4o",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant that generates R scripts based on given pseudocode."},
        {"role": "user", "content": prompt_instructions}
    ],
    "max_tokens": 3000,
    "temperature": 0.0
}
# Send API request to OpenAI
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}
response = requests.post(API_URL, json=payload, headers=headers)
# Handle response
if response.status_code == 200:
    response_data = response.json()
    r_script = response_data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
    if r_script:
        print("Generated R Script:\n")
        print(r_script)
        # Execute the R script using subprocess
        print("\nExecuting the R script...\n")
        result = subprocess.run(["Rscript", "-e", r_script], capture_output=True, text=True)
        # Display the output
        print("Standard Output:\n", result.stdout)
        print("Standard Error:\n", result.stderr)
    else:
        print("No R script was returned. Check the API response.")
else:
    print(f"Error {response.status_code}: {response.text}")

