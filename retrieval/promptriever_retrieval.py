import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel, PeftConfig
import numpy as np
from Bio import Entrez
class Promptriever:
    def __init__(self, model_name_or_path):
        self.model, self.tokenizer = self.get_model(model_name_or_path)
        self.model.eval().cuda()
    def get_model(self, peft_model_name):
        # Load the PEFT configuration to get the base model name
        peft_config = PeftConfig.from_pretrained(peft_model_name)
        base_model_name = peft_config.base_model_name_or_path
        # Load the base model and tokenizer
        base_model = AutoModel.from_pretrained(base_model_name)
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "right"
        # Load and merge the PEFT model
        model = PeftModel.from_pretrained(base_model, peft_model_name)
        model = model.merge_and_unload()
        # Configure max length for the model
        model.config.max_length = 512
        tokenizer.model_max_length = 512
        return model, tokenizer
    def create_batch_dict(self, tokenizer, input_texts):
        max_length = self.model.config.max_length
        batch_dict = tokenizer(
            input_texts,
            max_length=max_length - 1,
            return_token_type_ids=False,
            return_attention_mask=False,
            padding=False,
            truncation=True,
        )
        batch_dict["input_ids"] = [
            input_ids + [tokenizer.eos_token_id]
            for input_ids in batch_dict["input_ids"]
        ]
        return tokenizer.pad(
            batch_dict,
            padding=True,
            pad_to_multiple_of=8,
            return_attention_mask=True,
            return_tensors="pt",
        )
    def encode(self, sentences, max_length: int = 2048, batch_size: int = 4):
        all_embeddings = []
        for i in range(0, len(sentences), batch_size):
            batch_texts = sentences[i : i + batch_size]
            batch_dict = self.create_batch_dict(self.tokenizer, batch_texts)
            batch_dict = {
                key: value.to(self.model.device) for key, value in batch_dict.items()
            }
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    outputs = self.model(**batch_dict)
                    last_hidden_state = outputs.last_hidden_state
                    sequence_lengths = batch_dict["attention_mask"].sum(dim=1) - 1
                    batch_size = last_hidden_state.shape[0]
                    reps = last_hidden_state[
                        torch.arange(batch_size, device=last_hidden_state.device),
                        sequence_lengths,
                    ]
                    embeddings = F.normalize(reps, p=2, dim=-1)
                    all_embeddings.append(embeddings.cpu().numpy())
        return np.concatenate(all_embeddings, axis=0)
# PubMed Search Function
def search_pubmed(query, max_results=100):
    Entrez.email = "achilleas.livieratos@gmail.com"  # Add your email here
    handle = Entrez.esearch(
        db="pubmed",
        term=query,
        retmax=max_results,
        retmode="xml",
        sort="relevance",
    )
    results = Entrez.read(handle)
    handle.close()
    return results["IdList"]
# Initialize the Promptriever model
model = Promptriever("samaya-ai/promptriever-llama2-7b-v1")
# PubMed query
query = (
    "ulcerative colitis AND placebo AND (mirikizumab OR upadacitinib OR filgotinib OR ustekinumab OR tofacitinib OR etrasimod)"
)
# Fetch PubMed IDs
pubmed_ids = search_pubmed(query, max_results=100)
print("PubMed IDs retrieved:", pubmed_ids)

if pubmed_ids:
    # Fetch articles from PubMed
    handle = Entrez.efetch(db="pubmed", id=pubmed_ids, retmode="xml")
    records = Entrez.read(handle)
    handle.close()
    # Extract abstracts and metadata
    documents = []
    metadata = []
    for article in records["PubmedArticle"]:
        title = article["MedlineCitation"]["Article"].get("ArticleTitle", "No Title")
        journal = article["MedlineCitation"]["Article"]["Journal"].get("Title", "No Journal")
        pub_date = article["MedlineCitation"]["Article"]["Journal"].get("JournalIssue", {}).get("PubDate", "No Date")
        abstract = article["MedlineCitation"]["Article"].get("Abstract", {}).get("AbstractText", [])
        if abstract:
            documents.append(" ".join(abstract))
            metadata.append({"Title": title, "Journal": journal, "PubDate": pub_date})
    # Specify target journals
    TARGET_JOURNALS = {"Lancet (London, England)", "The New England journal of medicine"}
    # Filter documents for specific journals
    filtered_documents = []
    filtered_metadata = []
    for idx, meta in enumerate(metadata):
        if meta["Journal"] in TARGET_JOURNALS:
            filtered_documents.append(documents[idx])
            filtered_metadata.append(meta)
    # Check if filtered results are available
    if not filtered_documents:
        print("No relevant articles found in the specified journals.")
    else:
        # Define query instruction
        instruction = (
            "A relevant document would describe clinical trials on ulcerative colitis "
            "where patients were tested against placebo. The document should detail the use of treatments like "
            "mirikizumab, upadacitinib, filgotinib, or ustekinumab or etrasimod or tofacitinib and include clinical efficacy results."
        )
        input_text = f"query: {query.strip()} {instruction.strip()}".strip()
        # Encode query and the filtered documents
        query_embedding = model.encode([input_text])
        doc_embeddings = model.encode(filtered_documents)
        # Calculate similarities
        similarities = np.dot(query_embedding, doc_embeddings.T)[0]
        sorted_indices = np.argsort(similarities)[::-1]
        # Display top results
        print("Top relevant PubMed articles in specified journals:")
        for idx in sorted_indices[:10]:  # Top 10 results
            print(f"Title: {filtered_metadata[idx]['Title']}")
            print(f"Journal: {filtered_metadata[idx]['Journal']}")
            print(f"Publication Date: {filtered_metadata[idx]['PubDate']}")
            print(f"Abstract: {filtered_documents[idx]}")
            print(f"Similarity: {similarities[idx]:.4f}\n")
else:
    print("No PubMed articles found.")

