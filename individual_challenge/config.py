#swap to "qwen2.5:3b" or "phi4-mini" or "llama3.1:8b" .
#MODEL = "phi4-mini"
#MODEL = "qwen2.5:3b"  
MODEL = "llama3.1:8b"  # local model served by Ollama; swap to qwen2.5:3b / phi4-mini if slow
EMBEDDING = "all-MiniLM-L6-v2"  # sentence-transformers embedding model; swap if you want a different one
# Preset companies for the dropdown:  name -> Yahoo Finance ticker
# (.DE = Frankfurt, .PA = Paris; US tickers have no suffix)
COMPANIES = {
    "NVIDIA":    "NVDA",
    "Tesla":     "TSLA",
    "SAP":       "SAP",
    "DHL":       "DHL.DE",
    "BMW":       "BMW.DE",
    "Lufthansa": "LHA.DE",
    "Siemens":   "SIE.DE",
    "Airbus":    "AIR.PA",
}
INDUSTRIES = {
    "NVIDIA":    "Semiconductors ",
    "Tesla":     "Automotive",
    "SAP":       "Enterprise Software",
    "DHL":       "Logistics",
    "BMW":       "Automotive",
    "Lufthansa": "Aviation",
    "Siemens":   "Industrial",
    "Airbus":    "Aerospace",
}
def collection_name(company):
    slug = "".join(c if c.isalnum() else "_" for c in company.lower()).strip("_")
    return f"docs_{slug}"[:63]

