from datasets import load_dataset, DatasetDict

def load_and_map_dataset(tiny_slice: bool = False):
    """
    Loads and cleans Hugging Face LexGLUE SCOTUS dataset.
    Maps categories deterministically as per ADR 0001:
      0 -> Financial Crime
      1 -> Fraud
      2 -> Regulatory Breach
    """
    if tiny_slice:
        # Fast local mock DatasetDict for blazingly fast unit tests
        from datasets import Dataset
        mock_data = {
            "text": [
                "The corporate executive was indicted for money laundering and offshore bank structuring to evade banking disclosures.",
                "The vendor was guilty of bribery, falsifying billing invoices, and receiving illegal mail kickbacks.",
                "The administrative commission issued fines for statutory environmental standard breaches violating EPA regulation."
            ],
            "label": [0, 1, 2]
        }
        train_ds = Dataset.from_dict(mock_data)
        val_ds = Dataset.from_dict(mock_data)
        return DatasetDict({"train": train_ds, "validation": val_ds})
        
    # Load actual LexGLUE SCOTUS benchmark
    dataset = load_dataset("lex_glue", "scotus")
    
    # Heuristics mapping keywords
    fin_crime_kws = {"laundering", "insider trading", "antitrust", "monopoly", "tax evasion", "securities", "banking", "offshore"}
    fraud_kws = {"bribery", "mail fraud", "wire fraud", "kickback", "embezzled", "deceit", "theft", "falsified"}
    reg_breach_kws = {"administrative agency", "environmental regulation", "statutory standard", "interstate commerce", "breach of regulation", "statutory guideline", "regulatory rule"}
    
    def map_scotus_record(example):
        text = example["text"].lower()
        scotus_label = example["label"]
        
        mapped_label = -1
        
        # SCOTUS Label 7: Economic Activity, Label 11: Federal Taxation
        if scotus_label in {7, 11}:
            if any(kw in text for kw in fin_crime_kws):
                mapped_label = 0
            elif any(kw in text for kw in fraud_kws):
                mapped_label = 1
            elif any(kw in text for kw in reg_breach_kws):
                mapped_label = 2
        # SCOTUS Label 0: Criminal Procedure
        elif scotus_label == 0:
            if any(kw in text for kw in fraud_kws):
                mapped_label = 1
            elif any(kw in text for kw in fin_crime_kws):
                mapped_label = 0
        # SCOTUS Label 9: Federalism, Label 10: Interstate Relations
        elif scotus_label in {9, 10}:
            if any(kw in text for kw in reg_breach_kws):
                mapped_label = 2
                
        example["mapped_label"] = mapped_label
        return example
        
    # Execute mapping
    mapped_dataset = dataset.map(map_scotus_record)
    
    # Filter unmapped categories
    filtered_train = mapped_dataset["train"].filter(lambda x: x["mapped_label"] != -1)
    filtered_val = mapped_dataset["validation"].filter(lambda x: x["mapped_label"] != -1)
    filtered_test = mapped_dataset["test"].filter(lambda x: x["mapped_label"] != -1)
    
    # Finalize labels schema
    def finalize_schema(example):
        example["label"] = example["mapped_label"]
        return example
        
    final_train = filtered_train.map(finalize_schema).remove_columns(["mapped_label"])
    final_val = filtered_val.map(finalize_schema).remove_columns(["mapped_label"])
    final_test = filtered_test.map(finalize_schema).remove_columns(["mapped_label"])
    
    return DatasetDict({
        "train": final_train,
        "validation": final_val,
        "test": final_test
    })
