import os
import sys
import argparse
from models.classifier import ComplianceClassifier

def main():
    parser = argparse.ArgumentParser(description="Regulatory Compliance NLP System")
    parser.add_argument("--mode", required=True, choices=["classify", "train", "query"], help="Execution mode")
    parser.add_argument("--input", help="Path to input text file (required for classify mode)")
    parser.add_argument("--question", help="Question text (required for query mode)")
    
    args = parser.parse_args()
    
    if args.mode == "classify":
        if not args.input:
            print("Error: --input <file_path> is required in classify mode.")
            sys.exit(1)
            
        if not os.path.exists(args.input):
            print(f"Error: File '{args.input}' does not exist.")
            sys.exit(1)
            
        try:
            with open(args.input, "r", encoding="utf-8") as f:
                text = f.read().strip()
        except Exception as e:
            print(f"Error: Failed to read file '{args.input}': {e}")
            sys.exit(1)
            
        if not text:
            print("Error: Input file is empty.")
            sys.exit(1)
            
        print("Initializing classifier...")
        try:
            classifier = ComplianceClassifier()
            result = classifier.classify(text)
            
            # Extract Named Entities in parallel
            from preprocessing.nlp_pipeline import CompliancePreprocessor
            preprocessor = CompliancePreprocessor()
            entities = preprocessor.extract_entities(text)
            
            print("\n--- Compliance Analysis Results ---")
            print(f"Category:   {result['category']}")
            print(f"Confidence: {result['confidence']:.4f}")
            print("-----------------------------------")
            
            print("\n--- Extracted Named Entities ---")
            if entities:
                print(f"{'Entity Text':<30} | {'Label':<10} | {'Offsets':<10}")
                print("-" * 56)
                for ent in entities:
                    offsets = f"{ent['start']}-{ent['end']}"
                    print(f"{ent['text']:<30} | {ent['label']:<10} | {offsets:<10}")
            else:
                print("No entities extracted (Date, Org, Money).")
            print("-----------------------------------")
            
        except Exception as e:
            print(f"Error: Classification or Entity Extraction failed: {e}")
            sys.exit(1)
            
    elif args.mode == "train":
        print("Training mode is not yet implemented. Run this in Slice 5.")
        sys.exit(0)
    elif args.mode == "query":
        print("Query mode is not yet implemented. Run this in Slice 4.")
        sys.exit(0)

if __name__ == "__main__":
    main()
