"""
Script to create CSV dataset from the Urdu ghazals data
"""
import os
import json
import pandas as pd
from pathlib import Path
import re

def clean_line(line):
    """Clean a single line of text"""
    # Remove extra whitespace
    line = re.sub(r'\s+', ' ', line.strip())
    # Remove empty lines
    if not line:
        return None
    return line

def process_poet_files(poet_dir):
    """Process all files for a single poet"""
    urdu_dir = os.path.join(poet_dir, 'ur')
    roman_dir = os.path.join(poet_dir, 'en')  # Roman transliteration is in 'en' folder
    
    pairs = []
    
    if not os.path.exists(urdu_dir) or not os.path.exists(roman_dir):
        return pairs
    
    # Get list of files in Urdu directory
    urdu_files = os.listdir(urdu_dir)
    
    for urdu_file in urdu_files:
        if urdu_file.startswith('.'):
            continue
            
        urdu_path = os.path.join(urdu_dir, urdu_file)
        roman_path = os.path.join(roman_dir, urdu_file)
        
        # Check if corresponding Roman file exists
        if not os.path.exists(roman_path):
            continue
            
        try:
            # Read Urdu file
            with open(urdu_path, 'r', encoding='utf-8') as f:
                urdu_lines = f.readlines()
            
            # Read Roman file
            with open(roman_path, 'r', encoding='utf-8') as f:
                roman_lines = f.readlines()
            
            # Clean and pair lines
            urdu_cleaned = [clean_line(line) for line in urdu_lines if clean_line(line)]
            roman_cleaned = [clean_line(line) for line in roman_lines if clean_line(line)]
            
            # Pair lines (assuming they correspond line by line)
            min_len = min(len(urdu_cleaned), len(roman_cleaned))
            for i in range(min_len):
                if urdu_cleaned[i] and roman_cleaned[i]:
                    pairs.append({
                        'urdu': urdu_cleaned[i],
                        'roman': roman_cleaned[i],
                        'poet': os.path.basename(poet_dir),
                        'poem': urdu_file
                    })
        
        except Exception as e:
            print(f"Error processing {urdu_file}: {e}")
            continue
    
    return pairs

def create_dataset():
    """Create the complete dataset"""
    base_dir = "data/urdu_ghazals_rekhta/dataset/dataset"
    
    if not os.path.exists(base_dir):
        print(f"Dataset directory not found: {base_dir}")
        return
    
    all_pairs = []
    poets = os.listdir(base_dir)
    
    print(f"Processing {len(poets)} poets...")
    
    for poet in poets:
        if poet.startswith('.'):
            continue
            
        poet_dir = os.path.join(base_dir, poet)
        if not os.path.isdir(poet_dir):
            continue
        
        print(f"Processing poet: {poet}")
        poet_pairs = process_poet_files(poet_dir)
        all_pairs.extend(poet_pairs)
        print(f"  Found {len(poet_pairs)} pairs")
    
    print(f"\nTotal pairs collected: {len(all_pairs)}")
    
    # Create DataFrame
    df = pd.DataFrame(all_pairs)
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['urdu', 'roman'])
    print(f"After removing duplicates: {len(df)}")
    
    # Save to CSV
    output_path = "data/urdu_ghazals_rekhta_dataset.csv"
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Dataset saved to: {output_path}")
    
    # Show sample data
    print("\nSample data:")
    print(df[['urdu', 'roman']].head(10))
    
    # Show statistics
    print(f"\nDataset Statistics:")
    print(f"Total pairs: {len(df)}")
    print(f"Unique poets: {df['poet'].nunique()}")
    print(f"Average Urdu length: {df['urdu'].str.len().mean():.1f}")
    print(f"Average Roman length: {df['roman'].str.len().mean():.1f}")
    
    return output_path

if __name__ == "__main__":
    dataset_path = create_dataset()
    print(f"\nDataset ready for training: {dataset_path}")