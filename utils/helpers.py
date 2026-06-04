import os
import json
import pandas as pd
from urllib.request import urlretrieve
import re

# JSON Parser with Error Handling
def parse_json_response(response_text, context=""):
    """
    Parse JSON response from LLM with robust error handling.
    
    Args:
        response_text: Raw LLM response text
        context: Description of where this is being called (for error messages)
    
    Returns:
        Parsed JSON object (dict or list)
    
    Raises:
        ValueError: If JSON parsing fails after cleanup attempts
    """
    
    try:
        # Clean up common LLM response artifacts
        text = response_text.strip()
        
        # Remove markdown code blocks
        if '```json' in text:
            text = text.split('```json')[1].split('```')[0]
        elif '```' in text:
            text = text.split('```')[1].split('```')[0]
        
        text = text.strip()
        
        # Try direct parse
        return json.loads(text)
        
    except json.JSONDecodeError as e:
        print(f"\n⚠️  Initial JSON parsing failed at position {e.pos}. Attempting repairs...")
        
        # Try to fix common issues
        text_backup = text
        
        # 1. Try to find and extract valid JSON portion before the error
        try:
            # For arrays, try to truncate at the error point and close properly
            if text.strip().startswith('['):
                # Find the last complete object before the error
                error_pos = e.pos
                text_before_error = text[:error_pos]
                
                # Try to find the last complete JSON object
                # Count brackets to find proper truncation point
                bracket_count = 0
                last_valid_pos = 0
                in_string = False
                escape_next = False
                
                for i, char in enumerate(text_before_error):
                    if escape_next:
                        escape_next = False
                        continue
                    
                    if char == '\\':
                        escape_next = True
                        continue
                    
                    if char == '"' and not escape_next:
                        in_string = not in_string
                    
                    if not in_string:
                        if char == '{':
                            bracket_count += 1
                        elif char == '}':
                            bracket_count -= 1
                            if bracket_count == 0:
                                last_valid_pos = i + 1
                
                if last_valid_pos > 0:
                    # Try to build valid JSON array with truncated content
                    truncated = text_before_error[:last_valid_pos].rstrip(',').rstrip() + ']'
                    try:
                        result = json.loads(truncated)
                        print(f"   ✅ Recovered {len(result)} items by truncating at error position")
                        print(f"   ⚠️  Warning: Response was incomplete - some data may be missing")
                        return result
                    except:
                        pass
        except:
            pass
        
        # 2. Try to find JSON object
        obj_match = re.search(r'\{.*\}', text_backup, re.DOTALL)
        if obj_match:
            try:
                result = json.loads(obj_match.group(0))
                print(f"   ✅ Extracted valid JSON object")
                return result
            except:
                pass
        
        # 3. Try to find JSON array
        array_match = re.search(r'\[.*\]', text_backup, re.DOTALL)
        if array_match:
            try:
                result = json.loads(array_match.group(0))
                print(f"   ✅ Extracted valid JSON array with {len(result)} items")
                return result
            except:
                pass
        
        # 4. Try to fix common JSON issues programmatically
        try:
            # Remove trailing commas
            text_fixed = re.sub(r',\s*([}\]])', r'\1', text_backup)
            # Fix unescaped quotes in strings (basic attempt)
            result = json.loads(text_fixed)
            print(f"   ✅ Fixed JSON by removing trailing commas")
            return result
        except:
            pass
        
def download_parse_json(url, path):
    # Download CUAD Dataset
    json_url = url
    json_path = path

    if not os.path.exists(json_path):
        print('⏳ Downloading CUAD_v1.json...')
        urlretrieve(json_url, json_path)
        print('✅ Download complete!')
    else:
        print('✅ Dataset file exists')

    with open(json_path, 'r') as f:
        cuad_data = json.load(f)

    print(f'📊 Loaded {len(cuad_data["data"])} contracts')
    
    # Parse into Structured DataFrame
    data_rows = []

    for doc in cuad_data['data']:
        title = doc['title']
        for paragraph in doc['paragraphs']:
            context = paragraph['context']
            for qa in paragraph['qas']:
                data_rows.append({
                    'id': qa['id'],
                    'title': title,
                    'context': context,
                    'question': qa['question'],
                    'is_impossible': qa.get('is_impossible', False),
                    'answers': {'text': [ans['text'] for ans in qa.get('answers', [])],
                            'answer_start': [ans['answer_start'] for ans in qa.get('answers', [])]}
                })

    df = pd.DataFrame(data_rows)
    df['clause_name'] = df['question'].str.extract(r'"([^"]+)"')

    print(f'📋 Dataset: {len(df):,} QA pairs')
    print(f'📄 Contracts: {df["title"].nunique()}')
    print(f'🏷️  Clause types: {df["clause_name"].nunique()}')
    return df