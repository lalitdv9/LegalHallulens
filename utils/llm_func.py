from utils.llm import call_llm, count_tokens, get_model_name
from utils.helpers import parse_json_response
from utils.clauses_prompts import CLAUSE_DEFINITIONS, CLAUSE_TO_TYPE
import pandas as pd
import time
import traceback
import inspect
# from utils.llm_ollama import call_llm

# LLM Judge for Semantic Equivalence
def judge_semantic_match(clause_name, ai_answer, gt_answer, model):
    """Determine if AI answer is semantically equivalent to ground truth."""
    prompt = f"""You are an expert legal contract verifier.

TASK:
Determine whether ANSWER 1 (AI) is semantically equivalent to ANSWER 2 (Ground Truth) for the SAME clause and contract.

CLAUSE: {clause_name}

ANSWER 1 (AI Generated):
{ai_answer}

ANSWER 2 (Ground Truth):
{gt_answer}

DECISION CRITERIA (BE STRICT ON PRECISION):
Return "equivalent": true ONLY IF all of the following hold:
1) CORE FACTS MATCH: The same parties/actors, rights/obligations, and conditions are stated.
2) NUMERIC PRECISION MATCHES: Any amounts, percentages, thresholds, caps, quantities, and units (including time basis like per month/per year) are the same. Any mismatch => equivalent=false.
3) TEMPORAL PRECISION MATCHES: Any dates, durations, notice periods, renewal terms, survival periods, and timelines are the same. Any mismatch => equivalent=false.
4) MODALITY/POLARITY MATCHES: must/shall vs may, prohibited vs permitted, and any negation (not/unless/except) must match. Any mismatch => equivalent=false.
5) EXCEPTIONS/CARVE-OUTS: If either answer includes an exception, carve-out, or condition, the other must include the same exception/condition in substance. Otherwise => equivalent=false.

ALLOWABLE DIFFERENCES:
- Formatting, whitespace, and punctuation.
- Reordering of equivalent statements.
- Minor paraphrases that do not change any of the precise facts above.

OUTPUT (JSON ONLY):
Return ONLY a valid JSON object:
{{
  "equivalent": true/false, 
  "reason": "one short sentence", 
  "mismatch_type": "none|numeric|temporal|obligation|scope|missing_condition|extra_condition|other"
}}

RULES:
- If either answer is empty or says "Not present" while the other contains content, equivalent=false.
- If the AI answer is a subset of the ground truth but misses a required condition/exception, equivalent=false.
- Do not add any extra text outside the JSON.
"""
    
    # Keep compatibility across llm backends: only pass num_ctx if supported.
    call_kwargs = {
        'model': model,
        'temperature': 0.0,
        'max_tokens': 300,
    }
    if 'num_ctx' in inspect.signature(call_llm).parameters:
        call_kwargs['num_ctx'] = 4096

    text = call_llm(prompt, **call_kwargs)
    result = parse_json_response(text, context=f"judge_semantic_match for {clause_name}")
    judge_model = get_model_name(model)
    
    return result['equivalent'], result['reason'], result['mismatch_type'], judge_model
  
  
  

def extract_contract_clauses(contract_title, contract_df, model, run_id=1, num_runs=1, max_extraction_retries=2):
    """
    Extract clauses from a single contract (AI extraction only, no metrics).
    
    Args:
        contract_title: Title of the contract
        contract_df: DataFrame containing contract data
        model: Model name to use for extraction (required)
        run_id: Starting run number (default: 1)
        num_runs: Number of extraction runs to perform (default: 1)
        max_extraction_retries: Number of times to retry if JSON parsing fails per run
    
    Returns:
        If num_runs == 1:
            dict with keys: 
                - contract_title: str
                - run_id: int
                - merged_df: DataFrame (AI + GT merged with run_id column)
                - error: str or None (if failed)
        
        If num_runs > 1:
            list of dicts (one per run), each with same structure as above
    """
    
    # Multi-run wrapper
    if num_runs > 1:
        import uuid
        from datetime import datetime
        
        results = []
        contract_group_id = str(uuid.uuid4())
        
        print(f'\n{"="*80}')
        print(f'🔄 MULTI-RUN EXTRACTION: {num_runs} runs')
        print(f'📄 Contract: {contract_title[:60]}...')
        print(f'🔗 Group ID: {contract_group_id}')
        print(f'{"="*80}')
        
        for i in range(num_runs):
            current_run_id = run_id + i
            print(f'\n[Run {i+1}/{num_runs}] Starting extraction (run_id={current_run_id})...')
            
            # Call single extraction
            result = _extract_single_run(
                contract_title, 
                contract_df,
                model=model,
                run_id=current_run_id, 
                max_extraction_retries=max_extraction_retries
            )
            
            # Add multi-run metadata
            result['contract_group_id'] = contract_group_id
            result['run_timestamp'] = datetime.now().isoformat()
            result['total_runs'] = num_runs
            result['extraction_model'] = get_model_name(model)
            
            results.append(result)
            
            status = '✅' if result['error'] is None else '❌'
            print(f'{status} Run {i+1}/{num_runs} completed')
            
            # Brief pause between runs
            if i < num_runs - 1:
                time.sleep(0.5)
        
        successful = sum(1 for r in results if r['error'] is None)
        print(f'\n{"="*80}')
        print(f'✅ Multi-run complete: {successful}/{num_runs} successful')
        print(f'{"="*80}')
        
        return results
    
    # Single run (original behavior)
    else:
        return _extract_single_run(contract_title, contract_df, model, run_id, max_extraction_retries)


def _extract_single_run(contract_title, contract_df, model, run_id=1, max_extraction_retries=2):
    """
    Internal function to extract clauses from a single contract for one run.
    """
    try:
        print(f'\n{"="*80}')
        print(f'📄 Extracting: {contract_title[:70]}... (Run {run_id})')
        print(f'{"="*80}')
        
        # Extract contract text
        contract_text = contract_df['context'].iloc[0]
        gt_present_count = (~contract_df['is_impossible']).sum()
        print(f'📊 Length: {len(contract_text):,} chars | Clauses: {len(contract_df)} | GT-Present: {gt_present_count}')
        
        # AI Extraction (using original CUAD-compliant prompt)
        print('\n🤖 Running AI Extraction...')
        extraction_prompt = f"""You are a legal AI assistant analyzing a commercial contract.

You must answer each CUAD claim type below using ONLY the provided contract text.
For each claim type, you must decide whether it is present in the contract text.
If it is NOT present, do NOT provide any extracted spans.

CONTRACT UNDER REVIEW:
Title: {contract_title}

CONTRACT TEXT:
{contract_text}

================================================================================
TASK: CUAD-STYLE CLAUSE DETECTION (USE DEFINITIONS EXACTLY AS PROVIDED)
================================================================================

For EACH claim type definition in the list below:
- If present: return ALL contract span(s) that together fully capture the clause's operative meaning.
- If not present: mark it as impossible to answer and leave "answer" empty.

CRITICAL COMPLETENESS RULE (MOST IMPORTANT):
- If a clause type appears in multiple places (e.g., a restriction plus a termination trigger, or an insurance requirement plus cancellation/notice conditions), you MUST return ALL relevant spans.
- Do not stop after finding the first good match.

EXTRACTION RULES:
- Use only the contract text above (no outside knowledge).
- Spans must be verbatim (or near-verbatim) quotes from the contract.
- Each span should be minimal but COMPLETE for that clause fragment.
  - Include exceptions, carve-outs, conditions, triggers, notice periods, and "subject to / provided that / except / unless / however" language.
- If important details are in the next sentence, include it even if it makes the span longer.
- If multiple spans apply, return multiple spans in an array (do not merge distant sections).
- Do NOT guess section numbers. If a section/heading appears in the extracted text, keep it as part of the quote.
- Return results for ALL items in the list (do not omit any).
- Do not add explanations.

SELF-CHECK BEFORE FINAL OUTPUT (SILENT):
For each clause marked present, re-scan the contract text for:
- additional conditions, exceptions, or related enforcement/termination language
- notice periods, numeric thresholds, cancellation terms
- cross-references ("subject to Section…", "except as…", "provided that…")
If found, add those spans to the "answer" array.

CLAIM TYPE DEFINITIONS:
{chr(10).join(CLAUSE_DEFINITIONS)}

================================================================================
OUTPUT FORMAT (JSON ONLY)
================================================================================
Return ONLY a valid JSON array with the same number of items as the definitions list, in the same order.

Each item must be:
{{
  "clause_name": "<exact string from the list above>",
  "is_impossible": true | false,
  "answer": ["<extracted span 1>", "<extracted span 2>", ...]
}}

RULES:
- If "is_impossible" is true: "answer" MUST be an empty array [].
- If "is_impossible" is false: "answer" MUST contain one or more spans.
- Do not output any other keys.
- Do not output markdown or commentary.
- IMPORTANT: Complete ALL 41 clause types. Do not stop early.


BEGIN ANALYSIS NOW.
"""


        # Check token count before sending
        # print('\n📊 Checking token count before API call...')
        # token_count = count_tokens(extraction_prompt, model=model, verbose=True)
        
        # Retry logic for extraction
        ai_results = None
        extraction_error = None
        
        for attempt in range(max_extraction_retries):
            try:
                if attempt > 0:
                    print(f'\n🔄 Retry attempt {attempt + 1}/{max_extraction_retries} for extraction...')
                    # Add a small delay before retry
                    time.sleep(5)
                
                extraction_response = call_llm(extraction_prompt, model=model, temperature=0.0)
                
                # Parse JSON response
                ai_results = parse_json_response(extraction_response, context=f"extraction for {contract_title}")
                
                # Validate that we got all expected clauses
                if len(ai_results) < len(CLAUSE_DEFINITIONS):
                    print(f'   ⚠️  Warning: Expected {len(CLAUSE_DEFINITIONS)} clauses but got {len(ai_results)}')
                    if attempt < max_extraction_retries - 1:
                        print(f'   🔄 Retrying to get complete response...')
                        continue
                    else:
                        print(f'   ⚠️  Proceeding with {len(ai_results)} clauses (incomplete)')
                
                # Success!
                break
                
            except Exception as e:
                extraction_error = e
                if attempt < max_extraction_retries - 1:
                    print(f'   ⚠️  Extraction attempt {attempt + 1} failed: {str(e)[:100]}')
                else:
                    # Final attempt failed
                    raise extraction_error
        
        # Check if we got results
        if ai_results is None:
            raise ValueError(f"Failed to extract clauses after {max_extraction_retries} attempts")
        
        ai_df = pd.DataFrame(ai_results).add_suffix('_ai')
        
        print(f'   ✅ Extracted {len(ai_df)} clauses')
        print(f'   ✅ Clauses found: {(~ai_df["is_impossible_ai"]).sum()}')
        print(f'   ❌ Clauses not found: {ai_df["is_impossible_ai"].sum()}')
        
        # Merge with ground truth
        merged_df = ai_df.merge(
            contract_df,
            left_on='clause_name_ai',
            right_on='clause_name',
            how='inner'
        )
        
        # Add run_id column to merged_df
        merged_df['run_id'] = run_id
        merged_df['extraction_model'] = get_model_name(model)
        
        print(f'✅ Extraction complete for this contract! (Run {run_id})')
        
        return {
            'contract_title': contract_title,
            'run_id': run_id,
            'merged_df': merged_df,
            'extraction_model': get_model_name(model),
            'error': None
        }
        
    except Exception as e:
        print(f'\n❌ Error extracting {contract_title} (Run {run_id}):')
        print(f'   {str(e)}')
        print(traceback.format_exc())
        
        return {
            'contract_title': contract_title,
            'run_id': run_id,
            'merged_df': None,
            'error': str(e)
        }

print('✅ Contract extraction function defined')
