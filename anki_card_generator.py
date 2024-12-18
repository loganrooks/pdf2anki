import json
import re
from openai import OpenAI
from pdfminer.high_level import extract_text
import pymupdf4llm
import pdfminer
from itertools import product
import os
import math
import argparse


client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def get_pdf_title(pdf_path):
    # Extract text from the PDF
    text = extract_text(pdf_path, page_numbers=[0])
    # Split the text into paragraphs
    title = text.split('\n\n')[0].replace('\n', '')
    return title

def extract_text_from_pdf(pdf_path, line_overlap=0.6, line_margin=0.6, char_margin=3.0, ignore_patterns=[r'^\d+$', r'^\x0c']):
    params = pdfminer.layout.LAParams(line_overlap=line_overlap, line_margin=line_margin, char_margin=char_margin)
    text = extract_text(pdf_path, laparams=params)

    # Split the text into paragraphs
    paragraphs = text.split('\n\n')
    title = get_pdf_title(pdf_path)
    # Clean the paragraphs
    cleaned_paragraphs = clean_paragraphs(paragraphs, ignore_patterns=ignore_patterns)
    cleaned_text = '\n\n'.join(cleaned_paragraphs)
    return title, cleaned_text

def is_complete_sentence(text):
    # Define regex patterns for various sentence-ending formats
    sentence_end_patterns = [
        r'[.!?][\"\')\]]*$',  # Ends with punctuation followed by optional quotes, parentheses, or brackets
        r'[.!?][\"\')\]]*\s*\(\d+\)$',  # Ends with punctuation followed by optional quotes, parentheses, or brackets and a citation
        r'[.!?][\"\')\]]*\s*\[\d+\]$',  # Ends with punctuation followed by optional quotes, parentheses, or brackets and a footnote
        r'[.!?][\"\')\]]*$',  # Ends with punctuation followed by optional quotes, parentheses, or brackets and space
        r':$',  # Ends with colon (lead up to a block quotation)
    ]

    # Check if the text matches any of the sentence-ending patterns
    for pattern in sentence_end_patterns:
        if re.search(pattern, text.strip()):
            return True

    return False

def split_paragraph_with_punctuation(paragraph):
    # Use re.findall to capture sentences along with punctuation and spaces
    sentences = re.findall(r'.*?.!?\]]*|\s*\(\d+\)|\s*\[\d+\])?(?=\s+|$)', paragraph.strip())
    return sentences

def check_final_sentence(paragraph):
    # Split the paragraph by traditional sentence-ending punctuation
    # sentences = re.split(r'(?<=[.!?]) +', paragraph)
    sentences = split_paragraph_with_punctuation(paragraph)
 
    # If the last sentence is not complete, merge it with the previous one
    # if len(sentences) > 1 and not is_complete_sentence(sentences[-1]):
    #     final_sentence = sentences[-2] + sentences[-1]
    # else:
    #     final_sentence = sentences[-1]
    
    return sentences[-1], is_complete_sentence(sentences[-1])


def find_unintended_breaks_indices(paragraphs, header_re=r'^\x0c', page_number_re=r'^\d+$'):
    unintended_break_indices = []
    header_indices = []
    page_number_indices = []

    first_half, second_half = None

    for i, paragraph in enumerate(paragraphs):
        if re.search(header_re, paragraph):
            header_indices.append(i)
        elif len(re.findall(page_number_re, paragraph)) == 1:
            page_number_indices.append(i)

        elif check_final_sentence(paragraph)[1]:
            if first_half:
                second_half = i
                unintended_break_indices.append((first_half, second_half))
                first_half, second_half = None

            else:
                first_half = i

    return header_indices, page_number_indices, unintended_break_indices


def get_ignore_indices(paragraphs, ignore_patterns: list[str]):
    ignore_indices = []
    for i, paragraph in enumerate(paragraphs):
        for pattern in ignore_patterns:
            if re.search(pattern, paragraph):
                ignore_indices.append(i)
                break
    return ignore_indices

def is_ignored(paragraph, ignore_patterns: list[str]):
    for pattern in ignore_patterns:
        if re.search(pattern, paragraph):
            return True
    return False
           
def remove_line_breaks(lines: list[str]):
    merged_lines = ""
    for line in lines:
        merged_lines += line[:line.rfind("-")] if line.strip()[-1] == "-" else line
    return merged_lines


def clean_paragraphs(paragraphs: list[str], ignore_patterns=[r'^\[\d+\]$', r'^\x0c']):
    cleaned_paragraphs = []
    previous_paragraph_incomplete = False
    for paragraph in paragraphs:
        if not is_ignored(paragraph, ignore_patterns):
            lines = paragraph.splitlines()
            merged_paragraph = remove_line_breaks(lines)

            if previous_paragraph_incomplete:
                cleaned_paragraphs[-1] += (merged_paragraph)
            else:
                cleaned_paragraphs.append(merged_paragraph)
            
            previous_paragraph_incomplete = not is_complete_sentence(lines[-1])

    return cleaned_paragraphs

# Function to add brackets to the regex if they forgot to include them so the page number can be included
def ensure_brackets(s):
    if not s.startswith('('):
        s = '(' + s
    if not s.endswith(')'):
        s = s + ')'
    return s

def merge_adjacent_elements(lst=list, n=1):
    if n < 1:
        raise ValueError("n must be at least 1")
    return [''.join(lst[i:i + n]) for i in range(0, len(lst), n)]

# Function to split text based on regex or default to 8 paragraphs, ignores everything up to the first page number.
def split_text(text, regex=None):
    if regex:
        return merge_adjacent_elements(re.split(ensure_brackets(regex), text)[1:], n=2)
    else:
        paragraphs = text.split('\n\n')
        return ['\n'.join(paragraphs[i:i+8]) for i in range(0, len(paragraphs), 8)]

# CREATING ANKI CARDS

def create_anki_cards(text_chunk, system_prompt, temperature=0.7, max_completion_tokens=2000, top_p=0.5):
    response = client.chat.completions.create(model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"You are about to be given a section of the text, you will convert these into Anki cards in accordance with the guidelines stipulated in your system prompt. Emphasis here is on trying to have as many cloze deletions per card as you can, leave very few words undeleted. How you are to go about doing this is described in your system message. The text:\n{text_chunk}"}
    ],
    max_completion_tokens=max_completion_tokens,
    temperature=temperature,
    top_p=top_p)
    return response.choices[0].message.content.strip()

# FORMATING AS JSON
def format_as_json(output):
    try:
        # Find the position of the first '['
        start_index = output.find('[')
        
        # Initialize the counter for brackets
        counter = 0
        outer_scope_start = -1
        
        # Find the outermost open '{' bracket using the counter
        for i in range(len(output) - 1, -1, -1):
            if output[i] == '}':
                counter -= 1
            elif output[i] == '{':
                counter += 1
                if counter == 1:
                    outer_scope_start = i
                    counter = 0 
        print(f"Outer Scope Start: {outer_scope_start}\n")
        # If an outer scope '{' bracket is found, find the first closing '}' bracket that precedes it
        if outer_scope_start != -1:
            end_index = output.rfind('}', 0, outer_scope_start) + 1
            cleaned_output = output[start_index:end_index] + ']'

            remaining_start = output.find('{', end_index)
            remaining_content = output[remaining_start:-1].strip('```').strip()
            print(f"Remaining JSON: {remaining_content}")

        else:
            end_index = len(output)
            cleaned_output = output[start_index:end_index].strip('```')
            remaining_content = None
        
        print(f"Start Index: {start_index}\n")
        print(f"End Index: {end_index}\n")
    
        # Load the JSON data
        json_output = json.loads(cleaned_output)
        return (json_output, remaining_content), None
    except json.JSONDecodeError as e:
        print(f"Error: {str(e)}\nCleaned Beginning:{cleaned_output[:100]}\nCleaned End: {cleaned_output[-101:]}\n")
        return None, str(e)
    
def find_remaining_text(input_chunk, remaining_content):
    # Extract the content of the "Text" field from the remaining_content string
    text_start = remaining_content.find('"Text": "') + len('"Text": "')
    text_end = remaining_content.find('",', text_start)
    remaining_text = remaining_content[text_start:text_end]
    
    # Consider only the text up until the first "{"
    search_text = remaining_text.split("{")[0]

    search_text = search_text[:40] if len(search_text) > 40 else search_text
    
    # Find the position of the search_text in the input_chunk
    start_index = input_chunk.find(search_text)
    
    # If the search_text is found, return the section from start_index to the end of input_chunk
    if start_index != -1:
        return input_chunk[start_index:].strip()
    else:
        return None
    

# WRITING TO FILE
def write_json_to_file(output_json_path: str, output, args: argparse.ArgumentParser):
    with open(output_json_path, 'a+', encoding='utf8') as output_file:
            if args.test:
                output = [{"args": vars(args), "output": output.copy()}]
            if not args.overwrite:
                try: 
                    output_file.seek(0)
                    existing_content = json.loads(output_file.read())
                    output.extend(existing_content)          
                except:
                    print("File corrupted, empty or is not a list. Forcing overwrite.")
                finally:
                    output_file.seek(0)
                    output_file.truncate()
            
            json.dump(output, output_file, indent=4)





def main():
    parser = argparse.ArgumentParser(description='Generate cloze deletion anki cards from an epub file.')
    # I/O
    parser.add_argument('pdf_file', type=str, help='Path to the pdf file')
    parser.add_argument('-o', '--out', type=str, required=True, help='Output JSON file path')
    parser.add_argument('-w', '--overwrite', action='store_true', help='Overwrite what is in the output destination')

    # FOR EXTRACTING TEXT INFORMATION FROM PDF USING PDFMINER.SIX
    parser.add_argument("--ignore_patterns", type=str, nargs='*', help="Regular expression(s) to identify headers and page numbers", default=[r'^\[\d+\]$', r'^\x0c'])
    parser.add_argument("--line_overlap", type=float, help="If two characters have more overlap than this they are considered to be on the same line. The overlap is specified relative to the minimum height of both characters.", default=0.6)
    parser.add_argument("--line_margin", type=float, help="If two lines are are close together they are considered to be part of the same paragraph. The margin is specified relative to the height of a line.", default=0.6)
    parser.add_argument("--char_margin", type=float, help="If two characters are closer together than this margin they are considered part of the same line. The margin is specified relative to the width of the character.", default=3.0)

    # FOR TEXT BATCHING AS INPUT TO THE LLM 
    parser.add_argument('-r', '--regex', type=str, help='Regular expression to split the text', default=None)
    parser.add_argument('--pages_per_chunk', type=int, help='Number of pages per text chunk to be inputed to the model', default=1)
    parser.add_argument('--page_range', type=int, nargs=2, help='Number of total pages to be converted', default=[1,0])

    # FOR OPENAI GPT 4o MINI ANKI CARD GENERATION
    parser.add_argument('--prompt_file', type=str, help='Path to a text file containing system prompt instructions', default=None)
    parser.add_argument('--prompt_text', type=str, help='Prompt instructions as a string', default=None)
    parser.add_argument('-t', '--temperature', type=float, nargs='*', help='What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.', default=[0.7])
    parser.add_argument('--max_completion_tokens', type=int, nargs='*', help='An upper bound for the number of tokens that can be generated for a completion, including visible output tokens and reasoning tokens.', default=[3000])
    parser.add_argument('--top_p', type=float, nargs='*', help='An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass. So 0.1 means only the tokens comprising the top 10% probability mass are considered.', default=[0.5])

    # MISCELLANEOUS
    parser.add_argument('--test', action='store_true', help='Generate Anki cards for the first chunk only and append "_test" to the output filename')
    parser.add_argument('--use_example', action='store_true', help='Use example ANKI cards to test file writing')
    

    args = parser.parse_args()

    # Determine output paths

    if not args.use_example:

        # Extract text from epub file
        params = {"line_overlap": args.line_overlap, "line_margin": args.line_margin, "char_margin": args.char_margin, "ignore_patterns": args.ignore_patterns}
        title, text = extract_text_from_pdf(args.pdf_file, **params)

        # Split text based on regex or default to 8 paragraphs
        text_chunks = merge_adjacent_elements(split_text(text, args.regex), n=args.pages_per_chunk)

        # Load prompt instructions from file or use provided prompt text
        if args.prompt_file:
            with open(args.prompt_file, 'r') as file:
                system_prompt = file.read()
        elif args.prompt_text:
            system_prompt = args.prompt_text
        else:
            system_prompt = (
                f"You are a philosophy professor creating Anki flash cards from a given text for self-study purposes. "
                "You will be given a chunk of text from one of Martin Heidegger's books, to make Anki cloze deletion cards. "
                "Create as many flash cards as needed following these rules:\n"
                "- Do not create duplicates.\n"
                "- Provide only the JSON for the flash cards; any other text will be ignored.\n"
                "- Format the cards with cloze deletion for the front.\n"
                "- Include the text citation with page number under the field 'Citation'.\n"
                "- Do not invent anything; use only the given text.\n"
                "- Do not just remove single words for cloze deletion; include phrases or clauses as well.\n"
                "- Emphasis: Cloze delete roughly one-third of the input, ensuring all German, Greek, or Latin phrases/terms are cloze deleted. Aim for at least 30 cloze deletions per card, ideally 40, with high density. Cloze delete even regular words if you are unable to meet the quota.\n"
                "- The 'Text' field should contain at least a paragraph (4-5 sentences) with one-third cloze deletions per card but max 3 clozes (c1, c2, c3). Multiple 'c1's, 'c2's, and possibly 'c3's should be thematically related.\n"
                f"- Include the title '{title}' along with the page number in the citation.\n"
                "- Write in English (unless there are German, Latin, or Greek terms).\n"
                "- Ensure each 'Text' field has at least 4 sentences. Do not create so many cards that some have less than 4 sentences. It's okay for some cards to have up to 8-10 sentences.\n"
                "- Cloze delete significant nouns, verbs, words, and phrases (and every single Greek, German, and Latin term cloze deleted and tagged with 'c3'). Split other cloze deletions roughly 50/50 per card between 'c1' and 'c2', grouping them thematically.\n"
                "- Ensure each card is self-complete with enough context to recognize and understand its meaning vaguely on its own. If part of the passage includes a quote from another philosopher, provide enough context prior to the quote.\n"
                "- Ensure all instances and semantically similar instances of a deleted term are clozed. These cards should not be easy, I want no left over hints."
                "- Cloze delete all semantically similar instances of any phrase or term you cloze delete. Cloze delete all of Heidegger's terminology and all of his definitions. Most subjects and predicates of sentences must be cloze deleted."
                "- All hyphenated words must be cloze deleted (e.g. Being-in-the-world, within-the-world, Being-in, reference-relations, existential-ontological, ontico-existentiell etc.)."
                "- Example format for 'Text' field: \"The full {{c1::essence of truth}}, including its most proper {{c1::nonessence}}, keeps {{c2::Dasein}} in need by this perpetual {{c1::turning to and fro}}. {{c2::Dasein}} is a {{c1::turning into need}}. From the {{c2::Da-sein}} of human beings and from it alone arises the disclosure of {{c1::necessity}} and, as a result, the {{c1::possibility of being transposed}} into what is {{c2::inevitable}}. The disclosure of beings as such is {{c1::simultaneously}} and {{c1::intrinsically}} the {{c2::concealing of beings}}.\""
            )

        if args.regex:
            system_prompt += f"For citation purposes, you can locate the page numbers using the regex '{args.regex}'. Make sure to keep track of when you cross page numbers and cite accordingly. Every text chunk starts with a page number and there should be a total of {str(args.pages_per_chunk)} pages per text chunk."

        output_json_path = args.out

        if args.test:
            # output_json_path = os.path.splitext(output_json_path)[0] + "_test_t{}_tp{}_mct{}.json".format(str(args.temperature).translate(str.maketrans('', '',string.punctuation)), str(args.top_p).translate(str.maketrans('', '',string.punctuation)), args.max_completion_tokens)
            output_json_path = os.path.splitext(output_json_path)[0] + "_test.json"
            input_json_path = os.path.splitext(output_json_path)[0] + "_input.txt"

        error_log_path = os.path.splitext(output_json_path)[0] + "_errors.txt"
        remaining_text_path =  os.path.splitext(output_json_path)[0] + "_remaining.txt"

        all_outputs = []
        all_error_logs = []
        all_remaining_text = []


        # Limit chunks if test flag is set
        text_chunks = text_chunks[math.ceil(args.page_range[0]/args.pages_per_chunk) - 1: math.ceil(args.page_range[1]/args.pages_per_chunk) - 1]

        if args.test:
            text_chunks = text_chunks[:1] if len(text_chunks) > 1 else text_chunks
            print("Writing inputs to file...")
            write_json_to_file(output_json_path=input_json_path, output=text_chunks, args=args)

        
        for (temperature, max_completion_tokens, top_p) in product(args.temperature, args.max_completion_tokens, args.top_p):
            # For a given set of parameters, create anki cards for each chunk and handle errors
            variables = {"temperature": temperature, "max_completion_tokens": max_completion_tokens, "top_p": top_p}

            all_anki_cards = []
            error_log = []

            error_count = 0
            consecutive_errors = 0

            for i, chunk in enumerate(text_chunks):
                anki_cards_output = create_anki_cards(chunk, system_prompt, temperature=temperature, max_completion_tokens=max_completion_tokens, top_p=top_p)
                anki_cards_json, error = format_as_json(anki_cards_output)


                if anki_cards_json:
                    anki_cards_json_cleaned, anki_cards_json_remaining = anki_cards_json
                    all_anki_cards.extend(anki_cards_json_cleaned)
                    consecutive_errors = 0 # reset the counter because we didn't encounter an error in generating cards nor formatting as json

                    # check if there is any content remaining from an output that was incomplete i.e. with a card that didn't fully generate
                    if anki_cards_json_remaining:
                        # print(f"Output: {i}\nRemaining JSON: {anki_cards_json_remaining}\n")
                        remaining_text = find_remaining_text(chunk, anki_cards_json_remaining)

                        # TODO: if it cuts off at the field name and there is no text to go off of to search for where it cut off in the input, use the last couple of words from the last good card to search
                        if remaining_text:
                            print(f"Remaining Text: {remaining_text}\n")
                            if i < len(text_chunks) - 1:
                                text_chunks[i+1] = remaining_text + text_chunks[i+1]
                            else:
                                all_remaining_text.append({"remaining_text": remaining_text, "variables": variables})
                        else:
                            message = f"Output #{i} was incomplete but there was no remaining text."
                            print(message + "\nRemaining JSON: {anki_cards_json_remaining}\n")
                            error_log.append({"error": message , "chunk": chunk, "output": anki_cards_output, "remaining_json": anki_cards_json_remaining})
                
                else:
                    error_count += 1
                    consecutive_errors += 1
                    error_log.append({"error": error , "chunk": chunk, "output": anki_cards_output, "terminal": False})

                if consecutive_errors >= 3 or error_count >= len(text_chunks) * 0.2:
                    print("Too many errors encountered. Stopping execution.")
                    error_log[-1]["Terminal"] = True
                    break

            # append the outputs and error logs for this set of variables to the total list
            all_outputs.append({"anki_cards": all_anki_cards, "variables": variables})
            if error_log:
                all_error_logs.append({"error_log": error_log, "variables": variables})

        # write the outputs and the error logs to file    
        print("Writing outputs to file...")
        write_json_to_file(output_json_path=output_json_path, output=all_outputs, args=args)

        if all_remaining_text:
            print("Writing remaining text to file...")
            write_json_to_file(output_json_path=remaining_text_path, output=all_remaining_text, args=args)

        if all_error_logs:
            print("Writing error logs to file...")
            write_json_to_file(output_json_path=error_log_path, output=all_error_logs, args=args)


    else:
        output_json_path = os.path.dirname(__file__) + "/examples/example_output.json"
        example_json_path = os.path.dirname(__file__) + "/examples/example01.json"
        with open(example_json_path, 'r') as example_json:
            all_anki_cards = json.load(example_json)
        write_json_to_file(output_json_path, all_anki_cards, args)

if __name__ == "__main__":
    main()