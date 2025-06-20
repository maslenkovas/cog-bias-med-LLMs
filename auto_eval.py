import re
from tqdm import tqdm
import time
import os

search_str = "pmc-llama-13b"

# Get all the files in the directory containing the search string
f_names = [f for f in os.listdir() if search_str in f]

# f_names = ["bias_output_meditron-70b.txt"]

for f_name in f_names:
    with open(f_name, "r", encoding='utf8', errors='ignore') as f:
        lines = f.read()

    # Create empty text file
    f = open("ae_" + f_name, "w")

    # Split the content by the delimiter '~~~~~~~~~~~~~~~~~~~'
    sections = lines.split('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

    # Now, 'sections' is a list where each element is a section from the file
    # Trim whitespace and newlines from each section
    clean_sections = [section.strip() for section in sections if section.strip()]

    from openai import OpenAI
    import json

    with open("api_config.json", "r") as jsonfile:
        api_config = json.load(jsonfile)

    api_key = api_config['OpenAI']["API_KEY"]
    client = OpenAI(api_key=api_key)

    for i, line in enumerate(tqdm(clean_sections)):
        line = '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n' + line

        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')
        prompt = (
            "Evaluate a chatbot's accuracy in comparing responses with correct answers. "
            "For each scenario, assess the relationship between a given CORRECT ANSWER and RESPONSE. "
            "Choose 'True' if the RESPONSE explicitly and clearly matches the CORRECT ANSWER. Choose 'False' if the RESPONSE clearly does not match or contradicts the CORRECT ANSWER. "
            "Choose 'NR' (No Response/Unsure) if the RESPONSE is ambiguous, only partially related, provides no clear answer, or is entirely unrelated.\n\n"
            "Criteria for evaluation:\n"
            "- True: The RESPONSE must directly and unambiguously state the same option as the CORRECT ANSWER.\n"
            "- False: The RESPONSE clearly identifies an incorrect option or directly contradicts the CORRECT ANSWER.\n"
            "- NR: Use this for responses that do not directly address the question, are vague, provide additional unrelated context, or are ambiguous in any way. This includes responses that are related to the topic but do not clearly affirm or deny the CORRECT ANSWER, as well as incomplete or tangential responses.\n\n"
            "Apply these criteria consistently, with a cautious approach. When in doubt, or if the RESPONSE lacks clarity, lean towards 'NR'. Consider the overall intent and context of the RESPONSE carefully.\n\n"
            "Review each scenario and provide your evaluation (True, False, NR) based on the above criteria:\n\n"
        )

        if 'few_shot' or 'one_shot' in f_name:
            mult_examples = True

        start_response = False
        is_example = False
        for subline in line.split("\n"):
            if "### Example" in subline:
                is_example = True
            
            if "### Instruction" in subline:
                is_example = False

            if "Options" in subline and not is_example:
                # drop ### Options: from subline
                # subline = subline.split(":")[1].strip()
                prompt += subline + "\n"

            if "CORRECT ANSWER" in subline:
                prompt += subline + "\n"
            
            if "RESPONSE" in subline:
                prompt += subline
                start_response = True

            if "IS_CORRECT" in subline:
                start_response = False
            
            if "RESPONSE" not in subline and start_response:
                prompt += subline + "\n"
        
        prompt += "\nYour evaluation for each scenario (True, False, NR):"

        completion = client.chat.completions.create(
                    model='gpt-3.5-turbo-0613',
                    max_tokens=2048,
                    messages=[{"role": "system", "content": prompt}])
        response = completion.choices[0].message.content

        print(prompt)
        print(response)
        print("\n\n")

        # Edit RESPONSE: with response
        response = re.sub(r"IS_CORRECT: .*", "IS_CORRECT: " + response + "\n", line)

        f.write(response + "\n")

        # Pause for 1 second
        # time.sleep(1)

    f.close()