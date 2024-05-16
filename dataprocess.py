import re
from collections import OrderedDict
from sys import argv
def ppl(file_path):

    
 
    pattern = re.compile(r"\{'wikitext2': (\d+\.\d+), 'ptb': (\d+\.\d+)\}")


    with open(file_path, 'r') as file:
        for line in file:
            matches = pattern.finditer(line)
            for match in matches:
                number1, number2 = match.groups()
                print(f"{number1} {number2}")
      



def extract_block(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    results = []
    last_task = None

    for idx, line in enumerate(lines):
        if line.startswith("hf-causal-experimental"):

            current_block = lines[idx+3:idx+15]
            results.append(current_block)

    # Join all lines and return
    return results

def complete(lines):
    completed_lines = []
    for line in lines:
        if line.strip().startswith('|'):
            parts = line.split('|')
            if parts[1].strip() == '':
                # Fill in missing items from the last full line
                parts[1] = last_full_line.split('|')[1]
            completed_lines.append('|'.join(parts))
            last_full_line = '|'.join(parts)
        else:
            completed_lines.append(line)
        
    return completed_lines

def extract(data):

    lines = data.split('\n')
    res = []
    # Extract and print the fourth column from each line
    for line in lines:
        columns = line.split('|')
        if len(columns) > 4:
            res.append(columns[4].strip())
    [print(float(item)*100) for item in res]

def filter_and_sort_template(template, rules):
   
    rules_dict = {pair.split(':')[0]: pair.split(':')[1] for pair in rules.strip('{}').split('; ')}
    
  
    template_lines = template.strip().split('\n')

   
    filtered_lines = []
    for line in template_lines:
        parts = line.split('|')
        dataset, metric = parts[1].strip(), parts[3].strip()
        if dataset in rules_dict and rules_dict[dataset] == metric:
            filtered_lines.append(line)


    order = ["boolq", "piqa", "hellaswag", "winogrande", "arc_easy", "arc_challenge", "openbookqa"]
    sorted_lines = sorted(filtered_lines, key=lambda x: order.index(x.split('|')[1].strip()))

   
    return '\n'.join(sorted_lines)


# 需要保留的行规则
rules = "{winogrande:acc; arc_easy:acc; boolq:acc; piqa:acc_norm; openbookqa:acc_norm; hellaswag:acc_norm; arc_challenge:acc_norm}"



path = argv[1]
ppl(path)
output = extract_block(path)
for i in range(len(output)):
    output[i] = "".join(complete(output[i]))
    output[i] = filter_and_sort_template(output[i],rules)
    extract(output[i])
    print("***********")
   
# Output the result to a new file or print it


