from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

tokenizer = AutoTokenizer.from_pretrained("microsoft/GODEL-v1_1-base-seq2seq")
model = AutoModelForSeq2SeqLM.from_pretrained("microsoft/GODEL-v1_1-base-seq2seq")
model.config.output_logits=True
print(model)

def generate(instruction, knowledge, dialog):
    if knowledge != '':
        knowledge = '[KNOWLEDGE] ' + knowledge
    dialog = ' EOS '.join(dialog)
    #dialog = ' '
    query = f"{instruction} [CONTEXT] {dialog} {knowledge}"
    query = ''
    input_ids = tokenizer(f"{query}", return_tensors="pt").input_ids
    print(query)
    print(input_ids)
    #print(tokenizer.convert_ids_to_tokens([784, 1, 0]))
    outputs = model.generate(input_ids, max_length=12, min_length=8, top_p=0.9,
                             do_sample=True, 
                             output_scores=True,
                            return_dict_in_generate=True)

    scores = outputs['scores']
    scores = torch.log_softmax(torch.stack(scores).squeeze(1), -1)
    surprisals = -(scores/torch.log(torch.tensor(2.0)))
    print(surprisals)
    #scores[scores == float("-inf")] = 0
    sequence = outputs['sequences']
    seq = sequence.flatten().reshape(-1, 1)
    #print(torch.max(scores, dim=-1))
    #print(torch.sum(scores, dim=-1))
    print()
    print(seq)
    seq = seq[1:,:]
    print(scores.shape, seq.shape)
    print('seq', seq)
    by_word_surprisal = torch.gather(surprisals, 1, seq).flatten()
    print(by_word_surprisal)
    avg_surp = torch.sum(by_word_surprisal)/by_word_surprisal.size(0)
    print(avg_surp)
    print(torch.exp2(avg_surp))
    output = tokenizer.decode(sequence[0], skip_special_tokens=True)
    return output

# Instruction for a chitchat task
#instruction = f'Instruction: given a dialog context, you need to response empathically.'
instruction = ''
# Leave the knowldge empty
knowledge = ''
dialog = [
    'Does money buy happiness?']
#    'It is a question. Money buys you a lot of things, but not enough to buy happiness.',
#    'What is the best way to buy happiness ?'
#]
response = generate(instruction, knowledge, dialog)
print(response)
#print(model)
