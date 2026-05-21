import torch
import tqdm

from gpt.test import GPT
from gpt.tokenizer import tokenizer_from_file
import constant
from decompose.molConn import gen2mol
import rdkit.Chem as Chem
# from gpt.ESM import protein2vector, load_model

def gpt_test(path, n=0, method='top_k', save='smiles_generation_test.txt', batch=1, p_type='lora_1', conditional=False, prop = None, protein=None,pocket=None, model_name=None, arg=0.9):
    device = constant.device
    tokenizer = tokenizer_from_file(file_path=path + '/frag_tokenizer.json')
    torch.set_grad_enabled(False)
    model = GPT(vocab_size=tokenizer.get_vocab_size(), prop_len=constant.prop_len, p_type=p_type, conditional=conditional).to(device)
    if model_name is not None:
        model.load_state_dict(torch.load(path + '/' + model_name), strict=False)
    else:
        model.load_state_dict(torch.load(path + '/GPT.pt'), strict=False)
    model.eval()
    # 初始输入是空，每次加上后面的对话信息
    if n == 0:
        while True:
            sentence = ''
            temp_sentence = input("prompt:")

            sentence = constant.START_TOKEN + ' ' + temp_sentence
            answer = model.answer(sentence=sentence, tokenizer=tokenizer, method=method, protein=protein, prop=prop, pocket=pocket)
            print("molecule:", answer + '\n' + gen2mol([answer])[0][1])

    else:
        answers = []
        counter = 0

        from gpt.rACSF import cal_rACSF
        protein_rep = cal_rACSF(protein)
        protein = protein_rep[0]

        while True:
            sentence = constant.START_TOKEN
            try:
                answer = model.answer(sentence=sentence, tokenizer=tokenizer, method=method, protein=protein, batch=batch, prop=prop, model='raw', pocket=pocket, arg=arg)
                batch_size = len(answer)
                if batch_size + counter < n:
                    answers.extend(answer)
                    counter += batch_size
                    print(counter)
                else:
                    answers.extend(answer[batch_size + counter - n :])
                    break
            except Exception as e:
                print(e)

        result = gen2mol([x.replace('<start>', '').replace('</s>', '') for x in answers])
        smiles = [ i[1] for i in result]
        data = []
        for x in smiles:
            if '.' in x:
                pass
            data.append(x)

        if save is not None:
            with open(save, 'w') as f:
                f.write('\n'.join(data))


if __name__ == '__main__':
    gpt_test(n=1050, path='', method='temperature',
             save='gen/rACSF', batch=50, prop=None, conditional=['protein', 'protein_pocket'], protein='', arg=0.9, pocket=[])

    ## gpt_PL_test_auto(10, 'gpt/protein_test.txt')
    # fp_mol_test()
