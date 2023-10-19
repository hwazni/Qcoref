import spacy
import pandas as pd
from tqdm import tqdm
from discopy.rigid import Spider
from discopro.grammar import tensor
from discopro.anaphora import connect_anaphora_on_top
from lambeq import BobcatParser, NumpyModel, AtomicType, IQPAnsatz, remove_cups, Rewriter

# load model path
model_path = 'model_sllm_450.lt'
best_model = NumpyModel.from_checkpoint(model_path)

nlp = spacy.load("en_core_web_trf")

parser = BobcatParser()
rewriter = Rewriter(['auxiliary','connector','coordination','determiner','object_rel_pronoun',
                        'subject_rel_pronoun','postadverb','preadverb','prepositional_phrase'])

N = AtomicType.NOUN
S = AtomicType.SENTENCE
P = AtomicType.PREPOSITIONAL_PHRASE

ansatz = IQPAnsatz({N: 1, S: 1, P:1}, n_layers=1, n_single_qubit_params=3)

def generate_diagram(diagram, pro, ref):

    pro_box_idx = next(i for i, box in enumerate(diagram.boxes) if box.name.casefold() == pro.casefold())
    ref_box_idx = next(i for i, box in enumerate(diagram.boxes) if box.name.casefold() == ref.casefold())
    final_diagram = connect_anaphora_on_top(diagram, pro_box_idx, ref_box_idx)
    rewritten_diagram = rewriter(remove_cups(final_diagram)).normal_form()

    return rewritten_diagram

def anaphoraSent2dig(sentence1, sentence2, pro, ref):
    
    diagram1 = parser.sentence2diagram(sentence1)
    diagram2 = parser.sentence2diagram(sentence2)

    diagram = tensor(diagram1,diagram2)
    diagram = diagram >> Spider(2, 1, S)

    diag = generate_diagram(diagram, pro, ref)

    return diag


def noun_pronoun_clusters(s1, s2):
    
    nouns = []
    
    doc1 = nlp(s1)
    for token in doc1:
        if str(token.pos_) == 'NOUN':
            nouns.append(token.text)
            
    doc2 = nlp(s2)
    for token in doc2:
        if str(token.pos_) == 'PRON':
            pronoun = token.text
       
    return nouns, pronoun

# main 
df_test = pd.read_csv('test.csv', index_col=0)
results = []
for i in tqdm(range(df_test.shape[0])):
    
    s1 = df_test.iloc[i]['sentence1']
    s2 = df_test.iloc[i]['sentence2']
    nouns, pronoun = noun_pronoun_clusters(s1, s2)
    
    all_prob = []
    all_nouns = []
    for n in nouns:
        try:    
            probabilities = best_model([ansatz(anaphoraSent2dig(s1, s2, pronoun, n))])[0]
            all_prob.append(probabilities)
            all_nouns.append(n)
        except:
            pass
    
    try:
        p1 = all_prob[0]
        p2 = all_prob[1]

        pred_noun = ''
        if p1[0] > p2[0]:
            pred_noun =  all_nouns[0]
        else:
            pred_noun =  all_nouns[1]
        
        results.append(pred_noun)
    except:
        results.append('NA')
    
df_test['quantumcoref_scores'] = results
df_test.to_csv('scores_with_quantumcoref.csv')