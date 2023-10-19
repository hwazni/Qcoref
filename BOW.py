import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
from discopy.rigid import Spider
from discopro.grammar import tensor
from lambeq import NumpyModel, AtomicType, Dataset, QuantumTrainer, SPSAOptimizer , AtomicType, IQPAnsatz, spiders_reader

N = AtomicType.NOUN
S = AtomicType.SENTENCE
P = AtomicType.PREPOSITIONAL_PHRASE

ansatz = IQPAnsatz({N: 1, S: 1, P:1}, n_layers=1, n_single_qubit_params=3)


def anaphoraSent2dig(sentence1, sentence2):
    
    diagram1 = spiders_reader.sentence2diagram(sentence1)
    diagram2 = spiders_reader.sentence2diagram(sentence2)

    diagram = tensor(diagram1,diagram2)
    diagram = diagram >> Spider(2, 1, S)

    return diagram

def generate_diag_labels(df):

    circuits, labels, diagrams = [],[],[]
    selected_cols = ['referent' if i % 2 == 0 else 'wrong_referent' for i in range(len(df))]

    for i, row in tqdm(df.iterrows(), total=len(df)):

        ref = row[selected_cols[i]]
        label = [1.0, 0.0] if i % 2 == 0 else [0.0, 1.0]
        sent1, sent2, pro = row[['sentence1', 'sentence2', 'pronoun']]
        sent2 = sent2.replace(pro, 'The ' + ref)
        # print(sent1, sent2)

        try:
            diagram = anaphoraSent2dig(sent1.strip(), sent2.strip())
            diagrams.append(diagram)
            discopy_circuit = ansatz(diagram)
            circuits.append(discopy_circuit)
            labels.append(label)
        except Exception as e:
            # Print an error message if an exception occurs
            print("An error occurred:", e)

    return circuits, labels, diagrams


df_train = pd.read_csv('train.csv', index_col=0)
df_val = pd.read_csv('val.csv', index_col=0)
df_test = pd.read_csv('test.csv', index_col=0)

train_circuits, train_labels, train_diagrams = generate_diag_labels(df_train)
val_circuits, val_labels, val_diagrams = generate_diag_labels(df_val)
test_circuits, test_labels, test_diagrams = generate_diag_labels(df_test)

all_circuits = train_circuits + val_circuits + test_circuits
model = NumpyModel.from_diagrams(all_circuits, use_jit=True)

loss = lambda y_hat, y: -np.sum(y * np.log(y_hat)) / len(y)  # binary cross-entropy loss
acc = lambda y_hat, y: np.sum(np.round(y_hat) == np.array(y)) / len(y) / 2  # half due to double-counting
eval_metrics = {"acc": acc}

def main(EPOCHS,SEED,BATCH_SIZE):

    trainer = QuantumTrainer(
        model,
        loss_function=loss,
        epochs=EPOCHS,
        optimizer=SPSAOptimizer,
        optim_hyperparams={'a': 0.1, 'c': 0.06, 'A': 0.01 * EPOCHS},
        evaluate_functions=eval_metrics,
        evaluate_on_train=True,
        verbose='text',
        seed=SEED
    )

    train_dataset = Dataset(train_circuits, train_labels, batch_size=BATCH_SIZE)
    val_dataset = Dataset(val_circuits, val_labels, shuffle=False)

    now = datetime.datetime.now()
    t = now.strftime("%Y-%m-%d_%H_%M_%S")
    print(t)
    trainer.fit(train_dataset, val_dataset, evaluation_step=1, logging_step=10)
    test_acc = acc(model(test_circuits), test_labels)
    print('Test accuracy:', test_acc)
    
seed_arr = [0, 10, 50, 77, 100, 111, 150, 169, 200, 234, 250, 300, 350, 400, 450]
B_sizes = [2]
epochs_arr = [2000]

for SEED in seed_arr:
    for BATCH_SIZE in B_sizes:
        for EPOCHS in epochs_arr:
            print(EPOCHS, SEED, BATCH_SIZE)
            main(EPOCHS, SEED, BATCH_SIZE)