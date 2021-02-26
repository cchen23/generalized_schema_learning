'''Read and write weights.
Figure 5.
'''
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import plotnine as p9
import sys


def turn_filler_name_into_y_label(filler_name):
    new_label = filler_name.split('FILLER')[0].lower()
    new_label = new_label[0].upper() + new_label[1:]
    return new_label


def test_filename_to_filler_name(test_filename):
    filler_name = test_filename.split("_")[1]
    filler_name = filler_name[0:2] + filler_name[2:].lower()
    return filler_name


trial_num = sys.argv[1]
base_dir = '/home/cc27/Thesis/generalized_schema_learning'
ANALYSIS_DATA_DIR = os.path.join(base_dir, 'results/variablefiller_AllQs/variable_filler/variable_filler/weights')
DATA_DIR = os.path.join(base_dir, 'data/variablefiller_AllQs/')
SAVE_DIR = os.path.join(base_dir, 'paper_plot_scripts', 'figures')
TEMPLATE_READ = "read_weight_histories_DNC_30000epochs_trial{trial_num}_{test_filename}.npz"
TEMPLATE_WRITE = "write_weight_histories_DNC_30000epochs_trial{trial_num}_{test_filename}.npz"
FILLER_NAMES = ["DESSERTFILLER", "DRINKFILLER", "EMCEEFILLER", "FRIENDFILLER", "POETFILLER", "SUBJECTFILLER"]
FILLER_NAMES_TOYLABELS = {k: turn_filler_name_into_y_label(k) for k in FILLER_NAMES}
queries = ['QDessert', 'QDrink', 'QEmcee', 'QFriend', 'QPoet', 'QSubject']
test_filenames = ['test_' + query.upper() for query in queries]
query_colors = {"Dessert": "#1f77b4", "Drink": "#ff7f0e", "Emcee": "#2ca02c", "Friend": "#d62728", "Poet": "#9467bd", "Subject": "#8c564b"}
num_filler_names = len(FILLER_NAMES)
NUM_WORDS = 27
MEMORY_SIZE = 128
wordslist = [u'QSUBJECT', u'EMCEEFILLER', u'intro', u'expensive', u'decline', u'end', u'sit', u'perform', u'DESSERT', u'DRINK', u'poetry', u'QPOET', u'SUBJECTFILLER', u'FRIEND', u'begin', 'zzz', u'QEMCEE', u'QDRINK', u'QDESSERT', u'DESSERTFILLER', u'EMCEE', u'DRINKFILLER', u'QFRIEND', u'POETFILLER', '?', u'goodbye', u'POET', u'FRIENDFILLER', u'order', u'SUBJECT']  # wordslist.p


def get_max_weights(test_filenames):
    filler_w_maxes = {fillername: [] for fillername in FILLER_NAMES}
    filler_r_maxes = dict()

    for test_filename in test_filenames:
        with open(os.path.join(DATA_DIR, "{test_filename}.p".format(test_filename=test_filename)), "rb") as f:
            X, y = pickle.load(f)

        rw = np.load(os.path.join(ANALYSIS_DATA_DIR, TEMPLATE_READ.format(trial_num=trial_num, test_filename=test_filename)))["arr_0"]
        ww = np.load(os.path.join(ANALYSIS_DATA_DIR, TEMPLATE_WRITE.format(trial_num=trial_num, test_filename=test_filename)))["arr_0"]
        num_examples = rw.shape[0]
        num_words = rw.shape[-1]
        for example_index in range(num_examples):
            for word_index in range(num_words):
                word = wordslist[X[example_index, word_index, 0]]
                if word in FILLER_NAMES:
                    filler_w_maxes[word].append(np.argmax(ww[example_index, 0, :, word_index]))

        test_filename_filler_name = test_filename_to_filler_name(test_filename)
        filler_r_maxes[test_filename_filler_name] = []
        for example_index in range(num_examples):
            word_index = -1
            filler_r_maxes[test_filename_filler_name] += list(np.argmax(rw[example_index, :, :, word_index], axis=1))
    return filler_w_maxes, filler_r_maxes


filler_w_maxes, filler_r_maxes = get_max_weights(test_filenames)
fig, axes = plt.subplots(num_filler_names, 1)
for i in range(num_filler_names):
    filler_name = FILLER_NAMES[i]
    print(filler_name)
    filler_label = FILLER_NAMES_TOYLABELS[filler_name]
    axes[i].hist(filler_w_maxes[filler_name], color=query_colors[filler_label], range=(0, MEMORY_SIZE), bins=MEMORY_SIZE)
    axes[i].set_ylabel(filler_label)
    axes[i].tick_params('y', labelsize=8)
    axes[i].set_xlim([0, MEMORY_SIZE])
    if i != num_filler_names - 1:
        axes[i].set_xticks([])

axes[0].set_title("Histogram of Write Weights by Filler")
axes[num_filler_names - 1].set_xlabel("External Buffer Indices")
fig.align_ylabels()
plt.tight_layout()
plt.subplots_adjust(hspace=0.1)
plt.savefig(os.path.join(SAVE_DIR, "writeweights_histograms_" + trial_num))
plt.close()

fig, axes = plt.subplots(num_filler_names, 1)
for i in range(num_filler_names):
    test_filename = test_filenames[i]
    filler_name = test_filename_to_filler_name(test_filename)
    axes[i].hist(filler_r_maxes[filler_name], color=query_colors[filler_name[1:]], range=(0, MEMORY_SIZE), bins=MEMORY_SIZE)
    axes[i].set_ylabel(filler_name)
    axes[i].tick_params('y', labelsize=8)
    axes[i].set_xlim([0, MEMORY_SIZE])
    if i != num_filler_names - 1:
        axes[i].set_xticks([])

axes[0].set_title("Histogram of Read Weights by Query")
axes[num_filler_names - 1].set_xlabel("External Buffer Indices")
fig.align_ylabels()
plt.tight_layout()
plt.subplots_adjust(hspace=0.1)
plt.savefig(os.path.join(SAVE_DIR, "readweights_histograms_" + trial_num))
plt.close()


def compute_proportions(list_A):
    A_proportions = np.zeros(MEMORY_SIZE)
    for i in range(MEMORY_SIZE):
        A_proportions[i] = len(np.where(np.array(list_A) == i)[0]) / (1.0 * len(list_A))
    return A_proportions


overlap_correlations = np.zeros((num_filler_names, num_filler_names))
read_words = []
write_words = []
correlations = []
for i in range(num_filler_names):
    for j in range(num_filler_names):
        A_list = filler_w_maxes[FILLER_NAMES[i]]
        B_list = filler_r_maxes[test_filename_to_filler_name(test_filenames[j])]
        print(FILLER_NAMES[i], test_filenames[j])
        A_proportions = compute_proportions(A_list)
        B_proportions = compute_proportions(B_list)
        read_words.append(test_filename_to_filler_name(test_filenames[j]))
        write_words.append(FILLER_NAMES_TOYLABELS[FILLER_NAMES[i]])
        correlations.append(np.corrcoef(A_proportions, B_proportions)[0, 1])

# TODO: Change this to ggplot.
data_df = pd.DataFrame({'read_words': read_words, 'write_words': write_words, 'correlations': correlations})
plot = p9.ggplot(data_df, p9.aes(x='read_words', y='write_words', fill='correlations')) +\
        p9.geom_tile() +\
        p9.labels.labs(x='Read', y='Write') +\
        p9.theme(axis_text_x=p9.element_text(rotation=90))
plot.save(os.path.join(SAVE_DIR, "readwrite_correlations_" + trial_num), dpi=900)

# Make legend.
for role, color in query_colors.items():
    plt.plot([0, 0], [1, 1], color=color, label=role)
legend = plt.legend(ncol=1, bbox_to_anchor=(1.1, -0.1))
plt.savefig(os.path.join(SAVE_DIR, "decoding_legend_2col"), bbox_extra_artists=(legend,), bbox_inches="tight", dpi=900)
