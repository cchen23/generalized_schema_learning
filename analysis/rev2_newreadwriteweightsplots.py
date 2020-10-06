import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

ANALYSIS_DATA_DIR = "/Users/cathy/Documents/Projekte/generalized_schema_learning/analysis_data/"
RESULTS_DIR = "/Users/cathy/Documents/Projekte/generalized_schema_learning/analysis_data/results"

TEMPLATE_READ = "read_weights_histories_NTM2_30000epochs_trial0_{test_filename}.npz"
TEMPLATE_WRITE = "write_weights_histories_NTM2_30000epochs_trial0_{test_filename}.npz"
FILLER_NAMES = ["DessertFillerTest", "DrinkFillerTest", "EmceeFillerTest", "FriendFillerTest", "PoetFillerTest", "PersonFillerTest"]
FILLER_NAMES_TOYLABELS = {"PersonFillerTest":"Subject", "EmceeFillerTest":"Emcee", "PoetFillerTest":"Poet", "FriendFillerTest":"Friend", "DrinkFillerTest":"Drink", "DessertFillerTest":"Dessert"}
#FILLER_LABELS_TOCOLORS = query_colors = {"Dessert":"#1f77b4", "Drink":"#ff7f0e", "Emcee":"#2ca02c", "Friend":"#d62728", "Poet":"#9467bd", "Subject":"#8c564b"}
FILLER_LABELS_TOCOLORS = {"Dessert":"#006BA4", "Drink":"#FF800E", "Emcee":"#ABABAB", "Friend":"#595959", "Poet":"#5F9ED1", "Subject":"#C85200"}
num_filler_names = len(FILLER_NAMES)
NUM_WORDS = 29

def get_max_weights(test_filenames):
    wordslist = ['EmceeFillerTest', 'QSubject', 'QDrink_bought', 'Order_dessert', 'BEGIN', 'Too_expensive', 'FriendFillerTrain', 'PersonFillerTest', 'Subject_performs', 'Say_goodbye', 'Emcee_intro', 'Sit_down', '?', 'Order_drink', 'PoetFillerTrain', 'END', 'zzz', 'Poet_performs', 'QFriend', 'DrinkFillerTest', 'PoetFillerTest', 'QDessert_bought', 'QPoet', 'QEmcee', 'FriendFillerTest', 'EmceeFillerTrain', 'DessertFillerTrain', 'DrinkFillerTrain', 'PersonFillerTrain', 'DessertFillerTest', 'Subject_declines'] # From /home/cc27/Thesis/generalized_schema_learning/data/variablefiller_gensymbolicstates_100000_1_testunseen_AllQs/Xy_english.txt

    filler_w_maxes = {fillername:[] for fillername in FILLER_NAMES}
    filler_r_maxes = dict()

    for test_filename in test_filenames:
        with open(os.path.join(ANALYSIS_DATA_DIR, "readwriteweights", "split_test_data_foranalysis", "{test_filename}.p".format(test_filename=test_filename)), "rb") as f:
            X, y = pickle.load(f)

        rw = np.load(os.path.join(ANALYSIS_DATA_DIR, "readwriteweights", TEMPLATE_READ.format(test_filename=test_filename)))["arr_0"]
        ww = np.load(os.path.join(ANALYSIS_DATA_DIR, "readwriteweights", TEMPLATE_WRITE.format(test_filename=test_filename)))["arr_0"]
        num_examples = rw.shape[0]

        for example_index in range(num_examples):
            for word_index in range(NUM_WORDS):
                word = wordslist[X[example_index, word_index, 0]]
                if word in FILLER_NAMES:
                    filler_w_maxes[word].append(np.argmax(ww[example_index,0,:,word_index]))

        filler_r_maxes[test_filename] = []
        for example_index in range(num_examples):
            word_index = -1
            filler_r_maxes[test_filename] += (list(np.argmax(rw[example_index,:,:,word_index], axis=1)))

    return filler_w_maxes, filler_r_maxes

test_filenames = ["test_QDessert_bought", "test_QDrink_bought", "test_QEmcee", "test_QFriend", "test_QPoet", "test_QSubject"]
filler_w_maxes, filler_r_maxes = get_max_weights(test_filenames)
FILLER_NAMES_TOYLABELS = {"PersonFillerTest":"Subject", "EmceeFillerTest":"Emcee", "PoetFillerTest":"Poet", "FriendFillerTest":"Friend", "DrinkFillerTest":"Drink", "DessertFillerTest":"Dessert"}

fig, axes = plt.subplots(num_filler_names, 1)
for i in range(num_filler_names):
    filler_name = FILLER_NAMES[i]
    print(filler_name)
    filler_label = FILLER_NAMES_TOYLABELS[filler_name]
    axes[i].hist(filler_w_maxes[filler_name], color=FILLER_LABELS_TOCOLORS[filler_label], range=(0,128), bins=128)
    axes[i].set_ylabel(filler_label)
    axes[i].tick_params('y', labelsize=8)
    axes[i].set_xlim([0,128])
    if i != num_filler_names - 1:
        axes[i].set_xticks([])

axes[0].set_title("Histogram of Write Weights by Filler")
axes[num_filler_names - 1].set_xlabel("External Buffer Indices")
fig.align_ylabels()
plt.tight_layout()
plt.subplots_adjust(hspace=0.1)
plt.savefig(os.path.join(RESULTS_DIR, "writeweights_histograms"))
plt.close()

fig, axes = plt.subplots(num_filler_names, 1)
for i in range(num_filler_names):
    filler_label = test_filenames[i]
    print(filler_label)
    filler_name = filler_label.split("_")[1][1:]
    axes[i].hist(filler_r_maxes[filler_label], color=FILLER_LABELS_TOCOLORS[filler_name], range=(0,128), bins=128)
    axes[i].set_ylabel("Q" + filler_name)
    axes[i].tick_params('y', labelsize=8)
    axes[i].set_xlim([0,128])
    if i != num_filler_names - 1:
        axes[i].set_xticks([])

axes[0].set_title("Histogram of Read Weights by Query")
axes[num_filler_names - 1].set_xlabel("External Buffer Indices")
fig.align_ylabels()
plt.tight_layout()
plt.subplots_adjust(hspace=0.1)
plt.savefig(os.path.join(RESULTS_DIR, "readweights_histograms"))
plt.close()

def compute_proportions(list_A):
    MEMORY_SIZE = 128
    A_proportions = np.zeros(MEMORY_SIZE)
    for i in range(MEMORY_SIZE):
        A_proportions[i] = len(np.where(np.array(list_A) == i)[0]) / (1.0 * len(list_A))
    return A_proportions

overlap_correlations = np.zeros((num_filler_names, num_filler_names))
for i in range(num_filler_names):
    for j in range(num_filler_names):
        A_list = filler_w_maxes[FILLER_NAMES[i]]
        B_list = filler_r_maxes[test_filenames[j]]
        print(FILLER_NAMES[i], test_filenames[j])
        A_proportions = compute_proportion_overlap(A_list)
        B_proportions = compute_proportion_overlap(B_list)
        overlap_correlations[i,j] = np.corrcoef(A_proportions, B_proportions)[0,1]

plt.imshow(overlap_correlations, vmin=0, vmax=1)
locs, labels = plt.yticks()
plt.ylim([-0.5, 5.5])
plt.yticks(range(num_filler_names), [FILLER_NAMES_TOYLABELS[filler_name] for filler_name in FILLER_NAMES])
plt.xlim([-0.5, 5.5])
plt.xticks(range(num_filler_names), [test_filename.split("_")[1] for test_filename in test_filenames], rotation=90)
plt.ylabel("Role Read Weights", fontsize=10)
plt.xlabel("Query Write Weights", fontsize=10)
plt.colorbar()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "readwrite_correlations"), dpi=900)

# Make legend.
query_colors = {"Dessert":"#1f77b4", "Drink":"#ff7f0e", "Emcee":"#2ca02c", "Friend":"#d62728", "Poet":"#9467bd", "Subject":"#8c564b"}
for role, color in query_colors.items():
    plt.plot([0,0],[1,1], color=color, label=role)
legend = plt.legend(ncol=1, bbox_to_anchor=(1.1, -0.1))
plt.savefig(os.path.join(RESULTS_DIR, "decoding_legend_2col"), bbox_extra_artists=(legend,), bbox_inches="tight", dpi=900)

# Do binning stuff.
filler_w_maxes, filler_r_maxes = get_max_weights(test_filenames)
role_translation_dict = {
    "Dessert":["test_QDessert_bought", "DessertFillerTest"],
    "Drink":["test_QDrink_bought", "DrinkFillerTest"],
    "Emcee":["test_QEmcee", "EmceeFillerTest"],
    "Friend":["test_QFriend", "FriendFillerTest"],
    "Poet":["test_QPoet", "PoetFillerTest"],
    "Subject":["test_QSubject", "PersonFillerTest"]
}
num_keys = len(list(role_translation_dict.keys()))
MEMORY_SIZE = 128
read_proportions = np.empty((MEMORY_SIZE, num_keys))
write_proportions = np.empty((MEMORY_SIZE, num_keys))
for i, (key, values) in enumerate(role_translation_dict.items()):
    read_proportions[:,i] = compute_proportions(filler_r_maxes[values[0]])
    write_proportions[:,i] = compute_proportions(filler_w_maxes[values[1]])

memory_write_maxes = np.argmax(write_proportions, axis=1)
memory_read_maxes = np.argmax(read_proportions, axis=1)
order = np.argsort(memory_write_maxes)
plt.scatter(memory_write_maxes[order]+0.1*np.random.randn(MEMORY_SIZE), memory_read_maxes[order]+0.1*np.random.randn(MEMORY_SIZE), s=1)
