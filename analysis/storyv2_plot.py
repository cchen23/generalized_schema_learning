import matplotlib.pyplot as plt
import numpy as np

NUM_PERSONS = 1000

def get_values(architecture_name):
    run1 = np.load("%s_results_1000epochs_trial0.p" % architecture_name)["test_accuracy"][-1]
    run2 = np.load("%s_results_1000epochs_trial1.p" % architecture_name)["test_accuracy"][-1]
    run3 = np.load("%s_results_1000epochs_trial2.p" % architecture_name)["test_accuracy"][-1]
    return np.mean([run1, run2, run3]), np.max([run1, run2, run3]), np.min([run1, run2, run3])

width = 0.35
bar_values = []
bar_maxs = []
bar_mins = []
architecture_names = ["RNN-LN", "LSTM-LN", "NTM2", "RNN-LN-FW"]
for architecture_name in architecture_names:
    mean_val, max_val, min_val = get_values(architecture_name)
    bar_values.append(mean_val)
    bar_maxs.append(max_val - mean_val)
    bar_mins.append(mean_val - min_val)

xaxis_label = ["RNN", "LSTM", "NTM", "Fastweights"]
ind = np.arange(len(xaxis_label))
plt.bar(ind, bar_values, width, yerr=(bar_mins, bar_maxs), color='#0082c8')
chance_rate = 1.0/4119
plt.axhline(y=chance_rate, label="Chance Rate", color='k', linestyle='--')
plt.xticks(ind + width/2, xaxis_label)
plt.ylabel(info_type, fontsize=AXES_FONTSIZE)
plt.ylim([-0.01, 1.01])
TITLE_FONTSIZE = 16
plt.title("%s" % ("Correlation Violation"), fontsize=TITLE_FONTSIZE)
plt.ylabel("Test Accuracy")
plt.legend(loc="upper left")
plt.savefig("correlation_violation_1000persons", bbox_inches="tight", dpi=900)

# For shuffled stuff.

def get_values(architecture_name):
    run1 = np.load("%s_shuffled_1000epochs_testresults_trial0.p" % architecture_name)["test_accuracy"][-1]
    run2 = np.load("%s_results_1000epochs_trial1.p" % architecture_name)["test_accuracy"][-1]
    run3 = np.load("%s_results_1000epochs_trial2.p" % architecture_name)["test_accuracy"][-1]
    return np.mean([run1, run2, run3]), np.max([run1, run2, run3]), np.min([run1, run2, run3])

width = 0.35
bar_values = []
bar_maxs = []
bar_mins = []
architecture_names = ["RNN-LN", "LSTM-LN", "NTM2", "RNN-LN-FW"]
for architecture_name in architecture_names:
    mean_val, max_val, min_val = get_values(architecture_name)
    bar_values.append(mean_val)
    bar_maxs.append(max_val - mean_val)
    bar_mins.append(mean_val - min_val)

xaxis_label = ["RNN", "LSTM", "NTM", "Fastweights"]
ind = np.arange(len(xaxis_label))
plt.bar(ind, bar_values, width, yerr=(bar_mins, bar_maxs), color='#0082c8')
chance_rate = 1.0/4119
plt.axhline(y=chance_rate, label="Chance Rate", color='k', linestyle='--')
plt.xticks(ind + width/2, xaxis_label)
plt.ylabel(info_type, fontsize=AXES_FONTSIZE)
plt.ylim([-0.01, 1.01])
TITLE_FONTSIZE = 16
plt.title("%s" % ("Correlation Violation"), fontsize=TITLE_FONTSIZE)
plt.xlabel("Test Accuracy")
plt.legend(loc="upper left")
plt.savefig("correlation_violation_1000persons", bbox_inches="tight")

# For training curves.
fig, axes = plt.subplots(2, 1, sharex='col')
fig.suptitle("Accuracy, %d possible persons\n Train each filler in 3 roles, test on 4th role." % NUM_PERSONS)
axes[1].plot(rnn["test_accuracy"], label="rnn", color="c")
axes[0].plot(rnn["train_accuracy"], label="rnn", color="c", alpha=0.5)
axes[1].plot(lstm["test_accuracy"], label="lstm", color="b")
axes[0].plot(lstm["train_accuracy"], label="lstm", color="b", alpha=0.5)
axes[1].plot(ntm["test_accuracy"], label="ntm", color="g")
axes[0].plot(ntm["train_accuracy"], label="ntm", color="g", alpha=0.5)
axes[1].plot(fw["test_accuracy"], label="fastweights", color="r")
axes[0].plot(fw["train_accuracy"], label="fastweights", color="r", alpha=0.5)
axes[1].legend(bbox_to_anchor=(1, 1), loc='upper left', ncol=1)
axes[1].set_xlabel("num train epochs")
axes[0].set_ylabel("train accuracy")
axes[1].set_ylabel("test accuracy")
plt.savefig("train3roles_test4throle_%dpersons" % NUM_PERSONS, bbox_inches="tight")
plt.close()

fig, axes = plt.subplots(6, 2, sharex='col')
fig.suptitle("Test accuracies\n Previously seen (in diff role) and previously unseen (in any role)")
for i, query in enumerate(lstm_split["accuracies"].keys()):
    r, = axes[i][0].plot(rnn_split["accuracies"][query], color="c", label="rnn")
    ru, = axes[i][1].plot(rnn_unseen["accuracies"][query], color="c", alpha=0.5, label="rnn unseen")
    l, = axes[i][0].plot(lstm_split["accuracies"][query], color="b", label="lstm")
    lu, = axes[i][1].plot(lstm_unseen["accuracies"][query], color="b", alpha=0.5, label="lstm unseen")
    n, = axes[i][0].plot(ntm_split["accuracies"][query], color="g", label="ntm")
    nu, = axes[i][1].plot(ntm_unseen["accuracies"][query], color="g", alpha=0.5, label="ntm unseen")
    f, = axes[i][0].plot(fw_split["accuracies"][query], color="r", label="fastweights")
    fu, = axes[i][1].plot(fw_unseen["accuracies"][query], color="r", alpha=0.5, label="fastweights unseen")
    axes[i][0].set_ylabel(query, fontsize=9)
    axes[i][1].set_ylabel(query, fontsize=9)
    axes[i][0].set_ylim([0,1])
    axes[i][1].set_ylim([0,1])

axes[5][0].set_xlabel("num train epochs")
axes[5][1].set_xlabel("num train epochs")
axes[5][0].legend((r, l, n, f), ("rnn", "lstm", "ntm", "fastweights"), bbox_to_anchor=(0, -1), loc='upper left', ncol=1)
axes[5][1].legend((ru, lu, nu, fu), ("rnn unseen", "lstm unseen", "ntm unseen", "fastweights unseen"), bbox_to_anchor=(0, -1), loc='upper left', ncol=1)
plt.savefig("train3roles_test4throle_%dpersons_split" % NUM_PERSONS, bbox_inches="tight")
plt.close()


NUM_PERSONS = 1000

ntm = np.load("NTM2_shuffled_1000epochs_testresults_trial0.p")
fw = np.load("RNN-LN-FW_shuffled_1000epochs_testresults_trial0.p")
rnn = np.load("RNN-LN_shuffled_1000epochs_testresults_trial0.p")
lstm = np.load("LSTM-LN_shuffled_1000epochs_testresults_trial0.p")

networks = [ntm, fw, rnn, lstm]
network_names = ["ntm", "fastweights", "rnn", "lstm"]
queries = ["QSubject", "QFriend", "QPoet", "QEmcee"]
tasks = ["test_%s_shuffled1_sameroles", "test_%s_shuffled1_differentroles", "test_%s_shuffled1_unseen"]
query_colors = {"QDessert":"#1f77b4", "QDrink":"#ff7f0e", "QEmcee":"#2ca02c", "QFriend":"#d62728", "QPoet":"#9467bd", "QSubject":"#8c564b"}
width = 0.3
split_width = width/len(queries)
xaxis_length = len(tasks)
fig, axes = plt.subplots(4, 1, sharex='col')
fig.suptitle("Shuffled stories.")
chance_rate = 1.0/4119
for network_idx, network in enumerate(networks):
    accuracies = network["accuracy"]
    for query_idx, query in enumerate(queries):
        bar_values = [accuracies[task % query] for task in tasks]
        axes[network_idx].bar(np.arange(xaxis_length) + split_width * query_idx, bar_values, split_width, color=query_colors[query],  label=query, alpha=0.8)
        axes[network_idx].set_ylabel(network_names[network_idx])
        axes[network_idx].set_ylim([0, 1])
        axes[network_idx].axhline(y=chance_rate, label="Chance Rate", color='k', linestyle='--')

task_names = ["same roles", "correlation violation", "unseen fillers"]
ind = np.arange(xaxis_length)
axes[0].legend(bbox_to_anchor=(1, 1), loc='upper left', ncol=2)
plt.xticks(ind + width/2, task_names, fontsize=7)
plt.savefig("%dpersons_shuffled1" % NUM_PERSONS, bbox_inches="tight")
