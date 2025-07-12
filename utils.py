import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error, mean_absolute_error
from tabulate import tabulate
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import os
import matplotlib as mpl
import pickle

def creat_path(model_n):
    paretn_dir = os.path.abspath(os.getcwd())
    path = os.path.join(paretn_dir, model_n)
    if not os.path.exists(path):
        os.mkdir(path)
    return path

def get_key(c, n, m):
    L1 = list()
    L2 = list()
    for i in range(n):
        L1.append(c[i])
    for i in range(n, n+m):
        L2.append(c[i])
    return L1, L2

def scaler_fit(d):
    sclr = StandardScaler()
    sclr = sclr.fit(d)
    return sclr

def trnsfrm(data):
    temp = pd.DataFrame(columns=data.columns)
    for _, name in enumerate(data.columns):
        t = data[name].values.reshape(-1, 1)
        sclr = scaler_fit(t)
        temp[name] = sclr.transform(t).reshape(-1)
    return temp

def invrs_trnsfrm(y, data, lbl_clmns):

    y_t = np.reshape(
        y, [y.shape[0]*y.shape[1], y.shape[2]])
    temp = np.empty(y_t.shape)

    for i in range(len(lbl_clmns)):
        t = data[lbl_clmns[i]].values.reshape(-1, 1)
        sclr = scaler_fit(t)
        temp[:, i] = sclr.inverse_transform(
            y_t[:, i].reshape(-1, 1)).reshape(-1)

    temp = temp.reshape(y.shape)

    return temp

def evaluate_forecast(y_true, y_pred, m, titel, model_name, model):
    t = []
    r2a = 0
    for i in range(m):
        y_true_i = y_true[:, :, i].flatten()
        y_pred_i = y_pred[:, :, i].flatten()
        e1 = mean_squared_error(y_true_i, y_pred_i)
        e2 = mean_absolute_error(y_true_i, y_pred_i)
        e3 = mean_absolute_percentage_error(y_true_i, y_pred_i)
        e4 = r2_score(y_true_i, y_pred_i)
        r2a = r2a + e4
        t.append([titel[i], e1, e2, e3, e4])
    mse = mean_squared_error(y_true.flatten(), y_pred.flatten())
    mae = mean_absolute_error(y_true.flatten(), y_pred.flatten())
    mape = mean_absolute_percentage_error(y_true.flatten(), y_pred.flatten())
    t.append([titel[-1], mse, mae, mape, r2a/6])
    path = creat_path(model_name)
    head = ["Output", "mse", "mae", "mape", "r2"]
    with open(path+"/"+model_name+'.txt', 'w') as f:
        f.write("Model number : b\n")
        f.write("\nModel Summary")
        stringlist = []
        model.summary(print_fn=lambda x: stringlist.append(x))
        short_model_summary = "\n".join(stringlist)
        f.write(short_model_summary+"\n\n")
        f.write(tabulate(t, headers=head, tablefmt="grid"))

    return t

def plot_forecast(
        y_history, y_true, y_pred, window, labels_col, data, model_name):
    """plots input, label and prediction for the last example in
    train, validation or test sets.
    """
    y_pred = invrs_trnsfrm(y_pred, data, labels_col)
    for _, plot_col in enumerate(labels_col):
        plt.figure()
        ax = plt.gca()
        if plot_col not in (window.feature_columns and window.label_columns):
            raise ValueError(
                "The chosen plot column does not exist in input/label data!")

        if y_true.shape[2] == 1:
            label_col_index = 0
        elif y_true.shape[2] > 1:
            label_col_index = window.label_columns.index(plot_col)

        if y_history.shape[2] == 1:
            input_col_index = 0
        elif y_history.shape[2] > 1:
            input_col_index = window.feature_columns.index(plot_col)-18


        ax.plot(window.input_indices,
                y_history[1, :, input_col_index], "g-", label="History",linewidth=2,alpha=0.5)
        ax.plot(window.label_indices,
                y_true[1, :, label_col_index], "b", label="True",linewidth=2)
        ax.plot(window.label_indices,
                y_pred[1, :, label_col_index], "r", label="Predicted",linewidth=2)
        ax.legend(loc="best")
        ax.set_xlabel("time index")
        ax.set_ylabel(plot_col)
        path = creat_path(model_name)
        plt.savefig(path+"/"+model_name + "_" +
                    plot_col + "_forecast" + '.svg')
        plt.savefig(path+"/"+model_name + "_" +
                    plot_col + "_forecast" + '.eps')
        plt.show()
    return

def plot_rslt_metrics(t,lbl_clmns,model_n):
    mpl.rcParams['figure.figsize'] = (15, 10)
    mpl.rcParams['axes.grid'] = False


    fig, ax1 = plt.subplots()

    x = np.arange(len(lbl_clmns))
    width = 0.4

    color = "#1f77b4"
    ax1.set_ylabel("MSE over features and time-steps", color=color, fontsize=15)
    ax1.bar(x- width/2 , [t[i][1] for i in range(len(lbl_clmns))], width, color=color)
    ax1.bar_label(ax1.containers[0],fmt='%.2e', fontsize=12, padding=1)
    ax1.set_xticks(ticks=x)
    ax1.set_xticklabels(labels=lbl_clmns, fontsize=15)
    ax1.xaxis.set_tick_params(rotation=45)

    ax2 = ax1.twinx()

    color = "#ff7f0e"
    ax2.set_ylabel("MAE over features and time-steps", color=color, fontsize=15)

    ax2.bar(x + width/2, [t[i][2] for i in range(len(lbl_clmns))], width, color=color)
    ax2.bar_label(ax2.containers[0],fmt='%.2e', fontsize=12, padding=1)
    fig.tight_layout()
    path = creat_path(model_n)
    plt.savefig(path+"/"+model_n+"_err_metrics" +'.svg', dpi=300)
    plt.savefig(path+"/"+model_n+"_err_metrics" +'.eps', dpi=300)
    plt.show()

def plot_rslt_r2_metrics(t,lbl_clmns,model_n):
    mpl.rcParams['figure.figsize'] = (15, 10)
    mpl.rcParams['axes.grid'] = False
    
    
    fig, ax1 = plt.subplots()
    
    x = np.arange(len(lbl_clmns))
    width = 0.3
    
    
    ax1.set_ylabel("r² over features and time-steps", fontsize=15)
    ax1.bar(x, [t[i][-1] for i in range(len(lbl_clmns))], width)
    ax1.bar_label(ax1.containers[0],fmt='%.4f', fontsize=12)
    ax1.set_xticks(ticks=x)
    ax1.set_xticklabels(labels=lbl_clmns, fontsize=15)
    ax1.xaxis.set_tick_params(rotation=45)
    fig.tight_layout()
    path = creat_path(model_n)
    plt.savefig(path+"/"+model_n+"_r2_metric" +'.svg', dpi=300)
    plt.savefig(path+"/"+model_n+"_r2_metric" +'.eps', dpi=300)
    plt.show()

def plot_rslt_err_metrics(t, lbl_clmns, model_n):
    mpl.rcParams['figure.figsize'] = (15, 10)
    mpl.rcParams['axes.grid'] = False

    fig, ax1 = plt.subplots()

    x = np.arange(len(lbl_clmns))
    width = 0.4

    color = "#4cc776"
    ax1.set_ylabel("MSE over features and time-steps",
                   color=color, fontsize=15)
    ax1.bar(x - width/2, [t[i][1]
            for i in range(len(lbl_clmns))], width, color=color)
    ax1.bar_label(ax1.containers[0], fmt='%.2e', fontsize=12, padding=1)
    ax1.set_xticks(ticks=x)
    ax1.set_xticklabels(labels=lbl_clmns, fontsize=15)
    ax1.xaxis.set_tick_params(rotation=45)

    ax2 = ax1.twinx()

    color = "#3d997b"
    ax2.set_ylabel("MAE over features and time-steps",
                   color=color, fontsize=15)

    ax2.bar(x + width/2, [t[i][2]
            for i in range(len(lbl_clmns))], width, color=color)
    ax2.bar_label(ax2.containers[0], fmt='%.2e', fontsize=12, padding=1)
    fig.tight_layout()
    path = creat_path(model_n)
    plt.savefig(path+"/"+model_n+"_err_metrics" + '.svg', dpi=fig.dpi)
    plt.show()

def plot_rslts_radar(y_true, y_pred, lbl_clmns, model_n):
    lim = [[2600, 3700, 640, 1500, 140, 50],[3250, 4200, 760, 1750, 175, 350]]
    for i in range(6):

        plt.style.use('default')
        ytest = y_true[:, :, i]
        ypred = y_pred[:, :, i]
        ytest = np.reshape(ytest, [ytest.shape[0]*ytest.shape[1]])
        ypred = np.reshape(ypred, [ypred.shape[0]*ypred.shape[1]])
        subjects = np.linspace(
            1, len(ytest), num=len(ytest), dtype=int).tolist()

        true = ytest.tolist()
        pred = ypred.tolist()

        angles = np.linspace(0, 2*np.pi, len(subjects), endpoint=False)

        print(angles)

        angles = np.concatenate((angles, [angles[0]]))

        print(angles)

        subjects.append(subjects[0])

        true.append(true[0])

        pred.append(pred[0])

        fig = plt.figure(figsize=(10, 10))

        ax = fig.add_subplot(polar=True)

        ax.plot(angles, true, ls='-', color='g',
                label='Actual '+lbl_clmns[i], linewidth=8, alpha=0.5)

        ax.plot(angles, pred, ls='dotted',
                color='r', label='Predicted '+lbl_clmns[i], linewidth=2)

        ax.grid(True, ls='-.')

        plt.yticks(fontsize=18, weight="bold")

        ax.set_rlabel_position(90)
        ax.set_ylim(lim[0][i], lim[1][i])
        plt.xticks([])
        plt.tight_layout()
        #plt.legend(loc='upper right',fontsize=18)
        path = creat_path(model_n)
        plt.savefig(path+"/"+model_n + "_" + lbl_clmns[i] + "_radar" + '.svg')
        plt.savefig(path+"/"+model_n + "_" + lbl_clmns[i] + "_radar" + '.eps')
        plt.show()

def save_to_pickle(file_path, data):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

def load_from_pickle(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data
