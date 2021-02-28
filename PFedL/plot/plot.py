import os
import numpy as np
import matplotlib.pyplot as plt
import re
plt.rc('font', family='Times New Roman')
plt.rcParams['axes.unicode_minus']=False  # display negative sign
plt.style.use('science')  # use science image setting

def load_data(filename, compiler_list, ):
    result = []
    with open(filename, 'r') as f:
        text_list = f.readlines()
    text = "\n".join(text_list)
    for compiler in compiler_list:
        tmp = np.asarray(re.findall(compiler, text), dtype=np.float32)
        result.append(tmp)
    return result

def main(filename_list, compiler_list, algorithm_list, ylabel_list):
    num_plots = len(compiler_list)

    figure, ax_list= plt.subplots(1, num_plots, figsize=(9,6))
    # plt.subplots_adjust(left=0.2, right=0.85, top=0.9, bottom=0.15, wspace=0.01, hspace=0.1)
    plot_data = []
    for filepath in filename_list:
        plot_data.append(load_data(filepath, compiler_list))
    for i, ax in enumerate(ax_list):
        for j, data in enumerate(plot_data):
            rounds = len(data[i])
            ax.plot(np.arange(1, rounds+1), data[i], label=algorithm_list[j])
        ax.set_xlim(0, rounds)
        ax.set_ylabel(ylabel_list[i], fontdict={'family' : 'Times New Roman', 'size': 12})
        ax.set_xlabel("Communication Rounds")
        ax.legend()
        if ylabel_list[i] == "accuracy":
            plt.axhline(y=0.89,linestyle=":", color="k")
            plt.axhline(y=0.95, linestyle=":", color="k")
            plt.annotate(s=r"$0.89$", xy=(170, 0.88), xytext=(175, 0.8), weight='bold',
                         arrowprops={"arrowstyle":"->", "connectionstyle":"arc3", "color":"black",
                                     })
            plt.annotate(s="$0.95$", xy=(20, 0.949), xytext=(10, 0.85), weight='bold',
                         arrowprops={"arrowstyle":"->", "connectionstyle":"arc3", "color":"black",
                                     })
    # plt.imsave("../result//result.png", figure)
    figure.savefig("../result//result.png")
    plt.show(figure)

if __name__ == "__main__":
    base_path = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.dirname(base_path)

    file_list = os.listdir("../result/")
    text_list = [file for file in file_list if file.endswith(".txt")]
    assert len(text_list) > 0
    file_name = [os.path.splitext(tmp)[0] for tmp in text_list]
    algorithm_list = []
    filelist = []
    for file in file_name:
        data_name, algorithm_name = file.split("_")
        if data_name == "mnist":
            fileabs = os.path.join("result", ".".join([file, "txt"]))
            filelist.append(os.path.join(base_path, fileabs))
            algorithm_list.append(algorithm_name)
        
            
    loss_compiler = re.compile(r'test loss is\: (.*?),')
    acc_compiler = re.compile(r'metrics result is \[(.*?)\]')
    # filelist = ["../result/mnist_fedavg.txt", "../result/mnist_cafl.txt", "../result/mnist_lg.txt", "../result/mnist_pfedl.txt",
    #             "../result/mnist_apfl.txt"]
    compiller_list = [loss_compiler, acc_compiler]
    # algorithm_list = ["fedavg", "cafl", "lg", "pfedl", "apfl"]
    ylabel_list = ["test loss", "accuracy"]
    
    main(filelist, compiller_list, algorithm_list, ylabel_list)

'''
    pfedl_loss = np.asarray(re.findall(loss_compiler, pfedl_text), dtype=np.float32)
    pfedl_acc = np.asarray(re.findall(acc_compiler, pfedl_text), dtype=np.float32)
    fedavg_loss = np.asarray(re.findall(loss_compiler, fedavg_text), dtype=np.float32)
    fedavg_acc = np.asarray(re.findall(acc_compiler, fedavg_text), dtype=np.float32)
    lg_loss = np.asarray(re.findall(loss_compiler, lg_text), dtype=np.float32)
    lg_acc = np.asarray(re.findall(acc_compiler, lg_text), dtype=np.float32)
    with plt.style.context(['science', 'no-latex']):
        plt.figure(figsize=(8, 6))
        fig, ax1 = plt.subplots(figsize=(6, 4.5), dpi=200)
        ax1.plot(np.arange(1, len(pfedl_loss)+1), pfedl_loss, label="ours")
        ax1.plot(np.arange(1, len(fedavg_loss)+1), fedavg_loss, label="baseline")
        ax1.plot(np.arange(1, len(lg_loss)+1), lg_loss, label="mixture")
        ax1.set(xlabel="Iteration")
        ax1.set(ylabel="test loss")
        # ax1.autoscale(tight=True)
        ax1.set_ylim(0, 2.4)
        ax1.set_xlim(1, 101)
        ax1.legend(title="Method", loc='center right')
        # ax1.spines['top'].set_visible(False)


        ax2 = ax1.twinx()
        ax2.plot(np.arange(1, len(pfedl_acc)+1), pfedl_acc, label="ours", ls='--')
        ax2.plot(np.arange(1, len(fedavg_acc)+1), fedavg_acc, label="baseline", ls='--')
        ax2.plot(np.arange(1, len(lg_acc)+1), lg_acc, label="mixture", ls='--')
        ax2.set(ylabel="test accuracy")
        # ax2.autoscale(tight=True)
        ax2.set_ylim(0.15, 1)
        fig.savefig(r'D:\fig.png',
                    width=4, height=3, dpi=800)
'''
