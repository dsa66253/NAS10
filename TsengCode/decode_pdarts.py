import os
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
# this file just use to plot figure that shows alphas' variation during training

def tensor_to_list(input_op):
    return input_op.numpy().tolist()
def pickSecondMax(input):
    # set the max to zero to and find the max again 
    # to get the index of second large value of original input
    alphas = np.copy(input)
    alphasMaxIndex = np.argmax(alphas, -1)

    for i in range(len(alphas)):
        alphas[i, alphasMaxIndex[i]] = 0
    alphasSecondMaxIndex = np.argmax(alphas, -1)

    return alphasSecondMaxIndex
    

if __name__ == '__main__':
    # 實驗三種不同的初始權重與照片批次順序
    for i in range(3):
        name_number = str(i)
        print(f'i:{i}')
        num_of_cell = 5  # cell個數
        num_of_epoch = 45  # epoch總數
        num_of_op = 5  # 每個cell中的操作個數

        for cell in range(num_of_cell):
            label = []
            op1s = []
            op2s = []
            op3s = []
            op4s = []
            op5s = []

            for num in range(3, num_of_epoch + 1, 2):
                all_alpha_list = np.load(
                    './alpha_pdart_nodrop/alpha_prob_' + str(name_number) + '_' + str(num) + '.npy')

                op_list = []
                # import pdb

                # pdb.set_trace()
                for i in range(num_of_op):
                    op_list.append(all_alpha_list[cell][i])
                # print(op_list)
                # print(all_alpha_list)
                label.append(str(num))

                op1s.append(float('%.3f' % op_list[0]))
                op2s.append(float('%.3f' % op_list[1]))
                op3s.append(float('%.3f' % op_list[2]))
                op4s.append(float('%.3f' % op_list[3]))
                op5s.append(float('%.3f' % op_list[4]))

                # 存最後一個epoch的alpha機率值
                if (num == 45) & (cell == 0):
                    #* todo max_alphas = np.argmax(all_alpha_list, -1)
                    max_alphas = pickSecondMax(all_alpha_list)
                    # print("all_alpha_list", all_alpha_list)
                    # print("max_alphas", max_alphas)
                    gene_cell = np.array([np.array([i, j]) for i, j in enumerate(max_alphas)])
                    genotype_filename = os.path.join('./weights_pdarts_nodrop/',
                                                    'genotype_' + str(name_number))
                    # print("genotype_filename", genotype_filename)
                    np.save(genotype_filename, gene_cell)
                    # print('architecture search results:', network_path)
                    print('new cell structure:\n', gene_cell)

            x = np.arange(len(label))

            # 劃出alpha機率值的機率分布圖的訓練過程，以長條圖呈現
            width = 0.1
            fig, ax = plt.subplots(figsize=(10, 8))
            rects1 = ax.bar(x - 2 * width, op1s, width, label='3x3conv')
            rects2 = ax.bar(x - 1 * width, op2s, width, label='5x5conv')
            rects3 = ax.bar(x, op3s, width, label='7x7conv')
            rects4 = ax.bar(x + 1 * width, op4s, width, label='9x9conv')
            rects5 = ax.bar(x + 2 * width, op5s, width, label='11x11conv')

            plt.p(0, 1, 8)
            plt.legend(loc=2)
            ax.set_ylabel('probability')
            ax.set_xlabel('epoch')
            ax.set_title('cell' + str(cell))
            ax.set_xticks(x)
            ax.set_xticklabels(label)
            ax.legend()

            def autolabel(rects):
                for rect in rects:
                    height = rect.get_height()
                    # print(height)

            autolabel(rects1)
            autolabel(rects2)
            autolabel(rects3)
            autolabel(rects4)
            autolabel(rects5)
            fig.tight_layout()
            plt.savefig('./weights_pdarts_nodrop/' + str(name_number) + '_cell_' + str(cell) + '.png')
