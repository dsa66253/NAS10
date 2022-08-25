import os
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt


def tensor_to_list(input_op):
    return input_op.numpy().tolist()


if __name__ == '__main__':
    for i in range(3):
        name_number = str(i)
        num_of_cell = 10
        num_of_epoch = 45
        num_of_op = 7

        for cell in range(num_of_cell):
            label = []
            op1s = []
            op2s = []
            op3s = []
            op4s = []
            op5s = []
            op6s = []
            op7s = []

            for num in range(3, num_of_epoch + 1, 2):
                print('check')
                all_alpha_list = np.load(
                    './alpha_8cell/alpha_prob_' + str(name_number) + '_' + str(
                        num) + '.npy')

                op_list = []

                for i in range(num_of_op):
                    op_list.append(all_alpha_list[cell][i])

                label.append(str(num))
                op1s.append(float('%.3f' % op_list[0]))
                op2s.append(float('%.3f' % op_list[1]))
                op3s.append(float('%.3f' % op_list[2]))
                op4s.append(float('%.3f' % op_list[3]))
                op5s.append(float('%.3f' % op_list[4]))
                op6s.append(float('%.3f' % op_list[5]))
                op7s.append(float('%.3f' % op_list[6]))

                if (num == 45) & (cell == 0):
                    max_alphas = np.argmax(all_alpha_list, -1)
                    gene_cell = np.array([np.array([i, j]) for i, j in enumerate(max_alphas)])
                    genotype_filename = os.path.join('./weights_8cell/',
                                                     'genotype_' + str(name_number))
                    print(genotype_filename)
                    np.save(genotype_filename, gene_cell)
                    # print('architecture search results:', network_path)
                    print('new cell structure:\n', gene_cell)

            x = np.arange(len(label))

            width = 0.1
            fig, ax = plt.subplots(figsize=(10, 8))
            rects1 = ax.bar(x - 3 * width, op1s, width, label='3x3conv')
            rects2 = ax.bar(x - 2 * width, op2s, width, label='5x5conv')
            rects3 = ax.bar(x - 1 * width, op3s, width, label='7x7conv')
            rects4 = ax.bar(x, op4s, width, label='9x9conv')
            rects5 = ax.bar(x + 1 * width, op5s, width, label='11x11conv')
            rects6 = ax.bar(x + 2 * width, op6s, width, label='maxpooling')
            rects7 = ax.bar(x + 3 * width, op7s, width, label='avgpooling')

            plt.ylim(0, 1, 8)
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
                    print(height)


            autolabel(rects1)
            autolabel(rects2)
            autolabel(rects3)
            autolabel(rects4)
            autolabel(rects5)
            autolabel(rects6)
            autolabel(rects7)
            fig.tight_layout()
            plt.savefig(
                './weights_8cell/' + str(name_number) + '_cell_' + str(cell) + '.png')