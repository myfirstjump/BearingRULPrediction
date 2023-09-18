import matplotlib
# matplotlib.use('agg') # As a background for Linux running
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


class PlotDesign(object):

    def __init__(self):
        pass

    def plot(self, samples, size=[4, 4]):

        fig = plt.figure(figsize=(size[1], size[0]))
        gs = gridspec.GridSpec(size[0], size[1])
        gs.update(wspace=0, hspace=0.05)

        for i, sample in enumerate(samples):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')

            plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

        return fig

    def plot_saving(self, dest_path, filename, suffix):
        plt.savefig(fname=dest_path + filename + '.' + suffix)

    def learning_curve(self, obj):

        plt.plot(obj['train_loss'], label='train')
        plt.plot(obj['valid_loss'], label='valid')
        plt.legend()
        plt.show()

    def plot_RUL_prediction(self, pred_y, true_y, main_unit):

        plt.plot(true_y, label='True')
        plt.plot(pred_y, label='Prediction')
        plt.title("Engine Number # {} RUL Prediction".format(main_unit))
        plt.legend()
        plt.show()