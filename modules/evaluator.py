import matplotlib.pyplot as plt
import pandas as pd
import logging
from sklearn.calibration import calibration_curve

class Evaluator():
    def __init__(self, final_model, x_test, y_test):
        self.y_test = y_test
        self.x_test = x_test
        self.model = final_model

        pred_class_probs = self.model.predict_proba(self.x_test)
        true_index = list(self.model.classes_).index(1)
        self.true_preds = [pred[true_index] for pred in pred_class_probs]

    def create_plots(self):
        self.plot_calibration()

    def plot_calibration(self, outpath = 'output/test_calibration.png'):
        fig = plt.figure(1, figsize=(10, 10))
        ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
        ax2 = plt.subplot2grid((3, 1), (2, 0))
        fraction_of_positives, mean_predicted_value = \
                    calibration_curve(self.y_test, self.true_preds, n_bins=10)
        ax1.plot(mean_predicted_value, fraction_of_positives, "s-")
        ax2.hist(self.true_preds, range=(0, 1), bins=10,
                    histtype="step", lw=2)
        ax1.set_ylabel("Fraction of positives")
        ax1.set_ylim([-0.05, 1.05])
        ax1.legend(loc="lower right")
        ax1.set_title('Calibration plots  (reliability curve)')
        ax2.set_xlabel("Mean predicted value")
        ax2.set_ylabel("Count")
        ax2.legend(loc="upper center", ncol=2)
        plt.savefig(outpath)
        # TODO: split by class

    # def plot_roc(self, outpath = 'output/test_roc.png'):
        # by class

    # def sensitivity(self, alert_rate = 0.01):
        # by class

    def get_accuracy(self):
        logging.info(f"Test set score: {self.model.score(self.x_test, self.y_test)}")
        # TODO:
        logging.info(f"Test set score for true target class:")
        logging.info(f"Test set score for false target class:")
        # self.sensitivity()

    def get_feat_imp(self, top_n = 5, outpath = 'output/feature_importance.csv'):
        feat_imp = pd.Series(self.model.feature_importances_, index=self.x_test.columns.values)
        self.feat_imp = feat_imp.sort_values(ascending=False)
        logging.info(f"Top {top_n} features:")
        logging.info(self.feat_imp.head(top_n))
        self.feat_imp.to_csv(outpath)
