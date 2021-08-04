import matplotlib.pyplot as plt
import pandas as pd
import logging
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix
from tabulate import tabulate

class Evaluator():
    def __init__(self, final_model, x_test, y_test, outpath_prefix='output/test_'):
        self.y_test = y_test
        self.x_test = x_test
        self.model = final_model

        pred_class_probs = self.model.predict_proba(self.x_test)
        true_index = list(self.model.classes_).index(1)
        self.true_preds = [pred[true_index] for pred in pred_class_probs]

        self.outpath_prefix = outpath_prefix

    def create_plots(self):
        self.plot_calibration()
        # self.plot_roc()
        # self.plot_sensitivity()

    def plot_calibration(self):
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
        plt.savefig(self.outpath_prefix + "_calibration.png")
        # TODO: split by class

    # def plot_roc(self, outpath = 'output/test_roc.png'):
        # by class

    def sensitivity(self, alert_rate = 0.01):
        pass
        # TODO: & by class

    def plot_sensitivity(self):
        # TODO:
        pass

    def print_confusion_matrix(self, cm):
        cm_list = cm.tolist()
        cm_list[0].insert(0, 'Real True')
        cm_list[1].insert(0, 'Real False')
        print(tabulate(cm_list, headers=['Pred True','Pred False']))

    def get_accuracy(self):
        logging.info(f"Test set score: {self.model.score(self.x_test, self.y_test)}")
        cf_matrix = confusion_matrix(self.y_test, self.true_preds > 0.5)
        self.print_confusion_matrix(cf_matrix)
        self.sensitivity()

    def get_feat_imp(self, top_n = 5):
        # TODO: - because feature importance tends to inflate importance of high cardinality 
        # categorical variables and continuous, would try permutation importance with more 
        # computational power This shows how does random reshuffling of the data affect 
        # model performance?
        
        feat_imp = pd.Series(self.model.feature_importances_, index=self.x_test.columns.values)
        self.feat_imp = feat_imp.sort_values(ascending=False)
        logging.info(f"Top {top_n} features:")
        logging.info(self.feat_imp.head(top_n))
        self.feat_imp.to_csv(self.outpath_prefix + "_feature_importance.csv")
