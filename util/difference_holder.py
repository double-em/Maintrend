class DifferenceHolder:

    def __init__(self, threshold, logger, loud=False):
        self.total_difference = 0
        self.max_difference = 0
        self.min_difference = 0

        self.real_total_difference = 0
        self.real_max_difference = 0
        self.real_min_difference = 0

        self.threshold = threshold
        self.over_theshold_count = 0

        self.total_mean_difference = 0
        self.real_total_mean_difference = 0

        self.calculations = 0

        self.logger = logger
        self.loud = loud

    def difference_calc(self, single_prediction, true_value, dataset):

        pred_difference = abs(single_prediction - true_value)
        real_single_prediction = round(single_prediction)
        real_pred_difference = abs(real_single_prediction - true_value)
        self.total_difference += pred_difference

        if pred_difference > self.max_difference:
            self.max_difference = pred_difference

        if pred_difference < self.min_difference:
            self.min_difference = pred_difference

        self.real_total_difference += real_pred_difference
        
        if real_pred_difference > self.real_max_difference:
            self.real_max_difference = real_pred_difference

        if real_pred_difference < self.real_min_difference:
            self.real_min_difference = real_pred_difference

        if pred_difference > self.threshold:
            self.over_theshold_count += 1

            if self.loud:
                print("\n{:=^50}".format(" OVER TRESHOLD! "))
                print("Predicted: %s(%s) \nActually: %s \nDifference: %s(%s)" % (
                    single_prediction,
                    real_single_prediction,
                    true_value,
                    pred_difference,
                    real_pred_difference
                
                ))

                str_width = 10
                columns = [
                    "amount_down_today",
                    "times_down_today",
                    "produced_today",
                    "maintenance_day"#,
                    #"day_of_week"
                ]

                print("\nDataset:")
                for c in columns:
                    print("{:{width}.8}".format(c, width=str_width), end="")
                print()

                for i in range(len(dataset)):
                    j = dataset[i]
                    for k in j:
                        print("{:{width}.5}".format(str(k), width=str_width), end="")
                    print()

        self.calculations += 1
        
        self.total_mean_difference = self.total_difference / self.calculations
        self.real_total_mean_difference = self.real_total_difference / self.calculations



    def PrintFinal(self):
        if self.loud:
            print("\n========== Finished {} predictions! ==========".format(self.calculations))
            print("{:22} {:.3f}".format("Total Loss:", self.total_difference))
            print("{:22} {:.3f}".format("Total Mean Loss:", self.total_mean_difference))
            print("{:22} {:.3f}".format("Maximum Loss:", self.max_difference))
            print("{:22} {:.3f}".format("Minimum Loss:", self.min_difference))
            print()
            print("{:22} {}".format("Real Total Loss:", self.real_total_difference))
            print("{:22} {:.3f}".format("Real Total Mean Loss:", self.real_total_mean_difference))
            print("{:22} {}".format("Real Maximum Loss:", self.real_max_difference))
            print("{:22} {}".format("Real Minimum Loss:", self.real_min_difference))
            print()
            print("{:22} {}".format("Threshold:", self.threshold))
            print("{:22} {}".format("Total over threshold:", self.over_theshold_count))
        
        self.logger.debug(f"Predictions: {self.calculations}, MAE: {self.total_mean_difference}, Min: {self.min_difference}, Max: {self.max_difference}, Over treshold: {self.over_theshold_count}")