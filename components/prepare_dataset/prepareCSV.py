import csv

def prepare_csv(number, mode, landmark_list):
    if mode == 1 and (0 <= number <= 9):
        number = number + 10
        csv_path = 'mlModel/predict/dataset.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
    return
