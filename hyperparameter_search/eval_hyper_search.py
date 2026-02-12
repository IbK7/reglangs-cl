# eval_hyper_search.py
import csv
from pathlib import Path

import pandas as pd
# from defaultvalues import RESULT_PATH

def merge_experiment_files(path, model, problem):
    # merge all files together
    eval_file = path  / f"{model}_{problem}_evaluation.csv"
    if not Path(eval_file).exists():
        # create out_path folder of the stem
        path.parent.mkdir(parents=True, exist_ok=True)
        # get all files
        files_path = path / f"{model}" / f"{problem}"
        files = list(files_path.glob(f"{model}_{problem}*.csv"))
        if len(files) == 0:
            return
        # merge all files together
        df_full = None
        for file in files:
            df = pd.read_csv(file)
            if df_full is None:
                df_full = df
            else:
                df_full = pd.concat([df_full, df])
        # replace NaN values by None string
        df_full = df_full.fillna('No')
        # load csv file
        df = df_full
        # remove num_layers column
        df = df.drop(columns=['num_layers'])
        if model == 'lstm':
            df = df.drop(columns=['nonlinearity'])
        # save to csv
        df.to_csv(eval_file, index=False)


def evaluate_test_rnn(path, experiment_type, model):
    # create evaluation folder
    out_path = path / f'{experiment_type}'
    if not Path(out_path).exists():
        Path(out_path).mkdir(parents=True, exist_ok=True)
    else:
        return
    if not Path(path / f"{experiment_type}_evaluation.csv").exists():
        return
    # load csv file
    df = pd.read_csv(path / f"{experiment_type}_evaluation.csv")
    # get all columns
    columns = df.columns
    end_col_num = 9
    if model == 'lstm':
        end_col_num = 8

    hyperparameter_columns = columns[1:end_col_num]
    # get all unique column values + columns as keys
    unique_values = dict()
    for column in hyperparameter_columns:
        unique_values[column] = df[column].unique()
    
    lengths = df["length"].unique()

    headers = [ f'hyperparameter,hyperparameter_value,total_records,total_max_records,success_rate,max_accuracy'
    ,f'hyperparameter_1, hyperparameter_value_1, hyperparameter_2, hyperparameter_value_2,total_records,total_max_records,success_rate,max_accuracy'
    ,f'hyperparameter_1, hyperparameter_value_1, hyperparameter_2, hyperparameter_value_2, hyperparameter_3, hyperparameter_value_3,total_records,total_max_records,success_rate,max_accuracy'
    ,f'hyperparameter_1, hyperparameter_value_1, hyperparameter_2, hyperparameter_value_2, hyperparameter_3, hyperparameter_value_3, hyperparameter_4, hyperparameter_value_4,total_records,total_max_records,success_rate,max_accuracy'
    ,f'hyperparameter_1, hyperparameter_value_1, hyperparameter_2, hyperparameter_value_2, hyperparameter_3, hyperparameter_value_3, hyperparameter_4, hyperparameter_value_4, hyperparameter_5, hyperparameter_value_5,total_records,total_max_records,success_rate,max_accuracy'
    ,f'hyperparameter_1, hyperparameter_value_1, hyperparameter_2, hyperparameter_value_2, hyperparameter_3, hyperparameter_value_3, hyperparameter_4, hyperparameter_value_4, hyperparameter_5, hyperparameter_value_5, hyperparameter_6, hyperparameter_value_6,total_records,total_max_records,success_rate,max_accuracy']

    for i, header in enumerate(headers):
        with open(out_path / f"{experiment_type}_evaluation_{i+1}.csv", 'w') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(header.split(','))
        for length in lengths:
            with open(out_path / f"{experiment_type}_evaluation_{i+1}_length_{length}.csv", 'w') as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow(header.split(','))


    # filter df by unique values
    for key, value in unique_values.items():
        for v in value:
            filtered_df = df[df[key] == v]
            # total number of records
            total_records = len(filtered_df)
            if total_records > 0:
                # total number of 1.0 records
                total_max_accuracy = len(filtered_df[filtered_df["accuracy"] == 1.0])
                max_accuracy = filtered_df["accuracy"].max()
                success_rate = total_max_accuracy / total_records
                with open(out_path / f"{experiment_type}_evaluation_{1}.csv", 'a') as f:
                    csv_writer = csv.writer(f)
                    csv_writer.writerow([key, v, total_records, total_max_accuracy, success_rate, max_accuracy])

                for length in lengths:
                    filtered_df_length = filtered_df[filtered_df["length"] == length]
                    total_records = len(filtered_df_length)
                    if total_records > 0:
                        total_max_accuracy = len(filtered_df_length[filtered_df_length["accuracy"] == 1.0])
                        max_accuracy = filtered_df_length["accuracy"].max()
                        success_rate = total_max_accuracy / total_records
                        with open(out_path / f"{experiment_type}_evaluation_{1}_length_{length}.csv", 'a') as f:
                            csv_writer = csv.writer(f)
                            csv_writer.writerow([key, v, total_records, total_max_accuracy, success_rate, max_accuracy])

    # filter df by combinations of unique values (two)
    unique_combinations = set()
    tmp = [(key, value) for key, value in unique_values.items()]
    for i, (key_1, values) in enumerate(tmp):
        for v in values:
            for j, (key2, values2) in enumerate(tmp):
                if j > i:
                    for v2 in values2:
                        unique_combinations.add((key_1, v, key2, v2))
    for key1, v1, key2, v2 in unique_combinations:
        hyperparameter_dict = {key1: v1, key2: v2}
        sorted_dict = dict(sorted(hyperparameter_dict.items(), key=lambda item: item[0]))
        key1, key2 = sorted_dict.keys()
        v1, v2 = sorted_dict.values()
        filtered_df = df[(df[key1] == v1) & (df[key2] == v2)]
        # total number of records
        total_records = len(filtered_df)
        if total_records > 0:
            # total number of 1.0 records
            total_max_accuracy = len(filtered_df[filtered_df["accuracy"] == 1.0])
            max_accuracy = filtered_df["accuracy"].max()
            success_rate = total_max_accuracy / total_records
            with open(out_path / f"{experiment_type}_evaluation_{2}.csv", 'a') as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow([key1, v1, key2, v2, total_records, total_max_accuracy, success_rate, max_accuracy])

        for length in lengths:
            filtered_df_length = filtered_df[filtered_df["length"] == length]
            total_records = len(filtered_df_length)
            if total_records > 0:
                total_max_accuracy = len(filtered_df_length[filtered_df_length["accuracy"] == 1.0])
                max_accuracy = filtered_df_length["accuracy"].max()
                success_rate = total_max_accuracy / total_records
                with open(out_path / f"{experiment_type}_evaluation_{2}_length_{length}.csv", 'a') as f:
                    csv_writer = csv.writer(f)
                    csv_writer.writerow([key1, v1, key2, v2, total_records, total_max_accuracy, success_rate, max_accuracy])

    # filter df by combinations of unique values (three)
    unique_combinations = set()
    tmp = [(key, value) for key, value in unique_values.items()]
    for i, (key_1, values) in enumerate(tmp):
        for v in values:
            for j, (key2, values1) in enumerate(tmp):
                if j > i:
                    for v2 in values1:
                        for k, (key3, values2) in enumerate(tmp):
                            if k > j:
                                for v3 in values2:
                                    unique_combinations.add((key_1, v, key2, v2, key3, v3))
    for key1, v1, key2, v2, key3, v3 in unique_combinations:
        hyperparameter_dict = {key1: v1, key2: v2, key3: v3}
        sorted_dict = dict(sorted(hyperparameter_dict.items(), key=lambda item: item[0]))
        key1, key2, key3 = sorted_dict.keys()
        v1, v2, v3 = sorted_dict.values()

        filtered_df = df[(df[key1] == v1) & (df[key2] == v2) & (df[key3] == v3)]
        # total number of records
        total_records = len(filtered_df)
        if total_records > 0:
            # total number of 1.0 records
            total_max_accuracy = len(filtered_df[filtered_df["accuracy"] == 1.0])
            max_accuracy = filtered_df["accuracy"].max()
            success_rate = total_max_accuracy / total_records
            with open(out_path / f"{experiment_type}_evaluation_{3}.csv", 'a') as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow([key1, v1, key2, v2, key3, v3, total_records, total_max_accuracy, success_rate, max_accuracy])

        for length in lengths:
            filtered_df_length = filtered_df[filtered_df["length"] == length]
            total_records = len(filtered_df_length)
            if total_records > 0:
                total_max_accuracy = len(filtered_df_length[filtered_df_length["accuracy"] == 1.0])
                max_accuracy = filtered_df_length["accuracy"].max()
                success_rate = total_max_accuracy / total_records
                with open(out_path / f"{experiment_type}_evaluation_{3}_length_{length}.csv", 'a') as f:
                    csv_writer = csv.writer(f)
                    csv_writer.writerow([key1, v1, key2, v2, key3, v3, total_records, total_max_accuracy, success_rate, max_accuracy])


    # filter df by combinations of unique values (four)
    unique_combinations = set()
    tmp = [(key, value) for key, value in unique_values.items()]
    for i, (key_1, values) in enumerate(tmp):
        for v in values:
            for j, (key2, values1) in enumerate(tmp):
                if j > i:
                    for v2 in values1:
                        for k, (key3, values2) in enumerate(tmp):
                            if k > j:
                                for v3 in values2:
                                    for l, (key4, values3) in enumerate(tmp):
                                        if l > k:
                                            for v4 in values3:
                                                unique_combinations.add((key_1, v, key2, v2, key3, v3, key4, v4))
    for key1, v1, key2, v2, key3, v3, key4, v4 in unique_combinations:
        hyperparameter_dict = {key1: v1, key2: v2, key3: v3, key4: v4}
        sorted_dict = dict(sorted(hyperparameter_dict.items(), key=lambda item: item[0]))
        key1, key2, key3, key4 = sorted_dict.keys()
        v1, v2, v3, v4 = sorted_dict.values()

        filtered_df = df[(df[key1] == v1) & (df[key2] == v2) & (df[key3] == v3) & (df[key4] == v4)]
        # total number of records
        total_records = len(filtered_df)
        if total_records > 0:
            # total number of 1.0 records
            total_max_accuracy = len(filtered_df[filtered_df["accuracy"] == 1.0])
            max_accuracy = filtered_df["accuracy"].max()
            success_rate = total_max_accuracy / total_records
            with open(out_path / f"{experiment_type}_evaluation_{4}.csv", 'a') as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow([key1, v1, key2, v2, key3, v3, key4, v4, total_records, total_max_accuracy, success_rate, max_accuracy])

        for length in lengths:
            filtered_df_length = filtered_df[filtered_df["length"] == length]
            total_records = len(filtered_df_length)
            if total_records > 0:
                total_max_accuracy = len(filtered_df_length[filtered_df_length["accuracy"] == 1.0])
                max_accuracy = filtered_df_length["accuracy"].max()
                success_rate = total_max_accuracy / total_records
                with open(out_path / f"{experiment_type}_evaluation_{4}_length_{length}.csv", 'a') as f:
                    csv_writer = csv.writer(f)
                    csv_writer.writerow([key1, v1, key2, v2, key3, v3, key4, v4, total_records, total_max_accuracy, success_rate, max_accuracy])

    # filter df by combinations of unique values (five)
    unique_combinations = set()
    tmp = [(key, value) for key, value in unique_values.items()]
    for i, (key_1, values) in enumerate(tmp):
        for v in values:
            for j, (key2, values1) in enumerate(tmp):
                if j > i:
                    for v2 in values1:
                        for k, (key3, values2) in enumerate(tmp):
                            if k > j:
                                for v3 in values2:
                                    for l, (key4, values3) in enumerate(tmp):
                                        if l > k:
                                            for v4 in values3:
                                                for m, (key5, values4) in enumerate(tmp):
                                                    if m > l:
                                                        for v5 in values4:
                                                            unique_combinations.add((key_1, v, key2, v2, key3, v3, key4, v4, key5, v5))
    for key1, v1, key2, v2, key3, v3, key4, v4, key5, v5 in unique_combinations:
        hyperparameter_dict = {key1: v1, key2: v2, key3: v3, key4: v4, key5: v5}
        sorted_dict = dict(sorted(hyperparameter_dict.items(), key=lambda item: item[0]))
        key1, key2, key3, key4, key5 = sorted_dict.keys()
        v1, v2, v3, v4, v5 = sorted_dict.values()

        filtered_df = df[(df[key1] == v1) & (df[key2] == v2) & (df[key3] == v3) & (df[key4] == v4) & (df[key5] == v5)]
        # total number of records
        total_records = len(filtered_df)
        if total_records > 0:
            # total number of 1.0 records
            total_max_accuracy = len(filtered_df[filtered_df["accuracy"] == 1.0])
            max_accuracy = filtered_df["accuracy"].max()
            success_rate = total_max_accuracy / total_records
            with open(out_path / f"{experiment_type}_evaluation_{5}.csv", 'a') as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow([key1, v1, key2, v2, key3, v3, key4, v4, key5, v5, total_records, total_max_accuracy, success_rate, max_accuracy])

        for length in lengths:
            filtered_df_length = filtered_df[filtered_df["length"] == length]
            total_records = len(filtered_df_length)
            if total_records > 0:
                total_max_accuracy = len(filtered_df_length[filtered_df_length["accuracy"] == 1.0])
                max_accuracy = filtered_df_length["accuracy"].max()
                success_rate = total_max_accuracy / total_records
                with open(out_path / f"{experiment_type}_evaluation_{5}_length_{length}.csv", 'a') as f:
                    csv_writer = csv.writer(f)
                    csv_writer.writerow([key1, v1, key2, v2, key3, v3, key4, v4, key5, v5, total_records, total_max_accuracy, success_rate, max_accuracy])

    # filter df by combinations of unique values (six)
    unique_combinations = set()
    tmp = [(key, value) for key, value in unique_values.items()]
    for i, (key_1, values) in enumerate(tmp):
        for v in values:
            for j, (key2, values1) in enumerate(tmp):
                if j > i:
                    for v2 in values1:
                        for k, (key3, values2) in enumerate(tmp):
                            if k > j:
                                for v3 in values2:
                                    for l, (key4, values3) in enumerate(tmp):
                                        if l > k:
                                            for v4 in values3:
                                                for m, (key5, values4) in enumerate(tmp):
                                                    if m > l:
                                                        for v5 in values4:
                                                            for n, (key6, values5) in enumerate(tmp):
                                                                if n > m:
                                                                    for v6 in values5:
                                                                        unique_combinations.add((key_1, v, key2, v2, key3, v3, key4, v4, key5, v5, key6, v6))
    for key1, v1, key2, v2, key3, v3, key4, v4, key5, v5, key6, v6 in unique_combinations:
        hyperparameter_dict = {key1: v1, key2: v2, key3: v3, key4: v4, key5: v5, key6: v6}
        sorted_dict = dict(sorted(hyperparameter_dict.items(), key=lambda item: item[0]))
        key1, key2, key3, key4, key5, key6 = sorted_dict.keys()
        v1, v2, v3, v4, v5, v6 = sorted_dict.values()

        filtered_df = df[(df[key1] == v1) & (df[key2] == v2) & (df[key3] == v3) & (df[key4] == v4) & (df[key5] == v5) & (df[key6] == v6)]
        # total number of records
        total_records = len(filtered_df)
        if total_records > 0:
            # total number of 1.0 records
            total_max_accuracy = len(filtered_df[filtered_df["accuracy"] == 1.0])
            max_accuracy = filtered_df["accuracy"].max()
            success_rate = total_max_accuracy / total_records
            with open(out_path / f"{experiment_type}_evaluation_{6}.csv", 'a') as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow([key1, v1, key2, v2, key3, v3, key4, v4, key5, v5, key6, v6, total_records, total_max_accuracy, success_rate, max_accuracy])

        for length in lengths:
            filtered_df_length = filtered_df[filtered_df["length"] == length]
            total_records = len(filtered_df_length)
            if total_records > 0:
                total_max_accuracy = len(filtered_df_length[filtered_df_length["accuracy"] == 1.0])
                max_accuracy = filtered_df_length["accuracy"].max()
                success_rate = total_max_accuracy / total_records
                with open(out_path / f"{experiment_type}_evaluation_{6}_length_{length}.csv", 'a') as f:
                    csv_writer = csv.writer(f)
                    csv_writer.writerow([key1, v1, key2, v2, key3, v3, key4, v4, key5, v5, key6, v6, total_records, total_max_accuracy, success_rate, max_accuracy])


    pass


def merge_best_hyperparameters(path, problems, num_fixed=1, model='rnn'):
    # get the csv files and merge them
    df_full = None
    for i, problem in enumerate(problems):
        file_path = path / f'{model}_{problem}' / f'{model}_{problem}_evaluation_{num_fixed}.csv'
        if not Path(file_path).exists():
            return
        df = pd.read_csv(file_path)
        # add problem column
        df["problem"] = problem
        if df_full is None:
            df_full = df
        else:
            df_full = pd.concat([df_full, df])
    # create unique groups from the first ten columns
    # remove the problem column
    df_groups = df_full.drop(columns=['problem'])
    num_group_cols = num_fixed * 2 + 1
    # group by the first 11 columns and get the mean of the last three columns
    df_groups = df_groups.groupby(df_groups.columns.tolist()[:num_group_cols]).mean()
    # save to csv
    out_path = path / f'{model}_best_hyperparameters_{num_fixed}.csv'
    df_groups.to_csv(out_path)

    return







def main():
    problems = ['cycle_navigation_small']
    path = Path("/home/iailab73/khanm2/reglangs-cl/results/transformer_hyperparameter_search")

    for model in ['transformer']:
        for problem in problems:
            experiment_type = f'{model}_{problem}'
            merge_experiment_files(path, "",  problem)
            evaluate_test_rnn(path, experiment_type, model)
            # print(path)
        merge_best_hyperparameters(path, problems, num_fixed=1, model=model)
        merge_best_hyperparameters(path, problems, num_fixed=2, model=model)
        merge_best_hyperparameters(path, problems, num_fixed=3, model=model)
        merge_best_hyperparameters(path, problems, num_fixed=4, model=model)
        merge_best_hyperparameters(path, problems, num_fixed=5, model=model)
        merge_best_hyperparameters(path, problems, num_fixed=6, model=model)

if __name__ == "__main__":
    main()