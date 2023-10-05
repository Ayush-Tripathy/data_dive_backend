import numpy as np
max_display_rows = 10


class DTable:
    def __init__(self, initial_table: dict | np.ndarray = None):
        if initial_table is None:
            self.table = np.array([])

        if isinstance(initial_table, dict):
            column_names = np.array([key for key in initial_table.keys()])
            column_data = [initial_table[key] for key in initial_table.keys()]
            self.table = np.array(column_data, dtype="<U255").T
            self.table = np.insert(self.table, 0, column_names, axis=0)
            # self.table = self.table[1:].astype(float)
            self.column_types = {column: None for column in column_names}

        elif isinstance(initial_table, np.ndarray):
            column_names = initial_table[0]
            column_data = initial_table[1:]
            self.table = np.array(column_data, dtype="<U255")
            # self.table = np.insert(self.table, 0, column_names, axis=0)
            self.table = np.vstack((column_names, self.table))
            self.column_types = {column: None for column in column_names}

        elif initial_table is None:
            column_names = []
            self.column_types = {column: None for column in column_names}

        else:
            raise ValueError("'initial_table' should be a instance of dict or np.ndarray")

        for header, values in zip(column_names, self.table[1:].T):
            unique_values = set(values)
            if len(unique_values) == 2:
                self.column_types[header] = "Boolean"
            elif all(value.replace('.', '', 1).isdigit() or (value[1:].replace('.', '', 1).isdigit() and value[0] == '-') for value in values):
                self.column_types[header] = "Number"
            elif len(set(values)) < len(values) / 2:
                self.column_types[header] = 'Category'
            else:
                self.column_types[header] = "String"

    def __repr__(self):
        if len(self.table) == 0:
            rows = "Empty Table"
        elif len(self.table) < max_display_rows:
            rows = "\n".join("\t".join(str(cell)[:10] for cell in row) for row in self.table)
        else:
            rows = "\n".join("\t".join(str(cell)[:10] for cell in row) for row in
                             np.concatenate((self.table[0:max_display_rows - 4], self.table[len(self.table) - 4:])))
        return f"{rows}"

    def info(self):
        return {
            "rows": len(self.table) - 1,
            "cols": len(self.get_columns())
        }

    def get_columns(self):
        return self.table[0]

    def get_column_types(self):
        return self.column_types

    def get_column(self, column=None):
        if column is None or column not in self.get_columns():
            return np.array([])
        idx = np.where(self.get_columns() == column)[0][0]

        values = self.table[:, idx][1:]
        return_table = np.vstack((np.arange(len(values)), values)).T
        header = np.array(["", column])
        return_table = np.vstack((header, return_table))
        return DTable(return_table)

    def select(self, row_label, col_label):
        col_index = np.where(self.get_columns() == col_label)
        row_index = np.where(self.table[:, col_index] == row_label)
        print(row_index, col_index)
        return self.table[row_index, col_index]

    def get(self, row: int, col: int) -> str | float:
        r = self.table[row, col]

        try:
            r = float(r)
        except ValueError:
            r = str(r)

        return r

    def where(self, column: str, operator: str, value: str | int | float):

        try:
            col_index = np.where(self.get_columns() == column)
            col_type = self.get_column_types()[column]
        except KeyError:
            raise ValueError(f"'{column}' column not found")

        table = None
        if operator == "==":
            if col_type == "Number":
                col = self.table[:, col_index][1:].astype("float")
                idxs = np.where(col == value)[0]
                idxs = np.array([idx+1 for idx in idxs])
            else:
                col = self.table[:, col_index]
                idxs = np.where(col == value)[0]

            if len(idxs) != 0:
                table = np.vstack((self.get_columns(), [self.table[idx, ] for idx in idxs]))

        elif operator == ">":

            if col_type != "Number":
                raise ArithmeticError(f"Invalid operand for column '{column}'")

            col = self.table[:, col_index][1:].astype("float")
            idxs = np.where(col > value)[0]
            idxs = np.array([idx + 1 for idx in idxs])

            if len(idxs) != 0:
                table = np.vstack((self.get_columns(), [self.table[idx, ] for idx in idxs]))

        return DTable(table)


def read_csv(filename: str) -> DTable:
    with open(filename, 'r') as file:
        headers = file.readline().strip().split(',')

        lines = [line.strip() for line in file]

        data_dict = {header: [] for header in headers}

        for line in lines:
            values = []
            inside_quote = False
            current_value = ''

            for char in line:
                if char == '"':
                    inside_quote = not inside_quote
                elif char == ',' and not inside_quote:
                    values.append(current_value)
                    current_value = ''
                else:
                    current_value += char

            values.append(current_value)

            for header, value in zip(headers, values):
                data_dict[header].append(value)
        return DTable(data_dict)


# -------------- TESTING --------------------
dset = {
    "col1": [4, 2, 3, 7],
    "col2": [5, 4, 3, 0],
    "col3": [6, 5.4, 5., 0],
    "col4": ["hi", "bye", "hello", "hi"],
    "col5": ["Y", "Y", "N", "Y"],
    "col6": ["cat", "cat", "cat", "cat"]
}


t = DTable(dset)
# print(t.get_columns())
# print(t.get_column_types())
# print(t.get_column("col6"))
# print(t.select("col3", "0"))

import time

start = time.time()

dt = read_csv("dsets/ign.csv")
# print(dt.get_column("release_day"))
# print(dt.get(1, 2))
# print(type(dt.get(1, 2)))
# print(dt)
print(dt.where("score", ">", 9.5))
end = time.time()

print(f"T1: {end - start}")

# import pandas as pd
#
# s2 = time.time()
# df = pd.read_csv("dsets/ign.csv")
# print(df["release_day"])
# e2 = time.time()
#
# print(f"T2: {e2 - s2}")
