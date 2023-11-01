import numpy as np
import matplotlib.pyplot as plt

max_display_rows = 10


class DTable:
    """
    Represents a Table, in form of a 2D numpy array
    """
    
    def __init__(self,
                 initial_table: dict | np.ndarray = None,
                 columns: list | np.ndarray = None,
                 dtype: str = "<U255"):

        self.dtype = dtype

        if initial_table is None:
            self.table = np.array([])

        if isinstance(initial_table, dict):
            # If passed initial_table data is an instance of dict class

            self.columns = np.array([key for key in initial_table.keys()])
            if "" in self.columns:
                self.columns = np.array([f"Unnamed{idx}" if col == "" else col for idx, col in enumerate(self.columns)])
            column_data = [initial_table[key] for key in initial_table.keys()]
            self.table = np.array(column_data, dtype=dtype).T
            self.table = np.insert(self.table, 0, self.columns, axis=0)
            self.column_types = {column: None for column in self.columns}

        elif isinstance(initial_table, np.ndarray):
            # If passed initial_table data is an instance of numpy.ndarray class

            self.columns = initial_table[0]
            if "" in self.columns:
                self.columns = np.array([f"Unnamed{idx}" if col == "" else col for idx, col in enumerate(self.columns)])
            column_data = initial_table[1:]
            self.table = np.array(column_data, dtype=dtype)
            self.table = np.insert(self.table, 0, self.columns, axis=0)
            # self.table = np.vstack((self.columns, self.table))
            self.column_types = {column: None for column in self.columns}

        elif initial_table is None:
            self.columns = np.array([])
            self.column_types = {column: None for column in self.columns}

        else:
            raise ValueError(
                "arg1 of DTable should be a instance of dict or np.ndarray or None")

        if columns is not None:
            # if len(columns) != len(self.columns):
            #     raise Exception("Length of headers in data and length of columns should be same")

            self.columns = np.array(columns)

        # print(self.columns)
        # Classifying the column data into data types
        for header, values in zip(self.columns, self.table[1:].T):
            unique_values = set(values)
            if len(unique_values) == 2:
                self.column_types[header] = "Boolean"
            elif all(value.replace('.', '', 1).isdigit() or
                     (value[1:].replace('.', '', 1).isdigit() and value[0] == '-') for value in values):
                self.column_types[header] = "Number"
            elif len(set(values)) < len(values) / 2:
                self.column_types[header] = 'Category'
            else:
                self.column_types[header] = "String"

        # Fill the empty cells
        self.table = np.where(self.table == "", np.nan, self.table)

    def __repr__(self):
        if len(self.table) == 0:
            rows = "Empty Table"
        elif len(self.table) < max_display_rows:
            rows = "\n".join("\t".join(str(cell)[:10]
                             for cell in row) for row in self.table)
        else:
            idx_f = int(max_display_rows / 2) + 2
            idx_l = max_display_rows - idx_f + 1
            rows = "\n".join("\t".join(str(cell)[:10] for cell in row) for row in
                             np.concatenate((self.table[0:idx_f], self.table[len(self.table) - idx_l:])))
        return f"{rows}"

    def info(self):
        """
        Returns a dictionary having information about number of rows and
        columns present in the table
        :return: dict
        """
        if len(self.table) == 0:
            return {
                "rows": 0,
                "cols": len(self.columns)
            }

        return {
            "rows": len(self.table) - 1,
            "cols": len(self.get_columns())
        }

    def get_columns(self):
        """
        Returns list of columns in the table
        :return: list
        """
        return self.columns

    def get_column_types(self):
        """
        Returns dictionary consisting all the columns present in the table and
        their data types

        :return: dict
        """
        return self.column_types

    def select_column(self, column):
        """
        Creates a new DTable from the whole table consisting only the specified column of the table
        :param column:
        :return: DTable
        """
        if column not in self.get_columns():
            raise ValueError(f"No column named '{column}' found")
        idx = np.where(self.get_columns() == column)[0][0]

        values = self.table[:, idx][1:]
        return_table = np.vstack((np.arange(len(values)), values)).T
        header = np.array(["", column])
        return_table = np.vstack((header, return_table))

        return DTable(return_table, dtype=self.dtype)

    def select_columns(self, columns: list | np.ndarray):
        """
        Creates a new DTable from selecting all the provided column names
        :param columns:
        :return: DTable
        """

        # Raise ValueError if any of the provided column names is not in the table
        if any(column not in self.columns for column in columns):
            raise ValueError(
                "Column list contains a column, which is not present in the table")

        # Get indexes of the column names
        idxs = []
        for column in columns:
            idx = np.where(self.columns == column)[0][0]
            idxs.append(idx)
        idxs.sort()

        # Build new table using the indexes
        new_table = self.table[:, idxs]
        idx_row = np.arange(len(new_table))
        new_table = np.vstack((idx_row, new_table.T)).T

        # Return a DTable based on the new_table data
        return DTable(new_table, dtype=self.dtype)

    def get(self, row: int, col: int = None):
        """
        If col, it returns the element present at specified row and column of the table,
        If col is None returns the row present at specified index of the table

        :param row:
        :param col:
        :return: str | float | DTable
        """

        if col is None:
            # Building new table from the row index
            r = self.table[row + 1]
            r = np.vstack((self.columns, r))

            # Return a new DTable from only the row specified
            return DTable(r)

        r = self.table[row, col]
        col_name = self.columns[col]
        try:
            r = float(r)
        except ValueError:
            r = str(r)

        r = np.array(r)
        r = np.vstack((col_name, r))

        return DTable(r, dtype=self.dtype)

    def where(self, column: str, operator: str, value: str | int | float):
        """
        Returns a new subset table of the current table consisting of rows
        satisfying the condition.
        Example use:-

        >>> data_set = {"col1": [4, 2, 3, 7], "col2": [5, 4, 3, 0], "col3": [6, 5.4, 5., 0]}
        >>> dt_ = DTable(data_set)
        >>> new_dt = dt_.where("col3", ">", 5.5)

        Here, 'new_dt' will hold all the rows of the 'dt' Table in which
        column "col3" is greater than(>) 5.5

        :param column:
        :param operator:
        :param value:
        :return: DTable
        """

        # Try to get the index, type of the column
        # if column not found raise ValueError
        try:
            col_index = np.where(self.columns == column)
            col_type = self.get_column_types()[column]
        except KeyError:
            raise ValueError(f"'{column}' column not found")

        if operator == "equals" or operator == "==":
            if col_type == "Number":
                try:
                    value = float(value)
                except ValueError:
                    pass

                if type(value) not in [int, float]:
                    raise ArithmeticError(
                        f"Can't compare a {col_type} with {type(value)}")

                col = self.table[:, col_index][1:].astype("float")

                # Get the indexes of field satisfying the condition
                idxs = np.where(col == value)[0]
                idxs = np.array([idx+1 for idx in idxs])
            else:
                col = self.table[:, col_index][1:]

                # Get the indexes of field satisfying the condition
                idxs = np.where(col == value)[0]
                idxs = np.array([idx + 1 for idx in idxs])

        elif operator == "is greater than" or operator == ">":

            if col_type != "Number":
                raise ArithmeticError(f"Invalid operand for column '{column}'")

            try:
                value = float(value)
            except ValueError:
                pass

            if type(value) not in [int, float]:
                raise ArithmeticError(
                    f"Can't compare a {col_type} with {type(value)}")

            col = self.table[:, col_index][1:].astype("float")

            # Get the indexes of field satisfying the condition
            idxs = np.where(col > value)[0]
            idxs = np.array([idx + 1 for idx in idxs])

        elif operator == "is less than" or operator == "<":

            if col_type != "Number":
                raise ArithmeticError(f"Invalid operand for column '{column}'")

            try:
                value = float(value)
            except ValueError:
                pass

            if type(value) not in [int, float]:
                raise ArithmeticError(
                    f"Can't compare a {col_type} with {type(value)}")

            col = self.table[:, col_index][1:].astype("float")

            # Get the indexes of field satisfying the condition
            idxs = np.where(col < value)[0]
            idxs = np.array([idx + 1 for idx in idxs])

        elif operator == "!=" or (operator == "is not" and col_type == "Number"):
            try:
                value = float(value)
            except ValueError:
                pass

            if type(value) not in [int, float]:
                raise ArithmeticError(
                    f"Can't compare a {col_type} with {type(value)}")

            col = self.table[:, col_index][1:].astype("float")

            # Get the indexes of field satisfying the condition
            idxs = np.where(col != value)[0]
            idxs = np.array([idx+1 for idx in idxs])

        elif operator == "begins with":
            if col_type == "Number":
                raise ArithmeticError(f"Invalid operand for column '{column}'")

            col = self.table[:, col_index][1:]

            # Get the indexes of field satisfying the condition
            idxs = np.where(np.char.startswith(col, value))[0]
            idxs = np.array([idx + 1 for idx in idxs])

        elif operator == "contains":
            if col_type == "Number":
                raise ArithmeticError(f"Invalid operand for column '{column}'")

            col = self.table[:, col_index][1:]

            # Get the indexes of field satisfying the condition
            idxs = np.where(np.char.find(col, value) != -1)[0]
            idxs = np.array([idx + 1 for idx in idxs])

        elif operator == "is":
            if col_type == "Number":
                raise ArithmeticError(
                    f"Can't compare a {col_type} with {type(value)}")
            else:
                col = self.table[:, col_index][1:]

                # Get the indexes of field satisfying the condition
                idxs = np.where(col == value)[0]
                idxs = np.array([idx + 1 for idx in idxs])

        elif operator == "is not":
            if col_type == "Number":
                raise ArithmeticError(
                    f"Can't compare a {col_type} with {type(value)}")
            else:
                col = self.table[:, col_index][1:]

                # Get the indexes of field satisfying the condition
                idxs = np.where(col != value)[0]
                idxs = np.array([idx + 1 for idx in idxs])

        else:
            raise ValueError(f"Invalid operator '{operator}'")

        # Build new numpy array based on the indexes received
        if len(idxs) != 0:
            table = np.vstack(
                (self.get_columns(), [self.table[idx, ] for idx in idxs]))
        else:
            table = np.array([self.get_columns()])

        # Return new DTable using the new numpy array built
        return DTable(table, columns=self.columns, dtype=self.dtype)

    def intersection(self, dt):
        if not isinstance(dt, DTable):
            raise ValueError("Passed parameter is not an instance of DTable")

        table1 = self.table[1:]
        table2 = dt.table[1:]

        # Converting the rows to tuples to make them hashable.
        table1_tuples = [tuple(row) for row in table1]
        table2_tuples = [tuple(row) for row in table2] 

        # Finding intersection of the two tables
        intersection = set(table1_tuples) & set(table2_tuples)
        intersection = list(intersection)

        # Converting the intersection table again into numpy array
        intersection_table = np.array(intersection)
        if len(intersection_table) == 0:
            return DTable(np.array([self.columns]))
        intersection_table = np.vstack((self.columns, intersection_table))

        # Returning the intersection
        return DTable(intersection_table)
    
    def variance(self, column):
        # Storing in an array.
        data_array = self.select_column(column).table[:, 1][1:]
        data_array = data_array[data_array != "nan"]
        data_array = data_array.astype('float')
        
        # Calculating Variance
        overall_variance = np.var(data_array)
        
        # Returning the calculated Variance.
        return overall_variance
    
    def standard_deviation(self, column):
        data_array = self.select_column(column).table[:, 1][1:]
        data_array = data_array[data_array != "nan"]
        data_array = data_array.astype('float')

        # Calculating Standard Deviation
        overall_standard_deviation = np.std(data_array)

        # Returning the calculated Standard Deviation.
        return overall_standard_deviation

    def count(self, column: str) -> int:
        """
        Returns the count of (defined) values present in specified column
        """
        if column not in self.columns:
            raise ValueError(f"'{column}' not found")
        
        # Get column values
        col = self.select_column(column).table[:, 1][1:]
        
        # Filter not nan values
        filtered_col = col[col != "nan"]

        return len(filtered_col) - 1

    def mean(self, column):
        """
        This function will return the median of input column
        """
        if column not in self.columns:
            raise ValueError(f"'{column}' not found")

        col = self.select_column(column).table[:, 1][1:].astype("float")

        mean = np.mean(col)

        return mean

    def median(self, column):
        """
        Returns the median of input column 
        """
        if column not in self.columns:
            raise ValueError(f"'{column}' not found")
        
        col = self.select_column(column).table[:, 1][1:].astype("float")

        median = np.median(col)

        return median

    def mode(self, column):
        # Get column values
        col = self.select_column(column).table[:, 1][1:]

        # Find unique values in array
        values, counts = np.unique(col, return_counts=True)

        # Find mode indexes
        mode = np.argwhere(counts == np.max(counts))

        return values[mode].flatten()

    def to_html(self, file_name: str = None):
        """
        Builds HTML string for the DTable
        If file_name is not None then it writes the HTML string to the file
        Note: It overwrites the file

        :param file_name:
        :return: str
        """

        style = "<style>" \
                "table, td, th {" \
                "padding: 10px;" \
                "border: 1px solid black;" \
                "border-collapse: collapse;" \
                "}" \
                "</style>"

        # Create HTML table header row
        header_row = "<tr>" + "".join(f"<th>{col}</th>" for col in self.table[0]) + "</tr>"

        # Create HTML table data rows
        data_rows = "".join("<tr>" +
                            "".join(f"<td>"
                                    f"{value}"
                                    f"</td>" for value in row) + "</tr>" for row in self.table[1:])

        # Build the HTML table
        dt_html = style + f"<table>{header_row}{data_rows}</table>"

        if file_name is not None:
            with open(file_name, "w+", encoding='utf-8') as file:
                file.write(dt_html)

        # Return the HTML form of the DTable
        return dt_html

    def to_csv(self, filename: str) -> None:
        """
        This function converts the current table to CSV format
        :param filename:
        :return: None
        """
        with open(filename, "w+", encoding="utf-8") as file:
            csv_string = ''
            # Headers
            headers = ",".join(self.get_columns()) + "\n"
            csv_string += headers

            # Rest data
            data = "\n".join((",".join(f'"{value}"' if "," in value else value for value in row))
                             for row in self.table[1:])
            csv_string += data

            file.write(csv_string)

    def scatter_plot(self, x: str, y: str, drange: tuple = None) -> None:
        """
        Creates a scatter plot for given column names (x and y)
        :param x: Column name to be plotted on x-axis
        :param y: Column name to be plotted on y-axis
        :param drange: Range for number of points
        :return: None
        """
        if drange is not None:
            drange = (drange[0]+1, drange[1]+1)
        else:
            drange = (1, None)

        x_data = self.select_column(x).table[:, 1][drange[0]: drange[1]]
        y_data = self.select_column(y).table[:, 1][drange[0]: drange[1]]

        if self.get_column_types()[x] == "Number":
            x_data = x_data.astype("float")

        if self.get_column_types()[y] == "Number":
            y_data = y_data.astype("float")

        plt.scatter(x_data, y_data, s=15)
        plt.xlabel(x)
        plt.ylabel(y)
        plt.legend([y])
        if self.get_column_types()[x] == "String":
            plt.xticks(rotation=90)

    def bar_plot(self, x: str = None, y: str = None, drange: tuple = None, exclude: list = None) -> None:
        """
        Creates a bar plot based on given x and y, if x and y not specified
        the creates bar plot for all the numeric columns present in table.
        :param x:
        :param y:
        :param drange:
        :param exclude:
        :return: None
        """
        if drange is not None:
            drange = (drange[0]+1, drange[1]+1)
        else:
            drange = (1, None)

        if x is not None and y is not None:

            x_data = self.select_column(x).table[:, 1][drange[0]: drange[1]]
            y_data = self.select_column(y).table[:, 1][drange[0]: drange[1]]

            # if self.get_column_types()[x] == "Number":
            #     x_data = x_data.astype("int")

            if self.get_column_types()[y] == "Number":
                y_data = y_data.astype("float")

            indices = np.arange(len(x_data))
            plt.bar(indices, y_data, width=0.5, tick_label=x_data, label=y)
            if self.get_column_types()[x] == "String":
                plt.xticks(rotation=90)
            plt.legend()

        else:
            number_cols = []
            col_types = self.get_column_types()
            for k, v in col_types.items():
                if v == "Number" and k not in exclude:
                    number_cols.append(k)

            data_lists = []

            for col in number_cols:
                arr = self.select_column(col).table[:, 1][drange[0]: drange[1]].astype('float')
                data_lists.append(arr)

            if drange[1] is not None:
                x_ticks = np.arange(drange[1] - drange[0])
            else:
                x_ticks = np.arange(len(data_lists[0]))

            bar_width = 0.5
            number_cols_len = len(number_cols)
            offset = bar_width / number_cols_len
            current_offset = offset * (number_cols_len / 2)
            if number_cols_len > 1:
                current_offset *= -1
            else:
                current_offset = 0

            for idx, arr in enumerate(data_lists):
                plt.bar(x_ticks + current_offset, arr, width=offset, label=number_cols[idx])
                current_offset += offset

            plt.xticks(x_ticks)
            plt.legend()

    def pie_plot(self, x: str, y: str, drange: tuple = None) -> None:
        """
        Creates a pie plot for given column names (x and y)
        :param x: Column name to be plotted on x-axis
        :param y: Column name to be plotted on y-axis
        :param drange: Range for number of points
        :return: None
        """
        if drange is not None:
            drange = (drange[0]+1, drange[1]+1)
        else:
            drange = (1, None)

        x_data = self.select_column(x).table[:, 1][drange[0]: drange[1]]
        y_data = self.select_column(y).table[:, 1][drange[0]: drange[1]]

        if self.get_column_types()[x] == "Number":
            x_data = x_data.astype("float")

        if self.get_column_types()[y] == "Number":
            y_data = y_data.astype("float")

        plt.pie(x_data, labels=y_data)

    def stem_plot(self, x: str, y: str, drange: tuple = None) -> None:
        """
        Creates a stem plot for given column names (x and y)
        :param x: Column name to be plotted on x-axis
        :param y: Column name to be plotted on y-axis
        :param drange: Range for number of points
        :return: None
        """
        if drange is not None:
            drange = (drange[0]+1, drange[1]+1)
        else:
            drange = (1, None)

        x_data = self.select_column(x).table[:, 1][drange[0]: drange[1]]
        y_data = self.select_column(y).table[:, 1][drange[0]: drange[1]]

        if self.get_column_types()[x] == "Number":
            x_data = x_data.astype("float")

        if self.get_column_types()[y] == "Number":
            y_data = y_data.astype("float")

        plt.stem(x_data, y_data, linefmt=":", markerfmt="ro")
        plt.xlabel(x)
        plt.ylabel(y)
        plt.legend([y])
    
    def line_plot(self, x: str, drange: tuple = None) -> None:
        """
        Creates a line plot for given column names (x and y)
        """
        if drange is not None:
            drange = (drange[0]+1, drange[1]+1)
        else:
            drange = (1, None)

        x_data = self.select_column(x).table[:, 1][drange[0]: drange[1]]

        if self.get_column_types()[x] == "Number":
            x_data = x_data.astype("float")

        plt.plot(x_data)
    
    def histogram_plot(self, x: str, drange: tuple = None) -> None:
        """
        Creates a histogram plot for given column names (x and y)
        """
        if drange is not None:
            drange = (drange[0]+1, drange[1]+1)
        else:
            drange = (1, None)

        x_data = self.select_column(x).table[:, 1][drange[0]: drange[1]]

        if self.get_column_types()[x] == "Number":
            x_data = x_data.astype("float")

        plt.hist(x_data)


def read_csv(file_path: str) -> DTable:
    """
    Reads a csv file to create a dictionary holding keys as columns
    and each key holds a list of values of that column, and then returns
    a DTable object with the dictionary data.

    :param file_path:
    :return: ddive.DTable
    """
    from os.path import getsize  # Only used for checking the size of dataset file

    # size of file in KiB
    size_of_file = getsize(file_path) / 1024

    # 10 MiB
    mib_10 = 10 * 1024
    if size_of_file > mib_10:
        raise Exception(f"Size of file ({size_of_file} KiB) is too large")

    with open(file_path, 'r', encoding='utf-8') as file:

        # Get the column headers from the file
        # Fill the empty column headers
        headers = file.readline().strip().split(',')
        headers = [f"Unnamed{idx}" if col == "" else col for idx, col in enumerate(headers)]

        # Get the rest of the lines from the file
        lines = [line.strip() for line in file]

        data_dict = {header: [] for header in headers}

        # Iterate through the lines and split on getting a ','
        # Skip the split if ',' present inside double quotes symbol
        for line in lines:
            values = []
            inside_quote = False
            current_value = ''

            for char in line.strip():
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

        # Return DTable based on the values in file
        return DTable(data_dict)


def read_html(html_str: str) -> DTable:
    """
    Reads HTML string to create numpy.array() of it, and
    then returns a DTable object based on the created array.

    **Note**: This method assumes that the passed string is a valid HTML string
    constructed from DTable.to_html() method,
    if invalid HTML table string is passed it may cause errors,
    use it cautiously.

    :param html_str:
    :return: DTable
    """

    # Split the string to remove all 'tr' tags
    # Each element of the rows list will have single row of the HTML table
    rows = html_str.split("<tr>")[1:]
    rows = [v.split("</tr>")[0] for v in rows]

    # Parse the headers
    headers = np.array([header.split("</th>")[0] for header in rows[0].split("<th>") if "</th>" in header])

    # Parse the values
    values = np.array([[v.split("</td>")[0] for v in row.split("<td>") if "</td>" in v] for row in rows[1:]])

    # Build and return Table
    if len(values) != 0:
        data = np.vstack((headers, values))
    else:
        data = np.array([headers])

    return DTable(data)
