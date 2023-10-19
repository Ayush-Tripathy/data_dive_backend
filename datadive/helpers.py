from datadive.ddive import DTable


def get_column_type(dt: DTable, column: str) -> str:
    return dt.get_column_types()[column]


def get_related_condition_types(dtype: str):
    valid_types = ["Number", "String", "Category", "Boolean"]
    if dtype == valid_types[0]:
        return ["equals", "is not", "is less than", "is greater than"]
    elif dtype == valid_types[1]:
        return ["equals", "is not", "contains", "begins with"]
    elif dtype == valid_types[2]:
        return ["is", "is not"]
    elif dtype == valid_types[3]:
        return ["is", "is not"]


def read_csv_v2(file_path: str, dtype: str = "<U255") -> DTable:
    """
    Reads a csv file without using the csv module and creates a dictionary
    holding keys as columns and each key holds a list of values of that column,
    and then returns a DTable object with the dictionary data.

    :param file_path:
    :param dtype:
    :return: DTable
    """

    from os.path import getsize  # Only used for checking the size of dataset file
    # size of the file in KiB
    size_of_file = getsize(file_path) / 1024

    # 30 MiB
    mib_30 = 30 * 1024
    if size_of_file > mib_30:
        raise Exception(f"Size of file ({size_of_file} KiB) is too large")

    with open(file_path, 'r', encoding='utf-8') as file:
        # Get the column headers from the file
        headers = file.readline().strip().split(',')

        data_dict = {header: [] for header in headers}

        # Read the rest of the file
        inside_quote = False
        current_value = ''
        idx = 0
        for line in file:
            line = line.strip()
            for char in line:
                if char == '"':
                    inside_quote = not inside_quote
                elif char == ',' and not inside_quote:
                    data_dict[headers[idx]].append(current_value)
                    idx += 1
                    current_value = ''
                else:
                    current_value += char

            # Handle multiline values
            if inside_quote:
                current_value += '\n'
            else:

                # Append the last value after the loop
                data_dict[headers[idx]].append(current_value)
                idx = 0
                current_value = ''

    # Return a DTable based on the values in file
    return DTable(data_dict, dtype=dtype)


def select_rows(c_dt: DTable, conditions: list[list]) -> DTable:
    """
    Creates a DTable from another DTable's rows, the new DTable will
    hold rows which satisfy all the given conditions.
    :param c_dt:
    :param conditions:
    :return: DTable
    """

    new_dts = []

    # Get data tables by applying each condition
    for condition in conditions:
        t_dt = c_dt.where(condition[0], condition[1], condition[2])
        new_dts.append(t_dt)

    # Taking intersection of all data tables
    new_dt = new_dts[0]
    for dt in new_dts[1:]:
        new_dt = new_dt.intersection(dt)

    return new_dt
