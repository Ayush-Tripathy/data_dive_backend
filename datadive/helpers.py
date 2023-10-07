from ddive import DTable


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
