import datadive
import matplotlib.pyplot as plt


def print_menu():
    options = ["1. Print a Column", "2. Print rows based on conditions",
               "3. Calculate mean of a column", "4. Calculate median of a column",
               "5. Calculate variance of a column", "6. Calculate standard deviation of a column",
               "7. Count total number of not null values in a column", "8. Print all the column names",
               "9. Print element at position", "10. Print info of Table",
               "11. Print a Column types", "12. Print whole Table",
               "13. Reset selection to whole table", "14. Convert selection to CSV",
               "15. Set max display rows", "16. Create Scatter plot for column",
               "17. Create Bar plot for column", "18. Count number of non blank values"]
    cell_space = len(max(options, key=len))
    for i in range(0, len(options), 2):
        if i + 1 < len(options):
            print(f"{options[i]:<{cell_space}}\t{options[i + 1]:<{cell_space}}")
        else:
            print(f"{options[i]:<{cell_space}}")


def main():
    approval = ["y", "yes"]
    denial = ["n", "no"]

    filepath = input("Enter filepath: ")
    try:
        dt = datadive.read_csv(filepath)
        selected_dt = dt
        print("File loaded successfully. Specify what functions you want to run- ")
        print_menu()
        while True:
            choice = input("\nEnter your choice (enter 'm' for menu, 'q' to quit): ")
            if choice == "q":
                break

            elif choice == "m":
                print_menu()

            elif choice == "1":
                try:
                    col = input("Enter column name: ")
                    print(selected_dt.select_column(col))
                except ValueError as v:
                    print(v)

            elif choice == "2":
                conditions = []
                while True:
                    c_input = input("Enter condition [col operator value] ('q' to discard): ")
                    if c_input == "q":
                        break

                    c_ = c_input.strip().split(" ", 1)
                    if len(c_) != 2:
                        print("Invalid condition")
                        continue
                    if c_[1].startswith('"'):
                        c1 = c_[0]
                        c2 = c_[1].split('"')[1]
                        c3 = c_[1].split('"')[2][1:]
                        condition = [c1, c2, c3]
                    else:
                        condition = c_input.strip().split(" ")

                    if len(condition) != 3:
                        print("Invalid condition")
                        continue

                    try:
                        selected_dt.where(condition[0], condition[1], condition[2])
                        conditions.append(condition)
                    except Exception as e:
                        print(e)
                        continue

                    new_dts = []

                    # Get data tables by applying each condition
                    for condition in conditions:
                        t_dt = selected_dt.where(condition[0], condition[1], condition[2])
                        new_dts.append(t_dt)

                    # Taking intersection of all data tables
                    new_dt = new_dts[0]
                    for dt_ in new_dts[1:]:
                        new_dt = new_dt.intersection(dt_)
                    print(new_dt)
                    n_info = new_dt.info()
                    print(f"[{n_info['rows']} Rows * {n_info['cols']} Columns]")

                    print("Enter 'a' to add new condition or 's' to select current table based on entered conditions")
                    ch = input("Enter: ")
                    if ch == "s":
                        selected_dt = new_dt
                        break
                    elif ch == "a":
                        continue
                    else:
                        print("invalid choice")

            elif choice == "3":
                col = input("Enter column name: ")
                try:
                    print(f"Mean: {selected_dt.mean(col)}")
                except ValueError as v:
                    if str(v).split(":")[0] == f"No column named '{col}' found":
                        print(v)
                    else:
                        print("Column has non numeric values, try again with column that has only numeric values")

            elif choice == "4":
                col = input("Enter column name: ")
                try:
                    print(f"Median: {selected_dt.median(col)}")
                except ValueError as v:
                    if str(v).split(":")[0] == f"No column named '{col}' found":
                        print(v)
                    else:
                        print("Column has non numeric values, try again with column that has only numeric values")

            elif choice == "5":
                col = input("Enter column name: ")
                try:
                    print(f"Variance: {selected_dt.variance(col)}")
                except ValueError as v:
                    if str(v).split(":")[0] == f"No column named '{col}' found":
                        print(v)
                    else:
                        print("Column has non numeric values, try again with column that has only numeric values")

            elif choice == "6":
                col = input("Enter column name: ")
                try:
                    print(f"Standard deviation: {selected_dt.standard_deviation(col)}")
                except ValueError as v:
                    if str(v).split(":")[0] == f"No column named '{col}' found":
                        print(v)
                    else:
                        print("Column has non numeric values, try again with column that has only numeric values")

            elif choice == "7":
                try:
                    col = input("Enter column name: ")
                    print(f"Count: {selected_dt.count(col)}")
                except ValueError as v:
                    print(v)

            elif choice == "8":
                print(selected_dt.get_columns())

            elif choice == "9":
                try:
                    pos = input("Enter position [row col] (ignore column number to print whole row): ")
                    p = pos.strip().split(" ")
                    if len(p) == 2:
                        r, c = p
                        r = int(r)
                        c = int(c)
                        print(selected_dt.get(r, c))
                    elif len(p) == 1:
                        print(selected_dt.get(int(p[0])))

                except ValueError as v:
                    print(v)

            elif choice == "10":
                print("---INFO---")
                info = selected_dt.info()
                print(f"Rows: {info['rows']}")
                print(f"Columns: {info['cols']}")

            elif choice == "11":
                col_types = selected_dt.get_column_types()
                col_space = len(max(col_types.keys(), key=len))
                print(f"{'Column name':<{col_space}}\tColumn type")
                for col_, type_ in col_types.items():
                    print(f"{col_:<{col_space}}\t{type_}")

            elif choice == "12":
                print(selected_dt)

            elif choice == "13":
                selected_dt = dt
                print("Reset selection")

            elif choice == "14":
                filename = input("Enter filename: ")
                selected_dt.to_csv(filename)
                print("File saved successfully.")

            elif choice == "15":
                try:
                    n = int(input("Enter max number of rows to display: "))
                    datadive.ddive.max_display_rows = n
                except ValueError:
                    print("Invalid input, please try again with a number.")

            elif choice == "16":
                x_col = input("Enter column name for x-axis: ")
                y_col = input("Enter column name for y-axis: ")
                try:
                    l, u = map(int, input("Enter range [lower upper]: ").split(" "))

                    plt.figure(figsize=(7, 6))
                    selected_dt.scatter_plot(x_col, y_col, drange=(l, u))
                    plt.show()
                    to_save = input("Do you want to save the figure to image?[Y N] ")

                    if to_save.lower() in approval:
                        filepath_ = input("Enter filename (without extension): ")
                        selected_dt.scatter_plot(x_col, y_col, drange=(l, u))
                        plt.savefig(f"{filepath_}.png")
                    elif to_save.lower() in denial:
                        pass
                    else:
                        print("Invalid input. ")

                except ValueError:
                    print("Please enter valid range.")

            elif choice == "17":
                print("Leave both fields blank to construct bar plot for all numeric columns.")
                x_col = input("Enter column name for x-axis: ")
                y_col = input("Enter column name for y-axis: ")
                try:
                    l, u = map(int, input("Enter range [lower upper]: ").split(" "))

                    plt.figure()
                    if x_col == "" or y_col == "":
                        ex = input("Enter any column names to exclude from graph [col1 col2 ...]: ").split(" ")
                        selected_dt.bar_plot(drange=(l, u), exclude=ex)
                    else:
                        selected_dt.bar_plot(x_col, y_col, drange=(l, u))
                    plt.show()

                    to_save = input("Do you want to save the figure to image?[Y N] ")

                    if to_save.lower() in approval:
                        filepath_ = input("Enter filename (without extension): ")
                        if x_col == "" or y_col == "":
                            selected_dt.bar_plot(drange=(l, u), exclude=ex)
                        else:
                            selected_dt.bar_plot(x_col, y_col, drange=(l, u))

                        plt.savefig(f"{filepath_}.png")
                    elif to_save.lower() in denial:
                        pass
                    else:
                        print("Invalid input. ")
                except ValueError:
                    print("Please enter valid range.")

            elif choice == "18":
                col = input("Enter column name: ")
                try:
                    print(f"Mean: {selected_dt.count(col)}")
                except ValueError as v:
                    if str(v).split(":")[0] == f"No column named '{col}' found":
                        print(v)

            else:
                print("Invalid choice, try again.")

    except FileNotFoundError:
        print("File does not exists, please enter a valid filepath.")
    except ValueError:
        print("There may be irregularity in the file or the file is not supported.")
    except Exception as e:
        print(e)


if __name__ == "__main__":
    main()
