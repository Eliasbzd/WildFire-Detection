import csv
import os

class csv_handler():
    """
    Utility class to process adding/editing the final row of stat csv.
    Attributes:
    - path : path to files
    - fieldnames : defaults to what we need as a basis
    Functions:
    - add_rows: append a row at the end of the file, not necessarily complete
    - fetch_rows/show_rows: update/show rows with stored values
    - edit_last_rows: edit the values of the last row.
    - update: update csv
    """

    def __init__(self, path, fieldnames=['Model','Feature','Feature Transf','Dataset','Accuracy', 'F1-Score','Kappa (Cohen)','AUC', 'DATE']):
        self.fieldnames = fieldnames
        self.path = path
        try:
            with open(path,'r', newline='') as f:
                rd = csv.reader(f)
                self.rows = list(rd)
        except:
            with open(path,'w', newline='') as f:
                rd = csv.writer(f)
                rd.writerow(fieldnames)
            print("[ CSV ] Created CSV at {}".format(path))
            self.rows = [fieldnames]

    def add_row(self, fields):
        if len(fields) != len(self.fieldnames):
            return "error"
        self.rows += [fields]
        self.update()

    def fetch_rows(self):
        with open(self.path,'r', newline='') as f:
                rd = csv.reader(f)
                self.rows = list(rd)
        return self.rows

    def edit_last_row(self, fields):
        if len(fields) != len(self.fieldnames) or len(self.rows)==0:
            print("[ CSV ] Error: invalid action ({} vs{})".format(len(fields), len(self.fieldnames)))
        for index in range(len(fields)):
            if fields[index] != None:
                self.rows[-1][index] = fields[index]
        self.update()
        self.fetch_rows()

    def update(self):
        with open(self.path,'w', newline='') as f:
            wr = csv.writer(f)
            for element in self.rows:
                wr.writerow(element)
        print("[ CSV ] Updated")

    def show_rows(self):
        print(self.rows)